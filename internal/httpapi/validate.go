package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"unicode/utf8"
)

const maxJSONBody = 1 << 20

// fieldError mirrors one entry of FastAPI's 422 "detail" array.
type fieldError struct {
	Type  string         `json:"type"`
	Loc   []any          `json:"loc"`
	Msg   string         `json:"msg"`
	Input any            `json:"input,omitempty"`
	Ctx   map[string]any `json:"ctx,omitempty"`
}

type validationErrors []fieldError

func (v *validationErrors) add(e fieldError) { *v = append(*v, e) }

func (v validationErrors) write(w http.ResponseWriter) {
	writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": []fieldError(v)})
}

func bodyLoc(field string) []any { return []any{"body", field} }

// decodeJSON reads a JSON object body into dst. On failure it writes a 422 in
// FastAPI's format and returns false. Unknown fields are ignored like pydantic.
func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) bool {
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxJSONBody))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "Request body too large")
			return false
		}
		validationErrors{{Type: "json_invalid", Loc: []any{"body"}, Msg: "JSON decode error"}}.write(w)
		return false
	}
	if len(data) == 0 {
		validationErrors{{Type: "missing", Loc: []any{"body"}, Msg: "Field required"}}.write(w)
		return false
	}
	var probe any
	if err := json.Unmarshal(data, &probe); err != nil {
		validationErrors{{Type: "json_invalid", Loc: []any{"body"}, Msg: "JSON decode error"}}.write(w)
		return false
	}
	if _, ok := probe.(map[string]any); !ok {
		validationErrors{{Type: "model_attributes_type", Loc: []any{"body"}, Msg: "Input should be a valid dictionary or object to extract fields from", Input: probe}}.write(w)
		return false
	}
	if err := json.Unmarshal(data, dst); err != nil {
		var ute *json.UnmarshalTypeError
		if errors.As(err, &ute) {
			validationErrors{{Type: typeErrorKind(ute.Value, ute.Type.String()), Loc: bodyLoc(ute.Field), Msg: "Input should be a valid " + ute.Type.String()}}.write(w)
			return false
		}
		validationErrors{{Type: "json_invalid", Loc: []any{"body"}, Msg: "JSON decode error"}}.write(w)
		return false
	}
	return true
}

func typeErrorKind(got, want string) string {
	switch want {
	case "string":
		return "string_type"
	case "bool":
		return "bool_type"
	case "float64", "float32":
		return "float_type"
	default:
		return "int_type"
	}
}

// checkString validates a required string field's presence and length (in
// characters, like pydantic).
func checkString(errs *validationErrors, field string, value *string, min, max int) {
	if value == nil {
		errs.add(fieldError{Type: "missing", Loc: bodyLoc(field), Msg: "Field required"})
		return
	}
	n := utf8.RuneCountInString(*value)
	if n < min {
		errs.add(fieldError{Type: "string_too_short", Loc: bodyLoc(field),
			Msg: fmt.Sprintf("String should have at least %d character%s", min, plural(min)), Input: *value, Ctx: map[string]any{"min_length": min}})
	}
	if max > 0 && n > max {
		errs.add(fieldError{Type: "string_too_long", Loc: bodyLoc(field),
			Msg: fmt.Sprintf("String should have at most %d character%s", max, plural(max)), Input: *value, Ctx: map[string]any{"max_length": max}})
	}
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// readObject reads the request body as a JSON object (numbers as json.Number)
// and returns the raw bytes too. Failures are written as 422/413.
func readObject(w http.ResponseWriter, r *http.Request) (map[string]any, []byte, bool) {
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxJSONBody))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "Request body too large")
			return nil, nil, false
		}
		validationErrors{{Type: "json_invalid", Loc: []any{"body"}, Msg: "JSON decode error"}}.write(w)
		return nil, nil, false
	}
	if len(bytes.TrimSpace(data)) == 0 {
		validationErrors{{Type: "missing", Loc: []any{"body"}, Msg: "Field required"}}.write(w)
		return nil, nil, false
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil || dec.More() {
		validationErrors{{Type: "json_invalid", Loc: []any{"body"}, Msg: "JSON decode error"}}.write(w)
		return nil, nil, false
	}
	m, ok := v.(map[string]any)
	if !ok {
		validationErrors{{Type: "model_attributes_type", Loc: []any{"body"}, Msg: "Input should be a valid dictionary or object to extract fields from", Input: v}}.write(w)
		return nil, nil, false
	}
	return m, data, true
}
