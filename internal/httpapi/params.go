package httpapi

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/store"
)

// params collects request parameter problems so they can be reported together
// as a single FastAPI-style 422.
type params struct {
	r    *http.Request
	errs validationErrors
}

func newParams(r *http.Request) *params { return &params{r: r} }

func (p *params) fail(where, name, typ, msg string, input any, ctx map[string]any) {
	p.errs.add(fieldError{Type: typ, Loc: []any{where, name}, Msg: msg, Input: input, Ctx: ctx})
}

// pathUUID reads and validates a UUID path value.
func (p *params) pathUUID(name string) string {
	raw := p.r.PathValue(name)
	if u, ok := domain.CanonicalUUID(raw); ok {
		return u
	}
	p.fail("path", name, "uuid_parsing", "Input should be a valid UUID, invalid character: expected an optional prefix of `urn:uuid:` followed by [0-9a-fA-F-], found `"+badChar(raw)+"` at "+strconv.Itoa(badAt(raw)), raw, nil)
	return ""
}

func badAt(s string) int {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c == '-' || c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
			return i
		}
	}
	return len(s)
}

func badChar(s string) string {
	if i := badAt(s); i < len(s) {
		return string(s[i])
	}
	return ""
}

func (p *params) has(name string) (string, bool) {
	vals, ok := p.r.URL.Query()[name]
	if !ok || len(vals) == 0 {
		return "", false
	}
	return vals[0], true
}

// optString returns the raw string and whether it was supplied.
func (p *params) optString(name string) (string, bool) { return p.has(name) }

// reqString requires a non-empty (min_length=1) query string.
func (p *params) reqString(name string) string {
	v, ok := p.has(name)
	if !ok {
		p.fail("query", name, "missing", "Field required", nil, nil)
		return ""
	}
	if v == "" {
		p.fail("query", name, "string_too_short", "String should have at least 1 character", v, map[string]any{"min_length": 1})
	}
	return v
}

func (p *params) intParam(name string, def, min, max int) int {
	v, ok := p.has(name)
	if !ok {
		return def
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		p.fail("query", name, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v, nil)
		return def
	}
	if n < min {
		p.fail("query", name, "greater_than_equal", "Input should be greater than or equal to "+strconv.Itoa(min), v, map[string]any{"ge": min})
		return def
	}
	if max > 0 && n > max {
		p.fail("query", name, "less_than_equal", "Input should be less than or equal to "+strconv.Itoa(max), v, map[string]any{"le": max})
		return def
	}
	return n
}

func (p *params) dateParam(name string) *time.Time {
	v, ok := p.has(name)
	if !ok {
		return nil
	}
	d, err := domain.ParseDate(v)
	if err != nil {
		p.fail("query", name, "date_from_datetime_parsing", "Input should be a valid date or datetime, invalid character in year", v, nil)
		return nil
	}
	t := d.Time
	return &t
}

func (p *params) uuidParam(name string) *string {
	v, ok := p.has(name)
	if !ok {
		return nil
	}
	u, ok := domain.CanonicalUUID(v)
	if !ok {
		p.fail("query", name, "uuid_parsing", "Input should be a valid UUID, invalid character: expected an optional prefix of `urn:uuid:` followed by [0-9a-fA-F-], found `"+badChar(v)+"` at "+strconv.Itoa(badAt(v)), v, nil)
		return nil
	}
	return &u
}

// ok reports whether all parameters were valid, writing the 422 if not.
func (p *params) ok(w http.ResponseWriter) bool {
	if len(p.errs) == 0 {
		return true
	}
	p.errs.write(w)
	return false
}

// issuesToErrors converts domain issues to a 422 under the given location prefix.
func writeIssues(w http.ResponseWriter, issues domain.Issues, prefix ...any) {
	out := make(validationErrors, 0, len(issues))
	for _, i := range issues {
		loc := append(append([]any{}, prefix...), i.Loc...)
		out = append(out, fieldError{Type: i.Type, Loc: loc, Msg: i.Msg, Input: i.Input, Ctx: i.Ctx})
	}
	out.write(w)
}

func scopeOf(u userView) store.Scope { return store.Scope{UserID: u.ID, CanSeeAll: u.CanSeeAll} }
