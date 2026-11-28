package httpapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/shopspring/decimal"

	"github.com/tendstead/vision-bill/internal/benchmark"
	"github.com/tendstead/vision-bill/internal/domain"
)

func (s *Server) benchmarkRoutes(mux *http.ServeMux) {
	const base = "/api/v1/benchmarks"
	mux.HandleFunc("POST "+base, s.admin(s.createBenchmark))
	mux.HandleFunc("GET "+base, s.admin(s.listBenchmarks))
	mux.HandleFunc("GET "+base+"/capabilities", s.admin(func(w http.ResponseWriter, _ *http.Request, _ userView) {
		writeJSON(w, http.StatusOK, s.opts.Benchmarks.Capabilities())
	}))
	mux.HandleFunc("GET "+base+"/provider-options", s.admin(func(w http.ResponseWriter, _ *http.Request, _ userView) {
		writeJSON(w, http.StatusOK, s.opts.Benchmarks.ProviderOptions())
	}))
	mux.HandleFunc("GET "+base+"/compare", s.admin(s.compareBenchmarks))
	mux.HandleFunc("GET "+base+"/{run_id}/candidates", s.admin(s.compareCandidates))
	mux.HandleFunc("POST "+base+"/{run_id}/rerun", s.admin(s.rerunBenchmark))
	mux.HandleFunc("GET "+base+"/{run_id}/results", s.admin(s.benchmarkResults))
	mux.HandleFunc("GET "+base+"/{run_id}", s.admin(s.benchmarkStatus))
	mux.HandleFunc("POST "+base+"/{run_id}/reevaluate", s.admin(s.reevaluateBenchmark))
}

// bodyReader accumulates validation problems while reading typed fields.
type bodyReader struct {
	m    map[string]any
	loc  []any
	errs validationErrors
}

func (b *bodyReader) fail(key, typ, msg string, input any, ctx map[string]any) {
	b.errs.add(fieldError{Type: typ, Loc: append(append([]any{}, b.loc...), key), Msg: msg, Input: input, Ctx: ctx})
}

func (b *bodyReader) optString(key string) *string {
	v, ok := b.m[key]
	if !ok || v == nil {
		return nil
	}
	s, isStr := v.(string)
	if !isStr {
		b.fail(key, "string_type", "Input should be a valid string", v, nil)
		return nil
	}
	return &s
}

// label reads the optional free-text run label (1-200 characters).
func (b *bodyReader) label() *string {
	l := b.optString("label")
	if l == nil {
		return nil
	}
	if n := len([]rune(*l)); n < 1 || n > 200 {
		typ, msg, ctx := "string_too_short", "String should have at least 1 character", map[string]any{"min_length": 1}
		if n > 200 {
			typ, msg, ctx = "string_too_long", "String should have at most 200 characters", map[string]any{"max_length": 200}
		}
		b.fail("label", typ, msg, *l, ctx)
		return nil
	}
	return l
}

func (b *bodyReader) optInt(key string, lo, hi int) *int {
	v, ok := b.m[key]
	if !ok || v == nil {
		return nil
	}
	n, isNum := v.(json.Number)
	if !isNum {
		b.fail(key, "int_type", "Input should be a valid integer", v, nil)
		return nil
	}
	i, err := strconv.Atoi(n.String())
	if err != nil {
		b.fail(key, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v, nil)
		return nil
	}
	if i < lo {
		b.fail(key, "greater_than_equal", "Input should be greater than or equal to "+strconv.Itoa(lo), v, map[string]any{"ge": lo})
		return nil
	}
	if hi > 0 && i > hi {
		b.fail(key, "less_than_equal", "Input should be less than or equal to "+strconv.Itoa(hi), v, map[string]any{"le": hi})
		return nil
	}
	return &i
}

func (b *bodyReader) optBool(key string, def bool) bool {
	v, ok := b.m[key]
	if !ok {
		return def
	}
	bv, isBool := v.(bool)
	if !isBool {
		b.fail(key, "bool_type", "Input should be a valid boolean", v, nil)
		return def
	}
	return bv
}

func (b *bodyReader) optBoolPtr(key string) *bool {
	v, ok := b.m[key]
	if !ok || v == nil {
		return nil
	}
	bv, isBool := v.(bool)
	if !isBool {
		b.fail(key, "bool_type", "Input should be a valid boolean", v, nil)
		return nil
	}
	return &bv
}

func (b *bodyReader) optNonNegDecimal(key string) *string {
	v, ok := b.m[key]
	if !ok || v == nil {
		return nil
	}
	var text string
	switch t := v.(type) {
	case json.Number:
		text = t.String()
	case string:
		text = t
	default:
		b.fail(key, "decimal_type", "Decimal input should be an integer, float, string or Decimal object", v, nil)
		return nil
	}
	d, err := decimal.NewFromString(strings.TrimSpace(text))
	if err != nil {
		b.fail(key, "decimal_parsing", "Input should be a valid decimal", v, nil)
		return nil
	}
	if d.IsNegative() {
		b.fail(key, "greater_than_equal", "Input should be greater than or equal to 0", v, map[string]any{"ge": 0})
		return nil
	}
	return &text
}

func (b *bodyReader) enum(key, def string, allowed ...string) string {
	v, ok := b.m[key]
	if !ok {
		return def
	}
	if s, isStr := v.(string); isStr {
		for _, a := range allowed {
			if a == s {
				return s
			}
		}
	}
	quoted := make([]string, len(allowed))
	for i, a := range allowed {
		quoted[i] = "'" + a + "'"
	}
	exp := strings.Join(quoted[:len(quoted)-1], ", ") + " or " + quoted[len(quoted)-1]
	b.fail(key, "enum", "Input should be "+exp, v, map[string]any{"expected": exp})
	return def
}

func (b *bodyReader) stringList(key string, uuids bool) []string {
	v, ok := b.m[key]
	if !ok || v == nil {
		return nil
	}
	arr, isArr := v.([]any)
	if !isArr {
		b.fail(key, "list_type", "Input should be a valid list", v, nil)
		return nil
	}
	out := make([]string, 0, len(arr))
	for i, it := range arr {
		s, isStr := it.(string)
		if !isStr {
			b.errs.add(fieldError{Type: "string_type", Loc: append(append([]any{}, b.loc...), key, i), Msg: "Input should be a valid string", Input: it})
			continue
		}
		if uuids {
			u, ok := domain.CanonicalUUID(s)
			if !ok {
				b.errs.add(fieldError{Type: "uuid_parsing", Loc: append(append([]any{}, b.loc...), key, i), Msg: "Input should be a valid UUID", Input: s})
				continue
			}
			s = u
		}
		out = append(out, s)
	}
	return out
}

// parseExperiment reads the experiment options; only keys that are present
// are set, the caller applies them to its defaults.
func parseExperiment(raw any, loc ...any) (benchmark.ExperimentOverride, validationErrors) {
	if len(loc) == 0 {
		loc = []any{"body", "experiment"}
	}
	var ov benchmark.ExperimentOverride
	em, isObj := raw.(map[string]any)
	if !isObj {
		var errs validationErrors
		errs.add(fieldError{Type: "model_attributes_type", Loc: loc, Msg: "Input should be a valid dictionary or object to extract fields from", Input: raw})
		return ov, errs
	}
	eb := &bodyReader{m: em, loc: loc}
	if _, ok := em["pipeline"]; ok {
		v := eb.enum("pipeline", "upload", "upload", "direct_vision", "ocr")
		ov.Pipeline = &v
	}
	if v, ok := em["temperature"]; ok && v != nil {
		n, isNum := v.(json.Number)
		f, err := strconv.ParseFloat(string(n), 64)
		switch {
		case !isNum || err != nil:
			eb.fail("temperature", "float_type", "Input should be a valid number", v, nil)
		case f < 0:
			eb.fail("temperature", "greater_than_equal", "Input should be greater than or equal to 0", v, map[string]any{"ge": 0})
		case f > 2:
			eb.fail("temperature", "less_than_equal", "Input should be less than or equal to 2", v, map[string]any{"le": 2})
		default:
			ov.Temperature = &f
		}
	}
	if _, ok := em["think"]; ok {
		ov.ThinkSet, ov.Think = true, eb.optBoolPtr("think")
	}
	if _, ok := em["include_classification"]; ok {
		v := eb.optBool("include_classification", true)
		ov.IncludeClassification = &v
	}
	if _, ok := em["allow_ocr_fallback"]; ok {
		v := eb.optBool("allow_ocr_fallback", false)
		ov.AllowOCRFallback = &v
	}
	if raw, ok := em["provider_options"]; ok {
		ov.ProviderOptionsSet = true
		switch po := raw.(type) {
		case nil:
		case map[string]any:
			ov.ProviderOptions = po
		default:
			eb.fail("provider_options", "dict_type", "Input should be a valid dictionary", raw, nil)
		}
	}
	return ov, eb.errs
}

// parseVariants reads the sweep arms: [{label, experiment}].
func parseVariants(b *bodyReader) []benchmark.Variant {
	raw, ok := b.m["variants"]
	if !ok || raw == nil {
		return nil
	}
	arr, isArr := raw.([]any)
	if !isArr {
		b.fail("variants", "list_type", "Input should be a valid list", raw, nil)
		return nil
	}
	if len(arr) < 1 || len(arr) > 10 {
		b.fail("variants", "too_short", "List should have between 1 and 10 items", raw, map[string]any{"min_length": 1, "max_length": 10})
		return nil
	}
	out := make([]benchmark.Variant, 0, len(arr))
	for i, it := range arr {
		vm, isObj := it.(map[string]any)
		if !isObj {
			b.errs.add(fieldError{Type: "model_attributes_type", Loc: []any{"body", "variants", i}, Msg: "Input should be a valid dictionary or object to extract fields from", Input: it})
			continue
		}
		vb := &bodyReader{m: vm, loc: []any{"body", "variants", i}}
		v := benchmark.Variant{Label: vb.label()}
		if e, ok := vm["experiment"]; ok {
			ov, errs := parseExperiment(e, "body", "variants", i, "experiment")
			v.Experiment = ov
			vb.errs = append(vb.errs, errs...)
		}
		b.errs = append(b.errs, vb.errs...)
		out = append(out, v)
	}
	return out
}

func parseBenchmarkCreate(m map[string]any) (benchmark.CreateRequest, validationErrors) {
	b := &bodyReader{m: m, loc: []any{"body"}}
	req := benchmark.CreateRequest{Experiment: benchmark.ExperimentOptions{Pipeline: "upload", IncludeClassification: true}}
	if raw, ok := m["experiment"]; ok {
		ov, errs := parseExperiment(raw)
		req.Experiment = ov.Apply(req.Experiment)
		b.errs = append(b.errs, errs...)
	}
	req.Label = b.label()
	req.Variants = parseVariants(b)
	req.ModelIDs = b.stringList("model_ids", false)
	req.ReceiptIDs = b.stringList("receipt_ids", true)
	req.Category = b.optString("category")
	req.MaxSourceConfidence = b.optInt("max_source_confidence", 0, 100)
	req.Limit = b.optInt("limit", 1, 0)
	req.TimeoutSeconds = 300
	if t := b.optInt("request_timeout_seconds", 1, 0); t != nil {
		req.TimeoutSeconds = *t
	}
	req.CouncilPolicy = b.enum("council_policy", "all", "all", "material", "custom")
	req.AbsoluteThreshold = b.optNonNegDecimal("council_absolute_threshold")
	req.RelativeThreshold = b.optNonNegDecimal("council_relative_threshold")
	req.ApplyCouncilFlags = b.optBool("apply_council_flags", false)
	return req, b.errs
}

func (s *Server) createBenchmark(w http.ResponseWriter, r *http.Request, _ userView) {
	m, _, ok := readObject(w, r)
	if !ok {
		return
	}
	req, errs := parseBenchmarkCreate(m)
	if len(errs) > 0 {
		errs.write(w)
		return
	}
	runs, err := s.opts.Benchmarks.CreateRun(r.Context(), req)
	var ue *benchmark.UserError
	if errors.As(err, &ue) {
		writeError(w, http.StatusUnprocessableEntity, ue.Msg)
		return
	}
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if len(req.Variants) > 0 {
		// A sweep: the runs share a comparison_id; compare them with
		// GET /benchmarks/compare?runs=... or list them by comparison_id.
		cid := *runs[0].ComparisonID
		w.Header().Set("Location", "/api/v1/benchmarks?comparison_id="+cid)
		writeJSON(w, http.StatusAccepted, map[string]any{"comparison_id": cid, "runs": runs})
		return
	}
	w.Header().Set("Location", "/api/v1/benchmarks/"+runs[0].ID)
	writeJSON(w, http.StatusAccepted, runs[0])
}

func (s *Server) listBenchmarks(w http.ResponseWriter, r *http.Request, _ userView) {
	p := newParams(r)
	limit, offset := p.intParam("limit", 50, 1, 500), p.intParam("offset", 0, 0, 0)
	var comparison *string
	if q := r.URL.Query().Get("comparison_id"); q != "" {
		if u, ok := domain.CanonicalUUID(q); ok {
			comparison = &u
		} else {
			p.errs.add(fieldError{Type: "uuid_parsing", Loc: []any{"query", "comparison_id"}, Msg: "Input should be a valid UUID", Input: q})
		}
	}
	if !p.ok(w) {
		return
	}
	runs, err := s.db().BenchmarkRuns(r.Context(), comparison)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if offset > len(runs) {
		offset = len(runs)
	}
	end := min(offset+limit, len(runs))
	writeJSON(w, http.StatusOK, runs[offset:end])
}

func (s *Server) benchmarkResults(w http.ResponseWriter, r *http.Request, _ userView) {
	p := newParams(r)
	id := p.pathUUID("run_id")
	limit, offset := p.intParam("limit", 50, 1, 200), p.intParam("offset", 0, 0, 0)
	if !p.ok(w) {
		return
	}
	res, err := s.opts.Benchmarks.Results(r.Context(), id, limit, offset)
	if errors.Is(err, benchmark.ErrRunNotFound) {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) benchmarkStatus(w http.ResponseWriter, r *http.Request, _ userView) {
	p := newParams(r)
	id := p.pathUUID("run_id")
	if !p.ok(w) {
		return
	}
	st, err := s.opts.Benchmarks.Status(r.Context(), id)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if st == nil {
		writeError(w, http.StatusNotFound, "Benchmark run not found")
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) reevaluateBenchmark(w http.ResponseWriter, r *http.Request, _ userView) {
	p := newParams(r)
	id := p.pathUUID("run_id")
	m, _, ok := readObject(w, r)
	if !ok {
		return
	}
	var errs validationErrors
	errs = append(errs, p.errs...)
	var receiptID, modelID string
	if v, ok := m["receipt_id"]; !ok {
		errs.add(fieldError{Type: "missing", Loc: bodyLoc("receipt_id"), Msg: "Field required"})
	} else if str, isStr := v.(string); !isStr {
		errs.add(fieldError{Type: "uuid_type", Loc: bodyLoc("receipt_id"), Msg: "UUID input should be a string, bytes or UUID object", Input: v})
	} else if u, ok := domain.CanonicalUUID(str); !ok {
		errs.add(fieldError{Type: "uuid_parsing", Loc: bodyLoc("receipt_id"), Msg: "Input should be a valid UUID", Input: str})
	} else {
		receiptID = u
	}
	if v, ok := m["model_id"]; !ok {
		errs.add(fieldError{Type: "missing", Loc: bodyLoc("model_id"), Msg: "Field required"})
	} else if str, isStr := v.(string); !isStr {
		errs.add(fieldError{Type: "string_type", Loc: bodyLoc("model_id"), Msg: "Input should be a valid string", Input: v})
	} else if str == "" {
		errs.add(fieldError{Type: "string_too_short", Loc: bodyLoc("model_id"), Msg: "String should have at least 1 character", Input: str, Ctx: map[string]any{"min_length": 1}})
	} else {
		modelID = str
	}
	if len(errs) > 0 {
		errs.write(w)
		return
	}
	res, err := s.opts.Benchmarks.Reevaluate(r.Context(), id, receiptID, modelID)
	var ue *benchmark.UserError
	switch {
	case errors.As(err, &ue):
		writeError(w, http.StatusNotFound, ue.Msg)
	case errors.Is(err, benchmark.ErrTimeout):
		writeError(w, http.StatusGatewayTimeout, err.Error())
	case err != nil:
		s.internalError(w, r, err)
	default:
		writeJSON(w, http.StatusOK, res)
	}
}

func (s *Server) rerunBenchmark(w http.ResponseWriter, r *http.Request, _ userView) {
	p := newParams(r)
	id := p.pathUUID("run_id")
	m, _, ok := readObject(w, r)
	if !ok {
		return
	}
	b := &bodyReader{m: m, loc: []any{"body"}}
	b.errs = append(b.errs, p.errs...)
	var req benchmark.RerunRequest
	req.ModelIDs = b.stringList("model_ids", false)
	if raw, ok := m["experiment"]; ok {
		ov, errs := parseExperiment(raw)
		req.Experiment = ov
		b.errs = append(b.errs, errs...)
	}
	req.Label = b.label()
	if _, ok := m["provider_options"]; ok {
		// Top-level shorthand from the plan; it wins over experiment.provider_options.
		ov, errs := parseExperiment(map[string]any{"provider_options": m["provider_options"]}, "body")
		req.Experiment.ProviderOptions, req.Experiment.ProviderOptionsSet = ov.ProviderOptions, ov.ProviderOptionsSet
		b.errs = append(b.errs, errs...)
	}
	if len(b.errs) > 0 {
		b.errs.write(w)
		return
	}
	res, err := s.opts.Benchmarks.Rerun(r.Context(), id, req)
	var ue *benchmark.UserError
	switch {
	case errors.Is(err, benchmark.ErrRunNotFound):
		writeError(w, http.StatusNotFound, err.Error())
	case errors.As(err, &ue):
		writeError(w, http.StatusUnprocessableEntity, ue.Msg)
	case err != nil:
		s.internalError(w, r, err)
	default:
		w.Header().Set("Location", "/api/v1/benchmarks/"+res.ID)
		writeJSON(w, http.StatusAccepted, res)
	}
}

func (s *Server) compareBenchmarks(w http.ResponseWriter, r *http.Request, _ userView) {
	var ids []string
	var errs validationErrors
	seen := map[string]bool{}
	raw := r.URL.Query().Get("runs")
	if raw == "" {
		errs.add(fieldError{Type: "missing", Loc: []any{"query", "runs"}, Msg: "Field required"})
	}
	for _, part := range strings.Split(raw, ",") {
		if raw == "" {
			break
		}
		u, ok := domain.CanonicalUUID(strings.TrimSpace(part))
		if !ok {
			errs.add(fieldError{Type: "uuid_parsing", Loc: []any{"query", "runs"}, Msg: "Input should be a valid UUID", Input: part})
			continue
		}
		if !seen[u] {
			seen[u] = true
			ids = append(ids, u)
		}
	}
	if len(errs) == 0 && (len(ids) < 1 || len(ids) > 10) {
		errs.add(fieldError{Type: "value_error", Loc: []any{"query", "runs"}, Msg: "Value error, runs must list 1 to 10 distinct run ids", Input: raw})
	}
	if len(errs) > 0 {
		errs.write(w)
		return
	}
	res, err := s.opts.Benchmarks.Compare(r.Context(), ids)
	switch {
	case errors.Is(err, benchmark.ErrRunNotFound):
		writeError(w, http.StatusNotFound, err.Error())
	case err != nil:
		s.internalError(w, r, err)
	default:
		writeJSON(w, http.StatusOK, res)
	}
}

func (s *Server) compareCandidates(w http.ResponseWriter, r *http.Request, _ userView) {
	p := newParams(r)
	id := p.pathUUID("run_id")
	if !p.ok(w) {
		return
	}
	res, err := s.opts.Benchmarks.CompareCandidates(r.Context(), id)
	switch {
	case errors.Is(err, benchmark.ErrRunNotFound):
		writeError(w, http.StatusNotFound, "Benchmark run not found")
	case err != nil:
		s.internalError(w, r, err)
	default:
		writeJSON(w, http.StatusOK, res)
	}
}
