package httpapi

import (
	"net/http"
	"strings"
	"unicode/utf8"
)

func (s *Server) searchRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/search", s.authed(s.searchProducts))
	mux.HandleFunc("GET /api/v1/statistics", s.authed(s.statistics))
	mux.HandleFunc("GET /api/v1/statistics/items", s.authed(s.itemStatistics))
}

func (s *Server) searchProducts(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	query, has := p.optString("query")
	switch {
	case !has:
		p.fail("query", "query", "missing", "Field required", nil, nil)
	case query == "":
		p.fail("query", "query", "string_too_short", "String should have at least 1 character", query, map[string]any{"min_length": 1})
	case utf8.RuneCountInString(query) > 255:
		p.fail("query", "query", "string_too_long", "String should have at most 255 characters", query, map[string]any{"max_length": 255})
	}
	if !p.ok(w) {
		return
	}
	term := strings.TrimSpace(query)
	if term == "" {
		writeError(w, http.StatusUnprocessableEntity, "Query must not be blank")
		return
	}
	res, err := s.db().SearchProducts(r.Context(), term, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) statistics(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	weeks := p.intParam("weeks", 12, 1, 52)
	collection := p.uuidParam("collection_id")
	if !p.ok(w) {
		return
	}
	res, err := s.db().ReceiptStatistics(r.Context(), scopeOf(u), weeks, collection)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) itemStatistics(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	months := p.intParam("months", 12, 1, 36)
	collection := p.uuidParam("collection_id")
	if !p.ok(w) {
		return
	}
	res, err := s.db().ItemStatistics(r.Context(), scopeOf(u), months, collection, s.opts.Now())
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}
