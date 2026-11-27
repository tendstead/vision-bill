package httpapi

import (
	"net/http"
	"strings"
	"unicode"

	"github.com/tendstead/vision-bill/internal/domain"
)

func (s *Server) tagRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/tags/categories", s.authed(func(w http.ResponseWriter, _ *http.Request, _ userView) {
		out := make([]map[string]string, len(domain.SpendingCategories))
		for i, c := range domain.SpendingCategories {
			out[i] = map[string]string{"code": c}
		}
		writeJSON(w, http.StatusOK, out)
	}))
	mux.HandleFunc("GET /api/v1/tags/category-suggestion", s.authed(s.categorySuggestion))
	mux.HandleFunc("GET /api/v1/tags", s.authed(func(w http.ResponseWriter, r *http.Request, _ userView) {
		tags, err := s.db().ListTags(r.Context())
		if err != nil {
			s.internalError(w, r, err)
			return
		}
		writeJSON(w, http.StatusOK, tags)
	}))
	mux.HandleFunc("POST /api/v1/tags", s.authed(s.createTag))
}

func (s *Server) categorySuggestion(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	merchant, description := p.reqString("merchant_name"), p.reqString("description")
	if !p.ok(w) {
		return
	}
	c, err := s.db().CategorySuggestion(r.Context(), u.ID, merchant, description)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	var source *string
	if c != nil {
		h := "history"
		source = &h
	}
	writeJSON(w, http.StatusOK, map[string]*string{"category": c, "source": source})
}

func (s *Server) createTag(w http.ResponseWriter, r *http.Request, _ userView) {
	var body struct {
		Name *string `json:"name"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	var errs validationErrors
	checkString(&errs, "name", body.Name, 1, 100)
	if len(errs) > 0 {
		errs.write(w)
		return
	}
	name := strings.ToLower(strings.Join(strings.FieldsFunc(*body.Name, unicode.IsSpace), " "))
	switch {
	case name == "":
		writeError(w, http.StatusUnprocessableEntity, "Tag name must not be blank")
		return
	case len([]rune(name)) > 100:
		writeError(w, http.StatusUnprocessableEntity, "Tag name must be at most 100 characters long")
		return
	}
	created, err := s.db().CreateTag(r.Context(), name)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	writeJSON(w, status, map[string]any{"name": name, "created": created})
}
