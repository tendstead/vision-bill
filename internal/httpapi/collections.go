package httpapi

import (
	"errors"
	"net/http"
	"time"

	"github.com/tendstead/vision-bill/internal/domain"
)

func (s *Server) collectionRoutes(mux *http.ServeMux) {
	const base = "/api/v1/collections"
	mux.HandleFunc("GET "+base, s.authed(s.listCollections))
	mux.HandleFunc("POST "+base, s.authed(s.createCollection))
	mux.HandleFunc("GET "+base+"/active", s.authed(s.activeCollection))
	mux.HandleFunc("GET "+base+"/{collection_id}", s.authed(s.getCollection))
	mux.HandleFunc("PATCH "+base+"/{collection_id}", s.authed(s.updateCollection))
	mux.HandleFunc("DELETE "+base+"/{collection_id}", s.authed(s.deleteCollection))
	mux.HandleFunc("POST "+base+"/{collection_id}/receipts/{receipt_id}", s.authed(s.assignReceipt))
	mux.HandleFunc("DELETE "+base+"/{collection_id}/receipts/{receipt_id}", s.authed(s.unassignReceipt))
	mux.HandleFunc("POST "+base+"/{collection_id}/activate", s.authed(s.activateCollection))
	mux.HandleFunc("POST "+base+"/{collection_id}/deactivate", s.authed(s.deactivateCollection))
}

func (s *Server) listCollections(w http.ResponseWriter, r *http.Request, u userView) {
	out, err := s.db().ListCollections(r.Context(), scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, out)
}

type collectionFields struct {
	name     *string
	nameSet  bool
	color    *string
	colorSet bool
	start    *time.Time
	startSet bool
	end      *time.Time
	endSet   bool
	active   bool
	errs     validationErrors
	isCreate bool
}

// parseCollectionBody validates CollectionCreate / CollectionUpdate bodies.
func parseCollectionBody(m map[string]any, create bool) collectionFields {
	f := collectionFields{isCreate: create}
	loc := func(k string) []any { return []any{"body", k} }
	if v, ok := m["name"]; ok && (v != nil || create) {
		f.nameSet = true
		s, isStr := v.(string)
		switch {
		case !isStr:
			f.errs.add(fieldError{Type: "string_type", Loc: loc("name"), Msg: "Input should be a valid string", Input: v})
		case len([]rune(s)) < 1:
			f.errs.add(fieldError{Type: "string_too_short", Loc: loc("name"), Msg: "String should have at least 1 character", Input: s, Ctx: map[string]any{"min_length": 1}})
		case len([]rune(s)) > 100:
			f.errs.add(fieldError{Type: "string_too_long", Loc: loc("name"), Msg: "String should have at most 100 characters", Input: s, Ctx: map[string]any{"max_length": 100}})
		default:
			n := domain.NormalizeCollectionName(s)
			if n == "" {
				f.errs.add(fieldError{Type: "value_error", Loc: loc("name"), Msg: "Value error, name must not be blank", Input: s, Ctx: map[string]any{"error": map[string]any{}}})
			}
			f.name = &n
		}
	} else if !ok && create {
		f.errs.add(fieldError{Type: "missing", Loc: loc("name"), Msg: "Field required"})
	}
	if v, ok := m["color"]; ok && v != nil {
		if s, isStr := v.(string); isStr {
			f.color, f.colorSet = &s, true
		} else {
			f.errs.add(fieldError{Type: "string_type", Loc: loc("color"), Msg: "Input should be a valid string", Input: v})
		}
	}
	date := func(key string) *time.Time {
		v, ok := m[key]
		if !ok || v == nil {
			return nil
		}
		str, isStr := v.(string)
		if !isStr {
			f.errs.add(fieldError{Type: "date_type", Loc: loc(key), Msg: "Input should be a valid date", Input: v})
			return nil
		}
		d, err := domain.ParseDate(str)
		if err != nil {
			f.errs.add(fieldError{Type: "date_from_datetime_parsing", Loc: loc(key), Msg: "Input should be a valid date or datetime, invalid character in year", Input: str})
			return nil
		}
		t := d.Time
		return &t
	}
	f.start, f.end = date("start_date"), date("end_date")
	if create {
		if v, ok := m["active"]; ok {
			switch b := v.(type) {
			case bool:
				f.active = b
			default:
				f.errs.add(fieldError{Type: "bool_type", Loc: loc("active"), Msg: "Input should be a valid boolean", Input: v})
			}
		}
	}
	return f
}

func validateDates(start, end *time.Time) error {
	if start != nil && end != nil && start.After(*end) {
		return errors.New("start_date must be on or before end_date")
	}
	return nil
}

func (s *Server) createCollection(w http.ResponseWriter, r *http.Request, u userView) {
	m, _, ok := readObject(w, r)
	if !ok {
		return
	}
	f := parseCollectionBody(m, true)
	if len(f.errs) > 0 {
		f.errs.write(w)
		return
	}
	name := domain.NormalizeCollectionName(*f.name)
	if name == "" {
		writeError(w, http.StatusUnprocessableEntity, "Collection name must not be blank")
		return
	}
	if f.color != nil && !domain.IsValidHexColor(*f.color) {
		writeError(w, http.StatusUnprocessableEntity, "color must be a hex value like #RGB or #RRGGBB")
		return
	}
	if err := validateDates(f.start, f.end); err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	st := s.db()
	existing, err := st.FindCollectionByName(r.Context(), u.ID, name)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if existing != nil {
		writeError(w, http.StatusConflict, "A collection named '"+name+"' already exists")
		return
	}
	c, err := st.CreateCollection(r.Context(), u.ID, name, f.color, f.start, f.end, f.active)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusCreated, c)
}

// activeCollection answers 204 with an empty body when none is active.
func (s *Server) activeCollection(w http.ResponseWriter, r *http.Request, u userView) {
	c, err := s.db().ActiveCollection(r.Context(), scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if c == nil {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	writeJSON(w, http.StatusOK, c)
}

func (s *Server) getCollection(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	id := p.pathUUID("collection_id")
	if !p.ok(w) {
		return
	}
	d, err := s.db().CollectionDetail(r.Context(), id, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if d == nil {
		writeError(w, http.StatusNotFound, "Collection not found")
		return
	}
	writeJSON(w, http.StatusOK, d)
}

func (s *Server) updateCollection(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	id := p.pathUUID("collection_id")
	m, _, ok := readObject(w, r)
	if !ok {
		return
	}
	f := parseCollectionBody(m, false)
	p.errs = append(p.errs, f.errs...)
	if !p.ok(w) {
		return
	}
	st, sc := s.db(), scopeOf(u)
	current, err := st.GetCollection(r.Context(), id, sc)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if current == nil {
		writeError(w, http.StatusNotFound, "Collection not found")
		return
	}
	name := current.Name
	if f.name != nil {
		name = domain.NormalizeCollectionName(*f.name)
	}
	if name == "" {
		writeError(w, http.StatusUnprocessableEntity, "Collection name must not be blank")
		return
	}
	color := current.Color
	if f.color != nil {
		color = f.color
	}
	start, end := dateOrNil(current.StartDate), dateOrNil(current.EndDate)
	if f.start != nil {
		start = f.start
	}
	if f.end != nil {
		end = f.end
	}
	if color != nil && !domain.IsValidHexColor(*color) {
		writeError(w, http.StatusUnprocessableEntity, "color must be a hex value like #RGB or #RRGGBB")
		return
	}
	if err := validateDates(start, end); err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	if f.name != nil {
		owner := u.ID
		if owner == "" {
			owner = "00000000-0000-0000-0000-000000000000"
		}
		existing, err := st.FindCollectionByName(r.Context(), owner, name)
		if err != nil {
			s.internalError(w, r, err)
			return
		}
		if existing != nil && existing.ID != id {
			writeError(w, http.StatusConflict, "A collection named '"+name+"' already exists")
			return
		}
	}
	c, err := st.UpdateCollection(r.Context(), id, sc, name, color, start, end)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if c == nil {
		writeError(w, http.StatusNotFound, "Collection not found")
		return
	}
	writeJSON(w, http.StatusOK, c)
}

func dateOrNil(d *domain.Date) *time.Time {
	if d == nil {
		return nil
	}
	t := d.Time
	return &t
}

func (s *Server) deleteCollection(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	id := p.pathUUID("collection_id")
	if !p.ok(w) {
		return
	}
	ok, err := s.db().DeleteCollection(r.Context(), id, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if !ok {
		writeError(w, http.StatusNotFound, "Collection not found")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) assignReceipt(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	cid, rid := p.pathUUID("collection_id"), p.pathUUID("receipt_id")
	if !p.ok(w) {
		return
	}
	st, sc := s.db(), scopeOf(u)
	hasCollection, err := st.CollectionExists(r.Context(), cid, sc)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	var receipt *domain.ReceiptRow
	if hasCollection {
		if receipt, err = st.ReceiptByID(r.Context(), rid, sc); err != nil {
			s.internalError(w, r, err)
			return
		}
	}
	if !hasCollection || receipt == nil {
		writeError(w, http.StatusNotFound, "Collection or receipt not found")
		return
	}
	if err := st.AssignReceipt(r.Context(), cid, rid); err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"assigned": true})
}

func (s *Server) unassignReceipt(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	cid, rid := p.pathUUID("collection_id"), p.pathUUID("receipt_id")
	if !p.ok(w) {
		return
	}
	st := s.db()
	exists, err := st.CollectionExists(r.Context(), cid, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if !exists {
		writeError(w, http.StatusNotFound, "Collection not found")
		return
	}
	removed, err := st.UnassignReceipt(r.Context(), cid, rid)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if !removed {
		writeError(w, http.StatusNotFound, "Not assigned")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) activateCollection(w http.ResponseWriter, r *http.Request, u userView) {
	s.toggleCollection(w, r, u, true)
}

func (s *Server) deactivateCollection(w http.ResponseWriter, r *http.Request, u userView) {
	s.toggleCollection(w, r, u, false)
}

func (s *Server) toggleCollection(w http.ResponseWriter, r *http.Request, u userView, activate bool) {
	p := newParams(r)
	id := p.pathUUID("collection_id")
	if !p.ok(w) {
		return
	}
	var c *domain.Collection
	var err error
	if activate {
		c, err = s.db().ActivateCollection(r.Context(), id, scopeOf(u))
	} else {
		c, err = s.db().DeactivateCollection(r.Context(), id, scopeOf(u))
	}
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if c == nil {
		writeError(w, http.StatusNotFound, "Collection not found")
		return
	}
	writeJSON(w, http.StatusOK, c)
}
