package httpapi

import (
	"errors"
	"net/http"
	"strings"

	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/store"
)

func (s *Server) receiptRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/receipts", s.authed(s.listReceipts))
	mux.HandleFunc("GET /api/v1/receipts/{receipt_id}", s.authed(s.getReceipt))
	mux.HandleFunc("PUT /api/v1/receipts/{receipt_id}", s.authed(s.updateReceipt))
	mux.HandleFunc("POST /api/v1/receipts/{receipt_id}/verify", s.authed(s.verifyReceipt))
	mux.HandleFunc("DELETE /api/v1/receipts/{receipt_id}", s.authed(s.deleteReceipt))
}

func (s *Server) listReceipts(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	f := store.ListFilter{
		Limit:  p.intParam("limit", 50, 1, 500),
		Offset: p.intParam("offset", 0, 0, 0),
	}
	status, hasStatus := p.optString("status")
	f.DateFrom, f.DateTo = p.dateParam("date_from"), p.dateParam("date_to")
	if v, ok := p.optString("search"); ok {
		f.Search = v
	}
	f.CollectionID = p.uuidParam("collection_id")
	if !p.ok(w) {
		return
	}
	if hasStatus {
		for _, part := range strings.Split(status, ",") {
			if part = strings.TrimSpace(part); part != "" {
				f.Status = append(f.Status, part)
			}
		}
	}
	rows, err := s.db().ListReceipts(r.Context(), f, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, rows)
}

func (s *Server) getReceipt(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	id := p.pathUUID("receipt_id")
	if !p.ok(w) {
		return
	}
	d, err := s.db().ReceiptDetails(r.Context(), id, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if d == nil {
		writeError(w, http.StatusNotFound, "Receipt not found")
		return
	}
	writeJSON(w, http.StatusOK, d)
}

// updateReceipt commits the receipt edit first; the requested collection
// memberships are validated and applied afterwards (a bad collection yields
// 404 but the edit persists).
func (s *Server) updateReceipt(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	id := p.pathUUID("receipt_id")
	_, raw, ok := readObject(w, r)
	if !ok {
		return
	}
	receipt, issues := domain.ParseReceipt(raw, domain.Options{Update: true})
	if issues != nil {
		writeIssues(w, issues, "body")
		return
	}
	if !p.ok(w) {
		return
	}
	st := s.db()
	row, err := st.UpdateReceipt(r.Context(), id, receipt, scopeOf(u))
	var verr *store.ValidationError
	if errors.As(err, &verr) {
		writeError(w, http.StatusUnprocessableEntity, verr.Msg)
		return
	}
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if row == nil {
		writeError(w, http.StatusNotFound, "Receipt not found")
		return
	}
	if receipt.CollectionIDs != nil {
		if err := st.SetReceiptCollections(r.Context(), id, *receipt.CollectionIDs, scopeOf(u)); err != nil {
			if errors.Is(err, store.ErrNotFound) {
				writeError(w, http.StatusNotFound, "Collection not found")
				return
			}
			s.internalError(w, r, err)
			return
		}
	}
	writeJSON(w, http.StatusOK, row)
}

// verifyReceipt marks a receipt verified after moving its photos to permanent
// storage. The move is repeatable, so a retry after an interruption recovers.
func (s *Server) verifyReceipt(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	id := p.pathUUID("receipt_id")
	if !p.ok(w) {
		return
	}
	st, sc := s.db(), scopeOf(u)
	row, err := st.ReceiptByID(r.Context(), id, sc)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if row == nil {
		writeError(w, http.StatusNotFound, "Receipt not found")
		return
	}
	if row.Status == "verified" {
		writeError(w, http.StatusConflict, "Receipt already verified")
		return
	}
	if row.ImageID != nil {
		img, err := st.ImageByID(r.Context(), *row.ImageID, sc)
		if err != nil {
			s.internalError(w, r, err)
			return
		}
		if img != nil && img.ImagePath != nil && *img.ImagePath != "" {
			if err := s.opts.Analysis.MovePermanent(r.Context(), st, img, id); err != nil {
				s.internalError(w, r, err)
				return
			}
		}
	}
	verified, err := st.VerifyReceipt(r.Context(), id, sc)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if verified == nil {
		writeError(w, http.StatusNotFound, "Receipt not found")
		return
	}
	writeJSON(w, http.StatusOK, verified)
}

// deleteReceipt removes a receipt (line items and taxes cascade) and its photos.
func (s *Server) deleteReceipt(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	id := p.pathUUID("receipt_id")
	if !p.ok(w) {
		return
	}
	st, sc := s.db(), scopeOf(u)
	row, err := st.DeleteReceipt(r.Context(), id, sc)
	if errors.Is(err, store.ErrReceiptReferenced) {
		writeError(w, http.StatusConflict, "Receipt is referenced by a benchmark run and cannot be deleted")
		return
	}
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if row == nil {
		writeError(w, http.StatusNotFound, "Receipt not found")
		return
	}
	if row.ImageID != nil {
		img, err := st.ImageByID(r.Context(), *row.ImageID, sc)
		if err != nil {
			s.internalError(w, r, err)
			return
		}
		if img != nil {
			if err := st.DeleteImage(r.Context(), img.ID); err != nil {
				s.internalError(w, r, err)
				return
			}
			s.opts.Analysis.DeleteFiles(img)
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"deleted": id})
}
