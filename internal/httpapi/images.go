package httpapi

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/tendstead/vision-bill/internal/analysis"
	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/files"
	"github.com/tendstead/vision-bill/internal/llm"
	"github.com/tendstead/vision-bill/internal/store"
)

const (
	maxPhotoBytes = 20 * 1024 * 1024
	maxPhotos     = 10
	// formOverhead leaves room for multipart framing around the photos.
	formOverhead        = 1 << 20
	pendingQueueWarning = "LLM provider not available – image queued for background analysis"
)

func (s *Server) imageRoutes(mux *http.ServeMux) {
	const base = "/api/v1/images"
	mux.HandleFunc("POST "+base, s.authed(s.uploadImage))
	mux.HandleFunc("GET "+base, s.authed(s.listImages))
	mux.HandleFunc("POST "+base+"/analyze", s.authed(s.analyzePending))
	mux.HandleFunc("GET "+base+"/{image_id}", s.authed(s.getImage))
	mux.HandleFunc("GET "+base+"/{image_id}/file", s.authed(s.getImageFile))
	mux.HandleFunc("GET "+base+"/{image_id}/thumb", s.authed(s.getImageThumb))
	mux.HandleFunc("DELETE "+base+"/{image_id}", s.authed(s.deleteImage))
}

func location(id string) string { return "/api/v1/images/" + id }

func writeJSONHeaders(w http.ResponseWriter, status int, headers map[string]string, v any) {
	for k, val := range headers {
		w.Header().Set(k, val)
	}
	writeJSON(w, status, v)
}

type upload struct {
	filename string
	content  []byte
}

// readUploads streams the "receipt" file fields. ok=false means a response was written.
func (s *Server) readUploads(w http.ResponseWriter, r *http.Request) ([]upload, bool) {
	missing := func() ([]upload, bool) {
		validationErrors{{Type: "missing", Loc: bodyLoc("receipt"), Msg: "Field required"}}.write(w)
		return nil, false
	}
	ctype := r.Header.Get("Content-Type")
	if !strings.HasPrefix(strings.ToLower(ctype), "multipart/form-data") {
		return missing()
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxPhotos*maxPhotoBytes+formOverhead)
	mr, err := r.MultipartReader()
	if err != nil {
		return missing()
	}
	var out []upload
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				writeError(w, http.StatusRequestEntityTooLarge, "Each photo must be at most 20 MB")
				return nil, false
			}
			validationErrors{{Type: "missing", Loc: bodyLoc("receipt"), Msg: "Field required"}}.write(w)
			return nil, false
		}
		if part.FormName() != "receipt" || part.FileName() == "" {
			_ = part.Close()
			continue
		}
		if len(out) >= maxPhotos {
			writeError(w, http.StatusUnprocessableEntity, "Upload between 1 and 10 photos of one receipt")
			return nil, false
		}
		data, err := io.ReadAll(io.LimitReader(part, maxPhotoBytes+1))
		_ = part.Close()
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				writeError(w, http.StatusRequestEntityTooLarge, "Each photo must be at most 20 MB")
				return nil, false
			}
			writeError(w, http.StatusBadRequest, "Could not read upload")
			return nil, false
		}
		if len(data) > maxPhotoBytes {
			writeError(w, http.StatusRequestEntityTooLarge, "Each photo must be at most 20 MB")
			return nil, false
		}
		out = append(out, upload{filename: part.FileName(), content: data})
	}
	if len(out) == 0 {
		return missing()
	}
	return out, true
}

func (s *Server) uploadImage(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	modelParam, hasModel := p.optString("model_id")
	bypassParam := p.boolParam("bypass_review")
	if !p.ok(w) {
		return
	}
	uploads, ok := s.readUploads(w, r)
	if !ok {
		return
	}
	cfg := s.cfg()
	bypass := cfg.Images.BypassReviewDefault
	if bypassParam != nil {
		bypass = *bypassParam
	}
	var modelID *string
	if hasModel {
		modelID = &modelParam
	}
	st, fs := s.db(), s.opts.Files
	ctx := r.Context()

	var parts []domain.ImagePart
	cleanup := func() {
		for _, part := range parts {
			fs.Delete(part.ImagePath)
		}
	}
	for _, up := range uploads {
		mt, err := fs.Inspect(up.content)
		if err != nil {
			cleanup()
			var heic files.HEICUnavailableError
			if errors.As(err, &heic) {
				writeError(w, http.StatusUnsupportedMediaType, heic.Error())
				return
			}
			writeError(w, http.StatusUnsupportedMediaType, "Unsupported image type")
			return
		}
		path, err := fs.StoreTmp(up.content)
		if err != nil {
			cleanup()
			s.internalError(w, r, err)
			return
		}
		name, size := up.filename, int64(len(up.content))
		part := domain.ImagePart{ImagePath: path, OriginalFilename: &name, MediaType: &mt, SizeBytes: &size}
		parts = append(parts, part)
		if thumb := fs.GenerateThumbnail(ctx, path); thumb != "" {
			parts[len(parts)-1].ThumbnailPath = &thumb
		}
	}

	var models []llm.ModelInfo
	if len(parts) == 1 && s.opts.Provider != nil {
		pctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		if s.opts.Provider.CheckConnection(pctx) {
			if m, err := s.opts.Provider.Models(pctx); err == nil {
				models = m
			}
		}
		cancel()
		if len(models) == 0 {
			s.opts.Logger.Warn("LLM provider unreachable - queueing image for background analysis")
		}
	}

	first := parts[0]
	row, err := st.StoreImage(ctx, store.NewImage{
		ImagePath: first.ImagePath, OriginalFilename: first.OriginalFilename, MediaType: first.MediaType,
		SizeBytes: first.SizeBytes, UserID: u.ID, BypassReview: bypass, ThumbnailPath: first.ThumbnailPath,
		Additional: parts[1:], ModelID: modelID,
	})
	if err != nil {
		cleanup()
		s.internalError(w, r, err)
		return
	}
	loc := map[string]string{"Location": location(row.ID)}

	if len(parts) > 1 {
		if s.opts.Scheduler != nil {
			s.opts.Scheduler.Trigger()
		}
		writeJSONHeaders(w, http.StatusAccepted, loc, map[string]string{"image_id": row.ID, "status": "pending"})
		return
	}
	queued := func() {
		writeJSONHeaders(w, http.StatusAccepted, loc, map[string]string{"image_id": row.ID, "status": "pending", "warning": pendingQueueWarning})
	}
	if len(models) == 0 {
		queued()
		return
	}
	// The analysis outlives a client that disconnects: the outcome is always recorded.
	actx := context.WithoutCancel(ctx)
	claimed, err := st.ClaimImage(actx, row.ID)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if claimed == nil {
		s.opts.Logger.Info("image was claimed by the background worker", "image", row.ID)
		queued()
		return
	}
	chosen := analysis.ChooseModel(models, modelID, cfg.LLM.ModelName)

	receipt, err := s.opts.Analysis.Analyse(actx, chosen, claimed.Parts()[:1])
	if err != nil {
		var unreadable *llm.UnreadableError
		var timeout *llm.TimeoutError
		switch {
		case errors.As(err, &timeout), errors.As(err, &unreadable):
			status, code := "unreadable", http.StatusUnprocessableEntity
			if timeout != nil {
				status, code = "timed_out", http.StatusGatewayTimeout
			}
			_ = st.MarkImageTerminal(actx, row.ID, status, err.Error())
			writeJSONHeaders(w, code, loc, map[string]string{"detail": err.Error(), "image_id": row.ID, "status": status})
			return
		}
		_ = st.MarkImageFailed(actx, row.ID, err.Error())
		s.internalError(w, r, err)
		return
	}

	status := "unverified"
	if bypass {
		status = "verified"
	}
	rec, err := st.PersistReceipt(actx, receipt, &row.ID, status, bypass, &u.ID)
	if err == nil {
		err = st.MarkImageAnalyzed(actx, row.ID, rec.ID)
	}
	if err != nil {
		_ = st.MarkImageFailed(actx, row.ID, fmt.Sprintf("Failed to persist receipt: %v", err))
		s.internalError(w, r, err)
		return
	}
	if bypass {
		if err := s.opts.Analysis.MovePermanent(actx, st, claimed, rec.ID); err != nil {
			s.opts.Logger.Error("failed to move bypass-reviewed image to permanent storage", "image", row.ID, "err", err)
		}
	}
	writeJSONHeaders(w, http.StatusCreated, loc, map[string]any{
		"image_id": row.ID, "status": "analyzed", "receipt_id": rec.ID,
		"original_filename": claimed.OriginalFilename, "media_type": first.MediaType,
		"size_bytes": first.SizeBytes, "image_path": claimed.ImagePath,
	})
}

func (s *Server) listImages(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	limit, offset := p.intParam("limit", 50, 1, 500), p.intParam("offset", 0, 0, 0)
	status, hasStatus := p.optString("status")
	if !p.ok(w) {
		return
	}
	var statuses []string
	if hasStatus {
		for _, part := range strings.Split(status, ",") {
			if part = strings.TrimSpace(part); part != "" {
				statuses = append(statuses, part)
			}
		}
	}
	rows, err := s.db().ListImages(r.Context(), statuses, limit, offset, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, rows)
}

func (s *Server) analyzePending(w http.ResponseWriter, r *http.Request, _ userView) {
	results, err := s.opts.Scheduler.ProcessPending(context.WithoutCancel(r.Context()))
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

func (s *Server) imageFor(w http.ResponseWriter, r *http.Request, u userView, p *params) (*domain.ImageRow, bool) {
	id := p.pathUUID("image_id")
	if !p.ok(w) {
		return nil, false
	}
	img, err := s.db().ImageByID(r.Context(), id, scopeOf(u))
	if err != nil {
		s.internalError(w, r, err)
		return nil, false
	}
	return img, true
}

func (s *Server) getImage(w http.ResponseWriter, r *http.Request, u userView) {
	img, ok := s.imageFor(w, r, u, newParams(r))
	if !ok {
		return
	}
	if img == nil {
		writeError(w, http.StatusNotFound, "Image not found")
		return
	}
	writeJSON(w, http.StatusOK, img)
}

// serveFile streams a stored file with validators for conditional requests.
func serveFile(w http.ResponseWriter, r *http.Request, path, mediaType string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || info.IsDir() {
		return false
	}
	sum := md5.Sum([]byte(strconv.FormatInt(info.ModTime().UnixNano(), 10) + "-" + strconv.FormatInt(info.Size(), 10)))
	w.Header().Set("ETag", `"`+hex.EncodeToString(sum[:])+`"`)
	w.Header().Set("Content-Type", mediaType)
	http.ServeContent(w, r, "", info.ModTime(), f)
	return true
}

func (s *Server) getImageFile(w http.ResponseWriter, r *http.Request, u userView) {
	p := newParams(r)
	part := p.intParam("part", 0, 0, 0)
	img, ok := s.imageFor(w, r, u, p)
	if !ok {
		return
	}
	if img == nil || img.ImagePath == nil || *img.ImagePath == "" {
		writeError(w, http.StatusNotFound, "Image file not found")
		return
	}
	parts := img.Parts()
	if part >= len(parts) {
		writeError(w, http.StatusNotFound, "Image part not found")
		return
	}
	src := parts[part]
	mediaType := "application/octet-stream"
	if src.MediaType != nil && *src.MediaType != "" {
		mediaType = *src.MediaType
	}
	if !serveFile(w, r, src.ImagePath, mediaType) {
		writeError(w, http.StatusNotFound, "Image file not found")
	}
}

func (s *Server) getImageThumb(w http.ResponseWriter, r *http.Request, u userView) {
	img, ok := s.imageFor(w, r, u, newParams(r))
	if !ok {
		return
	}
	// A plain file server: the thumbnail or 404, never a fallback to the full image.
	if img == nil || img.ThumbnailPath == nil || *img.ThumbnailPath == "" || !serveFile(w, r, *img.ThumbnailPath, "image/webp") {
		writeError(w, http.StatusNotFound, "Image thumbnail not found")
	}
}

func (s *Server) deleteImage(w http.ResponseWriter, r *http.Request, u userView) {
	img, ok := s.imageFor(w, r, u, newParams(r))
	if !ok {
		return
	}
	if img == nil {
		writeError(w, http.StatusNotFound, "Image not found")
		return
	}
	switch img.Status {
	case "pending", "failed", "timed_out", "unreadable":
	default:
		writeError(w, http.StatusConflict, "Cannot delete an active or analyzed submission")
		return
	}
	if err := s.db().DeleteImage(r.Context(), img.ID); err != nil {
		s.internalError(w, r, err)
		return
	}
	s.opts.Analysis.DeleteFiles(img)
	writeJSON(w, http.StatusOK, map[string]string{"deleted": img.ID})
}
