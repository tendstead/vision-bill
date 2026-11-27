package httpapi

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
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
	// maxUploadBytes caps one whole request (all photos plus multipart
	// framing). Photos are streamed to disk, so this bounds disk and time,
	// not memory.
	maxUploadBytes      = 200 * 1024 * 1024
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

// upload is one received photo, already stored in the temporary area.
type upload struct {
	filename string
	path     string
	size     int64
}

func isTimeout(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// readUploads streams the "receipt" file fields to temporary files, so a
// request of up to maxUploadBytes never has to fit in memory. ok=false means
// a response was written and every stored file was removed again.
func (s *Server) readUploads(w http.ResponseWriter, r *http.Request) ([]upload, bool) {
	var out []upload
	abort := func(write func()) ([]upload, bool) {
		for _, up := range out {
			s.opts.Files.Delete(up.path)
		}
		write()
		return nil, false
	}
	missing := func() ([]upload, bool) {
		return abort(func() {
			validationErrors{{Type: "missing", Loc: bodyLoc("receipt"), Msg: "Field required"}}.write(w)
		})
	}
	// failed maps a read/copy error to its response.
	failed := func(err error) ([]upload, bool) {
		var tooBig *http.MaxBytesError
		switch {
		case errors.As(err, &tooBig):
			return abort(func() { writeError(w, http.StatusRequestEntityTooLarge, "An upload must be at most 200 MB in total") })
		case errors.Is(err, files.ErrTooLarge):
			return abort(func() { writeError(w, http.StatusRequestEntityTooLarge, "Each photo must be at most 20 MB") })
		case isTimeout(err):
			return abort(func() { writeError(w, http.StatusRequestTimeout, "The upload took too long") })
		}
		return abort(func() { writeError(w, http.StatusBadRequest, "Could not read upload") })
	}
	ctype := r.Header.Get("Content-Type")
	if !strings.HasPrefix(strings.ToLower(ctype), "multipart/form-data") {
		return missing()
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	mr, err := r.MultipartReader()
	if err != nil {
		return missing()
	}
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) || isTimeout(err) {
				return failed(err)
			}
			return missing()
		}
		filename := rawFileName(part)
		if part.FormName() != "receipt" || filename == "" {
			_ = part.Close()
			continue
		}
		if len(out) >= maxPhotos {
			_ = part.Close()
			return abort(func() { writeError(w, http.StatusUnprocessableEntity, "Upload between 1 and 10 photos of one receipt") })
		}
		path, size, err := s.opts.Files.StoreTmpStream(part, maxPhotoBytes)
		_ = part.Close()
		if err != nil {
			return failed(err)
		}
		out = append(out, upload{filename: filename, path: path, size: size})
	}
	if len(out) == 0 {
		return missing()
	}
	return out, true
}

// rawFileName returns the client-supplied file name unchanged (the standard
// library reduces it to a base name). It is only stored as metadata: files on
// disk are always named by generated identifiers.
func rawFileName(p *multipart.Part) string {
	_, params, err := mime.ParseMediaType(p.Header.Get("Content-Disposition"))
	if err != nil {
		return p.FileName()
	}
	return params["filename"]
}

func (s *Server) uploadImage(w http.ResponseWriter, r *http.Request, u userView) {
	// Bound concurrent uploads (disk, CPU for validation and thumbnails). The
	// slot is released once the photos are stored, not held during analysis.
	select {
	case s.uploadSem <- struct{}{}:
	case <-r.Context().Done():
		return
	}
	releaseSlot := sync.OnceFunc(func() { <-s.uploadSem })
	defer releaseSlot()
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
		for _, up := range uploads {
			fs.Delete(up.path)
		}
	}
	for _, up := range uploads {
		mt, err := fs.ValidateFile(ctx, up.path)
		if err != nil {
			cleanup()
			var heic files.HEICUnavailableError
			var undecodable *files.UndecodableError
			switch {
			case errors.As(err, &heic):
				writeError(w, http.StatusUnsupportedMediaType, heic.Error())
			case errors.As(err, &undecodable):
				s.opts.Logger.Warn("image could not be decoded", "file", up.filename, "err", undecodable.Err)
				writeError(w, http.StatusUnprocessableEntity, undecodable.Error())
			default:
				writeError(w, http.StatusUnsupportedMediaType, "Unsupported image type")
			}
			return
		}
		name, size := up.filename, up.size
		part := domain.ImagePart{ImagePath: up.path, OriginalFilename: &name, MediaType: &mt, SizeBytes: &size}
		parts = append(parts, part)
		if thumb := fs.GenerateThumbnail(ctx, up.path); thumb != "" {
			parts[len(parts)-1].ThumbnailPath = &thumb
		}
	}
	releaseSlot()

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
		case errors.As(err, &timeout):
			// Not final: the image stays queued and the worker retries it later.
			_ = st.MarkImageTimedOut(actx, row.ID, analysis.TimedOutMessage)
			writeJSONHeaders(w, http.StatusGatewayTimeout, loc, map[string]string{"detail": err.Error(), "image_id": row.ID, "status": "timed_out"})
			return
		case errors.As(err, &unreadable):
			_ = st.MarkImageTerminal(actx, row.ID, "unreadable", err.Error())
			writeJSONHeaders(w, http.StatusUnprocessableEntity, loc, map[string]string{"detail": err.Error(), "image_id": row.ID, "status": "unreadable"})
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

// analyzePending lets every user wake the analysis worker. Only an
// administrator runs the cycle synchronously and sees its results, because they
// describe the whole queue (other users' images and error texts).
func (s *Server) analyzePending(w http.ResponseWriter, r *http.Request, u userView) {
	if !u.IsAdmin {
		s.opts.Scheduler.Trigger()
		writeJSON(w, http.StatusOK, map[string]any{"results": []analysis.Result{}})
		return
	}
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
