// Package httpapi wires the HTTP surface: JSON API routes, health endpoints
// and the embedded single-page app.
package httpapi

import (
	"context"
	"encoding/json"
	"io/fs"
	"log/slog"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/tendstead/vision-bill/internal/analysis"
	"github.com/tendstead/vision-bill/internal/config"
	"github.com/tendstead/vision-bill/internal/files"
	"github.com/tendstead/vision-bill/internal/llm"
	"github.com/tendstead/vision-bill/internal/store"
)

// Options are the dependencies of the HTTP server.
type Options struct {
	Config *config.Holder
	// Store returns the live store, or nil while the database is unavailable.
	Store  func() *store.Store
	Static fs.FS
	Now    func() time.Time
	Logger *slog.Logger
	// ConfigSources locates the dotenv/YAML files for the settings endpoints.
	ConfigSources config.Sources
	// OnSettingsChanged is called after an admin updates runtime settings.
	OnSettingsChanged func(config.Config)
	Files             *files.Service
	Provider          llm.Provider
	Analysis          *analysis.Service
	Scheduler         *analysis.Scheduler
}

// Server holds the dependencies shared by handlers.
type Server struct {
	opts    Options
	hashSem chan struct{} // bounds concurrent Argon2 work (64 MiB each)
}

// New returns the application handler.
func New(opts Options) http.Handler {
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	if opts.Store == nil {
		opts.Store = func() *store.Store { return nil }
	}
	s := &Server{opts: opts, hashSem: make(chan struct{}, 4)}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("GET /readyz", s.readyz)
	s.routes(mux)
	mux.HandleFunc("/api/", s.apiNotFound)
	mux.HandleFunc("/", s.spa)
	return s.recoverer(mux)
}

func (s *Server) cfg() config.Config { return s.opts.Config.Get() }

func (s *Server) db() *store.Store { return s.opts.Store() }

func (s *Server) routes(mux *http.ServeMux) {
	s.authRoutes(mux)
	s.systemRoutes(mux)
	s.receiptRoutes(mux)
	s.tagRoutes(mux)
	s.collectionRoutes(mux)
	s.imageRoutes(mux)
	s.llmRoutes(mux)
}

func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				s.opts.Logger.Error("panic", "path", r.URL.Path, "panic", rec)
				writeError(w, http.StatusInternalServerError, "Internal Server Error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError writes the API's {"detail": ...} error body.
func writeError(w http.ResponseWriter, status int, detail string) {
	writeJSON(w, status, map[string]string{"detail": detail})
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// readyz reports DB connectivity and that the schema migration is recorded.
func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	st := s.db()
	if st == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "database unavailable"})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	if ok, err := store.SchemaReady(ctx, st.Pool); err != nil || !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "schema not ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// apiNotFound keeps unknown /api paths as JSON 404s instead of the SPA shell.
func (s *Server) apiNotFound(w http.ResponseWriter, _ *http.Request) {
	writeError(w, http.StatusNotFound, "Not Found")
}

// spa serves built assets and falls back to index.html for client routes.
func (s *Server) spa(w http.ResponseWriter, r *http.Request) {
	if s.opts.Static == nil || (r.Method != http.MethodGet && r.Method != http.MethodHead) {
		writeError(w, http.StatusNotFound, "Not Found")
		return
	}
	name := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
	if name == "" {
		name = "index.html"
	}
	if info, err := fs.Stat(s.opts.Static, name); err != nil || info.IsDir() {
		if path.Ext(name) != "" {
			writeError(w, http.StatusNotFound, "Not Found")
			return
		}
		name = "index.html"
	}
	if _, err := fs.Stat(s.opts.Static, name); err != nil {
		writeError(w, http.StatusNotFound, "Not Found")
		return
	}
	http.ServeFileFS(w, r, s.opts.Static, name)
}
