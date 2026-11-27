// Package httpapi wires the HTTP surface: JSON API routes, health endpoints
// and the embedded single-page app.
package httpapi

import (
	"encoding/json"
	"io/fs"
	"net/http"
	"path"
	"strings"
)

// Server holds the dependencies shared by handlers.
type Server struct {
	static fs.FS
}

// New returns the application handler. static may be nil (no SPA build).
func New(static fs.FS) http.Handler {
	s := &Server{static: static}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("/api/", s.apiNotFound)
	mux.HandleFunc("/", s.spa)
	return mux
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError uses FastAPI's {"detail": ...} error body.
func writeError(w http.ResponseWriter, status int, detail string) {
	writeJSON(w, status, map[string]string{"detail": detail})
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// apiNotFound keeps unknown /api paths as JSON 404s instead of the SPA shell.
func (s *Server) apiNotFound(w http.ResponseWriter, _ *http.Request) {
	writeError(w, http.StatusNotFound, "Not Found")
}

// spa serves built assets and falls back to index.html for client routes.
func (s *Server) spa(w http.ResponseWriter, r *http.Request) {
	if s.static == nil || (r.Method != http.MethodGet && r.Method != http.MethodHead) {
		writeError(w, http.StatusNotFound, "Not Found")
		return
	}
	name := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
	if name == "" {
		name = "index.html"
	}
	if info, err := fs.Stat(s.static, name); err != nil || info.IsDir() {
		if path.Ext(name) != "" {
			writeError(w, http.StatusNotFound, "Not Found")
			return
		}
		name = "index.html"
	}
	if _, err := fs.Stat(s.static, name); err != nil {
		writeError(w, http.StatusNotFound, "Not Found")
		return
	}
	http.ServeFileFS(w, r, s.static, name)
}
