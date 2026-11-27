package httpapi

import (
	"net/http"

	"github.com/tendstead/vision-bill/internal/llm"
)

func (s *Server) llmRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/llm/models", s.authed(s.listModels))
}

// listModels asks the provider for its vision-capable models. The provider is
// built from the current settings, so host changes apply without a restart.
func (s *Server) listModels(w http.ResponseWriter, r *http.Request, _ userView) {
	provider, err := llm.New(s.cfg().LLM)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	models, err := provider.Models(r.Context())
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if models == nil {
		models = []llm.ModelInfo{}
	}
	writeJSON(w, http.StatusOK, models)
}
