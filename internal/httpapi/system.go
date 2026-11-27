package httpapi

import (
	"net/http"
	"strings"

	"github.com/tendstead/vision-bill/internal/config"
)

type llmSettingsView struct {
	Provider    string  `json:"provider"`
	Host        string  `json:"host"`
	ModelName   string  `json:"model_name"`
	Temperature float64 `json:"temperature"`
}

type settingsView struct {
	LLM               llmSettingsView   `json:"llm"`
	AllowRegistration bool              `json:"allow_registration"`
	Sources           map[string]string `json:"sources"`
	RestartRequired   bool              `json:"restart_required"`
}

func (s *Server) systemRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/system/ui-config", s.uiConfig)
	mux.HandleFunc("GET /api/v1/system/settings", s.admin(func(w http.ResponseWriter, _ *http.Request, _ userView) {
		writeJSON(w, http.StatusOK, s.settingsView())
	}))
	mux.HandleFunc("PUT /api/v1/system/settings", s.admin(s.updateSettings))
}

// uiConfig is public and available even while the database is down.
func (s *Server) uiConfig(w http.ResponseWriter, _ *http.Request) {
	c := s.cfg()
	writeJSON(w, http.StatusOK, map[string]bool{
		"bypass_review_default": c.Images.BypassReviewDefault,
		"registration_open":     c.Auth.AllowRegistration,
	})
}

func (s *Server) settingsView() settingsView {
	c := s.cfg()
	return settingsView{
		LLM:               llmSettingsView{c.LLM.Provider, c.LLM.Host, c.LLM.ModelName, c.LLM.Temperature},
		AllowRegistration: c.Auth.AllowRegistration,
		Sources:           config.EditableSources(s.opts.ConfigSources),
		RestartRequired:   s.opts.Config.RestartRequired(),
	}
}

type settingsUpdate struct {
	LLM *struct {
		Provider    *string  `json:"provider"`
		Host        *string  `json:"host"`
		ModelName   *string  `json:"model_name"`
		Temperature *float64 `json:"temperature"`
	} `json:"llm"`
	AllowRegistration *bool `json:"allow_registration"`
}

func (s *Server) updateSettings(w http.ResponseWriter, r *http.Request, _ userView) {
	var body settingsUpdate
	if !decodeJSON(w, r, &body) {
		return
	}
	var errs validationErrors
	if body.LLM == nil {
		errs.add(fieldError{Type: "missing", Loc: bodyLoc("llm"), Msg: "Field required"})
	} else {
		l := body.LLM
		if l.Provider == nil {
			errs.add(fieldError{Type: "missing", Loc: []any{"body", "llm", "provider"}, Msg: "Field required"})
		} else if p := *l.Provider; p != "ollama" && p != "anthropic" && p != "openai" {
			errs.add(fieldError{Type: "enum", Loc: []any{"body", "llm", "provider"},
				Msg: "Input should be 'ollama', 'anthropic' or 'openai'", Input: p,
				Ctx: map[string]any{"expected": "'ollama', 'anthropic' or 'openai'"}})
		}
		for _, f := range []struct {
			name string
			v    *string
		}{{"host", l.Host}, {"model_name", l.ModelName}} {
			if f.v == nil {
				errs.add(fieldError{Type: "missing", Loc: []any{"body", "llm", f.name}, Msg: "Field required"})
			} else if *f.v == "" {
				errs.add(fieldError{Type: "string_too_short", Loc: []any{"body", "llm", f.name},
					Msg: "String should have at least 1 character", Input: "", Ctx: map[string]any{"min_length": 1}})
			}
		}
		if l.Temperature == nil {
			errs.add(fieldError{Type: "missing", Loc: []any{"body", "llm", "temperature"}, Msg: "Field required"})
		} else if *l.Temperature < 0 {
			errs.add(fieldError{Type: "greater_than_equal", Loc: []any{"body", "llm", "temperature"},
				Msg: "Input should be greater than or equal to 0", Input: *l.Temperature, Ctx: map[string]any{"ge": 0}})
		} else if *l.Temperature > 2 {
			errs.add(fieldError{Type: "less_than_equal", Loc: []any{"body", "llm", "temperature"},
				Msg: "Input should be less than or equal to 2", Input: *l.Temperature, Ctx: map[string]any{"le": 2}})
		}
	}
	if body.AllowRegistration == nil {
		errs.add(fieldError{Type: "missing", Loc: bodyLoc("allow_registration"), Msg: "Field required"})
	}
	if len(errs) > 0 {
		errs.write(w)
		return
	}

	sources := config.EditableSources(s.opts.ConfigSources)
	cur := s.cfg()
	type change struct {
		field   string
		changed bool
	}
	var locked []string
	for _, c := range []change{
		{"llm.provider", *body.LLM.Provider != cur.LLM.Provider},
		{"llm.host", *body.LLM.Host != cur.LLM.Host},
		{"llm.model_name", *body.LLM.ModelName != cur.LLM.ModelName},
		{"llm.temperature", *body.LLM.Temperature != cur.LLM.Temperature},
		{"auth.allow_registration", *body.AllowRegistration != cur.Auth.AllowRegistration},
	} {
		if c.changed && sources[c.field] == "environment" {
			locked = append(locked, c.field)
		}
	}
	if len(locked) > 0 {
		writeError(w, http.StatusConflict, "These settings are controlled by environment variables: "+strings.Join(locked, ", "))
		return
	}

	apply := func(c *config.Config) error {
		c.LLM.Provider = *body.LLM.Provider
		c.LLM.Host = *body.LLM.Host
		c.LLM.ModelName = *body.LLM.ModelName
		c.LLM.Temperature = *body.LLM.Temperature
		c.Auth.AllowRegistration = *body.AllowRegistration
		return nil
	}
	candidate := cur
	_ = apply(&candidate)
	if err := config.Write(s.opts.ConfigSources.YAMLPath, &candidate); err != nil {
		s.opts.Logger.Error("write config", "err", err)
		writeError(w, http.StatusServiceUnavailable, "Could not write application config")
		return
	}
	_ = s.opts.Config.Update(apply)
	if s.opts.OnSettingsChanged != nil {
		s.opts.OnSettingsChanged(s.cfg())
	}
	writeJSON(w, http.StatusOK, s.settingsView())
}
