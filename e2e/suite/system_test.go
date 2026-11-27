//go:build e2e

package suite

import (
	"math"
	"strings"
	"testing"
)

func TestUIConfigIsPublic(t *testing.T) {
	body := newEnv(t).Get("/system/ui-config").Is(200).M()
	if body["registration_open"] != true || body["bypass_review_default"] != false {
		t.Fatalf("%v", body)
	}
	if body["heic_supported"] != true {
		t.Fatalf("the image always ships its HEIC decoder: %v", body)
	}
}

func TestHealthReadinessAndDocs(t *testing.T) {
	c := newEnv(t)
	c.Raw("GET", "/healthz", nil, nil).Is(200)
	if r := c.Raw("GET", "/readyz", nil, nil).Is(200); str(r.M(), "status") != "ready" {
		t.Fatal(r)
	}
	if spec := c.Raw("GET", "/openapi.json", nil, nil).Is(200); !strings.Contains(string(spec.Body), `"/api/v1/receipts"`) {
		t.Fatal("contract does not describe receipts")
	}
	c.Raw("GET", "/docs", nil, nil).Is(200)
}

func TestLLMModelsListsTheStubModel(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "sys")
	found := false
	for _, m := range c.Get("/llm/models").Is(200).L() {
		if str(m, "id") == "e2e-vision" {
			found = true
		}
	}
	if !found {
		t.Fatal("stub model not listed")
	}
	newClient(t).Get("/llm/models").Is(401)
}

func TestSettingsSourcesAndEnvironmentConflict(t *testing.T) {
	newEnv(t)
	c := admin(t)
	view := c.Get("/system/settings").Is(200).M()
	if str(view, "llm", "provider") != "openai" || str(view, "sources", "llm.provider") != "environment" || str(view, "sources", "llm.temperature") == "environment" {
		t.Fatalf("%v", view)
	}
	llm := func(provider string, temp float64) map[string]any {
		return map[string]any{"llm": map[string]any{"provider": provider, "host": str(view, "llm", "host"),
			"model_name": str(view, "llm", "model_name"), "temperature": temp}, "allow_registration": view["allow_registration"]}
	}
	temp := mp(view, "llm")["temperature"].(float64)
	// An environment-controlled key cannot be changed...
	r := c.Put("/system/settings", llm("ollama", temp)).Is(409)
	if !strings.Contains(str(r.M(), "detail"), "llm.provider") {
		t.Fatal(r)
	}
	// ...a file-controlled one applies immediately.
	next := math.Round((temp+0.1)*100) / 100
	got := c.Put("/system/settings", llm(str(view, "llm", "provider"), next)).Is(200).M()
	if mp(got, "llm")["temperature"].(float64) != next {
		t.Fatalf("%v", got)
	}
}

func TestIndexAndClientRoutesServeTheSPA(t *testing.T) {
	c := newEnv(t)
	for _, p := range []string{"/", "/statistics", "/receipts/00000000-0000-0000-0000-000000000000", "/receipts/%2E%2E%2Fetc%2Fpasswd"} {
		r := c.Raw("GET", p, nil, nil).Is(200)
		if !strings.Contains(r.Header.Get("Content-Type"), "text/html") || !strings.Contains(string(r.Body), "</html>") {
			t.Errorf("%s: %v", p, r.Header.Get("Content-Type"))
		}
	}
	// Unknown API paths stay JSON 404s, missing assets are real 404s.
	if r := c.Raw("GET", "/api/v1/nope", nil, nil).Is(404); !strings.Contains(r.Header.Get("Content-Type"), "json") {
		t.Fatal("unknown API path must not fall back to the SPA")
	}
	c.Raw("GET", "/_app/immutable/missing.js", nil, nil).Is(404)
}
