package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func num(s string) json.Number { return json.Number(s) }

func TestSanitizeProviderOptions(t *testing.T) {
	got, err := SanitizeProviderOptions("OllamaProvider", map[string]any{
		"num_gpu": num("20"), "num_ctx": num("8192"), "top_p": num("0.9"), "seed": num("-5"), "repeat_penalty": num("1.1")})
	if err != nil || got["num_gpu"] != int64(20) || got["num_ctx"] != int64(8192) || got["top_p"] != 0.9 || got["seed"] != int64(-5) {
		t.Fatalf("%v %v", got, err)
	}
	if got, err := SanitizeProviderOptions("OllamaProvider", nil); got != nil || err != nil {
		t.Fatal("empty options stay empty")
	}
	if _, err := SanitizeProviderOptions("OpenAIProvider", map[string]any{"seed": num("1"), "top_p": num("0.5")}); err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		kind string
		in   map[string]any
		msg  string
	}{
		"openai has no num_ctx": {"OpenAIProvider", map[string]any{"num_ctx": num("1"), "num_gpu": num("1")}, "not supported by this provider: num_ctx, num_gpu"},
		"unknown key":           {"OllamaProvider", map[string]any{"temperature": num("1")}, "not supported by this provider: temperature"},
		"unknown provider":      {"Other", map[string]any{"seed": num("1")}, "not supported by this provider: seed"},
		"fraction for integer":  {"OllamaProvider", map[string]any{"num_ctx": num("2.5")}, "must be an integer"},
		"below minimum":         {"OllamaProvider", map[string]any{"num_ctx": num("0")}, "must be at least 1"},
		"above maximum":         {"OllamaProvider", map[string]any{"top_p": num("1.5")}, "must be at most 1"},
		"not a number":          {"OllamaProvider", map[string]any{"top_k": "5"}, "must be a number"},
		"zero penalty":          {"OllamaProvider", map[string]any{"repeat_penalty": num("0")}, "greater than 0"},
	} {
		if _, err := SanitizeProviderOptions(tc.kind, tc.in); err == nil || !strings.Contains(err.Error(), tc.msg) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestProviderOptionsReachTheRequest(t *testing.T) {
	var ollama, openai map[string]any
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&ollama)
		_, _ = w.Write([]byte(`{"message":{"content":"x"}}`))
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&openai)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	ctx := context.Background()
	msgs := []Message{{Role: "user", Text: "hi"}}
	opts := ChatOptions{Options: map[string]any{"num_ctx": int64(4096), "temperature": 9.0, "seed": int64(3), "top_p": 0.5}}

	if _, err := NewOllama(srv.URL, 0.3, time.Second).Chat(ctx, "m", msgs, opts); err != nil {
		t.Fatal(err)
	}
	o := ollama["options"].(map[string]any)
	// The configured temperature always wins over a stray option.
	if o["num_ctx"] != float64(4096) || o["temperature"] != 0.3 || o["seed"] != float64(3) {
		t.Fatalf("ollama options %v", o)
	}
	if _, err := NewOpenAI(srv.URL+"/v1", "k", 0.3, time.Second).Chat(ctx, "m", msgs, opts); err != nil {
		t.Fatal(err)
	}
	// Only the whitelisted standard fields are forwarded.
	if openai["seed"] != float64(3) || openai["top_p"] != 0.5 || openai["temperature"] != 0.3 || openai["num_ctx"] != nil {
		t.Fatalf("openai body %v", openai)
	}
}
