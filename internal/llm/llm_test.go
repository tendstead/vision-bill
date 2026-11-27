package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const goodReceipt = `{"confidence": 90, "merchant_name": "Shop", "date": "2025-03-04", "currency": "EUR", "category": "grocery",
 "line_items": [{"description": "Milk", "spending_category": "milk", "quantity": 1, "unit_price": "1.00", "total_price": "1.00", "tags": []}],
 "subtotal": "1.00", "total": "1.00"}`

func TestParseOllamaHost(t *testing.T) {
	cases := map[string]string{
		"localhost:11434":            "http://localhost:11434",
		"localhost":                  "http://localhost:11434",
		"host.docker.internal:11434": "http://host.docker.internal:11434",
		"http://gpu-box":             "http://gpu-box:80",
		"https://ollama.example.com": "https://ollama.example.com:443",
		"http://x:9000/prefix/":      "http://x:9000/prefix",
		"":                           "http://127.0.0.1:11434",
		"http://[::1]:11434":         "http://[::1]:11434",
	}
	for in, want := range cases {
		if got := ParseOllamaHost(in); got != want {
			t.Errorf("%q -> %q, want %q", in, got, want)
		}
	}
}

type ollamaStub struct {
	mu      sync.Mutex
	replies []string
	chats   []map[string]any
}

func (s *ollamaStub) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/tags", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"model":"vis:1","digest":"abc","details":{"parameter_size":"7B"}},{"model":"text:1"}]}`))
	})
	mux.HandleFunc("POST /api/show", func(w http.ResponseWriter, r *http.Request) {
		var b map[string]string
		_ = json.NewDecoder(r.Body).Decode(&b)
		caps := `["completion"]`
		if b["model"] == "vis:1" {
			caps = `["completion","vision"]`
		}
		_, _ = w.Write([]byte(`{"capabilities":` + caps + `}`))
	})
	mux.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		s.mu.Lock()
		s.chats = append(s.chats, b)
		reply := ""
		if len(s.replies) > 0 {
			reply, s.replies = s.replies[0], s.replies[1:]
		}
		s.mu.Unlock()
		out, _ := json.Marshal(map[string]any{"message": map[string]string{"role": "assistant", "content": reply}})
		_, _ = w.Write(out)
	})
	return mux
}

func writeImage(t *testing.T, name string) string {
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte("\x89PNGfake"), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestOllamaModelsAndAnalyse(t *testing.T) {
	stub := &ollamaStub{replies: []string{"not json at all", "```json\n" + goodReceipt + "\n```"}}
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()
	p := NewOllama(srv.URL, 0.3, 10*time.Second)
	ctx := context.Background()
	if !p.CheckConnection(ctx) {
		t.Fatal("should be reachable")
	}
	models, err := p.Models(ctx)
	if err != nil || len(models) != 1 || models[0].ID != "vis:1" || *models[0].ParameterSize != "7B" || *models[0].Digest != "abc" {
		t.Fatalf("models %+v %v", models, err)
	}
	a, err := Analyse(ctx, p, "vis:1", []string{writeImage(t, "temp_x.png")}, []string{"food"}, PromptOptions{}, ChatOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if a.Attempts != 2 || a.Receipt.MerchantName != "Shop" {
		t.Fatalf("%+v", a)
	}
	// The retry carries the failed output and a repair instruction.
	msgs := stub.chats[1]["messages"].([]any)
	if len(msgs) != 3 || msgs[1].(map[string]any)["role"] != "assistant" || msgs[1].(map[string]any)["content"] != "not json at all" ||
		!strings.Contains(msgs[2].(map[string]any)["content"].(string), "Please respond again with ONLY corrected JSON.") {
		t.Fatalf("repair messages: %v", msgs)
	}
	first := stub.chats[0]
	if first["stream"] != false || first["options"].(map[string]any)["temperature"] != 0.3 {
		t.Fatalf("request %v", first)
	}
	if imgs := first["messages"].([]any)[0].(map[string]any)["images"].([]any); len(imgs) != 1 {
		t.Fatalf("images %v", imgs)
	}
	p.SetTemperature(0.9)
	stub.replies = []string{goodReceipt}
	if _, err := Analyse(ctx, p, "vis:1", []string{writeImage(t, "a.png")}, nil, PromptOptions{}, ChatOptions{}); err != nil {
		t.Fatal(err)
	}
	if stub.chats[len(stub.chats)-1]["options"].(map[string]any)["temperature"] != 0.9 {
		t.Fatal("runtime temperature not applied")
	}
}

func TestAnalyseFailureModes(t *testing.T) {
	ctx := context.Background()
	run := func(replies ...string) error {
		stub := &ollamaStub{replies: replies}
		srv := httptest.NewServer(stub.handler())
		defer srv.Close()
		_, err := Analyse(ctx, NewOllama(srv.URL, 0, time.Second), "m", []string{writeImage(t, "a.png")}, nil, PromptOptions{}, ChatOptions{})
		return err
	}
	var unreadable *UnreadableError
	if err := run(`{"error": {"code": "unreadable", "message": "bottom is cut off"}}`); !errors.As(err, &unreadable) || unreadable.Message != "bottom is cut off" {
		t.Fatalf("unreadable: %v", err)
	}
	if err := run("Please provide an image so I can help"); err == nil || !strings.Contains(err.Error(), "cannot process images") {
		t.Fatalf("no-vision: %v", err)
	}
	if err := run("bad", "bad", "bad"); err == nil || !strings.Contains(err.Error(), "after 3 attempts") {
		t.Fatalf("exhausted: %v", err)
	}
	// Empty replies count as attempts but add no repair message.
	if err := run("", "", ""); err == nil || !strings.Contains(err.Error(), "after 3 attempts") {
		t.Fatalf("empty: %v", err)
	}
	if _, err := Analyse(ctx, NewOllama("http://127.0.0.1:1", 0, time.Second), "m", []string{"/nonexistent.png"}, nil, PromptOptions{}, ChatOptions{}); err == nil ||
		!strings.Contains(err.Error(), "Image not found at") {
		t.Fatalf("missing file: %v", err)
	}
	if _, err := Analyse(ctx, NewOllama("http://127.0.0.1:1", 0, 300*time.Millisecond), "m", []string{writeImage(t, "a.png")}, nil, PromptOptions{}, ChatOptions{}); err == nil {
		t.Fatal("expected connection error")
	}
}

func TestOpenAIProvider(t *testing.T) {
	var gotAuth string
	var chat map[string]any
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		// llama.cpp hybrid body
		_, _ = w.Write([]byte(`{"data":[{"id":"gemma","meta":{"n_params":27320697856,"modalities":["text","image"]}},{"id":"txt"}],
		 "models":[{"model":"gemma","capabilities":["completion","multimodal"],"digest":"d1","details":{}},{"model":"txt","capabilities":["completion"]}]}`))
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&chat)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":` + mustJSON(goodReceipt) + `}}]}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	p := NewOpenAI(srv.URL+"/v1", "secret", 0.5, 5*time.Second)
	ctx := context.Background()
	models, err := p.Models(ctx)
	if err != nil || len(models) != 1 || models[0].ID != "gemma" || *models[0].ParameterSize != "27.3B" || *models[0].Digest != "d1" {
		t.Fatalf("models %+v %v", models, err)
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("auth %q", gotAuth)
	}
	a, err := Analyse(ctx, p, "gemma", []string{writeImage(t, "temp_x.png")}, nil, PromptOptions{}, ChatOptions{})
	if err != nil || a.Receipt.MerchantName != "Shop" {
		t.Fatalf("%v %+v", err, a)
	}
	if chat["reasoning_effort"] != "low" || chat["temperature"] != 0.5 || chat["model"] != "gemma" {
		t.Fatalf("request %v", chat)
	}
	content := chat["messages"].([]any)[0].(map[string]any)["content"].([]any)
	img := content[1].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	if content[0].(map[string]any)["type"] != "text" || !strings.HasPrefix(img, "data:image/png;base64,") {
		t.Fatalf("content %v", content)
	}
	if !p.CheckConnection(ctx) || NewOpenAI("http://127.0.0.1:1/v1", "k", 0, time.Second).CheckConnection(ctx) {
		t.Fatal("connection checks")
	}
	if got := formatParameterSize(1_500_000); got != "1.5M" {
		t.Fatal(got)
	}
	if got := formatParameterSize(7_000_000_000); got != "7B" {
		t.Fatal(got)
	}
}

func mustJSON(s string) string { b, _ := json.Marshal(s); return string(b) }

const multiReceipt = `{"confidence": 90, "merchant_name": "Shop", "date": "2025-03-04", "subtotal": "3.00", "total": "3.00",
 "line_items": [
  {"description": "A", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 0},
  {"description": "B", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 0},
  {"description": "b ", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 1},
  {"description": "C", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 1}],
 "duplicate_candidates": [{"first_index": 1, "second_index": 2}]}`

func TestMultiImageReconcile(t *testing.T) {
	r, err := ParseResponse(multiReceipt, 2)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, it := range r.LineItems {
		names = append(names, it.Description)
		if it.SourceImage != nil {
			t.Fatal("source_image must be stripped")
		}
	}
	if strings.Join(names, ",") != "A,B,C" {
		t.Fatalf("overlap not removed: %v", names)
	}
	// A candidate that does not match exactly keeps all rows (repeat purchase).
	notSame := strings.Replace(multiReceipt, `"description": "b "`, `"description": "different"`, 1)
	if r, err := ParseResponse(notSame, 2); err != nil || len(r.LineItems) != 4 {
		t.Fatalf("non-identical rows must stay: %v %v", err, r)
	}
	for name, mod := range map[string]string{
		"photo order": strings.Replace(multiReceipt, `"source_image": 0},
  {"description": "B"`, `"source_image": 1},
  {"description": "B"`, 1),
		"source range":      strings.Replace(multiReceipt, `"source_image": 1}],`, `"source_image": 5}],`, 1),
		"out of range":      strings.Replace(multiReceipt, `"second_index": 2`, `"second_index": 9`, 1),
		"not adjacent":      strings.Replace(multiReceipt, `{"first_index": 1, "second_index": 2}`, `{"first_index": 0, "second_index": 1}`, 1),
		"not boundary":      strings.Replace(multiReceipt, `{"first_index": 1, "second_index": 2}`, `{"first_index": 0, "second_index": 3}`, 1),
		"missing candidate": strings.Replace(multiReceipt, `"duplicate_candidates": [{"first_index": 1, "second_index": 2}]`, `"x": 1`, 1),
	} {
		if _, err := ParseResponse(mod, 2); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestIsTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { time.Sleep(200 * time.Millisecond) }))
	defer srv.Close()
	_, err := NewOllama(srv.URL, 0, time.Second).Chat(ctx, "m", nil, ChatOptions{})
	if !IsTimeout(err) {
		t.Fatalf("not a timeout: %v", err)
	}
}
