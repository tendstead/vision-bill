// Command stub is a deterministic OpenAI-compatible LLM stub for the e2e stack.
//
// It serves ground-truth receipt JSON keyed by the sha256 of the uploaded image
// bytes, so the app's real OpenAI provider, retry loop and persistence all run.
// Fixture pairs (<name>.png|jpg|jpeg + <name>.json) are read from -data (opt-in
// local pairs) and -fixtures (committed pairs; they win on a hash collision).
//
// Control API (e2e only):
//
//	POST /__mode  {"mode": "ok"|"down"|"repair_first"|"broken"|"unreadable"|"slow"}
//	GET  /__requests
//	GET  /__fixtures/<name>[.json]
package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

var modes = map[string]bool{"ok": true, "down": true, "repair_first": true, "broken": true, "unreadable": true, "slow": true}

const brokenReply = `{"merchant_name": "broken", oops`

var genericReceipt = map[string]any{
	"confidence": 99, "merchant_name": "E2E Generic Store", "date": "2026-01-01",
	"line_items": []any{map[string]any{"description": "Generic Item", "quantity": 1.0, "unit_price": "5.00", "total_price": "5.00"}},
	"subtotal":   "5.00", "tax_total": "0", "total": "5.00", "payment_method": "cash",
}

type stub struct {
	fixturesDir string

	mu       sync.Mutex
	mode     string
	broken   map[string]bool // digests that already got their one broken answer (repair_first)
	requests []map[string]any
	truth    map[string]string // sha256(image) -> ground-truth JSON text
}

func newStub(dataDir, fixturesDir string) *stub {
	s := &stub{fixturesDir: fixturesDir, mode: "ok", broken: map[string]bool{}, requests: []map[string]any{}, truth: map[string]string{}}
	// Committed fixtures win over opt-in data on a hash collision.
	for _, dir := range []string{dataDir, fixturesDir} {
		for k, v := range loadFixtures(dir) {
			s.truth[k] = v
		}
	}
	return s
}

func loadFixtures(dir string) map[string]string {
	out := map[string]string{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return out
	}
	for _, e := range entries {
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
			continue
		}
		base := strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
		raw, err := os.ReadFile(filepath.Join(dir, base+".json"))
		if err != nil {
			continue
		}
		var probe map[string]any
		if json.Unmarshal(raw, &probe) != nil {
			log.Printf("stub: skipping %s.json (not a valid JSON object)", base)
			continue
		}
		img, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		sum := sha256.Sum256(img)
		out[hex.EncodeToString(sum[:])] = string(raw)
	}
	log.Printf("stub: loaded %d fixture pair(s) from %s", len(out), dir)
	return out
}

func (s *stub) logRequest(digest *string, served string) {
	var d any
	if digest != nil {
		d = *digest
	}
	s.requests = append(s.requests, map[string]any{"image_sha256": d, "served": served})
}

// photos returns the sha256 of every image in the chat request, in order.
func photos(body map[string]any) []string {
	var out []string
	msgs, _ := body["messages"].([]any)
	for _, m := range msgs {
		content, _ := m.(map[string]any)["content"].([]any)
		for _, part := range content {
			p, _ := part.(map[string]any)
			if p["type"] != "image_url" {
				continue
			}
			iu, _ := p["image_url"].(map[string]any)
			url, _ := iu["url"].(string)
			if !strings.HasPrefix(url, "data:") {
				continue
			}
			_, b64, _ := strings.Cut(url, ",")
			raw, err := base64.StdEncoding.DecodeString(b64)
			if err != nil {
				continue
			}
			sum := sha256.Sum256(raw)
			out = append(out, hex.EncodeToString(sum[:]))
		}
	}
	return out
}

func (s *stub) singleContent(digests []string) string {
	var digest *string
	if len(digests) > 0 {
		digest = &digests[0]
	}
	switch {
	case s.mode == "broken":
		s.logRequest(nil, "broken")
		return brokenReply
	case s.mode == "repair_first" && digest != nil && !s.broken[*digest]:
		s.broken[*digest] = true
		s.logRequest(digest, "broken")
		return brokenReply
	}
	if digest != nil {
		if truth, ok := s.truth[*digest]; ok {
			s.logRequest(digest, "fixture")
			return truth
		}
	}
	s.logRequest(digest, "generic")
	b, _ := json.Marshal(genericReceipt)
	return string(b)
}

func (s *stub) content(body map[string]any) string {
	s.mu.Lock()
	mode := s.mode
	s.mu.Unlock()
	switch mode {
	case "unreadable":
		return `{"error": {"code": "unreadable", "message": "Photo is too blurry"}}`
	case "slow":
		time.Sleep(3 * time.Second)
	}
	digests := photos(body)
	s.mu.Lock()
	content := s.singleContent(digests)
	s.mu.Unlock()
	if len(digests) <= 1 {
		return content
	}
	var payload map[string]any
	if json.Unmarshal([]byte(content), &payload) != nil {
		return content
	}
	// Multi-photo cases upload the same fixture several times. Answer with all
	// copies plus explicit overlap candidates to exercise the verification.
	items, _ := payload["line_items"].([]any)
	var all []any
	for source := range digests {
		for _, it := range items {
			row := map[string]any{}
			for k, v := range it.(map[string]any) {
				row[k] = v
			}
			row["source_image"] = source
			all = append(all, row)
		}
	}
	payload["line_items"] = all
	var cands []any
	for source := 0; source < len(digests)-1; source++ {
		for row := range items {
			cands = append(cands, map[string]any{"first_index": source*len(items) + row, "second_index": (source+1)*len(items) + row})
		}
	}
	payload["duplicate_candidates"] = cands
	b, _ := json.Marshal(payload)
	return string(b)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *stub) down() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.mode == "down"
}

var fixtureName = regexp.MustCompile(`^[a-z0-9]+(\.json)?$`)

func (s *stub) serveFixture(w http.ResponseWriter, name string) {
	if !fixtureName.MatchString(name) {
		writeJSON(w, 400, map[string]string{"error": "bad fixture name"})
		return
	}
	isJSON := strings.HasSuffix(name, ".json")
	base := strings.TrimSuffix(name, ".json")
	path, ctype := filepath.Join(s.fixturesDir, base+".png"), "image/png"
	if isJSON {
		path, ctype = filepath.Join(s.fixturesDir, base+".json"), "application/json"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		writeJSON(w, 404, map[string]string{"error": fmt.Sprintf("no fixture %q", base)})
		return
	}
	w.Header().Set("Content-Type", ctype)
	_, _ = w.Write(data)
}

func (s *stub) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("POST /__mode", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Mode string `json:"mode"`
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		if body.Mode == "" {
			body.Mode = "ok"
		}
		if !modes[body.Mode] {
			writeJSON(w, 400, map[string]string{"error": fmt.Sprintf("unknown mode %q", body.Mode)})
			return
		}
		s.mu.Lock()
		s.mode, s.broken = body.Mode, map[string]bool{} // reset per-mode broken-once tracking
		s.mu.Unlock()
		writeJSON(w, 200, map[string]string{"mode": body.Mode})
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		if s.down() {
			writeJSON(w, 503, map[string]string{"error": "stub is in down mode"})
			return
		}
		writeJSON(w, 200, map[string]any{"object": "list", "data": []any{map[string]any{
			"id": "e2e-vision", "object": "model", "owned_by": "e2e",
			"architecture": map[string]any{"input_modalities": []string{"image", "text"}, "output_modalities": []string{"text"}},
		}}})
	})
	mux.HandleFunc("GET /__requests", func(w http.ResponseWriter, _ *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()
		writeJSON(w, 200, map[string]any{"requests": s.requests})
	})
	mux.HandleFunc("GET /__fixtures/{name}", func(w http.ResponseWriter, r *http.Request) {
		if s.down() {
			writeJSON(w, 503, map[string]string{"error": "stub is in down mode"})
			return
		}
		s.serveFixture(w, r.PathValue("name"))
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if s.down() {
			writeJSON(w, 503, map[string]string{"error": "stub is in down mode"})
			return
		}
		var body map[string]any
		raw, _ := io.ReadAll(r.Body)
		if json.Unmarshal(raw, &body) != nil {
			writeJSON(w, 400, map[string]string{"error": "bad request"})
			return
		}
		writeJSON(w, 200, map[string]any{
			"id": "e2e-stub", "object": "chat.completion", "created": time.Now().Unix(), "model": "e2e-vision",
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": s.content(body)}, "finish_reason": "stop"}},
			"usage":   map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
		})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, 404, map[string]string{"error": "not found"})
	})
	return mux
}

func main() {
	addr := flag.String("addr", ":9123", "listen address")
	data := flag.String("data", "/data", "opt-in fixture pairs")
	fixtures := flag.String("fixtures", "/fixtures", "committed fixture pairs")
	health := flag.Bool("healthcheck", false, "probe the running stub and exit")
	flag.Parse()
	if *health {
		_, port, _ := strings.Cut(*addr, ":")
		resp, err := http.Get("http://127.0.0.1:" + port + "/healthz")
		if err != nil || resp.StatusCode != 200 {
			os.Exit(1)
		}
		return
	}
	s := newStub(*data, *fixtures)
	log.Fatal((&http.Server{Addr: *addr, Handler: s.handler(), ReadHeaderTimeout: 10 * time.Second}).ListenAndServe())
}
