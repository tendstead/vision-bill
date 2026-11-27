package main

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBenchmarkPayloadOmitsUnsetOptions(t *testing.T) {
	cat, lim := "grocery", 5
	p := benchmarkPayload([]string{"m"}, nil, &cat, nil, &lim, 300, "all", nil, nil, false)
	raw, _ := json.Marshal(p)
	for _, want := range []string{`"model_ids":["m"]`, `"category":"grocery"`, `"limit":5`, `"request_timeout_seconds":300`, `"council_policy":"all"`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("%s missing in %s", want, raw)
		}
	}
	for _, absent := range []string{"receipt_ids", "max_source_confidence", "apply_council_flags", "council_absolute_threshold"} {
		if strings.Contains(string(raw), absent) {
			t.Errorf("%s must be omitted: %s", absent, raw)
		}
	}
}

func TestBenchmarkCommandLogsInAndPrintsRunID(t *testing.T) {
	var loggedIn bool
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/auth/login", func(w http.ResponseWriter, r *http.Request) {
		loggedIn = true
		http.SetCookie(w, &http.Cookie{Name: "vb_session", Value: "tok", Path: "/"})
		_, _ = w.Write([]byte(`{}`))
	})
	mux.HandleFunc("POST /api/v1/benchmarks", func(w http.ResponseWriter, r *http.Request) {
		if c, err := r.Cookie("vb_session"); err != nil || c.Value != "tok" {
			w.WriteHeader(401)
			return
		}
		w.WriteHeader(202)
		_, _ = w.Write([]byte(`{"id":"run-123"}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	stdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	err := runBenchmark([]string{"-api-url", srv.URL, "-username", "a", "-password", "b", "-model", "x", "-limit", "3"})
	w.Close()
	os.Stdout = stdout
	out := make([]byte, 64)
	n, _ := r.Read(out)
	if err != nil || !loggedIn || strings.TrimSpace(string(out[:n])) != "run-123" {
		t.Fatalf("err=%v loggedIn=%v out=%q", err, loggedIn, out[:n])
	}
	if err := runBenchmark([]string{"-username", "only"}); err == nil {
		t.Fatal("username without password must be a usage error")
	} else if _, ok := err.(usageError); !ok {
		t.Fatalf("%T", err)
	}
}

func TestPercentileAndPct(t *testing.T) {
	v := []float64{1, 2, 3, 4, 10}
	if percentile(v, 50) != 3 || math.Abs(percentile(v, 90)-7.6) > 1e-9 || percentile(nil, 50) != 0 || percentile([]float64{5}, 90) != 5 {
		t.Fatalf("%v %v", percentile(v, 50), percentile(v, 90))
	}
	if pct(1, 4) != "25.0%" || pct(1, 0) != "n/a" {
		t.Fatal("pct")
	}
}

func TestResolveStored(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "x.png")
	_ = os.WriteFile(file, []byte("x"), 0o644)
	if resolveStored(file, nil) != file {
		t.Fatal("direct path")
	}
	if got := resolveStored("/app/uploads/x.png", []string{dir}); got != file {
		t.Fatalf("container path mapped to %q", got)
	}
	if resolveStored("/nope/y.png", []string{dir}) != "" || resolveStored("  ", nil) != "" {
		t.Fatal("missing")
	}
}

func TestContextBudgetAgainstStubOllama(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/tags", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"model":"vis:1","details":{"context_length":8192,"capabilities":["vision"]}}]}`))
	})
	mux.HandleFunc("POST /api/generate", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Prompt string   `json:"prompt"`
			Images []string `json:"images"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		n := len(body.Prompt) / 4
		if len(body.Images) > 0 {
			n += 300
		}
		_ = json.NewEncoder(w).Encode(map[string]int{"prompt_eval_count": n})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	img := filepath.Join(t.TempDir(), "i.png")
	_ = os.WriteFile(img, []byte("\x89PNGdata"), 0o644)
	tags := filepath.Join(t.TempDir(), "tags.json")
	_ = os.WriteFile(tags, []byte(`["a","b"]`), 0o644)

	stdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	err := runContextBudget([]string{"-host", srv.URL, "-image", img, "-tags-file", tags, "-tag-probe", "50"})
	w.Close()
	os.Stdout = stdout
	buf := make([]byte, 1<<16)
	n, _ := r.Read(buf)
	out := string(buf[:n])
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Model: vis:1", "Context window:          8192 tokens", "Tokens per tag (avg):", "Image tokens (i.png): 300 tokens", "Current tags (live):   2", "At 2 tags:"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
}
