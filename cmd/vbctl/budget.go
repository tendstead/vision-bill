package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tendstead/vision-bill/internal/llm"
	"github.com/tendstead/vision-bill/internal/store"
)

// A realistic pool of grocery tags (multi-word on purpose) used to measure
// the average token cost per tag without needing the live database.
var tagPool = []string{
	"alcohol", "beverage", "cheese", "coffee", "deposit", "electronics", "food", "fresh", "gift", "household",
	"household goods", "hygiene", "meat", "meat product", "office", "other", "pet", "poultry", "service",
	"subscription", "supplement", "travel", "bakery", "dairy", "frozen", "snack", "spice", "canned goods",
	"detergent", "paper products",
}

func tagsPrompt(count int) string {
	tags := make([]string, count)
	for i := range tags {
		if count <= len(tagPool) {
			tags[i] = tagPool[i%len(tagPool)]
		} else {
			tags[i] = fmt.Sprintf("tag_%d", i)
		}
	}
	return llm.BuildPrompt(tags, 1, llm.PromptOptions{})
}

func percentile(sorted []float64, pct float64) float64 {
	switch len(sorted) {
	case 0:
		return 0
	case 1:
		return sorted[0]
	}
	rank := float64(len(sorted)-1) * (pct / 100)
	lo, hi := int(math.Floor(rank)), int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	return sorted[lo] + (sorted[hi]-sorted[lo])*(rank-float64(lo))
}

func pct(part, whole int) string {
	if whole == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%%", 100*float64(part)/float64(whole))
}

type budget struct {
	host       string
	model      string
	numCtxCap  int
	client     *http.Client
	ctx        int
	baseTokens int
	perTag     float64
	imageToks  int
}

func (b *budget) call(method, path string, body any, out any) error {
	var rd io.Reader
	if body != nil {
		raw, _ := json.Marshal(body)
		rd = bytes.NewReader(raw)
	}
	req, err := http.NewRequest(method, b.host+path, rd)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("%s returned HTTP %d: %s", path, resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return json.Unmarshal(data, out)
}

func (b *budget) contextLength() error {
	var tags struct {
		Models []struct {
			Model   string `json:"model"`
			Name    string `json:"name"`
			Details struct {
				ContextLength int `json:"context_length"`
			} `json:"details"`
		} `json:"models"`
	}
	if err := b.call("GET", "/api/tags", nil, &tags); err != nil {
		return err
	}
	for _, m := range tags.Models {
		if m.Model == b.model || m.Name == b.model {
			if m.Details.ContextLength > 0 {
				b.ctx = m.Details.ContextLength
				return nil
			}
		}
	}
	var show struct {
		ModelInfo map[string]any `json:"model_info"`
	}
	if err := b.call("POST", "/api/show", map[string]string{"name": b.model}, &show); err != nil {
		return err
	}
	for k, v := range show.ModelInfo {
		if strings.HasSuffix(k, "context_length") {
			if f, ok := v.(float64); ok {
				b.ctx = int(f)
				return nil
			}
		}
	}
	return nil
}

// count returns the exact prompt token count (Ollama's prompt_eval_count).
func (b *budget) count(prompt, imageB64 string) (int, error) {
	guess := max(len(prompt)/3, 4096)
	if imageB64 != "" {
		guess = max(len(prompt)/3+6000, 4096)
	}
	numCtx := guess
	if b.ctx > 0 {
		numCtx = min(b.ctx, max(guess, b.numCtxCap))
	}
	payload := map[string]any{"model": b.model, "prompt": prompt, "stream": false, "num_predict": 1, "options": map[string]any{"num_ctx": numCtx}}
	if imageB64 != "" {
		payload["images"] = []string{imageB64}
	}
	var out struct {
		PromptEvalCount int `json:"prompt_eval_count"`
	}
	if err := b.call("POST", "/api/generate", payload, &out); err != nil {
		return 0, err
	}
	return out.PromptEvalCount, nil
}

func (b *budget) measure(tagProbe int, imageB64 string) error {
	if err := b.contextLength(); err != nil {
		return err
	}
	base := llm.BuildPrompt(nil, 1, llm.PromptOptions{})
	var err error
	if b.baseTokens, err = b.count(base, ""); err != nil {
		return err
	}
	big, err := b.count(tagsPrompt(tagProbe), "")
	if err != nil {
		return err
	}
	if tagProbe > 0 {
		b.perTag = float64(big-b.baseTokens) / float64(tagProbe)
	}
	if imageB64 != "" {
		withImage, err := b.count(base, imageB64)
		if err != nil {
			return err
		}
		b.imageToks = withImage - b.baseTokens
	}
	return nil
}

func firstVisionModel(host string, client *http.Client) (string, error) {
	b := &budget{host: host, client: client}
	var tags struct {
		Models []struct {
			Model   string `json:"model"`
			Details struct {
				Capabilities []string `json:"capabilities"`
			} `json:"details"`
		} `json:"models"`
	}
	if err := b.call("GET", "/api/tags", nil, &tags); err != nil {
		return "", fmt.Errorf("Could not reach Ollama at %s: %w", host, err)
	}
	for _, m := range tags.Models {
		for _, c := range m.Details.Capabilities {
			if m.Model != "" && c == "vision" {
				return m.Model, nil
			}
		}
	}
	for _, m := range tags.Models {
		if m.Model == "" {
			continue
		}
		var show struct {
			Capabilities []string `json:"capabilities"`
		}
		if err := b.call("POST", "/api/show", map[string]string{"name": m.Model}, &show); err != nil {
			continue
		}
		for _, c := range show.Capabilities {
			if c == "vision" {
				return m.Model, nil
			}
		}
	}
	return "", errors.New("No vision-capable model found. Pass --model explicitly.")
}

func runContextBudget(args []string) error {
	fs := flag.NewFlagSet("context-budget", flag.ContinueOnError)
	model := fs.String("model", "", "Ollama model name (default: first vision-capable model)")
	host := fs.String("host", envOr("OLLAMA_HOST", "http://localhost:11434"), "Ollama host")
	image := fs.String("image", "", "path to a sample image to measure image-token cost")
	appURL := fs.String("app-url", envOr("VB_APP_URL", "http://localhost:8080"), "app URL (reads the live tag count; needs a session via -tags-file otherwise)")
	tagsFile := fs.String("tags-file", "", "JSON file of the current tag list (overrides -app-url)")
	outputBudget := fs.Int("output-budget", 2000, "tokens reserved for the model's JSON receipt output")
	tagProbe := fs.Int("tag-probe", 250, "tag count used to measure the average tokens per tag")
	numCtxCap := fs.Int("num-ctx-cap", 16384, "max num_ctx to request during measurement")
	timeout := fs.Int("timeout", 180, "per-call HTTP timeout in seconds")
	dbSample := fs.Int("db-sample", 0, "sample N random images from the database (e.g. 16)")
	if err := fs.Parse(args); err != nil {
		return usageError("")
	}
	h := strings.TrimRight(*host, "/")
	client := &http.Client{Timeout: time.Duration(*timeout) * time.Second}
	name := *model
	if name == "" {
		var err error
		if name, err = firstVisionModel(h, &http.Client{Timeout: 30 * time.Second}); err != nil {
			return err
		}
	}
	var currentTags []string
	if *tagsFile != "" {
		raw, err := os.ReadFile(*tagsFile)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(raw, &currentTags); err != nil {
			return err
		}
	} else if resp, err := (&http.Client{Timeout: 5 * time.Second}).Get(strings.TrimRight(*appURL, "/") + "/api/v1/tags"); err == nil {
		defer resp.Body.Close()
		if resp.StatusCode == 200 {
			_ = json.NewDecoder(resp.Body).Decode(&currentTags)
		}
	}
	var imageB64 string
	if *image != "" {
		raw, err := os.ReadFile(*image)
		if err != nil {
			return err
		}
		imageB64 = base64.StdEncoding.EncodeToString(raw)
	}

	b := &budget{host: h, model: name, numCtxCap: *numCtxCap, client: client}
	fmt.Printf("Model: %s\n\n", name)
	if err := b.measure(*tagProbe, imageB64); err != nil {
		return fmt.Errorf("Measurement failed (is the model loaded / reachable?): %w", err)
	}
	fmt.Printf("Context window:          %d tokens\n", b.ctx)
	fmt.Printf("Base prompt (instr+schema): %d tokens  (%s of ctx)\n", b.baseTokens, pct(b.baseTokens, b.ctx))
	fmt.Printf("Tokens per tag (avg):    %.2f\n", b.perTag)
	if imageB64 != "" {
		fmt.Printf("Image tokens (%s): %d tokens\n", filepath.Base(*image), b.imageToks)
	}
	if len(currentTags) > 0 {
		fmt.Printf("Current tags (live):   %d\n", len(currentTags))
	}
	fmt.Printf("Reserved output budget:  %d tokens\n\n", *outputBudget)

	if *dbSample > 0 {
		reportDBSample(b, *dbSample, len(currentTags), *outputBudget)
	}

	promptTokens := func(tags int) int {
		p := b.baseTokens + int(math.Round(float64(tags)*b.perTag))
		if imageB64 != "" {
			p += b.imageToks
		}
		return p
	}
	title := "Tag vocabulary growth (base + tags"
	if imageB64 != "" {
		title += " + image"
	}
	fmt.Println(title + "):")
	fmt.Printf("  %6s  %8s  %6s  %16s\n", "tags", "prompt", "%ctx", "payload headroom")
	counts := map[int]bool{0: true, 25: true, 50: true, 100: true, 250: true, 500: true, 1000: true}
	if len(currentTags) > 0 {
		counts[len(currentTags)] = true
	}
	var ns []int
	for n := range counts {
		ns = append(ns, n)
	}
	sort.Ints(ns)
	for _, n := range ns {
		p := promptTokens(n)
		fmt.Printf("  %6d  %8d  %6s  %16d\n", n, p, pct(p, b.ctx), max(0, b.ctx-p-*outputBudget))
	}
	if b.perTag > 0 {
		for _, frac := range []float64{0.5, 0.8} {
			limit := int((frac*float64(b.ctx) - float64(b.baseTokens)) / b.perTag)
			suffix := ""
			if imageB64 != "" {
				suffix = " (before image)"
			}
			fmt.Printf("\nPrompt hits %.0f%% of context at ~%d tags%s.\n", frac*100, limit, suffix)
		}
	}
	nCur := len(currentTags)
	pCur := promptTokens(nCur)
	headroom := max(0, b.ctx-pCur-*outputBudget)
	fmt.Printf("\nAt %d tags: prompt = %d tokens, leaving %d tokens (%s) for the image + JSON output.\n", nCur, pCur, headroom, pct(headroom, b.ctx))
	if b.imageToks > 0 {
		fits := headroom >= b.imageToks
		verdict := "fits"
		if !fits {
			verdict = "DOES NOT FIT"
		}
		fmt.Printf("  -> sample image costs %d tokens; %s with %d tokens of headroom.\n", b.imageToks, verdict, headroom)
		if !fits {
			fmt.Println("  WARNING: reduce the image resolution or tag list to make room for the output.")
		}
	}
	return nil
}

func reportDBSample(b *budget, n, tagCount, outputBudget int) {
	fmt.Printf("Image token cost from the database (random sample of up to %d):\n", n)
	cfg, err := loadToolConfig()
	if err != nil {
		fmt.Printf("  Could not load the configuration: %v\n\n", err)
		return
	}
	ctx := context.Background()
	st, err := store.Open(ctx, cfg.PG.DSN())
	if err != nil {
		fmt.Printf("  Could not query the images table: %v\n\n", err)
		return
	}
	defer st.Close()
	rows, err := st.Pool.Query(ctx, `SELECT image_path FROM images WHERE image_path IS NOT NULL ORDER BY random() LIMIT $1`, n)
	if err != nil {
		fmt.Printf("  Could not query the images table: %v\n\n", err)
		return
	}
	var paths []string
	for rows.Next() {
		var p string
		if rows.Scan(&p) == nil {
			paths = append(paths, p)
		}
	}
	rows.Close()
	if len(paths) == 0 {
		fmt.Println("  No images with a path found in the database yet.")
		fmt.Println()
		return
	}
	dirs := []string{cfg.Images.SaveDir, cfg.Images.TmpDir, "uploads", "uploads_tmp"}
	base := llm.BuildPrompt(nil, 1, llm.PromptOptions{})
	var vals []float64
	unresolvable := 0
	for _, p := range paths {
		local := resolveStored(p, dirs)
		if local == "" {
			unresolvable++
			continue
		}
		raw, err := os.ReadFile(local)
		if err != nil {
			unresolvable++
			continue
		}
		total, err := b.count(base, base64.StdEncoding.EncodeToString(raw))
		if err != nil {
			fmt.Printf("  Measurement failed: %v\n\n", err)
			return
		}
		vals = append(vals, float64(total-b.baseTokens))
	}
	if len(vals) == 0 {
		fmt.Printf("  %d rows found but none resolved to a local file (stored paths may be container-absolute).\n\n", unresolvable)
		return
	}
	sort.Float64s(vals)
	tail := ""
	if unresolvable > 0 {
		tail = fmt.Sprintf(", %d unresolvable", unresolvable)
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	fmt.Printf("  sampled %d rows, measured %d images%s\n", len(paths), len(vals), tail)
	fmt.Printf("  min=%.0f  median=%.0f  mean=%.0f  p90=%.0f  max=%.0f  tokens\n", vals[0], percentile(vals, 50), sum/float64(len(vals)), percentile(vals, 90), vals[len(vals)-1])
	baseTags := b.baseTokens + int(math.Round(float64(tagCount)*b.perTag))
	for _, c := range []struct {
		label string
		img   float64
	}{{"median", percentile(vals, 50)}, {"p90", percentile(vals, 90)}} {
		total := baseTags + int(c.img) + outputBudget
		fmt.Printf("  base + %d tags + %s image + output = %d tokens (%s of ctx)\n", tagCount, c.label, total, pct(total, b.ctx))
	}
	fmt.Println()
}
