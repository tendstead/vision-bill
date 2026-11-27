package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"
)

const defaultImageMime = "image/jpeg"

// guessMime derives an image MIME type from the file extension.
func guessMime(name string) string {
	switch strings.ToLower(path.Ext(name)) {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg", ".jpe":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".heic":
		return "image/heic"
	case ".heif":
		return "image/heif"
	case ".bmp":
		return "image/bmp"
	case ".tif", ".tiff":
		return "image/tiff"
	case ".svg":
		return "image/svg+xml"
	}
	return defaultImageMime
}

// OpenAI talks to an operator-selected OpenAI-compatible endpoint.
type OpenAI struct {
	base    string
	host    string
	key     string
	timeout time.Duration
	client  *http.Client

	mu   sync.RWMutex
	temp float64
}

// NewOpenAI creates the provider; host is the API base URL (e.g. http://x/v1).
func NewOpenAI(host, apiKey string, temperature float64, timeout time.Duration) *OpenAI {
	return &OpenAI{base: strings.TrimRight(host, "/"), host: host, key: apiKey, timeout: timeout, client: httpClient(), temp: temperature}
}

func (o *OpenAI) Host() string           { return o.host }
func (o *OpenAI) SupportsThinking() bool { return false }
func (o *OpenAI) Kind() string           { return "OpenAIProvider" }

// WithTemperature returns an independent copy with a different temperature.
func (o *OpenAI) WithTemperature(t float64) Provider {
	return &OpenAI{base: o.base, host: o.host, key: o.key, timeout: o.timeout, client: o.client, temp: t}
}

// EndpointFingerprint hashes the base URL.
func (o *OpenAI) EndpointFingerprint() string { return sha256Hex(o.base) }
func (o *OpenAI) Timeout() time.Duration      { return o.timeout }
func (o *OpenAI) SetTemperature(t float64)    { o.mu.Lock(); o.temp = t; o.mu.Unlock() }
func (o *OpenAI) temperature() float64        { o.mu.RLock(); defer o.mu.RUnlock(); return o.temp }
func (o *OpenAI) Temperature() float64        { return o.temperature() }

func (o *OpenAI) do(ctx context.Context, method, p string, body any) ([]byte, error) {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, o.base+p, rd)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+o.key)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, &StatusError{resp.StatusCode, truncateRunes(string(data), 500)}
	}
	return data, nil
}

// CheckConnection lists models; any failure means "unreachable".
func (o *OpenAI) CheckConnection(ctx context.Context) bool {
	_, err := o.do(ctx, http.MethodGet, "/models", nil)
	return err == nil
}

func formatParameterSize(n int64) string {
	for _, t := range []struct {
		threshold int64
		suffix    string
	}{{1_000_000_000, "B"}, {1_000_000, "M"}, {1_000, "K"}} {
		if n >= t.threshold {
			s := fmt.Sprintf("%.1f%s", float64(n)/float64(t.threshold), t.suffix)
			return strings.Replace(s, ".0", "", 1)
		}
	}
	return fmt.Sprint(n)
}

func isVisionCapable(entry map[string]any, capabilities []string) bool {
	for _, c := range capabilities {
		if c == "multimodal" || c == "vision" {
			return true
		}
	}
	if arch, ok := entry["architecture"].(map[string]any); ok {
		if mods, ok := arch["input_modalities"].([]any); ok && containsAny(mods, "image") {
			return true
		}
	}
	if meta, ok := entry["meta"].(map[string]any); ok {
		for _, v := range meta {
			if list, ok := v.([]any); ok && containsAny(list, "image") {
				return true
			}
		}
	}
	return false
}

func containsAny(list []any, want string) bool {
	for _, v := range list {
		if s, ok := v.(string); ok && s == want {
			return true
		}
	}
	return false
}

func strField(m map[string]any, key string) string {
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

func stringList(v any) []string {
	list, ok := v.([]any)
	if !ok {
		return []string{}
	}
	out := []string{}
	for _, it := range list {
		if s, ok := it.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// Models lists vision-capable models; llama.cpp's hybrid body (an Ollama-style
// "models" array beside OpenAI's "data") is understood.
func (o *OpenAI) Models(ctx context.Context) ([]ModelInfo, error) {
	data, err := o.do(ctx, http.MethodGet, "/models", nil)
	if err != nil {
		return nil, err
	}
	var body struct {
		Data   []any `json:"data"`
		Models []any `json:"models"`
	}
	if err := json.Unmarshal(data, &body); err != nil {
		return nil, err
	}
	dataByID := map[string]map[string]any{}
	for _, d := range body.Data {
		if m, ok := d.(map[string]any); ok {
			dataByID[strField(m, "id")] = m
		}
	}
	out := []ModelInfo{}
	for _, e := range body.Models {
		entry, ok := e.(map[string]any)
		if !ok {
			continue
		}
		id := strField(entry, "model")
		if id == "" {
			id = strField(entry, "name")
		}
		if id == "" {
			continue
		}
		caps := stringList(entry["capabilities"])
		if !isVisionCapable(dataByID[id], caps) {
			continue
		}
		size := ""
		if d, ok := entry["details"].(map[string]any); ok {
			size = strField(d, "parameter_size")
		}
		if meta, ok := dataByID[id]["meta"].(map[string]any); ok && size == "" {
			if n, ok := meta["n_params"].(float64); ok && n > 0 && n == float64(int64(n)) {
				size = formatParameterSize(int64(n))
			}
		}
		mi := ModelInfo{ID: id, Capabilities: caps}
		if d := strField(entry, "digest"); d != "" {
			mi.Digest = &d
		}
		if size != "" {
			mi.ParameterSize = &size
		}
		out = append(out, mi)
	}
	if len(body.Models) == 0 {
		for _, d := range body.Data {
			entry, ok := d.(map[string]any)
			if !ok {
				continue
			}
			id := strField(entry, "id")
			if id == "" || !isVisionCapable(entry, nil) {
				continue
			}
			mi := ModelInfo{ID: id, Capabilities: stringList(entry["capabilities"])}
			if dg := strField(entry, "digest"); dg != "" {
				mi.Digest = &dg
			}
			out = append(out, mi)
		}
	}
	return out, nil
}

// Chat sends a chat completion with a capped reasoning effort.
func (o *OpenAI) Chat(ctx context.Context, model string, msgs []Message, opts ChatOptions) (string, error) {
	wire := make([]map[string]any, len(msgs))
	for i, m := range msgs {
		if len(m.Images) == 0 {
			wire[i] = map[string]any{"role": m.Role, "content": m.Text}
			continue
		}
		parts := []map[string]any{{"type": "text", "text": m.Text}}
		for _, img := range m.Images {
			mime := img.MediaType
			if mime == "" {
				mime = guessMime(img.Name)
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]string{
				"url": "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(img.Data)}})
		}
		wire[i] = map[string]any{"role": m.Role, "content": parts}
	}
	body := map[string]any{"model": model, "messages": wire, "temperature": o.temperature(), "reasoning_effort": "low"}
	started := time.Now()
	data, err := o.do(ctx, http.MethodPost, "/chat/completions", body)
	if err != nil {
		return "", err
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content *string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens     *int `json:"prompt_tokens"`
			CompletionTokens *int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return "", err
	}
	if opts.Trace != nil {
		tele := telemetry(msgs, time.Since(started))
		if resp.Usage != nil {
			if resp.Usage.PromptTokens != nil {
				tele["prompt_tokens"] = *resp.Usage.PromptTokens
			}
			if resp.Usage.CompletionTokens != nil {
				tele["completion_tokens"] = *resp.Usage.CompletionTokens
			}
		}
		opts.Trace(tele)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == nil {
		return "", nil
	}
	return *resp.Choices[0].Message.Content, nil
}
