package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// telemetry builds the request-independent part of a call trace entry.
func telemetry(msgs []Message, elapsed time.Duration) map[string]any {
	type m struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	prompt := make([]m, len(msgs))
	for i, x := range msgs {
		prompt[i] = m{x.Role, x.Text}
	}
	raw, _ := json.Marshal(prompt)
	return map[string]any{"elapsed_ms": float64(elapsed.Microseconds()) / 1000, "prompt_sha256": sha256Hex(string(raw))}
}

// httpClient returns a client with a 5s connect timeout and no overall
// timeout (the caller's context carries the extraction deadline).
func httpClient() *http.Client {
	return &http.Client{Transport: &http.Transport{
		DialContext:         (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
		TLSHandshakeTimeout: 5 * time.Second,
		MaxIdleConnsPerHost: 4,
		IdleConnTimeout:     60 * time.Second,
	}}
}

// StatusError is a non-2xx response from the model server.
type StatusError struct {
	Code int
	Body string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("model server returned HTTP %d: %s", e.Code, e.Body)
}

// IsTimeout reports context deadline or network timeout errors.
func IsTimeout(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// ParseOllamaHost normalises the configured host:
// "host", "host:port", "http(s)://host[:port][/path]".
func ParseOllamaHost(host string) string {
	port := "11434"
	scheme, hostport, found := strings.Cut(host, "://")
	if !found || hostport == "" {
		scheme, hostport = "http", host
	} else if scheme == "http" {
		port = "80"
	} else if scheme == "https" {
		port = "443"
	}
	u, err := url.Parse(scheme + "://" + hostport)
	h := "127.0.0.1"
	if err == nil && u.Hostname() != "" {
		h = u.Hostname()
	}
	if err == nil && u.Port() != "" {
		port = u.Port()
	}
	if ip := net.ParseIP(h); ip != nil && strings.Contains(h, ":") {
		h = "[" + h + "]"
	}
	out := scheme + "://" + h + ":" + port
	if err == nil {
		if p := strings.Trim(u.Path, "/"); p != "" {
			out += "/" + p
		}
	}
	return out
}

// Ollama talks to a local Ollama server.
type Ollama struct {
	base    string
	host    string
	timeout time.Duration
	client  *http.Client

	mu   sync.RWMutex
	temp float64
}

// NewOllama creates an Ollama provider for host.
func NewOllama(host string, temperature float64, timeout time.Duration) *Ollama {
	return &Ollama{base: ParseOllamaHost(host), host: host, timeout: timeout, client: httpClient(), temp: temperature}
}

func (o *Ollama) Host() string             { return o.host }
func (o *Ollama) Timeout() time.Duration   { return o.timeout }
func (o *Ollama) SetTemperature(t float64) { o.mu.Lock(); o.temp = t; o.mu.Unlock() }
func (o *Ollama) temperature() float64     { o.mu.RLock(); defer o.mu.RUnlock(); return o.temp }
func (o *Ollama) Temperature() float64     { return o.temperature() }
func (o *Ollama) SupportsThinking() bool   { return true }
func (o *Ollama) Kind() string             { return "OllamaProvider" }

// WithTemperature returns an independent copy with a different temperature.
func (o *Ollama) WithTemperature(t float64) Provider {
	return &Ollama{base: o.base, host: o.host, timeout: o.timeout, client: o.client, temp: t}
}

// EndpointFingerprint hashes the normalised base URL.
func (o *Ollama) EndpointFingerprint() string { return sha256Hex(o.base) }

func (o *Ollama) do(ctx context.Context, method, path string, body any, out any) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, o.base+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		return &StatusError{resp.StatusCode, truncateRunes(string(data), 500)}
	}
	return json.Unmarshal(data, out)
}

// CheckConnection lists models; any failure means "unreachable".
func (o *Ollama) CheckConnection(ctx context.Context) bool {
	var out struct{}
	return o.do(ctx, http.MethodGet, "/api/tags", nil, &out) == nil
}

// Models lists models having the vision capability.
func (o *Ollama) Models(ctx context.Context) ([]ModelInfo, error) {
	var list struct {
		Models []struct {
			Model   string `json:"model"`
			Name    string `json:"name"`
			Digest  string `json:"digest"`
			Details *struct {
				ParameterSize string `json:"parameter_size"`
			} `json:"details"`
		} `json:"models"`
	}
	if err := o.do(ctx, http.MethodGet, "/api/tags", nil, &list); err != nil {
		return nil, err
	}
	out := []ModelInfo{}
	for _, m := range list.Models {
		name := m.Model
		if name == "" {
			name = m.Name
		}
		if name == "" {
			continue
		}
		var show struct {
			Capabilities []string `json:"capabilities"`
		}
		if err := o.do(ctx, http.MethodPost, "/api/show", map[string]string{"model": name}, &show); err != nil {
			return nil, err
		}
		vision := false
		for _, c := range show.Capabilities {
			if c == "vision" {
				vision = true
			}
		}
		if !vision {
			continue
		}
		mi := ModelInfo{ID: name, Capabilities: show.Capabilities}
		if m.Digest != "" {
			d := m.Digest
			mi.Digest = &d
		}
		if m.Details != nil && m.Details.ParameterSize != "" {
			ps := m.Details.ParameterSize
			mi.ParameterSize = &ps
		}
		out = append(out, mi)
	}
	return out, nil
}

// Chat sends a non-streaming chat request.
func (o *Ollama) Chat(ctx context.Context, model string, msgs []Message, opts ChatOptions) (string, error) {
	type wireMsg struct {
		Role    string   `json:"role"`
		Content string   `json:"content"`
		Images  []string `json:"images,omitempty"`
	}
	wire := make([]wireMsg, len(msgs))
	for i, m := range msgs {
		w := wireMsg{Role: m.Role, Content: m.Text}
		for _, img := range m.Images {
			w.Images = append(w.Images, base64.StdEncoding.EncodeToString(img.Data))
		}
		wire[i] = w
	}
	options := map[string]any{}
	for k, v := range opts.Options {
		options[k] = v
	}
	options["temperature"] = o.temperature()
	body := map[string]any{
		"model": model, "messages": wire, "stream": false,
		"options": options,
	}
	if opts.Think != nil {
		body["think"] = *opts.Think
	}
	var resp struct {
		Message struct {
			Content  string  `json:"content"`
			Thinking *string `json:"thinking"`
		} `json:"message"`
		LoadDuration       *float64 `json:"load_duration"`
		PromptEvalDuration *float64 `json:"prompt_eval_duration"`
		EvalDuration       *float64 `json:"eval_duration"`
		PromptEvalCount    *int     `json:"prompt_eval_count"`
		EvalCount          *int     `json:"eval_count"`
	}
	started := time.Now()
	if err := o.do(ctx, http.MethodPost, "/api/chat", body, &resp); err != nil {
		return "", err
	}
	if opts.Trace != nil {
		tele := telemetry(msgs, time.Since(started))
		for k, v := range map[string]*float64{"load_ms": resp.LoadDuration, "prompt_eval_ms": resp.PromptEvalDuration, "eval_ms": resp.EvalDuration} {
			if v != nil {
				tele[k] = *v / 1_000_000
			}
		}
		if resp.PromptEvalCount != nil {
			tele["prompt_eval_count"] = *resp.PromptEvalCount
		}
		if resp.EvalCount != nil {
			tele["eval_count"] = *resp.EvalCount
		}
		if resp.Message.Thinking != nil {
			tele["thinking_characters"] = len([]rune(*resp.Message.Thinking))
		}
		opts.Trace(tele)
	}
	return resp.Message.Content, nil
}
