// Package llm contains the model-provider adapters (Ollama and
// OpenAI-compatible), the shared prompt and the extraction/repair loop.
// Receipt data is only ever sent to the endpoint the operator configured.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/tendstead/vision-bill/internal/domain"
)

// RetryLimit is the number of attempts including the first.
const RetryLimit = 3

// ModelInfo describes a vision-capable model.
type ModelInfo struct {
	ID            string   `json:"id"`
	ParameterSize *string  `json:"parameter_size"`
	Digest        *string  `json:"digest"`
	Capabilities  []string `json:"capabilities"`
}

// Analysis is a validated extraction with telemetry.
type Analysis struct {
	Receipt     *domain.Receipt
	Attempts    int
	ElapsedMS   float64
	ModelDigest *string
}

// UnreadableError means the model cannot read the receipt reliably.
type UnreadableError struct{ Message string }

func (e *UnreadableError) Error() string { return e.Message }

// TimeoutError means an extraction exceeded its deadline.
type TimeoutError struct{ Message string }

func (e *TimeoutError) Error() string { return e.Message }

// NewTimeoutError returns the standard timeout error.
func NewTimeoutError() *TimeoutError {
	return &TimeoutError{"The LLM request timed out. Please try again."}
}

// ValidationFailure is a model reply that did not match the schema (retryable).
type ValidationFailure struct{ Detail string }

func (e *ValidationFailure) Error() string { return "LLM output failed validation: " + e.Detail }

// Image is one input photo.
type Image struct {
	Name      string // file name (MIME fallback when MediaType is empty)
	MediaType string
	Data      []byte
}

// Message is a chat message; user messages may carry images.
type Message struct {
	Role   string
	Text   string
	Images []Image
}

// Provider is the model backend abstraction.
type Provider interface {
	Models(ctx context.Context) ([]ModelInfo, error)
	CheckConnection(ctx context.Context) bool
	// Chat sends messages and returns the reply text.
	Chat(ctx context.Context, model string, msgs []Message, opts ChatOptions) (string, error)
	SetTemperature(t float64)
	// Temperature is the temperature currently in use.
	Temperature() float64
	// WithTemperature returns an independent copy using temperature t; the
	// live provider is never mutated by experiments.
	WithTemperature(t float64) Provider
	// SupportsThinking reports whether the "think" request option is honoured.
	SupportsThinking() bool
	// Kind names the provider implementation ("OllamaProvider", "OpenAIProvider").
	Kind() string
	// EndpointFingerprint identifies the configured endpoint without exposing it.
	EndpointFingerprint() string
	// Timeout is the complete extraction deadline.
	Timeout() time.Duration
	// Host is the configured host string.
	Host() string
}

// ChatOptions are per-request options (experiments).
type ChatOptions struct {
	Think *bool
	// Trace, when set, receives request telemetry after every chat call.
	Trace func(map[string]any)
}

var providePattern = regexp.MustCompile(`(?is)provide.*image`)

// Analyse runs the extraction on the ordered photos with up to RetryLimit
// self-correcting attempts.
func Analyse(ctx context.Context, p Provider, model string, images []Image, tags []string, popts PromptOptions, copts ChatOptions) (*Analysis, error) {
	started := time.Now()
	msgs := []Message{{Role: "user", Text: BuildPrompt(tags, len(images), popts), Images: images}}

	var last error
	for attempt := 1; attempt <= RetryLimit; attempt++ {
		content, err := p.Chat(ctx, model, msgs, copts)
		if err != nil {
			return nil, err
		}
		if content == "" {
			last = errors.New("Empty response from model")
			continue
		}
		trimmed := strings.TrimLeft(content, " \t\r\n")
		if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "```") && providePattern.MatchString(content) {
			return nil, fmt.Errorf("Model '%s' returned a message indicating it cannot process images. "+
				"Please ensure the model supports vision capabilities.", model)
		}
		receipt, perr := ParseResponse(content, len(images))
		if perr == nil {
			return &Analysis{Receipt: receipt, Attempts: attempt, ElapsedMS: float64(time.Since(started).Microseconds()) / 1000}, nil
		}
		var unreadable *UnreadableError
		if errors.As(perr, &unreadable) {
			return nil, perr
		}
		last = perr
		msgs = append(msgs,
			Message{Role: "assistant", Text: content},
			Message{Role: "user", Text: "That response was not valid JSON matching the required schema. " +
				"Error: " + perr.Error() + "\n\nPlease respond again with ONLY corrected JSON."})
	}
	return nil, fmt.Errorf("Failed to get a valid response from model '%s' after %d attempts: %w", model, RetryLimit, last)
}

// ParseResponse validates a model reply (JSON, optionally fenced) into a Receipt.
func ParseResponse(response string, imageCount int) (*domain.Receipt, error) {
	raw := strings.TrimSpace(response)
	raw = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(raw, "```json"), "```"))
	if msg, ok := unreadableMessage(raw); ok {
		return nil, &UnreadableError{Message: truncateRunes(msg, 1000)}
	}
	if imageCount > 1 {
		r, issues := domain.ParseReceipt([]byte(raw), domain.Options{MultiImage: true})
		if issues != nil {
			return nil, &ValidationFailure{Detail: issues.Error()}
		}
		out, err := Reconcile(r, imageCount)
		if err != nil {
			return nil, err
		}
		return out, nil
	}
	r, issues := domain.ParseReceipt([]byte(raw), domain.Options{})
	if issues != nil {
		return nil, &ValidationFailure{Detail: issues.Error()}
	}
	return r, nil
}

func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) > n {
		return string(r[:n])
	}
	return s
}

// Reconcile removes verified overlap rows from a multi-photo extraction.
// Errors are plain (retryable) validation failures.
func Reconcile(r *domain.Receipt, imageCount int) (*domain.Receipt, error) {
	items := r.LineItems
	sources := make([]int, len(items))
	for i, it := range items {
		sources[i] = *it.SourceImage
	}
	for i := range sources {
		if i > 0 && sources[i] < sources[i-1] || sources[i] >= imageCount {
			return nil, &ValidationFailure{"Line items must be grouped in input photo order with valid source_image"}
		}
	}
	type pair struct{ a, b int }
	groups := map[int][]pair{}
	for _, c := range r.DuplicateCandidates {
		a, b := c.First, c.Second
		if !(0 <= a && a < b && b < len(items)) {
			return nil, &ValidationFailure{"Duplicate candidate indices are out of order or out of range"}
		}
		if sources[b] != sources[a]+1 {
			return nil, &ValidationFailure{"Duplicate candidates must come from adjacent photos"}
		}
		groups[sources[a]] = append(groups[sources[a]], pair{a, b})
	}
	removed := map[int]bool{}
	// Deterministic order of source groups.
	for source := 0; source < imageCount; source++ {
		pairs := groups[source]
		if len(pairs) == 0 {
			continue
		}
		sortPairs(pairs, func(x, y pair) bool { return x.a < y.a || x.a == y.a && x.b < y.b })
		var earlier, later []int
		for i, v := range sources {
			if v == source {
				earlier = append(earlier, i)
			}
			if v == source+1 {
				later = append(later, i)
			}
		}
		n := len(pairs)
		ok := n <= len(earlier) && n <= len(later)
		if ok {
			for k := 0; k < n; k++ {
				if pairs[k].a != earlier[len(earlier)-n+k] || pairs[k].b != later[k] {
					ok = false
				}
			}
		}
		if !ok {
			return nil, &ValidationFailure{"Duplicate candidates must describe an ordered photo-boundary overlap"}
		}
		all := true
		for _, p := range pairs {
			if !verifyDuplicate(items[p.a], items[p.b]) {
				all = false
			}
		}
		if all {
			for _, p := range pairs {
				removed[p.b] = true
			}
		}
	}
	out := *r
	out.DuplicateCandidates = nil
	out.LineItems = nil
	for i, it := range items {
		if !removed[i] {
			it.SourceImage = nil
			out.LineItems = append(out.LineItems, it)
		}
	}
	if out.LineItems == nil {
		out.LineItems = []domain.LineItem{}
	}
	return &out, nil
}

func sortPairs[T any](s []T, less func(a, b T) bool) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && less(s[j], s[j-1]); j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// verifyDuplicate demands equal normalised name, quantity and total.
func verifyDuplicate(a, b domain.LineItem) bool {
	norm := func(s string) string { return domain.NormalizeHistoryKey(s) }
	l, r := norm(a.Description), norm(b.Description)
	return l != "" && r != "" && l == r && a.Quantity.Equal(b.Quantity) && a.TotalPrice.Equal(b.TotalPrice)
}

// unreadableMessage extracts {"error": {"code": "unreadable", "message": str}}.
func unreadableMessage(raw string) (string, bool) {
	var doc struct {
		Error *struct {
			Code    any `json:"code"`
			Message any `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &doc); err != nil || doc.Error == nil {
		return "", false
	}
	code, _ := doc.Error.Code.(string)
	msg, ok := doc.Error.Message.(string)
	return msg, code == "unreadable" && ok
}
