package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// OptionSpec describes one whitelisted per-request provider option.
type OptionSpec struct {
	Name    string   `json:"name"`
	Type    string   `json:"type"` // "integer" or "number"
	Min     *float64 `json:"min"`
	Max     *float64 `json:"max"`
	Summary string   `json:"summary"`
}

func f64(v float64) *float64 { return &v }

// Settings that live in the inference server's start flags (llama.cpp -ngl,
// Ollama environment) cannot be changed per request and are not listed.
var ollamaOptions = []OptionSpec{
	{"num_gpu", "integer", f64(-1), nil, "Layers offloaded to the GPU (0 = CPU only, -1 = all)"},
	{"num_ctx", "integer", f64(1), nil, "Context window in tokens"},
	{"num_thread", "integer", f64(0), nil, "CPU threads used for generation"},
	{"top_p", "number", f64(0), f64(1), "Nucleus sampling"},
	{"top_k", "integer", f64(0), nil, "Top-k sampling"},
	{"seed", "integer", nil, nil, "Random seed"},
	{"repeat_penalty", "number", f64(0), f64(10), "Repetition penalty"},
}

var openAIOptions = []OptionSpec{
	{"seed", "integer", nil, nil, "Random seed"},
	{"top_p", "number", f64(0), f64(1), "Nucleus sampling"},
}

// ProviderOptionSpecs lists the options a provider kind accepts per request.
func ProviderOptionSpecs(kind string) []OptionSpec {
	switch kind {
	case "OllamaProvider":
		return ollamaOptions
	case "OpenAIProvider":
		return openAIOptions
	}
	return []OptionSpec{}
}

// SanitizeProviderOptions validates per-request options against the provider's
// whitelist and returns normalised values (int64 or float64). Unknown keys
// are reported as not supported by the provider.
func SanitizeProviderOptions(kind string, in map[string]any) (map[string]any, error) {
	if len(in) == 0 {
		return nil, nil
	}
	specs := map[string]OptionSpec{}
	for _, s := range ProviderOptionSpecs(kind) {
		specs[s.Name] = s
	}
	var unsupported []string
	for k := range in {
		if _, ok := specs[k]; !ok {
			unsupported = append(unsupported, k)
		}
	}
	if len(unsupported) > 0 {
		sort.Strings(unsupported)
		return nil, fmt.Errorf("Provider option(s) not supported by this provider: %s", strings.Join(unsupported, ", "))
	}
	out := map[string]any{}
	for k, v := range in {
		spec := specs[k]
		var f float64
		switch n := v.(type) {
		case json.Number:
			x, err := n.Float64()
			if err != nil {
				return nil, fmt.Errorf("Provider option %s must be a number", k)
			}
			f = x
		case float64:
			f = n
		case int:
			f = float64(n)
		case int64:
			f = float64(n)
		default:
			return nil, fmt.Errorf("Provider option %s must be a number", k)
		}
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, fmt.Errorf("Provider option %s must be a finite number", k)
		}
		if spec.Type == "integer" {
			if f != math.Trunc(f) || math.Abs(f) > 1<<53 {
				return nil, fmt.Errorf("Provider option %s must be an integer", k)
			}
		}
		if spec.Min != nil && f < *spec.Min {
			return nil, fmt.Errorf("Provider option %s must be at least %v", k, *spec.Min)
		}
		if spec.Max != nil && f > *spec.Max {
			return nil, fmt.Errorf("Provider option %s must be at most %v", k, *spec.Max)
		}
		if k == "repeat_penalty" && f == 0 {
			return nil, fmt.Errorf("Provider option %s must be greater than 0", k)
		}
		if spec.Type == "integer" {
			out[k] = int64(f)
		} else {
			out[k] = f
		}
	}
	return out, nil
}
