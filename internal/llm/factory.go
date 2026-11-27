package llm

import (
	"fmt"
	"time"

	"github.com/tendstead/vision-bill/internal/config"
)

// New builds the configured provider. anthropic is declared but unsupported.
func New(c config.LLM) (Provider, error) {
	timeout := time.Duration(c.AnalysisTimeoutSeconds * float64(time.Second))
	switch c.Provider {
	case "ollama":
		return NewOllama(c.Host, c.Temperature, timeout), nil
	case "openai":
		return NewOpenAI(c.Host, c.APIKey, c.Temperature, timeout), nil
	}
	return nil, fmt.Errorf("Unsupported LLM provider: %s", c.Provider)
}
