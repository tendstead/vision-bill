package config

import (
	"os"
	"testing"
)

// Skipped unless a Python-written config is provided (VB_PY_CONFIG).
func TestLoadsPythonWrittenYAML(t *testing.T) {
	p := os.Getenv("VB_PY_CONFIG")
	if p == "" {
		t.Skip("VB_PY_CONFIG not set")
	}
	c, err := Load(envSrc(nil, nil, p))
	if err != nil {
		t.Fatal(err)
	}
	if c.LLM.Provider == "" || c.Auth.SecretKey == "" || c.Auth.SessionMaxAgeSeconds != 1209600 {
		t.Fatalf("%+v", c)
	}
}
