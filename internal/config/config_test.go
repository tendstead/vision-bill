package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func envSrc(m map[string]string, files []string, yamlPath string) Sources {
	return Sources{
		Env: func(k string) string { return m[k] },
		EnvNames: func() []string {
			var n []string
			for k := range m {
				n = append(n, k)
			}
			return n
		},
		DotenvFiles: files,
		YAMLPath:    yamlPath,
	}
}

var required = map[string]string{
	"LLM__HOST": "h", "LLM__API_KEY": "k", "LLM__MODEL_NAME": "m", "LLM__TEMPERATURE": "0.5",
	"PG__USER": "u", "PG__PASSWORD": "p", "AUTH__SECRET_KEY": "s",
}

func TestDefaultsAndRequired(t *testing.T) {
	c, err := Load(envSrc(required, nil, "/nonexistent.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if c.API.Port != 8080 || c.PG.DB != "vision_bill" || c.Worker.CheckIntervalSeconds != 300 ||
		!c.Auth.AllowRegistration || c.Auth.SessionCookieName != "vb_session" || c.LLM.Provider != "ollama" {
		t.Fatalf("defaults: %+v", c)
	}
	_, err = Load(envSrc(nil, nil, "/nonexistent.yaml"))
	if err == nil || !strings.Contains(err.Error(), "auth.secret_key") || !strings.Contains(err.Error(), "pg.user") {
		t.Fatalf("missing keys not reported: %v", err)
	}
}

func TestPrecedenceEnvDotenvYAML(t *testing.T) {
	dir := t.TempDir()
	yml := filepath.Join(dir, "config.yaml")
	os.WriteFile(yml, []byte("llm:\n  host: yaml-host\n  model_name: yaml-model\n  api_key: yk\n  temperature: 0.2\npg:\n  user: yu\n  password: yp\nauth:\n  secret_key: ys\n  allow_registration: false\n"), 0o600)
	env1 := filepath.Join(dir, ".env")
	os.WriteFile(env1, []byte("# c\nLLM__HOST=dot-host\nexport LLM__MODEL_NAME=\"dot model\"\nPG__USER=du # trailing\n"), 0o600)
	env2 := filepath.Join(dir, ".env.local")
	os.WriteFile(env2, []byte("PG__USER=local-user\n"), 0o600)
	c, err := Load(envSrc(map[string]string{"llm__model_name": "env-model"}, []string{env1, env2}, yml))
	if err != nil {
		t.Fatal(err)
	}
	if c.LLM.ModelName != "env-model" || c.LLM.Host != "dot-host" || c.PG.User != "local-user" ||
		c.PG.Password != "yp" || c.Auth.SecretKey != "ys" || c.Auth.AllowRegistration {
		t.Fatalf("precedence wrong: %+v", c)
	}
	src := EditableSources(envSrc(map[string]string{"LLM__MODEL_NAME": "x"}, []string{env1}, yml))
	want := map[string]string{
		"llm.model_name": "environment", "llm.host": "environment", "llm.temperature": "config",
		"llm.provider": "default", "auth.allow_registration": "config",
	}
	for k, v := range want {
		if src[k] != v {
			t.Errorf("%s = %s, want %s", k, src[k], v)
		}
	}
}

func TestValidation(t *testing.T) {
	m := map[string]string{"LLM__PROVIDER": "bogus", "API__PORT": "x", "LLM__ANALYSIS_TIMEOUT_SECONDS": "601", "AUTH__ALLOW_REGISTRATION": "maybe"}
	for k, v := range required {
		m[k] = v
	}
	_, err := Load(envSrc(m, nil, "/nonexistent.yaml"))
	if err == nil {
		t.Fatal("expected error")
	}
	for _, k := range []string{"llm.provider", "api.port", "llm.analysis_timeout_seconds", "auth.allow_registration"} {
		if !strings.Contains(err.Error(), k) {
			t.Errorf("missing %s in %v", k, err)
		}
	}
}

func TestWriteInitializeRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "config.yaml")
	c, _ := Load(envSrc(required, nil, path))
	created, err := Initialize(path, c)
	if err != nil || !created {
		t.Fatalf("%v %v", created, err)
	}
	if created, _ = Initialize(path, c); created {
		t.Fatal("must not overwrite")
	}
	st, _ := os.Stat(path)
	if st.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v", st.Mode())
	}
	back, err := Load(envSrc(nil, nil, path))
	if err != nil || back.LLM.Host != "h" || back.Auth.SecretKey != "s" || back.LLM.Temperature != 0.5 {
		t.Fatalf("round trip: %v %+v", err, back)
	}
}

func TestDSNEscapes(t *testing.T) {
	p := PG{User: "a@b", Password: "p:/ w", Host: "h", Port: 5432, DB: "d"}
	if got := p.DSN(); got != "postgres://a%40b:p%3A%2F%20w@h:5432/d" {
		t.Fatal(got)
	}
}
