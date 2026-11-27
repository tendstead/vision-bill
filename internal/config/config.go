// Package config loads settings from, in order of precedence: process environment > .env/.env.local > persistent YAML.
package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

const (
	ConfigPathEnv     = "VISION_BILL_CONFIG_PATH"
	DefaultConfigPath = "/app/config/config.yaml"
)

// EditableKeys maps admin-editable settings to their environment variables.
var EditableKeys = map[string]string{
	"llm.provider":            "LLM__PROVIDER",
	"llm.host":                "LLM__HOST",
	"llm.model_name":          "LLM__MODEL_NAME",
	"llm.temperature":         "LLM__TEMPERATURE",
	"auth.allow_registration": "AUTH__ALLOW_REGISTRATION",
}

type Config struct {
	API    API    `yaml:"api"`
	Images Images `yaml:"images"`
	LLM    LLM    `yaml:"llm"`
	PG     PG     `yaml:"pg"`
	Worker Worker `yaml:"worker"`
	Auth   Auth   `yaml:"auth"`
}

type API struct {
	Port     int    `yaml:"port"`
	LogLevel string `yaml:"log_level"`
}

type Images struct {
	SaveDir             string `yaml:"save_dir"`
	TmpDir              string `yaml:"tmp_dir"`
	BypassReviewDefault bool   `yaml:"bypass_review_default"`
	ThumbnailMaxEdge    int    `yaml:"thumbnail_max_edge"`
	ThumbnailQuality    int    `yaml:"thumbnail_quality"`
}

type LLM struct {
	Provider               string  `yaml:"provider"`
	Host                   string  `yaml:"host"`
	APIKey                 string  `yaml:"api_key"`
	ModelName              string  `yaml:"model_name"`
	Temperature            float64 `yaml:"temperature"`
	AnalysisTimeoutSeconds float64 `yaml:"analysis_timeout_seconds"`
}

type PG struct {
	User     string `yaml:"user"`
	Password string `yaml:"password"`
	DB       string `yaml:"db"`
	Host     string `yaml:"host"`
	Port     int    `yaml:"port"`
}

type Worker struct {
	CheckIntervalSeconds int `yaml:"check_interval_seconds"`
}

type Auth struct {
	SecretKey            string  `yaml:"secret_key"`
	SessionCookieName    string  `yaml:"session_cookie_name"`
	SessionMaxAgeSeconds int     `yaml:"session_max_age_seconds"`
	SessionSecure        bool    `yaml:"session_secure"`
	BootstrapUsername    *string `yaml:"bootstrap_username"`
	BootstrapPassword    *string `yaml:"bootstrap_password"`
	AllowRegistration    bool    `yaml:"allow_registration"`
	AdminCanSeeAll       bool    `yaml:"admin_can_see_all"`
	Pepper               *string `yaml:"pepper"`
}

// PGDSN returns a connection URL for pgx.
func (p PG) DSN() string {
	return fmt.Sprintf("postgres://%s:%s@%s:%d/%s", escape(p.User), escape(p.Password), p.Host, p.Port, escape(p.DB))
}

func escape(s string) string {
	var b strings.Builder
	for _, c := range []byte(s) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-', c == '_', c == '.', c == '~':
			b.WriteByte(c)
		default:
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}

// EffectivePepper is the pepper in play (explicit pepper, else secret key).
func (a Auth) EffectivePepper() string {
	if a.Pepper != nil && *a.Pepper != "" {
		return *a.Pepper
	}
	return a.SecretKey
}

// Path returns the YAML path from the environment or the default.
func Path(env func(string) string) string {
	if p := env(ConfigPathEnv); p != "" {
		return expandHome(p)
	}
	return DefaultConfigPath
}

func expandHome(p string) string {
	if strings.HasPrefix(p, "~/") {
		if h, err := os.UserHomeDir(); err == nil {
			return filepath.Join(h, p[2:])
		}
	}
	return p
}

// Sources describes where values are read from; used by tests.
type Sources struct {
	Env         func(string) string
	EnvNames    func() []string
	DotenvFiles []string
	YAMLPath    string
}

// OSSources reads the real process environment, .env/.env.local in the
// working directory and the configured YAML path.
func OSSources() Sources {
	return Sources{
		Env: os.Getenv,
		EnvNames: func() []string {
			var n []string
			for _, kv := range os.Environ() {
				k, _, _ := strings.Cut(kv, "=")
				n = append(n, k)
			}
			return n
		},
		DotenvFiles: []string{".env", ".env.local"},
		YAMLPath:    Path(os.Getenv),
	}
}

// flat is a case-insensitive "section.key" -> raw string map.
type flat map[string]string

func envFlat(s Sources) flat {
	out := flat{}
	for _, name := range s.EnvNames() {
		if k, ok := envKey(name); ok {
			out[k] = s.Env(name)
		}
	}
	return out
}

func envKey(name string) (string, bool) {
	lower := strings.ToLower(name)
	section, key, ok := strings.Cut(lower, "__")
	if !ok || section == "" || key == "" || strings.Contains(key, "__") {
		return "", false
	}
	return section + "." + key, true
}

func dotenvFlat(files []string) flat {
	out := flat{}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for name, value := range parseDotenv(string(b)) {
			if k, ok := envKey(name); ok {
				out[k] = value
			}
		}
	}
	return out
}

func parseDotenv(text string) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(strings.TrimRight(line, "\r"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		name, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		out[strings.TrimSpace(name)] = unquote(strings.TrimSpace(value))
	}
	return out
}

func unquote(v string) string {
	if len(v) >= 2 && (v[0] == '"' || v[0] == '\'') {
		if end := strings.IndexByte(v[1:], v[0]); end >= 0 {
			return v[1 : end+1]
		}
	}
	if i := strings.Index(v, " #"); i >= 0 {
		v = v[:i]
	}
	return strings.TrimSpace(v)
}

func yamlFlat(path string) (flat, error) {
	out := flat{}
	b, err := os.ReadFile(path)
	if err != nil {
		return out, nil // absent or unreadable: treated as empty
	}
	var doc map[string]any
	if err := yaml.Unmarshal(b, &doc); err != nil {
		return out, nil
	}
	for section, v := range doc {
		m, ok := v.(map[string]any)
		if !ok {
			continue
		}
		for key, val := range m {
			if val == nil {
				continue
			}
			out[strings.ToLower(section)+"."+strings.ToLower(key)] = fmt.Sprint(val)
		}
	}
	return out, nil
}

// Load resolves the configuration. Missing required values produce an error
// listing every missing key.
func Load(s Sources) (*Config, error) {
	yml, err := yamlFlat(s.YAMLPath)
	if err != nil {
		return nil, err
	}
	layers := []flat{envFlat(s), dotenvFlat(s.DotenvFiles), yml}
	get := func(key string) (string, bool) {
		for _, l := range layers {
			if v, ok := l[key]; ok {
				return v, true
			}
		}
		return "", false
	}
	r := &resolver{get: get}
	c := &Config{}
	c.API.Port = r.intv("api.port", 8080, 1, 65535)
	c.API.LogLevel = r.str("api.log_level", "INFO")
	c.Images.SaveDir = r.str("images.save_dir", "/app/uploads/")
	c.Images.TmpDir = r.str("images.tmp_dir", "/app/uploads_tmp/")
	c.Images.BypassReviewDefault = r.boolv("images.bypass_review_default", false)
	c.Images.ThumbnailMaxEdge = r.intv("images.thumbnail_max_edge", 512, 64, 1<<30)
	c.Images.ThumbnailQuality = r.intv("images.thumbnail_quality", 80, 1, 100)
	c.LLM.Provider = r.enum("llm.provider", "ollama", "ollama", "anthropic", "openai")
	c.LLM.Host = r.required("llm.host")
	c.LLM.APIKey = r.required("llm.api_key")
	c.LLM.ModelName = r.required("llm.model_name")
	c.LLM.Temperature = r.floatRequired("llm.temperature")
	c.LLM.AnalysisTimeoutSeconds = r.floatv("llm.analysis_timeout_seconds", 600, 0, 600, true)
	c.PG.User = r.required("pg.user")
	c.PG.Password = r.required("pg.password")
	c.PG.DB = r.str("pg.db", "vision_bill")
	c.PG.Host = r.str("pg.host", "localhost")
	c.PG.Port = r.intv("pg.port", 5432, 1, 65535)
	c.Worker.CheckIntervalSeconds = r.intv("worker.check_interval_seconds", 300, 1, 1<<30)
	c.Auth.SecretKey = r.required("auth.secret_key")
	c.Auth.SessionCookieName = r.str("auth.session_cookie_name", "vb_session")
	c.Auth.SessionMaxAgeSeconds = r.intv("auth.session_max_age_seconds", 1_209_600, 1, 1<<30)
	c.Auth.SessionSecure = r.boolv("auth.session_secure", false)
	c.Auth.BootstrapUsername = r.optional("auth.bootstrap_username")
	c.Auth.BootstrapPassword = r.optional("auth.bootstrap_password")
	c.Auth.AllowRegistration = r.boolv("auth.allow_registration", true)
	c.Auth.AdminCanSeeAll = r.boolv("auth.admin_can_see_all", false)
	c.Auth.Pepper = r.optional("auth.pepper")
	if len(r.errs) > 0 {
		return nil, errors.New("invalid configuration: " + strings.Join(r.errs, "; "))
	}
	return c, nil
}

type resolver struct {
	get  func(string) (string, bool)
	errs []string
}

func (r *resolver) fail(key, msg string) { r.errs = append(r.errs, key+": "+msg) }

func (r *resolver) str(key, def string) string {
	if v, ok := r.get(key); ok {
		return v
	}
	return def
}

func (r *resolver) required(key string) string {
	v, ok := r.get(key)
	if !ok {
		r.fail(key, "field required")
	}
	return v
}

func (r *resolver) optional(key string) *string {
	if v, ok := r.get(key); ok {
		return &v
	}
	return nil
}

func (r *resolver) enum(key, def string, allowed ...string) string {
	v := r.str(key, def)
	for _, a := range allowed {
		if v == a {
			return v
		}
	}
	r.fail(key, "must be one of "+strings.Join(allowed, ", "))
	return def
}

func (r *resolver) intv(key string, def, lo, hi int) int {
	v, ok := r.get(key)
	if !ok {
		return def
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil || n < lo || n > hi {
		r.fail(key, "invalid integer")
		return def
	}
	return n
}

func (r *resolver) floatv(key string, def, lo, hi float64, loExclusive bool) float64 {
	v, ok := r.get(key)
	if !ok {
		return def
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil || f > hi || f < lo || (loExclusive && f == lo) {
		r.fail(key, "invalid number")
		return def
	}
	return f
}

func (r *resolver) floatRequired(key string) float64 {
	v, ok := r.get(key)
	if !ok {
		r.fail(key, "field required")
		return 0
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		r.fail(key, "invalid number")
	}
	return f
}

func (r *resolver) boolv(key string, def bool) bool {
	v, ok := r.get(key)
	if !ok {
		return def
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "t", "yes", "y", "on":
		return true
	case "0", "false", "f", "no", "n", "off":
		return false
	}
	r.fail(key, "invalid boolean")
	return def
}

// EditableSources reports "environment", "config" or "default" for each
// admin-editable key.
func EditableSources(s Sources) map[string]string {
	names := map[string]bool{}
	for _, n := range s.EnvNames() {
		names[n] = true
	}
	for _, f := range s.DotenvFiles {
		if b, err := os.ReadFile(f); err == nil {
			for n := range parseDotenv(string(b)) {
				names[n] = true
			}
		}
	}
	yml, _ := yamlFlat(s.YAMLPath)
	out := map[string]string{}
	for key, envName := range EditableKeys {
		switch {
		case names[envName]:
			out[key] = "environment"
		default:
			if _, ok := yml[key]; ok {
				out[key] = "config"
			} else {
				out[key] = "default"
			}
		}
	}
	return out
}

// Write atomically persists the configuration with 0600 permissions.
func Write(path string, c *Config) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	b, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	tmp := filepath.Join(filepath.Dir(path), "."+filepath.Base(path)+".tmp")
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	if err := os.Chmod(tmp, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Initialize creates the YAML file on first start; failures are non-fatal
// (read-only deployments may omit the config volume) and returned for logging.
func Initialize(path string, c *Config) (created bool, err error) {
	if _, statErr := os.Stat(path); statErr == nil {
		return false, nil
	}
	if err := Write(path, c); err != nil {
		return false, err
	}
	return true, nil
}

// Holder provides concurrency-safe access to the live configuration, which
// the admin settings endpoint may update at runtime.
type Holder struct {
	mu              sync.RWMutex
	c               Config
	startupProvider string
	startupHost     string
}

// NewHolder wraps c and records the provider/host the process starts with.
func NewHolder(c *Config) *Holder {
	return &Holder{c: *c, startupProvider: c.LLM.Provider, startupHost: c.LLM.Host}
}

// Get returns a snapshot of the current configuration.
func (h *Holder) Get() Config {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.c
}

// Update applies fn to a copy and publishes it if fn returns nil.
func (h *Holder) Update(fn func(*Config) error) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	next := h.c
	if err := fn(&next); err != nil {
		return err
	}
	h.c = next
	return nil
}

// RestartRequired reports whether the editable provider/host differs from the
// identity the running process initialised with.
func (h *Holder) RestartRequired() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.startupProvider != h.c.LLM.Provider || h.startupHost != h.c.LLM.Host
}
