package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/tendstead/vision-bill/internal/analysis"
	"github.com/tendstead/vision-bill/internal/benchmark"
	"github.com/tendstead/vision-bill/internal/config"
	"github.com/tendstead/vision-bill/internal/files"
	"github.com/tendstead/vision-bill/internal/store"
	"github.com/tendstead/vision-bill/internal/testdb"
)

const testSecret = "test-secret-key"

func testConfig(t *testing.T) (*config.Holder, config.Sources) {
	t.Helper()
	dir := t.TempDir()
	yml := filepath.Join(dir, "config.yaml")
	env := map[string]string{
		"LLM__HOST": "h", "LLM__API_KEY": "k", "LLM__MODEL_NAME": "m", "LLM__TEMPERATURE": "0.5",
		"PG__USER": "u", "PG__PASSWORD": "p", "AUTH__SECRET_KEY": testSecret,
	}
	src := config.Sources{
		Env: func(k string) string { return env[k] },
		EnvNames: func() []string {
			var n []string
			for k := range env {
				n = append(n, k)
			}
			return n
		},
		YAMLPath: yml,
	}
	c, err := config.Load(src)
	if err != nil {
		t.Fatal(err)
	}
	return config.NewHolder(c), src
}

type testEnv struct {
	h         http.Handler
	cfg       *config.Holder
	st        *store.Store
	provider  *fakeProvider
	scheduler *analysis.Scheduler
	bench     *benchmark.Service
	files     *files.Service
	dir       string
}

// newEnv builds a server. When withDB is true it uses a migrated throwaway DB.
func newEnv(t *testing.T, withDB bool) *testEnv {
	t.Helper()
	holder, src := testConfig(t)
	dir := t.TempDir()
	fileSvc, err := files.New(files.Options{TmpDir: filepath.Join(dir, "tmp"), SaveDir: filepath.Join(dir, "save"), ThumbMaxEdge: 512, ThumbQuality: 80})
	if err != nil {
		t.Fatal(err)
	}
	e := &testEnv{cfg: holder, provider: &fakeProvider{}, files: fileSvc, dir: dir}
	opts := Options{
		Config: holder, ConfigSources: src, Files: fileSvc, Provider: e.provider,
		Static: fstest.MapFS{"index.html": {Data: []byte("<html>app</html>")}, "_app/main.js": {Data: []byte("js")}},
	}
	if withDB {
		pool, _ := testdb.New(t)
		if err := store.Migrate(context.Background(), pool); err != nil {
			t.Fatal(err)
		}
		e.st = &store.Store{Pool: pool}
		opts.Store = func() *store.Store { return e.st }
	}
	svc := &analysis.Service{Store: func() *store.Store { return e.st }, Files: fileSvc, Provider: e.provider, Config: holder}
	opts.Analysis = svc
	e.scheduler = analysis.NewScheduler(svc)
	opts.Scheduler = e.scheduler
	e.bench = &benchmark.Service{Store: func() *store.Store { return e.st }, Provider: e.provider, Analysis: svc}
	opts.Benchmarks = e.bench
	e.h = New(opts)
	return e
}

type resp struct {
	*httptest.ResponseRecorder
}

func (r resp) json(t *testing.T) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(r.Body.Bytes(), &m); err != nil {
		t.Fatalf("body %q: %v", r.Body.String(), err)
	}
	return m
}

func (e *testEnv) do(method, path string, body any, cookies ...*http.Cookie) resp {
	var rd *bytes.Reader
	switch b := body.(type) {
	case nil:
		rd = bytes.NewReader(nil)
	case string:
		rd = bytes.NewReader([]byte(b))
	default:
		raw, _ := json.Marshal(b)
		rd = bytes.NewReader(raw)
	}
	req := httptest.NewRequest(method, path, rd)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for _, c := range cookies {
		req.AddCookie(c)
	}
	rec := httptest.NewRecorder()
	e.h.ServeHTTP(rec, req)
	return resp{rec}
}

func sessionCookie(t *testing.T, r resp) *http.Cookie {
	t.Helper()
	for _, c := range r.Result().Cookies() {
		if c.Name == "vb_session" && c.Value != "" {
			return c
		}
	}
	t.Fatalf("no session cookie in %v", r.Result().Header)
	return nil
}

func TestRouting(t *testing.T) {
	e := newEnv(t, false)
	cases := []struct {
		path   string
		status int
		body   string
	}{
		{"/healthz", 200, `{"status":"ok"}`},
		{"/", 200, "<html>app</html>"},
		{"/receipts/123", 200, "<html>app</html>"}, // deep link
		{"/_app/main.js", 200, "js"},
		{"/_app/missing.js", 404, `{"detail":"Not Found"}`},
		{"/api/v1/nothing", 404, `{"detail":"Not Found"}`},
		{"/readyz", 503, "database unavailable"},
		{"/receipts/../etc/passwd", 200, "<html>app</html>"}, // no redirect for unclean paths
	}
	for _, c := range cases {
		rec := e.do("GET", c.path, nil)
		if rec.Code != c.status || !strings.Contains(rec.Body.String(), c.body) {
			t.Errorf("%s: got %d %q", c.path, rec.Code, rec.Body.String())
		}
	}
}

func TestDegradedWithoutDatabase(t *testing.T) {
	e := newEnv(t, false)
	if r := e.do("GET", "/api/v1/system/ui-config", nil); r.Code != 200 || r.json(t)["registration_open"] != true {
		t.Fatalf("ui-config: %d %s", r.Code, r.Body)
	}
	if r := e.do("GET", "/api/v1/auth/me", nil); r.Code != 503 || r.json(t)["detail"] != "Authentication is unavailable" {
		t.Fatalf("me: %d %s", r.Code, r.Body)
	}
	if r := e.do("POST", "/api/v1/auth/login", map[string]string{"username": "a", "password": "b"}); r.Code != 503 {
		t.Fatalf("login: %d", r.Code)
	}
	// Request validation runs before the database check.
	if r := e.do("POST", "/api/v1/auth/login", map[string]string{"username": ""}); r.Code != 422 {
		t.Fatalf("validation first: %d", r.Code)
	}
}

func TestAuthFlow(t *testing.T) {
	e := newEnv(t, true)
	creds := map[string]string{"username": "alice", "password": "pässword"}

	reg := e.do("POST", "/api/v1/auth/register", creds)
	if reg.Code != 201 {
		t.Fatalf("register: %d %s", reg.Code, reg.Body)
	}
	u := reg.json(t)
	if u["username"] != "alice" || u["is_admin"] != false || u["can_see_all"] != false || len(u["id"].(string)) != 36 {
		t.Fatalf("user body %v", u)
	}
	cookie := sessionCookie(t, reg)
	if !cookie.HttpOnly || cookie.MaxAge != 1209600 || cookie.SameSite != http.SameSiteLaxMode || cookie.Path != "/" {
		t.Fatalf("cookie attrs %+v", cookie)
	}

	if r := e.do("POST", "/api/v1/auth/register", creds); r.Code != 409 || r.json(t)["detail"] != "Username is already taken" {
		t.Fatalf("duplicate: %d %s", r.Code, r.Body)
	}
	me := e.do("GET", "/api/v1/auth/me", nil, cookie)
	if me.Code != 200 || me.json(t)["id"] != u["id"] {
		t.Fatalf("me: %d %s", me.Code, me.Body)
	}
	if r := e.do("GET", "/api/v1/auth/me", nil); r.Code != 401 || r.json(t)["detail"] != "Not authenticated" {
		t.Fatalf("anonymous me: %d %s", r.Code, r.Body)
	}
	if r := e.do("GET", "/api/v1/auth/me", nil, &http.Cookie{Name: "vb_session", Value: "bad.token.value"}); r.Code != 401 {
		t.Fatalf("bad cookie: %d", r.Code)
	}

	if r := e.do("POST", "/api/v1/auth/login", creds); r.Code != 200 || sessionCookie(t, r) == nil {
		t.Fatalf("login: %d %s", r.Code, r.Body)
	}
	for _, bad := range []map[string]string{{"username": "alice", "password": "nope"}, {"username": "ghost", "password": "x"}} {
		if r := e.do("POST", "/api/v1/auth/login", bad); r.Code != 401 || r.json(t)["detail"] != "Invalid username or password" {
			t.Fatalf("bad login: %d %s", r.Code, r.Body)
		}
	}

	out := e.do("POST", "/api/v1/auth/logout", nil)
	if out.Code != 200 || out.json(t)["ok"] != true {
		t.Fatalf("logout: %d %s", out.Code, out.Body)
	}
	if got := out.Result().Header.Get("Set-Cookie"); !strings.Contains(got, "Max-Age=0") {
		t.Fatalf("logout cookie %q", got)
	}
}

func TestRegistrationValidationAndToggle(t *testing.T) {
	e := newEnv(t, true)
	r := e.do("POST", "/api/v1/auth/register", map[string]string{"username": "", "password": strings.Repeat("x", 201)})
	if r.Code != 422 {
		t.Fatalf("%d", r.Code)
	}
	detail := r.json(t)["detail"].([]any)
	if len(detail) != 2 || detail[0].(map[string]any)["type"] != "string_too_short" || detail[1].(map[string]any)["type"] != "string_too_long" {
		t.Fatalf("detail %v", detail)
	}
	if r := e.do("POST", "/api/v1/auth/register", map[string]string{"username": "x"}); r.Code != 422 ||
		r.json(t)["detail"].([]any)[0].(map[string]any)["type"] != "missing" {
		t.Fatalf("missing: %d %s", r.Code, r.Body)
	}
	if r := e.do("POST", "/api/v1/auth/register", "{not json"); r.Code != 422 {
		t.Fatalf("bad json: %d", r.Code)
	}
	_ = e.cfg.Update(func(c *config.Config) error { c.Auth.AllowRegistration = false; return nil })
	if r := e.do("POST", "/api/v1/auth/register", map[string]string{"username": "u", "password": "p"}); r.Code != 403 {
		t.Fatalf("disabled: %d", r.Code)
	}
}

func TestIssuedSessionCookieAccepted(t *testing.T) {
	// A session cookie issued before this deployment must keep working.
	e := newEnv(t, true)
	ctx := context.Background()
	id := "11111111-2222-3333-4444-555555555555"
	if _, err := e.st.Pool.Exec(ctx, `INSERT INTO users (id, username, hashed_password) VALUES ($1, 'py', 'x')`, id); err != nil {
		t.Fatal(err)
	}
	const pyToken = "11111111-2222-3333-4444-555555555555.2795789108.cbf9596157c76a680a4b3733d1a2acd97f8fbe8d017bbd926f35978004959968"
	r := e.do("GET", "/api/v1/auth/me", nil, &http.Cookie{Name: "vb_session", Value: pyToken})
	if r.Code != 200 || r.json(t)["username"] != "py" {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
}

func TestAdminSettings(t *testing.T) {
	e := newEnv(t, true)
	ctx := context.Background()
	hash := "x"
	var adminID, userID string
	e.st.Pool.QueryRow(ctx, `INSERT INTO users (username, hashed_password, is_admin) VALUES ('adm', $1, true) RETURNING id::text`, hash).Scan(&adminID)
	e.st.Pool.QueryRow(ctx, `INSERT INTO users (username, hashed_password) VALUES ('usr', $1) RETURNING id::text`, hash).Scan(&userID)
	tok := func(id string) *http.Cookie {
		return &http.Cookie{Name: "vb_session", Value: mintToken(id)}
	}
	if r := e.do("GET", "/api/v1/system/settings", nil, tok(userID)); r.Code != 403 {
		t.Fatalf("non-admin: %d", r.Code)
	}
	if r := e.do("GET", "/api/v1/system/settings", nil); r.Code != 401 {
		t.Fatalf("anon: %d", r.Code)
	}
	r := e.do("GET", "/api/v1/system/settings", nil, tok(adminID))
	if r.Code != 200 {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	v := r.json(t)
	if v["restart_required"] != false || v["sources"].(map[string]any)["llm.host"] != "environment" {
		t.Fatalf("view %v", v)
	}
	if _, leaked := v["llm"].(map[string]any)["api_key"]; leaked {
		t.Fatal("api key leaked")
	}
	body := map[string]any{"llm": map[string]any{"provider": "ollama", "host": "other", "model_name": "m", "temperature": 0.5}, "allow_registration": true}
	if r := e.do("PUT", "/api/v1/system/settings", body, tok(adminID)); r.Code != 409 ||
		!strings.Contains(r.json(t)["detail"].(string), "llm.host") {
		t.Fatalf("env-locked: %d %s", r.Code, r.Body)
	}
	bad := map[string]any{"llm": map[string]any{"provider": "x", "host": "", "model_name": "m", "temperature": 3}, "allow_registration": true}
	if r := e.do("PUT", "/api/v1/system/settings", bad, tok(adminID)); r.Code != 422 || len(r.json(t)["detail"].([]any)) != 3 {
		t.Fatalf("validation: %d %s", r.Code, r.Body)
	}
}

func mintToken(id string) string {
	return mustToken(id)
}

func TestMain(m *testing.M) { os.Exit(m.Run()) }

func (r resp) json2(t *testing.T) []map[string]any {
	t.Helper()
	var out []map[string]any
	if err := json.Unmarshal(r.Body.Bytes(), &out); err != nil {
		t.Fatalf("body %q: %v", r.Body.String(), err)
	}
	return out
}

func writeFile(path string, data []byte) error { return os.WriteFile(path, data, 0o644) }
