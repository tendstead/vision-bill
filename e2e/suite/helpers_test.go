//go:build e2e

// Package suite is the black-box end-to-end suite. It talks HTTP to the running
// compose stack (docker-compose.e2e.yml: app, PostgreSQL and the LLM stub); run
// it with `make test-e2e`. The stub's mode is global state, so tests run
// sequentially and every test starts with the stub in "ok" mode.
package suite

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/cookiejar"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tendstead/vision-bill/internal/ids"
)

const (
	adminUser = "admin"
	adminPass = "admin-e2e-pass"
	testPass  = "e2e-pw-12345"
	// secretKey is AUTH__SECRET_KEY of the e2e app container.
	secretKey = "e2e-secret-do-not-use"
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

var (
	baseURL = envOr("VB_E2E_BASE_URL", "http://localhost:53625")
	stubURL = envOr("VB_E2E_STUB_URL", "http://localhost:9123")
	pgURL   = envOr("VB_E2E_DATABASE_URL", "postgres://vision_bill:vision_bill@localhost:5433/vision_bill")
)

// dir returns the e2e directory (the parent of this package).
func dir(parts ...string) string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(append([]string{filepath.Dir(filepath.Dir(file))}, parts...)...)
}

// ── HTTP client ──────────────────────────────────────────────────────────

// R is an HTTP response.
type R struct {
	t      *testing.T
	Status int
	Body   []byte
	Header http.Header
}

func (r R) String() string { return fmt.Sprintf("%d %s", r.Status, strings.TrimSpace(string(r.Body))) }

// M decodes the body as a JSON object.
func (r R) M() map[string]any {
	r.t.Helper()
	var m map[string]any
	if err := json.Unmarshal(r.Body, &m); err != nil {
		r.t.Fatalf("response is not a JSON object (%v): %s", err, r)
	}
	return m
}

// L decodes the body as a JSON array.
func (r R) L() []any {
	r.t.Helper()
	var l []any
	if err := json.Unmarshal(r.Body, &l); err != nil {
		r.t.Fatalf("response is not a JSON array (%v): %s", err, r)
	}
	return l
}

// Is fails the test unless the status is one of want.
func (r R) Is(want ...int) R {
	r.t.Helper()
	for _, w := range want {
		if r.Status == w {
			return r
		}
	}
	r.t.Fatalf("status %d, want one of %v: %s", r.Status, want, r)
	return r
}

// Client is an HTTP client with its own cookie jar.
type Client struct {
	t  *testing.T
	hc *http.Client
}

func newClient(t *testing.T) *Client {
	jar, _ := cookiejar.New(nil)
	return &Client{t: t, hc: &http.Client{Jar: jar, Timeout: 60 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
}

// Raw performs a request against an absolute path (no /api/v1 prefix).
func (c *Client) Raw(method, path string, body io.Reader, headers map[string]string) R {
	c.t.Helper()
	req, err := http.NewRequest(method, baseURL+path, body)
	if err != nil {
		c.t.Fatal(err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		c.t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	return R{c.t, resp.StatusCode, data, resp.Header}
}

// Do performs a JSON request against /api/v1.
func (c *Client) Do(method, path string, body any) R {
	c.t.Helper()
	var rd io.Reader
	headers := map[string]string{}
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			c.t.Fatal(err)
		}
		rd = bytes.NewReader(raw)
		headers["Content-Type"] = "application/json"
	}
	return c.Raw(method, "/api/v1"+path, rd, headers)
}

func (c *Client) Get(path string) R            { c.t.Helper(); return c.Do("GET", path, nil) }
func (c *Client) Post(path string, body any) R { c.t.Helper(); return c.Do("POST", path, body) }
func (c *Client) Put(path string, body any) R  { c.t.Helper(); return c.Do("PUT", path, body) }
func (c *Client) Delete(path string) R         { c.t.Helper(); return c.Do("DELETE", path, nil) }
func (c *Client) cookie(name string) string {
	u, _ := http.NewRequest("GET", baseURL, nil)
	for _, ck := range c.hc.Jar.Cookies(u.URL) {
		if ck.Name == name {
			return ck.Value
		}
	}
	return ""
}

// File is one multipart photo.
type File struct {
	Name, ContentType string
	Data              []byte
}

// Upload posts photos to /images (field "receipt"); query is e.g. "?bypass_review=true".
func (c *Client) Upload(query string, files ...File) R {
	c.t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, f := range files {
		h := textproto.MIMEHeader{}
		h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="receipt"; filename=%q`, f.Name))
		ct := f.ContentType
		if ct == "" {
			ct = "application/octet-stream"
		}
		h.Set("Content-Type", ct)
		part, err := mw.CreatePart(h)
		if err != nil {
			c.t.Fatal(err)
		}
		_, _ = part.Write(f.Data)
	}
	_ = mw.Close()
	return c.Raw("POST", "/api/v1/images"+query, &buf, map[string]string{"Content-Type": mw.FormDataContentType()})
}

// ── users and the stub ───────────────────────────────────────────────────

// newEnv resets the stub to "ok" and returns an anonymous client.
func newEnv(t *testing.T) *Client {
	t.Helper()
	stubMode(t, "ok")
	return newClient(t)
}

// register creates a fresh isolated user and returns its logged-in client and JSON.
func register(t *testing.T, prefix string) (*Client, map[string]any) {
	t.Helper()
	c := newClient(t)
	name := fmt.Sprintf("e2e-%s-%s", prefix, ids.UUID()[:8])
	r := c.Post("/auth/register", map[string]string{"username": name, "password": testPass}).Is(201)
	return c, r.M()
}

func admin(t *testing.T) *Client {
	t.Helper()
	c := newClient(t)
	c.Post("/auth/login", map[string]string{"username": adminUser, "password": adminPass}).Is(200)
	return c
}

func stubCall(t *testing.T, method, path string, body any) R {
	t.Helper()
	var rd io.Reader
	if body != nil {
		raw, _ := json.Marshal(body)
		rd = bytes.NewReader(raw)
	}
	req, _ := http.NewRequest(method, stubURL+path, rd)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("stub %s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	return R{t, resp.StatusCode, data, resp.Header}
}

func stubMode(t *testing.T, mode string) {
	t.Helper()
	stubCall(t, "POST", "/__mode", map[string]string{"mode": mode}).Is(200)
}

func stubRequests(t *testing.T) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, v := range stubCall(t, "GET", "/__requests", nil).Is(200).M()["requests"].([]any) {
		out = append(out, v.(map[string]any))
	}
	return out
}

// ── generic helpers ──────────────────────────────────────────────────────

func waitUntil(t *testing.T, timeout time.Duration, what string, fn func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		if fn() {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %v waiting for %s", timeout, what)
		}
		time.Sleep(300 * time.Millisecond)
	}
}

// waitImage polls an image until its status leaves pending/processing (or
// matches one of want), driving analysis cycles through /images/analyze when drive is set.
func waitImage(t *testing.T, c *Client, imageID string, drive bool, want ...string) map[string]any {
	t.Helper()
	var row map[string]any
	waitUntil(t, 60*time.Second, "image "+imageID+" to finish", func() bool {
		if drive {
			c.Post("/images/analyze", nil)
		}
		row = c.Get("/images/" + imageID).Is(200).M()
		st, _ := row["status"].(string)
		if len(want) > 0 {
			for _, w := range want {
				if st == w {
					return true
				}
			}
			return false
		}
		return st != "pending" && st != "processing"
	})
	return row
}

// str reads a nested string from decoded JSON ("a", "b" => m["a"]["b"]).
func str(v any, path ...string) string {
	for _, p := range path {
		m, ok := v.(map[string]any)
		if !ok {
			return ""
		}
		v = m[p]
	}
	s, _ := v.(string)
	return s
}

func mp(v any, path ...string) map[string]any {
	for _, p := range path {
		m, _ := v.(map[string]any)
		v = m[p]
	}
	m, _ := v.(map[string]any)
	return m
}

func list(v any, path ...string) []any {
	for _, p := range path {
		m, _ := v.(map[string]any)
		v = m[p]
	}
	l, _ := v.([]any)
	return l
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func escape(s string) string { return url.QueryEscape(s) }
