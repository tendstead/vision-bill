package httpapi

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/fstest"
)

func get(h http.Handler, p string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, p, nil))
	return rec
}

func TestRouting(t *testing.T) {
	h := New(fstest.MapFS{
		"index.html":   {Data: []byte("<html>app</html>")},
		"_app/main.js": {Data: []byte("js")},
	})
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
	}
	for _, c := range cases {
		rec := get(h, c.path)
		if rec.Code != c.status || !contains(rec.Body.String(), c.body) {
			t.Errorf("%s: got %d %q", c.path, rec.Code, rec.Body.String())
		}
	}
}

func TestNoBuild(t *testing.T) {
	if rec := get(New(nil), "/"); rec.Code != 404 {
		t.Fatalf("got %d", rec.Code)
	}
}

func contains(s, sub string) bool { return len(s) >= len(sub) && (s == sub || indexOf(s, sub) >= 0) }

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
