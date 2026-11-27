package httpapi

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"testing"
)

// Every operation in the API contract must be routed. Unauthenticated calls
// never reach a handler's business logic, so a registered route answers with
// anything except the router's own 404/405.
func TestContractRoutesAreRegistered(t *testing.T) {
	var spec struct {
		Paths map[string]map[string]json.RawMessage `json:"paths"`
	}
	if err := json.Unmarshal(openAPISpec, &spec); err != nil {
		t.Fatal(err)
	}
	e := newEnv(t, false)
	param := regexp.MustCompile(`\{[^}]+\}`)
	ops := 0
	for path, methods := range spec.Paths {
		url := param.ReplaceAllString(path, "11111111-2222-3333-4444-555555555555")
		for method := range methods {
			m := strings.ToUpper(method)
			if m != "GET" && m != "POST" && m != "PUT" && m != "PATCH" && m != "DELETE" {
				continue
			}
			ops++
			r := e.do(m, url, nil)
			body := r.Body.String()
			if r.Code == http.StatusMethodNotAllowed || (r.Code == http.StatusNotFound && strings.Contains(body, `"Not Found"`)) {
				t.Errorf("%s %s is in the contract but not routed (%d %s)", m, path, r.Code, strings.TrimSpace(body))
			}
		}
	}
	if ops < 40 {
		t.Fatalf("only %d operations checked", ops)
	}
}

func TestDocsEndpoints(t *testing.T) {
	e := newEnv(t, false)
	if r := e.do("GET", "/openapi.json", nil); r.Code != 200 || r.Header().Get("Content-Type") != "application/json" || !strings.Contains(r.Body.String(), `"/api/v1/receipts"`) {
		t.Fatalf("openapi: %d", r.Code)
	}
	for _, p := range []string{"/docs", "/redoc"} {
		if r := e.do("GET", p, nil); r.Code != 200 || !strings.Contains(r.Body.String(), "/openapi.json") {
			t.Errorf("%s: %d", p, r.Code)
		}
	}
}
