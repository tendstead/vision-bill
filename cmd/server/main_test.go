package main

import "testing"

func TestHealthURL(t *testing.T) {
	cases := map[string]string{
		":8080":          "http://127.0.0.1:8080/healthz",
		"0.0.0.0:9000":   "http://127.0.0.1:9000/healthz",
		"[::]:8080":      "http://127.0.0.1:8080/healthz",
		"[::1]:8080":     "http://[::1]:8080/healthz",
		"[fd00::5]:8081": "http://[fd00::5]:8081/healthz",
		"10.0.0.2:8080":  "http://10.0.0.2:8080/healthz",
		"localhost:80":   "http://localhost:80/healthz",
	}
	for addr, want := range cases {
		got, err := healthURL(addr)
		if err != nil || got != want {
			t.Errorf("%q: got %q, %v; want %q", addr, got, err, want)
		}
	}
	if _, err := healthURL("8080"); err == nil {
		t.Error("address without port must be rejected")
	}
}
