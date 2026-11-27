package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"strings"
	"time"
)

type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, ",") }
func (l *listFlag) Set(v string) error { *l = append(*l, v); return nil }

// benchmarkPayload keeps only the options that were supplied.
func benchmarkPayload(models, receipts []string, category *string, maxConf, limit *int, timeout int, policy string, abs, rel *string, apply bool) map[string]any {
	p := map[string]any{"request_timeout_seconds": timeout, "council_policy": policy}
	if len(models) > 0 {
		p["model_ids"] = models
	}
	if len(receipts) > 0 {
		p["receipt_ids"] = receipts
	}
	if category != nil {
		p["category"] = *category
	}
	if maxConf != nil {
		p["max_source_confidence"] = *maxConf
	}
	if limit != nil {
		p["limit"] = *limit
	}
	if abs != nil {
		p["council_absolute_threshold"] = *abs
	}
	if rel != nil {
		p["council_relative_threshold"] = *rel
	}
	if apply {
		p["apply_council_flags"] = true
	}
	return p
}

func runBenchmark(args []string) error {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	apiURL := fs.String("api-url", envOr("VB_API_URL", "http://127.0.0.1:8080"), "Vision Bill API URL")
	username := fs.String("username", envOr("VB_USERNAME", ""), "admin username (or VB_USERNAME)")
	password := fs.String("password", envOr("VB_PASSWORD", ""), "admin password (or VB_PASSWORD)")
	var models, receipts listFlag
	fs.Var(&models, "model", "model id (repeatable)")
	fs.Var(&receipts, "receipt-id", "verified receipt id (repeatable)")
	category := fs.String("category", "", "only receipts of this category")
	maxConf := fs.Int("max-source-confidence", -1, "only receipts at or below this source confidence")
	limit := fs.Int("limit", 0, "maximum number of receipts")
	timeout := fs.Int("request-timeout-seconds", 300, "per-task timeout")
	policy := fs.String("council-policy", "all", "all | material | custom")
	abs := fs.String("council-absolute-threshold", "", "council absolute threshold")
	rel := fs.String("council-relative-threshold", "", "council relative threshold")
	apply := fs.Bool("apply-council-flags", false, "apply council flags")
	if err := fs.Parse(args); err != nil {
		return usageError("")
	}
	if (*username == "") != (*password == "") {
		return usageError("Provide both --username and --password (or VB_USERNAME and VB_PASSWORD).")
	}
	if *policy != "all" && *policy != "material" && *policy != "custom" {
		return usageError("--council-policy must be one of: all, material, custom")
	}
	opt := func(s string) *string {
		if s == "" {
			return nil
		}
		return &s
	}
	var maxConfP, limitP *int
	if *maxConf >= 0 {
		maxConfP = maxConf
	}
	if *limit > 0 {
		limitP = limit
	}
	payload := benchmarkPayload(models, receipts, opt(*category), maxConfP, limitP, *timeout, *policy, opt(*abs), opt(*rel), *apply)

	jar, _ := cookiejar.New(nil)
	client := &http.Client{Jar: jar, Timeout: 30 * time.Second}
	base := strings.TrimRight(*apiURL, "/")
	post := func(path string, body any) ([]byte, error) {
		raw, _ := json.Marshal(body)
		resp, err := client.Post(base+path, "application/json", bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		if resp.StatusCode/100 != 2 {
			return nil, fmt.Errorf("%s returned HTTP %d: %s", path, resp.StatusCode, strings.TrimSpace(string(data)))
		}
		return data, nil
	}
	if *username != "" {
		if _, err := post("/api/v1/auth/login", map[string]string{"username": *username, "password": *password}); err != nil {
			return fmt.Errorf("Unable to queue benchmark: %w", err)
		}
	}
	data, err := post("/api/v1/benchmarks", payload)
	if err != nil {
		return fmt.Errorf("Unable to queue benchmark: %w", err)
	}
	var run struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(data, &run); err != nil || run.ID == "" {
		return fmt.Errorf("unexpected response: %s", strings.TrimSpace(string(data)))
	}
	fmt.Println(run.ID)
	return nil
}
