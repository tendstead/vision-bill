//go:build e2e

package suite

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/tendstead/vision-bill/internal/ids"
)

func adminVerifiedReceipt(t *testing.T, a *Client) (string, fixtureCase) {
	t.Helper()
	fc := firstCase(t)
	r := a.Upload("?bypass_review=true", fc.file(t)).Is(201)
	return str(r.M(), "receipt_id"), fc
}

func TestBenchmarkExperimentUsesFrozenVerifiedData(t *testing.T) {
	newEnv(t)
	a := admin(t)
	rid, fc := adminVerifiedReceipt(t, a)
	run := a.Post("/benchmarks", map[string]any{"model_ids": []string{"e2e-vision"}, "receipt_ids": []string{rid},
		"experiment": map[string]any{"temperature": 0, "include_classification": false}}).Is(202).M()
	if run["scoring_version"] != "3" || mp(run, "experiment", "options")["temperature"] != float64(0) {
		t.Fatalf("%v", run)
	}
	runID := str(run, "id")
	waitUntil(t, 30*time.Second, "the benchmark task", func() bool {
		return a.Get("/benchmarks/" + runID).Is(200).M()["terminal"] == float64(1)
	})
	results := a.Get("/benchmarks/" + runID + "/results").Is(200).L()
	if len(results) != 1 {
		t.Fatalf("%d results", len(results))
	}
	res := results[0].(map[string]any)
	if res["success"] != true || mp(res, "scores")["normalized"] != float64(1) || mp(res, "scores")["classification"] != nil {
		t.Fatalf("%v", res)
	}
	meta := mp(res, "metadata")
	if meta["image_count"] != float64(1) || meta["used_pipeline"] != "direct_vision" || len(list(meta, "calls")) == 0 ||
		str(list(meta, "calls")[0], "prompt_sha256") == "" || str(res, "expected", "merchant_name") != fc.Expected.MerchantName {
		t.Fatalf("%v", meta)
	}
	// Editing the source receipt later must not replace the run's target.
	var edited map[string]any
	if err := json.Unmarshal(fc.Raw, &edited); err != nil {
		t.Fatal(err)
	}
	edited["merchant_name"] = "Edited after benchmark creation"
	a.Put("/receipts/"+rid, edited).Is(200)
	re := a.Post("/benchmarks/"+runID+"/reevaluate", map[string]string{"receipt_id": rid, "model_id": "e2e-vision"}).Is(200).M()
	if str(re, "expected", "merchant_name") != fc.Expected.MerchantName {
		t.Fatalf("frozen expectation replaced: %v", re["expected"])
	}
}

func TestBenchmarkValidationCapabilitiesAndAccess(t *testing.T) {
	newEnv(t)
	a := admin(t)
	caps := a.Get("/benchmarks/capabilities").Is(200).M()
	if caps["ocr_available"] != false || caps["thinking_control"] != false || len(caps) != 2 {
		t.Fatalf("%v", caps)
	}
	rid, _ := adminVerifiedReceipt(t, a)
	for _, exp := range []map[string]any{{"pipeline": "ocr"}, {"think": false}} {
		a.Post("/benchmarks", map[string]any{"model_ids": []string{"e2e-vision"}, "receipt_ids": []string{rid}, "experiment": exp}).Is(422)
	}
	newClient(t).Get("/benchmarks/capabilities").Is(401)
	user, _ := register(t, "bench")
	user.Get("/benchmarks").Is(403)
}

func TestBenchmarkReevaluateIsTransient(t *testing.T) {
	newEnv(t)
	a := admin(t)
	up := a.Upload("", File{Name: "admin.png", ContentType: "image/png", Data: genericPNG(t)}).Is(201).M()
	rid := str(up, "receipt_id")
	a.Post("/receipts/"+rid+"/verify", nil).Is(200)
	run := a.Post("/benchmarks", map[string]any{"model_ids": []string{"e2e-vision"}, "receipt_ids": []string{rid}}).Is(202).M()
	runID := str(run, "id")
	body := a.Post("/benchmarks/"+runID+"/reevaluate", map[string]string{"receipt_id": rid, "model_id": "e2e-vision"}).Is(200).M()
	if body["scores"] == nil || str(body, "receipt_id") != rid {
		t.Fatalf("%v", body)
	}
	a.Post("/benchmarks/"+ids.UUID()+"/reevaluate", map[string]string{"receipt_id": rid, "model_id": "e2e-vision"}).Is(404)
}

func TestBenchmarkRerunAndCompare(t *testing.T) {
	newEnv(t)
	a := admin(t)
	rid, fc := adminVerifiedReceipt(t, a)
	first := a.Post("/benchmarks", map[string]any{"model_ids": []string{"e2e-vision"}, "receipt_ids": []string{rid},
		"label": "baseline", "experiment": map[string]any{"temperature": 0}}).Is(202).M()
	firstID := str(first, "id")
	finished := func(id string) func() bool {
		return func() bool { return a.Get("/benchmarks/" + id).Is(200).M()["terminal"] == float64(1) }
	}
	waitUntil(t, 30*time.Second, "the first benchmark run", finished(firstID))

	// The receipt is edited afterwards: the rerun keeps the frozen expected values and reports it.
	var edited map[string]any
	if err := json.Unmarshal(fc.Raw, &edited); err != nil {
		t.Fatal(err)
	}
	edited["merchant_name"] = "Edited after benchmark creation"
	a.Put("/receipts/"+rid, edited).Is(200)

	second := a.Post("/benchmarks/"+firstID+"/rerun", map[string]any{"label": "warmer", "experiment": map[string]any{"temperature": 0.1}}).Is(202).M()
	secondID := str(second, "id")
	if second["parent_run_id"] != firstID || second["label"] != "warmer" || mp(second, "experiment", "options")["temperature"] != 0.1 ||
		len(list(second, "changes")) != 1 || str(list(second, "changes")[0], "change") != "expected_changed" {
		t.Fatalf("%v", second)
	}
	waitUntil(t, 30*time.Second, "the rerun", finished(secondID))

	cmp := a.Get("/benchmarks/compare?runs=" + firstID + "," + secondID).Is(200).M()
	if cmp["baseline_run_id"] != firstID || mp(cmp, "comparable")["comparable"] == nil || len(list(cmp, "receipts")) != 1 {
		t.Fatalf("%v", cmp)
	}
	cells := list(list(cmp, "receipts")[0], "cells")
	if len(cells) != 2 || str(cells[1], "run_id") != secondID || mp(cells[1])["score"] != float64(1) || mp(cells[1])["score_delta"] != float64(0) {
		t.Fatalf("%v", cells)
	}
	a.Get("/benchmarks/compare").Is(422)
	a.Get("/benchmarks/compare?runs=" + firstID).Is(200)
	if c := a.Get("/benchmarks/" + secondID + "/candidates").Is(200).L(); len(c) != 1 || c[0].(map[string]any)["selectable"] != true {
		t.Fatalf("%v", c)
	}
	a.Post("/benchmarks/"+ids.UUID()+"/rerun", map[string]any{}).Is(404)
}

func TestBenchmarkSweepAndProviderOptions(t *testing.T) {
	newEnv(t)
	a := admin(t)
	rid, _ := adminVerifiedReceipt(t, a)
	base := map[string]any{"model_ids": []string{"e2e-vision"}, "receipt_ids": []string{rid}}
	with := func(extra map[string]any) map[string]any {
		b := map[string]any{}
		for k, v := range base {
			b[k] = v
		}
		for k, v := range extra {
			b[k] = v
		}
		return b
	}
	// The stub speaks the OpenAI protocol: only seed and top_p are supported there.
	opts := a.Get("/benchmarks/provider-options").Is(200).M()
	if str(opts, "provider") != "OpenAIProvider" || len(list(opts, "options")) != 2 {
		t.Fatalf("%v", opts)
	}
	a.Post("/benchmarks", with(map[string]any{"experiment": map[string]any{"provider_options": map[string]any{"num_ctx": 4096}}})).Is(422)

	sweep := a.Post("/benchmarks", with(map[string]any{"variants": []map[string]any{
		{"label": "seed 1", "experiment": map[string]any{"provider_options": map[string]any{"seed": 1}}},
		{"label": "seed 2", "experiment": map[string]any{"provider_options": map[string]any{"seed": 2, "top_p": 0.9}}},
	}})).Is(202).M()
	cid := str(sweep, "comparison_id")
	runs := list(sweep, "runs")
	if len(runs) != 2 || str(runs[1], "comparison_arm") != "seed 2" {
		t.Fatalf("%v", sweep)
	}
	var runIDs []string
	for _, r := range runs {
		runIDs = append(runIDs, str(r, "id"))
	}
	for _, id := range runIDs {
		id := id
		waitUntil(t, 30*time.Second, "a sweep run", func() bool {
			return a.Get("/benchmarks/" + id).Is(200).M()["terminal"] == float64(1)
		})
	}
	if got := a.Get("/benchmarks?comparison_id=" + cid).Is(200).L(); len(got) != 2 {
		t.Fatalf("%d runs in the sweep", len(got))
	}
	cmp := a.Get("/benchmarks/compare?runs=" + runIDs[0] + "," + runIDs[1]).Is(200).M()
	varied := map[string]bool{}
	for _, v := range list(cmp, "varied") {
		varied[str(v, "field")] = true
	}
	if mp(cmp, "comparable")["comparable"] != true || !varied["provider_options"] || !varied["label"] {
		t.Fatalf("%v", cmp)
	}
}
