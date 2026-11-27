package httpapi

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tendstead/vision-bill/internal/store"
)

// verifiedReceiptWithPhoto creates a verified receipt whose photo is in permanent storage.
func (e *testEnv) verifiedReceiptWithPhoto(t *testing.T, owner actor, photo []byte) string {
	t.Helper()
	e.provider.set(false, modelReceipt)
	r := e.upload(t, owner, "?bypass_review=true", photo)
	if r.Code != 201 {
		t.Fatalf("setup upload: %d %s", r.Code, r.Body)
	}
	return r.json(t)["receipt_id"].(string)
}

func (e *testEnv) drain(t *testing.T) {
	t.Helper()
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		task, err := e.st.LeaseBenchmarkTask(ctx, time.Now())
		if err != nil {
			t.Fatal(err)
		}
		if task == nil {
			return
		}
		e.bench.Execute(ctx, e.st, *task)
	}
	t.Fatal("benchmark queue did not drain")
}

func TestBenchmarkRunLifecycle(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	rid := e.verifiedReceiptWithPhoto(t, user, pngData)

	if r := e.do("GET", "/api/v1/benchmarks", nil, user.cookie); r.Code != 403 {
		t.Fatalf("non-admin: %d", r.Code)
	}
	if r := e.do("POST", "/api/v1/benchmarks", `{}`, user.cookie); r.Code != 403 {
		t.Fatalf("non-admin create: %d", r.Code)
	}
	caps := e.do("GET", "/api/v1/benchmarks/capabilities", nil, admin.cookie).json(t)
	if caps["ocr_available"] != false || caps["thinking_control"] != true {
		t.Fatalf("caps %v", caps)
	}

	e.provider.set(false, modelReceipt)
	r := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:1"], "experiment": {"temperature": 0.2}}`, admin.cookie)
	if r.Code != 202 {
		t.Fatalf("create: %d %s", r.Code, r.Body)
	}
	run := r.json(t)
	runID := run["id"].(string)
	if r.Header().Get("Location") != "/api/v1/benchmarks/"+runID || run["status"] != "queued" ||
		run["receipt_ids"].([]any)[0] != rid || run["model_ids"].([]any)[0] != "vis:1" || run["scoring_version"] != "3" || run["prompt_version"] != "4" {
		t.Fatalf("run %v", run)
	}
	exp := run["experiment"].(map[string]any)
	if exp["inputs"] != nil || exp["tags"] != nil || exp["options"].(map[string]any)["temperature"] != 0.2 || exp["thinking_default"] != "server-controlled" {
		t.Fatalf("experiment %v", exp)
	}

	st := e.do("GET", "/api/v1/benchmarks/"+runID, nil, admin.cookie).json(t)
	if st["queued"] != float64(1) || st["terminal"] != float64(0) {
		t.Fatalf("status before: %v", st)
	}
	e.provider.set(false, modelReceipt)
	e.drain(t)
	st = e.do("GET", "/api/v1/benchmarks/"+runID, nil, admin.cookie).json(t)
	sum := st["summaries"].([]any)[0].(map[string]any)
	if st["run"].(map[string]any)["status"] != "completed" || st["run"].(map[string]any)["completed_at"] == nil ||
		st["terminal"] != float64(1) || sum["succeeded"] != float64(1) || sum["average_score"].(float64) < 0.99 || sum["average_attempts"] != float64(1) {
		t.Fatalf("status after: %v", st)
	}
	res := e.do("GET", "/api/v1/benchmarks/"+runID+"/results", nil, admin.cookie)
	items := res.json2(t)
	if len(items) != 1 || items[0]["success"] != true || items[0]["receipt_id"] != rid || items[0]["model_id"] != "vis:1" {
		t.Fatalf("results %s", res.Body)
	}
	scores := items[0]["scores"].(map[string]any)
	// "0" vs the stored "0.00" differ only in formatting: normalized is perfect, strict is not.
	if scores["normalized"] != 1.0 || scores["strict"].(float64) >= 1.0 || items[0]["legacy_scores"] == nil || items[0]["metadata"].(map[string]any)["used_pipeline"] != "direct_vision" {
		t.Fatalf("payload %v", items[0])
	}
	for _, d := range items[0]["differences"].([]any) {
		if d.(map[string]any)["equivalent"] != true {
			t.Fatalf("only formatting differences expected: %v", d)
		}
	}

	// Transient re-evaluation returns the diff and stores nothing.
	e.provider.set(false, strings.Replace(modelReceipt, `"total": "1.00"`, `"total": "1.05"`, 1))
	re := e.do("POST", "/api/v1/benchmarks/"+runID+"/reevaluate", `{"receipt_id": "`+rid+`", "model_id": "vis:1"}`, admin.cookie)
	if re.Code != 200 {
		t.Fatalf("reevaluate: %d %s", re.Code, re.Body)
	}
	rj := re.json(t)
	var totalDiff map[string]any
	for _, d := range rj["differences"].([]any) {
		if d.(map[string]any)["path"] == "total" {
			totalDiff = d.(map[string]any)
		}
	}
	if rj["scores"].(map[string]any)["normalized"].(float64) >= 1 || totalDiff == nil || totalDiff["equivalent"] != false || rj["expected"] == nil {
		t.Fatalf("reevaluate body %v", rj)
	}
	if got := e.do("GET", "/api/v1/benchmarks/"+runID+"/results", nil, admin.cookie).json2(t); len(got) != 1 {
		t.Fatal("re-evaluation must not persist results")
	}
	for body, want := range map[string]int{
		`{"receipt_id": "99999999-9999-9999-9999-999999999999", "model_id": "vis:1"}`: 404,
		`{"receipt_id": "` + rid + `", "model_id": "other"}`:                          404,
		`{"receipt_id": "` + rid + `"}`:                                               422,
		`{"receipt_id": "nope", "model_id": "x"}`:                                     422,
	} {
		if r := e.do("POST", "/api/v1/benchmarks/"+runID+"/reevaluate", body, admin.cookie); r.Code != want {
			t.Errorf("%s: %d %s", body, r.Code, r.Body)
		}
	}
	if r := e.do("GET", "/api/v1/benchmarks", nil, admin.cookie); r.Code != 200 || !strings.Contains(r.Body.String(), runID) {
		t.Fatalf("list: %d", r.Code)
	}
	if r := e.do("GET", "/api/v1/benchmarks?limit=1&offset=5", nil, admin.cookie); strings.TrimSpace(r.Body.String()) != "[]" {
		t.Fatalf("offset beyond end: %s", r.Body)
	}
	if r := e.do("GET", "/api/v1/benchmarks/99999999-9999-9999-9999-999999999999", nil, admin.cookie); r.Code != 404 {
		t.Fatalf("unknown run: %d", r.Code)
	}
	if r := e.do("GET", "/api/v1/benchmarks/99999999-9999-9999-9999-999999999999/results", nil, admin.cookie); r.Code != 404 {
		t.Fatalf("unknown results: %d", r.Code)
	}
	// A receipt referenced by a run cannot be deleted.
	if r := e.do("DELETE", "/api/v1/receipts/"+rid, nil, user.cookie); r.Code != 200 {
		// the finished task rows are gone, so deletion is allowed after completion
		t.Fatalf("delete after completion: %d %s", r.Code, r.Body)
	}
}

func TestBenchmarkValidationAndSelection(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	e.provider.set(false)
	if r := e.do("POST", "/api/v1/benchmarks", `{}`, admin.cookie); r.Code != 422 || r.json(t)["detail"] != "No verified receipts match the selection" {
		t.Fatalf("no receipts: %d %s", r.Code, r.Body)
	}
	rid := e.verifiedReceiptWithPhoto(t, user, pngData)
	// An unverified receipt is not benchmark data.
	unverified := e.seed(t, user, receiptBody, "unverified")
	e.provider.set(false, modelReceipt)
	for body, want := range map[string]string{
		`{"model_ids": ["nope"]}`:                 "Selected model(s) unavailable: nope",
		`{"receipt_ids": ["` + unverified + `"]}`: "No verified receipts match the selection",
		`{"experiment": {"pipeline": "ocr"}}`:     "OCR extension is not configured; choose normal upload or direct vision",
		`{"category": "fuel"}`:                    "No verified receipts match the selection",
		`{"max_source_confidence": 10}`:           "No verified receipts match the selection",
	} {
		if r := e.do("POST", "/api/v1/benchmarks", body, admin.cookie); r.Code != 422 || r.json(t)["detail"] != want {
			t.Errorf("%s: %d %s", body, r.Code, r.Body)
		}
	}
	for body, typ := range map[string]string{
		`{"experiment": {"pipeline": "x"}}`:  "enum",
		`{"experiment": {"temperature": 3}}`: "less_than_equal",
		`{"max_source_confidence": 101}`:     "less_than_equal",
		`{"limit": 0}`:                       "greater_than_equal",
		`{"request_timeout_seconds": 0}`:     "greater_than_equal",
		`{"council_policy": "x"}`:            "enum",
		`{"council_absolute_threshold": -1}`: "greater_than_equal",
		`{"receipt_ids": ["nope"]}`:          "uuid_parsing",
		`{"apply_council_flags": "yes"}`:     "bool_type",
	} {
		if r := e.do("POST", "/api/v1/benchmarks", body, admin.cookie); r.Code != 422 || !strings.Contains(r.Body.String(), typ) {
			t.Errorf("%s: %d %s", body, r.Code, r.Body)
		}
	}
	ok := e.do("POST", "/api/v1/benchmarks", `{"receipt_ids": ["`+rid+`", "`+unverified+`"], "council_policy": "material", "council_relative_threshold": 0.25, "experiment": {"think": false, "include_classification": false}}`, admin.cookie)
	if ok.Code != 202 || ok.json(t)["council_policy"] != "material" {
		t.Fatalf("%d %s", ok.Code, ok.Body)
	}
	if got := ok.json(t)["receipt_ids"].([]any); len(got) != 1 || got[0] != rid {
		t.Fatalf("only verified receipts are selected: %v", got)
	}
	// Unverified or deleted receipts are kept out of the run and the run keeps its frozen inputs.
	var n int
	_ = e.st.Pool.QueryRow(context.Background(), "SELECT count(*) FROM benchmark_tasks").Scan(&n)
	if n != 2 { // 2 models x 1 receipt
		t.Fatalf("tasks: %d", n)
	}
	if r := e.do("DELETE", "/api/v1/receipts/"+rid, nil, user.cookie); r.Code != 409 {
		t.Fatalf("receipt referenced by a queued benchmark must not be deleted: %d %s", r.Code, r.Body)
	}
}

func TestBenchmarkFailuresAndChangedInputs(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	rid := e.verifiedReceiptWithPhoto(t, user, pngData)
	e.provider.set(false, "garbage", "garbage", "garbage")
	r := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:1"]}`, admin.cookie)
	runID := r.json(t)["id"].(string)
	e.provider.set(false, "garbage", "garbage", "garbage")
	e.drain(t)
	items := e.do("GET", "/api/v1/benchmarks/"+runID+"/results", nil, admin.cookie).json2(t)
	if len(items) != 1 || items[0]["success"] != false || !strings.Contains(items[0]["error"].(string), "ValueError") {
		t.Fatalf("failure result %v", items)
	}
	if st := e.do("GET", "/api/v1/benchmarks/"+runID, nil, admin.cookie).json(t); st["run"].(map[string]any)["status"] != "completed" ||
		st["summaries"].([]any)[0].(map[string]any)["failed"] != float64(1) {
		t.Fatalf("%v", st)
	}

	// A changed photo invalidates replay of the frozen input.
	e.provider.set(false, modelReceipt)
	run2 := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:1"]}`, admin.cookie).json(t)["id"].(string)
	var path string
	_ = e.st.Pool.QueryRow(context.Background(), `SELECT i.image_path FROM images i JOIN receipts r ON r.image_id = i.id WHERE r.id = $1::uuid`, rid).Scan(&path)
	if err := writeFile(path, append(pngData, 1)); err != nil {
		t.Fatal(err)
	}
	e.drain(t)
	items = e.do("GET", "/api/v1/benchmarks/"+run2+"/results", nil, admin.cookie).json2(t)
	if len(items) != 1 || items[0]["success"] != false || items[0]["error"] != "Receipt image changed since run creation" {
		t.Fatalf("changed image: %v", items)
	}
	if r := e.do("POST", "/api/v1/benchmarks/"+run2+"/reevaluate", `{"receipt_id": "`+rid+`", "model_id": "vis:1"}`, admin.cookie); r.Code != 404 || r.json(t)["detail"] != "Receipt image changed since run creation" {
		t.Fatalf("reevaluate changed image: %d %s", r.Code, r.Body)
	}
}

func TestBenchmarkWaitsWhenModelDisappears(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	e.verifiedReceiptWithPhoto(t, user, pngData)
	e.provider.set(false, modelReceipt)
	runID := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:2"]}`, admin.cookie).json(t)["id"].(string)
	// The model is withdrawn: the task waits instead of failing.
	e.provider.mu.Lock()
	e.provider.models = e.provider.models[:1]
	e.provider.mu.Unlock()
	ctx := context.Background()
	task, _ := e.st.LeaseBenchmarkTask(ctx, time.Now())
	e.bench.Execute(ctx, e.st, *task)
	st := e.do("GET", "/api/v1/benchmarks/"+runID, nil, admin.cookie).json(t)
	if st["waiting_for_model"] != float64(1) || st["terminal"] != float64(0) {
		t.Fatalf("%v", st)
	}
	// Provider down => transient retry with backoff.
	e.provider.set(true)
	if next, _ := e.st.LeaseBenchmarkTask(ctx, time.Now()); next != nil {
		t.Fatal("task must not be runnable before its backoff elapses")
	}
}

func TestLegacyRunWithoutManifestStillRuns(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	rid := e.verifiedReceiptWithPhoto(t, user, pngData)
	run, err := e.st.CreateBenchmarkRun(context.Background(), store.NewBenchmarkRun{
		ModelIDs: []string{"vis:1"}, ReceiptIDs: []string{rid}, Fingerprint: "legacy", ModelDigests: map[string]*string{"vis:1": nil},
		TimeoutSeconds: 30, Policy: "all", PromptVersion: "3", ScoringVersion: "2",
	})
	if err != nil {
		t.Fatal(err)
	}
	if run.Experiment != nil {
		t.Fatal("legacy runs have no experiment")
	}
	e.provider.set(false, modelReceipt)
	e.drain(t)
	st := e.do("GET", "/api/v1/benchmarks/"+run.ID, nil, admin.cookie).json(t)
	sum := st["summaries"].([]any)[0].(map[string]any)
	if st["run"].(map[string]any)["status"] != "completed" || sum["succeeded"] != float64(1) || sum["average_score"].(float64) < 0.99 {
		t.Fatalf("%v", st)
	}
	if items := e.do("GET", "/api/v1/benchmarks/"+run.ID+"/results", nil, admin.cookie).json2(t); len(items) != 0 {
		t.Fatalf("legacy runs store no per-task payload: %v", items)
	}
	// Re-evaluation of a legacy run uses the live verified receipt.
	e.provider.set(false, strings.Replace(modelReceipt, `"merchant_name": "Shop"`, `"merchant_name": "Other"`, 1))
	re := e.do("POST", "/api/v1/benchmarks/"+run.ID+"/reevaluate", `{"receipt_id": "`+rid+`", "model_id": "vis:1"}`, admin.cookie)
	if re.Code != 200 {
		t.Fatalf("%d %s", re.Code, re.Body)
	}
	diff := re.json(t)["diff"].(map[string]any)
	if diff["merchant_name"] == nil || diff["total"] != nil {
		t.Fatalf("diff %v", diff)
	}
}
