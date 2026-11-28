package httpapi

import (
	"context"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tendstead/vision-bill/internal/llm"
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

// Tasks run model by model, so the inference server never swaps models between
// receipts of the same run.
func TestBenchmarkTasksRunModelByModel(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	for i := 0; i < 3; i++ {
		e.verifiedReceiptWithPhoto(t, user, append(append([]byte{}, pngData...), byte(i)))
	}
	e.provider.set(false, modelReceipt)
	// Requested order (vis:2 before vis:1) is kept, not sorted alphabetically.
	r := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:2", "vis:1"]}`, admin.cookie)
	if r.Code != 202 {
		t.Fatalf("create: %d %s", r.Code, r.Body)
	}
	ctx := context.Background()
	var order []string
	for {
		task, err := e.st.LeaseBenchmarkTask(ctx, time.Now())
		if err != nil {
			t.Fatal(err)
		}
		if task == nil {
			break
		}
		order = append(order, task.ModelID)
		e.bench.Execute(ctx, e.st, *task)
	}
	if got := strings.Join(order, ","); got != "vis:2,vis:2,vis:2,vis:1,vis:1,vis:1" {
		t.Fatalf("model order %s", got)
	}
}

// setDigests re-publishes the fake models with a digest (set resets them).
func (e *testEnv) setDigests(digest string) {
	e.provider.mu.Lock()
	defer e.provider.mu.Unlock()
	d := digest
	e.provider.models = []llm.ModelInfo{{ID: "vis:1", Digest: &d}, {ID: "vis:2", Digest: &d}}
}

func TestBenchmarkRerunAndCompare(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	r1 := e.verifiedReceiptWithPhoto(t, user, pngData)
	r2 := e.verifiedReceiptWithPhoto(t, user, append(append([]byte{}, pngData...), 9))
	ctx := context.Background()

	e.provider.set(false, modelReceipt)
	e.setDigests("sha-a")
	a := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:1"], "label": "baseline", "experiment": {"temperature": 0.2}}`, admin.cookie).json(t)
	runA := a["id"].(string)
	if a["label"] != "baseline" || a["parent_run_id"] != nil || a["comparison_id"] != nil {
		t.Fatalf("run %v", a)
	}
	e.drain(t)

	// Rerun with a changed setting: same frozen receipts, parent recorded.
	rr := e.do("POST", "/api/v1/benchmarks/"+runA+"/rerun", `{"label": "warm", "experiment": {"temperature": 0.7}}`, admin.cookie)
	if rr.Code != 202 {
		t.Fatalf("rerun: %d %s", rr.Code, rr.Body)
	}
	c := rr.json(t)
	runC := c["id"].(string)
	opts := c["experiment"].(map[string]any)["options"].(map[string]any)
	if rr.Header().Get("Location") != "/api/v1/benchmarks/"+runC || c["parent_run_id"] != runA || c["label"] != "warm" ||
		opts["temperature"] != 0.7 || opts["include_classification"] != true || len(c["receipt_ids"].([]any)) != 2 ||
		c["dataset_fingerprint"] != a["dataset_fingerprint"] || len(c["changes"].([]any)) != 0 || c["model_ids"].([]any)[0] != "vis:1" {
		t.Fatalf("rerun body %v", c)
	}
	e.drain(t)

	cmp := e.do("GET", "/api/v1/benchmarks/compare?runs="+runA+","+runC, nil, admin.cookie)
	if cmp.Code != 200 {
		t.Fatalf("compare: %d %s", cmp.Code, cmp.Body)
	}
	cj := cmp.json(t)
	comparable := cj["comparable"].(map[string]any)
	varied := map[string]bool{}
	for _, v := range cj["varied"].([]any) {
		varied[v.(map[string]any)["field"].(string)] = true
	}
	runs := cj["runs"].([]any)
	if cj["baseline_run_id"] != runA || comparable["comparable"] != true || len(comparable["reasons"].([]any)) != 0 ||
		!varied["temperature"] || !varied["label"] || varied["think"] || len(runs) != 2 || len(cj["receipts"].([]any)) != 2 {
		t.Fatalf("compare body %s", cmp.Body)
	}
	if runs[0].(map[string]any)["baseline_run_id"] != nil || runs[1].(map[string]any)["baseline_run_id"] != runA ||
		len(runs[1].(map[string]any)["deltas"].([]any)) != 1 {
		t.Fatalf("baselines %v", runs)
	}
	for _, rc := range cj["receipts"].([]any) {
		cells := rc.(map[string]any)["cells"].([]any)
		if len(cells) != 2 {
			t.Fatalf("cells %v", cells)
		}
		cell := cells[1].(map[string]any)
		if cell["run_id"] != runC || cell["success"] != true || cell["score"] != 1.0 || cell["score_delta"] != 0.0 ||
			cell["model_digest"] != "sha-a" || cell["latency_ms"] == nil || cell["attempts"] != 1.0 {
			t.Fatalf("cell %v", cell)
		}
	}

	// The model is updated, one receipt is deleted and another edited since.
	e.setDigests("sha-b")
	if r := e.do("DELETE", "/api/v1/receipts/"+r2, nil, user.cookie); r.Code != 200 {
		t.Fatalf("delete: %d %s", r.Code, r.Body)
	}
	if _, err := e.st.Pool.Exec(ctx, `UPDATE receipts SET merchant_name = 'Edited' WHERE id = $1::uuid`, r1); err != nil {
		t.Fatal(err)
	}
	rd := e.do("POST", "/api/v1/benchmarks/"+runA+"/rerun", `{}`, admin.cookie)
	if rd.Code != 202 {
		t.Fatalf("rerun after changes: %d %s", rd.Code, rd.Body)
	}
	d := rd.json(t)
	runD := d["id"].(string)
	got := map[string]string{}
	for _, ch := range d["changes"].([]any) {
		m := ch.(map[string]any)
		got[m["receipt_id"].(string)] = m["change"].(string)
	}
	if len(got) != 2 || got[r1] != "expected_changed" || got[r2] != "deleted" || len(d["receipt_ids"].([]any)) != 1 ||
		d["dataset_fingerprint"] == a["dataset_fingerprint"] || d["experiment"].(map[string]any)["options"].(map[string]any)["temperature"] != 0.2 {
		t.Fatalf("changes %v", d)
	}
	e.drain(t)
	// The frozen expected value is kept, so the edit does not change the score.
	items := e.do("GET", "/api/v1/benchmarks/"+runD+"/results", nil, admin.cookie).json2(t)
	if len(items) != 1 || items[0]["success"] != true || items[0]["scores"].(map[string]any)["normalized"] != 1.0 {
		t.Fatalf("rerun results %v", items)
	}

	cmp = e.do("GET", "/api/v1/benchmarks/compare?runs="+runA+","+runD, nil, admin.cookie)
	cj = cmp.json(t)
	codes := map[string]bool{}
	for _, r := range cj["comparable"].(map[string]any)["reasons"].([]any) {
		m := r.(map[string]any)
		codes[m["code"].(string)] = true
		if m["code"] == "model_digest_changed" && (m["model_id"] != "vis:1" || m["run_ids"].([]any)[0] != runD) {
			t.Fatalf("reason %v", m)
		}
	}
	if cmp.Code != 200 || cj["comparable"].(map[string]any)["comparable"] != false || !codes["model_digest_changed"] ||
		!codes["dataset_changed"] || len(codes) != 2 {
		t.Fatalf("not comparable: %s", cmp.Body)
	}

	// An unfinished run is reported as partial.
	runE := e.do("POST", "/api/v1/benchmarks/"+runD+"/rerun", `{"model_ids": ["vis:2"]}`, admin.cookie).json(t)["id"].(string)
	cj = e.do("GET", "/api/v1/benchmarks/compare?runs="+runD+","+runE, nil, admin.cookie).json(t)
	if cj["comparable"].(map[string]any)["comparable"] != false || cj["comparable"].(map[string]any)["reasons"].([]any)[0].(map[string]any)["code"] != "run_incomplete" {
		t.Fatalf("incomplete: %v", cj["comparable"])
	}
}

func TestBenchmarkRerunAndCompareValidation(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	rid := e.verifiedReceiptWithPhoto(t, user, pngData)
	e.provider.set(false, modelReceipt)
	run := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:1"]}`, admin.cookie).json(t)["id"].(string)
	const unknown = "99999999-9999-9999-9999-999999999999"

	if r := e.do("POST", "/api/v1/benchmarks/"+run+"/rerun", `{}`, user.cookie); r.Code != 403 {
		t.Fatalf("non-admin rerun: %d", r.Code)
	}
	if r := e.do("GET", "/api/v1/benchmarks/compare?runs="+run+","+run, nil, user.cookie); r.Code != 403 {
		t.Fatalf("non-admin compare: %d", r.Code)
	}
	for url, want := range map[string]int{
		"/api/v1/benchmarks/compare":                             422,
		"/api/v1/benchmarks/compare?runs=" + run:                 422,
		"/api/v1/benchmarks/compare?runs=" + run + "," + run:     422,
		"/api/v1/benchmarks/compare?runs=" + run + ",nope":       422,
		"/api/v1/benchmarks/compare?runs=" + run + "," + unknown: 404,
	} {
		if r := e.do("GET", url, nil, admin.cookie); r.Code != want {
			t.Errorf("%s: %d %s", url, r.Code, r.Body)
		}
	}
	for body, want := range map[string]int{
		`{"model_ids": ["nope"]}`:             422,
		`{"label": ""}`:                       422,
		`{"label": 5}`:                        422,
		`{"experiment": {"temperature": 5}}`:  422,
		`{"experiment": {"pipeline": "ocr"}}`: 422,
		`{"model_ids": "vis:1"}`:              422,
	} {
		if r := e.do("POST", "/api/v1/benchmarks/"+run+"/rerun", body, admin.cookie); r.Code != want {
			t.Errorf("%s: %d %s", body, r.Code, r.Body)
		}
	}
	if r := e.do("POST", "/api/v1/benchmarks/"+unknown+"/rerun", `{}`, admin.cookie); r.Code != 404 {
		t.Fatalf("unknown parent: %d", r.Code)
	}

	// Runs without frozen inputs cannot be rerun.
	legacy, err := e.st.CreateBenchmarkRun(context.Background(), store.NewBenchmarkRun{
		ModelIDs: []string{"vis:1"}, ReceiptIDs: []string{rid}, Fingerprint: "legacy", ModelDigests: map[string]*string{"vis:1": nil},
		TimeoutSeconds: 30, Policy: "all", PromptVersion: "3", ScoringVersion: "2",
	})
	if err != nil {
		t.Fatal(err)
	}
	if r := e.do("POST", "/api/v1/benchmarks/"+legacy.ID+"/rerun", `{}`, admin.cookie); r.Code != 422 ||
		!strings.Contains(r.Body.String(), "cannot be rerun") {
		t.Fatalf("legacy rerun: %d %s", r.Code, r.Body)
	}
	cj := e.do("GET", "/api/v1/benchmarks/compare?runs="+run+","+legacy.ID, nil, admin.cookie).json(t)
	codes := map[string]bool{}
	for _, r := range cj["comparable"].(map[string]any)["reasons"].([]any) {
		codes[r.(map[string]any)["code"].(string)] = true
	}
	if !codes["legacy_run"] || !codes["dataset_changed"] || !codes["scoring_changed"] {
		t.Fatalf("legacy compare: %v", cj["comparable"])
	}
}

func TestBenchmarkProviderOptions(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	e.verifiedReceiptWithPhoto(t, user, pngData)
	e.provider.set(false, modelReceipt)

	info := e.do("GET", "/api/v1/benchmarks/provider-options", nil, admin.cookie).json(t)
	names := map[string]bool{}
	for _, o := range info["options"].([]any) {
		names[o.(map[string]any)["name"].(string)] = true
	}
	if info["provider"] != "OllamaProvider" || !names["num_gpu"] || !names["num_ctx"] || !names["repeat_penalty"] || names["temperature"] {
		t.Fatalf("whitelist %v", info)
	}
	if r := e.do("GET", "/api/v1/benchmarks/provider-options", nil, user.cookie); r.Code != 403 {
		t.Fatalf("non-admin: %d", r.Code)
	}

	for body, want := range map[string]string{
		`{"experiment": {"provider_options": {"temperature": 1}}}`: "not supported by this provider: temperature",
		`{"experiment": {"provider_options": {"num_ctx": 0}}}`:     "must be at least 1",
		`{"experiment": {"provider_options": {"top_p": 2}}}`:       "must be at most 1",
		`{"experiment": {"provider_options": {"num_ctx": 1.5}}}`:   "must be an integer",
		`{"experiment": {"provider_options": {"seed": "x"}}}`:      "must be a number",
		`{"experiment": {"provider_options": [1]}}`:                "dict_type",
	} {
		if r := e.do("POST", "/api/v1/benchmarks", body, admin.cookie); r.Code != 422 || !strings.Contains(r.Body.String(), want) {
			t.Errorf("%s: %d %s", body, r.Code, r.Body)
		}
	}
	var n int
	_ = e.st.Pool.QueryRow(context.Background(), "SELECT count(*) FROM benchmark_runs").Scan(&n)
	if n != 0 {
		t.Fatalf("rejected requests must not create runs: %d", n)
	}

	r := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:1"], "experiment": {"provider_options": {"num_ctx": 4096, "num_gpu": 0, "top_p": 0.5}}}`, admin.cookie)
	if r.Code != 202 {
		t.Fatalf("create: %d %s", r.Code, r.Body)
	}
	run := r.json(t)
	po := run["experiment"].(map[string]any)["options"].(map[string]any)["provider_options"].(map[string]any)
	if po["num_ctx"] != 4096.0 || po["num_gpu"] != 0.0 || po["top_p"] != 0.5 {
		t.Fatalf("manifest options %v", po)
	}
	e.drain(t)
	e.provider.mu.Lock()
	sent := e.provider.chatOpts[len(e.provider.chatOpts)-1]
	e.provider.mu.Unlock()
	if sent["num_ctx"] != int64(4096) && sent["num_ctx"] != float64(4096) || sent["num_gpu"] == nil {
		t.Fatalf("options reached the provider as %v", sent)
	}

	// A rerun can replace or clear them; compare shows them as a varied setting.
	runID := run["id"].(string)
	rr := e.do("POST", "/api/v1/benchmarks/"+runID+"/rerun", `{"provider_options": {"num_ctx": 8192}}`, admin.cookie)
	if rr.Code != 202 {
		t.Fatalf("rerun: %d %s", rr.Code, rr.Body)
	}
	rj := rr.json(t)
	po = rj["experiment"].(map[string]any)["options"].(map[string]any)["provider_options"].(map[string]any)
	if len(po) != 1 || po["num_ctx"] != 8192.0 {
		t.Fatalf("replaced options %v", po)
	}
	e.drain(t)
	cj := e.do("GET", "/api/v1/benchmarks/compare?runs="+runID+","+rj["id"].(string), nil, admin.cookie).json(t)
	found := false
	for _, v := range cj["varied"].([]any) {
		found = found || v.(map[string]any)["field"] == "provider_options"
	}
	if !found || cj["comparable"].(map[string]any)["comparable"] != true {
		t.Fatalf("provider options are varied, not incomparable: %v", cj)
	}
	cleared := e.do("POST", "/api/v1/benchmarks/"+runID+"/rerun", `{"provider_options": null}`, admin.cookie).json(t)
	if _, has := cleared["experiment"].(map[string]any)["options"].(map[string]any)["provider_options"]; has {
		t.Fatalf("null clears the options: %v", cleared["experiment"])
	}
	if r := e.do("POST", "/api/v1/benchmarks/"+runID+"/rerun", `{"provider_options": {"bogus": 1}}`, admin.cookie); r.Code != 422 {
		t.Fatalf("rerun whitelist: %d", r.Code)
	}
}

func TestBenchmarkSweepAndLeaseOrder(t *testing.T) {
	e := newEnv(t, true)
	admin, user := e.user(t, "admin", true), e.user(t, "user", false)
	e.verifiedReceiptWithPhoto(t, user, pngData)
	e.verifiedReceiptWithPhoto(t, user, append(append([]byte{}, pngData...), 9))
	e.provider.set(false, modelReceipt)
	ctx := context.Background()

	// A plain run queued first is finished before the sweep starts.
	plain := e.do("POST", "/api/v1/benchmarks", `{"model_ids": ["vis:1"]}`, admin.cookie).json(t)["id"].(string)

	body := `{"model_ids": ["vis:2", "vis:1"], "label": "gpu box", "experiment": {"think": false}, "variants": [
		{"label": "cold", "experiment": {"temperature": 0}},
		{"experiment": {"temperature": 0.1, "provider_options": {"num_ctx": 2048}}},
		{"label": "hot", "experiment": {"temperature": 0.7}}]}`
	r := e.do("POST", "/api/v1/benchmarks", body, admin.cookie)
	if r.Code != 202 {
		t.Fatalf("sweep: %d %s", r.Code, r.Body)
	}
	sw := r.json(t)
	cid := sw["comparison_id"].(string)
	runs := sw["runs"].([]any)
	if len(runs) != 3 || r.Header().Get("Location") != "/api/v1/benchmarks?comparison_id="+cid {
		t.Fatalf("sweep body %s", r.Body)
	}
	wantArm := []string{"cold", "variant 2", "hot"}
	wantLabel := []any{"cold", "gpu box", "hot"}
	temps := []float64{0, 0.1, 0.7}
	byVariant := map[string]int{}
	for i, x := range runs {
		run := x.(map[string]any)
		opts := run["experiment"].(map[string]any)["options"].(map[string]any)
		if run["comparison_id"] != cid || run["variant_index"] != float64(i) || run["comparison_arm"] != wantArm[i] || run["label"] != wantLabel[i] ||
			opts["temperature"] != temps[i] || opts["think"] != false || run["dataset_fingerprint"] != runs[0].(map[string]any)["dataset_fingerprint"] {
			t.Fatalf("run %d: %v", i, run)
		}
		byVariant[run["id"].(string)] = i
	}
	list := e.do("GET", "/api/v1/benchmarks?comparison_id="+cid, nil, admin.cookie).json2(t)
	if len(list) != 3 {
		t.Fatalf("list by comparison: %d", len(list))
	}
	if r := e.do("GET", "/api/v1/benchmarks?comparison_id=nope", nil, admin.cookie); r.Code != 422 {
		t.Fatalf("bad comparison id: %d", r.Code)
	}

	// Lease order: plain run first; then per model (requested order) every
	// variant before the next model, then receipts.
	type step struct {
		model string
		run   string
	}
	var order []step
	for {
		task, err := e.st.LeaseBenchmarkTask(ctx, time.Now())
		if err != nil {
			t.Fatal(err)
		}
		if task == nil {
			break
		}
		name := "plain"
		if task.RunID != plain {
			name = strconv.Itoa(byVariant[task.RunID])
		}
		order = append(order, step{task.ModelID, name})
	}
	var want []step
	want = append(want, step{"vis:1", "plain"}, step{"vis:1", "plain"})
	for _, m := range []string{"vis:2", "vis:1"} {
		for v := 0; v < 3; v++ {
			want = append(want, step{m, strconv.Itoa(v)}, step{m, strconv.Itoa(v)}) // two receipts
		}
	}
	if !reflect.DeepEqual(order, want) {
		t.Fatalf("lease order\n got %v\nwant %v", order, want)
	}

	// An invalid arm rejects the whole sweep.
	var before int
	_ = e.st.Pool.QueryRow(ctx, "SELECT count(*) FROM benchmark_runs").Scan(&before)
	for body, want := range map[string]string{
		`{"variants": [{"experiment": {"provider_options": {"bogus": 1}}}]}`: "not supported by this provider",
		`{"variants": [{"experiment": {"temperature": 9}}]}`:                 "less_than_equal",
		`{"variants": []}`:              "too_short",
		`{"variants": [{"label": ""}]}`: "string_too_short",
		`{"variants": "x"}`:             "list_type",
		`{"variants": [1]}`:             "model_attributes_type",
	} {
		if r := e.do("POST", "/api/v1/benchmarks", body, admin.cookie); r.Code != 422 || !strings.Contains(r.Body.String(), want) {
			t.Errorf("%s: %d %s", body, r.Code, r.Body)
		}
	}
	var after int
	_ = e.st.Pool.QueryRow(ctx, "SELECT count(*) FROM benchmark_runs").Scan(&after)
	if after != before {
		t.Fatalf("rejected sweeps created runs: %d -> %d", before, after)
	}
}
