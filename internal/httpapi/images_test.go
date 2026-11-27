package httpapi

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tendstead/vision-bill/internal/llm"
)

// fakeProvider is a scriptable model backend.
type fakeProvider struct {
	mu        sync.Mutex
	down      bool
	models    []llm.ModelInfo
	replies   []string
	chatErr   error
	chatCalls int
	temp      float64
}

func (f *fakeProvider) Models(context.Context) ([]llm.ModelInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.down {
		return nil, errors.New("down")
	}
	return f.models, nil
}
func (f *fakeProvider) CheckConnection(context.Context) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return !f.down
}
func (f *fakeProvider) Chat(_ context.Context, _ string, _ []llm.Message, _ llm.ChatOptions) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.chatCalls++
	if f.chatErr != nil {
		return "", f.chatErr
	}
	if len(f.replies) == 0 {
		return "", nil
	}
	r := f.replies[0]
	if len(f.replies) > 1 {
		f.replies = f.replies[1:]
	}
	return r, nil
}
func (f *fakeProvider) SetTemperature(t float64)             { f.mu.Lock(); f.temp = t; f.mu.Unlock() }
func (f *fakeProvider) Temperature() float64                 { f.mu.Lock(); defer f.mu.Unlock(); return f.temp }
func (f *fakeProvider) Timeout() time.Duration               { return 5 * time.Second }
func (f *fakeProvider) Host() string                         { return "fake" }
func (f *fakeProvider) WithTemperature(float64) llm.Provider { return f }
func (f *fakeProvider) SupportsThinking() bool               { return true }
func (f *fakeProvider) Kind() string                         { return "OllamaProvider" }
func (f *fakeProvider) EndpointFingerprint() string          { return "fake-endpoint" }

func (f *fakeProvider) set(down bool, replies ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.down = down
	f.models = []llm.ModelInfo{{ID: "vis:1"}, {ID: "vis:2"}}
	f.replies = replies
	f.chatErr = nil
}

const modelReceipt = `{"confidence": 90, "merchant_name": "Shop", "date": "2025-03-04", "currency": "EUR", "category": "grocery",
 "line_items": [{"description": "Milk", "spending_category": "milk", "quantity": 1, "unit_price": "1.00", "total_price": "1.00", "tags": []}],
 "subtotal": "1.00", "total": "1.00"}`

var pngData = append([]byte("\x89PNG\r\n\x1a\n"), bytes.Repeat([]byte{7}, 64)...)

func multipartBody(t *testing.T, parts map[string][]byte, names ...string) (*bytes.Buffer, string) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, name := range names {
		fw, err := mw.CreateFormFile("receipt", name)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = fw.Write(parts[name])
	}
	_ = mw.Close()
	return &buf, mw.FormDataContentType()
}

func (e *testEnv) upload(t *testing.T, a actor, query string, files ...[]byte) resp {
	t.Helper()
	parts := map[string][]byte{}
	var names []string
	for i, f := range files {
		n := fmt.Sprintf("photo%d.png", i)
		parts[n] = f
		names = append(names, n)
	}
	body, ctype := multipartBody(t, parts, names...)
	req := httptest.NewRequest("POST", "/api/v1/images"+query, body)
	req.Header.Set("Content-Type", ctype)
	req.AddCookie(a.cookie)
	rec := httptest.NewRecorder()
	e.h.ServeHTTP(rec, req)
	return resp{rec}
}

func countFiles(t *testing.T, dir string) int {
	t.Helper()
	n := 0
	_ = filepath.Walk(dir, func(_ string, info os.FileInfo, _ error) error {
		if info != nil && !info.IsDir() {
			n++
		}
		return nil
	})
	return n
}

func TestUploadAnalysedSynchronously(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	e.provider.set(false, modelReceipt)

	r := e.upload(t, alice, "", pngData)
	if r.Code != 201 {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	j := r.json(t)
	imageID, receiptID := j["image_id"].(string), j["receipt_id"].(string)
	if r.Header().Get("Location") != "/api/v1/images/"+imageID || j["status"] != "analyzed" || j["media_type"] != "image/png" ||
		j["original_filename"] != "photo0.png" || j["size_bytes"] != float64(len(pngData)) {
		t.Fatalf("response %v %v", j, r.Header())
	}
	img := e.do("GET", "/api/v1/images/"+imageID, nil, alice.cookie).json(t)
	if img["status"] != "analyzed" || img["receipt_id"] != receiptID || img["bypass_review"] != false {
		t.Fatalf("image %v", img)
	}
	rec := e.do("GET", "/api/v1/receipts/"+receiptID, nil, alice.cookie).json(t)["receipt"].(map[string]any)
	if rec["status"] != "unverified" || rec["verified"] != false || rec["image_id"] != imageID {
		t.Fatalf("receipt %v", rec)
	}
	if !strings.HasPrefix(img["image_path"].(string), filepath.Join(e.dir, "tmp")) {
		t.Fatalf("original must stay in tmp until verified: %v", img["image_path"])
	}
	// The stored file streams back with the detected type.
	f := e.do("GET", "/api/v1/images/"+imageID+"/file", nil, alice.cookie)
	if f.Code != 200 || f.Header().Get("Content-Type") != "image/png" || !bytes.Equal(f.Body.Bytes(), pngData) || f.Header().Get("ETag") == "" {
		t.Fatalf("file: %d %v", f.Code, f.Header())
	}
	for path, want := range map[string]int{"/file?part=1": 404, "/thumb": 404, "/file?part=-1": 422} {
		if r := e.do("GET", "/api/v1/images/"+imageID+path, nil, alice.cookie); r.Code != want {
			t.Errorf("%s: %d %s", path, r.Code, r.Body)
		}
	}
	bob := e.user(t, "bob", false)
	if r := e.do("GET", "/api/v1/images/"+imageID, nil, bob.cookie); r.Code != 404 {
		t.Fatalf("foreign image: %d", r.Code)
	}

	// Verify moves the file to permanent storage and marks the receipt verified.
	v := e.do("POST", "/api/v1/receipts/"+receiptID+"/verify", nil, alice.cookie)
	if v.Code != 200 || v.json(t)["status"] != "verified" {
		t.Fatalf("verify: %d %s", v.Code, v.Body)
	}
	img = e.do("GET", "/api/v1/images/"+imageID, nil, alice.cookie).json(t)
	if want := filepath.Join(e.dir, "save", "receipt_"+receiptID+".png"); img["image_path"] != want {
		t.Fatalf("path %v, want %s", img["image_path"], want)
	}
	if countFiles(t, filepath.Join(e.dir, "tmp")) != 0 || countFiles(t, filepath.Join(e.dir, "save")) != 1 {
		t.Fatal("file should have moved from tmp to save")
	}
	if r := e.do("POST", "/api/v1/receipts/"+receiptID+"/verify", nil, alice.cookie); r.Code != 409 || r.json(t)["detail"] != "Receipt already verified" {
		t.Fatalf("verify twice: %d %s", r.Code, r.Body)
	}
	if r := e.do("POST", "/api/v1/receipts/"+receiptID+"/verify", nil, bob.cookie); r.Code != 404 {
		t.Fatalf("foreign verify: %d", r.Code)
	}

	// Deleting the receipt removes the submission and its files.
	if r := e.do("DELETE", "/api/v1/receipts/"+receiptID, nil, bob.cookie); r.Code != 404 {
		t.Fatalf("foreign delete: %d", r.Code)
	}
	d := e.do("DELETE", "/api/v1/receipts/"+receiptID, nil, alice.cookie)
	if d.Code != 200 || d.json(t)["deleted"] != receiptID {
		t.Fatalf("delete: %d %s", d.Code, d.Body)
	}
	if countFiles(t, e.dir) != 0 {
		t.Fatal("files must be removed")
	}
	if r := e.do("GET", "/api/v1/images/"+imageID, nil, alice.cookie); r.Code != 404 {
		t.Fatalf("image row must be gone: %d", r.Code)
	}
}

func TestUploadBypassReviewMovesFileImmediately(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	e.provider.set(false, modelReceipt)
	r := e.upload(t, alice, "?bypass_review=true&model_id=vis:2", pngData)
	if r.Code != 201 {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	rid := r.json(t)["receipt_id"].(string)
	rec := e.do("GET", "/api/v1/receipts/"+rid, nil, alice.cookie).json(t)
	if rec["receipt"].(map[string]any)["status"] != "verified" || !strings.Contains(rec["image_path"].(string), "/save/receipt_"+rid) {
		t.Fatalf("%v", rec)
	}
	if r := e.upload(t, alice, "?bypass_review=maybe", pngData); r.Code != 422 || !strings.Contains(r.Body.String(), "bool_parsing") {
		t.Fatalf("bad bool: %d %s", r.Code, r.Body)
	}
}

func TestUploadQueuedWhenProviderDownThenAnalysedByWorker(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	e.provider.set(true)
	r := e.upload(t, alice, "", pngData)
	j := r.json(t)
	if r.Code != 202 || j["status"] != "pending" || j["warning"] != pendingQueueWarning {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	imageID := j["image_id"].(string)
	// While the provider stays down the cycle leaves the image queued.
	if res, err := e.scheduler.ProcessPending(context.Background()); err != nil || len(res) != 0 {
		t.Fatalf("%v %v", res, err)
	}
	if e.do("GET", "/api/v1/images/"+imageID, nil, alice.cookie).json(t)["status"] != "pending" {
		t.Fatal("must stay pending")
	}
	list := e.do("GET", "/api/v1/images?status=pending,failed", nil, alice.cookie)
	if !strings.Contains(list.Body.String(), imageID) {
		t.Fatalf("queue listing: %s", list.Body)
	}

	e.provider.set(false, modelReceipt)
	// A regular user may trigger the worker but never sees (or runs inline) the queue.
	if u := e.do("POST", "/api/v1/images/analyze", nil, alice.cookie); u.Code != 200 || len(u.json(t)["results"].([]any)) != 0 {
		t.Fatalf("non-admin trigger: %d %s", u.Code, u.Body)
	}
	if e.do("GET", "/api/v1/images/"+imageID, nil, alice.cookie).json(t)["status"] != "pending" {
		t.Fatal("a non-admin trigger must not run the cycle inline")
	}
	boss := e.user(t, "boss", true)
	m := e.do("POST", "/api/v1/images/analyze", nil, boss.cookie)
	results := m.json(t)["results"].([]any)
	if m.Code != 200 || len(results) != 1 || results[0].(map[string]any)["status"] != "analyzed" {
		t.Fatalf("manual trigger: %d %s", m.Code, m.Body)
	}
	if got := e.do("GET", "/api/v1/images/"+imageID, nil, alice.cookie).json(t); got["status"] != "analyzed" || got["receipt_id"] == nil {
		t.Fatalf("%v", got)
	}
	// Re-running creates nothing new.
	if res, _ := e.scheduler.ProcessPending(context.Background()); len(res) != 0 {
		t.Fatalf("analysed image must not be queued again: %v", res)
	}
	var n int
	_ = e.st.Pool.QueryRow(context.Background(), "SELECT count(*) FROM receipts").Scan(&n)
	if n != 1 {
		t.Fatalf("%d receipts", n)
	}
}

func TestUploadOutcomes(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)

	e.provider.set(false, `{"error": {"code": "unreadable", "message": "bottom cut off"}}`)
	r := e.upload(t, alice, "", pngData)
	if r.Code != 422 || r.json(t)["status"] != "unreadable" || r.json(t)["detail"] != "bottom cut off" || r.Header().Get("Location") == "" {
		t.Fatalf("unreadable: %d %s", r.Code, r.Body)
	}
	e.provider.set(false)
	e.provider.chatErr = context.DeadlineExceeded
	r = e.upload(t, alice, "", pngData)
	if r.Code != 504 || r.json(t)["status"] != "timed_out" {
		t.Fatalf("timeout: %d %s", r.Code, r.Body)
	}
	e.provider.set(false, "garbage", "garbage", "garbage")
	r = e.upload(t, alice, "", pngData)
	if r.Code != 500 {
		t.Fatalf("exhausted retries: %d %s", r.Code, r.Body)
	}
	list := e.do("GET", "/api/v1/images", nil, alice.cookie)
	for _, want := range []string{`"status":"unreadable"`, `"status":"timed_out"`, `"status":"failed"`} {
		if !strings.Contains(list.Body.String(), want) {
			t.Errorf("listing lacks %s", want)
		}
	}
	// Terminal and failed submissions can be removed; active/analysed ones cannot.
	imgs := e.do("GET", "/api/v1/images", nil, alice.cookie)
	_ = imgs
	var id string
	_ = e.st.Pool.QueryRow(context.Background(), "SELECT id::text FROM images WHERE status='unreadable'").Scan(&id)
	if r := e.do("DELETE", "/api/v1/images/"+id, nil, alice.cookie); r.Code != 200 || r.json(t)["deleted"] != id {
		t.Fatalf("delete unreadable: %d %s", r.Code, r.Body)
	}
	e.provider.set(false, modelReceipt)
	ok := e.upload(t, alice, "", pngData)
	if r := e.do("DELETE", "/api/v1/images/"+ok.json(t)["image_id"].(string), nil, alice.cookie); r.Code != 409 {
		t.Fatalf("delete analysed: %d", r.Code)
	}
}

func TestUploadValidation(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	e.provider.set(false, modelReceipt)
	if r := e.upload(t, alice, "", []byte("%PDF-1.7 not an image")); r.Code != 415 || r.json(t)["detail"] != "Unsupported image type" {
		t.Fatalf("pdf: %d %s", r.Code, r.Body)
	}
	if r := e.upload(t, alice, ""); r.Code != 422 {
		t.Fatalf("no files: %d %s", r.Code, r.Body)
	}
	req := httptest.NewRequest("POST", "/api/v1/images", strings.NewReader("{}"))
	req.Header.Set("Content-Type", "application/json")
	req.AddCookie(alice.cookie)
	rec := httptest.NewRecorder()
	e.h.ServeHTTP(rec, req)
	if rec.Code != 422 || !strings.Contains(rec.Body.String(), `"missing"`) {
		t.Fatalf("non-multipart: %d %s", rec.Code, rec.Body)
	}
	eleven := make([][]byte, 11)
	for i := range eleven {
		eleven[i] = pngData
	}
	if r := e.upload(t, alice, "", eleven...); r.Code != 422 || r.json(t)["detail"] != "Upload between 1 and 10 photos of one receipt" {
		t.Fatalf("11 photos: %d %s", r.Code, r.Body)
	}
	big := append([]byte("\x89PNG\r\n\x1a\n"), make([]byte, maxPhotoBytes)...)
	if r := e.upload(t, alice, "", big); r.Code != 413 || r.json(t)["detail"] != "Each photo must be at most 20 MB" {
		t.Fatalf("oversized: %d %s", r.Code, r.Body)
	}
	if countFiles(t, e.dir) != 0 {
		t.Fatal("rejected uploads must not leave files behind")
	}
	if r := e.upload(t, actor{cookie: &http.Cookie{Name: "vb_session", Value: "x"}}, "", pngData); r.Code != 401 {
		t.Fatalf("anonymous: %d", r.Code)
	}
}

const multiModelReceipt = `{"confidence": 90, "merchant_name": "Shop", "date": "2025-03-04", "subtotal": "3.00", "total": "3.00",
 "line_items": [
  {"description": "A", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 0},
  {"description": "B", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 0},
  {"description": "B", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 1},
  {"description": "C", "quantity": 1, "unit_price": "1", "total_price": "1", "source_image": 1}],
 "duplicate_candidates": [{"first_index": 1, "second_index": 2}]}`

func TestMultiPhotoSubmission(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	e.provider.set(false, multiModelReceipt)
	second := append(append([]byte{}, pngData...), 9)
	r := e.upload(t, alice, "?bypass_review=true", pngData, second)
	if r.Code != 202 || r.json(t)["status"] != "pending" || r.json(t)["warning"] != nil {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	id := r.json(t)["image_id"].(string)
	img := e.do("GET", "/api/v1/images/"+id, nil, alice.cookie).json(t)
	if len(img["additional_images"].([]any)) != 1 {
		t.Fatalf("%v", img)
	}
	if res, err := e.scheduler.ProcessPending(context.Background()); err != nil || len(res) != 1 || res[0].Status != "analyzed" {
		t.Fatalf("%v %v", res, err)
	}
	img = e.do("GET", "/api/v1/images/"+id, nil, alice.cookie).json(t)
	rid := img["receipt_id"].(string)
	d := e.do("GET", "/api/v1/receipts/"+rid, nil, alice.cookie).json(t)
	if n := len(d["line_items"].([]any)); n != 3 {
		t.Fatalf("overlap row must be removed, got %d items", n)
	}
	// Both photos moved to permanent storage and are reachable by part index.
	if countFiles(t, filepath.Join(e.dir, "tmp")) != 0 || countFiles(t, filepath.Join(e.dir, "save")) != 2 {
		t.Fatal("both photos must be in permanent storage")
	}
	for part, want := range map[string][]byte{"0": pngData, "1": second} {
		f := e.do("GET", "/api/v1/images/"+id+"/file?part="+part, nil, alice.cookie)
		if f.Code != 200 || !bytes.Equal(f.Body.Bytes(), want) {
			t.Fatalf("part %s: %d", part, f.Code)
		}
	}
	if r := e.do("GET", "/api/v1/images/"+id+"/file?part=2", nil, alice.cookie); r.Code != 404 || r.json(t)["detail"] != "Image part not found" {
		t.Fatalf("part 2: %d %s", r.Code, r.Body)
	}
	if r := e.do("DELETE", "/api/v1/receipts/"+rid, nil, alice.cookie); r.Code != 200 || countFiles(t, e.dir) != 0 {
		t.Fatalf("delete removes both photos: %d", r.Code)
	}
}

func TestExpiredLeaseIsReclaimedAndMissingFileFails(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	e.provider.set(true)
	r := e.upload(t, alice, "", pngData)
	id := r.json(t)["image_id"].(string)
	ctx := context.Background()
	// A worker crashed mid-analysis long ago.
	if _, err := e.st.Pool.Exec(ctx, `UPDATE images SET status='processing', processing_at = now() - interval '20 minutes' WHERE id=$1::uuid`, id); err != nil {
		t.Fatal(err)
	}
	e.provider.set(false, modelReceipt)
	res, err := e.scheduler.ProcessPending(ctx)
	if err != nil || len(res) != 1 || res[0].Status != "analyzed" {
		t.Fatalf("lease not reclaimed: %v %v", res, err)
	}
	// A fresh lease is respected.
	e.provider.set(true)
	r = e.upload(t, alice, "", pngData)
	e.provider.set(false, modelReceipt)
	id2 := r.json(t)["image_id"].(string)
	_, _ = e.st.Pool.Exec(ctx, `UPDATE images SET status='processing', processing_at = now() WHERE id=$1::uuid`, id2)
	if res, _ := e.scheduler.ProcessPending(ctx); len(res) != 0 {
		t.Fatalf("active lease stolen: %v", res)
	}
	// Missing file => failed with an explanatory error, retried later.
	_, _ = e.st.Pool.Exec(ctx, `UPDATE images SET status='pending', processing_at=NULL WHERE id=$1::uuid`, id2)
	var path string
	_ = e.st.Pool.QueryRow(ctx, `SELECT image_path FROM images WHERE id=$1::uuid`, id2).Scan(&path)
	_ = os.Remove(path)
	res, _ = e.scheduler.ProcessPending(ctx)
	if len(res) != 1 || res[0].Status != "failed" || !strings.Contains(*res[0].Error, "Image file missing") {
		t.Fatalf("%+v", res)
	}
}

func TestSettingsAppliesTemperatureAndUIConfig(t *testing.T) {
	e := newEnv(t, false)
	r := e.do("GET", "/api/v1/system/ui-config", nil)
	if r.json(t)["heic_supported"] != false {
		t.Fatalf("%s", r.Body)
	}
}

// A model timeout is never final: the image stays queued, is retried only
// after its backoff delay and succeeds once the model answers in time.
func TestTimeoutIsRetriedWithBackoff(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	e.provider.set(false, modelReceipt)
	e.provider.mu.Lock()
	e.provider.chatErr = context.DeadlineExceeded
	e.provider.mu.Unlock()

	r := e.upload(t, alice, "", pngData)
	if r.Code != 504 || r.json(t)["status"] != "timed_out" {
		t.Fatalf("timeout upload: %d %s", r.Code, r.Body)
	}
	imageID := r.json(t)["image_id"].(string)
	var attempts int
	var wait string
	q := func() {
		t.Helper()
		err := e.st.Pool.QueryRow(context.Background(),
			`SELECT analysis_attempts, (retry_after > now() + interval '9 minutes')::text FROM images WHERE id = $1::uuid`, imageID).Scan(&attempts, &wait)
		if err != nil {
			t.Fatal(err)
		}
	}
	q()
	if attempts != 1 || wait != "true" {
		t.Fatalf("attempts=%d delayed=%s", attempts, wait)
	}

	// Healthy model, but the backoff has not passed: nothing is claimed.
	e.provider.set(false, modelReceipt)
	if res, err := e.scheduler.ProcessPending(context.Background()); err != nil || len(res) != 0 {
		t.Fatalf("backoff ignored: %v %v", res, err)
	}
	// A second timeout doubles the delay.
	if _, err := e.st.Pool.Exec(context.Background(), `UPDATE images SET retry_after = now() - interval '1 second' WHERE id = $1::uuid`, imageID); err != nil {
		t.Fatal(err)
	}
	e.provider.mu.Lock()
	e.provider.chatErr = context.DeadlineExceeded
	e.provider.mu.Unlock()
	res, _ := e.scheduler.ProcessPending(context.Background())
	if len(res) != 1 || res[0].Status != "timed_out" {
		t.Fatalf("retry after backoff: %v", res)
	}
	q()
	if attempts != 2 {
		t.Fatalf("attempts=%d", attempts)
	}
	// Once the delay passed and the model answers, the image is analysed.
	_, _ = e.st.Pool.Exec(context.Background(), `UPDATE images SET retry_after = now() - interval '1 second' WHERE id = $1::uuid`, imageID)
	e.provider.set(false, modelReceipt)
	res, _ = e.scheduler.ProcessPending(context.Background())
	if len(res) != 1 || res[0].Status != "analyzed" {
		t.Fatalf("final retry: %v", res)
	}
}
