package benchmark

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shopspring/decimal"

	"github.com/tendstead/vision-bill/internal/analysis"
	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/llm"
	"github.com/tendstead/vision-bill/internal/store"
)

// UserError is a request problem reported to the client (422/404).
type UserError struct{ Msg string }

func (e *UserError) Error() string { return e.Msg }

func userErr(format string, a ...any) error { return &UserError{fmt.Sprintf(format, a...)} }

// ErrRunNotFound reports an unknown benchmark run.
var ErrRunNotFound = errors.New("Benchmark run not found")

// ErrTimeout reports that a transient re-evaluation exceeded the run timeout.
var ErrTimeout = errors.New("Benchmark reevaluation exceeded its run timeout")

// ExperimentOptions are the request-scoped experiment settings.
type ExperimentOptions struct {
	Pipeline              string   `json:"pipeline"`
	Temperature           *float64 `json:"temperature"`
	Think                 *bool    `json:"think"`
	IncludeClassification bool     `json:"include_classification"`
	AllowOCRFallback      bool     `json:"allow_ocr_fallback"`
}

// CreateRequest is a validated run request.
type CreateRequest struct {
	Experiment          ExperimentOptions
	ModelIDs            []string
	ReceiptIDs          []string
	Category            *string
	MaxSourceConfidence *int
	Limit               *int
	TimeoutSeconds      int
	CouncilPolicy       string
	AbsoluteThreshold   *string
	RelativeThreshold   *string
	ApplyCouncilFlags   bool
}

// Service creates runs and executes their tasks.
type Service struct {
	Store    func() *store.Store
	Provider llm.Provider
	Analysis *analysis.Service
	Log      *slog.Logger
	Now      func() time.Time

	mu     sync.Mutex
	cancel context.CancelFunc
	done   chan struct{}
}

func (s *Service) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

func (s *Service) log() *slog.Logger {
	if s.Log != nil {
		return s.Log
	}
	return slog.Default()
}

// Capabilities reports what experiments can use. OCR is an optional
// extension that is not part of this build.
func (s *Service) Capabilities() map[string]bool {
	return map[string]bool{"ocr_available": false, "thinking_control": s.Provider.SupportsThinking()}
}

// Start launches the background worker.
func (s *Service) Start(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cancel != nil {
		return
	}
	ctx, s.cancel = context.WithCancel(ctx)
	s.done = make(chan struct{})
	go s.work(ctx)
}

// Stop ends the worker.
func (s *Service) Stop() {
	s.mu.Lock()
	cancel, done := s.cancel, s.done
	s.cancel = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
		<-done
	}
}

func sleep(ctx context.Context, d time.Duration) {
	select {
	case <-ctx.Done():
	case <-time.After(d):
	}
}

func (s *Service) work(ctx context.Context) {
	defer close(s.done)
	for ctx.Err() == nil {
		st := s.Store()
		if st == nil {
			sleep(ctx, time.Second)
			continue
		}
		task, err := st.LeaseBenchmarkTask(ctx, s.now())
		if err != nil {
			if ctx.Err() == nil {
				s.log().Error("benchmark worker iteration failed", "err", err)
				sleep(ctx, 5*time.Second)
			}
			continue
		}
		if task == nil {
			sleep(ctx, time.Second)
			continue
		}
		s.Execute(ctx, st, *task)
	}
}

// ── run creation ─────────────────────────────────────────────────────────

func promptOptions(o ExperimentOptions) llm.PromptOptions {
	return llm.PromptOptions{ExtractionOnly: !o.IncludeClassification}
}

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

type inputImage struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type manifestInput struct {
	ReceiptID string          `json:"receipt_id"`
	Expected  json.RawMessage `json:"expected"`
	Images    []inputImage    `json:"images"`
	Pipeline  string          `json:"pipeline"`
}

type manifest struct {
	Engine              string            `json:"engine,omitempty"`
	Options             ExperimentOptions `json:"options"`
	Tags                []string          `json:"tags"`
	Inputs              []manifestInput   `json:"inputs"`
	Provider            string            `json:"provider"`
	EndpointFingerprint *string           `json:"endpoint_fingerprint"`
	OCRFingerprint      *string           `json:"ocr_fingerprint"`
	PromptHashes        map[string]string `json:"prompt_hashes"`
	ValidationSchemaSHA string            `json:"validation_schema_sha256"`
	ThinkingDefault     string            `json:"thinking_default"`
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// expectedFromDetails builds the expected extraction from a verified receipt.
func expectedFromDetails(d *domain.ReceiptWithDetails) (*domain.Receipt, error) {
	r := d.Receipt
	dec := func(s domain.DecimalStr) decimal.Decimal {
		v, _ := decimal.NewFromString(string(s))
		return v
	}
	date, err := domain.ParseDate(r.Date.String())
	if err != nil {
		return nil, err
	}
	out := &domain.Receipt{
		Confidence: r.Confidence, MerchantName: r.MerchantName, MerchantAddress: r.MerchantAddress, ReceiptNumber: r.ReceiptNumber,
		Date: date, Time: r.Time, Currency: r.Currency, Category: r.Category, PaymentMethod: r.PaymentMethod,
		Subtotal: dec(r.Subtotal), DiscountTotal: dec(r.DiscountTotal), TaxTotal: dec(r.TaxTotal), Total: dec(r.Total),
		LineItems: []domain.LineItem{}, Taxes: []domain.TaxLine{},
	}
	if r.Tip != nil {
		t := dec(*r.Tip)
		out.Tip = &t
	}
	for _, li := range d.LineItems {
		q, _ := decimal.NewFromString(strconv.FormatFloat(li.Quantity, 'f', -1, 64))
		out.LineItems = append(out.LineItems, domain.LineItem{
			Description: li.Description, SpendingCategory: li.SpendingCategory, Quantity: q,
			UnitPrice: dec(li.UnitPrice), TotalPrice: dec(li.TotalPrice), Tags: li.Tags,
		})
	}
	for _, t := range d.Taxes {
		tl := domain.TaxLine{Name: t.Name, Amount: dec(t.Amount)}
		if t.Rate != nil {
			rate, _ := decimal.NewFromString(strconv.FormatFloat(*t.Rate, 'f', -1, 64))
			tl.Rate = &rate
		}
		out.Taxes = append(out.Taxes, tl)
	}
	return out, nil
}

func (s *Service) expected(ctx context.Context, st *store.Store, receiptID string) (*domain.Receipt, *domain.ReceiptWithDetails, error) {
	d, err := st.ReceiptForBenchmark(ctx, receiptID)
	if err != nil {
		return nil, nil, err
	}
	if d == nil || !d.Receipt.Verified {
		return nil, nil, userErr("Benchmark data must be a verified receipt")
	}
	r, err := expectedFromDetails(d)
	return r, d, err
}

func (s *Service) selectReceipts(ctx context.Context, st *store.Store, req CreateRequest) ([]string, error) {
	if req.ReceiptIDs != nil {
		out := []string{}
		for _, id := range req.ReceiptIDs {
			row, err := st.ReceiptByID(ctx, id, store.Scope{CanSeeAll: true})
			if err != nil {
				return nil, err
			}
			if row != nil && row.Verified {
				out = append(out, id)
			}
		}
		return out, nil
	}
	limit := 10_000
	if req.Limit != nil {
		limit = *req.Limit
	}
	rows, err := st.ListReceipts(ctx, store.ListFilter{Limit: limit, Status: []string{"verified"}}, store.Scope{CanSeeAll: true})
	if err != nil {
		return nil, err
	}
	out := []string{}
	for _, r := range rows {
		if (req.Category == nil || r.Category == *req.Category) && (req.MaxSourceConfidence == nil || r.Confidence <= *req.MaxSourceConfidence) {
			out = append(out, r.ID)
		}
	}
	return out, nil
}

func (s *Service) captureInput(ctx context.Context, st *store.Store, receiptID string) (manifestInput, *domain.Receipt, error) {
	expected, details, err := s.expected(ctx, st, receiptID)
	if err != nil {
		return manifestInput{}, nil, err
	}
	if details.ImagePath == nil || *details.ImagePath == "" {
		return manifestInput{}, nil, userErr("Verified receipt image is unavailable")
	}
	paths := []string{*details.ImagePath}
	if details.Receipt.ImageID != nil {
		img, err := st.ImageByID(ctx, *details.Receipt.ImageID, store.Scope{CanSeeAll: true})
		if err != nil {
			return manifestInput{}, nil, err
		}
		if img != nil {
			paths = paths[:0]
			for _, p := range img.Parts() {
				paths = append(paths, p.ImagePath)
			}
		}
	}
	in := manifestInput{ReceiptID: receiptID, Pipeline: "direct_vision"}
	for _, p := range paths {
		sum, err := fileSHA256(p)
		if err != nil {
			return manifestInput{}, nil, userErr("Receipt image is unavailable: %s", p)
		}
		in.Images = append(in.Images, inputImage{p, sum})
	}
	raw, err := json.Marshal(Dump(expected))
	if err != nil {
		return manifestInput{}, nil, err
	}
	in.Expected = raw
	return in, expected, nil
}

// CreateRun validates the request, freezes the inputs and queues all
// (model x receipt) tasks.
func (s *Service) CreateRun(ctx context.Context, req CreateRequest) (*domain.BenchmarkRun, error) {
	st := s.Store()
	models, err := s.Provider.Models(ctx)
	if err != nil {
		return nil, err
	}
	available := map[string]llm.ModelInfo{}
	var order []string
	for _, m := range models {
		available[m.ID] = m
		order = append(order, m.ID)
	}
	selected := req.ModelIDs
	if len(selected) == 0 {
		selected = order
	}
	if len(selected) == 0 {
		return nil, userErr("No vision models are available")
	}
	var unknown []string
	for _, id := range selected {
		if _, ok := available[id]; !ok {
			unknown = append(unknown, id)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return nil, userErr("Selected model(s) unavailable: %s", strings.Join(dedupe(unknown), ", "))
	}
	receiptIDs, err := s.selectReceipts(ctx, st, req)
	if err != nil {
		return nil, err
	}
	if len(receiptIDs) == 0 {
		return nil, userErr("No verified receipts match the selection")
	}
	if req.Experiment.Think != nil && !s.Provider.SupportsThinking() {
		return nil, userErr("Thinking override is supported only by the Ollama provider")
	}
	if req.Experiment.Pipeline == "ocr" {
		return nil, userErr("OCR extension is not configured; choose normal upload or direct vision")
	}
	tags, err := st.ListTags(ctx)
	if err != nil {
		return nil, err
	}
	var inputs []manifestInput
	var expected []*domain.Receipt
	for _, id := range receiptIDs {
		in, exp, err := s.captureInput(ctx, st, id)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, in)
		expected = append(expected, exp)
	}

	options := req.Experiment
	if options.Temperature == nil {
		t := s.Provider.Temperature()
		options.Temperature = &t
	}
	counts := map[int]bool{}
	for _, in := range inputs {
		counts[len(in.Images)] = true
	}
	hashes := map[string]string{}
	for c := range counts {
		hashes[strconv.Itoa(c)] = sha(llm.BuildPrompt(tags, c, promptOptions(options)))
	}
	thinking := "server-controlled"
	if options.Think != nil {
		thinking = "explicit"
	}
	endpoint := s.Provider.EndpointFingerprint()
	m := manifest{Engine: "go", Options: options, Tags: orEmpty(tags), Inputs: inputs, Provider: s.Provider.Kind(),
		EndpointFingerprint: &endpoint, PromptHashes: hashes, ValidationSchemaSHA: llm.SchemaSHA256(), ThinkingDefault: thinking}
	raw, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	digests := map[string]*string{}
	for _, id := range selected {
		digests[id] = available[id].Digest
	}
	return st.CreateBenchmarkRun(ctx, store.NewBenchmarkRun{
		ModelIDs: dedupe(selected), ReceiptIDs: receiptIDs, Fingerprint: DatasetFingerprint(expected), ModelDigests: digests,
		TimeoutSeconds: req.TimeoutSeconds, Policy: req.CouncilPolicy, AbsoluteThreshold: req.AbsoluteThreshold,
		RelativeThreshold: req.RelativeThreshold, ApplyFlags: req.ApplyCouncilFlags,
		PromptVersion: PromptVersion, ScoringVersion: DetailScoringVersion, Manifest: raw,
	})
}

func orEmpty(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

func dedupe(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// ── queries ──────────────────────────────────────────────────────────────

// Status returns live queue state and summaries; nil when the run is unknown.
func (s *Service) Status(ctx context.Context, id string) (*domain.BenchmarkStatus, error) {
	st := s.Store()
	run, err := st.BenchmarkRun(ctx, id)
	if err != nil || run == nil {
		return nil, err
	}
	q, r, w, rt, term, err := st.TaskCounts(ctx, id)
	if err != nil {
		return nil, err
	}
	sums, err := st.BenchmarkSummaries(ctx, id)
	if err != nil {
		return nil, err
	}
	return &domain.BenchmarkStatus{Run: *run, Queued: q, Running: r, WaitingForModel: w, Retrying: rt, Terminal: term, Summaries: sums}, nil
}

// Results pages stored task results.
func (s *Service) Results(ctx context.Context, id string, limit, offset int) ([]map[string]any, error) {
	st := s.Store()
	run, err := st.BenchmarkRun(ctx, id)
	if err != nil {
		return nil, err
	}
	if run == nil {
		return nil, ErrRunNotFound
	}
	return st.BenchmarkResults(ctx, id, limit, offset)
}

// ── execution ────────────────────────────────────────────────────────────

func parseManifest(raw json.RawMessage) (*manifest, error) {
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// Execute runs one leased task to a terminal outcome or schedules a retry.
func (s *Service) Execute(ctx context.Context, st *store.Store, task store.BenchmarkTask) {
	rec := context.WithoutCancel(ctx)
	retry := func(waiting bool) {
		if err := st.RetryBenchmarkTask(rec, task, waiting, s.now()); err != nil {
			s.log().Error("could not schedule benchmark retry", "err", err)
		}
	}
	mctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	models, err := s.Provider.Models(mctx)
	cancel()
	if err != nil {
		retry(false)
		return
	}
	run, err := st.BenchmarkRun(rec, task.RunID)
	if err != nil || run == nil {
		return
	}
	digests, timeout, err := st.BenchmarkMeta(rec, task.RunID)
	if err != nil {
		retry(false)
		return
	}
	var model *llm.ModelInfo
	for i := range models {
		if models[i].ID == task.ModelID {
			model = &models[i]
		}
	}
	if model == nil {
		retry(true)
		return
	}
	if want := digests[task.ModelID]; want != nil && *want != "" && model.Digest != nil && *model.Digest != "" && *want != *model.Digest {
		retry(true)
		return
	}
	raw, err := st.BenchmarkManifest(rec, task.RunID)
	if err != nil {
		retry(false)
		return
	}
	deadline := time.Duration(timeout) * time.Second
	if raw != nil {
		s.executeExperiment(ctx, rec, st, task, raw, deadline)
		return
	}
	s.executeLegacy(ctx, rec, st, task, deadline, retry)
}

func (s *Service) executeLegacy(ctx, rec context.Context, st *store.Store, task store.BenchmarkTask, deadline time.Duration, retry func(bool)) {
	fail := func() { _ = st.FinishBenchmarkTask(rec, task, store.FinishBenchmark{Success: false, Attempts: 0}) }
	expected, details, err := s.expected(rec, st, task.ReceiptID)
	if err != nil {
		var ue *UserError
		if errors.As(err, &ue) {
			fail()
		} else {
			retry(false)
		}
		return
	}
	if details.ImagePath == nil || !fileExists(*details.ImagePath) {
		fail()
		return
	}
	actx, cancel := context.WithTimeout(ctx, deadline)
	defer cancel()
	images, err := s.Analysis.LoadImages(actx, []domain.ImagePart{{ImagePath: *details.ImagePath}})
	if err != nil {
		fail()
		return
	}
	a, err := llm.Analyse(actx, s.Provider, task.ModelID, images, nil, llm.PromptOptions{}, llm.ChatOptions{})
	if err != nil {
		var unreadable *llm.UnreadableError
		var vf *llm.ValidationFailure
		switch {
		case errors.As(err, &unreadable), errors.As(err, &vf), strings.HasPrefix(err.Error(), "Failed to get a valid response"):
			fail()
		default:
			s.log().Warn("transient benchmark failure", "run", task.RunID, "model", task.ModelID, "receipt", task.ReceiptID, "err", err)
			retry(false)
		}
		return
	}
	_ = st.FinishBenchmarkTask(rec, task, store.FinishBenchmark{Success: true, Score: ScoreReceipts(expected, a.Receipt)["overall"],
		Confidence: a.Receipt.Confidence, Attempts: a.Attempts, LatencyMS: a.ElapsedMS})
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// experimentResult is the outcome of one frozen-input extraction.
type experimentResult struct {
	Receipt  *domain.Receipt
	Attempts int
	Elapsed  float64
	Metadata map[string]any
}

type experimentFailure struct {
	msg      string
	metadata map[string]any
}

func (e *experimentFailure) Error() string { return e.msg }

func errKind(err error) string {
	var timeout *llm.TimeoutError
	var unreadable *llm.UnreadableError
	switch {
	case errors.As(err, &timeout):
		return "AnalysisTimeoutError"
	case errors.As(err, &unreadable):
		return "UnreadableReceiptError"
	}
	return "ValueError"
}

// runExperiment verifies the frozen inputs still match and extracts again.
func (s *Service) runExperiment(ctx context.Context, m *manifest, receiptID, modelID string) (*domain.Receipt, *experimentResult, error) {
	current := s.Provider.Kind()
	if m.Provider != current {
		return nil, nil, userErr("Provider changed since run creation; create a new run")
	}
	var item *manifestInput
	for i := range m.Inputs {
		if m.Inputs[i].ReceiptID == receiptID {
			item = &m.Inputs[i]
		}
	}
	if item == nil {
		return nil, nil, userErr("Receipt is absent from the frozen benchmark inputs")
	}
	if item.Pipeline == "ocr" {
		return nil, nil, userErr("OCR provider endpoint or client settings changed; create a new run")
	}
	if m.Engine == "go" {
		// Fingerprints of this implementation's endpoint, prompt and schema can
		// be compared; earlier manifests record values that are not comparable.
		if m.EndpointFingerprint == nil || *m.EndpointFingerprint != s.Provider.EndpointFingerprint() {
			return nil, nil, userErr("Provider endpoint changed since run creation; create a new run")
		}
		if m.ValidationSchemaSHA != llm.SchemaSHA256() {
			return nil, nil, userErr("Receipt validation schema changed; create a new run")
		}
		want := m.PromptHashes[strconv.Itoa(len(item.Images))]
		if got := sha(llm.BuildPrompt(m.Tags, len(item.Images), promptOptions(m.Options))); got != want {
			return nil, nil, userErr("Extraction prompt changed since run creation; create a new run")
		}
	}
	for _, img := range item.Images {
		sum, err := fileSHA256(img.Path)
		if err != nil {
			return nil, nil, err
		}
		if sum != img.SHA256 {
			return nil, nil, userErr("Receipt image changed since run creation")
		}
	}
	expected, issues := domain.ParseReceipt(item.Expected, domain.Options{})
	if issues != nil {
		return nil, nil, userErr("Frozen expected receipt is invalid: %s", issues.Error())
	}

	started := time.Now()
	temperature := s.Provider.Temperature()
	if m.Options.Temperature != nil {
		temperature = *m.Options.Temperature
	}
	provider := s.Provider.WithTemperature(temperature)
	metadata := map[string]any{
		"requested_pipeline": m.Options.Pipeline, "fallback_used": false, "image_count": len(item.Images),
		"provider": provider.Kind(), "options": m.Options,
	}
	parts := make([]domain.ImagePart, len(item.Images))
	for i, img := range item.Images {
		parts[i] = domain.ImagePart{ImagePath: img.Path}
		if !fileExists(img.Path) {
			return nil, nil, fmt.Errorf("Receipt image or additional photo is unavailable")
		}
	}
	var calls []map[string]any
	var callsMu sync.Mutex
	copts := llm.ChatOptions{Think: m.Options.Think, Trace: func(t map[string]any) {
		callsMu.Lock()
		calls = append(calls, t)
		callsMu.Unlock()
	}}
	actx, cancel := context.WithTimeout(ctx, provider.Timeout())
	defer cancel()
	var a *llm.Analysis
	err := s.Analysis.Inference(actx, func(c context.Context) error {
		images, err := s.Analysis.LoadImages(c, parts)
		if err != nil {
			return err
		}
		a, err = llm.Analyse(c, provider, modelID, images, m.Tags, promptOptions(m.Options), copts)
		return err
	})
	callsOrEmpty := func() []map[string]any {
		callsMu.Lock()
		defer callsMu.Unlock()
		if calls == nil {
			return []map[string]any{}
		}
		return append([]map[string]any(nil), calls...)
	}
	if err != nil {
		if llm.IsTimeout(err) {
			err = llm.NewTimeoutError()
		}
		metadata["calls"] = callsOrEmpty()
		metadata["elapsed_ms"] = float64(time.Since(started).Microseconds()) / 1000
		return nil, nil, &experimentFailure{msg: errKind(err) + ": " + err.Error(), metadata: metadata}
	}
	metadata["used_pipeline"] = "direct_vision"
	metadata["calls"] = callsOrEmpty()
	llmMS := 0.0
	for _, c := range calls {
		if v, ok := c["elapsed_ms"].(float64); ok {
			llmMS += v
		}
	}
	metadata["llm_ms"] = llmMS
	elapsed := float64(time.Since(started).Microseconds()) / 1000
	metadata["elapsed_ms"] = elapsed
	metadata["resolved_pipeline"] = item.Pipeline
	hashes := make([]string, len(item.Images))
	for i, img := range item.Images {
		hashes[i] = img.SHA256
	}
	metadata["image_hashes"] = hashes
	n := len(calls)
	if n == 0 {
		n = 1
	}
	return expected, &experimentResult{Receipt: a.Receipt, Attempts: n, Elapsed: elapsed, Metadata: metadata}, nil
}

func (s *Service) executeExperiment(ctx, rec context.Context, st *store.Store, task store.BenchmarkTask, raw json.RawMessage, deadline time.Duration) {
	started := time.Now()
	m, err := parseManifest(raw)
	if err == nil {
		rctx, cancel := context.WithTimeout(ctx, deadline)
		var expected *domain.Receipt
		var result *experimentResult
		expected, result, err = s.runExperiment(rctx, m, task.ReceiptID, task.ModelID)
		timedOut := rctx.Err() == context.DeadlineExceeded
		cancel()
		if err == nil {
			detail := DetailedScore(expected, result.Receipt, m.Options.IncludeClassification)
			payload, _ := json.Marshal(map[string]any{
				"version": detail.Version, "scores": detail.Scores, "differences": detail.Differences, "rows": detail.Rows, "checks": detail.Checks,
				"expected": Dump(expected), "actual": Dump(result.Receipt), "metadata": result.Metadata,
				"legacy_scores": ScoreReceipts(expected, result.Receipt),
			})
			score, _ := detail.Scores["normalized"].(float64)
			_ = st.FinishBenchmarkTask(rec, task, store.FinishBenchmark{Success: true, Score: score,
				Confidence: result.Receipt.Confidence, Attempts: result.Attempts, LatencyMS: result.Elapsed, Result: payload})
			return
		}
		if timedOut {
			err = errors.New("TimeoutError")
		}
	}
	metadata := map[string]any{}
	var ef *experimentFailure
	if errors.As(err, &ef) {
		metadata = ef.metadata
	} else if m != nil {
		metadata["options"] = m.Options
	}
	elapsed := float64(time.Since(started).Microseconds()) / 1000
	metadata["elapsed_ms"] = elapsed
	if m != nil {
		metadata["requested_pipeline"] = m.Options.Pipeline
	}
	attempts := 1
	if calls, ok := metadata["calls"].([]map[string]any); ok && len(calls) > 0 {
		attempts = len(calls)
	}
	msg := err.Error()
	if msg == "" {
		msg = "Error"
	}
	payload, _ := json.Marshal(map[string]any{"error": msg, "metadata": metadata})
	_ = st.FinishBenchmarkTask(rec, task, store.FinishBenchmark{Success: false, Attempts: attempts, LatencyMS: elapsed, Result: payload})
}

// Reevaluate transiently re-runs one receipt with one model; nothing is stored.
func (s *Service) Reevaluate(ctx context.Context, runID, receiptID, modelID string) (map[string]any, error) {
	st := s.Store()
	run, err := st.BenchmarkRun(ctx, runID)
	if err != nil {
		return nil, err
	}
	if run == nil || !contains(run.ReceiptIDs, receiptID) || !contains(run.ModelIDs, modelID) {
		return nil, userErr("Receipt/model is not part of this benchmark run")
	}
	raw, err := st.BenchmarkManifest(ctx, runID)
	if err != nil {
		return nil, err
	}
	digests, timeout, err := st.BenchmarkMeta(ctx, runID)
	if err != nil {
		return nil, err
	}
	deadline := time.Duration(timeout) * time.Second
	if raw != nil {
		models, err := s.Provider.Models(ctx)
		if err != nil {
			return nil, err
		}
		var current *llm.ModelInfo
		for i := range models {
			if models[i].ID == modelID {
				current = &models[i]
			}
		}
		if want := digests[modelID]; current == nil || (want != nil && *want != "" && current.Digest != nil && *current.Digest != "" && *want != *current.Digest) {
			return nil, userErr("Selected model is unavailable or its digest changed; create a new run")
		}
		m, err := parseManifest(raw)
		if err != nil {
			return nil, err
		}
		rctx, cancel := context.WithTimeout(ctx, deadline)
		defer cancel()
		expected, result, err := s.runExperiment(rctx, m, receiptID, modelID)
		if rctx.Err() == context.DeadlineExceeded {
			return nil, ErrTimeout
		}
		if err != nil {
			var ef *experimentFailure
			if errors.As(err, &ef) {
				return nil, &UserError{ef.msg}
			}
			return nil, asUserError(err)
		}
		detail := DetailedScore(expected, result.Receipt, m.Options.IncludeClassification)
		return map[string]any{
			"receipt_id": receiptID, "model_id": modelID, "version": detail.Version, "scores": detail.Scores,
			"differences": detail.Differences, "rows": detail.Rows, "checks": detail.Checks,
			"expected": Dump(expected), "actual": Dump(result.Receipt), "metadata": result.Metadata,
		}, nil
	}

	expected, details, err := s.expected(ctx, st, receiptID)
	if err != nil {
		return nil, err
	}
	if details.ImagePath == nil || !fileExists(*details.ImagePath) {
		return nil, userErr("Verified receipt image is unavailable")
	}
	images, err := s.Analysis.LoadImages(ctx, []domain.ImagePart{{ImagePath: *details.ImagePath}})
	if err != nil {
		return nil, err
	}
	a, err := llm.Analyse(ctx, s.Provider, modelID, images, nil, llm.PromptOptions{}, llm.ChatOptions{})
	if err != nil {
		return nil, err
	}
	ed, ad := toMap(Dump(expected)), toMap(Dump(a.Receipt))
	diff := map[string]any{}
	for k, v := range ed {
		if !jsonEqual(v, ad[k]) {
			diff[k] = map[string]any{"expected": v, "actual": ad[k]}
		}
	}
	return map[string]any{"receipt_id": receiptID, "model_id": modelID, "scores": ScoreReceipts(expected, a.Receipt),
		"expected": ed, "actual": ad, "diff": diff}, nil
}

func asUserError(err error) error {
	var ue *UserError
	if errors.As(err, &ue) || errors.Is(err, os.ErrNotExist) {
		return &UserError{err.Error()}
	}
	return &UserError{err.Error()}
}

func toMap(v any) map[string]any {
	b, _ := json.Marshal(v)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}
