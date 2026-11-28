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
	"github.com/tendstead/vision-bill/internal/ids"
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
	// ProviderOptions are whitelisted per-request inference options (num_ctx,
	// num_gpu, seed, ...). They are part of the manifest, so runs show them
	// as varied settings.
	ProviderOptions map[string]any `json:"provider_options,omitempty"`
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
	Label               *string
	// Variants turn the request into a sweep: one run per variant, all with the
	// same frozen inputs and a shared comparison_id. Each variant's experiment
	// overrides Experiment.
	Variants []Variant
}

// Variant is one arm of a sweep.
type Variant struct {
	Label      *string
	Experiment ExperimentOverride
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

// ProviderOptionInfo lists the per-request inference options the configured
// provider accepts in experiments.
type ProviderOptionInfo struct {
	Provider string           `json:"provider"`
	Options  []llm.OptionSpec `json:"options"`
}

// ProviderOptions reports the whitelist of the configured provider.
func (s *Service) ProviderOptions() ProviderOptionInfo {
	return ProviderOptionInfo{Provider: s.Provider.Kind(), Options: llm.ProviderOptionSpecs(s.Provider.Kind())}
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

// resolveModels validates the requested model ids (all available models when
// none are requested) and returns them with the provider's model list.
func (s *Service) resolveModels(ctx context.Context, requested []string) ([]string, map[string]llm.ModelInfo, error) {
	models, err := s.Provider.Models(ctx)
	if err != nil {
		return nil, nil, err
	}
	available := map[string]llm.ModelInfo{}
	var order []string
	for _, m := range models {
		available[m.ID] = m
		order = append(order, m.ID)
	}
	selected := requested
	if len(selected) == 0 {
		selected = order
	}
	if len(selected) == 0 {
		return nil, nil, userErr("No vision models are available")
	}
	var unknown []string
	for _, id := range selected {
		if _, ok := available[id]; !ok {
			unknown = append(unknown, id)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return nil, nil, userErr("Selected model(s) unavailable: %s", strings.Join(dedupe(unknown), ", "))
	}
	return selected, available, nil
}

// prepareOptions validates the experiment against the provider and
// normalises its provider options.
func (s *Service) prepareOptions(o ExperimentOptions) (ExperimentOptions, error) {
	if o.Think != nil && !s.Provider.SupportsThinking() {
		return o, userErr("Thinking override is supported only by the Ollama provider")
	}
	if o.Pipeline == "ocr" {
		return o, userErr("OCR extension is not configured; choose normal upload or direct vision")
	}
	clean, err := llm.SanitizeProviderOptions(s.Provider.Kind(), o.ProviderOptions)
	if err != nil {
		return o, &UserError{err.Error()}
	}
	o.ProviderOptions = clean
	return o, nil
}

// CreateRun validates the request, freezes the inputs and queues all
// (model x receipt) tasks. With variants it creates one run per variant (a
// sweep sharing one comparison_id); otherwise exactly one run.
func (s *Service) CreateRun(ctx context.Context, req CreateRequest) ([]domain.BenchmarkRun, error) {
	st := s.Store()
	selected, available, err := s.resolveModels(ctx, req.ModelIDs)
	if err != nil {
		return nil, err
	}
	receiptIDs, err := s.selectReceipts(ctx, st, req)
	if err != nil {
		return nil, err
	}
	if len(receiptIDs) == 0 {
		return nil, userErr("No verified receipts match the selection")
	}
	type arm struct {
		options ExperimentOptions
		label   *string
		arm     *string
	}
	var arms []arm
	if len(req.Variants) == 0 {
		o, err := s.prepareOptions(req.Experiment)
		if err != nil {
			return nil, err
		}
		arms = []arm{{options: o, label: req.Label}}
	}
	for i, v := range req.Variants {
		o, err := s.prepareOptions(v.Experiment.Apply(req.Experiment))
		if err != nil {
			return nil, err
		}
		name := fmt.Sprintf("variant %d", i+1)
		if v.Label != nil {
			name = *v.Label
		}
		label := v.Label
		if label == nil && req.Label != nil {
			label = req.Label
		}
		arms = append(arms, arm{options: o, label: label, arm: &name})
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
	settings := store.BenchmarkRunSettings{TimeoutSeconds: req.TimeoutSeconds, Policy: req.CouncilPolicy,
		AbsoluteThreshold: req.AbsoluteThreshold, RelativeThreshold: req.RelativeThreshold, ApplyFlags: req.ApplyCouncilFlags}
	var comparison *string
	if len(req.Variants) > 0 {
		id := ids.UUID()
		comparison = &id
	}
	var news []store.NewBenchmarkRun
	for i, a := range arms {
		n, err := s.buildRun(ctx, st, queueParams{
			Options: a.options, Selected: selected, Available: available, ReceiptIDs: receiptIDs, Inputs: inputs, Expected: expected,
			Settings: settings, Label: a.label,
		})
		if err != nil {
			return nil, err
		}
		if comparison != nil {
			idx := i
			n.ComparisonID, n.ComparisonArm, n.VariantIndex = comparison, a.arm, &idx
		}
		news = append(news, n)
	}
	return st.CreateBenchmarkRuns(ctx, news)
}

// queueParams is a fully validated run with frozen inputs, ready to be stored.
type queueParams struct {
	Options    ExperimentOptions
	Selected   []string
	Available  map[string]llm.ModelInfo
	ReceiptIDs []string
	Inputs     []manifestInput
	Expected   []*domain.Receipt
	Settings   store.BenchmarkRunSettings
	Label      *string
	Parent     *string
}

// queueRun builds and stores one run.
func (s *Service) queueRun(ctx context.Context, st *store.Store, p queueParams) (*domain.BenchmarkRun, error) {
	n, err := s.buildRun(ctx, st, p)
	if err != nil {
		return nil, err
	}
	runs, err := st.CreateBenchmarkRuns(ctx, []store.NewBenchmarkRun{n})
	if err != nil {
		return nil, err
	}
	return &runs[0], nil
}

// buildRun records the manifest (current prompt, schema and endpoint) of a run.
func (s *Service) buildRun(ctx context.Context, st *store.Store, p queueParams) (store.NewBenchmarkRun, error) {
	var zero store.NewBenchmarkRun
	tags, err := st.ListTags(ctx)
	if err != nil {
		return zero, err
	}
	options := p.Options
	if options.Temperature == nil {
		t := s.Provider.Temperature()
		options.Temperature = &t
	}
	counts := map[int]bool{}
	for _, in := range p.Inputs {
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
	m := manifest{Engine: "go", Options: options, Tags: orEmpty(tags), Inputs: p.Inputs, Provider: s.Provider.Kind(),
		EndpointFingerprint: &endpoint, PromptHashes: hashes, ValidationSchemaSHA: llm.SchemaSHA256(), ThinkingDefault: thinking}
	raw, err := json.Marshal(m)
	if err != nil {
		return zero, err
	}
	digests := map[string]*string{}
	for _, id := range p.Selected {
		digests[id] = p.Available[id].Digest
	}
	return store.NewBenchmarkRun{
		ModelIDs: dedupe(p.Selected), ReceiptIDs: p.ReceiptIDs, Fingerprint: DatasetFingerprint(p.Expected), ModelDigests: digests,
		TimeoutSeconds: p.Settings.TimeoutSeconds, Policy: p.Settings.Policy, AbsoluteThreshold: p.Settings.AbsoluteThreshold,
		RelativeThreshold: p.Settings.RelativeThreshold, ApplyFlags: p.Settings.ApplyFlags,
		PromptVersion: PromptVersion, ScoringVersion: DetailScoringVersion, Manifest: raw,
		ParentRunID: p.Parent, Label: p.Label,
	}, nil
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
	issues, err := st.BenchmarkIssues(ctx, id)
	if err != nil {
		return nil, err
	}
	return &domain.BenchmarkStatus{Run: *run, Queued: q, Running: r, WaitingForModel: w, Retrying: rt, Terminal: term, Summaries: sums, Issues: issues}, nil
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
	retry := func(waiting bool, reason string) {
		if err := st.RetryBenchmarkTask(rec, task, waiting, s.now(), reason); err != nil {
			s.log().Error("could not schedule benchmark retry", "err", err)
		}
	}
	mctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	models, err := s.Provider.Models(mctx)
	cancel()
	if err != nil {
		retry(false, "The inference server could not be reached to list its models: "+err.Error())
		return
	}
	run, err := st.BenchmarkRun(rec, task.RunID)
	if err != nil || run == nil {
		return
	}
	digests, timeout, err := st.BenchmarkMeta(rec, task.RunID)
	if err != nil {
		retry(false, "Database error while loading the run: "+err.Error())
		return
	}
	var model *llm.ModelInfo
	for i := range models {
		if models[i].ID == task.ModelID {
			model = &models[i]
		}
	}
	if model == nil {
		retry(true, fmt.Sprintf("Model '%s' is not available on the inference server; the task waits until it is back", task.ModelID))
		return
	}
	if want := digests[task.ModelID]; want != nil && *want != "" && model.Digest != nil && *model.Digest != "" && *want != *model.Digest {
		retry(true, fmt.Sprintf("Model '%s' was updated since this run was created (digest differs); the task waits for the original version. Create a new run (rerun) to test the new version.", task.ModelID))
		return
	}
	raw, err := st.BenchmarkManifest(rec, task.RunID)
	if err != nil {
		retry(false, "Database error while loading the run inputs: "+err.Error())
		return
	}
	deadline := time.Duration(timeout) * time.Second
	if raw != nil {
		s.executeExperiment(ctx, rec, st, task, raw, deadline, retry)
		return
	}
	s.executeLegacy(ctx, rec, st, task, deadline, retry)
}

func (s *Service) executeLegacy(ctx, rec context.Context, st *store.Store, task store.BenchmarkTask, deadline time.Duration, retry func(bool, string)) {
	fail := func(reason string) {
		payload, _ := json.Marshal(map[string]any{"error": reason, "metadata": map[string]any{}})
		_ = st.FinishBenchmarkTask(rec, task, store.FinishBenchmark{Success: false, Attempts: 0, Result: payload})
	}
	expected, details, err := s.expected(rec, st, task.ReceiptID)
	if err != nil {
		var ue *UserError
		if errors.As(err, &ue) {
			fail(ue.Msg)
		} else {
			retry(false, "Database error while loading the receipt: "+err.Error())
		}
		return
	}
	if details.ImagePath == nil || !fileExists(*details.ImagePath) {
		fail("Receipt image is unavailable")
		return
	}
	// The deadline starts once an inference slot is held: waiting for a slot
	// must not be recorded as a model timeout.
	var a *llm.Analysis
	loadFailed := false
	err = s.Analysis.Inference(ctx, func(c context.Context) error {
		actx, cancel := context.WithTimeout(c, deadline)
		defer cancel()
		images, err := s.Analysis.LoadImages(actx, []domain.ImagePart{{ImagePath: *details.ImagePath}})
		if err != nil {
			loadFailed = true
			return err
		}
		a, err = llm.Analyse(actx, s.Provider, task.ModelID, images, nil, llm.PromptOptions{}, llm.ChatOptions{})
		return err
	})
	if loadFailed {
		fail("The receipt image could not be prepared for the model: " + err.Error())
		return
	}
	if err != nil {
		var unreadable *llm.UnreadableError
		var vf *llm.ValidationFailure
		switch {
		case errors.As(err, &unreadable), errors.As(err, &vf), strings.HasPrefix(err.Error(), "Failed to get a valid response"):
			fail(errKind(err) + ": " + err.Error())
		default:
			s.log().Warn("transient benchmark failure", "run", task.RunID, "model", task.ModelID, "receipt", task.ReceiptID, "err", err)
			retry(false, "Transient failure, retrying with backoff: "+err.Error())
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

// runInSlot runs one frozen-input extraction while holding an inference slot.
// The run's deadline starts after the slot is acquired so that waiting behind
// uploads is never recorded as a model timeout. waited is non-nil when ctx
// ended before a slot became free.
func (s *Service) runInSlot(ctx context.Context, deadline time.Duration, m *manifest, receiptID, modelID string) (expected *domain.Receipt, result *experimentResult, timedOut bool, err, waited error) {
	waited = s.Analysis.Inference(ctx, func(c context.Context) error {
		rctx, cancel := context.WithTimeout(c, deadline)
		defer cancel()
		expected, result, err = s.runExperiment(rctx, m, receiptID, modelID)
		timedOut = rctx.Err() == context.DeadlineExceeded
		return nil
	})
	return
}

// runExperiment verifies the frozen inputs still match and extracts again.
// The caller must hold an inference slot.
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
	copts := llm.ChatOptions{Think: m.Options.Think, Options: m.Options.ProviderOptions, Trace: func(t map[string]any) {
		callsMu.Lock()
		calls = append(calls, t)
		callsMu.Unlock()
	}}
	// The caller holds the inference slot (see runInSlot).
	actx, cancel := context.WithTimeout(ctx, provider.Timeout())
	defer cancel()
	var a *llm.Analysis
	err := func() error {
		images, err := s.Analysis.LoadImages(actx, parts)
		if err != nil {
			return err
		}
		a, err = llm.Analyse(actx, provider, modelID, images, m.Tags, promptOptions(m.Options), copts)
		return err
	}()
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

func (s *Service) executeExperiment(ctx, rec context.Context, st *store.Store, task store.BenchmarkTask, raw json.RawMessage, deadline time.Duration, retry func(bool, string)) {
	started := time.Now()
	m, err := parseManifest(raw)
	if err == nil {
		expected, result, timedOut, runErr, waited := s.runInSlot(ctx, deadline, m, task.ReceiptID, task.ModelID)
		if waited != nil {
			// Shutting down before a slot was free: nothing ran, try again later.
			retry(false, "Interrupted before an inference slot was free; will retry")
			return
		}
		err = runErr
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
			err = fmt.Errorf("TimeoutError: the model did not finish within the run timeout of %d s", int(deadline.Seconds()))
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
		expected, result, timedOut, err, waited := s.runInSlot(ctx, deadline, m, receiptID, modelID)
		if waited != nil {
			return nil, waited
		}
		if timedOut {
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
	var a *llm.Analysis
	err = s.Analysis.Inference(ctx, func(c context.Context) error {
		actx, cancel := context.WithTimeout(c, deadline)
		defer cancel()
		images, err := s.Analysis.LoadImages(actx, []domain.ImagePart{{ImagePath: *details.ImagePath}})
		if err != nil {
			return err
		}
		a, err = llm.Analyse(actx, s.Provider, modelID, images, nil, llm.PromptOptions{}, llm.ChatOptions{})
		return err
	})
	if err != nil {
		if llm.IsTimeout(err) {
			return nil, ErrTimeout
		}
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
