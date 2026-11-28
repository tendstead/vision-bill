package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/store"
)

// The rerun/compare design (parent/comparison columns, a run that reuses the
// frozen inputs of another run) follows the paired-comparison ideas of PR #16
// by David Hupperich (comparison_id / comparison_arm naming).

// ExperimentOverride holds the experiment options a rerun changes; absent
// fields keep the value of the parent run.
type ExperimentOverride struct {
	Pipeline              *string
	Temperature           *float64
	Think                 *bool
	ThinkSet              bool // an explicit null clears the parent's override
	IncludeClassification *bool
	AllowOCRFallback      *bool
	ProviderOptions       map[string]any
	ProviderOptionsSet    bool // replaces the parent's options; null or {} clears them
}

// Apply returns base with the overrides applied.
func (o ExperimentOverride) Apply(base ExperimentOptions) ExperimentOptions {
	if o.Pipeline != nil {
		base.Pipeline = *o.Pipeline
	}
	if o.Temperature != nil {
		base.Temperature = o.Temperature
	}
	if o.ThinkSet {
		base.Think = o.Think
	}
	if o.IncludeClassification != nil {
		base.IncludeClassification = *o.IncludeClassification
	}
	if o.AllowOCRFallback != nil {
		base.AllowOCRFallback = *o.AllowOCRFallback
	}
	if o.ProviderOptionsSet {
		base.ProviderOptions = o.ProviderOptions
	}
	return base
}

// RerunRequest is a validated rerun request.
type RerunRequest struct {
	ModelIDs   []string // empty: the parent's models
	Experiment ExperimentOverride
	Label      *string
}

// RerunChange reports a receipt whose live state differs from the frozen
// inputs the rerun reuses.
type RerunChange struct {
	ReceiptID string `json:"receipt_id"`
	Change    string `json:"change"` // deleted, unverified, expected_changed, image_changed, image_missing
	Detail    string `json:"detail"`
}

// RerunResult is the new run plus what changed since the parent was created.
type RerunResult struct {
	domain.BenchmarkRun
	Changes []RerunChange `json:"changes"`
}

// Rerun queues a new run that reuses the frozen receipt set and expected
// values of the parent (so the comparison stays fair) with optional
// overrides. Receipts that were deleted since are left out, changed ones are
// kept; both are reported in Changes.
func (s *Service) Rerun(ctx context.Context, parentID string, req RerunRequest) (*RerunResult, error) {
	st := s.Store()
	parent, err := st.BenchmarkRun(ctx, parentID)
	if err != nil {
		return nil, err
	}
	if parent == nil {
		return nil, ErrRunNotFound
	}
	raw, err := st.BenchmarkManifest(ctx, parentID)
	if err != nil {
		return nil, err
	}
	if raw == nil {
		return nil, userErr("This run has no frozen inputs and cannot be rerun")
	}
	m, err := parseManifest(raw)
	if err != nil {
		return nil, err
	}
	options, err := s.prepareOptions(req.Experiment.Apply(m.Options))
	if err != nil {
		return nil, err
	}
	requested := req.ModelIDs
	if len(requested) == 0 {
		requested = parent.ModelIDs
	}
	selected, available, err := s.resolveModels(ctx, requested)
	if err != nil {
		return nil, err
	}
	settings, err := st.BenchmarkSettings(ctx, parentID)
	if err != nil {
		return nil, err
	}

	changes := []RerunChange{}
	var inputs []manifestInput
	var expected []*domain.Receipt
	var receiptIDs []string
	for _, in := range m.Inputs {
		frozen, issues := domain.ParseReceipt(in.Expected, domain.Options{})
		if issues != nil {
			return nil, userErr("Frozen expected receipt is invalid: %s", issues.Error())
		}
		d, err := st.ReceiptForBenchmark(ctx, in.ReceiptID)
		if err != nil {
			return nil, err
		}
		if d == nil {
			changes = append(changes, RerunChange{in.ReceiptID, "deleted", "The receipt no longer exists and is left out of the rerun"})
			continue
		}
		if !d.Receipt.Verified {
			changes = append(changes, RerunChange{in.ReceiptID, "unverified", "The receipt is no longer verified; the frozen expected values are still used"})
		}
		if live, err := expectedFromDetails(d); err != nil || DatasetFingerprint([]*domain.Receipt{live}) != DatasetFingerprint([]*domain.Receipt{frozen}) {
			changes = append(changes, RerunChange{in.ReceiptID, "expected_changed", "The receipt was edited after the parent run; the frozen expected values are still used"})
		}
		for _, img := range in.Images {
			sum, err := fileSHA256(img.Path)
			switch {
			case err != nil:
				changes = append(changes, RerunChange{in.ReceiptID, "image_missing", "A frozen photo is unavailable; its tasks will fail"})
			case sum != img.SHA256:
				changes = append(changes, RerunChange{in.ReceiptID, "image_changed", "A frozen photo changed; its tasks will fail"})
			}
		}
		inputs = append(inputs, in)
		expected = append(expected, frozen)
		receiptIDs = append(receiptIDs, in.ReceiptID)
	}
	if len(receiptIDs) == 0 {
		return nil, userErr("None of the receipts of the parent run exist any more")
	}
	run, err := s.queueRun(ctx, st, queueParams{
		Options: options, Selected: selected, Available: available, ReceiptIDs: receiptIDs, Inputs: inputs, Expected: expected,
		Settings: settings, Label: req.Label, Parent: &parent.ID,
	})
	if err != nil {
		return nil, err
	}
	return &RerunResult{BenchmarkRun: *run, Changes: changes}, nil
}

// ── compare ──────────────────────────────────────────────────────────────

// CompareReason explains why runs are not directly comparable.
type CompareReason struct {
	Code    string   `json:"code"`
	Message string   `json:"message"`
	RunIDs  []string `json:"run_ids"`
	ModelID *string  `json:"model_id,omitempty"`
}

// Comparability is the result of the strict comparison rule: same dataset
// fingerprint, same model digest per model, same prompt/schema version, same
// scoring version and same council policy.
type Comparability struct {
	Comparable bool            `json:"comparable"`
	Reasons    []CompareReason `json:"reasons"`
}

// CompareDelta is a run/model aggregate against the same model in the baseline run.
type CompareDelta struct {
	ModelID          string   `json:"model_id"`
	AverageScore     *float64 `json:"average_score"`
	AverageLatencyMS *float64 `json:"average_latency_ms"`
	Failed           int      `json:"failed"`
}

// CompareRun is one run in a comparison.
type CompareRun struct {
	Run           domain.BenchmarkRun       `json:"run"`
	BaselineRunID *string                   `json:"baseline_run_id"`
	ModelDigests  map[string]*string        `json:"model_digests"`
	Summaries     []domain.BenchmarkSummary `json:"summaries"`
	Deltas        []CompareDelta            `json:"deltas"`
}

// CompareCell is one model's outcome for one receipt in one run.
type CompareCell struct {
	RunID        string         `json:"run_id"`
	ModelID      string         `json:"model_id"`
	ModelDigest  *string        `json:"model_digest"`
	Success      bool           `json:"success"`
	Score        *float64       `json:"score"`
	Scores       map[string]any `json:"scores"`
	ScoreDelta   *float64       `json:"score_delta"`
	LatencyMS    *float64       `json:"latency_ms"`
	LoadMS       *float64       `json:"load_ms"`       // model load time, when the server reports it
	GenerationMS *float64       `json:"generation_ms"` // prompt evaluation plus generation, when reported
	Attempts     int            `json:"attempts"`
	Error        *string        `json:"error"`
}

// CompareReceipt lists the cells of every run for one receipt.
type CompareReceipt struct {
	ReceiptID string        `json:"receipt_id"`
	Cells     []CompareCell `json:"cells"`
}

// VariedField is a setting that differs between the compared runs.
type VariedField struct {
	Field  string         `json:"field"`
	Values map[string]any `json:"values"` // run id -> value
}

// CompareResult is the response of the compare endpoint.
type CompareResult struct {
	BaselineRunID string           `json:"baseline_run_id"`
	Comparable    Comparability    `json:"comparable"`
	Varied        []VariedField    `json:"varied"`
	Runs          []CompareRun     `json:"runs"`
	Receipts      []CompareReceipt `json:"receipts"`
}

type compareInput struct {
	run      domain.BenchmarkRun
	settings store.BenchmarkRunSettings
	digests  map[string]*string
	manifest *manifest
	sums     []domain.BenchmarkSummary
	cells    map[string]store.BenchmarkCell // model\x00receipt
}

func cellKey(model, receipt string) string { return model + "\x00" + receipt }

func digestValue(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// Compare lines up two or more runs. The first run is the baseline unless a
// run's parent is part of the comparison; then the parent is its baseline.
func (s *Service) Compare(ctx context.Context, ids []string) (*CompareResult, error) {
	st := s.Store()
	ins := make([]*compareInput, 0, len(ids))
	byID := map[string]*compareInput{}
	for _, id := range ids {
		run, err := st.BenchmarkRun(ctx, id)
		if err != nil {
			return nil, err
		}
		if run == nil {
			return nil, fmt.Errorf("%w: %s", ErrRunNotFound, id)
		}
		in := &compareInput{run: *run, cells: map[string]store.BenchmarkCell{}}
		if in.settings, err = st.BenchmarkSettings(ctx, id); err != nil {
			return nil, err
		}
		if in.digests, _, err = st.BenchmarkMeta(ctx, id); err != nil {
			return nil, err
		}
		raw, err := st.BenchmarkManifest(ctx, id)
		if err != nil {
			return nil, err
		}
		if raw != nil {
			if in.manifest, err = parseManifest(raw); err != nil {
				return nil, err
			}
		}
		if in.sums, err = st.BenchmarkSummaries(ctx, id); err != nil {
			return nil, err
		}
		cells, err := st.BenchmarkCells(ctx, id)
		if err != nil {
			return nil, err
		}
		for _, c := range cells {
			in.cells[cellKey(c.ModelID, c.ReceiptID)] = c
		}
		ins = append(ins, in)
		byID[id] = in
	}
	baselineOf := func(in *compareInput) *compareInput {
		if p := in.run.ParentRunID; p != nil && *p != in.run.ID {
			if b := byID[*p]; b != nil {
				return b
			}
		}
		if in == ins[0] {
			return nil
		}
		return ins[0]
	}

	out := &CompareResult{BaselineRunID: ins[0].run.ID, Comparable: s.comparability(ins, baselineOf), Varied: variedFields(ins)}
	for _, in := range ins {
		cr := CompareRun{Run: in.run, ModelDigests: in.digests, Summaries: in.sums, Deltas: []CompareDelta{}}
		if b := baselineOf(in); b != nil {
			cr.BaselineRunID = &b.run.ID
			base := map[string]domain.BenchmarkSummary{}
			for _, bs := range b.sums {
				base[bs.ModelID] = bs
			}
			for _, sm := range in.sums {
				bs, ok := base[sm.ModelID]
				if !ok {
					continue
				}
				d := CompareDelta{ModelID: sm.ModelID, Failed: sm.Failed - bs.Failed}
				d.AverageScore = diffPtr(sm.AverageScore, bs.AverageScore)
				d.AverageLatencyMS = diffPtr(sm.AverageLatencyMS, bs.AverageLatencyMS)
				cr.Deltas = append(cr.Deltas, d)
			}
		}
		out.Runs = append(out.Runs, cr)
	}

	var order []string
	seen := map[string]bool{}
	for _, in := range ins {
		for _, r := range in.run.ReceiptIDs {
			if !seen[r] {
				seen[r] = true
				order = append(order, r)
			}
		}
	}
	out.Receipts = []CompareReceipt{}
	for _, rid := range order {
		row := CompareReceipt{ReceiptID: rid, Cells: []CompareCell{}}
		for _, in := range ins {
			for _, model := range in.run.ModelIDs {
				c, ok := in.cells[cellKey(model, rid)]
				if !ok {
					continue
				}
				cell := CompareCell{RunID: in.run.ID, ModelID: model, ModelDigest: in.digests[model], Success: c.Success, Scores: c.Scores,
					LatencyMS: c.ElapsedMS, LoadMS: c.LoadMS, GenerationMS: c.GenMS, Attempts: max(c.Calls, 1), Error: c.Error}
				if v, ok := c.Scores["normalized"].(float64); ok && c.Success {
					cell.Score = &v
				}
				if b := baselineOf(in); b != nil {
					if bc, ok := b.cells[cellKey(model, rid)]; ok && bc.Success {
						if bv, ok := bc.Scores["normalized"].(float64); ok {
							cell.ScoreDelta = diffPtr(cell.Score, &bv)
						}
					}
				}
				row.Cells = append(row.Cells, cell)
			}
		}
		out.Receipts = append(out.Receipts, row)
	}
	return out, nil
}

func diffPtr(a, b *float64) *float64 {
	if a == nil || b == nil {
		return nil
	}
	d := *a - *b
	return &d
}

func (s *Service) comparability(ins []*compareInput, baselineOf func(*compareInput) *compareInput) Comparability {
	type key struct{ code, model string }
	reasons := map[key]*CompareReason{}
	var keys []key
	add := func(code, model, msg, runID string) {
		k := key{code, model}
		r := reasons[k]
		if r == nil {
			r = &CompareReason{Code: code, Message: msg, RunIDs: []string{}}
			if model != "" {
				m := model
				r.ModelID = &m
			}
			reasons[k] = r
			keys = append(keys, k)
		}
		r.RunIDs = append(r.RunIDs, runID)
	}
	for _, in := range ins {
		id := in.run.ID
		if in.run.Status != "completed" {
			add("run_incomplete", "", "The run has not finished; its results are partial", id)
		}
		if in.manifest == nil {
			add("legacy_run", "", "The run predates frozen inputs: no stored per-receipt results and no prompt/schema record", id)
		}
		b := baselineOf(in)
		if b == nil {
			continue
		}
		if in.run.DatasetFingerprint != b.run.DatasetFingerprint {
			add("dataset_changed", "", "The expected receipts differ (other receipts, or the expected values were edited)", id)
		}
		if in.run.ScoringVersion != b.run.ScoringVersion {
			add("scoring_changed", "", "The scoring version differs", id)
		}
		if in.settings.Policy != b.settings.Policy || in.settings.ApplyFlags != b.settings.ApplyFlags ||
			digestValue(in.settings.AbsoluteThreshold) != digestValue(b.settings.AbsoluteThreshold) ||
			digestValue(in.settings.RelativeThreshold) != digestValue(b.settings.RelativeThreshold) {
			add("council_policy_changed", "", "The council policy differs", id)
		}
		if in.run.PromptVersion != b.run.PromptVersion {
			add("prompt_changed", "", "The prompt version differs", id)
		} else if in.manifest != nil && b.manifest != nil {
			for k, h := range in.manifest.PromptHashes {
				if bh, ok := b.manifest.PromptHashes[k]; ok && bh != h {
					add("prompt_changed", "", "The extraction prompt differs (different instructions or category list)", id)
					break
				}
			}
			if in.manifest.ValidationSchemaSHA != b.manifest.ValidationSchemaSHA {
				add("schema_changed", "", "The receipt validation schema differs", id)
			}
		}
		for model, d := range in.digests {
			bd, ok := b.digests[model]
			if ok && digestValue(d) != digestValue(bd) {
				add("model_digest_changed", model, "The model was updated: its digest differs from the baseline", id)
			}
		}
	}
	res := Comparability{Comparable: len(keys) == 0, Reasons: []CompareReason{}}
	for _, k := range keys {
		res.Reasons = append(res.Reasons, *reasons[k])
	}
	return res
}

// variedFields lists the settings that are not identical across the runs; they
// are what the comparison measures, not a reason for incomparability.
func variedFields(ins []*compareInput) []VariedField {
	fields := []string{"label", "pipeline", "temperature", "think", "include_classification", "allow_ocr_fallback", "provider_options", "provider", "endpoint_fingerprint"}
	values := func(in *compareInput) map[string]any {
		v := map[string]any{"label": in.run.Label}
		if m := in.manifest; m != nil {
			v["pipeline"], v["temperature"], v["think"] = m.Options.Pipeline, m.Options.Temperature, m.Options.Think
			v["include_classification"], v["allow_ocr_fallback"] = m.Options.IncludeClassification, m.Options.AllowOCRFallback
			if len(m.Options.ProviderOptions) > 0 {
				v["provider_options"] = m.Options.ProviderOptions
			}
			v["provider"], v["endpoint_fingerprint"] = m.Provider, m.EndpointFingerprint
		}
		return v
	}
	per := make([]map[string]any, len(ins))
	for i, in := range ins {
		// Normalise through JSON so pointers compare by value.
		b, _ := json.Marshal(values(in))
		_ = json.Unmarshal(b, &per[i])
	}
	out := []VariedField{}
	for _, f := range fields {
		differs := false
		for i := 1; i < len(ins); i++ {
			if !reflect.DeepEqual(per[i][f], per[0][f]) {
				differs = true
			}
		}
		if !differs {
			continue
		}
		vf := VariedField{Field: f, Values: map[string]any{}}
		for i, in := range ins {
			vf.Values[in.run.ID] = per[i][f]
		}
		out = append(out, vf)
	}
	return out
}
