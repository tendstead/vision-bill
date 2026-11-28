package domain

// BenchmarkRun is a durable benchmark request and its state.
type BenchmarkRun struct {
	ID                 string         `json:"id"`
	Status             string         `json:"status"`
	ModelIDs           []string       `json:"model_ids"`
	ReceiptIDs         []string       `json:"receipt_ids"`
	DatasetFingerprint string         `json:"dataset_fingerprint"`
	PromptVersion      string         `json:"prompt_version"`
	ScoringVersion     string         `json:"scoring_version"`
	CouncilPolicy      string         `json:"council_policy"`
	ApplyCouncilFlags  bool           `json:"apply_council_flags"`
	CreatedAt          Timestamp      `json:"created_at"`
	CompletedAt        *Timestamp     `json:"completed_at"`
	Experiment         map[string]any `json:"experiment"`
	ParentRunID        *string        `json:"parent_run_id"`
	Label              *string        `json:"label"`
	ComparisonID       *string        `json:"comparison_id"`
	ComparisonArm      *string        `json:"comparison_arm"`
	VariantIndex       *int           `json:"variant_index"`
}

// BenchmarkSummary aggregates one model's results within a run.
type BenchmarkSummary struct {
	ModelID           string   `json:"model_id"`
	ModelDigest       *string  `json:"model_digest"`
	Completed         int      `json:"completed"`
	Succeeded         int      `json:"succeeded"`
	Failed            int      `json:"failed"`
	AverageScore      *float64 `json:"average_score"`
	AverageConfidence *float64 `json:"average_confidence"`
	AverageAttempts   *float64 `json:"average_attempts"`
	AverageLatencyMS  *float64 `json:"average_latency_ms"`
	CouncilCandidates int      `json:"council_candidates"`
	CouncilFindings   int      `json:"council_findings"`
}

// BenchmarkStatus is live queue state plus per-model summaries.
type BenchmarkStatus struct {
	Run             BenchmarkRun       `json:"run"`
	Queued          int                `json:"queued"`
	Running         int                `json:"running"`
	WaitingForModel int                `json:"waiting_for_model"`
	Retrying        int                `json:"retrying"`
	Terminal        int                `json:"terminal"`
	Summaries       []BenchmarkSummary `json:"summaries"`
}
