package store

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/tendstead/vision-bill/internal/domain"
)

// NewBenchmarkRun holds the values of a run to create.
type NewBenchmarkRun struct {
	ModelIDs          []string
	ReceiptIDs        []string
	Fingerprint       string
	ModelDigests      map[string]*string
	TimeoutSeconds    int
	Policy            string
	AbsoluteThreshold *string
	RelativeThreshold *string
	ApplyFlags        bool
	PromptVersion     string
	ScoringVersion    string
	Manifest          json.RawMessage // nil for legacy runs
	ParentRunID       *string
	Label             *string
	ComparisonID      *string
	ComparisonArm     *string
	VariantIndex      *int
}

const runCols = `id::text, status, model_ids, receipt_ids, dataset_fingerprint, prompt_version, scoring_version,
	council_policy, apply_council_flags, created_at, completed_at, experiment_manifest,
	parent_run_id::text, label, comparison_id::text, comparison_arm, variant_index`

func scanRun(row pgx.Row) (*domain.BenchmarkRun, error) {
	var r domain.BenchmarkRun
	var models, receipts, manifest []byte
	var created time.Time
	var completed *time.Time
	if err := row.Scan(&r.ID, &r.Status, &models, &receipts, &r.DatasetFingerprint, &r.PromptVersion, &r.ScoringVersion,
		&r.CouncilPolicy, &r.ApplyCouncilFlags, &created, &completed, &manifest,
		&r.ParentRunID, &r.Label, &r.ComparisonID, &r.ComparisonArm, &r.VariantIndex); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	if err := json.Unmarshal(models, &r.ModelIDs); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(receipts, &r.ReceiptIDs); err != nil {
		return nil, err
	}
	r.CreatedAt = domain.Timestamp{Time: created}
	r.CompletedAt = ts(completed)
	if len(manifest) > 0 && string(manifest) != "null" {
		var m map[string]any
		if err := json.Unmarshal(manifest, &m); err != nil {
			return nil, err
		}
		delete(m, "inputs")
		delete(m, "tags")
		r.Experiment = m
	}
	return &r, nil
}

// CreateBenchmarkRun inserts one run with its tasks and summaries atomically.
func (s *Store) CreateBenchmarkRun(ctx context.Context, n NewBenchmarkRun) (*domain.BenchmarkRun, error) {
	runs, err := s.CreateBenchmarkRuns(ctx, []NewBenchmarkRun{n})
	if err != nil {
		return nil, err
	}
	return &runs[0], nil
}

// CreateBenchmarkRuns inserts several runs (a sweep) in one transaction so a
// sweep is either fully queued or not at all.
func (s *Store) CreateBenchmarkRuns(ctx context.Context, news []NewBenchmarkRun) ([]domain.BenchmarkRun, error) {
	out := []domain.BenchmarkRun{}
	err := s.inTx(ctx, func(tx pgx.Tx) error {
		for _, n := range news {
			r, err := insertBenchmarkRun(ctx, tx, n)
			if err != nil {
				return err
			}
			out = append(out, *r)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func insertBenchmarkRun(ctx context.Context, tx pgx.Tx, n NewBenchmarkRun) (*domain.BenchmarkRun, error) {
	models, _ := json.Marshal(n.ModelIDs)
	receipts, _ := json.Marshal(n.ReceiptIDs)
	digests, _ := json.Marshal(n.ModelDigests)
	var manifest *string
	if n.Manifest != nil {
		m := string(n.Manifest)
		manifest = &m
	}
	var id string
	if err := tx.QueryRow(ctx, `INSERT INTO benchmark_runs
		(model_ids, receipt_ids, dataset_fingerprint, model_digests, request_timeout_seconds, council_policy,
		 council_absolute_threshold, council_relative_threshold, apply_council_flags, prompt_version, scoring_version, experiment_manifest,
		 parent_run_id, label, comparison_id, comparison_arm, variant_index)
		VALUES ($1::jsonb,$2::jsonb,$3,$4::jsonb,$5,$6,$7::numeric,$8::numeric,$9,$10,$11,$12::jsonb,
		        $13::uuid,$14,$15::uuid,$16,$17) RETURNING id::text`,
		string(models), string(receipts), n.Fingerprint, string(digests), n.TimeoutSeconds, n.Policy,
		n.AbsoluteThreshold, n.RelativeThreshold, n.ApplyFlags, n.PromptVersion, n.ScoringVersion, manifest,
		n.ParentRunID, n.Label, n.ComparisonID, n.ComparisonArm, n.VariantIndex).Scan(&id); err != nil {
		return nil, err
	}
	for _, m := range n.ModelIDs {
		for _, r := range n.ReceiptIDs {
			if _, err := tx.Exec(ctx, `INSERT INTO benchmark_tasks (run_id, model_id, receipt_id) VALUES ($1::uuid,$2,$3::uuid)`, id, m, r); err != nil {
				return nil, err
			}
		}
		if _, err := tx.Exec(ctx, `INSERT INTO benchmark_summaries (run_id, model_id, model_digest) VALUES ($1::uuid,$2,$3)`, id, m, n.ModelDigests[m]); err != nil {
			return nil, err
		}
	}
	return scanRun(tx.QueryRow(ctx, `SELECT `+runCols+` FROM benchmark_runs WHERE id = $1::uuid`, id))
}

// BenchmarkRun returns one run, or nil.
func (s *Store) BenchmarkRun(ctx context.Context, id string) (*domain.BenchmarkRun, error) {
	return scanRun(s.Pool.QueryRow(ctx, `SELECT `+runCols+` FROM benchmark_runs WHERE id = $1::uuid`, id))
}

// BenchmarkRuns lists runs, newest first, optionally only one comparison group.
func (s *Store) BenchmarkRuns(ctx context.Context, comparisonID *string) ([]domain.BenchmarkRun, error) {
	rows, err := s.Pool.Query(ctx, `SELECT `+runCols+` FROM benchmark_runs WHERE ($1::uuid IS NULL OR comparison_id = $1::uuid)
		ORDER BY created_at DESC, variant_index DESC NULLS LAST, id DESC`, comparisonID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.BenchmarkRun{}
	for rows.Next() {
		r, err := scanRun(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *r)
	}
	return out, rows.Err()
}

// BenchmarkMeta returns the recorded model digests and request timeout.
func (s *Store) BenchmarkMeta(ctx context.Context, id string) (digests map[string]*string, timeoutSeconds int, err error) {
	var raw []byte
	if err = s.Pool.QueryRow(ctx, `SELECT model_digests, request_timeout_seconds FROM benchmark_runs WHERE id = $1::uuid`, id).Scan(&raw, &timeoutSeconds); err != nil {
		return nil, 0, err
	}
	digests = map[string]*string{}
	if len(raw) > 0 {
		err = json.Unmarshal(raw, &digests)
	}
	return digests, timeoutSeconds, err
}

// BenchmarkManifest returns the frozen inputs of a run, or nil for legacy runs.
func (s *Store) BenchmarkManifest(ctx context.Context, id string) (json.RawMessage, error) {
	var raw []byte
	if err := s.Pool.QueryRow(ctx, `SELECT experiment_manifest FROM benchmark_runs WHERE id = $1::uuid`, id).Scan(&raw); err != nil {
		return nil, err
	}
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	return raw, nil
}

// TaskCounts reports the queue state of a run.
func (s *Store) TaskCounts(ctx context.Context, id string) (queued, running, waiting, retrying, terminal int, err error) {
	rows, err := s.Pool.Query(ctx, `SELECT status, count(*)::int FROM benchmark_tasks WHERE run_id = $1::uuid GROUP BY status`, id)
	if err != nil {
		return
	}
	for rows.Next() {
		var status string
		var n int
		if err = rows.Scan(&status, &n); err != nil {
			rows.Close()
			return
		}
		switch status {
		case "queued":
			queued = n
		case "running":
			running = n
		case "waiting_for_model":
			waiting = n
		case "retrying":
			retrying = n
		}
	}
	rows.Close()
	if err = rows.Err(); err != nil {
		return
	}
	err = s.Pool.QueryRow(ctx, `SELECT COALESCE(sum(completed), 0)::int FROM benchmark_summaries WHERE run_id = $1::uuid`, id).Scan(&terminal)
	return
}

// BenchmarkSummaries returns the per-model aggregates of a run.
func (s *Store) BenchmarkSummaries(ctx context.Context, id string) ([]domain.BenchmarkSummary, error) {
	rows, err := s.Pool.Query(ctx, `SELECT model_id, model_digest, completed, succeeded, failed, total_score, total_confidence,
		total_attempts, total_latency_ms, council_candidates, council_findings
		FROM benchmark_summaries WHERE run_id = $1::uuid ORDER BY model_id`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.BenchmarkSummary{}
	for rows.Next() {
		var m domain.BenchmarkSummary
		var score, conf, latency float64
		var attempts int
		if err := rows.Scan(&m.ModelID, &m.ModelDigest, &m.Completed, &m.Succeeded, &m.Failed, &score, &conf, &attempts, &latency, &m.CouncilCandidates, &m.CouncilFindings); err != nil {
			return nil, err
		}
		if m.Succeeded > 0 {
			a, b := score/float64(m.Succeeded), conf/float64(m.Succeeded)
			m.AverageScore, m.AverageConfidence = &a, &b
		}
		if m.Completed > 0 {
			a, b := float64(attempts)/float64(m.Completed), latency/float64(m.Completed)
			m.AverageAttempts, m.AverageLatencyMS = &a, &b
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// BenchmarkTask is a leased unit of work.
type BenchmarkTask struct {
	RunID, ModelID, ReceiptID string
	Attempts                  int // attempts before this lease
}

// LeaseBenchmarkTask claims the next runnable task for six minutes.
func (s *Store) LeaseBenchmarkTask(ctx context.Context, now time.Time) (*BenchmarkTask, error) {
	var t *BenchmarkTask
	err := s.inTx(ctx, func(tx pgx.Tx) error {
		var task BenchmarkTask
		// Model-major order: a comparison group (a sweep; a plain run is its own
		// group) first-come-first-served, then models in the order they were
		// requested, then the variants of a sweep, then receipts. The inference
		// server therefore loads each model once per group instead of swapping
		// models for every receipt.
		err := tx.QueryRow(ctx, `SELECT t.run_id::text, t.model_id, t.receipt_id::text, t.attempts
			FROM benchmark_tasks t JOIN benchmark_runs r ON r.id = t.run_id
			WHERE (t.status IN ('queued','retrying','waiting_for_model') AND (t.retry_at IS NULL OR t.retry_at <= $1))
			   OR (t.status = 'running' AND t.leased_until < $1)
			ORDER BY (SELECT min(g.created_at) FROM benchmark_runs g WHERE COALESCE(g.comparison_id, g.id) = COALESCE(r.comparison_id, r.id)),
			         COALESCE(r.comparison_id, r.id),
			         (SELECT e.pos FROM jsonb_array_elements_text(r.model_ids) WITH ORDINALITY AS e(v, pos) WHERE e.v = t.model_id),
			         COALESCE(r.variant_index, 0), r.created_at, t.run_id,
			         t.receipt_id
			FOR UPDATE OF t SKIP LOCKED LIMIT 1`, now).Scan(&task.RunID, &task.ModelID, &task.ReceiptID, &task.Attempts)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil
		}
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE benchmark_tasks SET status='running', attempts=attempts+1, started_at=$2, leased_until=$3
			WHERE run_id=$1::uuid AND model_id=$4 AND receipt_id=$5::uuid`, task.RunID, now, now.Add(6*time.Minute), task.ModelID, task.ReceiptID); err != nil {
			return err
		}
		t = &task
		return nil
	})
	return t, err
}

// RetryBenchmarkTask releases a task for a later attempt with capped backoff.
func (s *Store) RetryBenchmarkTask(ctx context.Context, t BenchmarkTask, waiting bool, now time.Time) error {
	exp := t.Attempts
	if exp < 0 {
		exp = 0
	}
	delay := 300
	if exp < 6 {
		delay = min(5*(1<<exp), 300)
	}
	status := "retrying"
	if waiting {
		status = "waiting_for_model"
	}
	_, err := s.Pool.Exec(ctx, `UPDATE benchmark_tasks SET status=$1, leased_until=NULL, retry_at=$2
		WHERE run_id=$3::uuid AND model_id=$4 AND receipt_id=$5::uuid`, status, now.Add(time.Duration(delay)*time.Second), t.RunID, t.ModelID, t.ReceiptID)
	return err
}

// FinishBenchmark carries a task outcome.
type FinishBenchmark struct {
	Success    bool
	Score      float64
	Confidence int
	Attempts   int
	LatencyMS  float64
	Result     json.RawMessage // optional payload
}

// FinishBenchmarkTask records a terminal outcome exactly once and completes
// the run when its queue is empty.
func (s *Store) FinishBenchmarkTask(ctx context.Context, t BenchmarkTask, f FinishBenchmark) error {
	return s.inTx(ctx, func(tx pgx.Tx) error {
		tag, err := tx.Exec(ctx, `DELETE FROM benchmark_tasks WHERE run_id=$1::uuid AND model_id=$2 AND receipt_id=$3::uuid`, t.RunID, t.ModelID, t.ReceiptID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return nil // a recovered lease must not count the same task twice
		}
		if f.Result != nil {
			if _, err := tx.Exec(ctx, `INSERT INTO benchmark_results (run_id, model_id, receipt_id, success, payload)
				VALUES ($1::uuid,$2,$3::uuid,$4,$5::jsonb) ON CONFLICT (run_id, model_id, receipt_id) DO NOTHING`,
				t.RunID, t.ModelID, t.ReceiptID, f.Success, string(f.Result)); err != nil {
				return err
			}
		}
		succeeded, failed := 0, 1
		if f.Success {
			succeeded, failed = 1, 0
		}
		if _, err := tx.Exec(ctx, `UPDATE benchmark_summaries SET completed=completed+1, succeeded=succeeded+$1, failed=failed+$2,
			total_score=total_score+$3, total_confidence=total_confidence+$4, total_attempts=total_attempts+$5, total_latency_ms=total_latency_ms+$6
			WHERE run_id=$7::uuid AND model_id=$8`, succeeded, failed, f.Score, f.Confidence, f.Attempts, f.LatencyMS, t.RunID, t.ModelID); err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE benchmark_runs SET status='completed', completed_at=now()
			WHERE id=$1::uuid AND NOT EXISTS (SELECT 1 FROM benchmark_tasks WHERE run_id=$1::uuid)`, t.RunID)
		return err
	})
}

// BenchmarkResults pages the stored per-task results of a run.
func (s *Store) BenchmarkResults(ctx context.Context, id string, limit, offset int) ([]map[string]any, error) {
	rows, err := s.Pool.Query(ctx, `SELECT model_id, receipt_id::text, success, payload FROM benchmark_results
		WHERE run_id = $1::uuid ORDER BY model_id, receipt_id LIMIT $2 OFFSET $3`, id, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var model, receipt string
		var success bool
		var payload []byte
		if err := rows.Scan(&model, &receipt, &success, &payload); err != nil {
			return nil, err
		}
		item := map[string]any{}
		if len(payload) > 0 {
			if err := json.Unmarshal(payload, &item); err != nil {
				return nil, err
			}
		}
		item["model_id"], item["receipt_id"], item["success"] = model, receipt, success
		out = append(out, item)
	}
	return out, rows.Err()
}

// ReceiptForBenchmark loads a verified receipt (any owner) as the expected
// extraction together with the path of its first photo.
func (s *Store) ReceiptForBenchmark(ctx context.Context, receiptID string) (*domain.ReceiptWithDetails, error) {
	return s.ReceiptDetails(ctx, receiptID, Scope{CanSeeAll: true})
}

// BenchmarkRunSettings are the run-level settings a rerun inherits.
type BenchmarkRunSettings struct {
	TimeoutSeconds    int
	Policy            string
	AbsoluteThreshold *string
	RelativeThreshold *string
	ApplyFlags        bool
}

// BenchmarkSettings returns the timeout and council settings of a run.
func (s *Store) BenchmarkSettings(ctx context.Context, id string) (BenchmarkRunSettings, error) {
	var b BenchmarkRunSettings
	err := s.Pool.QueryRow(ctx, `SELECT request_timeout_seconds, council_policy, council_absolute_threshold::text,
		council_relative_threshold::text, apply_council_flags FROM benchmark_runs WHERE id = $1::uuid`, id).
		Scan(&b.TimeoutSeconds, &b.Policy, &b.AbsoluteThreshold, &b.RelativeThreshold, &b.ApplyFlags)
	return b, err
}

// BenchmarkCell is the compact outcome of one (model, receipt) task, without
// the bulky expected/actual payloads.
type BenchmarkCell struct {
	ModelID   string
	ReceiptID string
	Success   bool
	Scores    map[string]any
	ElapsedMS *float64
	Calls     int
	Error     *string
}

// BenchmarkCells returns every stored task outcome of a run.
func (s *Store) BenchmarkCells(ctx context.Context, id string) ([]BenchmarkCell, error) {
	rows, err := s.Pool.Query(ctx, `SELECT model_id, receipt_id::text, success, payload->'scores',
		(payload->'metadata'->>'elapsed_ms')::float8,
		CASE WHEN jsonb_typeof(payload->'metadata'->'calls') = 'array' THEN jsonb_array_length(payload->'metadata'->'calls') ELSE 0 END,
		payload->>'error'
		FROM benchmark_results WHERE run_id = $1::uuid ORDER BY model_id, receipt_id`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []BenchmarkCell{}
	for rows.Next() {
		var c BenchmarkCell
		var scores []byte
		if err := rows.Scan(&c.ModelID, &c.ReceiptID, &c.Success, &scores, &c.ElapsedMS, &c.Calls, &c.Error); err != nil {
			return nil, err
		}
		if len(scores) > 0 && string(scores) != "null" {
			if err := json.Unmarshal(scores, &c.Scores); err != nil {
				return nil, err
			}
		}
		out = append(out, c)
	}
	return out, rows.Err()
}
