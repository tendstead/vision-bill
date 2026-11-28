-- Comparable benchmarks: a run can be derived from another run (rerun) and
-- runs can be grouped for side-by-side comparison. All columns are nullable
-- and additive, so the previous release keeps working on this schema.
--
-- comparison_id / comparison_arm follow the naming of the paired-comparison
-- design in PR #16 (David Hupperich); comparison_arm is a free-form label.
ALTER TABLE benchmark_runs
    ADD COLUMN parent_run_id uuid REFERENCES benchmark_runs(id) ON DELETE SET NULL,
    ADD COLUMN label text,
    ADD COLUMN comparison_id uuid,
    ADD COLUMN comparison_arm text,
    ADD COLUMN variant_index integer;

CREATE INDEX benchmark_runs_parent_run_id_idx ON benchmark_runs (parent_run_id) WHERE parent_run_id IS NOT NULL;
CREATE INDEX benchmark_runs_comparison_id_idx ON benchmark_runs (comparison_id) WHERE comparison_id IS NOT NULL;
