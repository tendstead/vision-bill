"""Frozen benchmark inputs and durable experiment diagnostics."""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE benchmark_runs ADD COLUMN experiment_manifest JSONB")
    op.execute("""CREATE TABLE benchmark_results (
        run_id UUID NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
        model_id TEXT NOT NULL,
        receipt_id UUID NOT NULL,
        success BOOLEAN NOT NULL,
        payload JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (run_id, model_id, receipt_id)
    )""")


def downgrade() -> None:
    op.execute("DROP TABLE benchmark_results")
    op.execute("ALTER TABLE benchmark_runs DROP COLUMN experiment_manifest")
