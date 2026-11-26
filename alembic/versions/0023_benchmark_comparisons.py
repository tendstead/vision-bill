"""Group independently fenced benchmark runs into a paired comparison."""

from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE benchmark_runs
        ADD COLUMN comparison_id uuid,
        ADD COLUMN comparison_arm text,
        ADD CHECK ((comparison_id IS NULL AND comparison_arm IS NULL)
            OR (comparison_id IS NOT NULL AND comparison_arm IS NOT NULL
                AND comparison_arm IN ('direct_vision', 'ocr')))""")
    op.execute("""CREATE UNIQUE INDEX benchmark_runs_comparison_arm
        ON benchmark_runs (comparison_id, comparison_arm) WHERE comparison_id IS NOT NULL""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM benchmark_runs WHERE comparison_id IS NOT NULL)
        THEN RAISE EXCEPTION 'benchmark comparison downgrade refused'; END IF;
        END $$""")
    op.execute("ALTER TABLE benchmark_runs DROP COLUMN comparison_arm, DROP COLUMN comparison_id")
