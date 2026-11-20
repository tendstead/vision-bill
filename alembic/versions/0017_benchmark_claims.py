"""Fenced reservations and explicitly post-upgrade inference accounting.

Stop/drain ALL old benchmark workers before upgrading: their PK-only writes
cannot obey the new fence. `attempts` is preserved historical control-poll
evidence, not a known inference count. Zero in the new counter means zero
transitions recorded under policy 0017. An unfinished legacy task may receive
up to three further tracked attempts; its lifetime inference baseline is unknown.
Completed reports, manifests and scoring are deliberately untouched.
"""

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE benchmark_tasks ADD COLUMN claim_token UUID")
    op.execute("""ALTER TABLE benchmark_tasks ADD COLUMN inference_attempt_count
        INTEGER NOT NULL DEFAULT 0 CHECK(inference_attempt_count BETWEEN 0 AND 3)""")
    op.execute("ALTER TABLE benchmark_tasks ADD COLUMN inference_started BOOLEAN NOT NULL DEFAULT FALSE")
    op.execute("""UPDATE benchmark_tasks SET status='retrying', leased_until=NULL,
        retry_at=clock_timestamp() WHERE status='running'""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS(SELECT 1 FROM benchmark_tasks WHERE
        claim_token IS NOT NULL OR inference_attempt_count>0 OR inference_started) THEN
        RAISE EXCEPTION 'destructive benchmark claim downgrade refused'; END IF; END $$""")
    op.execute("ALTER TABLE benchmark_tasks DROP COLUMN inference_started")
    op.execute("ALTER TABLE benchmark_tasks DROP COLUMN inference_attempt_count")
    op.execute("ALTER TABLE benchmark_tasks DROP COLUMN claim_token")
