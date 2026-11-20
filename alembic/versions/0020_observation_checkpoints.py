"""Persist complete observation metadata and distinguish claim lease purposes."""

from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE images
        ADD COLUMN observation_checkpoint JSONB CHECK (jsonb_typeof(observation_checkpoint)='object'),
        ADD COLUMN claim_purpose TEXT CHECK (claim_purpose IN ('inference','completion'))""")
    # Old workers must be stopped. Their leases all represent inference attempts.
    op.execute("UPDATE images SET claim_purpose='inference' WHERE status='processing'")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (
        SELECT 1 FROM images WHERE observation_checkpoint IS NOT NULL
        OR claim_purpose='completion'
        OR legacy_events @> '[{"event":"observation_checkpoint_retired"}]'::jsonb
        ) THEN RAISE EXCEPTION 'observation checkpoint downgrade refused'; END IF; END $$""")
    op.execute("ALTER TABLE images DROP COLUMN claim_purpose, DROP COLUMN observation_checkpoint")
