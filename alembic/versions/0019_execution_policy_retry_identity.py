"""Persist explicit retry policy identity; stop/drain older workers before upgrade."""

from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE image_retry_requests ADD COLUMN request_options JSONB
        NOT NULL DEFAULT '{}'::jsonb CHECK (request_options IN
        ('{}'::jsonb,'{"policy":"preserve"}'::jsonb,'{"policy":"current"}'::jsonb))""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM image_retry_requests
        WHERE request_options->>'policy'='current') OR EXISTS (SELECT 1 FROM images,
        jsonb_array_elements(legacy_events) event WHERE event->>'event'='execution_policy_adopted')
        THEN RAISE EXCEPTION 'execution policy adoption downgrade refused'; END IF; END $$""")
    op.execute("ALTER TABLE image_retry_requests DROP COLUMN request_options")
