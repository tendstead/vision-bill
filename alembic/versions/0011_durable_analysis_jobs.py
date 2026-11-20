"""Durable generation/lease queue; stop legacy workers before migrating."""
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for column in (
        "generation INTEGER NOT NULL DEFAULT 1 CHECK (generation >= 1)",
        "attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0)",
        "max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 3)",
        "retry_at TIMESTAMPTZ", "claim_token UUID", "lease_expires_at TIMESTAMPTZ",
        "source_manifest JSONB NOT NULL DEFAULT '[]'", "requested_options JSONB NOT NULL DEFAULT '{}'",
        "effective_options JSONB NOT NULL DEFAULT '{}'", "request_fingerprint TEXT",
        "idempotency_key TEXT", "result_code TEXT", "wait_reason TEXT",
        "draft_available BOOLEAN NOT NULL DEFAULT FALSE", "ocr_checkpoint JSONB",
        "legacy_events JSONB NOT NULL DEFAULT '[]'",
    ):
        op.execute(f"ALTER TABLE images ADD COLUMN {column}")
    op.execute("ALTER TABLE images DROP CONSTRAINT images_status_check")
    op.execute("""ALTER TABLE images ADD CONSTRAINT images_status_check CHECK
        (status IN ('pending','processing','analyzed','needs_review','failed','timed_out','unreadable'))""")
    # Preserve all receipt content, verified flags, NUMERICs and historical paths.
    op.execute("""UPDATE images i SET receipt_id = (
        SELECT r.id FROM receipts r WHERE r.image_id=i.id ORDER BY r.created_at, r.id LIMIT 1
        ) WHERE i.receipt_id IS NULL AND EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=i.id)""")
    op.execute("""UPDATE receipts r SET image_id=i.id FROM images i
        WHERE i.receipt_id=r.id AND r.image_id IS NULL AND NOT EXISTS
        (SELECT 1 FROM receipts other WHERE other.image_id=i.id)
        AND i.id=(SELECT ii.id FROM images ii WHERE ii.receipt_id=r.id ORDER BY ii.created_at,ii.id LIMIT 1)""")
    op.execute("""UPDATE images SET legacy_events=jsonb_build_array(jsonb_build_object(
        'status',status,'error',error,'receipt_count',
        (SELECT count(*) FROM receipts r WHERE r.image_id=images.id))),
        status='analyzed', processing_at=NULL WHERE receipt_id IS NOT NULL""")
    op.execute("""UPDATE images SET legacy_events=jsonb_build_array(jsonb_build_object(
        'status',status,'error',error,'processing_at',processing_at)),
        status='pending', processing_at=NULL, result_code='legacy_requeued'
        WHERE status IN ('failed','processing') AND receipt_id IS NULL
        AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id)""")
    op.execute("CREATE UNIQUE INDEX ux_images_owner_upload_key ON images(user_id,idempotency_key) WHERE idempotency_key IS NOT NULL")
    op.execute("CREATE INDEX ix_images_due ON images(retry_at,created_at,id) WHERE status IN ('pending','processing')")
    op.execute("""CREATE TABLE image_retry_requests (
        image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        request_key TEXT NOT NULL, expected_generation INTEGER NOT NULL,
        result JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY(image_id,request_key))""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM images WHERE request_fingerprint IS NOT NULL
        OR generation<>1 OR attempt_count<>0 OR status='needs_review'
        OR ocr_checkpoint IS NOT NULL OR source_manifest<>'[]'::jsonb
        OR legacy_events<>'[]'::jsonb)
        OR EXISTS (SELECT 1 FROM image_retry_requests) THEN
        RAISE EXCEPTION 'destructive workflow downgrade refused'; END IF; END $$""")
    op.execute("DROP TABLE image_retry_requests")
    op.execute("DROP INDEX ix_images_due")
    op.execute("DROP INDEX ux_images_owner_upload_key")
    op.execute("ALTER TABLE images DROP CONSTRAINT images_status_check")
    # Existing timed_out/unreadable meanings cannot be destroyed by a downgrade.
    op.execute("ALTER TABLE images ADD CONSTRAINT images_status_check CHECK (status IN ('pending','processing','analyzed','failed','timed_out','unreadable'))")
    for column in ("generation", "attempt_count", "max_attempts", "retry_at", "claim_token",
                   "lease_expires_at", "source_manifest", "requested_options", "effective_options",
                   "request_fingerprint", "idempotency_key", "result_code", "wait_reason",
                   "draft_available", "ocr_checkpoint", "legacy_events"):
        op.execute(f"ALTER TABLE images DROP COLUMN {column}")
