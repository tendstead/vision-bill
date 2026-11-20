"""Immutable extraction attempts, ordered sources, saved views and mutable drafts."""
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE image_source_parts (
        part_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        photo_index INTEGER NOT NULL CHECK(photo_index>=0), source_sha256 TEXT NOT NULL
        CHECK(source_sha256 ~ '^[0-9a-f]{64}$'), image_path TEXT NOT NULL,
        metadata JSONB NOT NULL, UNIQUE(image_id,photo_index))""")
    op.execute("""CREATE TABLE extraction_working_views (
        view_id UUID PRIMARY KEY, image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        part_id UUID NOT NULL REFERENCES image_source_parts(part_id) ON DELETE CASCADE,
        artifact_path TEXT NOT NULL, metadata JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
    op.execute("""CREATE TABLE receipt_extractions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        generation INTEGER NOT NULL, attempt_number INTEGER NOT NULL,
        outcome_kind TEXT NOT NULL CHECK(outcome_kind IN ('complete','partial','retry','terminal')),
        result_code TEXT, envelope JSONB, review JSONB, source_views JSONB NOT NULL DEFAULT '[]',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(image_id,generation,attempt_number))""")
    op.execute("""CREATE TABLE receipt_drafts (
        image_id UUID PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
        extraction_id UUID NOT NULL REFERENCES receipt_extractions(id), generation INTEGER NOT NULL,
        revision INTEGER NOT NULL DEFAULT 1, document JSONB NOT NULL, amount_policy JSONB NOT NULL,
        findings JSONB NOT NULL, field_sources JSONB NOT NULL, human_edited BOOLEAN NOT NULL DEFAULT FALSE,
        finalized_receipt_id UUID REFERENCES receipts(id), created_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
    op.execute("""CREATE FUNCTION vision_bill_immutable_artifact() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'immutable extraction/review artifact'; END $$""")
    for table in ("receipt_extractions", "extraction_working_views"):
        op.execute(f"CREATE TRIGGER immutable_artifact BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION vision_bill_immutable_artifact()")
    op.execute("""CREATE FUNCTION vision_bill_artifact_no_delete() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN IF pg_trigger_depth()=1 THEN RAISE EXCEPTION 'immutable final artifact';
        END IF; RETURN OLD; END $$""")
    for table in ("receipt_extractions", "extraction_working_views"):
        op.execute(f"CREATE TRIGGER immutable_artifact_delete BEFORE DELETE ON {table} FOR EACH ROW EXECUTE FUNCTION vision_bill_artifact_no_delete()")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM receipt_extractions)
        OR EXISTS (SELECT 1 FROM receipt_drafts) OR EXISTS (SELECT 1 FROM image_source_parts)
        OR EXISTS (SELECT 1 FROM extraction_working_views) THEN
        RAISE EXCEPTION 'destructive extraction downgrade refused'; END IF; END $$""")
    for table in ("receipt_drafts", "receipt_extractions", "extraction_working_views", "image_source_parts"):
        op.execute(f"DROP TABLE {table}")
    op.execute("DROP FUNCTION vision_bill_immutable_artifact()")
    op.execute("DROP FUNCTION vision_bill_artifact_no_delete()")
