"""Explicit immutable benchmark references and honest nullable historical ownership."""
from alembic import op

revision = '0016'
down_revision = '0015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('ALTER TABLE taxes ADD COLUMN position INTEGER')
    op.execute("""WITH ordered AS (SELECT id,row_number() OVER(PARTITION BY receipt_id ORDER BY id)-1 AS pos FROM taxes) UPDATE taxes SET position=ordered.pos FROM ordered WHERE taxes.id=ordered.id""")
    op.execute('ALTER TABLE taxes ALTER COLUMN position SET NOT NULL')
    op.execute('ALTER TABLE taxes ALTER COLUMN position SET DEFAULT 0')
    op.execute('ALTER TABLE taxes ADD CONSTRAINT taxes_position_nonnegative CHECK(position>=0)')
    op.execute('ALTER TABLE benchmark_runs ADD COLUMN owner_id UUID REFERENCES users(id) ON DELETE SET NULL')
    op.execute('''CREATE TABLE benchmark_references (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        receipt_id UUID REFERENCES receipts(id) ON DELETE SET NULL,
        owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
        qualified_by UUID REFERENCES users(id) ON DELETE SET NULL,
        receipt_identity UUID NOT NULL, owner_identity UUID, actor_identity UUID NOT NULL,
        qualified_at TIMESTAMPTZ NOT NULL DEFAULT now(), review_revision INTEGER NOT NULL,
        document_sha256 TEXT NOT NULL, expected_sha256 TEXT NOT NULL,
        expected JSONB NOT NULL, scored_fields JSONB NOT NULL, sources JSONB NOT NULL,
        schema_version TEXT NOT NULL, scoring_version TEXT NOT NULL,
        findings JSONB NOT NULL, acknowledgements JSONB NOT NULL,
        UNIQUE(receipt_identity,review_revision,document_sha256))''')
    op.execute('''CREATE FUNCTION vision_bill_immutable_reference() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
        IF TG_OP='UPDATE' AND pg_trigger_depth()>1
          AND to_jsonb(NEW)-ARRAY['receipt_id','owner_id','qualified_by'] =
              to_jsonb(OLD)-ARRAY['receipt_id','owner_id','qualified_by']
          AND (NEW.receipt_id IS NOT DISTINCT FROM OLD.receipt_id OR
               (NEW.receipt_id IS NULL AND NOT EXISTS(SELECT 1 FROM receipts WHERE id=OLD.receipt_id)))
          AND (NEW.owner_id IS NOT DISTINCT FROM OLD.owner_id OR
               (NEW.owner_id IS NULL AND NOT EXISTS(SELECT 1 FROM users WHERE id=OLD.owner_id)))
          AND (NEW.qualified_by IS NOT DISTINCT FROM OLD.qualified_by OR
               (NEW.qualified_by IS NULL AND NOT EXISTS(SELECT 1 FROM users WHERE id=OLD.qualified_by)))
          THEN RETURN NEW;
        END IF;
        RAISE EXCEPTION 'immutable benchmark reference'; END $$''')
    op.execute('''CREATE TRIGGER immutable_reference BEFORE UPDATE OR DELETE ON benchmark_references
        FOR EACH ROW EXECUTE FUNCTION vision_bill_immutable_reference()''')


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS(SELECT 1 FROM (SELECT position,row_number() OVER(PARTITION BY receipt_id ORDER BY id)-1 AS old_position FROM taxes) ordered WHERE position<>old_position) THEN RAISE EXCEPTION 'destructive tax order downgrade refused'; END IF; END $$""")
    op.execute('''DO $$ BEGIN IF EXISTS(SELECT 1 FROM benchmark_references)
        OR EXISTS(SELECT 1 FROM benchmark_runs WHERE owner_id IS NOT NULL) THEN
        RAISE EXCEPTION 'destructive benchmark reference downgrade refused'; END IF; END $$''')
    op.execute('DROP TABLE benchmark_references')
    op.execute('DROP FUNCTION vision_bill_immutable_reference()')
    op.execute('ALTER TABLE benchmark_runs DROP COLUMN owner_id')
    op.execute('ALTER TABLE taxes DROP COLUMN position')
