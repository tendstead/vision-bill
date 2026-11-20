"""Receipt review metadata and append-only operation history."""
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for column in (
        "extraction_id UUID REFERENCES receipt_extractions(id) ON DELETE SET NULL",
        "operation_revision INTEGER NOT NULL DEFAULT 1 CHECK(operation_revision>=1)",
        "review_revision INTEGER NOT NULL DEFAULT 1 CHECK(review_revision>=1)",
        "amount_policy JSONB NOT NULL DEFAULT '{}'", "findings JSONB NOT NULL DEFAULT '[]'",
        "field_sources JSONB NOT NULL DEFAULT '{}'", "checks_version TEXT", "verification_source TEXT",
    ):
        op.execute(f"ALTER TABLE receipts ADD COLUMN {column}")
    op.execute("UPDATE receipts SET verification_source='legacy' WHERE verified=TRUE OR status='verified'")
    op.execute("""CREATE TABLE receipt_review_revisions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        receipt_id UUID NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
        operation_revision INTEGER NOT NULL, review_revision INTEGER NOT NULL,
        event_kind TEXT NOT NULL, actor_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
        snapshot JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE(receipt_id,operation_revision))""")
    op.execute("""CREATE FUNCTION vision_bill_immutable_review() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
        -- Only the actor FK's referential action may null its deleted parent.
        -- Every other column, including the original snapshot, remains immutable.
        IF pg_trigger_depth()>1 AND OLD.actor_user_id IS NOT NULL AND NEW.actor_user_id IS NULL
           AND to_jsonb(NEW)-'actor_user_id'=to_jsonb(OLD)-'actor_user_id'
           AND NOT EXISTS (SELECT 1 FROM users WHERE id=OLD.actor_user_id) THEN RETURN NEW;
        END IF;
        RAISE EXCEPTION 'immutable review history'; END $$""")
    op.execute("""CREATE TRIGGER immutable_artifact BEFORE UPDATE ON receipt_review_revisions
        FOR EACH ROW EXECUTE FUNCTION vision_bill_immutable_review()""")
    # Direct deletion is forbidden; cascading deletion with the receipt remains supported.
    op.execute("""CREATE FUNCTION vision_bill_review_no_delete() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN IF pg_trigger_depth()=1 THEN RAISE EXCEPTION 'append-only review history';
        END IF; RETURN OLD; END $$""")
    op.execute("""CREATE TRIGGER append_only_review BEFORE DELETE ON receipt_review_revisions
        FOR EACH ROW EXECUTE FUNCTION vision_bill_review_no_delete()""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM receipt_review_revisions)
        OR EXISTS (SELECT 1 FROM receipts WHERE extraction_id IS NOT NULL OR operation_revision<>1
        OR review_revision<>1 OR verification_source NOT IN ('legacy')
        OR amount_policy<>'{}'::jsonb OR findings<>'[]'::jsonb OR field_sources<>'{}'::jsonb
        OR checks_version IS NOT NULL) THEN
        RAISE EXCEPTION 'destructive review downgrade refused'; END IF; END $$""")
    op.execute("DROP TABLE receipt_review_revisions")
    op.execute("DROP FUNCTION vision_bill_immutable_review()")
    op.execute("DROP FUNCTION vision_bill_review_no_delete()")
    for column in ("extraction_id", "operation_revision", "review_revision", "amount_policy",
                   "findings", "field_sources", "checks_version", "verification_source"):
        op.execute(f"ALTER TABLE receipts DROP COLUMN {column}")
