"""Independent human-authored benchmark cases and typed task input identities."""

from alembic import op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE benchmark_cases (
        id UUID PRIMARY KEY,
        owner_id UUID NOT NULL REFERENCES users(id),
        name TEXT NOT NULL CHECK (length(btrim(name)) BETWEEN 1 AND 160),
        revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
        sources JSONB NOT NULL CHECK (
            jsonb_typeof(sources) = 'array' AND jsonb_array_length(sources) BETWEEN 1 AND 10),
        expected JSONB NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(expected) = 'object'),
        confirmation_id UUID UNIQUE,
        confirmed_by UUID REFERENCES users(id),
        confirmed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CHECK ((confirmation_id IS NULL) = (confirmed_by IS NULL)
            AND (confirmation_id IS NULL) = (confirmed_at IS NULL))
    )""")
    op.execute("CREATE INDEX benchmark_cases_by_owner ON benchmark_cases (owner_id, created_at DESC, id)")
    # Keep the existing task key and every legacy caller. Each task still has a
    # real FK for its selected input type, including the original deletion fence.
    op.execute("""ALTER TABLE benchmark_tasks
        DROP CONSTRAINT benchmark_tasks_receipt_id_fkey,
        ADD COLUMN input_kind TEXT NOT NULL DEFAULT 'receipt' CHECK (input_kind IN ('receipt','case')),
        ADD COLUMN ledger_receipt_id UUID GENERATED ALWAYS AS
            (CASE WHEN input_kind='receipt' THEN receipt_id END) STORED REFERENCES receipts(id),
        ADD COLUMN case_id UUID GENERATED ALWAYS AS
            (CASE WHEN input_kind='case' THEN receipt_id END) STORED REFERENCES benchmark_cases(id)""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS(SELECT 1 FROM benchmark_cases) THEN
        RAISE EXCEPTION 'destructive benchmark case downgrade refused'; END IF; END $$""")
    op.execute("""ALTER TABLE benchmark_tasks
        DROP COLUMN ledger_receipt_id, DROP COLUMN case_id, DROP COLUMN input_kind,
        ADD CONSTRAINT benchmark_tasks_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES receipts(id)""")
    op.execute("DROP TABLE benchmark_cases")
