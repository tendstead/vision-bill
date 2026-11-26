"""Retain authorized file cleanup identities and detach explicitly deleted draft receipts."""

from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE resource_deletions (
        resource_id UUID NOT NULL, resource_kind TEXT NOT NULL CHECK(resource_kind IN ('receipt','image')),
        owner_id UUID REFERENCES users(id) ON DELETE SET NULL, files JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY(resource_id,resource_kind))""")
    op.execute(
        "ALTER TABLE receipt_drafts DROP CONSTRAINT receipt_drafts_finalized_receipt_id_fkey"
    )
    op.execute(
        "ALTER TABLE receipt_drafts ADD CONSTRAINT receipt_drafts_finalized_receipt_id_fkey FOREIGN KEY(finalized_receipt_id) REFERENCES receipts(id) ON DELETE SET NULL"
    )


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS(SELECT 1 FROM resource_deletions) THEN
        RAISE EXCEPTION 'pending file cleanup downgrade refused'; END IF; END $$""")
    op.execute("DROP TABLE resource_deletions")
    op.execute(
        "ALTER TABLE receipt_drafts DROP CONSTRAINT receipt_drafts_finalized_receipt_id_fkey"
    )
    op.execute(
        "ALTER TABLE receipt_drafts ADD CONSTRAINT receipt_drafts_finalized_receipt_id_fkey FOREIGN KEY(finalized_receipt_id) REFERENCES receipts(id)"
    )
