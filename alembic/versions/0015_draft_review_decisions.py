"""Keep editable draft overlap decisions separate from immutable extraction evidence."""
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE receipt_drafts ADD COLUMN overlap_decisions JSONB NOT NULL DEFAULT '[]'")
    op.execute("""UPDATE receipt_drafts d SET overlap_decisions=e.envelope->'overlap_decisions'
        FROM receipt_extractions e WHERE e.id=d.extraction_id
        AND jsonb_typeof(e.envelope->'overlap_decisions')='array'""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM receipt_drafts WHERE overlap_decisions <> '[]'::jsonb)
        THEN RAISE EXCEPTION 'Cannot drop nonempty draft overlap decisions'; END IF; END $$""")
    op.execute("ALTER TABLE receipt_drafts DROP COLUMN overlap_decisions")
