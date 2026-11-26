"""Allow unprinted receipt values to remain unknown after review."""

from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE receipts ALTER COLUMN discount_total DROP NOT NULL")
    op.execute("""ALTER TABLE line_items
        ALTER COLUMN quantity DROP NOT NULL,
        ALTER COLUMN unit_price DROP NOT NULL""")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM receipts WHERE discount_total IS NULL)
          OR EXISTS (SELECT 1 FROM line_items WHERE quantity IS NULL OR unit_price IS NULL)
        THEN RAISE EXCEPTION 'optional receipt values downgrade refused'; END IF;
        END $$""")
    op.execute("ALTER TABLE receipts ALTER COLUMN discount_total SET NOT NULL")
    op.execute("""ALTER TABLE line_items
        ALTER COLUMN quantity SET NOT NULL,
        ALTER COLUMN unit_price SET NOT NULL""")
