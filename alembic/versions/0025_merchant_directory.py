"""Keep the printed merchant alongside an editable name and private directory."""

from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE receipt_drafts ADD COLUMN item_category_sources JSONB NOT NULL DEFAULT '[]'::jsonb")
    op.execute("ALTER TABLE receipts ADD COLUMN merchant_display_name TEXT")
    op.execute("""
        CREATE TABLE merchants (
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name_key TEXT NOT NULL,
            display_name TEXT NOT NULL,
            PRIMARY KEY (user_id, name_key)
        )
    """)
    op.execute("""
        CREATE TABLE merchant_aliases (
            user_id UUID NOT NULL,
            printed_key TEXT NOT NULL,
            address_key TEXT NOT NULL,
            name_key TEXT NOT NULL,
            PRIMARY KEY (user_id, printed_key, address_key),
            FOREIGN KEY (user_id, name_key) REFERENCES merchants(user_id, name_key)
                ON DELETE CASCADE
        )
    """)


def downgrade() -> None:
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT FROM merchants) OR EXISTS (SELECT FROM merchant_aliases)
           OR EXISTS (SELECT FROM receipts WHERE merchant_display_name IS NOT NULL)
           OR EXISTS (SELECT FROM receipt_drafts WHERE document->>'merchant_display_name' IS NOT NULL
                     OR item_category_sources <> '[]'::jsonb)
        THEN RAISE EXCEPTION 'destructive merchant directory downgrade refused';
        END IF;
    END $$""")
    op.execute("ALTER TABLE receipt_drafts DROP COLUMN item_category_sources")
    op.execute("DROP TABLE merchant_aliases")
    op.execute("DROP TABLE merchants")
    op.execute("ALTER TABLE receipts DROP COLUMN merchant_display_name")
