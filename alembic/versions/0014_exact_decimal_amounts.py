"""Lossless review-number storage and current review decisions."""

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

MONEY = {
    "receipts": ("subtotal", "discount_total", "tax_total", "tip", "total"),
    "line_items": ("unit_price", "total_price"),
    "taxes": ("amount",),
}
# Quantity and rate are material review inputs too; retain their former bounds
# only as downgrade guards, never as implicit rounding on new writes.
NUMERIC_COLUMNS = [(table, column, 14, 2) for table, columns in MONEY.items() for column in columns]
NUMERIC_COLUMNS += [("line_items", "quantity", 10, 4), ("taxes", "rate", 6, 4)]


def upgrade() -> None:
    for table, column, _precision, _scale in NUMERIC_COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE NUMERIC")
    op.execute("ALTER TABLE receipts ADD COLUMN overlap_decisions JSONB NOT NULL DEFAULT '[]'")
    op.execute("ALTER TABLE receipts ADD COLUMN acknowledgements JSONB NOT NULL DEFAULT '[]'")


def downgrade() -> None:
    # Refuse, never round or discard newer review decisions.
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM receipts WHERE
        overlap_decisions<>'[]'::jsonb OR acknowledgements<>'[]'::jsonb) THEN
        RAISE EXCEPTION 'review decision downgrade refused'; END IF; END $$""")
    for table, column, precision, scale in NUMERIC_COLUMNS:
        limit = 10 ** (precision - scale)
        op.execute(f"""DO $$ BEGIN IF EXISTS (SELECT 1 FROM {table} WHERE
            {column}<>round({column},{scale}) OR abs({column})>={limit}) THEN
            RAISE EXCEPTION 'precision losing numeric downgrade refused';
            END IF; END $$""")
    for table, column, precision, scale in NUMERIC_COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE NUMERIC({precision},{scale})")
    op.execute("ALTER TABLE receipts DROP COLUMN acknowledgements, DROP COLUMN overlap_decisions")
