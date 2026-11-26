"""Tests for ReceiptDB — pool lifecycle and SQL behaviour with mocked asyncpg."""

import logging
from datetime import UTC, datetime
from datetime import date as Date
from datetime import time as Time
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from vision_bill.api.receipts import LineItemUpdate, ReceiptUpdate
from vision_bill.config import Settings
from vision_bill.model.receipt import LineItem, Receipt, TaxLine
from vision_bill.provider.db.receipt_db import (
    DELETE_CATEGORY_ALIAS_SQL,
    DELETE_LINE_ITEMS_SQL,
    DELETE_RECEIPT_SQL,
    DELETE_TAXES_SQL,
    GET_RECEIPT_WITH_IMAGE_SQL,
    INSERT_LINE_ITEM_SQL,
    INSERT_TAG_SQL,
    INSERT_TAX_SQL,
    LIST_LINE_ITEMS_SQL,
    LIST_RECEIPTS_BASE_SQL,
    LIST_TAGS_SQL,
    SEARCH_PRODUCTS_BASE_SQL,
    UPSERT_CATEGORY_ALIAS_SQL,
    ReceiptDB,
)

PATCH_TARGET = "vision_bill.provider.db.receipt_db.asyncpg"
RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000001")
OTHER_RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000002")
IMAGE_ID = UUID("00000000-0000-4000-8000-000000000003")
LINE_ITEM_ID = UUID("00000000-0000-4000-8000-000000000004")
TAX_ID = UUID("00000000-0000-4000-8000-000000000005")


def _make_pool(conn: AsyncMock) -> MagicMock:
    """Build a mock pool whose acquire() yields the given conn."""
    conn.transaction = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    conn.fetch = AsyncMock(return_value=[])
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    return pool


def _receipt_row(
    receipt_id: UUID = RECEIPT_ID,
    **overrides: object,
) -> dict[str, Any]:
    """A realistic asyncpg receipts row (driver types, not strings)."""
    row: dict[str, Any] = {
        "id": receipt_id,
        "confidence": 95,
        "merchant_name": "Test Store",
        "merchant_address": "123 Main St",
        "receipt_number": "R-001",
        "date": Date(2024, 1, 15),
        "time": Time(14, 30),
        "currency": "USD",
        "category": "other",
        "subtotal": Decimal("50.00"),
        "discount_total": Decimal("0.00"),
        "tax_total": Decimal("4.50"),
        "tip": None,
        "total": Decimal("54.50"),
        "payment_method": "credit_card",
        "status": "unverified",
        "image_id": None,
        "created_at": datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
        "verified": False,
    }
    row.update(overrides)
    return row


def _line_item_row(receipt_id: UUID = RECEIPT_ID) -> dict[str, Any]:
    return {
        "id": LINE_ITEM_ID,
        "receipt_id": receipt_id,
        "description": "Item A",
        "quantity": Decimal("2.0000"),
        "unit_price": Decimal("10.00"),
        "total_price": Decimal("20.00"),
        "tags": ["test"],
        "position": 0,
    }


def _pfand_line_item_row(receipt_id: UUID = RECEIPT_ID) -> dict[str, Any]:
    """Driver row for a German bottle-deposit refund (negative prices)."""
    return {
        "id": LINE_ITEM_ID,
        "receipt_id": receipt_id,
        "description": "Pfand (Flaschen)",
        "quantity": Decimal("3.0000"),
        "unit_price": Decimal("-0.25"),
        "total_price": Decimal("-0.75"),
        "tags": ["other"],
        "position": 0,
    }


def _tax_row(receipt_id: UUID = RECEIPT_ID) -> dict[str, Any]:
    return {
        "id": TAX_ID,
        "receipt_id": receipt_id,
        "name": "VAT",
        "rate": Decimal("0.1900"),
        "amount": Decimal("4.50"),
    }


def _review_update(receipt):
    result = receipt if isinstance(receipt, ReceiptUpdate) else ReceiptUpdate.model_validate(receipt.model_dump())
    result.expected_operation_revision = 1
    return result


def _make_receipt(merchant_name: str = "Test Store") -> Receipt:
    return Receipt(
        confidence=95,
        merchant_name=merchant_name,
        merchant_address="123 Main St",
        receipt_number="R-001",
        date=Date(2024, 1, 15),
        time="14:30",
        currency="USD",
        category="grocery",
        line_items=[
            LineItem(
                description="Item A",
                quantity=2,
                unit_price=Decimal("10.00"),
                total_price=Decimal("20.00"),
                tags=["test"],
            )
        ],
        taxes=[TaxLine(name="VAT", rate=0.19, amount=Decimal("4.50"))],
        subtotal=Decimal("50.00"),
        discount_total=Decimal(0),
        tax_total=Decimal("4.50"),
        total=Decimal("54.50"),
        payment_method="credit_card",
    )


@pytest.fixture
def db(settings: Settings) -> ReceiptDB:
    return ReceiptDB(settings.pg)


# ── Connection pool lifecycle tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_init_db_creates_pool(db: ReceiptDB, caplog: pytest.LogCaptureFixture) -> None:
    """init_db should create the pool, run the to_regclass check, and skip DDL."""
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"missing": False})
    mock_pool = _make_pool(mock_conn)

    with (
        patch(PATCH_TARGET) as mock_asyncpg,
        caplog.at_level(logging.INFO, logger="vision_bill.provider.db.receipt_db"),
    ):
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
        await db.init_db()

    mock_asyncpg.create_pool.assert_called_once()
    assert db._pool is mock_pool
    # No DDL in the app code anymore — migrations own the schema.
    mock_conn.execute.assert_not_called()
    # The to_regclass sanity check IS executed against the acquired connection.
    mock_conn.fetchrow.assert_called_once()
    assert "to_regclass('receipts')" in mock_conn.fetchrow.call_args.args[0]
    # Table present -> no "schema not initialised" warning.
    assert not any("Schema not initialised" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_init_db_warns_when_schema_missing(
    db: ReceiptDB, caplog: pytest.LogCaptureFixture
) -> None:
    """init_db should warn (not raise) when the receipts table is missing."""
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"missing": True})
    mock_pool = _make_pool(mock_conn)

    with (
        patch(PATCH_TARGET) as mock_asyncpg,
        caplog.at_level(logging.INFO, logger="vision_bill.provider.db.receipt_db"),
    ):
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
        await db.init_db()

    assert db._pool is mock_pool
    mock_conn.execute.assert_not_called()
    assert any("Schema not initialised" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_destroy_db_closes_pool(db: ReceiptDB) -> None:
    """destroy_db should close the pool and set it to None."""
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    db._pool = mock_pool

    await db.destroy_db()

    mock_pool.close.assert_called_once()
    assert db._pool is None


@pytest.mark.asyncio
async def test_init_db_skips_when_already_init(db: ReceiptDB) -> None:
    """init_db should skip if the pool already exists."""
    db._pool = MagicMock()

    with patch(PATCH_TARGET) as mock_asyncpg:
        mock_asyncpg.create_pool = AsyncMock()
        await db.init_db()

    mock_asyncpg.create_pool.assert_not_called()


def test_pool_property_raises_when_not_init(db: ReceiptDB) -> None:
    """pool property should raise RuntimeError if not initialised."""
    with pytest.raises(RuntimeError, match="Database pool not initialised"):
        _ = db.pool


# ── Persist and query tests (mocked pool) ────────────────────────────


@pytest.mark.asyncio
async def test_persist_receipt(db: ReceiptDB) -> None:
    """persist_receipt should insert receipt + line items + taxes."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=_receipt_row(image_id=IMAGE_ID, category="grocery"))
    mock_conn.execute = AsyncMock()

    receipt = _make_receipt()

    result = await db.persist_receipt(receipt, image_id=IMAGE_ID, status="unverified")

    assert result.id == RECEIPT_ID
    assert result.confidence == 95
    assert result.merchant_name == "Test Store"
    assert result.status == "unverified"
    assert result.category == "grocery"
    assert result.verified is False
    assert result.image_id == IMAGE_ID
    assert result.time == "14:30:00"
    assert result.created_at == Date(2024, 1, 15)

    # The TIME column must be bound as datetime.time, never a string
    fetchrow_call = mock_conn.fetchrow.call_args
    assert fetchrow_call is not None
    # confidence is the first bound parameter
    assert fetchrow_call.args[1] == 95
    bound_time = fetchrow_call.args[6]
    assert isinstance(bound_time, Time)
    assert bound_time == Time(14, 30)
    # category is the 8th bound parameter
    assert fetchrow_call.args[8] == "grocery"
    # status, verified, user_id and image_id are the last bound parameters
    assert fetchrow_call.args[15] == "unverified"
    assert fetchrow_call.args[16] is False
    assert fetchrow_call.args[18] == IMAGE_ID

    assert mock_conn.fetchrow.call_count == 1
    # Deletes replace any previous children, followed by one insert each for
    # the line item and the tax.
    assert mock_conn.execute.call_count == 4


@pytest.mark.asyncio
async def test_persist_receipt_binds_negative_line_item_prices(db: ReceiptDB) -> None:
    """Negative Pfand line-item prices must be bound to the INSERT as negative values."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=_receipt_row(image_id=IMAGE_ID))
    mock_conn.execute = AsyncMock()

    receipt = _make_receipt()
    receipt.line_items.append(
        LineItem(
            description="Pfand (Flaschen)",
            quantity=3,
            unit_price=Decimal("-0.25"),
            total_price=Decimal("-0.75"),
        )
    )

    await db.persist_receipt(receipt, image_id=IMAGE_ID, status="unverified")

    insert_calls = [
        call for call in mock_conn.execute.call_args_list if call.args[0] == INSERT_LINE_ITEM_SQL
    ]
    assert len(insert_calls) == 2
    pfand_call = insert_calls[1]
    # Monetary binds preserve exact Decimal values, including negative prices.
    assert pfand_call.args[4] == -0.25
    assert pfand_call.args[5] == -0.75
    assert isinstance(pfand_call.args[4], Decimal)
    assert isinstance(pfand_call.args[5], Decimal)


@pytest.mark.asyncio
async def test_get_receipt_by_found(db: ReceiptDB) -> None:
    """get_receipt_by_id should return a ReceiptRow when found."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(
        return_value=_receipt_row(
            receipt_id=RECEIPT_ID, merchant_name="Found Store", status="verified"
        )
    )

    result = await db.get_receipt_by_id(RECEIPT_ID)

    assert result is not None
    assert result.id == RECEIPT_ID
    assert result.merchant_name == "Found Store"
    assert result.status == "verified"


@pytest.mark.asyncio
async def test_get_receipt_by_not_found(db: ReceiptDB) -> None:
    """get_receipt_by_id should return None when the receipt doesn't exist."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=None)

    result = await db.get_receipt_by_id(OTHER_RECEIPT_ID)

    assert result is None


@pytest.mark.asyncio
async def test_list_receipts(db: ReceiptDB) -> None:
    """list_receipts should return a paginated list of receipts."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetch = AsyncMock(
        return_value=[
            _receipt_row(receipt_id=OTHER_RECEIPT_ID, merchant_name="Store B"),
            _receipt_row(receipt_id=RECEIPT_ID, merchant_name="Store A"),
        ]
    )

    results = await db.list_receipts(limit=10, offset=0)

    assert "ORDER BY r.date DESC, r.id" in mock_conn.fetch.call_args.args[0]

    assert len(results) == 2
    assert results[0].merchant_name == "Store B"
    assert results[1].merchant_name == "Store A"


@pytest.mark.asyncio
async def test_search_products(db: ReceiptDB) -> None:
    """search_products should match verified line items and map purchase data."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetch = AsyncMock(
        return_value=[
            {
                **_line_item_row(RECEIPT_ID),
                "description": "Gouda Mittelalt",
                "merchant_name": "Test Store",
                "date": Date(2024, 1, 15),
                "time": Time(14, 30),
                "currency": "EUR",
            }
        ]
    )

    results = await db.search_products("gouda")

    assert len(results) == 1
    assert results[0].description == "Gouda Mittelalt"
    assert results[0].unit_price == Decimal("10.00")
    assert results[0].time == "14:30:00"
    fetch_call = mock_conn.fetch.call_args
    assert fetch_call is not None
    assert fetch_call.args[0].startswith(SEARCH_PRODUCTS_BASE_SQL)
    assert "li.description ILIKE $1" in fetch_call.args[0]
    assert fetch_call.args[1] == "%gouda%"


@pytest.mark.asyncio
async def test_get_receipt_with_details(db: ReceiptDB) -> None:
    """get_receipt_with_details should combine row, line items, taxes and image path.

    fetchrow serves the LEFT JOIN query, so the row carries the receipt columns
    plus an extra ``image_path`` key (the mapper drops it, the detail keeps it).
    """
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(
        return_value=_receipt_row(
            receipt_id=RECEIPT_ID, image_id=IMAGE_ID, image_path="/save/img.png"
        )
    )

    def fetch_side_effect(sql: str, *args: object) -> list[dict[str, Any]]:
        if "line_items" in sql:
            return [_line_item_row(RECEIPT_ID)]
        if "taxes" in sql:
            return [_tax_row(RECEIPT_ID)]
        return []

    mock_conn.fetch = AsyncMock(side_effect=fetch_side_effect)

    details = await db.get_receipt_with_details(RECEIPT_ID)

    assert details is not None
    assert details.receipt.id == RECEIPT_ID
    assert details.receipt.image_id == IMAGE_ID
    assert details.image_path == "/save/img.png"
    assert len(details.line_items) == 1
    assert details.line_items[0].description == "Item A"
    assert details.line_items[0].quantity == 2.0
    assert details.line_items[0].tags == ["test"]
    assert len(details.taxes) == 1
    assert details.taxes[0].rate == Decimal("0.19")


@pytest.mark.asyncio
async def test_get_receipt_with_details_reads_back_negative_line_item(db: ReceiptDB) -> None:
    """Negative prices returned by the driver must round-trip onto a LineItemRow."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=_receipt_row())

    def fetch_side_effect(sql: str, *args: object) -> list[dict[str, Any]]:
        if "line_items" in sql:
            return [_pfand_line_item_row(RECEIPT_ID)]
        if "taxes" in sql:
            return []
        return []

    mock_conn.fetch = AsyncMock(side_effect=fetch_side_effect)

    details = await db.get_receipt_with_details(RECEIPT_ID)

    assert details is not None
    item = details.line_items[0]
    assert item.unit_price == Decimal("-0.25")
    assert item.total_price == Decimal("-0.75")
    assert item.unit_price < 0
    assert item.total_price < 0


# ── Update and verify ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_receipt(db: ReceiptDB) -> None:
    """update_receipt should UPDATE ... RETURNING, delete children, re-insert."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(
        return_value=_receipt_row(receipt_id=RECEIPT_ID, merchant_name="Updated Store")
    )
    mock_conn.execute = AsyncMock()

    receipt = _make_receipt(merchant_name="Updated Store")

    result = await db.update_receipt(RECEIPT_ID, _review_update(receipt))

    assert result is not None
    assert result.merchant_name == "Updated Store"

    fetchrow_call = mock_conn.fetchrow.call_args
    assert fetchrow_call is not None
    assert "UPDATE receipts" in fetchrow_call.args[0]
    assert "RETURNING *" in fetchrow_call.args[0]

    executed_sqls = [call.args[0] for call in mock_conn.execute.call_args_list]
    assert DELETE_LINE_ITEMS_SQL in executed_sqls
    assert DELETE_TAXES_SQL in executed_sqls
    assert INSERT_LINE_ITEM_SQL in executed_sqls
    assert INSERT_TAX_SQL in executed_sqls
    # Children must be replaced: deletes happen before re-inserts
    assert executed_sqls.index(DELETE_LINE_ITEMS_SQL) < executed_sqls.index(INSERT_LINE_ITEM_SQL)


@pytest.mark.asyncio
async def test_update_receipt_not_found(db: ReceiptDB) -> None:
    """update_receipt should return None when no receipt row matched."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()

    result = await db.update_receipt(OTHER_RECEIPT_ID, _make_receipt())

    assert result is None
    mock_conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_verify_receipt_rejects_already_verified(db: ReceiptDB) -> None:
    """The locked state, rather than an earlier API read, rejects duplicate verification."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=_receipt_row(status="verified", verified=True))
    with pytest.raises(ValueError, match="already_verified"):
        await db.verify_receipt(RECEIPT_ID)
    mock_conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_receipt_not_found(db: ReceiptDB) -> None:
    """verify_receipt should return None when no receipt row matched."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=None)

    result = await db.verify_receipt(OTHER_RECEIPT_ID)

    assert result is None


# ── Delete ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_receipt(db: ReceiptDB) -> None:
    """delete_receipt should DELETE ... RETURNING and map the removed row."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(
        return_value=_receipt_row(
            receipt_id=RECEIPT_ID, image_id=IMAGE_ID, merchant_name="Doomed Store"
        )
    )

    result = await db.delete_receipt(RECEIPT_ID)

    assert result is not None
    assert result.id == RECEIPT_ID
    assert result.image_id == IMAGE_ID

    fetchrow_call = mock_conn.fetchrow.call_args
    assert fetchrow_call is not None
    assert fetchrow_call.args[0] == DELETE_RECEIPT_SQL
    assert fetchrow_call.args[1] == RECEIPT_ID


@pytest.mark.asyncio
async def test_delete_receipt_not_found(db: ReceiptDB) -> None:
    """delete_receipt should return None when no receipt row matched."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=None)

    result = await db.delete_receipt(OTHER_RECEIPT_ID)

    assert result is None


# ── Tags (vocabulary) ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tags(db: ReceiptDB) -> None:
    """list_tags maps the name column and preserves the DB ordering."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetch = AsyncMock(return_value=[{"name": "coffee"}, {"name": "food"}])

    result = await db.list_tags()

    assert result == ["coffee", "food"]
    fetch_call = mock_conn.fetch.await_args
    assert fetch_call is not None
    assert fetch_call.args[0] == LIST_TAGS_SQL


@pytest.mark.asyncio
async def test_create_tag_new(db: ReceiptDB) -> None:
    """create_tag inserts the name and reports a fresh creation."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value={"name": "coffee"})

    created = await db.create_tag("coffee")

    assert created == ("coffee", True)
    fetchrow_call = mock_conn.fetchrow.await_args
    assert fetchrow_call is not None
    assert fetchrow_call.args[0] == INSERT_TAG_SQL
    assert fetchrow_call.args[1] == "coffee"


@pytest.mark.asyncio
async def test_create_tag_existing_is_not_an_error(db: ReceiptDB) -> None:
    """Look up the stored spelling after the insert skips an existing name."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=None)

    mock_conn.fetchval = AsyncMock(return_value="Coffee")
    created = await db.create_tag("coffee")

    assert created == ("Coffee", False)
    # The insert must be conflict-safe, not a plain INSERT.
    insert_sql = mock_conn.fetchrow.await_args.args[0]
    assert "ON CONFLICT (name) DO NOTHING" in insert_sql


def test_insert_line_item_sql_has_spending_category() -> None:
    """Item spending categories are separate from the receipt-level category."""
    assert "spending_category" in INSERT_LINE_ITEM_SQL
    assert "category_source" in INSERT_LINE_ITEM_SQL


@pytest.mark.asyncio
async def test_get_statistics_scopes_to_collection(db: ReceiptDB) -> None:
    """When collection_id is set, every STATS query gains the EXISTS scope."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])  # all six queries return empty
    db._pool = _make_pool(conn)
    coll_id = UUID("00000000-0000-4000-8000-0000000000c1")
    await db.get_statistics(
        user_id=UUID("00000000-0000-4000-8000-00000000000a"), weeks=12, collection_id=coll_id
    )
    first_sql = conn.fetch.call_args_list[0].args[0]
    assert "EXISTS" in first_sql
    assert "receipt_collections" in first_sql
    # collection id is a bound parameter, not interpolated
    args = conn.fetch.call_args_list[0].args[1:]
    assert coll_id in args
    # The weekly query (separate arg list) must also be scoped, with the
    # collection id still bound ahead of the weeks offset (no off-by-one).
    weekly_sql = conn.fetch.call_args_list[5].args[0]
    assert "EXISTS" in weekly_sql
    weekly_args = conn.fetch.call_args_list[5].args[1:]
    assert coll_id in weekly_args
    # The weeks offset is the LAST bound param — pins the ordering.
    assert weekly_args[-1] == 11  # weeks=12 -> max(weeks - 1, 0)


@pytest.mark.asyncio
async def test_get_statistics_without_collection_has_no_scope(db: ReceiptDB) -> None:
    """No collection_id -> no EXISTS scope is added to any of the six queries."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    db._pool = _make_pool(conn)
    await db.get_statistics(user_id=UUID("00000000-0000-4000-8000-00000000000a"), weeks=12)
    assert len(conn.fetch.call_args_list) == 6  # guard against a vacuous loop below
    for call in conn.fetch.call_args_list:
        assert "EXISTS" not in call.args[0]


@pytest.mark.asyncio
async def test_list_receipts_filters_by_collection(db: ReceiptDB) -> None:
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    db._pool = _make_pool(conn)
    coll_id = UUID("00000000-0000-4000-8000-0000000000c1")
    await db.list_receipts(
        collection_id=coll_id, user_id=UUID("00000000-0000-4000-8000-00000000000a")
    )
    sql = conn.fetch.call_args.args[0]
    assert "receipt_collections" in sql
    assert "collection_id" in sql
    # The collection id is a bound parameter, not interpolated.
    assert coll_id in conn.fetch.call_args.args[1:]


@pytest.mark.asyncio
async def test_persist_receipt_line_items_get_position(db: ReceiptDB) -> None:
    """Each persisted line item must receive a 0-based position matching its array index."""
    mock_conn = AsyncMock()
    db._pool = _make_pool(mock_conn)
    mock_conn.fetchrow = AsyncMock(return_value=_receipt_row(image_id=IMAGE_ID))
    mock_conn.execute = AsyncMock()

    receipt = _make_receipt()
    # Add more items so we can verify 0-based indexing across multiple inserts.
    receipt.line_items.extend(
        [
            LineItem(
                description="Item B",
                quantity=1,
                unit_price=Decimal("5.00"),
                total_price=Decimal("5.00"),
            ),
            LineItem(
                description="Item C",
                quantity=3,
                unit_price=Decimal("2.00"),
                total_price=Decimal("6.00"),
            ),
        ]
    )

    await db.persist_receipt(receipt, image_id=IMAGE_ID, status="unverified")

    insert_calls = [
        call for call in mock_conn.execute.call_args_list if call.args[0] == INSERT_LINE_ITEM_SQL
    ]
    assert len(insert_calls) == 3
    for idx, call in enumerate(insert_calls):
        # The 7th bound parameter ($7) is position (args[7] because args[0] is SQL).
        assert call.args[7] == idx


def test_list_line_items_sql_orders_by_position() -> None:
    """Line items must be fetched ordered by position, not by UUID."""
    assert "ORDER BY position" in LIST_LINE_ITEMS_SQL


def test_detail_sql_exposes_thumbnail_path() -> None:
    """Detail query must join images and select thumbnail_path."""
    assert "i.thumbnail_path" in GET_RECEIPT_WITH_IMAGE_SQL
    assert "LEFT JOIN images i ON i.id = r.image_id" in GET_RECEIPT_WITH_IMAGE_SQL


def test_list_sql_exposes_thumbnail_path() -> None:
    """List query must expose thumbnail_path via correlated subquery."""
    assert "thumbnail_path" in LIST_RECEIPTS_BASE_SQL
    assert "FROM receipts r" in LIST_RECEIPTS_BASE_SQL


@pytest.mark.asyncio
async def test_confirmed_history_overrides_model_without_learning(db: ReceiptDB) -> None:
    conn = AsyncMock()
    conn.fetchval.return_value = None  # No previously confirmed merchant alias.
    db._pool = _make_pool(conn)
    owner = UUID("00000000-0000-4000-8000-000000000099")
    conn.fetchrow = AsyncMock(return_value=_receipt_row(user_id=owner))
    conn.fetch = AsyncMock(return_value=[{"description_key": "item a", "category": "milk"}])
    receipt = _make_receipt()
    receipt.line_items[0].spending_category = "cheese"

    await db.persist_receipt(receipt, user_id=owner, verified=True)

    inserted = next(c for c in conn.execute.call_args_list if c.args[0] == INSERT_LINE_ITEM_SQL)
    assert inserted.args[9:12] == ("milk", "history", "Item A")
    assert conn.fetch.await_args.args[1] == owner
    assert not any(c.args[0] == UPSERT_CATEGORY_ALIAS_SQL for c in conn.execute.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("remember", [False, True])
async def test_update_preserves_original_and_owner_history(
    db: ReceiptDB, remember: bool
) -> None:
    conn = AsyncMock()
    db._pool = _make_pool(conn)
    owner = UUID("00000000-0000-4000-8000-000000000099")
    admin = UUID("00000000-0000-4000-8000-000000000098")
    conn.fetchrow = AsyncMock(return_value=_receipt_row(user_id=owner))
    conn.fetch = AsyncMock(side_effect=lambda sql, *args: [{
        **_line_item_row(), "description": "Milk", "original_description": "H-MILCH 1,5%",
        "spending_category": "milk", "category_source": "user",
    }] if sql == LIST_LINE_ITEMS_SQL else [])
    receipt = ReceiptUpdate.model_validate(_make_receipt().model_dump())
    receipt.line_items = [LineItemUpdate(
        id=LINE_ITEM_ID, description="Organic milk", quantity=1,
        unit_price=Decimal(2), total_price=Decimal(2),
        remember_category=remember,
    )]

    await db.update_receipt(RECEIPT_ID, _review_update(receipt), user_id=admin, can_see_all=True)

    inserted = next(c for c in conn.execute.call_args_list if c.args[0] == INSERT_LINE_ITEM_SQL)
    assert inserted.args[8:12] == (LINE_ITEM_ID, "milk", "user", "H-MILCH 1,5%")
    aliases = [c for c in conn.execute.call_args_list if c.args[0] == UPSERT_CATEGORY_ALIAS_SQL]
    assert len(aliases) == (2 if remember else 0)
    if remember:
        assert {c.args[3] for c in aliases} == {"h-milch 1,5%", "organic milk"}
        assert all(c.args[1] == owner for c in aliases)


@pytest.mark.asyncio
async def test_legacy_replacement_does_not_inherit_removed_item(db: ReceiptDB) -> None:
    conn = AsyncMock()
    db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_receipt_row())
    conn.fetch = AsyncMock(side_effect=lambda sql, *args: [{
        **_line_item_row(), "description": "Removed item",
        "original_description": "Removed item", "spending_category": "milk",
        "category_source": "user",
    }] if sql == LIST_LINE_ITEMS_SQL else [])
    receipt = _make_receipt()
    await db.update_receipt(RECEIPT_ID, _review_update(receipt))
    inserted = next(c for c in conn.execute.call_args_list if c.args[0] == INSERT_LINE_ITEM_SQL)
    assert inserted.args[8:12] == (None, "unknown", "unknown", "Item A")


@pytest.mark.asyncio
async def test_explicit_unknown_removes_both_confirmed_aliases(db: ReceiptDB) -> None:
    conn = AsyncMock()
    conn.fetchval.return_value = None  # No previously confirmed merchant alias.
    db._pool = _make_pool(conn)
    owner = UUID("00000000-0000-4000-8000-000000000099")
    conn.fetchrow = AsyncMock(return_value=_receipt_row(user_id=owner))
    conn.fetch = AsyncMock(side_effect=lambda sql, *args: [{
        **_line_item_row(), "description": "Milk", "original_description": "H-MILCH 1,5%",
        "spending_category": "milk", "category_source": "history",
    }] if sql == LIST_LINE_ITEMS_SQL else [])
    receipt = ReceiptUpdate.model_validate(_make_receipt().model_dump())
    receipt.line_items = [LineItemUpdate(
        id=LINE_ITEM_ID, description="Milk", quantity=1,
        unit_price=Decimal(2), total_price=Decimal(2),
        spending_category="unknown", remember_category=True,
    )]
    await db.update_receipt(RECEIPT_ID, _review_update(receipt), user_id=owner)
    inserted = next(c for c in conn.execute.call_args_list if c.args[0] == INSERT_LINE_ITEM_SQL)
    assert inserted.args[9:11] == ("unknown", "user")
    deletes = [c for c in conn.execute.call_args_list if c.args[0] == DELETE_CATEGORY_ALIAS_SQL]
    assert {c.args[3] for c in deletes} == {"h-milch 1,5%", "milk"}


@pytest.mark.asyncio
async def test_details_and_replacement_only_access_visible_collection_links(db: ReceiptDB) -> None:
    owner, target = OTHER_RECEIPT_ID, IMAGE_ID
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=_receipt_row())
    db._pool = _make_pool(conn)
    conn.fetch.side_effect = [[], [], [{"collection_id": target}], [{"id": target}]]
    detail = await db.get_receipt_with_details(RECEIPT_ID, user_id=owner)
    assert detail is not None and detail.collection_ids == [target]
    read_sql, *read_args = conn.fetch.await_args.args
    assert "JOIN collections" in read_sql and "c.user_id" in read_sql
    assert read_args == [RECEIPT_ID, owner]

    await db._set_collections(conn, RECEIPT_ID, [target], owner, False)
    delete_sql, *delete_args = conn.execute.await_args_list[0].args
    assert "USING collections" in delete_sql and "c.user_id" in delete_sql
    assert delete_args == [RECEIPT_ID, owner]


@pytest.mark.parametrize("candidate, expected", [
    ("Globus Baumarkt", "Globus Baumarkt"),
    ("Globus Markthallen", "Globus Markthallen"),
    ("Unrelated chain", "Globus"),
])
async def test_automatic_receipt_accepts_only_candidates_for_printed_brand(db, candidate, expected):
    from vision_bill.provider.db.receipt_db import INSERT_RECEIPT_SQL

    conn = AsyncMock()
    conn.fetchrow.return_value = _receipt_row()
    db._pool = _make_pool(conn)
    receipt = _make_receipt("Globus")
    receipt.merchant_display_name = candidate
    await db.persist_receipt(receipt)
    insert = conn.fetchrow.await_args
    assert insert.args[0] == INSERT_RECEIPT_SQL
    assert insert.args[2] == "Globus"
    assert insert.args[19] == expected
