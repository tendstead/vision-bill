"""Per-user data scoping tests (Part 6).

The security property under test: a caller without ``can_see_all`` is confined
to rows they own (``user_id``); an admin with ``can_see_all`` sees everything.
The confinement happens in the SQL each provider emits, so the bulk of these
tests assert on the generated SQL and bound parameters against a mocked asyncpg
pool. A thin API-level section then proves the handler threads the
authenticated user all the way down to the query.

Row-level mapping (the ``user_id`` column) and the background-worker ownership
hand-off (image -> receipt) are covered in ``test_image_db`` /
``test_analysis_scheduler`` respectively.
"""

from collections.abc import Generator
from datetime import UTC, datetime
from datetime import date as Date
from datetime import time as Time
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import vision_bill.main as main_module
from vision_bill.api import images as images_api_module
from vision_bill.api.receipts import ReceiptUpdate
from vision_bill.api.system import main as system_api_module
from vision_bill.config import Settings
from vision_bill.model.receipt import LineItem, Receipt
from vision_bill.provider.db import image_db as image_db_module
from vision_bill.provider.db import receipt_db as receipt_db_module
from vision_bill.provider.db.image_db import (
    GET_IMAGE_SQL,
    LIST_IMAGES_BASE_SQL,
    ImageDB,
)
from vision_bill.provider.db.receipt_db import (
    GET_RECEIPT_SQL,
    GET_RECEIPT_WITH_IMAGE_SQL,
    LIST_RECEIPTS_BASE_SQL,
    SEARCH_PRODUCTS_BASE_SQL,
    UPDATE_RECEIPT_SQL,
    ReceiptDB,
)
from vision_bill.provider.llm.base import ModelInfo
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User

# Two distinct users; the tests only ever rely on them being unequal.
USER_A = UUID("00000000-0000-4000-8000-000000000001")
USER_B = UUID("00000000-0000-4000-8000-000000000002")
RESOURCE_ID = UUID("00000000-0000-4000-8000-000000000003")

RECEIPTS_URL = "/api/v1/receipts"


# ── Helpers ───────────────────────────────────────────────────────────


def _make_pool(conn: AsyncMock) -> MagicMock:
    """Build a mock pool whose acquire() yields the given conn."""
    conn.transaction = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    return pool


def _receipt_row(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": RESOURCE_ID,
        "confidence": 95,
        "merchant_name": "Store",
        "merchant_address": None,
        "receipt_number": "R-1",
        "date": Date(2024, 1, 15),
        "time": Time(14, 30),
        "currency": "USD",
        "category": "other",
        "subtotal": Decimal("10.00"),
        "discount_total": Decimal("0.00"),
        "tax_total": Decimal("0.00"),
        "tip": None,
        "total": Decimal("10.00"),
        "payment_method": "card",
        "status": "unverified",
        "image_id": None,
        "verified": False,
        "created_at": datetime(2024, 1, 15, 12, 0, tzinfo=UTC),
        "user_id": USER_A,
    }
    base.update(overrides)
    return base


def _image_row(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": RESOURCE_ID,
        "original_filename": "a.png",
        "media_type": "image/png",
        "size_bytes": 1,
        "image_path": "/tmp/a.png",
        "status": "pending",
        "error": None,
        "receipt_id": None,
        "bypass_review": False,
        "user_id": USER_A,
        "created_at": datetime(2024, 1, 15, 12, 0, tzinfo=UTC),
        "analyzed_at": None,
    }
    base.update(overrides)
    return base


def _make_receipt() -> Receipt:
    return Receipt(
        confidence=95,
        merchant_name="Store",
        date=Date(2024, 1, 15),
        currency="USD",
        line_items=[
            LineItem(
                description="Item",
                quantity=1,
                unit_price=Decimal("10.00"),
                total_price=Decimal("10.00"),
            )
        ],
        subtotal=Decimal("10.00"),
        total=Decimal("10.00"),
    )


@pytest.fixture
def receipt_db(settings: Settings) -> ReceiptDB:
    return ReceiptDB(settings.pg)


@pytest.fixture
def image_db(settings: Settings) -> ImageDB:
    return ImageDB(settings.pg)


# ── ReceiptDB scoping ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_receipt_by_id_scoped_to_owner(receipt_db: ReceiptDB) -> None:
    """A non-see-all caller only matches rows owned by their user_id."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_receipt_row())

    await receipt_db.get_receipt_by_id(RESOURCE_ID, user_id=USER_A, can_see_all=False)

    call = conn.fetchrow.call_args
    assert call.args[0] == GET_RECEIPT_SQL + " AND user_id = $2"
    assert call.args[1] == RESOURCE_ID
    assert call.args[2] == USER_A


@pytest.mark.asyncio
async def test_get_receipt_by_id_see_all_unscoped(receipt_db: ReceiptDB) -> None:
    """can_see_all bypasses the owner filter entirely (base query)."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_receipt_row())

    await receipt_db.get_receipt_by_id(RESOURCE_ID, can_see_all=True)

    call = conn.fetchrow.call_args
    assert call.args[0] == GET_RECEIPT_SQL
    assert call.args[1:] == (RESOURCE_ID,)


@pytest.mark.asyncio
async def test_get_receipt_by_id_unowned_is_unscoped(receipt_db: ReceiptDB) -> None:
    """A missing user_id (legacy/unowned) never narrows the query."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=None)

    await receipt_db.get_receipt_by_id(RESOURCE_ID)  # user_id defaults to None

    call = conn.fetchrow.call_args
    assert call.args[0] == GET_RECEIPT_SQL
    assert call.args[1:] == (RESOURCE_ID,)


@pytest.mark.asyncio
async def test_get_receipt_with_details_scoped_to_owner(receipt_db: ReceiptDB) -> None:
    """The detail query scopes on the aliased receipts table (r.user_id)."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=None)

    await receipt_db.get_receipt_with_details(RESOURCE_ID, user_id=USER_A, can_see_all=False)

    call = conn.fetchrow.call_args
    assert call.args[0] == GET_RECEIPT_WITH_IMAGE_SQL + " AND r.user_id = $2"
    assert call.args[2] == USER_A


@pytest.mark.asyncio
async def test_list_receipts_scoped_to_owner(receipt_db: ReceiptDB) -> None:
    """list_receipts injects user_id as the first WHERE predicate."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetch = AsyncMock(return_value=[])

    await receipt_db.list_receipts(user_id=USER_A, can_see_all=False)

    call = conn.fetch.call_args
    sql = call.args[0]
    assert sql.startswith(LIST_RECEIPTS_BASE_SQL)
    assert "user_id = $1" in sql
    assert "ORDER BY date DESC LIMIT $2 OFFSET $3" in sql
    assert call.args[1] == USER_A
    assert call.args[2:] == (50, 0)  # default limit / offset


@pytest.mark.asyncio
async def test_list_receipts_see_all_unscoped(receipt_db: ReceiptDB) -> None:
    """A see-all listing uses the plain paged base query (no owner filter)."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetch = AsyncMock(return_value=[])

    await receipt_db.list_receipts(can_see_all=True)

    call = conn.fetch.call_args
    assert "user_id" not in call.args[0]
    assert call.args[0] == LIST_RECEIPTS_BASE_SQL + " ORDER BY date DESC LIMIT $1 OFFSET $2"


@pytest.mark.asyncio
async def test_search_products_scoped_to_owner(receipt_db: ReceiptDB) -> None:
    """Product search must not expose another user's verified line items."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetch = AsyncMock(return_value=[])

    await receipt_db.search_products("Gouda", user_id=USER_A, can_see_all=False)

    call = conn.fetch.call_args
    assert call is not None
    assert call.args[0].startswith(SEARCH_PRODUCTS_BASE_SQL)
    assert "r.status = 'verified'" in call.args[0]
    assert "r.verified = TRUE" in call.args[0]
    assert "r.user_id = $2" in call.args[0]
    assert call.args[1] == "%Gouda%"
    assert call.args[2] == USER_A


@pytest.mark.asyncio
async def test_update_receipt_scoped_to_owner(receipt_db: ReceiptDB) -> None:
    """update_receipt only touches rows the caller owns (WHERE id AND user_id)."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_receipt_row())
    conn.execute = AsyncMock()

    await receipt_db.update_receipt(RESOURCE_ID, ReceiptUpdate.model_validate(dict(_make_receipt().model_dump(), expected_operation_revision=1)), user_id=USER_A, can_see_all=False)

    call = next(c for c in conn.fetchrow.call_args_list if "confidence      = $1" in c.args[0])
    assert "AND user_id = $16 RETURNING *" in call.args[0]
    assert call.args[-1] == USER_A


@pytest.mark.asyncio
async def test_update_receipt_see_all_unscoped(receipt_db: ReceiptDB) -> None:
    """A see-all update issues the unmodified statement."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_receipt_row())
    conn.execute = AsyncMock()

    await receipt_db.update_receipt(RESOURCE_ID, ReceiptUpdate.model_validate(dict(_make_receipt().model_dump(), expected_operation_revision=1)), can_see_all=True)

    call = next(c for c in conn.fetchrow.call_args_list if "confidence      = $1" in c.args[0])
    assert call.args[0] == UPDATE_RECEIPT_SQL
    assert "user_id" not in call.args[0]


@pytest.mark.asyncio
async def test_verify_receipt_scoped_to_owner(receipt_db: ReceiptDB) -> None:
    """verify_receipt cannot flip a row owned by someone else."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=None)

    await receipt_db.verify_receipt(RESOURCE_ID, user_id=USER_A, can_see_all=False)

    call = conn.fetchrow.call_args
    assert "user_id=$2" in call.args[0]
    assert call.args[3] is False
    assert call.args[1] == RESOURCE_ID
    assert call.args[2] == USER_A


@pytest.mark.asyncio
async def test_delete_receipt_scoped_to_owner(receipt_db: ReceiptDB) -> None:
    """delete_receipt cannot remove a row owned by someone else."""
    conn = AsyncMock()
    receipt_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=None)

    await receipt_db.delete_receipt(RESOURCE_ID, user_id=USER_A, can_see_all=False)

    call = conn.fetchrow.call_args
    assert "user_id=$2" in call.args[0]
    assert call.args[1] == RESOURCE_ID
    assert call.args[2] == USER_A
    assert call.args[3] is False
    assert conn.fetchrow.await_count == 1  # Invisible rows are not locked or deleted.


# ── ImageDB scoping ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_image_records_owner(image_db: ImageDB) -> None:
    """store_image stamps the new row with the uploading user's id."""
    conn = AsyncMock()
    image_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_image_row(user_id=USER_A))

    await image_db.store_image("/tmp/a.png", user_id=USER_A)

    call = conn.fetchrow.call_args
    # user_id is the 7th bound value in INSERT_IMAGE_SQL ($7), after thumbnail_path.
    assert call.args[7] == USER_A


@pytest.mark.asyncio
async def test_get_image_by_id_scoped_to_owner(image_db: ImageDB) -> None:
    """A non-see-all caller only fetches their own image rows."""
    conn = AsyncMock()
    image_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_image_row())

    await image_db.get_image_by_id(RESOURCE_ID, user_id=USER_A, can_see_all=False)

    call = conn.fetchrow.call_args
    assert call.args[0] == GET_IMAGE_SQL + " AND user_id = $2"
    assert call.args[1] == RESOURCE_ID
    assert call.args[2] == USER_A


@pytest.mark.asyncio
async def test_get_image_by_id_see_all_unscoped(image_db: ImageDB) -> None:
    """can_see_all fetches by id alone."""
    conn = AsyncMock()
    image_db._pool = _make_pool(conn)
    conn.fetchrow = AsyncMock(return_value=_image_row())

    await image_db.get_image_by_id(RESOURCE_ID, can_see_all=True)

    call = conn.fetchrow.call_args
    assert call.args[0] == GET_IMAGE_SQL
    assert call.args[1:] == (RESOURCE_ID,)


@pytest.mark.asyncio
async def test_list_images_scoped_to_owner(image_db: ImageDB) -> None:
    """list_images injects user_id as the first WHERE predicate."""
    conn = AsyncMock()
    image_db._pool = _make_pool(conn)
    conn.fetch = AsyncMock(return_value=[])

    await image_db.list_images(user_id=USER_A, can_see_all=False)

    call = conn.fetch.call_args
    sql = call.args[0]
    assert sql.startswith(LIST_IMAGES_BASE_SQL)
    assert "user_id = $1" in sql
    assert call.args[1] == USER_A


@pytest.mark.asyncio
async def test_list_images_see_all_unscoped(image_db: ImageDB) -> None:
    """A see-all listing carries no owner filter."""
    conn = AsyncMock()
    image_db._pool = _make_pool(conn)
    conn.fetch = AsyncMock(return_value=[])

    await image_db.list_images(can_see_all=True)

    assert "user_id" not in conn.fetch.call_args.args[0]


# ── API-level wiring: the handler threads the authenticated user through ──


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.check_connection = AsyncMock(return_value=True)
    provider.get_available_models = AsyncMock(return_value=[ModelInfo(id="test-model")])
    return provider


def _patch_app(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, conn: AsyncMock, provider: MagicMock
) -> None:
    """Patch settings, provider factory and the asyncpg pools the DB providers use."""
    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(images_api_module, "settings", settings)
    monkeypatch.setattr(system_api_module, "settings", settings)
    monkeypatch.setattr(main_module, "get_llm_provider", lambda cfg: provider)

    fake_asyncpg = MagicMock()
    fake_asyncpg.create_pool = AsyncMock(return_value=_make_pool(conn))
    monkeypatch.setattr(receipt_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(image_db_module, "asyncpg", fake_asyncpg)


@pytest.fixture
def user_api_context(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, AsyncMock], None, None]:
    """App booted against a mocked DB, authenticated as a non-admin user."""
    conn = AsyncMock()
    # {"missing": False} satisfies the to_regclass startup checks; bootstrap_admin
    # has no credentials in the test settings and any probe error is swallowed by
    # the lifespan, so startup succeeds.
    conn.fetchrow = AsyncMock(return_value={"missing": False})
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()
    provider = _make_provider()
    _patch_app(monkeypatch, settings, conn, provider)

    # A regular user (id=USER_A) without the see-all privilege.
    main_module.app.dependency_overrides[get_current_user] = lambda: User(
        id=USER_A, username="alice", is_admin=False, can_see_all=False
    )
    with TestClient(main_module.app) as client:
        yield client, conn
    main_module.app.dependency_overrides.clear()


def test_api_receipt_list_threads_user(user_api_context: tuple[TestClient, AsyncMock]) -> None:
    """GET /receipts scopes the list to the authenticated user's id."""
    client, conn = user_api_context
    conn.fetch = AsyncMock(return_value=[])

    response = client.get(RECEIPTS_URL)

    assert response.status_code == 200
    sql = conn.fetch.call_args.args[0]
    assert "user_id = $1" in sql
    assert conn.fetch.call_args.args[1] == USER_A


def test_api_receipt_detail_threads_user(user_api_context: tuple[TestClient, AsyncMock]) -> None:
    """GET /receipts/{id} scopes the detail lookup to the authenticated user."""
    client, conn = user_api_context
    conn.fetchrow = AsyncMock(return_value=_receipt_row(id=RESOURCE_ID))

    response = client.get(f"{RECEIPTS_URL}/{RESOURCE_ID}")

    assert response.status_code == 200
    call = conn.fetchrow.call_args
    assert call.args[0].startswith(GET_RECEIPT_WITH_IMAGE_SQL)
    assert "AND r.user_id = $2" in call.args[0]
    assert call.args[2] == USER_A
