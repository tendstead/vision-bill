"""CollectionService — validation, uniqueness, delegation (mocked CollectionDB)."""

from datetime import date as Date
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from vision_bill.model.collection import CollectionCreate, CollectionUpdate
from vision_bill.service.collection_service import (
    CollectionService,
    NameConflictError,
    NotFoundError,
)

CID = UUID("00000000-0000-4000-8000-0000000000c1")
RID = UUID("00000000-0000-4000-8000-0000000000d1")
UID = UUID("00000000-0000-4000-8000-0000000000a1")


def _service() -> tuple[CollectionService, AsyncMock]:
    svc = CollectionService.__new__(CollectionService)
    db = AsyncMock()
    svc._db = db
    svc._receipt_service = AsyncMock()
    return svc, db


@pytest.mark.asyncio
async def test_create_rejects_bad_hex() -> None:
    svc, db = _service()
    with pytest.raises(ValueError):
        await svc.create(CollectionCreate(name="X", color="red"), UID)
    db.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_rejects_inverted_dates() -> None:
    svc, db = _service()
    with pytest.raises(ValueError):
        await svc.create(
            CollectionCreate(name="X", start_date=Date(2026, 6, 9), end_date=Date(2026, 6, 3)), UID
        )


@pytest.mark.asyncio
async def test_create_rejects_blank_name() -> None:
    svc, db = _service()
    with pytest.raises(ValueError):
        await svc.create(CollectionCreate(name="   "), UID)


@pytest.mark.asyncio
async def test_create_raises_name_conflict() -> None:
    svc, db = _service()
    db.find_by_name = AsyncMock(return_value=object())
    with pytest.raises(NameConflictError):
        await svc.create(CollectionCreate(name="Berlin"), UID)


@pytest.mark.asyncio
async def test_create_normalizes_and_delegates() -> None:
    svc, db = _service()
    db.find_by_name = AsyncMock(return_value=None)
    db.create = AsyncMock(return_value=object())
    await svc.create(CollectionCreate(name="  Berlin  "), UID)
    assert db.create.call_args.args[1] == "Berlin"  # normalized name


@pytest.mark.asyncio
async def test_assign_missing_collection_raises_not_found() -> None:
    svc, db = _service()
    db.exists = AsyncMock(return_value=False)
    with pytest.raises(NotFoundError):
        await svc.assign(CID, RID, UID, can_see_all=False)


@pytest.mark.asyncio
async def test_assign_foreign_receipt_raises_not_found() -> None:
    """A caller may only attach receipts they can see; a foreign receipt_id must
    not be linkable to their own collection (cross-user data leak)."""
    svc, db = _service()
    db.exists = AsyncMock(return_value=True)  # collection belongs to caller
    svc._receipt_service.get_receipt_by_id = AsyncMock(return_value=None)  # foreign receipt
    with pytest.raises(NotFoundError):
        await svc.assign(CID, RID, UID, can_see_all=False)
    db.assign.assert_not_called()
    svc._receipt_service.get_receipt_by_id.assert_awaited_once_with(
        RID, user_id=UID, can_see_all=False
    )


@pytest.mark.asyncio
async def test_assign_own_receipt_delegates() -> None:
    svc, db = _service()
    db.exists = AsyncMock(return_value=True)
    svc._receipt_service.get_receipt_by_id = AsyncMock(return_value=object())  # caller's receipt
    db.assign = AsyncMock()
    await svc.assign(CID, RID, UID, can_see_all=False)
    db.assign.assert_awaited_once_with(CID, RID, user_id=UID, can_see_all=False)


@pytest.mark.asyncio
async def test_update_omitted_fields_keep_current() -> None:
    svc, db = _service()
    current = type(
        "C", (), {"name": "Berlin", "color": "#111", "start_date": None, "end_date": None}
    )()
    db.get = AsyncMock(return_value=current)
    db.find_by_name = AsyncMock(return_value=None)
    db.update = AsyncMock(return_value=current)
    await svc.update(CID, CollectionUpdate(name="New Berlin"), UID, can_see_all=False)
    # color/dates resolved from current (unchanged)
    kwargs = db.update.call_args.args
    assert kwargs[4] == "#111"


@pytest.mark.asyncio
async def test_update_same_name_is_not_conflict() -> None:
    svc, db = _service()
    current = type(
        "C",
        (),
        {"id": CID, "name": "Berlin", "color": "#111", "start_date": None, "end_date": None},
    )()
    db.get = AsyncMock(return_value=current)
    # find_by_name returns the SAME collection (self) -> resubmitting its own name
    # (e.g. a color-only change) must NOT be a conflict.
    db.find_by_name = AsyncMock(return_value=current)
    db.update = AsyncMock(return_value=current)
    await svc.update(CID, CollectionUpdate(name="Berlin", color="#222"), UID, can_see_all=False)
    db.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_activate_delegates() -> None:
    svc, db = _service()
    db.activate = AsyncMock(return_value=object())
    await svc.activate(CID, UID, can_see_all=False)
    db.activate.assert_awaited_once_with(CID, UID, can_see_all=False)


@pytest.mark.asyncio
async def test_activate_not_found_raises() -> None:
    svc, db = _service()
    db.activate = AsyncMock(return_value=None)
    with pytest.raises(NotFoundError):
        await svc.activate(CID, UID, can_see_all=False)


@pytest.mark.asyncio
async def test_deactivate_not_found_raises() -> None:
    svc, db = _service()
    db.deactivate = AsyncMock(return_value=None)
    with pytest.raises(NotFoundError):
        await svc.deactivate(CID, UID, can_see_all=False)


@pytest.mark.asyncio
async def test_active_collection_delegates() -> None:
    svc, db = _service()
    db.active_collection = AsyncMock(return_value=None)
    assert await svc.active_collection(UID, can_see_all=False) is None
    db.active_collection.assert_awaited_once_with(UID, can_see_all=False)


@pytest.mark.asyncio
async def test_create_passes_active_flag() -> None:
    svc, db = _service()
    db.find_by_name = AsyncMock(return_value=None)
    db.create = AsyncMock(return_value=object())
    await svc.create(CollectionCreate(name="Berlin", active=True), UID)
    assert db.create.call_args.kwargs["active"] is True


@pytest.mark.asyncio
async def test_set_receipt_collections_rejects_foreign_receipt() -> None:
    """A caller cannot (re)assign collections on a receipt they cannot see."""
    svc, db = _service()
    svc._receipt_service.get_receipt_by_id = AsyncMock(return_value=None)  # foreign receipt
    with pytest.raises(NotFoundError):
        await svc.set_receipt_collections(RID, [CID], UID, can_see_all=False)
    db.set_receipt_collections.assert_not_called()


@pytest.mark.asyncio
async def test_set_receipt_collections_rejects_foreign_collection() -> None:
    """A caller cannot attach their receipt to a collection they do not own
    (would leak their data into a foreign collection's detail view)."""
    svc, db = _service()
    svc._receipt_service.get_receipt_by_id = AsyncMock(return_value=object())  # caller's receipt
    db.exists = AsyncMock(return_value=False)  # the target collection is someone else's
    with pytest.raises(NotFoundError):
        await svc.set_receipt_collections(RID, [CID], UID, can_see_all=False)
    db.set_receipt_collections.assert_not_called()


@pytest.mark.asyncio
async def test_set_receipt_collections_delegates_when_owned() -> None:
    svc, db = _service()
    svc._receipt_service.get_receipt_by_id = AsyncMock(return_value=object())
    db.exists = AsyncMock(return_value=True)  # every target collection is the caller's
    db.set_receipt_collections = AsyncMock()
    await svc.set_receipt_collections(RID, [CID], UID, can_see_all=False)
    db.set_receipt_collections.assert_awaited_once_with(RID, [CID], user_id=UID, can_see_all=False)
