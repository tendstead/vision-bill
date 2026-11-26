"""Business logic for collections: validation, uniqueness, delegation to CollectionDB."""

from datetime import date
from uuid import UUID

from ..model.collection import (
    Collection,
    CollectionCreate,
    CollectionDetail,
    CollectionSummary,
    CollectionUpdate,
    is_valid_hex_color,
    normalize_collection_name,
)
from ..provider.db.collection_db import CollectionDB
from .receipt_service import ReceiptService


class NameConflictError(Exception):
    """Raised when a collection name already exists for the user."""


class NotFoundError(Exception):
    """Raised when a referenced collection/receipt does not exist for the caller."""


def _validate_dates(start_date: date | None, end_date: date | None) -> None:
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")


class CollectionService:
    def __init__(self, receipt_service: ReceiptService):
        self._receipt_service = receipt_service
        self._db = CollectionDB(receipt_service.pool)

    @property
    def db_ready(self) -> bool:
        return self._receipt_service.db_ready

    async def create(self, body: CollectionCreate, user_id: UUID) -> Collection:
        name = normalize_collection_name(body.name)
        if not name:
            raise ValueError("Collection name must not be blank")
        if body.color is not None and not is_valid_hex_color(body.color):
            raise ValueError("color must be a hex value like #RGB or #RRGGBB")
        _validate_dates(body.start_date, body.end_date)
        if await self._db.find_by_name(user_id, name) is not None:
            raise NameConflictError(f"A collection named '{name}' already exists")
        return await self._db.create(
            user_id, name, body.color, body.start_date, body.end_date, active=body.active
        )

    async def list_summaries(
        self, user_id: UUID | None, can_see_all: bool
    ) -> list[CollectionSummary]:
        return await self._db.list_with_totals(user_id, can_see_all)

    async def get_detail(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> CollectionDetail | None:
        return await self._db.get_detail(collection_id, user_id, can_see_all)

    async def active_collection(self, user_id: UUID | None, can_see_all: bool) -> Collection | None:
        return await self._db.active_collection(user_id, can_see_all=can_see_all)

    async def activate(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> Collection | None:
        result = await self._db.activate(collection_id, user_id, can_see_all=can_see_all)
        if result is None:
            raise NotFoundError("Collection not found")
        return result

    async def deactivate(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> Collection | None:
        result = await self._db.deactivate(collection_id, user_id, can_see_all)
        if result is None:
            raise NotFoundError("Collection not found")
        return result

    async def update(
        self, collection_id: UUID, body: CollectionUpdate, user_id: UUID | None, can_see_all: bool
    ) -> Collection | None:
        current = await self._db.get(collection_id, user_id, can_see_all)
        if current is None:
            return None
        name = normalize_collection_name(body.name) if body.name is not None else current.name
        if not name:
            raise ValueError("Collection name must not be blank")
        color = body.color if body.color is not None else current.color
        start = body.start_date if body.start_date is not None else current.start_date
        end = body.end_date if body.end_date is not None else current.end_date
        if color is not None and not is_valid_hex_color(color):
            raise ValueError("color must be a hex value like #RGB or #RRGGBB")
        _validate_dates(start, end)
        if body.name is not None:
            existing = await self._db.find_by_name(
                user_id if user_id is not None else UUID(int=0), name
            )
            # A rename to the collection's own name (incl. a case variant) is not a
            # conflict — only a *different* collection with the same name is.
            if existing is not None and existing.id != collection_id:
                raise NameConflictError(f"A collection named '{name}' already exists")
        return await self._db.update(collection_id, user_id, can_see_all, name, color, start, end)

    async def delete(self, collection_id: UUID, user_id: UUID | None, can_see_all: bool) -> bool:
        return await self._db.delete(collection_id, user_id, can_see_all)

    async def assign(
        self, collection_id: UUID, receipt_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> None:
        if not await self._db.exists(collection_id, user_id, can_see_all):
            raise NotFoundError("Collection not found")
        # The receipt must also belong to the caller (or be visible to a
        # can_see_all admin); otherwise a user could attach a foreign receipt to
        # their own collection and read it back via get_detail.
        receipt = await self._receipt_service.get_receipt_by_id(
            receipt_id, user_id=user_id, can_see_all=can_see_all
        )
        if receipt is None:
            raise NotFoundError("Receipt not found")
        try:
            await self._db.assign(collection_id, receipt_id, user_id=user_id, can_see_all=can_see_all)
        except ValueError as exc:
            raise NotFoundError(str(exc)) from exc

    async def unassign(
        self, collection_id: UUID, receipt_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> bool:
        if not await self._db.exists(collection_id, user_id, can_see_all):
            raise NotFoundError("Collection not found")
        try:
            return await self._db.unassign(collection_id, receipt_id, user_id=user_id, can_see_all=can_see_all)
        except ValueError as exc:
            raise NotFoundError(str(exc)) from exc

    async def collection_ids_for_receipt(self, receipt_id: UUID) -> list[UUID]:
        return await self._db.collection_ids_for_receipt(receipt_id)

    async def set_receipt_collections(
        self, receipt_id: UUID, collection_ids: list[UUID], user_id: UUID | None, can_see_all: bool
    ) -> None:
        # The receipt must belong to the caller (or be visible to a can_see_all
        # admin); otherwise a user could rewrite another user's receipt links.
        receipt = await self._receipt_service.get_receipt_by_id(
            receipt_id, user_id=user_id, can_see_all=can_see_all
        )
        if receipt is None:
            raise NotFoundError("Receipt not found")
        # Each target collection must also belong to the caller; otherwise a user
        # could attach their receipt to a foreign collection and leak their data
        # into that collection's detail view (the reverse of the assign leak).
        for collection_id in collection_ids:
            if not await self._db.exists(collection_id, user_id, can_see_all):
                raise NotFoundError("Collection not found")
        try:
            await self._db.set_receipt_collections(receipt_id, collection_ids, user_id=user_id, can_see_all=can_see_all)
        except ValueError as exc:
            raise NotFoundError(str(exc)) from exc
