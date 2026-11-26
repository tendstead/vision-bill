"""Owns the asyncpg pool and all SQL for collections."""

from collections.abc import Mapping
from datetime import date as Date
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg

from ...model.collection import Collection, CollectionDetail, CollectionSummary, CurrencyTotal
from ...model.db.receipt import ReceiptRow
from .receipt_db import MEMBERSHIP_LOCK_SQL, ReceiptDB

CREATE_COLLECTION_SQL = (
    "INSERT INTO collections (user_id, name, color, start_date, end_date, active) "
    "VALUES ($1, $2, $3, $4, $5, $6) RETURNING *"
)
ACTIVATE_DEACTIVATE_SIBLINGS_SQL = "UPDATE collections SET active = FALSE WHERE user_id = $1"
SET_ACTIVE_SQL = "UPDATE collections SET active = TRUE WHERE id = $1 RETURNING *"
SET_INACTIVE_SQL = "UPDATE collections SET active = FALSE WHERE id = $1 RETURNING *"
GET_ACTIVE_SQL = "SELECT * FROM collections WHERE active = TRUE"
GET_COLLECTION_SQL = "SELECT * FROM collections WHERE id = $1"
# One query per list (no N+1): each collection carries its receipt count via a
# correlated subquery exposed as `cnt`.
LIST_COLLECTIONS_BASE_SQL = (
    "SELECT c.*, (SELECT COUNT(*)::int FROM receipt_collections rc "
    "JOIN receipts r ON r.id = rc.receipt_id WHERE rc.collection_id = c.id{receipt_scope}) "
    "AS cnt FROM collections c WHERE 1=1{collection_scope} ORDER BY c.name"
)
# Placeholder numbering must be contiguous from $1: an argument that is passed
# but never referenced (the old `WHERE id = $2` + duplicated id) makes Postgres
# reject the prepared statement with IndeterminateDatatypeError.
UPDATE_COLLECTION_SQL = (
    "UPDATE collections SET name = $2, color = $3, start_date = $4, end_date = $5 "
    "WHERE id = $1 RETURNING *"
)
DELETE_COLLECTION_SQL = "DELETE FROM collections WHERE id = $1 RETURNING id"
EXISTS_COLLECTION_SQL = "SELECT 1 AS x FROM collections WHERE id = $1"
COLLECTION_OWNER_CHECK_SQL = "SELECT 1 AS x FROM collections WHERE id = $1 AND user_id = $2"
TOTALS_PER_CURRENCY_SQL = (
    "SELECT r.currency AS currency, SUM(r.total) AS total "
    "FROM receipt_collections rc JOIN receipts r ON r.id = rc.receipt_id "
    "WHERE rc.collection_id = $1"
)
RECEIPTS_IN_COLLECTION_SQL = (
    "SELECT r.* FROM receipts r JOIN receipt_collections rc "
    "ON rc.receipt_id = r.id WHERE rc.collection_id = $1"
)
ASSIGN_SQL = (
    "INSERT INTO receipt_collections (receipt_id, collection_id) "
    "VALUES ($1, $2) ON CONFLICT (receipt_id, collection_id) DO NOTHING"
)
UNASSIGN_SQL = (
    "DELETE FROM receipt_collections WHERE receipt_id = $1 AND collection_id = $2 RETURNING 1"
)
LIST_COLLECTION_IDS_FOR_RECEIPT_SQL = (
    "SELECT collection_id FROM receipt_collections WHERE receipt_id = $1"
)
DELETE_RECEIPT_COLLECTIONS_SQL = "DELETE FROM receipt_collections WHERE receipt_id = $1"


class CollectionDB:
    """All SQL for collections; reuses the receipt DB's asyncpg pool (no own pool)."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @property
    def pool(self) -> asyncpg.Pool:
        return self._pool

    @staticmethod
    def _collection_from_row(row: Mapping[str, Any]) -> Collection:
        return Collection(
            id=row["id"],
            name=row["name"],
            color=row["color"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            active=row["active"],
            created_at=row["created_at"].date() if row["created_at"] is not None else None,
        )

    def _owner_scope(
        self, args: list[Any], user_id: UUID | None, can_see_all: bool, *, column: str = "user_id"
    ) -> str:
        """Return an extra WHERE fragment restricting to the owner when needed."""
        if not can_see_all and user_id is not None:
            args.append(user_id)
            return f" AND {column} = ${len(args)}"
        return ""

    def _scoped_returning(
        self, base_sql: str, args: list[Any], user_id: UUID | None, can_see_all: bool
    ) -> str:
        """Like ``_owner_scope`` but for statements ending in a RETURNING clause.

        The owner fragment must go in the WHERE clause, i.e. BEFORE ``RETURNING`` —
        appending it after would parse as part of the RETURNING expression
        (``RETURNING id AND user_id = $N``) and fail with a datatype error.
        """
        if not can_see_all and user_id is not None:
            args.append(user_id)
            return base_sql.replace(" RETURNING ", f" AND user_id = ${len(args)} RETURNING ")
        return base_sql

    async def create(
        self,
        user_id: UUID,
        name: str,
        color: str | None,
        start_date: Date | None,
        end_date: Date | None,
        active: bool = False,
    ) -> Collection:
        async with self.pool.acquire() as conn, conn.transaction():
            if active:
                await conn.execute(MEMBERSHIP_LOCK_SQL)
                await conn.execute(ACTIVATE_DEACTIVATE_SIBLINGS_SQL, user_id)
            row = await conn.fetchrow(
                CREATE_COLLECTION_SQL, user_id, name, color, start_date, end_date, active
            )
        return self._collection_from_row(row)

    async def list_with_totals(
        self, user_id: UUID | None, can_see_all: bool
    ) -> list[CollectionSummary]:
        args: list[Any] = []
        sql = LIST_COLLECTIONS_BASE_SQL.format(
            receipt_scope=self._owner_scope(args, user_id, can_see_all, column="r.user_id"),
            collection_scope=self._owner_scope(args, user_id, can_see_all, column="c.user_id"),
        )
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
            summaries: list[CollectionSummary] = []
            for row in rows:
                base = self._collection_from_row(row)
                count = int(row["cnt"])
                if count > 0:
                    totals = await self._currency_totals(conn, row["id"], user_id, can_see_all)
                else:
                    totals = []
                summaries.append(
                    CollectionSummary(**base.model_dump(), receipt_count=count, totals=totals)
                )
        return summaries

    async def get(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> Collection | None:
        args: list[Any] = [collection_id]
        sql = GET_COLLECTION_SQL + self._owner_scope(args, user_id, can_see_all)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return self._collection_from_row(row) if row is not None else None

    async def find_by_name(self, user_id: UUID, name: str) -> Collection | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM collections WHERE user_id = $1 AND lower(name) = lower($2)",
                user_id,
                name,
            )
        return self._collection_from_row(row) if row is not None else None

    async def update(
        self,
        collection_id: UUID,
        user_id: UUID | None,
        can_see_all: bool,
        name: str | None,
        color: str | None,
        start_date: Date | None,
        end_date: Date | None,
    ) -> Collection | None:
        """Update all four fields in one statement.

        ``None`` means "set the column to NULL" — it is NOT "leave unchanged".
        The caller (the service) must pass the collection's current values for
        any field it does not want to change, otherwise a blank write will fail
        the NOT NULL constraint on ``name``.
        """
        args: list[Any] = [collection_id, name, color, start_date, end_date]
        sql = self._scoped_returning(UPDATE_COLLECTION_SQL, args, user_id, can_see_all)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return self._collection_from_row(row) if row is not None else None

    async def delete(self, collection_id: UUID, user_id: UUID | None, can_see_all: bool) -> bool:
        args: list[Any] = [collection_id]
        sql = self._scoped_returning(DELETE_COLLECTION_SQL, args, user_id, can_see_all)
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(MEMBERSHIP_LOCK_SQL)
            visible = await conn.fetchrow(
                EXISTS_COLLECTION_SQL + self._owner_scope([collection_id], user_id, can_see_all),
                *args,
            )
            if visible is None:
                return False
            linked = await conn.fetch(
                "SELECT receipt_id FROM receipt_collections WHERE collection_id=$1 ORDER BY receipt_id",
                collection_id,
            )
            locked = []
            for link in linked:
                row = await ReceiptDB._lock_review_receipt(conn, link["receipt_id"], None, True)
                if row is not None:
                    locked.append(row["id"])
            deleted = await conn.fetchrow(sql, *args)
            if deleted is not None:
                for receipt_id in locked:
                    await self._membership_event(conn, receipt_id, user_id)
            return deleted is not None

    async def exists(self, collection_id: UUID, user_id: UUID | None, can_see_all: bool) -> bool:
        args: list[Any] = [collection_id]
        sql = EXISTS_COLLECTION_SQL + self._owner_scope(args, user_id, can_see_all)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return row is not None

    async def active_collection(self, user_id: UUID | None, can_see_all: bool) -> Collection | None:
        args: list[Any] = []
        sql = GET_ACTIVE_SQL + self._owner_scope(args, user_id, can_see_all) + " LIMIT 1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return self._collection_from_row(row) if row is not None else None

    async def activate(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> Collection | None:
        """Mark active, deactivating the user's other active collection (one active per user)."""
        args: list[Any] = [collection_id]
        set_sql = self._scoped_returning(SET_ACTIVE_SQL, args, user_id, can_see_all)
        visible_args: list[Any] = [collection_id]
        visible_sql = GET_COLLECTION_SQL + self._owner_scope(visible_args, user_id, can_see_all)
        async with self.pool.acquire() as conn, conn.transaction():
            if await conn.fetchrow(visible_sql, *visible_args) is None:
                return None
            # Same order as capture/delete: membership ownership before rows.
            # Recheck visibility after waiting; the target could have been deleted.
            await conn.execute(MEMBERSHIP_LOCK_SQL)
            target = await conn.fetchrow(visible_sql + " FOR UPDATE", *visible_args)
            if target is None:
                return None
            await conn.execute(ACTIVATE_DEACTIVATE_SIBLINGS_SQL, target["user_id"])
            row = await conn.fetchrow(set_sql, *args)
        return self._collection_from_row(row) if row is not None else None

    async def deactivate(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> Collection | None:
        args: list[Any] = [collection_id]
        set_sql = self._scoped_returning(SET_INACTIVE_SQL, args, user_id, can_see_all)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(set_sql, *args)
        return self._collection_from_row(row) if row is not None else None

    @staticmethod
    async def _membership_event(
        conn: asyncpg.Connection, receipt_id: UUID, actor: UUID | None
    ) -> None:
        row = await conn.fetchrow(
            "UPDATE receipts SET operation_revision=operation_revision+1 WHERE id=$1 RETURNING *",
            receipt_id,
        )
        if row is not None:
            await ReceiptDB._append_review_history(conn, row, "collections_changed", actor)

    @staticmethod
    async def _lock_membership(
        conn: asyncpg.Connection,
        collection_id: UUID,
        receipt_id: UUID,
        user_id: UUID | None,
        can_see_all: bool,
    ) -> None:
        await conn.execute(MEMBERSHIP_LOCK_SQL)
        if await ReceiptDB._lock_review_receipt(conn, receipt_id, user_id, can_see_all) is None:
            raise ValueError("receipt_not_found")
        visible = await conn.fetchrow(
            "SELECT id FROM collections WHERE id=$1 AND ($3 OR $2::uuid IS NULL OR user_id=$2) FOR SHARE",
            collection_id,
            user_id,
            can_see_all,
        )
        if visible is None:
            raise ValueError("collection_not_found")

    async def assign(
        self,
        collection_id: UUID,
        receipt_id: UUID,
        *,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> None:
        async with self.pool.acquire() as conn, conn.transaction():
            await self._lock_membership(conn, collection_id, receipt_id, user_id, can_see_all)
            changed = await conn.fetchrow(
                ASSIGN_SQL + " RETURNING receipt_id", receipt_id, collection_id
            )
            if changed is not None:
                await self._membership_event(conn, receipt_id, user_id)

    async def unassign(
        self,
        collection_id: UUID,
        receipt_id: UUID,
        *,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> bool:
        async with self.pool.acquire() as conn, conn.transaction():
            await self._lock_membership(conn, collection_id, receipt_id, user_id, can_see_all)
            row = await conn.fetchrow(UNASSIGN_SQL, receipt_id, collection_id)
            if row is not None:
                await self._membership_event(conn, receipt_id, user_id)
            return row is not None

    async def collection_ids_for_receipt(self, receipt_id: UUID) -> list[UUID]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(LIST_COLLECTION_IDS_FOR_RECEIPT_SQL, receipt_id)
        return [row["collection_id"] for row in rows]

    async def set_receipt_collections(
        self,
        receipt_id: UUID,
        collection_ids: list[UUID],
        *,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> None:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(MEMBERSHIP_LOCK_SQL)
            if await ReceiptDB._lock_review_receipt(conn, receipt_id, user_id, can_see_all) is None:
                raise ValueError("receipt_not_found")
            current = set(await ReceiptDB._visible_collection_ids(
                conn, receipt_id, user_id, can_see_all
            ))
            if current == set(collection_ids):
                return
            await ReceiptDB._set_collections(conn, receipt_id, collection_ids, user_id, can_see_all)
            await self._membership_event(conn, receipt_id, user_id)

    async def receipts_in_collection(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> list[ReceiptRow]:
        args: list[Any] = [collection_id]
        sql = RECEIPTS_IN_COLLECTION_SQL + self._owner_scope(
            args, user_id, can_see_all, column="r.user_id"
        )
        sql += " ORDER BY r.date DESC"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [self._receipt_row(r) for r in rows]

    @staticmethod
    def _receipt_row(row: Mapping[str, Any]) -> ReceiptRow:
        from ...provider.db.receipt_db import ReceiptDB

        return ReceiptDB._receipt_row_from_record(row)

    async def _currency_totals(
        self, conn: asyncpg.Connection, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> list[CurrencyTotal]:
        args: list[Any] = [collection_id]
        sql = TOTALS_PER_CURRENCY_SQL + self._owner_scope(
            args, user_id, can_see_all, column="r.user_id"
        )
        rows = await conn.fetch(sql + " GROUP BY r.currency ORDER BY total DESC", *args)
        return [CurrencyTotal(currency=r["currency"], total=Decimal(r["total"])) for r in rows]

    # Convenience detail assembly used by the service.
    async def get_detail(
        self, collection_id: UUID, user_id: UUID | None, can_see_all: bool
    ) -> CollectionDetail | None:
        coll = await self.get(collection_id, user_id, can_see_all)
        if coll is None:
            return None
        async with self.pool.acquire() as conn:
            totals = await self._currency_totals(conn, collection_id, user_id, can_see_all)
        receipts = await self.receipts_in_collection(collection_id, user_id, can_see_all)
        return CollectionDetail(**coll.model_dump(), totals=totals, receipts=receipts)
