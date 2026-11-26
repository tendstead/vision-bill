import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date as Date
from datetime import time as Time
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast
from uuid import UUID

import asyncpg

from ...config import PGSettings
from ...model.db.image import ImagePart, ImageRow
from ...model.db.receipt import (
    LineItemRow,
    MerchantOptions,
    ReceiptRow,
    ReceiptWithDetails,
    TaxLineRow,
)
from ...model.extraction import (
    ClassificationEnrichment,
    DraftLineItem,
    DraftTaxLine,
    ExtractionDraft,
    assert_metadata_only,
)
from ...model.item_category import CATEGORY_CODE_SET, SpendingCategory, normalize_history_key
from ...model.merchant_catalog import is_generic_merchant, merchant_category
from ...model.ocr import SourceRef
from ...model.receipt import Category, Receipt
from ...model.review import OriginalAvailability, ReviewEvaluation
from ...model.search import ProductPurchase
from ...model.statistics import (
    CurrencyStatistics,
    NamedStatistics,
    ReceiptStatistics,
    WeekdayStatistics,
    WeeklyStatistics,
)
from ...model.workflow import AccessScope, ReviewPreview, ReviewRequest
from ...service.receipt_review import (
    evaluate_receipt,
    material_document,
    persistent_findings,
    validate_acknowledgements,
)
from . import merchants
from .deadline import connection

# ── SQL (DML; DDL lives in alembic/versions/0001_initial_schema.py) ──

INSERT_RECEIPT_SQL = """
    INSERT INTO receipts
        (confidence, merchant_name, merchant_address, receipt_number, date, time,
         currency, category, subtotal, discount_total, tax_total, tip, total,
         payment_method, status, image_id, verified, user_id, merchant_display_name)
    VALUES
        ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
    RETURNING *
"""

UPDATE_RECEIPT_BY_IMAGE_SQL = """
    UPDATE receipts SET
        confidence = $1,
        merchant_name = $2,
        merchant_address = $3,
        receipt_number = $4,
        date = $5,
        time = $6,
        currency = $7,
        category = $8,
        subtotal = $9,
        discount_total = $10,
        tax_total = $11,
        tip = $12,
        total = $13,
        payment_method = $14,
        status = $15,
        verified = $16,
        user_id = $17,
        merchant_display_name = $19
    WHERE id = (
        SELECT id FROM receipts
        WHERE image_id = $18
        ORDER BY created_at ASC, id ASC
        LIMIT 1
    )
    RETURNING *
"""

UPDATE_RECEIPT_SQL = """
    UPDATE receipts SET
        confidence      = $1,
        merchant_name   = $2,
        merchant_address = $3,
        receipt_number  = $4,
        date            = $5,
        time            = $6,
        currency        = $7,
        category        = $8,
        subtotal        = $9,
        discount_total  = $10,
        tax_total       = $11,
        tip             = $12,
        total           = $13,
        payment_method  = $14,
        merchant_display_name = $16
    WHERE id = $15
    RETURNING *
"""

DELETE_LINE_ITEMS_SQL = "DELETE FROM line_items WHERE receipt_id = $1"
DELETE_TAXES_SQL = "DELETE FROM taxes WHERE receipt_id = $1"
DELETE_RECEIPT_SQL = "DELETE FROM receipts WHERE id = $1 RETURNING *"

INSERT_LINE_ITEM_SQL = """
    INSERT INTO line_items
        (receipt_id, description, quantity, unit_price,
         total_price, tags, position, id, spending_category, category_source,
         original_description)
    VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8, gen_random_uuid()), $9, $10, $11)
"""

GET_CATEGORY_ALIAS_SQL = """
    SELECT category FROM item_category_aliases
    WHERE user_id = $1 AND merchant_key = $2 AND description_key = $3
"""
LIST_CATEGORY_ALIASES_SQL = """
    SELECT description_key, category FROM item_category_aliases
    WHERE user_id = $1 AND merchant_key = $2 AND description_key = ANY($3::text[])
"""
UPSERT_CATEGORY_ALIAS_SQL = """
    INSERT INTO item_category_aliases (user_id, merchant_key, description_key, category)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT (user_id, merchant_key, description_key)
    DO UPDATE SET category = EXCLUDED.category
"""
DELETE_CATEGORY_ALIAS_SQL = """
    DELETE FROM item_category_aliases
    WHERE user_id = $1 AND merchant_key = $2 AND description_key = $3
"""

INSERT_TAX_SQL = """
    INSERT INTO taxes
        (receipt_id, name, rate, amount, position)
    VALUES ($1, $2, $3, $4, $5)
"""

GET_RECEIPT_SQL = "SELECT * FROM receipts WHERE id = $1"
GET_RECEIPT_BY_IMAGE_ID_SQL = (
    "SELECT * FROM receipts WHERE image_id = $1 ORDER BY created_at ASC, id ASC LIMIT 1"
)
# Join the images table so the detail response can expose the resolved
# image_path without a second round-trip.
GET_RECEIPT_WITH_IMAGE_SQL = """
    SELECT r.*, i.image_path, i.thumbnail_path,
           COALESCE(NULLIF(initial.snapshot -> 'classification_enrichment', 'null'::jsonb),
                    e.envelope -> 'classification_enrichment') AS classification_enrichment
    FROM receipts r
    LEFT JOIN images i ON i.id = r.image_id
    LEFT JOIN receipt_extractions e ON e.id = r.extraction_id AND e.image_id = r.image_id
    LEFT JOIN receipt_review_revisions initial ON initial.receipt_id = r.id
        AND initial.operation_revision = 1 AND initial.event_kind = 'checkpoint_completed'
    WHERE r.id = $1
"""
LIST_RECEIPTS_BASE_SQL = (
    "SELECT r.*, (SELECT i.thumbnail_path FROM images i WHERE i.id = r.image_id) "
    "AS thumbnail_path FROM receipts r"
)
SEARCH_PRODUCTS_BASE_SQL = """
    SELECT
        li.receipt_id,
        li.description,
        li.quantity,
        li.unit_price,
        r.merchant_name,
        r.merchant_display_name,
        r.date,
        r.time,
        r.currency
    FROM line_items AS li
    JOIN receipts AS r ON r.id = li.receipt_id
    WHERE r.status = 'verified'
      AND r.verified = TRUE
      AND li.description ILIKE $1
"""
STATS_CURRENCY_SQL = """
    SELECT
        currency,
        COUNT(*)::int AS receipt_count,
        SUM(total) AS total,
        AVG(total) AS average,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY total)::numeric AS median,
        MIN(total) AS minimum,
        MAX(total) AS maximum,
        SUM(subtotal) AS subtotal,
        COALESCE(SUM(discount_total), 0) AS discounts,
        SUM(tax_total) AS taxes,
        SUM(COALESCE(tip, 0)) AS tips
    FROM receipts
    WHERE status = 'verified' AND verified = TRUE{scope}
    GROUP BY currency
    ORDER BY total DESC, currency
"""
STATS_NAMED_SQL = """
    SELECT
        {field} AS name,
        currency,
        COUNT(*)::int AS receipt_count,
        SUM(total) AS total,
        AVG(total) AS average
    FROM receipts
    WHERE status = 'verified' AND verified = TRUE{scope}
    GROUP BY {field}, currency
    ORDER BY total DESC, name, currency
"""
STATS_WEEKDAY_SQL = """
    SELECT
        EXTRACT(ISODOW FROM date)::int AS weekday,
        currency,
        COUNT(*)::int AS receipt_count,
        SUM(total) AS total,
        AVG(total) AS average
    FROM receipts
    WHERE status = 'verified' AND verified = TRUE{scope}
    GROUP BY weekday, currency
    ORDER BY weekday, currency
"""
STATS_WEEKLY_SQL = """
    SELECT
        date_trunc('week', date)::date AS week_start,
        currency,
        COUNT(*)::int AS receipt_count,
        SUM(total) AS total,
        AVG(total) AS average
    FROM receipts
    WHERE status = 'verified' AND verified = TRUE
      AND date >= date_trunc('week', CURRENT_DATE) - ${weeks_param} * INTERVAL '1 week'{scope}
    GROUP BY week_start, currency
    ORDER BY week_start, currency
"""
LIST_LINE_ITEMS_SQL = "SELECT * FROM line_items WHERE receipt_id = $1 ORDER BY position, id"
LIST_TAXES_SQL = "SELECT * FROM taxes WHERE receipt_id = $1 ORDER BY position, id"
LIST_COLLECTION_IDS_FOR_RECEIPT_SQL = (
    "SELECT collection_id FROM receipt_collections WHERE receipt_id = $1"
)
LIST_TAGS_SQL = "SELECT name FROM tags ORDER BY name"
INSERT_TAG_SQL = """
    INSERT INTO tags (name) SELECT $1::text
    WHERE NOT EXISTS (SELECT FROM tags WHERE lower(name) = lower($1))
    ON CONFLICT (name) DO NOTHING RETURNING name
"""
EXISTING_TAG_SQL = """
    SELECT name FROM tags WHERE lower(name) = lower($1)
    ORDER BY (name = $1) DESC, name LIMIT 1
"""
VERIFY_RECEIPT_SQL = (
    "UPDATE receipts SET status = 'verified', verified = TRUE WHERE id = $1 RETURNING *"
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemCategoryDecision:
    id: UUID | None
    original_description: str
    category: str
    source: str


MEMBERSHIP_LOCK_SQL = "SELECT pg_advisory_xact_lock(78830741390211)"

OriginalChecker = Callable[[list[ImagePart]], Awaitable[list[OriginalAvailability]]]


class ReceiptDB:
    """Owns the asyncpg pool and all SQL for receipts."""

    def __init__(self, settings: PGSettings, *, original_checker: OriginalChecker | None = None):
        self._original_checker = original_checker
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    # ── Connection pool lifecycle ────────────────────────────────────
    async def init_db(self) -> None:
        """Create the connection pool and check that the schema is migrated."""
        if self._pool is not None:
            logger.warning("Database pool already initialised - skipping")
            return

        dsn = self._settings.pg_dsn
        logger.info("Creating asyncpg connection pool")
        self._pool = await asyncpg.create_pool(dsn=dsn)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT to_regclass('receipts') IS NULL AS missing")
        if row["missing"]:
            logger.warning(
                "Schema not initialised - run 'uv run alembic upgrade head' before starting the app"
            )

    async def destroy_db(self) -> None:
        """Close the connection pool and release all resources."""
        if self._pool is not None:
            logger.info("Closing database connection pool")
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    @property
    def is_ready(self) -> bool:
        """Whether the connection pool has been initialised."""
        return self._pool is not None

    # ── Helpers ──────────────────────────────────────────────────────

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool not initialised. Call init_db() first.")
        return self._pool

    @staticmethod
    def _receipt_time_value(receipt: Receipt) -> Time | None:
        """Convert the model's ISO time string ("HH:MM" / "HH:MM:SS") to a Time."""
        return Time.fromisoformat(receipt.time) if receipt.time else None

    @staticmethod
    def _receipt_row_from_record(row: Mapping[str, Any]) -> ReceiptRow:
        """Map a raw receipts row to a ReceiptRow with explicit conversions.

        Accepts either an asyncpg.Record or a plain dict. Extra keys (e.g. the
        joined ``image_path`` from the detail query) are dropped.
        """
        d = dict(row)
        d.pop("image_path", None)
        t: Time | None = d["time"]
        d["time"] = t.isoformat() if t is not None else None
        created_at: Date | None = d["created_at"].date() if d["created_at"] is not None else None
        d["created_at"] = created_at
        return ReceiptRow(**d)

    @staticmethod
    def _line_item_row_from_record(row: asyncpg.Record) -> LineItemRow:
        d = dict(row)
        return LineItemRow(
            id=d["id"],
            receipt_id=d["receipt_id"],
            description=d["description"],
            original_description=d.get("original_description"),
            spending_category=d.get("spending_category", "unknown"),
            category_source=d.get("category_source", "unknown"),
            quantity=d["quantity"],
            unit_price=Decimal(d["unit_price"]) if d["unit_price"] is not None else None,
            total_price=Decimal(d["total_price"]),
            tags=list(d.get("tags") or []),
            position=d["position"],
        )

    @staticmethod
    def _product_purchase_from_record(row: Mapping[str, Any]) -> ProductPurchase:
        d = dict(row)
        t: Time | None = d["time"]
        return ProductPurchase(
            receipt_id=d["receipt_id"],
            description=d["description"],
            merchant_name=d["merchant_name"],
            merchant_display_name=d.get("merchant_display_name"),
            date=d["date"],
            time=t.isoformat() if t is not None else None,
            quantity=d["quantity"],
            unit_price=Decimal(d["unit_price"]) if d["unit_price"] is not None else None,
            currency=d["currency"],
        )

    @staticmethod
    def _tax_row_from_record(row: asyncpg.Record) -> TaxLineRow:
        d = dict(row)
        return TaxLineRow(
            id=d["id"],
            receipt_id=d["receipt_id"],
            name=d["name"],
            rate=d["rate"],
            amount=Decimal(d["amount"]),
        )

    async def _insert_children(
        self, conn: asyncpg.Connection, receipt_id: UUID, receipt: Receipt,
        decisions: list[ItemCategoryDecision],
    ) -> None:
        """Insert the receipt's line items and taxes (shared insert logic)."""
        if receipt.line_items:
            for index, (item, decision) in enumerate(zip(receipt.line_items, decisions, strict=True)):
                await conn.execute(
                    INSERT_LINE_ITEM_SQL,
                    receipt_id,
                    item.description,
                    item.quantity,
                    item.unit_price,
                    item.total_price,
                    list(item.tags),
                    index,
                    decision.id,
                    decision.category,
                    decision.source,
                    decision.original_description,
                )

        if receipt.taxes:
            for position, tax in enumerate(receipt.taxes):
                await conn.execute(
                    INSERT_TAX_SQL,
                    receipt_id,
                    tax.name,
                    tax.rate,
                    tax.amount,
                    position,
                )

    async def suggested_merchant_name(
        self, user_id: UUID | None, merchant_name: str, merchant_address: str | None,
        *, deadline: float | None = None,
    ) -> str | None:
        async with connection(self.pool, deadline=deadline) as conn:
            return await merchants.suggested_name(conn, user_id, merchant_name, merchant_address, printed_fallback=False)

    async def merchant_options(
        self, scope: AccessScope, printed: str, address: str | None, query: str,
        receipt_id: UUID | None, image_id: UUID | None,
    ) -> MerchantOptions | None:
        async with self.pool.acquire() as conn:
            owner: UUID | None = scope.user_id
            if receipt_id or image_id:
                table = "receipts" if receipt_id else "images"
                resource = await conn.fetchrow(
                    f"SELECT user_id FROM {table} WHERE id=$1 AND (user_id=$2 OR $3)",
                    receipt_id or image_id, scope.user_id, scope.can_see_all,
                )
                if resource is None:
                    return None
                if resource["user_id"] != scope.user_id:
                    owner = None
            return await merchants.options(conn, owner, printed, address, query)

    async def get_category_suggestion(
        self, user_id: UUID, merchant_name: str, description: str, *,
        display_name: str | None = None, address: str | None = None, receipt_id: UUID | None = None,
    ) -> str | None:
        """Receipt editors only consult personal product history on their own receipt."""
        description_key = normalize_history_key(description)
        if not merchant_name.strip() or not description_key:
            return None
        async with self.pool.acquire() as conn:
            if receipt_id is not None and not await conn.fetchval(
                "SELECT EXISTS (SELECT FROM receipts WHERE id=$1 AND user_id=$2)", receipt_id, user_id,
            ):
                return None
            display = display_name or await merchants.suggested_name(conn, user_id, merchant_name, address)
            for key in self._merchant_history_keys(merchant_name, display, address):
                row = await conn.fetchrow(GET_CATEGORY_ALIAS_SQL, user_id, key, description_key)
                category = row["category"] if row else None
                if isinstance(category, str) and category in CATEGORY_CODE_SET and category != "unknown":
                    return category
        return None

    @staticmethod
    def _merchant_history_keys(
        printed: str, display: str | None, address: str | None,
    ) -> list[str]:
        unresolved = normalize_history_key(display or printed) == normalize_history_key(printed)
        if is_generic_merchant(display or printed) or (unresolved and merchants.needs_address(printed, address)):
            # Avoid sharing product learning across unrelated, unnamed shops.
            if not address:
                return []
            identity = json.dumps([normalize_history_key(printed), normalize_history_key(address)])
            return [f"branch:{sha256(identity.encode()).hexdigest()}"]
        names = [normalize_history_key(display or printed)]
        if not merchants.needs_address(printed, address):
            names.append(normalize_history_key(printed))
        return list(dict.fromkeys(
            name if len(name.encode()) <= 800 else f"merchant:{sha256(name.encode()).hexdigest()}"
            for name in names if name
        ))

    async def _history_for_items(
        self, conn: asyncpg.Connection, user_id: UUID | None, merchant_name: str,
        descriptions: list[str], display: str | None = None, address: str | None = None,
    ) -> dict[str, str]:
        keys = sorted({normalize_history_key(value) for value in descriptions if value.strip()})
        if user_id is None or not keys:
            return {}
        result: dict[str, str] = {}
        for merchant_key in self._merchant_history_keys(merchant_name, display, address):
            rows = await conn.fetch(LIST_CATEGORY_ALIASES_SQL, user_id, merchant_key, keys)
            for row in rows:
                if row["category"] in CATEGORY_CODE_SET and row["category"] != "unknown":
                    result.setdefault(row["description_key"], row["category"])
        return result

    async def suggest_draft_categories(
        self, conn: asyncpg.Connection, owner: UUID | None, document: ExtractionDraft, *, candidate: str | None = None,
    ) -> list[str]:
        printed = document.merchant_name or ""
        display = await merchants.suggested_name(conn, owner, printed, document.merchant_address, candidate=candidate)
        document.merchant_display_name = display
        if document.category is None and (category := merchant_category(display or "")):
            document.category = cast(Category, category)
        items = document.line_items or []
        history = await self._history_for_items(
            conn, owner, printed, [item.description or "" for item in items],
            display, document.merchant_address,
        )
        sources = []
        for item in items:
            if learned := history.get(normalize_history_key(item.description or "")):
                item.spending_category = cast(SpendingCategory, learned)
                sources.append("history")
            else:
                sources.append("llm" if item.spending_category not in (None, "unknown") else "unknown")
        return sources

    # ── Persist extracted data ───────────────────────────────────────

    async def insert_receipt(
        self, conn: asyncpg.Connection, receipt: Receipt, *, image_id: UUID,
        user_id: UUID | None, status: str = "unverified", verified: bool = False,
        human_save: bool = False, actor_id: UUID | None = None, category_missing: bool = False,
        draft_sources: list[str] | None = None,
    ) -> ReceiptRow:
        """Insert only on the caller's transaction; never overwrite a linked receipt."""
        display = (receipt.merchant_display_name if human_save else None) or await merchants.suggested_name(
            conn, user_id if not human_save or user_id == actor_id else None,
            receipt.merchant_name, receipt.merchant_address, candidate=receipt.merchant_display_name,
        )
        category = receipt.category
        if not human_save and category == "other" and category_missing:
            category = cast(Category, merchant_category(display or "") or category)
        row = await conn.fetchrow(
            INSERT_RECEIPT_SQL, receipt.confidence, receipt.merchant_name,
            receipt.merchant_address, receipt.receipt_number, receipt.date,
            self._receipt_time_value(receipt), receipt.currency, category,
            receipt.subtotal, receipt.discount_total, receipt.tax_total, receipt.tip,
            receipt.total, receipt.payment_method, status, image_id, verified, user_id, display,
        )
        history = {} if human_save else await self._history_for_items(
            conn, user_id, receipt.merchant_name, [i.description for i in receipt.line_items],
            display, receipt.merchant_address,
        )
        decisions = []
        for index, item in enumerate(receipt.line_items):
            learned = history.get(normalize_history_key(item.description))
            source = "llm" if item.spending_category != "unknown" else "unknown"
            if human_save:
                source = draft_sources[index] if draft_sources and index < len(draft_sources) else "unknown"
            decisions.append(ItemCategoryDecision(
                id=None, original_description=item.description,
                category=learned or item.spending_category, source="history" if learned else source,
            ))
        await self._insert_children(conn, row["id"], receipt, decisions)
        await self._capture_initial_collection(conn, row["id"], user_id)
        return self._receipt_row_from_record(row)

    @staticmethod
    async def _capture_initial_collection(
        conn: asyncpg.Connection, receipt_id: UUID, owner_id: UUID | None,
    ) -> None:
        if owner_id is None:
            return
        # Explicit savepoint operations avoid async-context cleanup awaiting
        # rollback after cancellation. Fatal errors go straight to the caller's
        # outer transaction/deadline adapter, which owns full rollback/discard.
        savepoint = conn.transaction()
        await savepoint.start()
        try:
            # Serialize with deletion's membership enumeration/history.
            # This receipt is new and cannot yet be locked by another caller.
            await conn.execute(MEMBERSHIP_LOCK_SQL)
            collection_id = await conn.fetchval(
                "SELECT id FROM collections WHERE user_id=$1 AND active=TRUE FOR SHARE",
                owner_id,
            )
            if collection_id is not None:
                await conn.execute(
                    """INSERT INTO receipt_collections(receipt_id,collection_id)
                    VALUES($1,$2) ON CONFLICT(receipt_id,collection_id) DO NOTHING""",
                    receipt_id, collection_id,
                )
        except asyncpg.IntegrityConstraintViolationError:
            # These operations remain inside the caller's bounded body; a
            # cancellation or unusable transaction during rollback propagates.
            await savepoint.rollback()
            logger.warning("Active collection capture rejected")
        else:
            await savepoint.commit()

    async def persist_receipt(
        self,
        receipt: Receipt,
        image_id: UUID | None = None,
        status: str = "unverified",
        verified: bool = False,
        user_id: UUID | None = None,
    ) -> ReceiptRow:
        """Insert a receipt (with line items and taxes) into PostgreSQL.

        ``image_id`` links the receipt to an ``images`` row (the FK swap); it
        is ``None`` for receipts created without an associated image. ``user_id``
        is the owning user (``None`` for legacy/unowned rows).
        """
        logger.info("Persisting receipt")

        values = (
            receipt.confidence,
            receipt.merchant_name,
            receipt.merchant_address,
            receipt.receipt_number,
            receipt.date,
            self._receipt_time_value(receipt),
            receipt.currency,
            receipt.category,
            receipt.subtotal,
            receipt.discount_total,
            receipt.tax_total,
            receipt.tip,
            receipt.total,
            receipt.payment_method,
        )

        async with self.pool.acquire() as conn, conn.transaction():
            display = await merchants.suggested_name(
                conn, user_id, receipt.merchant_name, receipt.merchant_address, candidate=receipt.merchant_display_name,
            )
            if "category" not in receipt.model_fields_set and (category := merchant_category(display or "")):
                values = (*values[:7], cast(Category, category), *values[8:])
            row = None
            if image_id is not None:
                row = await conn.fetchrow(
                    UPDATE_RECEIPT_BY_IMAGE_SQL,
                    *values,
                    status,
                    verified,
                    user_id,
                    image_id,
                    display,
                )
            if row is None:
                row = await conn.fetchrow(
                    INSERT_RECEIPT_SQL, *values, status, image_id, verified, user_id, display
                )
            receipt_id: UUID = row["id"]
            history = await self._history_for_items(
                conn, user_id, receipt.merchant_name,
                [item.description for item in receipt.line_items], display, receipt.merchant_address,
            )
            decisions = []
            for item in receipt.line_items:
                category = history.get(normalize_history_key(item.description))
                decisions.append(
                    ItemCategoryDecision(
                        id=None,
                        original_description=item.description,
                        category=category or item.spending_category,
                        source="history" if category else (
                            "llm" if item.spending_category != "unknown" else "unknown"
                        ),
                    )
                )
            await conn.execute(DELETE_LINE_ITEMS_SQL, receipt_id)
            await conn.execute(DELETE_TAXES_SQL, receipt_id)
            await self._insert_children(conn, receipt_id, receipt, decisions)

        return self._receipt_row_from_record(row)

    @staticmethod
    async def _lock_review_receipt(
        conn: asyncpg.Connection,
        receipt_id: UUID,
        user_id: UUID | None,
        can_see_all: bool,
        *, lock: bool = True,
    ) -> asyncpg.Record | None:
        if not lock:
            return await conn.fetchrow("SELECT * FROM receipts WHERE id=$1 AND ($3 OR user_id=$2)", receipt_id, user_id, can_see_all)
        # Match workflow's image-before-receipt lock order. Recheck scope after locking.
        visible = await conn.fetchrow(
            "SELECT image_id FROM receipts WHERE id=$1 AND ($3 OR $2::uuid IS NULL OR user_id=$2)",
            receipt_id,
            user_id,
            can_see_all,
        )
        if visible is None:
            return None
        if visible["image_id"] is not None:
            await conn.fetchrow("SELECT id FROM images WHERE id=$1 FOR UPDATE", visible["image_id"])
        return await conn.fetchrow(
            "SELECT * FROM receipts WHERE id=$1 AND ($3 OR $2::uuid IS NULL OR user_id=$2) FOR UPDATE",
            receipt_id,
            user_id,
            can_see_all,
        )

    async def _review_document(
        self, conn: asyncpg.Connection, row: Mapping[str, Any]
    ) -> ExtractionDraft:
        # Stored observations must remain repairable even when legacy values violate
        # today's model. Evaluation validates them explicitly rather than adding defaults.
        data = dict(row)
        stored_time = data.get("time")
        if isinstance(stored_time, Time):
            data["time"] = (
                stored_time.isoformat(timespec="minutes")
                if not stored_time.second and not stored_time.microsecond
                else stored_time.isoformat()
            )
        data["line_items"] = [
            DraftLineItem.model_validate(self._line_item_row_from_record(item).model_dump())
            for item in await conn.fetch(LIST_LINE_ITEMS_SQL, row["id"])
        ]
        data["taxes"] = [
            DraftTaxLine.model_validate(self._tax_row_from_record(item).model_dump())
            for item in await conn.fetch(LIST_TAXES_SQL, row["id"])
        ]
        return ExtractionDraft.model_construct(
            **{key: data.get(key) for key in ExtractionDraft.model_fields}
        )

    @staticmethod
    async def _validate_review_sources(
        conn: asyncpg.Connection,
        image_id: UUID | None,
        sources: dict[str, list[SourceRef]],
        overlaps: list[dict[str, Any]],
        *, lock: bool = True,
    ) -> None:
        assert_metadata_only(overlaps)
        references = [ref for refs in sources.values() for ref in refs]
        view_ids: set[UUID] = set()

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("view_id") is not None:
                    view_ids.add(UUID(str(value["view_id"])))
                if "part_id" in value:
                    references.append(SourceRef.model_validate(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(overlaps)
        for view_id in view_ids:
            if not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM extraction_working_views WHERE view_id=$1 AND image_id=$2)",
                view_id, image_id,
            ):
                raise ValueError("source reference view does not match saved artifact")
        if not references:
            return
        parts = {
            part["photo_index"]: part
            for part in await conn.fetch(
                "SELECT * FROM image_source_parts WHERE image_id=$1 ORDER BY photo_index" + (" FOR SHARE" if lock else ""),
                image_id,
            )
        }
        for ref in references:
            part = parts.get(ref.photo_index)
            if (
                part is None
                or part["part_id"] != ref.part_id
                or part["source_sha256"] != ref.source_sha256
            ):
                raise ValueError("source reference must bind to a real source part")
            if ref.coordinate_space == "working" and ref.view_id is None:
                raise ValueError("working source reference requires saved view")
            if ref.view_id is not None and not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM extraction_working_views WHERE view_id=$1 AND image_id=$2 AND part_id=$3)",
                ref.view_id,
                image_id,
                ref.part_id,
            ):
                raise ValueError("source reference view does not match saved artifact")

    async def _review_originals(
        self,
        conn: asyncpg.Connection,
        image_id: UUID | None,
        *, lock: bool = True,
    ) -> list[OriginalAvailability]:
        if image_id is None:
            return []
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1", image_id)
        if row is None:
            return []
        image = ImageRow.model_validate(dict(row))
        durable = await conn.fetch(
            "SELECT * FROM image_source_parts WHERE image_id=$1 ORDER BY photo_index" + (" FOR SHARE" if lock else ""),
            image_id,
        )
        by_index = {part["photo_index"]: part for part in durable}
        parts = image.parts()
        # Only pre-workflow images without any durable identity evidence may
        # retain unknown legacy identities. A missing manifest is corruption,
        # not permission to replace a known original with any readable file.
        snapshots = [row[name] for name in ("source_manifest", "requested_options", "effective_options")]
        has_snapshot = any(
            json.loads(value) if isinstance(value, str) else value for value in snapshots
        )
        requires_manifest = bool(
            durable or has_snapshot or row["request_fingerprint"] is not None
            or row["idempotency_key"] is not None or row["ocr_checkpoint"] is not None
            or row["attempt_count"] > 0 or row["generation"] > 1
        )
        if not requires_manifest:
            requires_manifest = await conn.fetchval(
                """SELECT EXISTS(SELECT 1 FROM receipt_extractions WHERE image_id=$1)
                OR EXISTS(SELECT 1 FROM image_retry_requests WHERE image_id=$1)""",
                image_id,
            )
        if requires_manifest and [part["photo_index"] for part in durable] != list(range(len(parts))):
            raise ValueError("original_integrity_failed")
        for index, part in enumerate(parts):
            saved = by_index.get(index)
            part.part_id = saved["part_id"] if saved else None
            part.photo_index = index
            part.source_sha256 = saved["source_sha256"] if saved else part.source_sha256
            if saved and saved["image_path"] != part.image_path:
                # An inconsistent manifest is not silently replaced by current file availability.
                return [
                    OriginalAvailability(
                        part_id=part.part_id, photo_index=index, available=False, sha256_matches=False
                    )
                ]
        if self._original_checker is None:
            raise ValueError("original_checker_required")
        result = await self._original_checker(parts)
        if [(o.part_id, o.photo_index) for o in result] != [(p.part_id, p.photo_index) for p in parts]:
            raise ValueError("original_integrity_failed")
        return result

    @staticmethod
    async def _append_review_history(
        conn: asyncpg.Connection,
        row: asyncpg.Record,
        event: str,
        actor: UUID | None,
    ) -> None:
        receipt = ReceiptDB._receipt_row_from_record(row)
        snapshot = {
            "receipt": receipt.model_dump(mode="json"),
            "line_items": [
                ReceiptDB._line_item_row_from_record(item).model_dump(mode="json")
                for item in await conn.fetch(LIST_LINE_ITEMS_SQL, receipt.id)
            ],
            "taxes": [
                ReceiptDB._tax_row_from_record(item).model_dump(mode="json")
                for item in await conn.fetch(LIST_TAXES_SQL, receipt.id)
            ],
            "collection_ids": [
                str(item["collection_id"])
                for item in await conn.fetch(
                    "SELECT collection_id FROM receipt_collections WHERE receipt_id=$1 ORDER BY collection_id",
                    receipt.id,
                )
            ],
        }
        await conn.execute(
            """INSERT INTO receipt_review_revisions(receipt_id,operation_revision,review_revision,
            event_kind,actor_user_id,snapshot) VALUES($1,$2,$3,$4,$5,$6::jsonb)""",
            receipt.id,
            receipt.operation_revision,
            receipt.review_revision,
            event,
            actor,
            json.dumps(snapshot),
        )

    @staticmethod
    async def _visible_collection_ids(
        conn: asyncpg.Connection, receipt_id: UUID, user_id: UUID | None, can_see_all: bool,
    ) -> list[UUID]:
        if can_see_all or user_id is None:
            rows = await conn.fetch(LIST_COLLECTION_IDS_FOR_RECEIPT_SQL, receipt_id)
        else:
            rows = await conn.fetch(
                "SELECT rc.collection_id FROM receipt_collections rc "
                "JOIN collections c ON c.id=rc.collection_id "
                "WHERE rc.receipt_id=$1 AND c.user_id=$2",
                receipt_id,
                user_id,
            )
        return [row["collection_id"] for row in rows]

    @staticmethod
    async def _set_collections(
        conn: asyncpg.Connection,
        receipt_id: UUID,
        collection_ids: list[UUID],
        user_id: UUID | None,
        can_see_all: bool,
    ) -> None:
        targets = set(collection_ids)
        rows = await conn.fetch(
            "SELECT id FROM collections WHERE id=ANY($1::uuid[]) AND ($3 OR user_id=$2) FOR SHARE",
            list(targets),
            user_id,
            can_see_all,
        )
        if {row["id"] for row in rows} != targets:
            raise ValueError("collection_not_found")
        if can_see_all or user_id is None:
            await conn.execute("DELETE FROM receipt_collections WHERE receipt_id=$1", receipt_id)
        else:
            await conn.execute(
                "DELETE FROM receipt_collections rc USING collections c "
                "WHERE rc.receipt_id=$1 AND rc.collection_id=c.id AND c.user_id=$2",
                receipt_id,
                user_id,
            )
        for target in sorted(targets):
            await conn.execute(
                "INSERT INTO receipt_collections(receipt_id,collection_id) VALUES($1,$2)",
                receipt_id,
                target,
            )

    async def update_receipt(
        self,
        receipt_id: UUID,
        receipt: Receipt,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> ReceiptRow | None:
        """Update a receipt row and replace its line items and taxes.

        Returns None when no receipt with the given id exists (or, for
        non-see-all callers, when it is not owned by ``user_id``).
        """
        logger.info("Updating receipt %s", receipt_id)

        args: list[Any] = [
            receipt.confidence,
            receipt.merchant_name,
            receipt.merchant_address,
            receipt.receipt_number,
            receipt.date,
            self._receipt_time_value(receipt),
            receipt.currency,
            receipt.category,
            receipt.subtotal,
            receipt.discount_total,
            receipt.tax_total,
            receipt.tip,
            receipt.total,
            receipt.payment_method,
            receipt_id,
            None,  # Filled from the locked receipt and this update below.
        ]
        sql = UPDATE_RECEIPT_SQL
        if not can_see_all and user_id is not None:
            args.append(user_id)
            sql = sql.replace("RETURNING *", f"AND user_id = ${len(args)} RETURNING *", 1)

        async with self.pool.acquire() as conn, conn.transaction():
            if getattr(receipt, "collection_ids", None) is not None:
                await conn.execute(MEMBERSHIP_LOCK_SQL)
            before = await self._lock_review_receipt(conn, receipt_id, user_id, can_see_all)
            if before is None:
                return None
            current = self._receipt_row_from_record(before)
            display = receipt.merchant_display_name
            if "merchant_display_name" not in receipt.model_fields_set and (
                current.merchant_name == receipt.merchant_name and current.merchant_address == receipt.merchant_address
            ):
                display = current.merchant_display_name
            own_account = user_id if user_id == current.user_id else None
            args[15] = display or await merchants.suggested_name(
                conn, own_account, receipt.merchant_name, receipt.merchant_address,
            )
            document = await self._review_document(conn, before)
            policy = getattr(receipt, "amount_policy", None) or current.amount_policy
            sources = getattr(receipt, "field_sources", None)
            sources = current.field_sources if sources is None else sources
            overlaps = getattr(receipt, "overlap_decisions", None)
            overlaps = current.overlap_decisions if overlaps is None else overlaps
            material = (
                material_document(document) != material_document(receipt)
                or policy != current.amount_policy
                or sources != current.field_sources
                or overlaps != current.overlap_decisions
            )
            expected = getattr(receipt, "expected_operation_revision", None)
            if (material and expected is None) or (
                expected is not None and expected != current.operation_revision
            ):
                raise ValueError("operation_revision_conflict")
            evaluation = None
            if material:
                await self._validate_review_sources(conn, current.image_id, sources, overlaps)
                originals = await self._review_originals(conn, current.image_id)
                evaluation = evaluate_receipt(
                    receipt,
                    policy,
                    sources,
                    overlaps,
                    originals,
                    retained_findings=persistent_findings(current.findings),
                )
            row = await conn.fetchrow(sql, *args)
            assert row is not None

            owner_id: UUID | None = dict(row).get("user_id")
            await merchants.learn_name(
                conn, owner_id, user_id, receipt.merchant_name, receipt.merchant_address, args[15],
            )
            old_rows = [dict(old) for old in await conn.fetch(LIST_LINE_ITEMS_SQL, receipt_id)]
            old_by_id = {old["id"]: old for old in old_rows}
            old_by_position = {old["position"]: old for old in old_rows}
            used_ids: set[UUID] = set()
            decisions: list[ItemCategoryDecision] = []
            merchant_keys = self._merchant_history_keys(receipt.merchant_name, args[15], receipt.merchant_address)
            merchant_key = merchant_keys[0] if merchant_keys else None
            legacy_items = all(getattr(item, "id", None) is None for item in receipt.line_items)
            for position, item in enumerate(receipt.line_items):
                item_id: UUID | None = getattr(item, "id", None)
                if item_id is not None and item_id not in old_by_id:
                    raise ValueError("Line item id does not belong to this receipt")
                old = old_by_id.get(item_id) if item_id is not None else None
                if old is None and legacy_items and len(old_rows) == len(receipt.line_items):
                    candidate = old_by_position.get(position)
                    if candidate is not None and candidate["description"] == item.description:
                        old = candidate
                if old is not None and old["id"] in used_ids:
                    raise ValueError("Duplicate line item id")
                if old is not None:
                    used_ids.add(old["id"])
                original = (
                    (old.get("original_description") or old["description"])
                    if old
                    else item.description
                )
                old_category = old.get("spending_category", "unknown") if old else "unknown"
                old_source = old.get("category_source", "unknown") if old else "unknown"
                explicit = "spending_category" in item.model_fields_set
                changed = explicit and item.spending_category != old_category
                remember = bool(getattr(item, "remember_category", False))
                category = item.spending_category if changed or old is None else old_category
                source = "user" if changed or remember else old_source
                if old is None and explicit and category != "unknown":
                    source = "user"
                decisions.append(
                    ItemCategoryDecision(
                        id=old["id"] if old else None,
                        original_description=original,
                        category=category,
                        source=source,
                    )
                )
                if remember and owner_id is not None and merchant_key:
                    for description_key in {
                        normalize_history_key(original),
                        normalize_history_key(item.description),
                    }:
                        if not description_key:
                            continue
                        if category == "unknown":
                            for key in merchant_keys:
                                await conn.execute(
                                    DELETE_CATEGORY_ALIAS_SQL, owner_id, key, description_key
                                )
                        else:
                            await conn.execute(
                                UPSERT_CATEGORY_ALIAS_SQL,
                                owner_id,
                                merchant_key,
                                description_key,
                                category,
                            )

            await conn.execute(DELETE_LINE_ITEMS_SQL, receipt_id)
            await conn.execute(DELETE_TAXES_SQL, receipt_id)
            await self._insert_children(conn, receipt_id, receipt, decisions)
            collections = getattr(receipt, "collection_ids", None)
            if collections is not None:
                await self._set_collections(conn, receipt_id, collections, user_id, can_see_all)
            if material:
                assert evaluation is not None
                verified = current.verified and evaluation.can_auto_verify
                row = await conn.fetchrow(
                    """UPDATE receipts SET operation_revision=operation_revision+1,
                    review_revision=review_revision+1,amount_policy=$2::jsonb,field_sources=$3::jsonb,
                    overlap_decisions=$4::jsonb,findings=$5::jsonb,checks_version=$6,
                    acknowledgements='[]'::jsonb,verified=$7,status=$8,verification_source=$9
                    WHERE id=$1 RETURNING *""",
                    receipt_id,
                    policy.model_dump_json(),
                    json.dumps(
                        {k: [r.model_dump(mode="json") for r in v] for k, v in sources.items()}
                    ),
                    json.dumps(overlaps),
                    json.dumps([f.model_dump(mode="json") for f in evaluation.findings]),
                    evaluation.checks_version,
                    verified,
                    "verified" if verified else "unverified",
                    current.verification_source if verified else None,
                )
            else:
                row = await conn.fetchrow(
                    "UPDATE receipts SET operation_revision=operation_revision+1 WHERE id=$1 RETURNING *",
                    receipt_id,
                )
            assert row is not None
            await self._append_review_history(
                conn, row, "material_edit" if material else "classification_edit", user_id
            )

        return self._receipt_row_from_record(row)

    async def _evaluate_current_review(
        self,
        conn: asyncpg.Connection,
        row: asyncpg.Record,
        current: ReceiptRow,
        *,
        lock: bool = True,
    ) -> ReviewEvaluation:
        document = await self._review_document(conn, row)
        await self._validate_review_sources(
            conn, current.image_id, current.field_sources, current.overlap_decisions, lock=lock
        )
        originals = await self._review_originals(conn, current.image_id, lock=lock)
        return evaluate_receipt(
            document,
            current.amount_policy,
            current.field_sources,
            current.overlap_decisions,
            originals,
            retained_findings=persistent_findings(current.findings),
        )

    async def preview_human_review(
        self, receipt_id: UUID, *, scope: AccessScope
    ) -> ReviewPreview | None:
        """Evaluate current findings on a read-only, owner-scoped snapshot."""
        async with (
            self.pool.acquire() as conn,
            conn.transaction(isolation="repeatable_read", readonly=True),
        ):
            row = await self._lock_review_receipt(
                conn, receipt_id, scope.user_id, scope.can_see_all, lock=False
            )
            if row is None:
                return None
            current = self._receipt_row_from_record(row)
            if (current.verified or current.status != "unverified") and not current.can_rereview():
                raise ValueError("human_rereview_not_available")
            evaluation = await self._evaluate_current_review(conn, row, current, lock=False)
            return ReviewPreview(
                operation_revision=current.operation_revision,
                review_revision=current.review_revision,
                findings=evaluation.findings,
            )

    async def verify_receipt(
        self,
        receipt_id: UUID,
        request: ReviewRequest | None = None,
        *,
        scope: AccessScope | None = None,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> ReceiptRow | None:
        """Lock, re-read evidence, check exact current exceptions, then append history."""
        request = request or ReviewRequest()
        if scope is not None:
            user_id, can_see_all = scope.user_id, scope.can_see_all
        async with self.pool.acquire() as conn, conn.transaction():
            row = await self._lock_review_receipt(conn, receipt_id, user_id, can_see_all)
            if row is None:
                return None
            current = self._receipt_row_from_record(row)
            if (
                request.expected_operation_revision is not None
                and request.expected_operation_revision != current.operation_revision
            ):
                raise ValueError("operation_revision_conflict")
            if (
                request.expected_review_revision is not None
                and request.expected_review_revision != current.review_revision
            ):
                raise ValueError("review_revision_conflict")
            missing_revision = (
                request.expected_operation_revision is None
                or request.expected_review_revision is None
            )
            if current.verified or current.status == "verified":
                if not current.can_rereview() or user_id is None:
                    raise ValueError("receipt_already_verified")
                if missing_revision:
                    raise ValueError("review_revision_required")
            evaluation = await self._evaluate_current_review(conn, row, current)
            validate_acknowledgements(evaluation, request.acknowledgements)
            if evaluation.findings and missing_revision:
                raise ValueError("review_revision_required")
            display = current.merchant_display_name or await merchants.suggested_name(
                conn, user_id if user_id == current.user_id else None,
                current.merchant_name, current.merchant_address,
            )
            await merchants.learn_name(
                conn, current.user_id, user_id, current.merchant_name, current.merchant_address, display,
            )
            saved = await conn.fetchrow(
                """UPDATE receipts SET status='verified',verified=TRUE,merchant_display_name=$5,verification_source='human',
                operation_revision=operation_revision+1,findings=$2::jsonb,checks_version=$3,
                acknowledgements=$4::jsonb WHERE id=$1 RETURNING *""",
                receipt_id,
                json.dumps([f.model_dump(mode="json") for f in evaluation.findings]),
                evaluation.checks_version,
                json.dumps([a.model_dump(mode="json") for a in request.acknowledgements]),
                display,
            )
            assert saved is not None
            await self._append_review_history(conn, saved, "verified", user_id)
            return self._receipt_row_from_record(saved)

    async def delete_receipt(
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ReceiptRow | None:
        """Delete a receipt row. Line items and taxes cascade via their foreign keys.

        Returns the deleted row, or None when no receipt with the given id exists
        (or is not owned by a non-see-all ``user_id``). A ``benchmark_tasks`` row
        still referencing the receipt raises ``asyncpg.ForeignKeyViolationError``
        (that FK has no ON DELETE rule).
        """
        logger.info("Deleting receipt %s", receipt_id)

        args: list[Any] = [receipt_id]
        sql = DELETE_RECEIPT_SQL
        if not can_see_all and user_id is not None:
            args.append(user_id)
            sql = sql.replace("RETURNING *", f"AND user_id = ${len(args)} RETURNING *", 1)

        async with self.pool.acquire() as conn, conn.transaction():
            # DELETE also locks the image via its receipt_id SET NULL action.
            # Acquire it first, as PUT/verify do, before touching the receipt.
            if await self._lock_review_receipt(conn, receipt_id, user_id, can_see_all) is None:
                return None
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return None
        return self._receipt_row_from_record(row)

    # ── Query helpers ────────────────────────────────────────────────

    async def get_receipt_by_id(
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ReceiptRow | None:
        """Fetch a single receipt row by its primary key, scoped to its owner.

        Non-see-all callers only match their own rows (other users' receipts
        resolve to ``None`` -> 404 upstream).
        """
        args: list[Any] = [receipt_id]
        sql = GET_RECEIPT_SQL
        if not can_see_all and user_id is not None:
            args.append(user_id)
            sql += f" AND user_id = ${len(args)}"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return None
        return self._receipt_row_from_record(row)

    async def get_receipt_by_image_id(self, image_id: UUID) -> ReceiptRow | None:
        """Return the existing receipt linked to an image, if any."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(GET_RECEIPT_BY_IMAGE_ID_SQL, image_id)
        if row is None:
            return None
        return self._receipt_row_from_record(row)

    async def list_receipts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: list[str] | None = None,
        date_from: Date | None = None,
        date_to: Date | None = None,
        search: str | None = None,
        collection_id: UUID | None = None,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> list[ReceiptRow]:
        """Return a paginated list of receipts ordered by date descending.

        Optional filters: ``status`` (IN list), an inclusive ``date_from`` /
        ``date_to`` range, and case-insensitive ``search`` over merchant name
        or receipt number. Non-see-all callers are restricted to their own
        rows via ``user_id``.
        """
        where: list[str] = []
        args: list[Any] = []
        if not can_see_all and user_id is not None:
            args.append(user_id)
            where.append(f"user_id = ${len(args)}")
        if status:
            args.append(status)
            where.append(f"status = ANY(${len(args)})")
        if date_from is not None:
            args.append(date_from)
            where.append(f"date >= ${len(args)}")
        if date_to is not None:
            args.append(date_to)
            where.append(f"date <= ${len(args)}")
        if search:
            pattern = f"%{search}%"
            args.append(pattern)
            where.append(f"(merchant_name ILIKE ${len(args)} OR merchant_display_name ILIKE ${len(args)} OR receipt_number ILIKE ${len(args)})")
        if collection_id is not None:
            args.append(collection_id)
            where.append(
                "EXISTS (SELECT 1 FROM receipt_collections rc "
                f"WHERE rc.receipt_id = r.id AND rc.collection_id = ${len(args)})"
            )

        sql = LIST_RECEIPTS_BASE_SQL
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY r.date DESC, r.id LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
        args.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [self._receipt_row_from_record(row) for row in rows]

    async def search_products(
        self,
        query: str,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> list[ProductPurchase]:
        """Find verified line items whose descriptions contain ``query``.

        Matching is case-insensitive and is deliberately performed by
        PostgreSQL; product search must not invoke the LLM. Non-see-all callers
        are restricted to their own receipts via ``user_id``.
        """
        args: list[Any] = [f"%{query}%"]
        sql = SEARCH_PRODUCTS_BASE_SQL
        if not can_see_all and user_id is not None:
            args.append(user_id)
            sql += f" AND r.user_id = ${len(args)}"
        sql += (
            " ORDER BY r.date DESC, r.time DESC NULLS LAST, r.created_at DESC NULLS LAST, "
            "r.id DESC, li.id DESC"
        )

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [self._product_purchase_from_record(row) for row in rows]

    async def get_statistics(
        self,
        user_id: UUID | None = None,
        can_see_all: bool = False,
        weeks: int = 12,
        collection_id: UUID | None = None,
    ) -> ReceiptStatistics:
        """Aggregate verified receipts for the statistics and dashboard views."""
        scope = ""
        scope_args: list[Any] = []
        if not can_see_all and user_id is not None:
            scope_args.append(user_id)
            scope = f" AND user_id = ${len(scope_args)}"
        if collection_id is not None:
            scope_args.append(collection_id)
            scope += (
                f" AND EXISTS (SELECT 1 FROM receipt_collections rc "
                f"WHERE rc.receipt_id = receipts.id AND rc.collection_id = ${len(scope_args)})"
            )

        weekly_args = [*scope_args, max(weeks - 1, 0)]
        weekly_sql = STATS_WEEKLY_SQL.format(
            scope=scope,
            weeks_param=len(weekly_args),
        )

        async with self.pool.acquire() as conn:
            currency_rows = await conn.fetch(STATS_CURRENCY_SQL.format(scope=scope), *scope_args)
            merchant_rows = await conn.fetch(
                STATS_NAMED_SQL.format(
                    field="COALESCE(merchant_display_name, NULLIF(merchant_name, ''), 'Unknown merchant')",
                    scope=scope,
                ),
                *scope_args,
            )
            category_rows = await conn.fetch(
                STATS_NAMED_SQL.format(field="category", scope=scope), *scope_args
            )
            payment_rows = await conn.fetch(
                STATS_NAMED_SQL.format(field="payment_method", scope=scope), *scope_args
            )
            weekday_rows = await conn.fetch(STATS_WEEKDAY_SQL.format(scope=scope), *scope_args)
            weekly_rows = await conn.fetch(weekly_sql, *weekly_args)

        def decimal(row: Mapping[str, Any], key: str) -> Decimal:
            value = row[key]
            return value if isinstance(value, Decimal) else Decimal(str(value))

        currencies = [
            CurrencyStatistics(
                currency=row["currency"],
                receipt_count=int(row["receipt_count"]),
                total=decimal(row, "total"),
                average=decimal(row, "average"),
                median=decimal(row, "median"),
                minimum=decimal(row, "minimum"),
                maximum=decimal(row, "maximum"),
                subtotal=decimal(row, "subtotal"),
                discounts=decimal(row, "discounts"),
                taxes=decimal(row, "taxes"),
                tips=decimal(row, "tips"),
            )
            for row in currency_rows
        ]

        def named(rows: list[Mapping[str, Any]]) -> list[NamedStatistics]:
            return [
                NamedStatistics(
                    name=row["name"],
                    currency=row["currency"],
                    receipt_count=int(row["receipt_count"]),
                    total=decimal(row, "total"),
                    average=decimal(row, "average"),
                )
                for row in rows
            ]

        return ReceiptStatistics(
            verified_receipt_count=sum(item.receipt_count for item in currencies),
            currencies=currencies,
            merchants=named(merchant_rows),
            categories=named(category_rows),
            payment_methods=named(payment_rows),
            weekdays=[
                WeekdayStatistics(
                    weekday=int(row["weekday"]),
                    currency=row["currency"],
                    receipt_count=int(row["receipt_count"]),
                    total=decimal(row, "total"),
                    average=decimal(row, "average"),
                )
                for row in weekday_rows
            ],
            weekly_spending=[
                WeeklyStatistics(
                    week_start=row["week_start"],
                    currency=row["currency"],
                    receipt_count=int(row["receipt_count"]),
                    total=decimal(row, "total"),
                    average=decimal(row, "average"),
                )
                for row in weekly_rows
            ],
        )

    async def get_receipt_with_details(
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False,
        *, deadline: float | None = None,
    ) -> ReceiptWithDetails | None:
        """Fetch a receipt together with its line items, taxes and image path.

        The image path is resolved via a LEFT JOIN on the images table so the
        response exposes it for frontends even though the receipts row only
        stores the FK. Non-see-all callers only match their own rows.
        """
        args: list[Any] = [receipt_id]
        sql = GET_RECEIPT_WITH_IMAGE_SQL
        if not can_see_all and user_id is not None:
            args.append(user_id)
            sql += f" AND r.user_id = ${len(args)}"
        async with connection(self.pool, deadline=deadline) as conn:
            row = await conn.fetchrow(sql, *args)
            if row is None:
                return None
            li_rows = await conn.fetch(LIST_LINE_ITEMS_SQL, receipt_id)
            tax_rows = await conn.fetch(LIST_TAXES_SQL, receipt_id)
            collection_ids = await self._visible_collection_ids(conn, receipt_id, user_id, can_see_all)

        d = dict(row)
        image_path: str | None = d.get("image_path")
        receipt = self._receipt_row_from_record(row)
        raw_enrichment = d.get("classification_enrichment")
        if isinstance(raw_enrichment, str):
            raw_enrichment = json.loads(raw_enrichment)

        return ReceiptWithDetails(
            receipt=receipt,
            line_items=[self._line_item_row_from_record(r) for r in li_rows],
            taxes=[self._tax_row_from_record(r) for r in tax_rows],
            image_path=image_path,
            collection_ids=collection_ids,
            classification_enrichment=(
                ClassificationEnrichment.model_validate(raw_enrichment)
                if raw_enrichment is not None else ClassificationEnrichment()
            ),
        )

    async def list_tags(self, *, deadline: float | None = None) -> list[str]:
        """Return the allowed line-item tag vocabulary, ordered by name."""
        async with connection(self.pool, deadline=deadline) as conn:
            rows = await conn.fetch(LIST_TAGS_SQL)
        return [row["name"] for row in rows]

    async def create_tag(self, name: str) -> tuple[str, bool]:
        """Return the stored spelling and whether the normalized tag was inserted."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(INSERT_TAG_SQL, name)
            if row is not None:
                return row["name"], True
            # A new statement also sees a concurrent creator that won ON CONFLICT.
            stored = await conn.fetchval(EXISTING_TAG_SQL, name)
            return stored, False
