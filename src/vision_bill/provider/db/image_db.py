import json
import logging
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import asyncpg

from ...config import PGSettings
from ...model.db.image import ImagePart, ImageRow

# ── SQL (DML; DDL lives in alembic/versions/0001_initial_schema.py) ──

INSERT_IMAGE_SQL = """
    INSERT INTO images
        (original_filename, media_type, size_bytes, image_path, thumbnail_path, status, user_id,
         bypass_review, additional_images, model_id)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
    RETURNING *
"""

GET_IMAGE_SQL = "SELECT * FROM images WHERE id = $1"

# The "queue": images that still need analysis (pending), are being retried
# (failed), or whose previous worker lease expired. Oldest first so the
# backlog drains FIFO.
LIST_PENDING_IMAGES_SQL = (
    "SELECT * FROM images WHERE request_fingerprint IS NULL AND claim_token IS NULL "
    "AND attempt_count = 0 AND (status IN ('pending', 'failed') "
    "OR (status = 'processing' AND (processing_at IS NULL OR "
    "processing_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'))) "
    "ORDER BY created_at ASC, id ASC"
)

CLAIM_IMAGE_SQL = """
    UPDATE images
    SET status = 'processing', processing_at = CURRENT_TIMESTAMP, error = NULL
    WHERE id = $1
      AND request_fingerprint IS NULL AND claim_token IS NULL AND attempt_count = 0
      AND (
          status IN ('pending', 'failed')
          OR (status = 'processing' AND (processing_at IS NULL OR
              processing_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'))
      )
    RETURNING *
"""

LIST_IMAGES_BASE_SQL = "SELECT * FROM images"

MARK_ANALYZED_SQL = (
    "UPDATE images SET status = 'analyzed', receipt_id = $2, error = NULL, "
    "analyzed_at = CURRENT_TIMESTAMP, processing_at = NULL WHERE id = $1 "
    "AND request_fingerprint IS NULL AND claim_token IS NULL AND attempt_count = 0"
)

MARK_FAILED_SQL = (
    "UPDATE images SET status = 'failed', error = $2, processing_at = NULL WHERE id = $1 "
    "AND request_fingerprint IS NULL AND claim_token IS NULL AND attempt_count = 0"
)

UPDATE_IMAGE_PATH_SQL = "UPDATE images SET image_path = $2 WHERE id = $1"

UPDATE_IMAGE_THUMB_PATH_SQL = "UPDATE images SET thumbnail_path = $2 WHERE id = $1"

DELETE_IMAGE_SQL = "DELETE FROM images WHERE id = $1"


logger = logging.getLogger(__name__)


class ImageDB:
    """Owns the asyncpg pool and all SQL for the images table.

    Mirrors :class:`ReceiptDB`'s pool lifecycle but manages the ``images``
    table independently so the analysis pipeline can read/write it without
    touching receipt SQL.
    """

    def __init__(self, settings: PGSettings):
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    # ── Connection pool lifecycle ────────────────────────────────────
    async def init_db(self) -> None:
        """Create the connection pool and check that the schema is migrated."""
        if self._pool is not None:
            logger.warning("Image database pool already initialised - skipping")
            return

        dsn = self._settings.pg_dsn
        logger.info("Creating asyncpg image connection pool")
        self._pool = await asyncpg.create_pool(dsn=dsn)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT to_regclass('images') IS NULL AS missing")
        if row["missing"]:
            logger.warning(
                "Images schema not initialised - run 'uv run alembic upgrade head' before starting the app"
            )

    async def destroy_db(self) -> None:
        """Close the connection pool and release all resources."""
        if self._pool is not None:
            logger.info("Closing image database connection pool")
            await self._pool.close()
            self._pool = None
            logger.info("Image database pool closed")

    @property
    def is_ready(self) -> bool:
        """Whether the connection pool has been initialised."""
        return self._pool is not None

    # ── Helpers ──────────────────────────────────────────────────────

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Image database pool not initialised. Call init_db() first.")
        return self._pool

    @staticmethod
    def _image_row_from_record(row: Mapping[str, Any]) -> ImageRow:
        """Map a raw images row to an ImageRow."""
        return ImageRow(**dict(row))

    # ── DML ──────────────────────────────────────────────────────────

    async def store_image(
        self,
        image_path: str,
        original_filename: str | None = None,
        media_type: str | None = None,
        size_bytes: int | None = None,
        status: str = "pending",
        user_id: UUID | None = None,
        bypass_review: bool = False,
        thumbnail_path: str | None = None,
        additional_images: list[ImagePart] | None = None,
        model_id: str | None = None,
    ) -> ImageRow:
        """Insert a new images row (default status ``pending``) and return it."""
        logger.info("Storing image row (status=%s)", status)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                INSERT_IMAGE_SQL,
                original_filename,
                media_type,
                size_bytes,
                image_path,
                thumbnail_path,
                status,
                user_id,
                bypass_review,
                json.dumps([part.model_dump() for part in additional_images or []]),
                model_id,
            )
        return self._image_row_from_record(row)

    async def get_image_by_id(
        self, image_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ImageRow | None:
        """Fetch a single image row by its primary key, scoped to its owner."""
        args: list[Any] = [image_id]
        sql = GET_IMAGE_SQL
        if not can_see_all and user_id is not None:
            args.append(user_id)
            sql += f" AND user_id = ${len(args)}"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row is None:
            return None
        return self._image_row_from_record(row)

    async def list_pending_images(self) -> list[ImageRow]:
        """Return unclaimed and expired analysis jobs, oldest first."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(LIST_PENDING_IMAGES_SQL)
        return [self._image_row_from_record(row) for row in rows]

    async def claim_for_analysis(self, image_id: UUID) -> ImageRow | None:
        """Atomically claim an image so only one worker analyzes it."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(CLAIM_IMAGE_SQL, image_id)
        return self._image_row_from_record(row) if row is not None else None

    async def list_images(
        self,
        status: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> list[ImageRow]:
        """List image rows (newest first), optionally filtered by status.

        Non-see-all callers are restricted to their own rows via ``user_id``.
        """
        where: list[str] = []
        args: list[Any] = []
        if not can_see_all and user_id is not None:
            args.append(user_id)
            where.append(f"user_id = ${len(args)}")
        if status:
            args.append(status)
            where.append(f"status = ANY(${len(args)})")

        sql = LIST_IMAGES_BASE_SQL
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY created_at DESC, id DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
        args.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [self._image_row_from_record(row) for row in rows]

    async def mark_analyzed(self, image_id: UUID, receipt_id: UUID) -> None:
        """Mark an image analyzed and link it to its receipt."""
        logger.info("Marking image %s analyzed (receipt %s)", image_id, receipt_id)
        async with self.pool.acquire() as conn:
            await conn.execute(MARK_ANALYZED_SQL, image_id, receipt_id)

    async def mark_failed(self, image_id: UUID, error: str) -> None:
        """Mark an image failed, recording the error for the next retry."""
        logger.warning("Marking image %s failed", image_id)
        async with self.pool.acquire() as conn:
            await conn.execute(MARK_FAILED_SQL, image_id, error)

    async def update_image_path(self, image_id: UUID, image_path: str) -> None:
        """Update the on-disk path of an image (e.g. tmp -> permanent on verify)."""
        async with self.pool.acquire() as conn:
            await conn.execute(UPDATE_IMAGE_PATH_SQL, image_id, image_path)

    async def update_image_thumbnail_path(self, image_id: UUID, thumbnail_path: str) -> None:
        """Update the stored thumbnail path for an image row."""
        async with self.pool.acquire() as conn:
            await conn.execute(UPDATE_IMAGE_THUMB_PATH_SQL, image_id, thumbnail_path)

    async def delete_image(self, image_id: UUID) -> None:
        """Delete an image row (the on-disk file is removed by the caller)."""
        async with self.pool.acquire() as conn:
            await conn.execute(DELETE_IMAGE_SQL, image_id)
