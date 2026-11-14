import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import asyncpg
import httpx
from fastapi import UploadFile
from openai import APITimeoutError

from ..config import ImageSettings, PGSettings
from ..model.db.image import ImagePart, ImageRow
from ..model.db.receipt import ReceiptRow, ReceiptWithDetails
from ..model.receipt import Receipt
from ..model.search import ProductSearchResponse
from ..model.statistics import ReceiptStatistics
from ..provider.db.collection_db import CollectionDB
from ..provider.db.image_db import ImageDB
from ..provider.db.receipt_db import ReceiptDB
from ..provider.db.user_db import UserDB
from ..provider.llm.base import LLMProvider, ModelInfo
from ..provider.llm.errors import AnalysisTimeoutError
from ..provider.ocr.base import OcrProvider
from .image_service import ImageService

if TYPE_CHECKING:
    from ..model.experiment import ExperimentOptions
    from .experiment_execution import ExperimentResult

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    model_id: str
    receipt: Receipt | None
    error: str | None = None


class ReceiptReferencedError(Exception):
    """Raised when a receipt cannot be deleted because a benchmark references it."""


class ReceiptService:
    """Orchestrates receipt upload → LLM extraction → persistence.

    All database access is delegated to the ``ReceiptDB`` provider.
    """

    _experiment_tags: list[str] | None = None

    def __init__(
        self,
        image_settings: ImageSettings,
        pg_settings: PGSettings,
        provider: LLMProvider,
        ocr_provider: OcrProvider | None = None,
    ):
        self._provider: LLMProvider = provider
        self._ocr_provider: OcrProvider | None = ocr_provider
        self._image_service = ImageService(image_settings)
        self._db = ReceiptDB(pg_settings)
        self._image_db = ImageDB(pg_settings)
        self._user_db = UserDB(pg_settings)
        self._collection_db: CollectionDB | None = None

    # ── Database delegation ──────────────────────────────────────────

    async def init_db(self) -> None:
        await self._db.init_db()
        await self._image_db.init_db()
        await self._user_db.init_db()

    async def destroy_db(self) -> None:
        await self._db.destroy_db()
        await self._image_db.destroy_db()
        await self._user_db.destroy_db()

    @property
    def pool(self) -> asyncpg.Pool:
        return self._db.pool

    @property
    def db_ready(self) -> bool:
        return self._db.is_ready

    @property
    def image_db(self) -> ImageDB:
        """The images-table provider, exposed for the scheduler and API."""
        return self._image_db

    @property
    def user_db(self) -> UserDB:
        """The users-table provider, exposed for auth and ownership backfill."""
        return self._user_db

    async def list_tags(self) -> list[str]:
        """Return the allowed line-item tag vocabulary from the database."""
        return await self._db.list_tags()

    async def get_category_suggestion(
        self, user_id: UUID, merchant_name: str, description: str
    ) -> str | None:
        return await self._db.get_category_suggestion(user_id, merchant_name, description)

    async def create_tag(self, raw_name: str) -> tuple[str, bool]:
        """Normalize a tag name and make sure it exists in the tag vocabulary.

        Returns ``(name, created)`` where ``created`` is False when a tag with
        this (normalized) name already existed. Raises ``ValueError`` for
        names that are blank or too long.
        """
        name = " ".join(raw_name.split()).lower()
        if not name:
            raise ValueError("Tag name must not be blank")
        if len(name) > 100:
            raise ValueError("Tag name must be at most 100 characters long")
        created = await self._db.create_tag(name)
        return name, created

    async def persist_receipt(
        self,
        receipt: Receipt,
        image_id: UUID | None = None,
        status: str = "unverified",
        verified: bool = False,
        user_id: UUID | None = None,
    ) -> ReceiptRow:
        row = await self._db.persist_receipt(
            receipt, image_id=image_id, status=status, verified=verified, user_id=user_id
        )
        await self._capture_to_active(row.id, user_id)
        return row

    def set_collection_db(self, collection_db: CollectionDB) -> None:
        """Attach a CollectionDB so persisted receipts can be auto-captured
        into the caller's active collection (fail-safe)."""
        self._collection_db = collection_db

    async def _capture_to_active(self, receipt_id: UUID, user_id: UUID | None) -> None:
        """Assign a freshly persisted receipt to the user's active collection.

        Fail-safe: any error (or no active collection / no owner) is logged and
        swallowed so the capture can never break the core persistence flow.
        """
        if user_id is None or self._collection_db is None:
            return
        try:
            active = await self._collection_db.active_collection(user_id, can_see_all=False)
            if active is not None:
                await self._collection_db.assign(active.id, receipt_id)
        except Exception:
            logger.warning("auto-capture to active collection failed", exc_info=True)

    async def get_receipt_by_id(
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ReceiptRow | None:
        return await self._db.get_receipt_by_id(
            receipt_id, user_id=user_id, can_see_all=can_see_all
        )

    async def get_receipt_by_image_id(self, image_id: UUID) -> ReceiptRow | None:
        """Find the receipt already produced for an image, if any."""
        return await self._db.get_receipt_by_image_id(image_id)

    async def list_receipts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        search: str | None = None,
        collection_id: UUID | None = None,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> list[ReceiptRow]:
        return await self._db.list_receipts(
            limit=limit,
            offset=offset,
            status=status,
            date_from=date_from,
            date_to=date_to,
            search=search,
            collection_id=collection_id,
            user_id=user_id,
            can_see_all=can_see_all,
        )

    async def search_products(
        self,
        query: str,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> ProductSearchResponse:
        """Search verified line items and calculate their unit-price summary."""
        purchases = await self._db.search_products(query, user_id=user_id, can_see_all=can_see_all)
        if not purchases:
            return ProductSearchResponse(query=query, purchases=[])

        prices = [purchase.unit_price for purchase in purchases]
        # Currency conversion is intentionally out of scope: the current
        # application assumes a user pays in one currency.
        # cheapest_price is the lowest non-negative unit price: a negative Pfand
        # refund line is money returned, not a price a customer pays, so it is
        # excluded (and None is reported when no such price exists).
        non_negative_prices = [price for price in prices if price >= 0]
        return ProductSearchResponse(
            query=query,
            purchases=purchases,
            latest_price=purchases[0].unit_price,
            cheapest_price=min(non_negative_prices) if non_negative_prices else None,
            average_price=sum(prices, Decimal(0)) / len(prices),
            currency=purchases[0].currency,
        )

    async def get_statistics(
        self,
        user_id: UUID | None = None,
        can_see_all: bool = False,
        weeks: int = 12,
        collection_id: UUID | None = None,
    ) -> ReceiptStatistics:
        """Return verified receipt aggregates scoped to the current user."""
        return await self._db.get_statistics(
            user_id=user_id,
            can_see_all=can_see_all,
            weeks=weeks,
            collection_id=collection_id,
        )

    async def get_receipt_with_details(
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ReceiptWithDetails | None:
        return await self._db.get_receipt_with_details(
            receipt_id, user_id=user_id, can_see_all=can_see_all
        )

    async def update_receipt(
        self,
        receipt_id: UUID,
        receipt: Receipt,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> ReceiptRow | None:
        return await self._db.update_receipt(
            receipt_id, receipt, user_id=user_id, can_see_all=can_see_all
        )

    async def verify_receipt(
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ReceiptRow | None:
        return await self._db.verify_receipt(receipt_id, user_id=user_id, can_see_all=can_see_all)

    async def delete_receipt(
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ReceiptRow | None:
        """Delete a receipt; raises ReceiptReferencedError if a benchmark references it."""
        try:
            return await self._db.delete_receipt(
                receipt_id, user_id=user_id, can_see_all=can_see_all
            )
        except asyncpg.ForeignKeyViolationError as exc:
            raise ReceiptReferencedError(str(exc)) from exc

    # ── Image (images table) delegation ──────────────────────────────

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
        return await self._image_db.store_image(
            image_path=image_path,
            original_filename=original_filename,
            media_type=media_type,
            size_bytes=size_bytes,
            status=status,
            user_id=user_id,
            bypass_review=bypass_review,
            thumbnail_path=thumbnail_path,
            additional_images=additional_images,
            model_id=model_id,
        )

    async def claim_image_for_analysis(self, image_id: UUID) -> ImageRow | None:
        """Claim a queued image before an in-request analysis starts."""
        return await self._image_db.claim_for_analysis(image_id)

    async def get_image_by_id(
        self, image_id: UUID, user_id: UUID | None = None, can_see_all: bool = False
    ) -> ImageRow | None:
        return await self._image_db.get_image_by_id(
            image_id, user_id=user_id, can_see_all=can_see_all
        )

    async def list_pending_images(self) -> list[ImageRow]:
        return await self._image_db.list_pending_images()

    async def list_images(
        self,
        status: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
        user_id: UUID | None = None,
        can_see_all: bool = False,
    ) -> list[ImageRow]:
        return await self._image_db.list_images(
            status=status, limit=limit, offset=offset, user_id=user_id, can_see_all=can_see_all
        )

    async def mark_image_analyzed(self, image_id: UUID, receipt_id: UUID) -> None:
        await self._image_db.mark_analyzed(image_id, receipt_id)

    async def mark_image_failed(self, image_id: UUID, error: str) -> None:
        await self._image_db.mark_failed(image_id, error)

    async def mark_image_terminal(self, image_id: UUID, status: str, error: str) -> None:
        await self._image_db.mark_terminal(image_id, status, error)

    async def move_additional_images(self, image: ImageRow, receipt_id: UUID) -> None:
        for part in image.additional_images:
            path, thumb = self._image_service.store_perm_image(Path(part.image_path), receipt_id)
            if path is not None:
                part.image_path = str(path)
            if thumb is not None:
                part.thumbnail_path = str(thumb)
            await self._image_db.update_additional_images(image.id, image.additional_images)

    async def update_image_path(self, image_id: UUID, image_path: str) -> None:
        await self._image_db.update_image_path(image_id, image_path)

    async def update_image_thumbnail_path(self, image_id: UUID, thumbnail_path: str) -> None:
        await self._image_db.update_image_thumbnail_path(image_id, thumbnail_path)

    async def delete_image_row(self, image_id: UUID) -> None:
        await self._image_db.delete_image(image_id)

    # ── Extraction methods ───────────────────────────────────────────

    async def get_available_models(self) -> list[ModelInfo]:
        logger.debug("Requesting available models from provider")
        return await self._provider.get_available_models()

    async def check_connection(self) -> bool:
        """Return True if the LLM backend is reachable, False otherwise."""
        return await self._provider.check_connection()

    async def _available_tags(self) -> list[str]:
        """The tag vocabulary for the prompt; empty when the database is unavailable."""
        if self._experiment_tags is not None:
            return list(self._experiment_tags)
        return await self._db.list_tags() if self.db_ready else []

    async def analyse_receipt_with_ocr(self, model_id: str, image_path: Path) -> Receipt:
        """Two-stage analysis: OCR extraction followed by LLM formatting.

        If OCR is not available or fails, falls back to direct LLM analysis.
        """
        logger.info("Starting two-stage OCR+LLM analysis for image %s", image_path.name)

        if self._ocr_provider is None:
            logger.info("OCR provider not configured, using direct LLM analysis")
            return await self.analyse_receipt_from_path(model_id, image_path)

        # Stage 1: OCR text extraction
        logger.info("Stage 1/2: Running OCR extraction on image %s", image_path.name)
        try:
            ocr_start = time.time()
            ocr_result = await self._ocr_provider.extract_text(image_path)
            ocr_elapsed = time.time() - ocr_start
            logger.info(
                "OCR extraction completed in %.2fs. Result type: %s, keys: %s",
                ocr_elapsed,
                type(ocr_result).__name__,
                list(ocr_result.keys()) if isinstance(ocr_result, dict) else "N/A",
            )
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.warning(
                "OCR extraction failed (%s), falling back to direct LLM analysis: %s",
                type(exc).__name__,
                exc,
            )
            return await self.analyse_receipt_from_path(model_id, image_path)

        # Stage 2: LLM formatting of OCR text
        logger.info("Stage 2/2: Sending OCR text to LLM for formatting (model: %s)", model_id)
        tags = await self._available_tags()
        try:
            llm_start = time.time()
            async with asyncio.timeout(self._provider.analysis_timeout_seconds):
                result = await self._provider.analyse_receipt_from_ocr_text(
                    model_id, ocr_result, tags=tags
                )
            llm_elapsed = time.time() - llm_start
            logger.info(
                "OCR+LLM pipeline completed in %.2fs total (OCR: %.2fs, LLM: %.2fs). "
                "Merchant: %s, Total: %s",
                ocr_elapsed + llm_elapsed,
                ocr_elapsed,
                llm_elapsed,
                result.merchant_name,
                result.total,
            )
            return result
        except (TimeoutError, httpx.TimeoutException, APITimeoutError) as exc:
            raise AnalysisTimeoutError("The LLM request timed out. Please try again.") from exc
        except Exception:
            logger.exception("Failed to analyse receipt with OCR+LLM pipeline")
            raise

    async def analyse_receipt_from_path(
        self, model_id: str, image_path: Path | Sequence[Path]
    ) -> Receipt:
        """Run LLM extraction on an image file whose lifetime the caller owns."""
        logger.info("Analysing receipt using model: %s", model_id)
        tags = await self._available_tags()
        try:
            async with asyncio.timeout(self._provider.analysis_timeout_seconds):
                result = await self._provider.analyse_receipt_from_model(
                    model_id, image_path, tags=tags
                )
            logger.info("Successfully analysed receipt with model: %s", model_id)
            return result
        except (TimeoutError, httpx.TimeoutException, APITimeoutError) as exc:
            raise AnalysisTimeoutError("The LLM request timed out. Please try again.") from exc
        except Exception:
            logger.exception("Failed to analyse receipt with model %s", model_id)
            raise

    async def analyse_receipt_from_model(self, model_id: str, image: bytes) -> Receipt:
        """Store the image temporarily, run extraction, then clean up the tmp file."""
        tmp_path = self._image_service.store_tmp_image(image)
        try:
            return await self.analyse_receipt_from_path(model_id, tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    async def extract_receipt_all_models(self, image: UploadFile) -> list[ModelResult]:
        """Run extraction concurrently across every available model."""
        logger.info("Starting concurrent extraction across all models for uploaded file")
        models = await self.get_available_models()
        tags = await self._available_tags()
        content = await image.read()
        tmp_path = self._image_service.store_tmp_image(content)

        async def run_one(model: ModelInfo) -> ModelResult:
            try:
                logger.debug("Running extraction for model: %s", model.id)
                receipt = await self._provider.analyse_receipt_from_model(
                    model.id, tmp_path, tags=tags
                )
                return ModelResult(model_id=model.id, receipt=receipt)
            except ValueError as e:
                logger.warning("Model %s failed with expected warning: %s", model.id, e)
                return ModelResult(model_id=model.id, receipt=None, error=str(e))
            except Exception as e:
                logger.exception("Unexpected failure for model %s", model.id)
                return ModelResult(model_id=model.id, receipt=None, error=str(e))

        try:
            results = await asyncio.gather(*(run_one(m) for m in models))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        logger.info("Completed concurrent extraction across all models")
        return results

    async def analyse_submission_with_metadata(
        self, model_id: str, paths: list[Path], options: "ExperimentOptions", tags: list[str]
    ) -> "ExperimentResult":
        """Run an isolated experiment through normal upload extraction paths."""
        from .experiment_execution import execute_submission
        return await execute_submission(self, model_id, paths, options, tags)
