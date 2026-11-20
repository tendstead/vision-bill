import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import UUID

import asyncpg
import httpx
from fastapi import UploadFile
from openai import APITimeoutError

from ..config import ImageSettings, OcrSettings, PGSettings, Settings, WorkerSettings
from ..model.db.image import ImagePart, ImageRow
from ..model.db.receipt import ReceiptRow, ReceiptWithDetails
from ..model.extraction import InferenceEnvelope
from ..model.handoff import HandoffStatus, HandoffWait
from ..model.ocr import OcrCapabilities, OcrResponse, SourceRef
from ..model.receipt import Receipt
from ..model.search import ProductSearchResponse
from ..model.statistics import ReceiptStatistics
from ..model.workflow import (
    AccessScope,
    EnqueueResult,
    ReviewPreview,
    ReviewRequest,
    SubmissionCreate,
    SubmissionStatus,
)
from ..provider.db.collection_db import CollectionDB
from ..provider.db.image_db import ImageDB
from ..provider.db.receipt_db import ReceiptDB
from ..provider.db.user_db import UserDB
from ..provider.db.workflow_db import WorkflowDB
from ..provider.llm.base import LLMProvider, ModelInfo
from ..provider.llm.errors import AnalysisTimeoutError
from ..provider.ocr.base import OcrProvider
from .image_service import ImageService

if TYPE_CHECKING:
    from ..model.experiment import ExperimentOptions
    from .experiment_execution import ExperimentResult
    from .extraction_pipeline import ExtractionPipeline, UnmanagedClassificationEvent
    from .gpu_handoff import ModelHandoff

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
        ocr_settings: OcrSettings | None = None,
        worker_settings: WorkerSettings | None = None,
    ):
        self._provider: LLMProvider = provider
        self._ocr_provider: OcrProvider | None = ocr_provider
        self._ocr_settings = ocr_settings or OcrSettings(enabled=ocr_provider is not None)
        self._image_service = ImageService(image_settings)
        self._db = ReceiptDB(pg_settings, original_checker=self._image_service.check_originals)
        self._image_db = ImageDB(pg_settings)
        self._user_db = UserDB(pg_settings)
        self._workflow_db = WorkflowDB(
            pg_settings, worker_settings=worker_settings,
            original_checker=self._image_service.check_originals,
            working_view_checker=self._image_service.read_working_image,
        )
        self._collection_db: CollectionDB | None = None

    # ── Database delegation ──────────────────────────────────────────

    async def init_db(self) -> None:
        await self._db.init_db()
        await self._image_db.init_db()
        await self._user_db.init_db()
        await self._workflow_db.init_db()

    async def destroy_db(self) -> None:
        await self._db.destroy_db()
        await self._image_db.destroy_db()
        await self._user_db.destroy_db()
        await self._workflow_db.destroy_db()

    @property
    def workflow_db(self) -> WorkflowDB:
        """Durable submission provider shared with the workflow worker."""
        return self._workflow_db

    @property
    def workflow_ready(self) -> bool:
        return self._workflow_db.is_ready

    async def enqueue_submission(
        self, submission: SubmissionCreate, *, scope: AccessScope, idempotency_key: str | None
    ) -> EnqueueResult:
        return await self._workflow_db.enqueue_submission(
            submission, scope=scope, idempotency_key=idempotency_key
        )

    async def get_submission(
        self, image_id: UUID, *, scope: AccessScope
    ) -> SubmissionStatus | None:
        return await self._workflow_db.get_submission(image_id, scope=scope)

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

    async def list_tags(self, *, deadline: float | None = None) -> list[str]:
        """Return the allowed line-item tag vocabulary from the database."""
        if deadline is None:
            return await self._db.list_tags()
        return await self._db.list_tags(deadline=deadline)

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
                await self._collection_db.assign(active.id, receipt_id, user_id=user_id)
        except Exception:  # noqa: BLE001 - optional capture must not fail extraction
            logger.warning("collection_capture_failed")

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

        prices = [purchase.unit_price for purchase in purchases if purchase.unit_price is not None]
        # Currency conversion is intentionally out of scope: the current
        # application assumes a user pays in one currency.
        # cheapest_price is the lowest non-negative unit price: a negative Pfand
        # refund line is money returned, not a price a customer pays, so it is
        # excluded (and None is reported when no such price exists).
        non_negative_prices = [price for price in prices if price >= 0]
        return ProductSearchResponse(
            query=query,
            purchases=purchases,
            latest_price=prices[0] if prices else None,
            cheapest_price=min(non_negative_prices) if non_negative_prices else None,
            average_price=sum(prices, Decimal(0)) / len(prices) if prices else None,
            currency=next(
                (purchase.currency for purchase in purchases if purchase.unit_price is not None),
                purchases[0].currency,
            ),
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
        self, receipt_id: UUID, user_id: UUID | None = None, can_see_all: bool = False,
        *, deadline: float | None = None,
    ) -> ReceiptWithDetails | None:
        if deadline is not None:
            return await self._db.get_receipt_with_details(
                receipt_id, user_id=user_id, can_see_all=can_see_all, deadline=deadline
            )
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

    async def preview_human_review(
        self, receipt_id: UUID, *, scope: AccessScope
    ) -> ReviewPreview | None:
        return await self._db.preview_human_review(receipt_id, scope=scope)

    async def verify_receipt(
        self, receipt_id: UUID, request: ReviewRequest | None = None, *,
        scope: AccessScope | None = None, user_id: UUID | None = None, can_see_all: bool = False,
    ) -> ReceiptRow | None:
        return await self._db.verify_receipt(receipt_id, request, scope=scope, user_id=user_id, can_see_all=can_see_all)

    async def delete_resource(self, resource_id: UUID, *, receipt: bool, scope: AccessScope, images: "ImageService") -> bool:
        try:
            deleted = await self.workflow_db.delete_resource(resource_id, receipt=receipt, scope=scope, validate_files=lambda deleted: images.validate_deleted_files(deleted, initial=True))
        except asyncpg.ForeignKeyViolationError as exc:
            raise ReceiptReferencedError("resource_referenced") from exc
        if deleted is None:
            return False
        await self.workflow_db.cleanup_deleted_resource(resource_id, receipt=receipt, scope=scope, remove_files=images.remove_deleted_files)
        return True

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

    async def update_image_path(self, image_id: UUID, image_path: str) -> None:
        await self._image_db.update_image_path(image_id, image_path)

    async def update_image_thumbnail_path(self, image_id: UUID, thumbnail_path: str) -> None:
        await self._image_db.update_image_thumbnail_path(image_id, thumbnail_path)

    async def delete_image_row(self, image_id: UUID) -> None:
        await self._image_db.delete_image(image_id)

    # ── Extraction methods ───────────────────────────────────────────

    async def get_available_models(self, *, capability: str = "vision") -> list[ModelInfo]:
        logger.debug("Requesting available models from provider")
        return await self._provider.get_available_models(capability=capability)

    async def check_connection(self) -> bool:
        """Return True if the LLM backend is reachable, False otherwise."""
        return await self._provider.check_connection()

    @property
    def inference_identity(self) -> str:
        return self._provider.inference_identity

    def managed_ocr_compatible(self) -> bool:
        """Check the actual LLM's handoff contract without resource side effects."""
        from .gpu_handoff import handoff_resource_identity

        if not self._provider.supports_model_release:
            return False
        try:
            handoff_resource_identity(self._provider, "llm")
        except ValueError:
            return False
        return True

    async def _available_tags(self) -> list[str]:
        """The tag vocabulary for the prompt; empty when the database is unavailable."""
        if self._experiment_tags is not None:
            return list(self._experiment_tags)
        return await self._db.list_tags() if self.db_ready else []

    async def ocr_capabilities(self, settings: OcrSettings) -> OcrCapabilities:
        provider = self._extraction_pipeline(settings).ocr
        if provider is None:
            return OcrCapabilities(enabled=False, configured=False, reachable=False, ready=False)
        return await provider.get_capabilities()

    def set_ocr_route(self, provider: OcrProvider | None, settings: OcrSettings) -> None:
        """Publish the qualified selection while the app holds activation admission."""
        self._ocr_provider = provider
        self._ocr_settings = settings

    def _extraction_pipeline(self, settings: OcrSettings | None = None) -> "ExtractionPipeline":
        from ..provider.factory import get_ocr_provider
        from .extraction_pipeline import ExtractionPipeline

        if settings is None or (settings == self._ocr_settings and self._ocr_provider is not None):
            return ExtractionPipeline(self._provider, self._ocr_provider, self._ocr_settings)
        return ExtractionPipeline(self._provider, get_ocr_provider(settings), settings)

    def execution_handoff(self, settings: OcrSettings) -> "ModelHandoff | None":
        from .execution_policy import managed_ocr

        selected = self._extraction_pipeline(settings).coordinator()
        live = self._extraction_pipeline().coordinator() if self._ocr_provider is not None else selected
        if ((managed_ocr(settings) and selected is None)
                or (selected is None) != (live is None)
                or (live is not None and selected is not None and not selected.matches(live))):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="execution_policy_changed",
            ))
        return selected

    def current_execution_snapshot(self, settings: Settings) -> dict[str, Any]:
        from .execution_policy import current_snapshot

        try:
            handoff = self.execution_handoff(settings.ocr)
        except (HandoffWait, ValueError, OSError):
            handoff = None
        return current_snapshot(settings, inference_identity=self.inference_identity, handoff=handoff)

    async def extract_receipt(
        self, model_id: str, paths: Sequence[Path], *, options: "ExperimentOptions",
        deadline: float, tags: Sequence[str],
        checkpoint: Callable[[OcrResponse], Awaitable[None]] | None = None,
        observation_checkpoint: Callable[[InferenceEnvelope], Awaitable[None]] | None = None,
        unmanaged_classification_event: "UnmanagedClassificationEvent | None" = None,
        resume_pages: Sequence[OcrResponse] = (), source_refs: Sequence[SourceRef] = (),
        ocr_settings: OcrSettings | None = None, max_output_tokens: int | None = None,
        expected_binding_sha256: str | None = None, handoff_deadline: float | None = None,
    ) -> InferenceEnvelope:
        """Explicit per-attempt snapshots; never inject state into shared providers."""
        if ocr_settings is not None:
            self.execution_handoff(ocr_settings)
        return await self._extraction_pipeline(ocr_settings).extract(
            model_id, paths, options=options, deadline=deadline, tags=tags,
            checkpoint=checkpoint, resume_pages=resume_pages, source_refs=source_refs,
            observation_checkpoint=observation_checkpoint,
            unmanaged_classification_event=unmanaged_classification_event,
            max_output_tokens=max_output_tokens, expected_binding_sha256=expected_binding_sha256,
            handoff_deadline=handoff_deadline,
        )

    async def analyse_receipt_from_path(
        self, model_id: str, image_path: Path | Sequence[Path]
    ) -> Receipt:
        """Run normal extraction on caller-owned originals using configured OCR."""
        logger.info("Analysing receipt using model: %s", model_id)
        tags = await self._available_tags()
        try:
            async with asyncio.timeout(self._provider.analysis_timeout_seconds):
                if self._ocr_settings.enabled:
                    from ..model.experiment import ExperimentOptions

                    paths = [image_path] if isinstance(image_path, Path) else list(image_path)
                    envelope = await self.extract_receipt(
                        model_id, paths, options=ExperimentOptions(),
                        deadline=monotonic() + self._provider.analysis_timeout_seconds, tags=tags,
                    )
                    if envelope.receipt is None:
                        raise ValueError(envelope.findings[-1].code if envelope.findings else "receipt_incomplete")
                    result = envelope.receipt
                else:
                    result = await self._provider.analyse_receipt_from_model(
                        model_id, image_path, tags=tags
                    )
            logger.info("Successfully analysed receipt with model: %s", model_id)
            return result
        except (TimeoutError, httpx.TimeoutException, APITimeoutError) as exc:
            raise AnalysisTimeoutError("The LLM request timed out. Please try again.") from exc
        except Exception:
            logger.warning("receipt_extraction_failed")
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
        models = await self.get_available_models(
            capability="text" if self._ocr_settings.enabled else "vision"
        )
        content = await image.read()
        tmp_path = self._image_service.store_tmp_image(content)

        async def run_one(model: ModelInfo) -> ModelResult:
            try:
                logger.debug("Running extraction for model: %s", model.id)
                receipt = await self.analyse_receipt_from_path(model.id, tmp_path)
                return ModelResult(model_id=model.id, receipt=receipt)
            except ValueError as e:
                logger.warning("receipt_model_failed: %s", type(e).__name__)
                return ModelResult(model_id=model.id, receipt=None, error=str(e))
            except Exception as e:  # noqa: BLE001 - isolate independent model failures
                logger.warning("receipt_model_failed")
                return ModelResult(model_id=model.id, receipt=None, error=str(e))

        try:
            results = await asyncio.gather(*(run_one(m) for m in models))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        logger.info("Completed concurrent extraction across all models")
        return results

    async def analyse_submission_with_metadata(
        self, model_id: str, paths: list[Path], options: "ExperimentOptions", tags: list[str],
        *, deadline: float | None = None,
    ) -> "ExperimentResult":
        """Run an isolated experiment through normal upload extraction paths."""
        from .experiment_execution import execute_submission
        return await execute_submission(self, model_id, paths, options, tags, deadline=deadline)
