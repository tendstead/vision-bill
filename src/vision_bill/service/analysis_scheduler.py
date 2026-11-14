import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from ..config import Settings
from ..model.db.image import ImageRow
from ..provider.db.image_db import ImageDB
from ..provider.llm.base import LLMProvider, ModelInfo
from ..provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError
from .image_service import ImageService
from .receipt_service import ReceiptService

logger = logging.getLogger(__name__)


@dataclass
class PendingImageResult:
    """Outcome of analysing a single queued image."""

    image_id: UUID
    status: str  # analyzed | failed | timed_out | unreadable
    receipt_id: UUID | None = None
    error: str | None = None


class AnalysisScheduler:
    """Drains the pending-image queue by running LLM extraction on each image.

    Runs as a background task started from the app lifespan and can also be
    triggered on demand via the ``POST /images/analyze`` endpoint. Both paths
    funnel through :meth:`process_pending`, which is guarded by an
    ``asyncio.Lock`` so a manual trigger and the periodic cycle never overlap.

    There is deliberately no terminal "unreachable" state: if the LLM provider
    is down at cycle time the images simply stay ``pending`` and are retried on
    the next cycle.
    """

    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        receipt_service: ReceiptService,
        image_db: ImageDB,
        image_service: ImageService | None = None,
    ):
        self._settings = settings
        self._provider = provider
        self._receipt_service = receipt_service
        self._image_db = image_db
        self._image_service = image_service
        self._interval = settings.worker.check_interval_seconds
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def check_interval_seconds(self) -> int:
        return self._interval

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is not None:
            logger.warning("Analysis scheduler already running - skipping start")
            return
        self._wake.set()
        self._task = asyncio.create_task(self._run_forever())
        logger.info("Analysis scheduler started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("Analysis scheduler stopped")

    def trigger(self) -> None:
        """Wake the existing worker immediately; work remains persisted in PostgreSQL."""
        self._wake.set()

    async def _run_forever(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                await self.process_pending()
            except Exception:
                logger.exception("Unexpected error in analysis cycle")

    # ── Core cycle ───────────────────────────────────────────────────

    async def process_pending(self) -> list[PendingImageResult]:
        """Run one analysis cycle, serialised against concurrent triggers."""
        async with self._lock:
            if not (self._receipt_service.db_ready and self._image_db.is_ready):
                logger.info("Database not ready - skipping analysis cycle")
                return []
            return await self._process_pending_unlocked()

    def _select_model(self, models: list[ModelInfo]) -> str:
        """Prefer the configured model, fall back to the first vision model."""
        preferred = self._settings.llm.model_name
        for model in models:
            if model.id == preferred:
                return model.id
        return models[0].id

    async def _process_pending_unlocked(self) -> list[PendingImageResult]:
        try:
            async with asyncio.timeout(5):
                models = await self._provider.get_available_models()
        except Exception:  # noqa: BLE001 - worker boundary: any failure keeps images queued
            logger.warning("LLM provider unreachable - pending images stay queued")
            return []

        if not models:
            logger.warning("No vision models available - pending images stay queued")
            return []

        model_id = self._select_model(models)
        pending = await self._image_db.list_pending_images()
        logger.info("Analysis cycle: %d image(s) queued, using model %s", len(pending), model_id)
        results: list[PendingImageResult] = []
        for image in pending:
            claimed = await self._image_db.claim_for_analysis(image.id)
            if claimed is None:
                logger.info("Image %s was claimed by another worker", image.id)
                continue
            chosen = claimed.model_id if claimed.model_id in {m.id for m in models} else model_id
            results.append(await self._analyze_one(claimed, chosen))
        return results

    async def _analyze_one(self, image: ImageRow, model_id: str) -> PendingImageResult:
        if image.receipt_id is not None:
            await self._image_db.mark_analyzed(image.id, image.receipt_id)
            return PendingImageResult(
                image_id=image.id, status="analyzed", receipt_id=image.receipt_id
            )

        existing = await self._receipt_service.get_receipt_by_image_id(image.id)
        if existing is not None:
            await self._image_db.mark_analyzed(image.id, existing.id)
            return PendingImageResult(image_id=image.id, status="analyzed", receipt_id=existing.id)

        image_path = Path(image.image_path) if image.image_path else None
        if image_path is None or not image_path.exists():
            error = f"Image file missing: {image.image_path}"
            logger.error(error)
            await self._image_db.mark_failed(image.id, error)
            return PendingImageResult(image_id=image.id, status="failed", error=error)

        try:
            if image.additional_images:
                paths = [Path(part.image_path) for part in image.parts()]
                receipt = await self._receipt_service.analyse_receipt_from_path(model_id, paths)
            else:
                receipt = await self._receipt_service.analyse_receipt_with_ocr(model_id, image_path)
        except (AnalysisTimeoutError, UnreadableReceiptError) as exc:
            status = "timed_out" if isinstance(exc, AnalysisTimeoutError) else "unreadable"
            await self._image_db.mark_terminal(image.id, status, str(exc))
            return PendingImageResult(image_id=image.id, status=status, error=str(exc))
        except Exception as e:  # noqa: BLE001 - worker boundary: fail the image, never the cycle
            # The provider already exhausted its self-correction retry loop.
            await self._image_db.mark_failed(image.id, str(e))
            return PendingImageResult(image_id=image.id, status="failed", error=str(e))

        bypass = image.bypass_review
        try:
            row = await self._receipt_service.persist_receipt(
                receipt,
                image_id=image.id,
                status="verified" if bypass else "unverified",
                verified=bypass,
                user_id=image.user_id,
            )
            await self._image_db.mark_analyzed(image.id, row.id)
        except Exception as e:  # noqa: BLE001 - worker boundary: fail the image, never the cycle
            error = f"Failed to persist receipt: {e}"
            await self._image_db.mark_failed(image.id, error)
            return PendingImageResult(image_id=image.id, status="failed", error=error)

        if bypass and self._image_service is not None:
            try:
                perm_path, perm_thumb = self._image_service.store_perm_image(image_path, row.id)
                if perm_path is not None:
                    await self._image_db.update_image_path(image.id, str(perm_path))
                if perm_thumb is not None:
                    await self._image_db.update_image_thumbnail_path(image.id, str(perm_thumb))
                if image.additional_images:
                    await self._receipt_service.move_additional_images(image, row.id)
            except Exception:
                logger.exception(
                    "Failed to move bypass-reviewed image %s to permanent storage", image.id
                )

        return PendingImageResult(image_id=image.id, status="analyzed", receipt_id=row.id)
