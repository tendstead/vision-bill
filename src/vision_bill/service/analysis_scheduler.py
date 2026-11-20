"""Wakeable durable queue consumer; no inference, image moves or unfenced writes."""

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from ..config import Settings
from ..model.handoff import HandoffWait
from ..model.workflow import AccessScope
from ..provider.db.image_db import ImageDB
from ..provider.llm.base import LLMProvider
from .activation_admission import inference_admission
from .image_service import ImageService
from .receipt_service import ReceiptService
from .submission_service import SubmissionService

logger = logging.getLogger(__name__)


@dataclass
class PendingImageResult:
    image_id: UUID
    status: str
    receipt_id: UUID | None = None
    error: str | None = None


class AnalysisScheduler:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        receipt_service: ReceiptService,
        image_db: ImageDB,
        image_service: ImageService | None = None,
        *,
        submission_service: SubmissionService | None = None,
    ):
        self._settings = settings
        self._provider = provider
        self._db = receipt_service.workflow_db
        self._submissions = submission_service or SubmissionService(
            settings, receipt_service, image_service or ImageService(settings.images)
        )
        self._interval = settings.worker.check_interval_seconds
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._deferred_image_ids: set[UUID] = set()
        self._task: asyncio.Task[None] | None = None

    @property
    def check_interval_seconds(self) -> int:
        return self._interval

    async def start(self) -> None:
        if self._task is None:
            self._wake.set()
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def trigger(self) -> None:
        self._wake.set()

    async def _run_forever(self) -> None:
        delay = float(self._interval)
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                await self.process_pending()
                delay = (
                    await self._db.next_wake_delay(
                        self._interval, deferred_image_ids=tuple(self._deferred_image_ids)
                    )
                    if self._db.is_ready
                    else float(self._interval)
                )
            except Exception:  # noqa: BLE001 - general logs must never contain receipt data
                logger.warning("Analysis cycle unavailable")
                delay = float(self._interval)

    async def process_pending(
        self, *, scope: AccessScope | None = None
    ) -> list[PendingImageResult]:
        async with self._lock:
            self._deferred_image_ids.clear()
            if not self._db.is_ready:
                return []
            await self._db.reap_expired_claims()
            results = []
            for image in await self._db.due_candidates(purpose=None):
                completion = image.observations_available
                try:
                    with inference_admission(self._provider):
                        try:
                            async with asyncio.timeout(5):
                                ready = await (
                                    self._submissions.ready_for_completion(image) if completion
                                    else self._submissions.ready_for(image)
                                )
                        except Exception:  # noqa: BLE001 - preclaim failure spends no attempt
                            ready = False
                        if not ready:
                            self._deferred_image_ids.add(image.id)
                            continue
                        lease = self._settings.worker.claim_lease_seconds
                        if completion:
                            claim = await self._db.claim_next(
                                lease_seconds=lease, image_id=image.id, purpose="completion"
                            )
                        else:
                            claim = await self._db.claim_next(lease_seconds=lease, image_id=image.id)
                        if claim is None:
                            # A competing claim must not create a zero-delay loop.
                            self._deferred_image_ids.add(image.id)
                            continue
                        result = await (
                            self._submissions.complete_checkpoint(claim) if completion
                            else self._submissions.process_claim(claim)
                        )
                except HandoffWait as exc:
                    await self._db.set_wait_reason(image, str(exc))
                    self._deferred_image_ids.add(image.id)
                    continue
                if result.submission is not None and result.submission.wait_reason:
                    self._deferred_image_ids.add(image.id)
                if result.submission is not None and (
                    scope is None or scope.can_see_all or claim.image.user_id == scope.user_id
                ):
                    status = result.submission
                    results.append(
                        PendingImageResult(
                            status.id,
                            status.status,
                            receipt_id=status.receipt_id,
                            error=status.result_code,
                        )
                    )
            return results
