"""One fenced attempt, ordered checkpoint recovery, and owner-scoped draft operations."""

import asyncio
import base64
import hashlib
import logging
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx

from ..config import OcrSettings, Settings
from ..helper.deadline import check_deadline, run_sync_before
from ..model.db.image import ImageRow
from ..model.db.receipt import ReceiptRow
from ..model.experiment import ExperimentOptions
from ..model.extraction import (
    EXTRACTION_SCHEMA_VERSION,
    SOURCE_PROJECTION_VERSION,
    DraftView,
    ExtractionDraft,
    InferenceEnvelope,
    pack_inference_metadata,
)
from ..model.handoff import HandoffStatus, HandoffWait
from ..model.ocr import OcrPageMetadata, OcrResponse, SourceRef, WorkingImage, WorkingImageMetadata
from ..model.ocr_errors import OCR_RESULT_CODES
from ..model.receipt import Receipt
from ..model.review import OriginalAvailability
from ..model.workflow import (
    AccessScope,
    AnalysisClaim,
    AttemptOutcome,
    CheckpointCompletionOutcome,
    CheckpointExecutionIdentity,
    CommitResult,
    DraftFinalize,
    DraftPatch,
    ObservationCheckpoint,
    OcrCheckpoint,
    PersistedSourcePart,
    PersistedWorkingView,
    SourcePartBinding,
)
from ..provider.db.workflow_db import OBSERVATION_RECOVERY_WAIT
from ..provider.llm.base import ModelInfo
from ..provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError
from ..provider.ocr.base import compact_ocr_text
from .benchmark_scoring import PROMPT_VERSION
from .execution_policy import (
    managed_ocr,
    policy_hold_reason,
    semantic_policy_sha256,
    snapshot_ocr,
)
from .extraction_pipeline import (
    OcrRuntimeChangedError,
    OriginalSourceChangedError,
    ReadinessHoldError,
    select_vision_model,
)
from .gpu_handoff import ModelHandoff, managed_ocr_enabled
from .image_service import ImageService
from .observation_admission import admit_observations
from .receipt_review import evaluate_receipt
from .receipt_service import ReceiptService

logger = logging.getLogger(__name__)


class StaleClaimError(Exception):
    """A worker lost its fence; it must not publish any outcome."""


class SourceIntegrityError(Exception):
    """Original bytes/path/order no longer match the durable submission."""


class PolicyHoldError(Exception):
    """The operator policy changed after readiness or while preparing dispatch."""


class CheckpointRecoveryError(Exception):
    """Owned resource reconciliation has not been confirmed."""


class SubmissionService:
    def __init__(self, settings: Settings, receipts: ReceiptService, images: ImageService):
        self.settings = settings
        self.worker = settings.worker
        self.receipts = receipts
        self.db = receipts.workflow_db
        self.images = images

    def current_snapshot(self) -> dict[str, Any]:
        return self.receipts.current_execution_snapshot(self.settings)

    @staticmethod
    def _initial_model(models: list[ModelInfo], preferred: str, *, vision: bool) -> ModelInfo | None:
        """Resolve the selected model from affirmative capability discovery."""
        if vision:
            return select_vision_model(models, preferred)
        return next((model for model in models if model.id == preferred), None)

    async def ready_for(self, image: ImageRow) -> bool:
        """Persist only safe reasons against the observed generation/status before a claim."""
        try:
            async with asyncio.timeout(4):
                reason = await self._readiness_reason(image)
        except TimeoutError:
            reason = "provider_unavailable"
        observed = await self.db.set_wait_reason(image, reason)
        return reason is None and observed

    def _completion_handoff(
        self, image: ImageRow, checkpoint: ObservationCheckpoint,
    ) -> ModelHandoff | None:
        snapshot = image.effective_options
        identity = checkpoint.execution_identity
        if (
            checkpoint.eligible_generation != image.generation
            or checkpoint.producer_attempt_number != image.attempt_count
            or checkpoint.model_id != image.model_id
            or identity.semantic_policy_sha256 != semantic_policy_sha256(snapshot)
        ):
            raise ValueError("checkpoint producer or frozen policy mismatch")
        if (
            policy_hold_reason(
                snapshot, self.settings.ocr, self.receipts.inference_identity,
                checkpoint_identity=identity.inference_identity,
            )
            or identity.inference_identity != self.receipts.inference_identity
            or identity.prompt_version != PROMPT_VERSION
            or identity.extraction_schema_version != EXTRACTION_SCHEMA_VERSION
            or identity.source_projection_version != SOURCE_PROJECTION_VERSION
        ):
            raise PolicyHoldError
        ocr = snapshot_ocr(snapshot)
        if managed_ocr(ocr) and not managed_ocr_enabled():
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="model_handoff_managed_ocr_required",
            ))
        return self.receipts.execution_handoff(ocr)

    async def ready_for_completion(self, image: ImageRow) -> bool:
        """Check frozen policy and owned-resource recovery without model discovery."""
        if image.unmanaged_classification_unsettled:
            await self.db.set_wait_reason(image, "model_request_unsettled")
            return False
        try:
            checkpoint = ObservationCheckpoint.model_validate(image.observation_checkpoint)
            handoff = self._completion_handoff(image, checkpoint)
            reason = None
            if handoff is not None:
                status = await handoff.inspect(
                    model_id=checkpoint.model_id, deadline=monotonic() + 1,
                    expected_binding_sha256=(image.effective_options.get("managed_binding_sha256") or "")
                    if handoff.shared_gpu else None,
                )
                if status.state != "ready":
                    reason = status.reason_code or OBSERVATION_RECOVERY_WAIT
        except PolicyHoldError:
            reason = "execution_policy_changed"
        except HandoffWait as exc:
            reason = exc.status.reason_code or OBSERVATION_RECOVERY_WAIT
        except (ValueError, KeyError):
            reason = "observation_checkpoint_invalid"
        observed = await self.db.set_wait_reason(image, reason)
        return reason is None and observed

    async def _readiness_reason(self, image: ImageRow) -> str | None:
        """Readiness precedes claim, and therefore never spends the attempt budget."""
        snapshot = image.effective_options
        try:
            reason = policy_hold_reason(snapshot, self.settings.ocr, self.receipts.inference_identity)
            if reason:
                return reason
            ocr = snapshot_ocr(snapshot)
        except (ValueError, KeyError):
            # Claim malformed saved options to record a terminal safe diagnostic.
            return None
        try:
            selected = image.model_id or self.settings.llm.model_name
            if managed_ocr(ocr):
                if not managed_ocr_enabled():
                    return "model_handoff_managed_ocr_required"
                if ocr.shared_gpu and not self.receipts.managed_ocr_compatible():
                    return "model_release_unsupported"
                handoff = self.receipts.execution_handoff(ocr)
                assert handoff is not None
                status = await handoff.inspect(
                    model_id=selected, deadline=monotonic() + 1,
                    expected_binding_sha256=(snapshot.get("managed_binding_sha256") or "")
                    if image.effective_options and handoff.shared_gpu else None,
                )
                if status.state != "ready":
                    return status.reason_code
            else:
                self.receipts.execution_handoff(ocr)
            if not await self.receipts.check_connection():
                return "provider_unavailable"
            capability = "vision"
            if ocr.enabled:
                capabilities = await self.receipts.ocr_capabilities(ocr)
                if capabilities.reason_code == "ocr_mode_mismatch":
                    return "ocr_mode_mismatch"
                if not (
                    capabilities.ready or (ocr.manage_model_lifecycle and capabilities.can_load)
                ):
                    return capabilities.reason_code if capabilities.reason_code in OCR_RESULT_CODES else "ocr_unavailable"
                capability = "text"
            models = await self.receipts.get_available_models(capability=capability)
            if self._initial_model(models, selected, vision=capability == "vision") is not None:
                return None
            return "vision_model_unavailable" if capability == "vision" else "formatter_model_unavailable"
        except HandoffWait as exc:
            return str(exc)
        except (httpx.HTTPError, OSError, ValueError, KeyError):
            return "provider_unavailable"

    async def process_claim(self, claim: AnalysisClaim) -> CommitResult:
        if claim.purpose != "inference":
            return CommitResult(stale=True)
        deadline = monotonic() + self.worker.attempt_timeout_seconds
        # Convert once: asyncio loop time need not share monotonic()'s epoch.
        loop = asyncio.get_running_loop()
        db_deadline = loop.time() + max(0.0, deadline - monotonic())
        owned: list[PersistedWorkingView] = []
        try:
            async with asyncio.timeout_at(db_deadline):
                return await self._process_claim(claim, deadline, owned, db_deadline=db_deadline)
        except TimeoutError:
            # A hung DB cannot extend the complete attempt. Lease recovery owns
            # any still-uncommitted outcome; stale fencing prevents a late write.
            return CommitResult(stale=True)
        finally:
            if owned and monotonic() < deadline and loop.time() < db_deadline:
                # The DB locks the submission before checking references, including
                # transactions whose commit acknowledgement was lost. Uncertain
                # outcomes retain artifacts; never delete a referenced file.
                try:
                    cleanup_deadline = min(deadline, monotonic() + 5.0)
                    cleanup_db_deadline = min(db_deadline, loop.time() + 5.0)
                    async with asyncio.timeout_at(cleanup_db_deadline):
                        unused = await self.db.unreferenced_views(
                            claim.image.id, owned, deadline=cleanup_db_deadline
                        )
                        for view in unused:
                            check_deadline(cleanup_deadline)
                            if loop.time() >= cleanup_db_deadline:
                                raise TimeoutError("database_attempt_deadline")
                            await run_sync_before(
                                cleanup_deadline,
                                partial(
                                    self.images.remove_unreferenced_working_image,
                                    deadline=cleanup_deadline,
                                ),
                                claim.image.id,
                                view.artifact_path,
                                view.metadata,
                            )
                except Exception:  # noqa: BLE001 - a safe orphan is preferable to lost evidence
                    logger.warning("Working artifact cleanup deferred")

    async def complete_checkpoint(self, claim: AnalysisClaim) -> CommitResult:
        checkpoint_id = claim.checkpoint_id
        if claim.purpose != "completion" or checkpoint_id is None:
            return CommitResult(stale=True)
        budget = self.worker.attempt_timeout_seconds
        deadline = monotonic() + budget
        db_deadline = asyncio.get_running_loop().time() + budget
        try:
            async with asyncio.timeout_at(db_deadline):
                return await self._complete_checkpoint(claim, checkpoint_id, deadline, db_deadline)
        except TimeoutError:
            return CommitResult(stale=True)

    async def _complete_checkpoint(
        self, claim: AnalysisClaim, checkpoint_id: UUID, deadline: float, db_deadline: float,
    ) -> CommitResult:
        try:
            async with asyncio.timeout_at(db_deadline - self._outcome_reserve()):
                checkpoint = await self.db.get_observation_checkpoint(claim, deadline=db_deadline)
                if checkpoint is None:
                    return await self.db.get_checkpoint_completion_result(
                        claim, checkpoint_id, deadline=db_deadline,
                    )
                handoff = self._completion_handoff(claim.image, checkpoint)
                envelope = checkpoint.recovered_envelope()
                originals = [OriginalAvailability(
                    part_id=part.part_id, photo_index=part.photo_index,
                    available=True, sha256_matches=True,
                ) for part in checkpoint.source_parts]
                outcome = CheckpointCompletionOutcome(
                    envelope=envelope, persisted_views=checkpoint.persisted_views,
                    review=evaluate_receipt(
                        admit_observations(envelope), envelope.amount_policy, envelope.field_sources,
                        envelope.overlap_decisions, originals, retained_findings=envelope.findings,
                    ),
                )

                async def check_before_recovery() -> None:
                    self._completion_handoff(claim.image, checkpoint)
                    if await self.db.get_observation_checkpoint(claim, deadline=db_deadline) != checkpoint:
                        raise StaleClaimError

                async with AsyncExitStack() as recovery:
                    if handoff is not None:
                        try:
                            await recovery.enter_async_context(handoff.recover_checkpoint(
                                deadline, model_id=checkpoint.model_id,
                                expected_binding_sha256=(claim.image.effective_options.get("managed_binding_sha256") or "")
                                if handoff.shared_gpu else None,
                                before_dispatch=check_before_recovery,
                            ))
                        except HandoffWait:
                            raise
                        except (OSError, RuntimeError, httpx.HTTPError) as exc:
                            raise CheckpointRecoveryError from exc
                    self._completion_handoff(claim.image, checkpoint)
                    return await self.db.finish_checkpoint_completion(
                        claim, checkpoint_id, outcome, deadline=db_deadline,
                    )
        except StaleClaimError:
            return CommitResult(stale=True)
        except PolicyHoldError:
            reason = "execution_policy_changed"
        except HandoffWait as exc:
            reason = exc.status.reason_code or OBSERVATION_RECOVERY_WAIT
        except (CheckpointRecoveryError, TimeoutError):
            reason = OBSERVATION_RECOVERY_WAIT
        except (ValueError, KeyError) as exc:
            reason = "source_integrity_failed" if str(exc) == "source_integrity_failed" else "observation_checkpoint_invalid"
        return await self.db.defer_checkpoint_completion(claim, checkpoint_id, reason, deadline=db_deadline)

    async def _process_claim(
        self, claim: AnalysisClaim, deadline: float, owned: list[PersistedWorkingView],
        *, db_deadline: float,
    ) -> CommitResult:
        try:
            # Keep the durable final write inside its own reserve on both clocks.
            reserve = self._outcome_reserve()
            work_deadline = deadline - reserve
            work_db_deadline = db_deadline - reserve
            async with asyncio.timeout_at(work_db_deadline):
                outcome = await self._attempt(
                    claim, work_deadline, owned, db_deadline=work_db_deadline,
                )
                check_deadline(work_deadline)
            check_deadline(deadline)
            if asyncio.get_running_loop().time() >= db_deadline:
                raise TimeoutError("database_attempt_deadline")
            return await self.db.finish_attempt(claim, outcome, deadline=db_deadline)
        except StaleClaimError:
            return CommitResult(stale=True)
        except PolicyHoldError:
            return await self.db.hold_claim_for_policy(
                claim, "execution_policy_changed", deadline=db_deadline
            )
        except (HandoffWait, ReadinessHoldError) as exc:
            return await self.db.hold_claim_for_readiness(claim, str(exc), deadline=db_deadline)
        except (SourceIntegrityError, OriginalSourceChangedError):
            outcome = AttemptOutcome(kind="terminal", result_code="source_integrity_failed")
        except (TimeoutError, AnalysisTimeoutError):
            outcome = AttemptOutcome(kind="retry", result_code="timed_out")
        except UnreadableReceiptError:
            outcome = AttemptOutcome(kind="terminal", result_code="unreadable")
        except OcrRuntimeChangedError:
            outcome = AttemptOutcome(kind="retry", result_code="ocr_runtime_changed")
        except (ValueError, KeyError):
            outcome = AttemptOutcome(kind="terminal", result_code="analysis_contract_failed")
        except Exception:  # noqa: BLE001 - no receipt content in error responses or logs
            outcome = AttemptOutcome(kind="retry", result_code="analysis_unavailable")
        check_deadline(deadline)
        if asyncio.get_running_loop().time() >= db_deadline:
            raise TimeoutError("database_attempt_deadline")
        async with asyncio.timeout_at(db_deadline):
            return await self.db.finish_attempt(claim, outcome, deadline=db_deadline)

    def _outcome_reserve(self) -> float:
        return min(5.0, self.worker.attempt_timeout_seconds * 0.05)

    @staticmethod
    def _bindings(image: ImageRow) -> list[SourcePartBinding]:
        try:
            return [
                SourcePartBinding(
                    photo_index=i,
                    image_path=p.image_path,
                    source_sha256=hashlib.sha256(Path(p.image_path).read_bytes()).hexdigest(),
                )
                for i, p in enumerate(image.parts())
            ]
        except OSError as exc:
            raise SourceIntegrityError from exc

    async def _attempt(
        self, claim: AnalysisClaim, deadline: float, owned: list[PersistedWorkingView],
        *, db_deadline: float,
    ) -> AttemptOutcome:
        image = claim.image
        snapshot = image.effective_options
        if policy_hold_reason(snapshot, self.settings.ocr, self.receipts.inference_identity):
            raise PolicyHoldError
        if image.model_id is None:
            # Resolve an unbound selection from affirmative provider discovery.
            # This is repeated under the attempt budget and saved exactly once.
            adopted_ocr = snapshot_ocr(snapshot)
            models = await self.receipts.get_available_models(
                capability="text" if adopted_ocr.enabled else "vision"
            )
            preferred = image.model_id or self.settings.llm.model_name
            selected = self._initial_model(models, preferred, vision=not adopted_ocr.enabled)
            if selected is None:
                raise ReadinessHoldError(
                    "formatter_model_unavailable" if adopted_ocr.enabled else "vision_model_unavailable"
                )
            saved = await self.db.ensure_execution_snapshot(
                claim, model_id=selected.id, effective_options=snapshot, deadline=db_deadline
            )
            if saved is None:
                raise StaleClaimError
            image = saved
        snapshot = image.effective_options
        if not image.model_id:
            raise ValueError("execution_snapshot_invalid")
        ocr = snapshot_ocr(snapshot)
        options = ExperimentOptions(temperature=snapshot["temperature"], allow_ocr_fallback=False)
        limit = snapshot["max_output_tokens"]
        if type(limit) is not int or limit <= 0:
            raise ValueError("execution_snapshot_invalid")
        bindings = await run_sync_before(deadline, self._bindings, image)
        try:
            parts = await self.db.bind_source_parts(claim, bindings, deadline=db_deadline)
        except ValueError as exc:
            raise SourceIntegrityError from exc
        if parts is None:
            raise StaleClaimError
        hashes = [p.source_sha256 for p in parts]
        refs = [
            SourceRef(
                part_id=p.part_id,
                photo_index=p.photo_index,
                source_sha256=p.source_sha256,
                coordinate_space="original",
            )
            for p in parts
        ]
        pages, manifests = await self._resume(claim, parts, ocr, deadline, db_deadline=db_deadline)
        observation_checkpoint_id: UUID | None = None

        async def save_page(page: OcrResponse) -> None:
            manifest = await self._save_view(
                image.id, page.photo_index, page.source_sha256, page.working_image, parts, deadline
            )
            owned.append(manifest)
            pages.append(page)
            manifests.append(manifest)
            checkpoint = OcrCheckpoint(
                ordered_source_sha256=hashes,
                profile_id=page.profile_id,
                runtime_fingerprint=page.runtime_fingerprint,
                pages=[p.model_dump() for p in pages],
                persisted_views=list(manifests),
            )
            if not await self.db.save_ocr_checkpoint(claim, checkpoint, deadline=db_deadline):
                raise StaleClaimError

        async def save_direct_views(envelope: InferenceEnvelope) -> None:
            for view in envelope.input_views:
                manifest = await self._save_view(
                    image.id, view.photo_index, view.source_sha256, view.working_image, parts, deadline
                )
                owned.append(manifest)
                manifests.append(manifest)

        async def save_observations(envelope: InferenceEnvelope) -> None:
            nonlocal observation_checkpoint_id
            if envelope.provenance.ordered_source_sha256 != hashes:
                raise SourceIntegrityError
            await save_direct_views(envelope)
            if await run_sync_before(deadline, self._bindings, image) != bindings:
                raise SourceIntegrityError
            provenance = envelope.provenance
            checkpoint = await run_sync_before(
                deadline,
                lambda: ObservationCheckpoint(
                    checkpoint_id=uuid4(),
                    producer_generation=claim.generation,
                    producer_attempt_number=claim.attempt_number,
                    eligible_generation=claim.generation,
                    source_parts=parts,
                    model_id=provenance.model_id,
                    execution_identity=CheckpointExecutionIdentity(
                        inference_identity=self.receipts.inference_identity,
                        semantic_policy_sha256=semantic_policy_sha256(snapshot),
                        model_id=provenance.model_id,
                        model_digest=provenance.model_digest,
                        prompt_version=PROMPT_VERSION,
                        extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
                        source_projection_version=envelope.source_projection_version,
                    ),
                    envelope=pack_inference_metadata(envelope),
                    persisted_views=manifests,
                ),
            )
            if not await self.db.save_observation_checkpoint(claim, checkpoint, deadline=db_deadline):
                raise StaleClaimError
            observation_checkpoint_id = checkpoint.checkpoint_id

        async def record_classification_event(state: Literal["started", "settled"]) -> None:
            if observation_checkpoint_id is None:
                raise StaleClaimError
            recorded = await self.db.record_unmanaged_classification_event(
                claim, observation_checkpoint_id, state=state, deadline=db_deadline,
            )
            if not recorded:
                raise StaleClaimError

        tags = await self.receipts.list_tags(deadline=db_deadline)
        expected_binding = (snapshot.get("managed_binding_sha256") or "") if managed_ocr(ocr) and ocr.shared_gpu else None
        for retry in range(2):
            try:
                if policy_hold_reason(snapshot, self.settings.ocr, self.receipts.inference_identity):
                    if retry:
                        # A real dispatch already happened in this whole attempt.
                        # Keep it counted; the next readiness cycle will persist the hold.
                        raise OcrRuntimeChangedError("ocr_runtime_changed")
                    raise PolicyHoldError
                envelope = await self.receipts.extract_receipt(
                    image.model_id,
                    [Path(p.image_path) for p in parts],
                    options=options,
                    deadline=deadline,
                    tags=tags,
                    checkpoint=save_page,
                    observation_checkpoint=save_observations,
                    unmanaged_classification_event=record_classification_event,
                    resume_pages=list(pages),
                    source_refs=refs,
                    ocr_settings=ocr,
                    max_output_tokens=limit,
                    expected_binding_sha256=expected_binding,
                    handoff_deadline=deadline if managed_ocr(ocr) else None,
                )
                break
            except OcrRuntimeChangedError:
                if not await self.db.clear_ocr_checkpoint(claim, deadline=db_deadline):
                    raise StaleClaimError from None
                pages.clear()
                manifests.clear()
                if retry:
                    raise
            except (HandoffWait, ReadinessHoldError) as exc:
                if retry:
                    raise OcrRuntimeChangedError("ocr_runtime_changed") from exc
                raise
        else:
            raise OcrRuntimeChangedError("ocr_runtime_changed")
        if envelope.provenance.ordered_source_sha256 != hashes:
            raise SourceIntegrityError
        if not manifests:
            # An incomplete Direct draft can still publish views with its partial outcome.
            await save_direct_views(envelope)
        # Check originals again before accepting any receipt; preserve bytes and paths.
        if await run_sync_before(deadline, self._bindings, image) != bindings:
            raise SourceIntegrityError
        return await run_sync_before(
            deadline,
            self._outcome,
            envelope,
            manifests,
            [
                OriginalAvailability(
                    part_id=p.part_id,
                    photo_index=p.photo_index,
                    available=True,
                    sha256_matches=True,
                )
                for p in parts
            ],
        )

    async def _save_view(
        self,
        image_id: UUID,
        index: int,
        digest: str,
        working: WorkingImage,
        parts: list[PersistedSourcePart],
        deadline: float,
    ) -> PersistedWorkingView:
        if index >= len(parts) or index < 0 or parts[index].source_sha256 != digest:
            raise SourceIntegrityError
        path = await self.images.save_working_image(image_id, working, deadline=deadline)
        return PersistedWorkingView(
            part_id=parts[index].part_id,
            photo_index=index,
            source_sha256=digest,
            artifact_path=str(path),
            metadata=WorkingImageMetadata.model_validate(working.model_dump()),
        )

    async def _resume(
        self,
        claim: AnalysisClaim,
        parts: list[PersistedSourcePart],
        ocr: OcrSettings,
        deadline: float,
        *,
        db_deadline: float,
    ) -> tuple[list[OcrResponse], list[PersistedWorkingView]]:
        checkpoint = await self.db.get_ocr_checkpoint(claim, deadline=db_deadline)
        if checkpoint is None:
            return [], []
        if checkpoint.ordered_source_sha256 != [p.source_sha256 for p in parts]:
            raise SourceIntegrityError
        try:
            capabilities = await self.receipts.ocr_capabilities(ocr)
            if ocr.enabled and not (capabilities.ready or capabilities.can_load):
                reason = capabilities.reason_code
                raise ReadinessHoldError(reason if reason in OCR_RESULT_CODES else "ocr_unavailable")
            if (
                not ocr.enabled
                or not capabilities.runtime_fingerprint
                or capabilities.profile_id != checkpoint.profile_id
                or capabilities.runtime_fingerprint != checkpoint.runtime_fingerprint
            ):
                raise ValueError("checkpoint_runtime_changed")
            if not 0 < len(checkpoint.pages) <= len(parts) or len(
                checkpoint.persisted_views
            ) != len(checkpoint.pages):
                raise ValueError("checkpoint_prefix_invalid")
            result: list[OcrResponse] = []
            seen: set[UUID] = set()
            for index, raw in enumerate(checkpoint.pages):
                page = OcrPageMetadata.model_validate(raw)
                compact_ocr_text({"text": page.text})
                view = checkpoint.persisted_views[index]
                if (
                    page.photo_index != index
                    or page.source_sha256 != parts[index].source_sha256
                    or page.profile_id != checkpoint.profile_id
                    or page.runtime_fingerprint != checkpoint.runtime_fingerprint
                    or view.metadata != page.working_image
                    or view.photo_index != index
                    or view.part_id != parts[index].part_id
                    or view.source_sha256 != parts[index].source_sha256
                    or view.metadata.view_id in seen
                ):
                    raise ValueError("checkpoint_identity_invalid")
                seen.add(view.metadata.view_id)
                png = await run_sync_before(
                    deadline,
                    self.images.read_working_image,
                    claim.image.id,
                    view.artifact_path,
                    view.metadata,
                )
                payload = page.model_dump()
                payload["working_image"]["png_base64"] = base64.b64encode(png).decode()
                result.append(OcrResponse.model_validate(payload))
            return result, list(checkpoint.persisted_views)
        except (TimeoutError, ReadinessHoldError):
            raise
        except (ValueError, OSError):
            if not await self.db.clear_ocr_checkpoint(claim, deadline=db_deadline):
                raise StaleClaimError from None
            return [], []

    @staticmethod
    def _outcome(
        envelope: InferenceEnvelope,
        views: list[PersistedWorkingView],
        originals: list[OriginalAvailability],
    ) -> AttemptOutcome:
        document = envelope.draft
        admission = envelope
        if (
            document is None
            and envelope.receipt is not None
            and envelope.source_projection_version == "1"
            and envelope.classification_enrichment.state == "not_recorded"
        ):
            # Historical receipt-only adapters predate nullable observation provenance.
            document = ExtractionDraft.model_validate(envelope.receipt.model_dump())
            admission = envelope.model_copy(update={"draft": document})
        if document is None:
            code = envelope.findings[-1].code if envelope.findings else "analysis_incomplete"
            if code not in OCR_RESULT_CODES and code not in {
                "ocr_unavailable",
                "ocr_page_failed",
                "ocr_disabled",
                "formatter_model_unavailable",
                "vision_model_unavailable",
                "vision_fallback_unavailable",
            }:
                code = "analysis_incomplete"
            return AttemptOutcome(
                kind="retry", result_code=code, envelope=envelope, persisted_views=views
            )
        admitted: Receipt | None
        try:
            admitted = admit_observations(admission)
        except ValueError:
            admitted = None
        review = evaluate_receipt(
            admitted if admitted is not None else document,
            envelope.amount_policy,
            envelope.field_sources,
            envelope.overlap_decisions,
            originals,
            retained_findings=envelope.findings,
        )
        if admitted is not None and review.can_finalize:
            return AttemptOutcome(
                kind="complete", envelope=envelope, review=review, persisted_views=views
            )
        envelope.draft = document
        return AttemptOutcome(
            kind="partial",
            envelope=envelope,
            review=review,
            result_code="needs_review",
            persisted_views=views,
        )

    async def get_draft(self, image_id: UUID, *, scope: AccessScope) -> DraftView | None:
        return await self.db.get_draft(image_id, scope=scope)

    async def patch_draft(
        self, image_id: UUID, patch: DraftPatch, *, scope: AccessScope
    ) -> DraftView | None:
        return await self.db.patch_draft(image_id, patch, scope=scope)

    async def finalize_draft(
        self, image_id: UUID, request: DraftFinalize, *, scope: AccessScope
    ) -> ReceiptRow | None:
        draft = await self.db.get_draft(image_id, scope=scope)
        if draft is None:
            return None
        availability = []
        if draft.finalized_receipt_id is None:
            parts = await self.db.get_source_parts(image_id, scope=scope)

            def inspect() -> list[OriginalAvailability]:
                result = []
                for part in parts:
                    try:
                        digest = hashlib.sha256(Path(part.image_path).read_bytes()).hexdigest()
                        result.append(
                            OriginalAvailability(
                                part_id=part.part_id,
                                photo_index=part.photo_index,
                                available=True,
                                sha256_matches=digest == part.source_sha256,
                            )
                        )
                    except OSError:
                        result.append(
                            OriginalAvailability(
                                part_id=part.part_id,
                                photo_index=part.photo_index,
                                available=False,
                                sha256_matches=False,
                            )
                        )
                return result

            availability = await run_sync_before(monotonic() + 30, inspect)
        return await self.db.finalize_draft(image_id, request, scope=scope, originals=availability)
