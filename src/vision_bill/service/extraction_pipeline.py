"""Deadline-bound, ordered all-photo OCR and formatting without persistence."""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from time import monotonic, perf_counter
from typing import Literal

import httpx

from ..config import OcrSettings
from ..helper.deadline import check_deadline, run_sync_before
from ..model.experiment import ExperimentOptions
from ..model.extraction import (
    InferenceEnvelope,
    InferenceTelemetry,
    PipelineProvenance,
    ReceiptInput,
)
from ..model.handoff import HandoffStatus, HandoffWait
from ..model.ocr import OcrPageMetadata, OcrResponse, SourceRef
from ..model.ocr_errors import OcrWorkerFailure
from ..model.review import ReviewFinding
from ..provider.llm.base import LLMProvider, ModelInfo
from ..provider.ocr.base import OcrProvider, compact_ocr_text
from .activation_admission import inference_admission
from .gpu_handoff import ModelHandoff, admission_deadline, current_llm_owner, current_owner
from .observation_admission import admit_observations

logger = logging.getLogger(__name__)

UnmanagedClassificationEvent = Callable[[Literal["started", "settled"]], Awaitable[None]]


def select_vision_model(models: Sequence[ModelInfo], requested_id: str) -> ModelInfo | None:
    """Select the requested model only when discovery affirms vision support."""
    available = {m.id: m for m in models if m.capabilities and
                 any(c in m.capabilities for c in ("vision", "multimodal"))}
    return available.get(requested_id)


class ReadinessHoldError(ValueError):
    """The selected model disappeared before new inference was dispatched."""


class OriginalSourceChangedError(ValueError):
    """The current originals no longer match their supplied persisted binding."""


class OcrRuntimeChangedError(ValueError):
    """A cached/live worker identity changed; discard the cache before a bounded restart."""


class ExtractionPipeline:
    def __init__(
        self,
        provider: LLMProvider,
        ocr_provider: OcrProvider | None = None,
        ocr_settings: OcrSettings | None = None,
    ):
        self.provider = provider
        self.ocr = ocr_provider
        self.settings = ocr_settings or OcrSettings(enabled=ocr_provider is not None)

    def coordinator(self, *, direct: bool = False) -> ModelHandoff | None:
        configured = getattr(self.provider, "handoff", None)
        if direct and not self.settings.shared_gpu:
            return None
        if self.ocr is None or not getattr(self.ocr, "managed_lifecycle", False):
            if isinstance(configured, ModelHandoff) and self.ocr is not None:
                raise HandoffWait(HandoffStatus(
                    state="configuration_required", reason_code="execution_policy_changed",
                ))
            return configured if isinstance(configured, ModelHandoff) else None
        handoff = ModelHandoff(
            self.settings.coordination_path, self.provider, self.ocr, shared_gpu=self.settings.shared_gpu,
        )
        if self.settings.shared_gpu and not self.provider.supports_model_release:
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="model_release_unsupported",
            ))
        if isinstance(configured, ModelHandoff) and not handoff.matches(configured):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="execution_policy_changed",
            ))
        return handoff

    async def extract(
        self,
        model_id: str,
        paths: Sequence[Path],
        *,
        options: ExperimentOptions,
        deadline: float,
        tags: Sequence[str],
        checkpoint: Callable[[OcrResponse], Awaitable[None]] | None = None,
        observation_checkpoint: Callable[[InferenceEnvelope], Awaitable[None]] | None = None,
        unmanaged_classification_event: UnmanagedClassificationEvent | None = None,
        resume_pages: Sequence[OcrResponse] = (),
        source_refs: Sequence[SourceRef] = (),
        max_output_tokens: int | None = None,
        expected_binding_sha256: str | None = None,
        handoff_deadline: float | None = None,
    ) -> InferenceEnvelope:
        owner = current_owner()
        if owner is not None:
            deadline = min(deadline, owner.deadline)
        started = perf_counter()
        direct = options.pipeline.value == "direct_vision"
        handoff = self.coordinator(direct=direct)
        with inference_admission(self.provider, handoff=handoff, uses_ocr=self.settings.enabled and not direct):
            async with asyncio.timeout(max(0, deadline - monotonic())), (
                handoff.session(
                    deadline, model_id=model_id,
                    expected_binding_sha256=expected_binding_sha256,
                    lock_deadline=admission_deadline(min(deadline, handoff_deadline)
                        if handoff_deadline is not None else deadline),
                ) if handoff is not None else nullcontext()
            ):
                result = await self._extract(
                    model_id,
                    tuple(paths),
                    options=options.model_copy(deep=True),
                    deadline=deadline,
                    tags=tuple(tags),
                    checkpoint=checkpoint,
                    observation_checkpoint=observation_checkpoint,
                    unmanaged_classification_event=unmanaged_classification_event,
                    resume_pages=tuple(p.model_copy(deep=True) for p in resume_pages),
                    source_refs=tuple(r.model_copy(deep=True) for r in source_refs),
                    max_output_tokens=max_output_tokens,
                )
        check_deadline(deadline)
        result.telemetry.elapsed_ms = (perf_counter() - started) * 1000
        return result

    async def _extract(
        self,
        model_id: str,
        paths: Sequence[Path],
        *,
        options: ExperimentOptions,
        deadline: float,
        tags: Sequence[str],
        checkpoint: Callable[[OcrResponse], Awaitable[None]] | None,
        observation_checkpoint: Callable[[InferenceEnvelope], Awaitable[None]] | None,
        unmanaged_classification_event: UnmanagedClassificationEvent | None,
        resume_pages: Sequence[OcrResponse],
        source_refs: Sequence[SourceRef],
        max_output_tokens: int | None,
    ) -> InferenceEnvelope:
        if resume_pages and (
            not self.settings.enabled or options.pipeline.value == "direct_vision"
        ):
            raise ValueError("ocr_resume_disabled")
        if not paths:
            raise ValueError("source_images_required")
        started = perf_counter()
        hashes = await run_sync_before(
            deadline, lambda: [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        )
        # Optional persisted bindings describe the complete ordered originals.
        refs = list(source_refs)
        if refs and (
            len(refs) != len(paths)
            or any(
                ref.photo_index != i
                or ref.source_sha256 != hashes[i]
                or ref.coordinate_space != "original"
                or ref.view_id is not None
                or ref.block_id is not None
                or ref.bbox is not None
                or ref.text_span is not None
                for i, ref in enumerate(refs)
            )
            or len({r.part_id for r in refs}) != len(refs)
        ):
            raise OriginalSourceChangedError("source_reference_mismatch")
        provider = self.provider.for_experiment(options)
        if max_output_tokens is not None:
            if max_output_tokens <= 0:
                raise ValueError("invalid_output_limit")
            provider.max_output_tokens = max_output_tokens
        result = InferenceEnvelope(
            provenance=PipelineProvenance(
                requested_pipeline=options.pipeline.value,
                used_pipeline="none",
                source_image_count=len(paths),
                ordered_source_sha256=hashes,
                model_id=model_id,
                requested_model_id=model_id,
            ),
            telemetry=InferenceTelemetry(
                elapsed_ms=0, max_output_tokens=provider.max_output_tokens
            ),
            field_sources={"input": refs},
        )
        async with asyncio.timeout(max(0, deadline - monotonic())):
            explicit_direct = options.pipeline.value == "direct_vision"
            use_ocr = not explicit_direct and self.settings.enabled
            if options.pipeline.value == "ocr" and not self.settings.enabled:
                return self._failed(result, "ocr_disabled", started)
            reason: str | None = None
            if use_ocr:
                owner = current_owner()
                model = None
                if owner is not None:
                    models = await provider.get_available_models(capability="text")
                    model = next((m for m in models if m.id == model_id), None)
                    if model is None:
                        return self._failed(result, "formatter_model_unavailable", started)
                if self.ocr is None:
                    reason = "ocr_unavailable"
                else:
                    cached_all = len(resume_pages) == len(paths)
                    if owner is not None and not cached_all:
                        try:
                            await owner.ocr_phase()
                        except OcrWorkerFailure as exc:
                            result.provenance.fallback_reason = str(exc)
                            return self._failed(result, str(exc), started)
                    capabilities = await self.ocr.get_capabilities()
                    available = capabilities.ready or (cached_all and owner is not None and capabilities.can_load)
                    if not available or not capabilities.profile_id:
                        reason = capabilities.reason_code or "ocr_unavailable"
                    else:
                        result.provenance.profile_id = capabilities.profile_id
                        result.telemetry.ocr_max_new_tokens = capabilities.ocr_max_new_tokens
                        ocr_started = perf_counter()
                        runtime = capabilities.runtime_fingerprint
                        if resume_pages:
                            if not runtime or len(resume_pages) > len(paths):
                                raise OcrRuntimeChangedError("ocr_resume_identity_mismatch")
                            for index, page in enumerate(resume_pages):
                                compact_ocr_text({"text": page.text})
                                if (
                                    page.photo_index != index
                                    or page.source_sha256 != hashes[index]
                                    or page.profile_id != capabilities.profile_id
                                    or page.runtime_fingerprint != runtime
                                ):
                                    raise OcrRuntimeChangedError("ocr_resume_identity_mismatch")
                            result.ocr_pages = list(resume_pages)
                            result.provenance.runtime_fingerprint = runtime
                        for index in range(len(resume_pages), len(paths)):
                            path = paths[index]
                            try:
                                page = await self.ocr.extract_page(
                                    path,
                                    photo_index=index,
                                    source_sha256=hashes[index],
                                    deadline=deadline,
                                    expected_profile_id=capabilities.profile_id,
                                )
                                compact_ocr_text({"text": page.text})
                                if runtime and page.runtime_fingerprint != runtime:
                                    raise OcrRuntimeChangedError("ocr_runtime_changed")
                                if (
                                    page.photo_index != index
                                    or page.source_sha256 != hashes[index]
                                    or page.profile_id != capabilities.profile_id
                                ):
                                    raise ValueError("ocr_identity_mismatch")
                            except OcrRuntimeChangedError:
                                raise
                            except OcrWorkerFailure as exc:
                                reason = str(exc)
                                break
                            except (httpx.HTTPError, OSError, ValueError) as exc:
                                # Transport failures share the same whole-receipt policy as blank text.
                                if isinstance(exc, TimeoutError):
                                    raise
                                if str(exc) == "ocr_source_hash_mismatch":
                                    raise OriginalSourceChangedError(
                                        "ocr_source_hash_mismatch"
                                    ) from exc
                                reason = "ocr_page_failed"
                                break
                            runtime = page.runtime_fingerprint
                            result.ocr_pages.append(page)
                            result.provenance.runtime_fingerprint = page.runtime_fingerprint
                            if checkpoint is not None:
                                # Durability failure is not an OCR failure and must not trigger fallback.
                                await checkpoint(page)
                        result.telemetry.stage_ms["ocr"] = (perf_counter() - ocr_started) * 1000
                if reason is None:
                    owner = current_owner()
                    if owner is not None:
                        await owner.llm_phase(model_id)
                    if owner is None:
                        models = await provider.get_available_models(capability="text")
                        model = next((m for m in models if m.id == model_id), None)
                    if model is None:
                        return self._failed(result, "formatter_model_unavailable", started)
                    result.provenance.model_digest = model.digest
                    format_started = perf_counter()
                    logger.info("ocr_formatter state=start pages=%d budget_ms=%d",
                                len(result.ocr_pages), int(max(0, deadline - monotonic()) * 1000))
                    formatted = await provider.extract_observations(
                        model_id,
                        ReceiptInput(
                            mode="ocr",
                            image_paths=[str(p) for p in paths],
                            ocr_pages=[
                                OcrPageMetadata.model_validate(p.model_dump())
                                for p in result.ocr_pages
                            ],
                            source_refs=refs,
                        ),
                        tags=tags,
                        deadline=deadline,
                    )
                    logger.info("ocr_formatter state=finished elapsed_ms=%d",
                                int((perf_counter() - format_started) * 1000))
                    formatted.telemetry.context_window_tokens = model.context_window_tokens
                    return await self._finish(
                        provider, formatted, result, started, tags, deadline, observation_checkpoint,
                        unmanaged_classification_event,
                    )
                result.provenance.fallback_reason = reason
                return self._failed(result, reason, started)
            # Only discovered Vision models may receive originals/normalized image derivatives.
            models = await provider.get_available_models(capability="vision")
            model = select_vision_model(models, model_id)
            if model is None:
                if current_owner() is None:
                    raise ReadinessHoldError("vision_model_unavailable")
                return self._failed(result, "vision_model_unavailable", started)
            vision_id = model.id
            formatted = await provider.extract_observations(
                vision_id,
                ReceiptInput(
                    mode="direct_vision", image_paths=[str(p) for p in paths], source_refs=refs
                ),
                tags=tags,
                deadline=deadline,
            )
            if [view.source_sha256 for view in formatted.input_views] != hashes:
                raise OriginalSourceChangedError("vision_source_hash_mismatch")
            result.provenance.model_id = vision_id
            result.provenance.model_digest = model.digest
            formatted.telemetry.context_window_tokens = model.context_window_tokens
            return await self._finish(
                provider, formatted, result, started, tags, deadline, observation_checkpoint,
                unmanaged_classification_event,
            )

    @staticmethod
    def _failed(result: InferenceEnvelope, code: str, started: float) -> InferenceEnvelope:
        result.findings.append(
            ReviewFinding(
                id=code,
                code=code,
                severity="integrity",
                field_path="input",
                source_refs=result.field_sources["input"],
            )
        )
        result.telemetry.elapsed_ms = (perf_counter() - started) * 1000
        return result

    @staticmethod
    async def _finish(
        provider: LLMProvider,
        formatted: InferenceEnvelope,
        original: InferenceEnvelope,
        started: float,
        tags: Sequence[str],
        deadline: float,
        observation_checkpoint: Callable[[InferenceEnvelope], Awaitable[None]] | None,
        unmanaged_classification_event: UnmanagedClassificationEvent | None,
    ) -> InferenceEnvelope:
        original.provenance.used_pipeline = formatted.provenance.used_pipeline
        formatted.provenance = original.provenance
        formatted.ocr_pages = original.ocr_pages
        formatted.findings = original.findings + formatted.findings
        formatted.telemetry.stage_ms.update(original.telemetry.stage_ms)
        formatted.telemetry.ocr_max_new_tokens = original.telemetry.ocr_max_new_tokens
        formatted.telemetry.elapsed_ms = (perf_counter() - started) * 1000
        if formatted.receipt is None:
            return formatted
        await run_sync_before(deadline, admit_observations, formatted)
        if observation_checkpoint is not None:
            await observation_checkpoint(formatted)
        event = None
        if formatted.classification_enrichment.state == "pending" and current_llm_owner() is None:
            event = unmanaged_classification_event
        if event is not None:
            await event("started")
        enriched = await provider.enrich_observations(
            formatted.provenance.model_id, formatted, tags, work_deadline=deadline,
        )
        if event is not None:
            # A normal unmanaged return proves a response or no dispatch.
            await event("settled")
        return enriched
