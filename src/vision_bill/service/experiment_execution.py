"""Benchmark execution through the same public receipt-service paths as uploads.

Forced managed OCR requires production OCR participation in the shared handoff.
Unmanaged CPU/replay extensions may be selected independently of production OCR.
All pipeline policy overrides are local to the experiment.
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, Any

from ..config import OcrSettings
from ..helper.deadline import run_sync_before
from ..model.experiment import ExperimentOptions, ExtractionPipeline
from ..model.extraction import (
    InferenceEnvelope,
    PersistedInferenceEnvelope,
    pack_inference_metadata,
)
from ..model.handoff import HandoffStatus, HandoffWait
from ..model.ocr_errors import OcrWorkerFailure, raise_for_worker_reason
from ..model.receipt import Receipt
from .activation_admission import inference_admission
from .extraction_pipeline import ExtractionPipeline as ReceiptExtractionPipeline
from .gpu_handoff import admission_deadline
from .observation_admission import admit_observations

if TYPE_CHECKING:
    from .receipt_service import ReceiptService

# Benchmark manifests bind their provider endpoint separately from upload snapshots.
BENCHMARK_EXECUTION_POLICY_VERSION = 3


@dataclass
class ExperimentResult:
    receipt: Receipt
    attempts: int
    elapsed_ms: float
    metadata: dict[str, Any]


class ExperimentFailure(ValueError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


class PostObservationFailure(ExperimentFailure):
    """Keep admitted evidence as a failed case without repeating inference."""


def ocr_fingerprint(provider: Any) -> str | None:
    """Identify the published OCR extension's client without exposing its URL.

    The full pipeline identity separately binds the extension's advertised
    profile and runtime fingerprint; this hash binds local client settings.
    """
    if provider is None:
        return None
    identity = {
        "provider": type(provider).__name__,
        "endpoint": getattr(provider, "service_url", None),
        "timeout": getattr(provider, "timeout", None),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def effective_policy(
    service: "ReceiptService",
    options: ExperimentOptions,
) -> tuple[Any, OcrSettings, ExperimentOptions]:
    """Keep upload policy, explicit direct vision and strict OCR distinct."""
    settings = service._ocr_settings.model_copy(update={
        "allow_vision_fallback": False, "vision_fallback_model_id": None,
    })
    options = options.model_copy(update={"allow_ocr_fallback": False})
    ocr = service._ocr_provider
    if options.pipeline == ExtractionPipeline.OCR:
        from .activation_admission import activation_enabled
        if activation_enabled() and not settings.enabled:
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="dedicated_ocr_disabled",
            ))
        settings = settings.model_copy(
            update={
                "enabled": True,
                "required": True,
                "allow_vision_fallback": False,
            }
        )
        if ocr is None:
            from ..provider.factory import get_ocr_provider

            ocr = get_ocr_provider(settings)
    # Direct vision still coordinates with the configured OCR sidecar; it simply
    # skips its load phase. This also protects direct benchmark jobs.
    return ocr, settings, options


async def execute_submission(
    service: "ReceiptService",
    model_id: str,
    paths: list[Path],
    options: ExperimentOptions,
    tags: list[str],
    *,
    deadline: float | None = None,
) -> ExperimentResult:
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("Receipt image or additional photo is unavailable")
    started = perf_counter()
    probe = copy(service)
    probe._ocr_provider, probe._ocr_settings, effective_options = effective_policy(service, options)
    provider = service._provider.for_experiment(effective_options)
    probe._provider = provider
    probe._experiment_tags = list(tags)
    provider_deadline = monotonic() + provider.analysis_timeout_seconds
    work_deadline = provider_deadline if deadline is None else min(deadline, provider_deadline)
    metadata: dict[str, Any] = {
        "requested_pipeline": options.pipeline.value,
        "requested_options": options.model_dump(mode="json"),
        "fallback_used": False,
        "image_count": len(paths),
        "provider": type(provider).__name__,
        "options": provider.experiment_options.model_dump(mode="json")
        if provider.experiment_options
        else options.model_dump(mode="json"),
    }
    observations_recorded = False

    async def remember_observations(envelope: InferenceEnvelope) -> None:
        nonlocal observations_recorded
        metadata.update(await run_sync_before(
            work_deadline, lambda: observation_metadata(pack_inference_metadata(envelope)),
        ))
        observations_recorded = True

    try:
        async with asyncio.timeout(max(0, work_deadline - monotonic())):
            envelope = await probe.extract_receipt(
                model_id,
                paths,
                options=effective_options,
                deadline=work_deadline,
                tags=tags,
                observation_checkpoint=remember_observations,
            )
            persisted = await run_sync_before(work_deadline, pack_inference_metadata, envelope)
            metadata.update(await run_sync_before(work_deadline, observation_metadata, persisted))
            if persisted.draft is None and persisted.receipt is None:
                code = persisted.findings[-1].code if persisted.findings else "receipt_incomplete"
                raise ValueError(code)
            receipt = await run_sync_before(work_deadline, admit_observations, persisted)
    except Exception as exc:
        if isinstance(exc, HandoffWait) and not observations_recorded:
            raise
        metadata.setdefault("calls", provider.experiment_trace or [])
        metadata["elapsed_ms"] = (perf_counter() - started) * 1000
        code = str(exc) if isinstance(exc, (OcrWorkerFailure, ValueError)) else type(exc).__name__
        failure = PostObservationFailure if observations_recorded else ExperimentFailure
        raise failure(code, metadata) from exc
    metadata["llm_ms"] = sum(call.get("elapsed_ms", 0) for call in metadata["calls"])
    elapsed = (perf_counter() - started) * 1000
    metadata["elapsed_ms"] = elapsed
    return ExperimentResult(receipt, len(metadata["calls"]) or 1, elapsed, metadata)


def observation_metadata(persisted: PersistedInferenceEnvelope) -> dict[str, Any]:
    evidence = persisted.model_dump(mode="json", exclude={"provenance", "receipt", "amount_policy"})
    return {
        **persisted.provenance.model_dump(mode="json"),
        **evidence,
        "calls": evidence["telemetry"]["calls"],
    }


async def pipeline_identity(
    service: "ReceiptService", options: ExperimentOptions, *, qualify: bool = False,
    managed_model_ids: Sequence[str] = (),
    expected_binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze effective local execution contracts without receipt/reference observations."""
    import inspect

    from ..helper.document_image import NORMALIZER_VERSION, document_rgb
    from ..model.extraction import (
        EXTRACTION_SCHEMA_VERSION,
        SOURCE_PROJECTION_VERSION,
        ReceiptInput,
    )
    from ..provider.llm.experiment import source_extraction_prompt

    ocr, settings, effective_options = effective_policy(service, options)
    provider = service._provider.for_experiment(effective_options)
    handoff = ReceiptExtractionPipeline(provider, ocr, settings).coordinator(
        direct=options.pipeline == ExtractionPipeline.DIRECT,
    )
    capabilities = None
    binding_sha256 = None
    uses_ocr = ocr is not None and options.pipeline != ExtractionPipeline.DIRECT
    with inference_admission(provider, handoff=handoff, uses_ocr=uses_ocr):
        if handoff is not None and handoff.shared_gpu:
            binding = handoff.require_binding(expected_binding_sha256)
            if qualify and any(model not in binding.model_ids for model in managed_model_ids):
                raise HandoffWait(HandoffStatus(
                    state="configuration_required", reason_code="model_handoff_model_not_adopted",
                ))
            binding_sha256 = handoff.binding_sha256(binding)
        elif expected_binding_sha256 is not None:
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="execution_policy_changed",
            ))
        if ocr is not None and uses_ocr:
            if qualify and handoff is not None:
                deadline = monotonic() + provider.analysis_timeout_seconds
                async with handoff.session(
                    deadline, model_ids=managed_model_ids, expected_binding_sha256=binding_sha256,
                    lock_deadline=admission_deadline(deadline),
                ) as owner:
                    await owner.ocr_phase()
                    qualified = await ocr.get_capabilities()
                    if not qualified.ready:
                        raise_for_worker_reason(qualified.reason_code)
                    if not qualified.ready or not qualified.runtime_fingerprint:
                        raise ValueError("ocr_qualification_required")
                    capabilities = qualified.model_dump(mode="json")
            else:
                capabilities = (await ocr.get_capabilities()).model_dump(mode="json")
            if getattr(ocr, "managed_lifecycle", False):
                # Residency and availability are observations, not frozen identity.
                capabilities = {key: capabilities[key] for key in (
                    "profile_id", "runtime_fingerprint", "ocr_max_new_tokens"
                )}
    return {
        "execution_policy_version": BENCHMARK_EXECUTION_POLICY_VERSION,
        "managed_binding_sha256": binding_sha256,
        "version": 1,
        "normalizer_version": NORMALIZER_VERSION,
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "source_projection_version": SOURCE_PROJECTION_VERSION,
        "source_prompt_sha256": hashlib.sha256(
            inspect.getsource(source_extraction_prompt).encode()
        ).hexdigest(),
        "input_schema_sha256": hashlib.sha256(
            json.dumps(ReceiptInput.model_json_schema(), sort_keys=True).encode()
        ).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(
            (
                inspect.getsource(document_rgb)
                + inspect.getsource(provider._prepare_receipt_input)
                + inspect.getsource(provider._authorized_source_refs)
            ).encode()
        ).hexdigest(),
        "ocr_client": ocr_fingerprint(ocr),
        "ocr_runtime": capabilities,
        "ocr_settings": settings.model_dump(
            mode="json", exclude={"service_url", "shared_gpu"} if settings.shared_gpu else {"service_url"},
        ),
        "requested_options": options.model_dump(mode="json"),
        "options": (provider.experiment_options or effective_options).model_dump(mode="json"),
        "formatter_output_tokens": provider.max_output_tokens,
        "context_window_tokens": getattr(provider, "context_window_tokens", None),
        "analysis_timeout_seconds": provider.analysis_timeout_seconds,
    }
