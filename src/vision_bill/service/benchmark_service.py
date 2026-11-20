"""Asynchronous, restart-safe local benchmark execution."""
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID

import httpx
from ollama import ResponseError
from openai import APIConnectionError, APIStatusError

from ..config import OcrSettings, WorkerSettings
from ..model.benchmark import (
    BenchmarkCreate,
    BenchmarkRun,
    BenchmarkStatus,
    GoldSnapshot,
    ReferencePreview,
    require_selected_models,
)
from ..model.experiment import ExperimentOptions, ExtractionPipeline
from ..model.handoff import HandoffStatus, HandoffWait
from ..model.ocr_errors import OCR_RESULT_CODES, OcrWorkerFailure, raise_for_worker_reason
from ..model.receipt import LineItem, Receipt, TaxLine
from ..model.workflow import AccessScope
from ..provider.db.benchmark_db import BenchmarkDB
from ..provider.db.deadline import connection
from ..provider.llm.base import LLMProvider
from ..provider.llm.errors import UnreadableReceiptError
from ..provider.llm.experiment import endpoint_fingerprint
from .activation_admission import inference_admission
from .benchmark_scoring import PROMPT_VERSION, dataset_fingerprint, score_receipts
from .experiment_execution import (
    ExperimentFailure,
    ExperimentResult,
    PostObservationFailure,
    effective_policy,
    image_digest,
    ocr_fingerprint,
    pipeline_identity,
)
from .experiment_scoring import DETAIL_SCORING_VERSION, detailed_score, detailed_score_v3
from .extraction_pipeline import OcrRuntimeChangedError, OriginalSourceChangedError
from .gpu_handoff import ModelHandoff, admission_deadline, handoff_resource_identity
from .receipt_service import ReceiptService

logger = logging.getLogger(__name__)


class _ReadinessWait(ValueError):
    """Unavailable dependency before inference, with no attempt to charge."""


class _StaleClaim(Exception):
    """The reservation no longer authorizes inference or persistence."""


class BenchmarkService:
    """Owns benchmark task execution; normal receipt/image records are read-only."""

    def __init__(self, provider: LLMProvider, receipt_service: ReceiptService,
                 *, worker: WorkerSettings | None = None):
        self._provider = provider
        self._receipts = receipt_service
        self._settings = worker or WorkerSettings()
        self._db = BenchmarkDB(receipt_service.pool, receipt_service._db, worker=self._settings)
        self._worker: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._worker is None:
            self._stopping.clear()
            self._worker = asyncio.create_task(self._work(), name="benchmark-worker")

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def create_run(self, request: BenchmarkCreate, *, scope: AccessScope) -> BenchmarkRun:
        selected_models = require_selected_models(getattr(request, "model_ids", None))
        use_ocr = request.experiment.pipeline == ExtractionPipeline.OCR or (
            request.experiment.pipeline == ExtractionPipeline.UPLOAD and self._receipts._ocr_settings.enabled
        )
        models = await self._provider.get_available_models(capability="text" if use_ocr else "vision")
        available = {m.id: m for m in models}
        unknown = set(selected_models) - set(available)
        if unknown:
            raise ValueError(f"Selected model(s) unavailable: {', '.join(sorted(unknown))}")
        selected_ids = await self._select_receipts(request, scope=scope)
        if not selected_ids:
            raise ValueError("No current qualified references match the selection")
        if request.experiment.think is not None and not self._provider.supports_thinking_control:
            raise ValueError("Thinking override is supported only by the Ollama provider")
        if request.experiment.pipeline == ExtractionPipeline.OCR and not self.capabilities()["ocr_available"]:
            raise ValueError("OCR extension is unavailable; managed OCR requires production OCR enabled with model lifecycle management")
        tags = await self._receipts.list_tags()
        inputs = [await self._capture_input(receipt_id, scope=scope) for receipt_id in selected_ids]
        expected = [Receipt.model_validate(item["expected"]) for item in inputs]
        probe = self._provider.for_experiment(request.experiment)
        options = probe.experiment_options or request.experiment
        counts = {len(item["images"]) for item in inputs}
        for item in inputs:
            item["pipeline"] = options.pipeline.value
        self._managed_coordinator(
            effective_policy(self._receipts, options)[1], direct=options.pipeline == ExtractionPipeline.DIRECT,
        )
        manifest = {"identity": await pipeline_identity(self._receipts, options, qualify=True, managed_model_ids=selected_models), "options": options.model_dump(mode="json"), "tags": tags, "inputs": inputs,
                    "provider": type(self._provider).__name__,
                    "endpoint_fingerprint": endpoint_fingerprint(self._provider),
                    "ocr_fingerprint": ocr_fingerprint(getattr(self._receipts, "_ocr_provider", None)),
                    "prompt_hashes": {str(count): sha256(probe.build_prompt(tags, image_count=count).encode()).hexdigest() for count in counts},
                    "validation_schema_sha256": sha256(json.dumps(Receipt.model_json_schema(), sort_keys=True).encode()).hexdigest(),
                    "thinking_default": "server-controlled" if options.think is None else "explicit"}
        return await self._db.create_run(
            model_ids=selected_models, receipt_ids=selected_ids, fingerprint=dataset_fingerprint(expected),
            model_digests={model_id: available[model_id].digest for model_id in selected_models}, timeout=request.request_timeout_seconds,
            policy=request.council_policy, absolute=request.council_absolute_threshold,
            relative=request.council_relative_threshold, apply_flags=request.apply_council_flags,
            prompt_version=PROMPT_VERSION, scoring_version=DETAIL_SCORING_VERSION, manifest=manifest, owner_id=scope.user_id, scope=scope,
        )

    async def get_status(self, run_id: UUID, *, scope: AccessScope) -> BenchmarkStatus | None:
        run = await self._db.run(run_id, scope=scope)
        if not run:
            return None
        counts = await self._db.task_counts(run_id)
        return BenchmarkStatus(run=run, summaries=await self._db.summaries(run_id), **counts)

    async def list_runs(self, *, scope: AccessScope) -> list[BenchmarkRun]:
        return await self._db.runs(scope=scope)

    async def reevaluate(self, run_id: UUID, receipt_id: UUID, model_id: str, *, scope: AccessScope) -> dict[str, Any]:
        """Transient inspection: extraction and field diff are returned, never stored."""
        run = await self._db.run(run_id, scope=scope)
        if not run or receipt_id not in run.receipt_ids or model_id not in run.model_ids:
            raise ValueError("Receipt/model is not part of this benchmark run")
        manifest = await self._db.manifest(run_id)
        if manifest:
            available = {m.id: m for m in await self._provider.get_available_models(capability="all")}
            current = available.get(model_id)
            digest = (await self._run_digests(run)).get(model_id)
            if current is None or (digest and digest != current.digest):
                raise ValueError("Selected model is unavailable or its digest changed; create a new run")
            budget = await self._timeout(run)
            with inference_admission(self._provider):
                async with asyncio.timeout(budget):
                    expected, result = await self._run_experiment(
                        manifest, receipt_id, model_id, managed_model_ids=run.model_ids,
                        inference_deadline=monotonic() + budget,
                    )
            detail = self._score(manifest, receipt_id, expected, result.receipt)
            item = next(item for item in manifest["inputs"] if item["receipt_id"] == str(receipt_id))
            return {"receipt_id": receipt_id, "model_id": model_id, **detail,
                    "expected": item["expected"] if item.get("reference") else expected.model_dump(mode="json"),
                    "actual": result.receipt.model_dump(mode="json"),
                    "metadata": result.metadata}
        raise ValueError("Historical benchmark has no execution manifest; create a new run")

    async def _select_receipts(self, request: BenchmarkCreate, *, scope: AccessScope) -> list[UUID]:
        if request.receipt_ids is not None:
            result = []
            for receipt_id in request.receipt_ids:
                row = await self._receipts.get_receipt_by_id(receipt_id, user_id=scope.user_id, can_see_all=scope.can_see_all)
                if row is None or not row.verified:
                    raise ValueError("Explicit receipt is unavailable or not verified")
                result.append(receipt_id)
            return result
        candidates = await self._db.reference_candidates(
            scope=scope, category=request.category, max_source_confidence=request.max_source_confidence,
        )
        selected = []
        for receipt_id in candidates:
            try:
                await self._db.current_reference(receipt_id, scope=scope)
            except ValueError:
                continue  # Filtered selection excludes stale/ineligible references.
            selected.append(receipt_id)
            if request.limit is not None and len(selected) >= request.limit:
                break
        return selected

    async def _expected_receipt_and_path(self, receipt_id: UUID, *, deadline: float | None = None) -> tuple[Receipt, str | None]:
        details = (await self._receipts.get_receipt_with_details(receipt_id, deadline=deadline)
                   if deadline is not None else await self._receipts.get_receipt_with_details(receipt_id))
        if details is None or not details.receipt.verified:
            raise ValueError("Benchmark data must be a verified receipt")
        r = details.receipt
        return Receipt.model_validate({"confidence": r.confidence, "merchant_name": r.merchant_name, "merchant_address": r.merchant_address, "receipt_number": r.receipt_number, "date": r.date, "time": r.time, "currency": r.currency, "category": r.category, "line_items": [LineItem.model_validate(x.model_dump()) for x in details.line_items], "taxes": [TaxLine.model_validate(x.model_dump()) for x in details.taxes], "subtotal": r.subtotal, "discount_total": r.discount_total, "tax_total": r.tax_total, "tip": r.tip, "total": r.total, "payment_method": r.payment_method}), details.image_path

    async def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                with inference_admission(self._provider):
                    task = await self._db.lease_task()
                    if task is not None:
                        await self._execute(dict(task))
                if task is None:
                    await asyncio.sleep(1)
            except HandoffWait:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - don't log private provider exception content
                logger.warning("Benchmark worker iteration unavailable")
                await asyncio.sleep(5)

    async def _execute(self, task: dict[str, Any]) -> None:
        started = asyncio.get_running_loop().time()
        deadline = started + self._settings.attempt_timeout_seconds
        # This outer budget also covers loading the run and its saved timeout.
        try:
            async with asyncio.timeout_at(deadline):
                run = await self._db.run(task["run_id"], deadline=deadline)
                if run is None:
                    return
                deadline = min(deadline, started + await self._timeout(run, deadline=deadline))
                async with asyncio.timeout_at(deadline):
                    manifest = await self._db.manifest(task["run_id"], deadline=deadline)
                    await self._execute_reserved(task, run, manifest, started, deadline)
        except TimeoutError:
            # Persistence/connection cleanup cannot extend the total budget.
            # Uncommitted work remains fenced for lease recovery, never a late finish.
            logger.warning("Benchmark attempt deadline reached")

    async def _execute_reserved(self, task: dict[str, Any], run: BenchmarkRun,
                                manifest: dict[str, Any] | None, started: float, deadline: float) -> None:
        if task["inference_attempt_count"] >= self._settings.max_attempts:
            # Expired final attempts terminalize independently of model readiness.
            await self._finish_failure(task, manifest, ValueError("inference_attempts_exhausted"), started, deadline)
            return
        reserve = min(5.0, (deadline - started) * 0.05)
        try:
            async with asyncio.timeout_at(deadline - reserve):
                if not manifest:
                    raise ValueError("Historical benchmark has no execution manifest; create a new run")
                try:
                    models = {m.id: m for m in await self._provider.get_available_models(capability="all")}
                except Exception as exc:
                    raise _ReadinessWait("model_discovery_unavailable") from exc
                model = models.get(task["model_id"])
                expected_digest = (await self._run_digests(run, deadline=deadline)).get(task["model_id"])
                if model is None or (expected_digest and expected_digest != model.digest):
                    raise _ReadinessWait("model_unavailable_or_changed")
                inference_deadline = monotonic() + max(0, deadline - reserve - asyncio.get_running_loop().time())
                await self._execute_experiment(
                    task, manifest, deadline, inference_deadline=inference_deadline,
                    managed_model_ids=run.model_ids,
                )
        except _StaleClaim:
            return
        except Exception as exc:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("benchmark_attempt_deadline") from exc
            before_inference = not task["inference_started"]
            retryable = self._retryable(exc)
            if before_inference and isinstance(exc, HandoffWait):
                if await self._db.hold_before_inference(task, str(exc), deadline=deadline):
                    logger.debug("Benchmark held before inference (%s)", exc.status.reason_code)
            elif before_inference and (isinstance(exc, _ReadinessWait) or retryable):
                await self._db.retry_task(task, waiting=True, deadline=deadline)
            elif retryable and task["inference_attempt_count"] < self._settings.max_attempts:
                await self._db.retry_task(task, deadline=deadline)
            else:
                await self._finish_failure(task, manifest, exc, started, deadline)

    async def _begin(self, task: dict[str, Any], deadline: float) -> None:
        updated = await self._db.begin_attempt(task, deadline=deadline)
        if updated is None:
            raise _StaleClaim
        task.update(updated)

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, PostObservationFailure):
            return False
        # ExperimentFailure deliberately preserves its cause. A wrapped ValueError
        # is not automatically permanent or retryable. OCR page failures include
        # both transport and blank/invalid pages; conservatively bound them to three.
        if isinstance(exc, ExperimentFailure) and exc.__cause__ is not None:
            if isinstance(exc.__cause__, (FileNotFoundError, OriginalSourceChangedError, UnreadableReceiptError)):
                return False
            if BenchmarkService._retryable(exc.__cause__):
                return True
        if isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError, APIConnectionError, OcrRuntimeChangedError, HandoffWait)):
            return True
        if isinstance(exc, (APIStatusError, ResponseError)):
            return exc.status_code == 429 or bool(exc.status_code and exc.status_code >= 500)
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        if isinstance(exc, (OcrWorkerFailure, ValueError)) and str(exc) in OCR_RESULT_CODES:
            return True
        return isinstance(exc, ValueError) and str(exc) in {
            "ocr_unavailable", "ocr_page_failed", "formatter_model_unavailable",
            "vision_model_unavailable", "vision_fallback_unavailable",
        }

    @staticmethod
    def _attempt_metadata(task: dict[str, Any]) -> dict[str, Any]:
        return {"inference_attempt_count": task["inference_attempt_count"],
                "legacy_control_poll_count": task["attempts"],
                # A poll count >0 never reveals how many old calls actually ran.
                "historical_inference_attempt_count": None if task["attempts"] else 0}

    async def _finish_failure(self, task: dict[str, Any], manifest: dict[str, Any] | None,
                              exc: Exception, started: float, deadline: float) -> None:
        if not manifest:
            await self._db.finish_task(task, success=False, deadline=deadline)
            return
        metadata = dict(exc.metadata) if isinstance(exc, ExperimentFailure) else {"options": manifest["options"]}
        elapsed = (asyncio.get_running_loop().time() - started) * 1000
        metadata.update(self._attempt_metadata(task))
        metadata.update(elapsed_ms=elapsed, requested_pipeline=manifest["options"]["pipeline"])
        item = next(x for x in manifest["inputs"] if x["receipt_id"] == str(task["receipt_id"]))
        failure_score = self._score(manifest, task["receipt_id"], Receipt.model_validate(item["expected"]), None) if "reference" in item else {}
        await self._db.finish_task(task, success=False, latency_ms=elapsed,
            attempts=len(metadata.get("calls", [])) or 1,
            result={**failure_score, "error": str(exc) or type(exc).__name__, "metadata": metadata}, deadline=deadline)

    async def _run_digests(self, run: BenchmarkRun, *, deadline: float | None = None) -> dict[str, str | None]:
        async with connection(self._receipts.pool, deadline=deadline) as conn:
            row = await conn.fetchrow("SELECT model_digests FROM benchmark_runs WHERE id=$1", run.id)
        if not row:
            return {}
        value = row["model_digests"]
        return dict(json.loads(value) if isinstance(value, str) else value)

    async def _timeout(self, run: BenchmarkRun, *, deadline: float | None = None) -> int:
        async with connection(self._receipts.pool, deadline=deadline) as conn:
            return int(await conn.fetchval("SELECT request_timeout_seconds FROM benchmark_runs WHERE id=$1", run.id))

    def capabilities(self) -> dict[str, bool]:
        from .gpu_handoff import managed_ocr_enabled
        ocr = self._receipts._ocr_settings
        return {"ocr_available": bool(ocr.service_url) and (
                    not ocr.manage_model_lifecycle or managed_ocr_enabled()),
                "ocr_enabled": self._receipts._ocr_settings.enabled,
                "thinking_control": self._provider.supports_thinking_control}

    async def results(self, run_id: UUID, *, limit: int, offset: int, scope: AccessScope) -> list[dict[str, Any]]:
        if await self._db.run(run_id, scope=scope) is None:
            raise ValueError("Benchmark run not found")
        return await self._db.results(run_id, limit=limit, offset=offset)

    async def reference_preview(self, receipt_id: UUID, *, scope: AccessScope) -> ReferencePreview:
        return await self._db.reference_preview(receipt_id, scope=scope)

    async def qualify_reference(self, receipt_id: UUID, expected_review_revision: int,
                                expected_document_sha256: str, *, scope: AccessScope) -> GoldSnapshot:
        return await self._db.qualify_reference(receipt_id, expected_review_revision,
                                                expected_document_sha256, scope=scope)

    async def _capture_input(self, receipt_id: UUID, *, scope: AccessScope) -> dict[str, Any]:
        reference = await self._db.current_reference(receipt_id, scope=scope)
        return {"receipt_id": str(receipt_id), "expected": reference.expected,
                "images": reference.sources, "reference": reference.model_dump(mode="json"),
                "scored_fields": reference.scored_fields}

    def _managed_coordinator(self, settings: OcrSettings, *, direct: bool = False) -> ModelHandoff | None:
        if direct and not settings.shared_gpu:
            return None
        handoff = self._receipts.execution_handoff(settings)
        if (handoff is not None and handoff.shared_gpu
                and handoff_resource_identity(self._provider, "llm") != handoff.resources.llm):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="execution_policy_changed",
            ))
        return handoff

    async def _run_experiment(
        self, manifest: dict[str, Any], receipt_id: UUID, model_id: str, *,
        before_inference: Callable[[], Awaitable[None]] | None = None,
        inference_deadline: float, managed_model_ids: Sequence[str],
    ) -> tuple[Receipt, ExperimentResult]:
        item = next((x for x in manifest["inputs"] if x["receipt_id"] == str(receipt_id)), None)
        if item is None:
            raise ValueError("Receipt is absent from the frozen benchmark inputs")
        options = ExperimentOptions.model_validate(manifest["options"])
        ocr, settings, _ = effective_policy(self._receipts, options)
        uses_ocr = settings.enabled and options.pipeline != ExtractionPipeline.DIRECT
        frozen_identity = manifest.get("identity")
        frozen_binding = (frozen_identity or {}).get("managed_binding_sha256")
        frozen_ocr = (frozen_identity or {}).get("ocr_settings") or {}
        frozen_shared_gpu = frozen_ocr.get("shared_gpu", True)
        was_managed = frozen_binding is not None or (
            frozen_ocr.get("enabled") is True and frozen_ocr.get("manage_model_lifecycle") is True
            and (frozen_shared_gpu or options.pipeline != ExtractionPipeline.DIRECT)
        )
        handoff = self._managed_coordinator(settings, direct=options.pipeline == ExtractionPipeline.DIRECT)
        shared_handoff = handoff is not None and handoff.shared_gpu
        was_or_is_managed = was_managed or handoff is not None
        frozen_path = frozen_ocr.get("coordination_path")
        coordinator_changed = was_managed and (
            handoff is None or (
                isinstance(frozen_path, str) and Path(frozen_path).resolve() != handoff.path.resolve()
            )
        )
        topology_error = None
        if frozen_shared_gpu != settings.shared_gpu:
            raise HandoffWait(HandoffStatus(state="configuration_required", reason_code="execution_policy_changed"))
        if coordinator_changed:
            topology_error = "Managed coordinator changed since run creation"
        elif manifest.get("endpoint_fingerprint") != endpoint_fingerprint(self._provider):
            topology_error = "Provider endpoint changed since run creation"
        elif manifest["provider"] != type(self._provider).__name__:
            topology_error = "Provider changed since run creation"
        if topology_error is not None:
            if was_or_is_managed:
                raise HandoffWait(HandoffStatus(
                    state="configuration_required", reason_code="execution_policy_changed",
                ))
            raise ValueError(f"{topology_error}; create a new run")
        schema_hash = sha256(json.dumps(Receipt.model_json_schema(), sort_keys=True).encode()).hexdigest()
        if manifest["validation_schema_sha256"] != schema_hash:
            raise ValueError("Receipt validation schema changed; create a new run")
        paths = [Path(image["path"]) for image in item["images"]]
        if not paths:
            raise ValueError("Frozen benchmark has no original images")
        for path, image in zip(paths, item["images"], strict=True):
            if await asyncio.to_thread(image_digest, path) != image["sha256"]:
                raise ValueError("Receipt image changed since run creation")
        if handoff is not None:
            # Historical direct vision may still use the current dedicated
            # coordinator. Historical managed OCR must prove its own binding.
            if shared_handoff and (uses_ocr or "managed_binding_sha256" in (frozen_identity or {})) and not frozen_binding:
                frozen_binding = ""
            status = await handoff.inspect(
                model_id=model_id, deadline=inference_deadline,
                expected_binding_sha256=frozen_binding,
            )
            if status.state != "ready":
                raise HandoffWait(status)
        if (item["pipeline"] != "direct_vision" or was_or_is_managed) and (
            manifest.get("ocr_fingerprint") != ocr_fingerprint(self._receipts._ocr_provider)
        ):
            # The client fingerprint includes the request timeout.
            # Recheck dedication because inspection released the lock.
            if handoff is not None and handoff.shared_gpu:
                handoff.require_binding(frozen_binding)
            raise ValueError("OCR provider endpoint or client settings changed; create a new run")
        current_identity = await pipeline_identity(
            self._receipts, options, expected_binding_sha256=frozen_binding,
        )
        if (current_identity["managed_binding_sha256"] is not None) != shared_handoff:
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="execution_policy_changed",
            ))
        runtime = current_identity.get("ocr_runtime") or {}
        runtime_reason = runtime.get("reason_code")
        if uses_ocr and runtime_reason == "ocr_mode_mismatch":
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="ocr_mode_mismatch",
            ))
        qualify_runtime = False
        if uses_ocr and handoff is not None and ocr is not None:
            available = await ocr.get_capabilities()
            runtime_reason = available.reason_code
            if runtime_reason == "ocr_mode_mismatch":
                raise HandoffWait(HandoffStatus(
                    state="configuration_required", reason_code="ocr_mode_mismatch",
                ))
            qualify_runtime = available.can_load and not runtime.get("runtime_fingerprint")
            runtime = {**runtime, "ready": available.ready or available.can_load}
        if uses_ocr and not runtime.get("ready"):
            # Check readiness BEFORE frozen identity: an outage is a wait, not
            # evidence that a different runtime permanently replaced this one.
            raise_for_worker_reason(runtime_reason)
            raise _ReadinessWait("ocr_unavailable")
        if frozen_identity is not None:
            if not uses_ocr:
                current_identity["ocr_runtime"] = frozen_identity.get("ocr_runtime")
            elif qualify_runtime:
                # A parked worker can still report partial identity. Check it
                # before authorizing the load that supplies its fingerprint.
                saved_runtime = frozen_identity.get("ocr_runtime") or {}
                observed_runtime = current_identity.get("ocr_runtime") or {}
                if any(value is not None and value != saved_runtime.get(key)
                       for key, value in observed_runtime.items()):
                    raise ValueError("Effective pipeline identity changed; create a new run")
                current_identity["ocr_runtime"] = saved_runtime
            if frozen_identity.get("execution_policy_version") in (None, 2):
                current_identity.pop("managed_binding_sha256")
                if "execution_policy_version" in frozen_identity:
                    current_identity["execution_policy_version"] = 2
                else:
                    current_identity.pop("execution_policy_version")
            if frozen_identity != current_identity:
                raise ValueError("Effective pipeline identity changed; create a new run")
        probe = self._provider.for_experiment(options)
        current_prompt = sha256(probe.build_prompt(list(manifest["tags"]), image_count=len(paths)).encode()).hexdigest()
        if current_prompt != manifest["prompt_hashes"][str(len(paths))]:
            raise ValueError("Extraction prompt changed since run creation; create a new run")
        options = options.model_copy(update={"pipeline": ExtractionPipeline(item["pipeline"])})
        expected = Receipt.model_validate(item["expected"])
        capability = "text" if uses_ocr else "vision"
        try:
            models = await self._provider.get_available_models(capability=capability)
        except Exception as exc:
            raise _ReadinessWait(f"{capability}_model_discovery_unavailable") from exc
        available_ids = {model.id for model in models if uses_ocr or (
            model.capabilities and any(c in model.capabilities for c in ("vision", "multimodal"))
        )}
        if model_id not in available_ids:
            raise _ReadinessWait(f"{capability}_model_unavailable")

        async def infer() -> ExperimentResult:
            return await self._receipts.analyse_submission_with_metadata(
                model_id, paths, options, list(manifest["tags"]), deadline=inference_deadline,
            )

        result: ExperimentResult | None = None
        if handoff is None:
            if before_inference is not None:
                await before_inference()
            result = await infer()
        else:
            try:
                async with handoff.session(
                    inference_deadline, model_id=model_id, model_ids=managed_model_ids,
                    expected_binding_sha256=frozen_binding, before_dispatch=before_inference,
                    lock_deadline=admission_deadline(inference_deadline),
                ) as owner:
                    if qualify_runtime and ocr is not None:
                        await owner.ocr_phase()
                        live = await ocr.get_capabilities()
                        frozen_runtime = (frozen_identity or {}).get("ocr_runtime") or {}
                        if not live.ready:
                            raise_for_worker_reason(live.reason_code)
                        if not live.ready or not live.runtime_fingerprint:
                            raise ValueError("ocr_unavailable")
                        if any(frozen_runtime.get(key) != getattr(live, key) for key in (
                            "profile_id", "runtime_fingerprint", "ocr_max_new_tokens",
                        )):
                            raise ValueError("Effective pipeline identity changed; create a new run")
                    result = await infer()
            except Exception as exc:
                if result is None:
                    raise
                # The outer owner can still fail releasing the model after inference returns.
                raise ExperimentFailure(type(exc).__name__, result.metadata) from exc
        assert result is not None
        if manifest.get("identity") is not None:
            if result.metadata.get("ordered_source_sha256") != [image["sha256"] for image in item["images"]]:
                raise ValueError("Executed source identity differs from frozen input")
            runtime = manifest["identity"].get("ocr_runtime")
            if runtime and result.metadata.get("used_pipeline") == "ocr" and any(
                runtime.get(key) != result.metadata.get(key)
                for key in ("profile_id", "runtime_fingerprint")
            ):
                raise ValueError("Executed OCR runtime differs from frozen input")
        result.metadata["requested_pipeline"] = manifest["options"]["pipeline"]
        result.metadata["resolved_pipeline"] = result.metadata.get("used_pipeline", item["pipeline"])
        result.metadata["image_hashes"] = [image["sha256"] for image in item["images"]]
        return expected, result

    @staticmethod
    def _score(manifest: dict[str, Any], receipt_id: UUID, expected: Receipt, actual: Receipt | None) -> dict[str, Any]:
        item = next(x for x in manifest["inputs"] if x["receipt_id"] == str(receipt_id))
        if "reference" not in item and actual is not None:
            return detailed_score_v3(expected, actual, classification=manifest["options"]["include_classification"])
        return detailed_score(item["expected"], actual, classification=manifest["options"]["include_classification"],
                              scored_fields=item.get("scored_fields"))

    async def _execute_experiment(
        self, task: dict[str, Any], manifest: dict[str, Any], deadline: float,
        *, inference_deadline: float, managed_model_ids: Sequence[str],
    ) -> None:
        async def begin() -> None:
            await self._begin(task, deadline)
        expected, result = await self._run_experiment(
            manifest, task["receipt_id"], task["model_id"], before_inference=begin,
            inference_deadline=inference_deadline, managed_model_ids=managed_model_ids,
        )
        detail = self._score(manifest, task["receipt_id"], expected, result.receipt)
        result.metadata.update(self._attempt_metadata(task))
        payload = {**detail, "expected": next(x["expected"] for x in manifest["inputs"] if x["receipt_id"] == str(task["receipt_id"])),
                   "actual": result.receipt.model_dump(mode="json"), "metadata": result.metadata,
                   "legacy_scores": score_receipts(expected, result.receipt)}
        await self._db.finish_task(task, success=True, score=detail["scores"]["normalized"],
                                   confidence=result.receipt.confidence, attempts=result.attempts,
                                   latency_ms=result.elapsed_ms, result=payload, deadline=deadline)
