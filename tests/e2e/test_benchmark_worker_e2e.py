"""Real benchmark worker/queue with controlled inference and readiness boundaries."""

import asyncio
import json
from time import monotonic
from unittest.mock import AsyncMock

import httpx
import pytest

from tests.e2e import test_benchmark_claims_e2e as claims
from tests.test_extraction_pipeline import formatter, sidecar
from vision_bill.config import ImageSettings, OcrSettings, WorkerSettings
from vision_bill.model.benchmark import BenchmarkCreate, CouncilPolicy
from vision_bill.model.experiment import ExperimentOptions
from vision_bill.provider.llm.base import AnalysisResult
from vision_bill.service.benchmark_service import BenchmarkService
from vision_bill.service.experiment_execution import ExperimentFailure, ExperimentResult
from vision_bill.service.receipt_service import ReceiptService

postgres, db, workflow_db = claims.postgres, claims.db, claims.workflow_db
pytestmark = pytest.mark.e2e


async def prepare(w, scope, tmp_path, *, kind="legacy", pipeline="direct_vision", worker=None, ocr_settings=None, permission=False):
    row, _body, path = await claims.references.human(w, scope, tmp_path)
    provider, ocr = formatter(), sidecar()
    # This helper controls CPU inference; it never loads managed native models.
    ocr_settings = (ocr_settings or OcrSettings(enabled=False)).model_copy(
        update={"manage_model_lifecycle": False}
    )
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path), tmp_dir=str(tmp_path)), w._settings,
        provider, ocr_provider=ocr, ocr_settings=ocr_settings)
    receipts._db = w._receipts
    service = BenchmarkService(provider, receipts, worker=worker) if worker else BenchmarkService(provider, receipts)
    if kind == "legacy":
        run = await service._db.create_run(model_ids=["fixture"], receipt_ids=[row.id], fingerprint="f",
            model_digests={}, timeout=600, policy=CouncilPolicy.ALL, absolute=None, relative=None,
            apply_flags=False, prompt_version="2", scoring_version="2")
    else:
        preview = await service.reference_preview(row.id, scope=scope)
        await service.qualify_reference(row.id, 1, preview.document_sha256, scope=scope)
        run = await service.create_run(BenchmarkCreate(model_ids=["fixture"], receipt_ids=[row.id],
            experiment=ExperimentOptions(pipeline=pipeline, include_classification=False, allow_ocr_fallback=permission)), scope=scope)
        if kind == "old_manifest":
            manifest = await service._db.manifest(run.id)
            manifest["inputs"][0].pop("reference")
            manifest["inputs"][0].pop("scored_fields")
            async with w.pool.acquire() as conn:
                await conn.execute("UPDATE benchmark_runs SET experiment_manifest=$2,scoring_version='3' WHERE id=$1", run.id, json.dumps(manifest))
    expected, _ = await service._expected_receipt_and_path(row.id)
    # Queue/preflight/scoring/persistence remain real; only external inference is controlled.
    result = AnalysisResult(expected, 7, 2.0) if kind == "legacy" else ExperimentResult(expected, 7, 2.0, {
        "ordered_source_sha256": [(await service._db.manifest(run.id))["inputs"][0]["images"][0]["sha256"]],
        "used_pipeline": "direct_vision", "calls": [{"elapsed_ms": 1}] * 7,
    })
    inference = AsyncMock(return_value=result)
    if kind == "legacy":
        provider.analyse_receipt_with_metadata = inference
    else:
        receipts.analyse_submission_with_metadata = inference
    return service, run, row, path, inference, ocr


async def test_unmanifested_history_fails_before_discovery_or_new_inference(workflow_db, tmp_path):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind="legacy")
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE benchmark_tasks SET attempts=431 WHERE run_id=$1", run.id)
    service._provider.get_available_models.side_effect = AssertionError("history must not rediscover")
    task = dict(await service._db.lease_task())
    assert task["inference_attempt_count"] == 0
    assert task["attempts"] == 431
    await service._execute(task)
    service._provider.get_available_models.assert_not_awaited()
    inference.assert_not_awaited()
    assert await service._db.lease_task() is None
    assert (await service._db.summaries(run.id))[0].failed == 1
    assert await service._db.results(run.id) == []


@pytest.mark.parametrize("kind", ["manifest", "old_manifest"])
@pytest.mark.parametrize("outcome", ["success", "exhausted"])
async def test_two_transient_failures_then_success_or_terminal_third(workflow_db, tmp_path, kind, outcome):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind=kind)
    success = inference.return_value
    failure = ExperimentFailure("ConnectError", {"calls": []})
    failure.__cause__ = httpx.ConnectError("provider unavailable")
    inference.side_effect = [failure, failure, success if outcome == "success" else failure]
    for attempt in (1, 2, 3):
        task = await service._db.lease_task()
        await service._execute(dict(task))
        summary = (await service._db.summaries(run.id))[0]
        if attempt < 3:
            current = await claims.saved(w, run.id)
            assert current is not None, "transient failure must not terminalize before exhaustion"
            assert current["inference_attempt_count"] == attempt
            assert current["status"] == "retrying"
            assert summary.completed == 0 and await service._db.results(run.id) == []
            await claims.due(w, run.id)
    assert inference.await_count == 3
    assert await service._db.lease_task() is None
    assert summary.completed == 1 and summary.failed == (outcome == "exhausted")
    assert (await service._db.run(run.id)).status == "completed"
    # Provider-call telemetry remains separate from durable automatic-attempt count.
    if outcome == "success":
        assert summary.average_attempts == 7
        assert summary.average_score == pytest.approx(1)
    if kind == "manifest":
        result = (await service._db.results(run.id))[0]
        assert result["metadata"]["inference_attempt_count"] == 3
        assert result["metadata"]["legacy_control_poll_count"] == 0
        assert result["metadata"]["historical_inference_attempt_count"] == 0
        assert summary.all_case_average_score == (1 if outcome == "success" else 0)
        assert summary.full_receipt_correct_rate == (1 if outcome == "success" else 0)
        if outcome == "exhausted":
            assert result["full_receipt_correct"] is False
            assert result["scores"]["normalized"] == 0
    else:
        assert summary.all_case_average_score is None and summary.full_receipt_correct_rate is None
        if kind == "old_manifest":
            result = (await service._db.results(run.id))[0]
            assert result.get("version") != "4"
            if outcome == "exhausted":
                assert "full_receipt_correct" not in result


@pytest.mark.parametrize("reason", ["discovery", "missing", "digest", "missing_digest", "ocr"])
async def test_repeated_readiness_waits_spend_no_attempts(workflow_db, tmp_path, reason):
    w, scope = workflow_db
    service, run, _, _, inference, ocr = await prepare(w, scope, tmp_path,
        kind="manifest", pipeline="ocr" if reason == "ocr" else "direct_vision")
    if reason == "discovery":
        service._provider.get_available_models.side_effect = httpx.ConnectError("offline")
    elif reason == "missing":
        service._provider.get_available_models.return_value = []
    elif reason in {"digest", "missing_digest"}:
        service._provider.get_available_models.return_value[0].digest = "changed" if reason == "digest" else None
        async with w.pool.acquire() as conn:
            await conn.execute("UPDATE benchmark_runs SET model_digests='{\"fixture\":\"frozen\"}' WHERE id=$1", run.id)
    else:
        ocr.get_capabilities.return_value.ready = False
        ocr.get_capabilities.return_value.runtime_fingerprint = None
    for _ in range(4):
        await service._execute(dict(await service._db.lease_task()))
        task = await claims.saved(w, run.id)
        assert task is not None and task["status"] == "waiting_for_model"
        assert task["inference_attempt_count"] == 0
        assert (await service._db.summaries(run.id))[0].completed == 0
        await claims.due(w, run.id)
    assert inference.await_count == 0


@pytest.mark.parametrize("kind", ["legacy", "manifest", "old_manifest"])
async def test_expired_third_attempt_terminalizes_even_without_discovery(workflow_db, tmp_path, kind):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind=kind)
    for _ in range(3):
        await service._db.begin_attempt(await service._db.lease_task())
        await claims.due(w, run.id, expire=True)
    service._provider.get_available_models.side_effect = AssertionError("exhausted work must not need discovery")
    recovery = dict(await service._db.lease_task())
    assert recovery["inference_attempt_count"] == 3
    await service._execute(recovery)
    await service._execute(recovery)
    assert await claims.saved(w, run.id) is None
    assert (await service._db.summaries(run.id))[0].failed == 1
    assert inference.await_count == 0
    if kind == "manifest":
        result = (await service._db.results(run.id))[0]
        assert result["scores"]["normalized"] == 0 and result["full_receipt_correct"] is False


@pytest.mark.parametrize("violation", ["missing", "hash", "endpoint", "prompt", "schema", "runtime"])
async def test_frozen_preflight_violations_are_terminal_without_inference(workflow_db, tmp_path, violation):
    w, scope = workflow_db
    service, run, _, path, inference, ocr = await prepare(w, scope, tmp_path, kind="manifest",
        pipeline="ocr" if violation == "runtime" else "direct_vision")
    if violation == "missing":
        path.unlink()
    elif violation == "hash":
        path.write_bytes(b"changed")
    elif violation == "runtime":
        ocr.get_capabilities.return_value.runtime_fingerprint = "changed while ready"
    else:
        manifest = await service._db.manifest(run.id)
        key = {"endpoint": "endpoint_fingerprint", "prompt": "prompt_hashes", "schema": "validation_schema_sha256"}[violation]
        manifest[key] = {"1": "changed"} if violation == "prompt" else "changed"
        async with w.pool.acquire() as conn:
            await conn.execute("UPDATE benchmark_runs SET experiment_manifest=$2 WHERE id=$1", run.id, json.dumps(manifest))
    await service._execute(dict(await service._db.lease_task()))
    assert inference.await_count == 0
    assert (await service._db.summaries(run.id))[0].failed == 1
    result = (await service._db.results(run.id))[0]
    assert result["metadata"]["inference_attempt_count"] == 0
    assert result["scores"]["normalized"] == 0


@pytest.mark.parametrize("kind", ["manifest", "old_manifest"])
async def test_unreadable_output_is_permanent_after_one_attempt(workflow_db, tmp_path, kind):
    from vision_bill.provider.llm.errors import UnreadableReceiptError
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind=kind)
    inference.side_effect = UnreadableReceiptError("unreadable")
    await service._execute(dict(await service._db.lease_task()))
    assert (await service._db.summaries(run.id))[0].failed == 1
    assert inference.await_count == 1 and await service._db.lease_task() is None


@pytest.mark.parametrize("pipeline,enabled,fallback,want_wait", [
    ("direct_vision", True, False, False), ("upload", False, False, False),
    ("upload", True, True, True), ("upload", True, False, True), ("ocr", False, True, True),
])
async def test_readiness_obeys_effective_policy(workflow_db, tmp_path, pipeline, enabled, fallback, want_wait):
    w, scope = workflow_db
    service, run, _, _, inference, ocr = await prepare(w, scope, tmp_path, kind="manifest",
        pipeline=pipeline, permission=fallback, ocr_settings=OcrSettings(enabled=enabled, allow_vision_fallback=fallback))
    ocr.get_capabilities.return_value.ready = False
    ocr.get_capabilities.return_value.reachable = False
    ocr.get_capabilities.return_value.profile_id = None
    await service._execute(dict(await service._db.lease_task()))
    summary = (await service._db.summaries(run.id))[0]
    assert summary.completed == (0 if want_wait else 1)
    assert inference.await_count == (0 if want_wait else 1)
    if want_wait:
        assert (await claims.saved(w, run.id))["inference_attempt_count"] == 0
    else:
        assert summary.succeeded == 1


@pytest.mark.parametrize("kind", ["manifest", "old_manifest"])
async def test_configured_single_attempt_and_unknown_legacy_baseline(workflow_db, tmp_path, kind):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind=kind,
        worker=WorkerSettings(max_attempts=1))
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE benchmark_tasks SET attempts=431 WHERE run_id=$1", run.id)
    inference.side_effect = TimeoutError("provider timeout")
    await service._execute(dict(await service._db.lease_task()))
    assert inference.await_count == 1
    assert await service._db.lease_task() is None
    assert (await service._db.summaries(run.id))[0].failed == 1
    metadata = (await service._db.results(run.id))[0]["metadata"]
    assert metadata["inference_attempt_count"] == 1
    assert metadata["legacy_control_poll_count"] == 431
    assert metadata["historical_inference_attempt_count"] is None


@pytest.mark.parametrize("failure_code", ["ocr_page_failed", "ocr_unavailable"])
async def test_wrapped_missing_source_is_permanent_even_with_availability_code(workflow_db, tmp_path, failure_code):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind="manifest")
    failure = ExperimentFailure(failure_code, {"calls": []})
    failure.__cause__ = FileNotFoundError("original disappeared")
    inference.side_effect = failure
    await service._execute(dict(await service._db.lease_task()))
    assert (await service._db.summaries(run.id))[0].failed == 1
    assert await claims.saved(w, run.id) is None
    assert inference.await_count == 1


async def test_stale_worker_stops_before_inference_after_preflight_barrier(workflow_db, tmp_path):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind="manifest")
    entered, release = asyncio.Event(), asyncio.Event()
    models = service._provider.get_available_models.return_value
    async def discovery(**kwargs):
        entered.set()
        await release.wait()
        return models
    service._provider.get_available_models.side_effect = discovery
    old = dict(await service._db.lease_task())
    work = asyncio.create_task(service._execute(old))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await claims.due(w, run.id, expire=True)
        current = dict(await service._db.lease_task())
        release.set()
        await asyncio.wait_for(work, 2)
        assert inference.await_count == 0
        assert dict(await claims.saved(w, run.id)) == current
        assert (await service._db.summaries(run.id))[0].completed == 0
    finally:
        release.set()
        await asyncio.gather(work, return_exceptions=True)


async def test_legacy_permission_cannot_bypass_ocr_readiness(workflow_db, tmp_path):
    w, scope = workflow_db
    service, run, _, _, inference, ocr = await prepare(w, scope, tmp_path, kind="manifest", pipeline="upload", permission=True,
        ocr_settings=OcrSettings(enabled=True, allow_vision_fallback=True))
    service._provider.get_available_models.return_value[0].capabilities = ["completion"]
    ocr.get_capabilities.return_value.ready = False
    await service._execute(dict(await service._db.lease_task()))
    task = await claims.saved(w, run.id)
    assert task is not None and task["status"] == "waiting_for_model"
    assert task["inference_attempt_count"] == 0 and inference.await_count == 0


@pytest.mark.parametrize("change", ["formatter_budget", "ocr_recovery_race"])
async def test_strict_upload_preserves_frozen_contract_and_executed_runtime(workflow_db, tmp_path, change):
    w, scope = workflow_db
    service, run, _, _, inference, _ocr = await prepare(w, scope, tmp_path, kind="manifest", pipeline="upload", permission=True,
        ocr_settings=OcrSettings(enabled=True, allow_vision_fallback=True))
    if change == "formatter_budget":
        service._provider.max_output_tokens += 1
    else:
        # Ready frozen OCR reported a different runtime during extraction.
        inference.return_value.metadata.update(used_pipeline="ocr", profile_id="different", runtime_fingerprint="different")
    await service._execute(dict(await service._db.lease_task()))
    assert (await service._db.summaries(run.id))[0].failed == 1
    expected_count = 0 if change == "formatter_budget" else 1
    assert inference.await_count == expected_count
    result = (await service._db.results(run.id))[0]
    assert result["metadata"]["inference_attempt_count"] == expected_count
    assert result["scores"]["normalized"] == 0


@pytest.mark.parametrize("budget", ["worker", "saved_run"])
@pytest.mark.parametrize("phase", ["inference", "preflight", "persistence", "preflight_bookkeeping"])
async def test_complete_worker_deadline_includes_preflight_and_persistence(workflow_db, tmp_path, phase, budget):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await prepare(w, scope, tmp_path, kind="manifest",
        worker=WorkerSettings(attempt_timeout_seconds=1 if budget == "worker" else 600))
    if budget == "saved_run":
        async with w.pool.acquire() as conn:
            await conn.execute("UPDATE benchmark_runs SET request_timeout_seconds=1 WHERE id=$1", run.id)
    reached = asyncio.Event()
    cancelled = asyncio.Event()
    release_inference = asyncio.Event()
    original_retry = service._db.retry_task
    async def blocked(*args, **kwargs):
        reached.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    if phase == "inference":
        inference.side_effect = blocked
    elif phase == "preflight":
        service._provider.get_available_models.side_effect = blocked
    elif phase == "preflight_bookkeeping":
        service._provider.get_available_models.side_effect = RuntimeError("discovery unavailable")
        async def blocked_retry(*args, **kwargs):
            reached.set()
            try:
                return await original_retry(*args, **kwargs)
            finally:
                cancelled.set()
        service._db.retry_task = blocked_retry
    if phase in ("persistence", "preflight_bookkeeping"):
        # Hold a real task lock through the deadline. Preflight blocks retry;
        # persistence blocks finish and retry after a real begin transition.
        blocker = await w.pool.acquire()
        tx = blocker.transaction()
        await tx.start()
        async def result_after_barrier(*args, **kwargs):
            reached.set()
            await release_inference.wait()
            return inference.return_value
        if phase == "persistence":
            inference.side_effect = result_after_barrier
    started = monotonic()
    original_claim = dict(await service._db.lease_task())
    if phase == "preflight_bookkeeping":
        await blocker.fetchval("SELECT 1 FROM benchmark_tasks WHERE run_id=$1 FOR UPDATE", run.id)
    work = asyncio.create_task(service._execute(dict(original_claim)))
    try:
        if phase == "persistence":
            await asyncio.wait_for(reached.wait(), 1)
            await blocker.fetchval("SELECT 1 FROM benchmark_tasks WHERE run_id=$1 FOR UPDATE", run.id)
            release_inference.set()
            await claims.wait_blocked(w.pool, 1)
        else:
            await asyncio.wait_for(reached.wait(), 1)
            if phase == "preflight_bookkeeping":
                await claims.wait_blocked(w.pool, 1)
        await asyncio.wait_for(work, 2)
        assert monotonic() - started < 1.8
    finally:
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
        if phase in ("persistence", "preflight_bookkeeping"):
            await tx.rollback()
            await w.pool.release(blocker)
        service._db.retry_task = original_retry
    task = await claims.saved(w, run.id)
    assert task is not None
    count = 0 if phase.startswith("preflight") else 1
    assert task["inference_attempt_count"] == count
    assert task["inference_started"] == bool(count)
    if phase in ("persistence", "preflight_bookkeeping"):
        assert task["status"] == "running"
    if task["status"] == "running":
        # Bookkeeping may exhaust its bounded reserve. Every reservation field
        # must remain intact, not just an arbitrary acceptable status string.
        assert dict(task) == {
            **original_claim, "inference_attempt_count": count, "inference_started": bool(count)
        }
    else:
        assert task["status"] == ("waiting_for_model" if count == 0 else "retrying")
        assert task["claim_token"] is None and task["leased_until"] is None
        assert task["retry_at"] is not None and task["retry_at"] > task["started_at"]
        assert dict(task) == {
            **original_claim, "inference_attempt_count": count, "inference_started": bool(count),
            "status": task["status"], "claim_token": None, "leased_until": None,
            "retry_at": task["retry_at"],
        }
    assert (await service._db.summaries(run.id))[0].completed == 0
    assert await service._db.results(run.id) == []
    if phase != "persistence":
        assert cancelled.is_set()
    service._provider.get_available_models.side_effect = None
    inference.side_effect = None
    await claims.due(w, run.id, expire=task["status"] == "running")
    recovered = dict(await service._db.lease_task())
    assert recovered["claim_token"] != original_claim["claim_token"]
    assert recovered["inference_attempt_count"] == count and not recovered["inference_started"]
    assert recovered["retry_at"] is None and recovered["leased_until"] is not None
    summaries = await service._db.summaries(run.id)
    run_before = await service._db.run(run.id)
    assert await service._db.begin_attempt(original_claim) is None
    assert not await service._db.retry_task(original_claim, waiting=True)
    assert not await service._db.finish_task(original_claim, success=True)
    assert dict(await claims.saved(w, run.id)) == recovered
    assert await service._db.summaries(run.id) == summaries
    assert await service._db.run(run.id) == run_before
    assert await service._db.results(run.id) == []
    await service._execute(recovered)
    assert await service._db.lease_task() is None
    assert (await service._db.summaries(run.id))[0].completed == 1
    results = await service._db.results(run.id)
    assert len(results) == 1 and results[0]["metadata"]["inference_attempt_count"] == count + 1


async def test_upload_without_permission_waits_despite_global_fallback(workflow_db, tmp_path):
    w, scope = workflow_db
    service, run, _, _, inference, ocr = await prepare(
        w, scope, tmp_path, kind="manifest", pipeline="upload",
        ocr_settings=OcrSettings(enabled=True, allow_vision_fallback=True),
    )
    ocr.get_capabilities.return_value.ready = False
    await service._execute(dict(await service._db.lease_task()))
    task = await claims.saved(w, run.id)
    assert task["status"] == "waiting_for_model"
    assert task["inference_attempt_count"] == 0 and inference.await_count == 0
