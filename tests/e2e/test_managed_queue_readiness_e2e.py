"""Managed queue admission, real PostgreSQL fences and native CPU lifecycle doubles."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from functools import partial
from time import monotonic
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from ocr_service.worker import OcrWorkerManager
from PIL import Image
from test_extraction_pipeline import formatter
from test_handoff_binding import adopt
from test_ocr_worker_lifecycle import FakeContext

from tests.e2e import test_benchmark_claims_e2e as claims
from vision_bill import config
from vision_bill.config import OcrSettings, WorkerSettings
from vision_bill.model.benchmark import BenchmarkCreate
from vision_bill.model.db.image import ImagePart
from vision_bill.model.experiment import ExperimentOptions
from vision_bill.model.handoff import HandoffWait
from vision_bill.model.ocr import OcrCapabilities
from vision_bill.model.workflow import RetryRequest, SubmissionCreate
from vision_bill.service.analysis_scheduler import AnalysisScheduler
from vision_bill.service.benchmark_service import BenchmarkService
from vision_bill.service.execution_policy import current_snapshot
from vision_bill.service.gpu_handoff import ModelHandoff
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.service.submission_service import SubmissionService

postgres, db, workflow_db = claims.postgres, claims.db, claims.workflow_db
pytestmark = pytest.mark.e2e


class Native:
    managed_lifecycle = True
    lifecycle_identity = "ocr:managed-fixture"

    def __init__(self):
        self.context = FakeContext()
        self.worker = OcrWorkerManager(mp_context=self.context, startup_timeout=0.2)
        self.events = []

    async def claim(self, token, deadline):
        self.worker.claim(token)
        self.events.append("claim")

    async def load(self, token, deadline):
        self.worker.load(token, deadline_monotonic=deadline)
        self.events.append("load")

    async def park(self, token, deadline):
        self.worker.park(token)
        self.events.append("park")

    async def get_capabilities(self):
        ready = self.worker.readiness()
        return OcrCapabilities(
            enabled=True,
            configured=True,
            reachable=True,
            ready=ready.ready,
            can_load=ready.reason_code == "ocr_parked",
            profile_id=ready.profile_id,
            runtime_fingerprint=ready.runtime_fingerprint,
            reason_code=ready.reason_code,
        )

    async def extract_page(self, *args, **kwargs):
        pytest.fail("unready work reached native inference")


@pytest.fixture
async def managed(workflow_db, settings, monkeypatch, tmp_path):
    from vision_bill.provider import factory

    w, _scope = workflow_db
    w._receipts._pool = w.pool
    settings.ocr = OcrSettings(
        enabled=True,
        service_url="http://native.invalid",
        manage_model_lifecycle=True,
        coordination_path=tmp_path / "A.lock",
    )
    settings.worker = WorkerSettings(attempt_timeout_seconds=1)
    settings.llm.model_name = "fixture"
    monkeypatch.setattr(config, "settings", settings)
    workers = []
    providers = []

    def make(native=None):
        if native is None:
            native = Native()
            workers.append(native)
        llm = formatter()
        providers.append(llm)
        llm._client.generate = AsyncMock(return_value=None)
        llm.check_connection = AsyncMock(return_value=True)
        llm.handoff = ModelHandoff(settings.ocr.coordination_path, llm, native)
        monkeypatch.setattr(factory, "get_ocr_provider", lambda _: native)
        receipts = ReceiptService(
            settings.images,
            w._settings,
            llm,
            ocr_provider=native,
            ocr_settings=settings.ocr,
            worker_settings=settings.worker,
        )
        receipts._db = w._receipts
        receipts._workflow_db = w
        submissions = SubmissionService(settings, receipts, receipts._image_service)
        scheduler = AnalysisScheduler(
            settings, llm, receipts, receipts._image_db, submission_service=submissions
        )
        benchmark = BenchmarkService(llm, receipts, worker=settings.worker)
        return llm, native, receipts, submissions, scheduler, benchmark

    yield make
    for native in workers:
        native.worker.stop()
    for llm in providers:
        await llm._client._client.aclose()


def frozen(settings, handoff):
    # The binding produced by explicit adoption is the only authority.
    binding = handoff.read_binding().model_dump()
    digest = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **current_snapshot(settings, inference_identity=handoff.provider.inference_identity),
        "managed_binding_sha256": digest,
    }


async def upload(w, scope, tmp_path, snapshot, *, model="fixture"):
    path = tmp_path / f"{uuid4()}.png"
    Image.new("RGB", (4, 5), (1, 0, 0)).save(path)
    return (
        await w.enqueue_submission(
            SubmissionCreate(
                id=uuid4(),
                parts=[ImagePart(image_path=str(path))],
                source_manifest=[
                    {
                        "photo_index": 0,
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ],
                requested_options={"model_id": model},
                request_fingerprint=uuid4().hex,
                effective_options=snapshot,
                model_id=model,
                bypass_review=True,
            ),
            scope=scope,
            idempotency_key=None,
        )
    ).submission


async def queued_benchmark(w, scope, tmp_path, benchmark, *, pipeline="direct_vision"):
    receipt, _, path = await claims.references.human(w, scope, tmp_path)
    Image.new("RGB", (4, 5), (1, 0, 0)).save(path, format="PNG")
    preview = await benchmark.reference_preview(receipt.id, scope=scope)
    await benchmark.qualify_reference(receipt.id, 1, preview.document_sha256, scope=scope)
    return await benchmark.create_run(
        BenchmarkCreate(
            model_ids=["fixture"],
            receipt_ids=[receipt.id],
            request_timeout_seconds=1,
            experiment=ExperimentOptions(pipeline=pipeline, include_classification=False),
        ),
        scope=scope,
    )


async def assert_upload_held(w, scope, image, reason, *, history=0, count=0):
    state = await w.get_submission(image.id, scope=scope)
    assert state.wait_reason == reason
    assert state.status == "pending" and state.attempt_count == count
    assert state.generation == image.generation
    async with w.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1", image.id)
        assert row["claim_token"] is None and row["lease_expires_at"] is None
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM receipt_extractions WHERE image_id=$1", image.id
            )
            == history
        )
    return row


async def test_old_native_generation_cannot_supersede_active_new_coordinator(
    workflow_db,
    settings,
    tmp_path,
    managed,
):
    w, scope = workflow_db
    old_llm, old_native, *_ = managed()
    await adopt(old_llm.handoff, ("fixture", "other"))
    async with old_llm.handoff.session(monotonic() + 2, model_id="fixture"):
        pass
    old_native.worker.stop()
    old = current_snapshot(settings, inference_identity=old_llm.inference_identity)
    old.pop("inference_identity")
    old.pop("managed_binding_sha256", None)
    old["execution_policy_version"] = 2
    image = await upload(w, scope, tmp_path, old)
    settings.ocr = settings.ocr.model_copy(update={"coordination_path": tmp_path / "B.lock"})
    llm, native, receipts, submissions, scheduler, _ = managed()
    await adopt(llm.handoff, ("fixture", "other"))
    old_journal = old_llm.handoff.journal.read_bytes()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def reply(**kwargs):
        entered.set()
        await finish.wait()
        return llm._client.chat.return_value

    llm._client.chat.side_effect = reply
    active = asyncio.create_task(
        llm.send_message("other", [{"role": "user", "content": "fixture"}])
    )
    await asyncio.wait_for(entered.wait(), 2)
    observed = list(native.events)
    try:
        assert await scheduler.process_pending() == []
        row = await assert_upload_held(w, scope, image, "execution_policy_changed")
        assert json.loads(row["effective_options"]) == old
        assert native.events == observed
        with pytest.raises(
            (HandoffWait, ValueError),
            match="execution_policy_changed|model_handoff_identity_changed",
        ):
            await receipts.extract_receipt(
                "fixture",
                [tmp_path / "unused.png"],
                options=ExperimentOptions(),
                deadline=monotonic() + 1,
                tags=[],
                ocr_settings=OcrSettings.model_validate(old["ocr"]),
            )
        assert native.events == observed and len(native.context.processes) == 0
        assert old_llm.handoff.journal.read_bytes() == old_journal
    finally:
        finish.set()
        await active
    replacement = submissions.current_snapshot()
    assert (
        replacement["managed_binding_sha256"]
        == frozen(settings, llm.handoff)["managed_binding_sha256"]
    )
    adopted = await w.retry_submission(
        image.id,
        RetryRequest(expected_generation=image.generation, request_key="current", policy="current"),
        scope=scope,
        replacement_snapshot=replacement,
    )
    assert adopted.generation == image.generation + 1 and adopted.attempt_count == 0
    candidate = (await w.due_candidates())[0]
    assert await submissions.ready_for(candidate)
    async with w.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT effective_options,legacy_events FROM images WHERE id=$1", image.id
        )
    assert json.loads(row["effective_options"]) == replacement
    assert json.loads(row["legacy_events"])[-1]["effective_options"] == old


async def test_same_path_new_resource_holds_before_actual_native_claim(
    workflow_db,
    settings,
    tmp_path,
    managed,
):
    w, scope = workflow_db
    llm, native, *_ = managed()
    await adopt(llm.handoff, ("fixture",))
    snapshot = frozen(settings, llm.handoff)
    image = await upload(w, scope, tmp_path, snapshot)
    native.lifecycle_identity = "ocr:new-endpoint"
    llm, native, _, _, scheduler, _ = managed(native)
    assert await scheduler.process_pending() == []
    row = await assert_upload_held(w, scope, image, "execution_policy_changed")
    assert json.loads(row["effective_options"]) == snapshot
    assert native.events == []


@pytest.mark.parametrize("state", ["busy", "unsettled", "race"])
async def test_both_queue_holds_keep_fences_and_attempts_across_restart(
    workflow_db,
    settings,
    tmp_path,
    managed,
    monkeypatch,
    state,
):
    w, scope = workflow_db
    llm, native, _, _, _, benchmark = managed()
    await adopt(llm.handoff, ("fixture",))
    snapshot = frozen(settings, llm.handoff)
    image = await upload(w, scope, tmp_path, snapshot)
    run = await queued_benchmark(w, scope, tmp_path, benchmark)
    manifest = await benchmark._db.manifest(run.id)
    expected = "model_request_unsettled" if state == "unsettled" else "model_handoff_busy"
    if state == "unsettled":
        llm.handoff.journal.write_text(
            json.dumps(
                {
                    "version": 2,
                    "coordinator": "vision-bill-model-handoff-v2",
                    "resources": llm.handoff.resources.model_dump(),
                    "generation": 1,
                    "models": ["fixture"],
                    "unsettled": True,
                    "ocr_pending": False,
                }
            )
        )
    original_journal = llm.handoff.journal.read_bytes() if state == "unsettled" else None
    first_token = None
    observed_benchmark_reasons = []

    for _ in range(2):
        llm, _, _, _, scheduler, benchmark = managed(native)
        hold = getattr(benchmark._db, "hold_before_inference", None)
        if hold is not None:

            async def record_hold(task, reason, *, deadline=None, hold=hold):
                observed_benchmark_reasons.append(reason)
                return await hold(task, reason, deadline=deadline)

            benchmark._db.hold_before_inference = record_hold
        reservation = await benchmark._db.lease_task()
        assert reservation["claim_token"] != first_token
        assert not reservation["inference_started"]
        first_token = reservation["claim_token"]
        if state == "busy":
            async with llm.handoff.exclusion(monotonic() + 1):
                await scheduler.process_pending()
                await benchmark._execute(dict(reservation))
        elif state == "race":
            reached, proceed = asyncio.Event(), asyncio.Event()
            enter = ModelHandoff.session

            @asynccontextmanager
            async def gate(
                coordinator, *args, reached=reached, proceed=proceed, enter=enter, **kwargs
            ):
                reached.set()
                await proceed.wait()
                async with enter(coordinator, *args, **kwargs) as owner:
                    yield owner

            monkeypatch.setattr(ModelHandoff, "session", gate)
            for call in (scheduler.process_pending, partial(benchmark._execute, dict(reservation))):
                reached.clear()
                proceed.clear()
                job = asyncio.create_task(call())
                try:
                    await asyncio.wait_for(reached.wait(), 2)
                    async with llm.handoff.exclusion(monotonic() + 1):
                        proceed.set()
                        await asyncio.wait_for(job, 2)
                finally:
                    if not job.done():
                        job.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await job
            monkeypatch.setattr(ModelHandoff, "session", enter)
        else:
            await scheduler.process_pending()
            await benchmark._execute(dict(reservation))
        row = await assert_upload_held(w, scope, image, expected)
        assert json.loads(row["effective_options"]) == snapshot
        held = await claims.saved(w, run.id)
        assert held is not None and held["status"] == "waiting_for_model"
        assert held["attempts"] == held["inference_attempt_count"] == 0
        assert held["claim_token"] is None and held["leased_until"] is None
        assert (await benchmark._db.summaries(run.id))[0].completed == 0
        assert await benchmark._db.results(run.id) == []
        assert await benchmark._db.manifest(run.id) == manifest
        await claims.due(w, run.id)

    assert native.events == []
    assert observed_benchmark_reasons == [expected, expected]
    if original_journal is None:
        assert not llm.handoff.journal.exists()
    else:
        assert llm.handoff.journal.read_bytes() == original_journal


async def test_upload_readiness_refund_cannot_touch_new_claim_or_prior_attempt(
    workflow_db,
    settings,
    tmp_path,
    managed,
):
    from vision_bill.model.workflow import AttemptOutcome

    w, scope = workflow_db
    llm, _, *_ = managed()
    await adopt(llm.handoff, ("fixture",))
    image = await upload(w, scope, tmp_path, frozen(settings, llm.handoff))
    first = await w.claim_next()
    reason = "model_handoff_busy"
    await w.finish_attempt(first, AttemptOutcome(kind="retry", retry_delay_seconds=0))
    waiting = await w.claim_next()
    await w.hold_claim_for_policy(waiting, "execution_policy_changed")
    current = await w.claim_next()
    assert current.claim_token != waiting.claim_token
    assert (await w.hold_claim_for_readiness(waiting, reason)).stale
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT claim_token FROM images WHERE id=$1", image.id)
            == current.claim_token
        )
    assert not (await w.hold_claim_for_readiness(current, reason)).stale
    await assert_upload_held(w, scope, image, reason, history=1, count=1)
    assert (await w.finish_attempt(current, AttemptOutcome(kind="terminal"))).stale


async def test_benchmark_readiness_hold_cannot_release_replaced_or_begun_claim(
    workflow_db, tmp_path
):
    w, scope = workflow_db
    benchmark, run, _, _ = await claims.queue(w, scope, tmp_path)
    stale = await benchmark.lease_task()
    await claims.due(w, run.id, expire=True)
    replacement = await benchmark.lease_task()
    assert replacement["claim_token"] != stale["claim_token"]
    assert not await benchmark.hold_before_inference(stale, "model_handoff_busy")
    assert (await claims.saved(w, run.id))["claim_token"] == replacement["claim_token"]
    begun = await benchmark.begin_attempt(replacement)
    assert begun["inference_attempt_count"] == 1
    assert not await benchmark.hold_before_inference(replacement, "model_handoff_busy")
    assert dict(await claims.saved(w, run.id)) == begun


async def test_historical_direct_benchmark_keeps_manifest_and_coordinates_without_ocr(
    workflow_db,
    tmp_path,
    managed,
):
    w, scope = workflow_db
    llm, native, _, _, _, benchmark = managed()
    await adopt(llm.handoff, ("fixture",))
    run = await queued_benchmark(w, scope, tmp_path, benchmark)
    old = await benchmark._db.manifest(run.id)
    old["identity"]["execution_policy_version"] = 2
    old["identity"].pop("managed_binding_sha256", None)
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE benchmark_runs SET experiment_manifest=$2 WHERE id=$1", run.id, json.dumps(old)
        )
    await benchmark._execute(dict(await benchmark._db.lease_task()))
    summary = (await benchmark._db.summaries(run.id))[0]
    assert summary.succeeded == 1
    assert await benchmark._db.manifest(run.id) == old
    assert native.events == ["claim", "park"]
    assert len(native.context.processes) == 0
    assert llm._client.chat.await_count == 1
    assert llm._client.chat.call_args.kwargs["messages"][0]["images"]
    assert (await benchmark._db.results(run.id))[0]["metadata"]["inference_attempt_count"] == 1


async def test_benchmark_qualification_checks_every_model_before_native_load(
    workflow_db, tmp_path, managed
):
    from vision_bill.provider.llm.base import ModelInfo

    w, scope = workflow_db
    llm, native, _, _, _, benchmark = managed()
    await adopt(llm.handoff, ("fixture",))
    other = ModelInfo("not-adopted")
    other.capabilities = ["completion"]
    llm.get_available_models.return_value.append(other)
    receipt, _, _ = await claims.references.human(w, scope, tmp_path)
    preview = await benchmark.reference_preview(receipt.id, scope=scope)
    await benchmark.qualify_reference(receipt.id, 1, preview.document_sha256, scope=scope)
    with pytest.raises(HandoffWait, match="model_handoff_model_not_adopted"):
        await benchmark.create_run(
            BenchmarkCreate(
                model_ids=["fixture", other.id],
                receipt_ids=[receipt.id],
                experiment=ExperimentOptions(pipeline="ocr", include_classification=False),
            ),
            scope=scope,
        )
    assert native.events == []
    assert not llm.handoff.journal.exists()
    assert not await benchmark._db.runs(scope=scope)


@pytest.mark.parametrize("scenario", ["semantic-drift", "native-attempt"])
async def test_cold_benchmark_qualification_obeys_the_frozen_contract_and_attempt_fence(
    workflow_db,
    tmp_path,
    managed,
    scenario,
):
    from vision_bill.service.extraction_pipeline import OcrRuntimeChangedError

    w, scope = workflow_db
    llm, native, _, _, _, benchmark = managed()
    await adopt(llm.handoff, ("fixture",))
    run = await queued_benchmark(w, scope, tmp_path, benchmark, pipeline="ocr")
    assert native.events == ["claim", "load", "park"]
    native.events.clear()

    if scenario == "semantic-drift":
        manifest = await benchmark._db.manifest(run.id)
        manifest["prompt_hashes"]["1"] = "changed"
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE benchmark_runs SET experiment_manifest=$2 WHERE id=$1",
                run.id,
                json.dumps(manifest),
            )
    else:

        async def failed_load(token, deadline):
            row = await claims.saved(w, run.id)
            assert row["inference_started"] and row["inference_attempt_count"] == 1
            native.events.append("native load request")
            raise OcrRuntimeChangedError("fixture_during_qualification")

        native.load = failed_load

    await benchmark._execute(dict(await benchmark._db.lease_task()))

    if scenario == "semantic-drift":
        assert native.events == []
        assert await claims.saved(w, run.id) is None
        assert (await benchmark._db.results(run.id))[0]["metadata"]["inference_attempt_count"] == 0
    else:
        assert native.events == ["claim", "native load request", "park"]
        row = await claims.saved(w, run.id)
        assert row["status"] == "retrying" and row["inference_attempt_count"] == 1
        assert (await benchmark._db.summaries(run.id))[0].completed == 0
    llm._client.chat.assert_not_awaited()


@pytest.mark.parametrize("change", ["rebind", "restarted_endpoint"])
async def test_frozen_benchmark_managed_topology_drift_keeps_reservation(
    workflow_db,
    tmp_path,
    managed,
    monkeypatch,
    change,
):
    w, scope = workflow_db
    llm, native, _, _, _, benchmark = managed()
    await adopt(llm.handoff, ("fixture",))
    run = await queued_benchmark(w, scope, tmp_path, benchmark)
    manifest = await benchmark._db.manifest(run.id)
    expected_reasons = []

    if change == "rebind":
        inspect = ModelHandoff.inspect

        async def inspect_then_readopt(coordinator, **kwargs):
            status = await inspect(coordinator, **kwargs)
            assert status.state == "ready"
            await adopt(coordinator, ("fixture", "replacement"))
            return status

        monkeypatch.setattr(ModelHandoff, "inspect", inspect_then_readopt)
    else:
        llm, _, _, _, _, benchmark = managed(native)
        llm._client._client.base_url = "http://replacement.invalid"
        llm.lifecycle_identity = "ollama:restarted-fixture"

    hold = benchmark._db.hold_before_inference

    async def record_hold(task, reason, *, deadline=None):
        expected_reasons.append(reason)
        return await hold(task, reason, deadline=deadline)

    benchmark._db.hold_before_inference = record_hold
    task = await benchmark._db.lease_task()
    await benchmark._execute(dict(task))

    held = await claims.saved(w, run.id)
    assert held is not None and held["status"] == "waiting_for_model"
    assert held["claim_token"] is None and held["leased_until"] is None
    assert not held["inference_started"] and held["inference_attempt_count"] == 0
    assert expected_reasons == ["execution_policy_changed"]
    assert await benchmark._db.manifest(run.id) == manifest
    assert await benchmark._db.results(run.id) == []
    assert (await benchmark._db.summaries(run.id))[0].completed == 0
    assert native.events == [] and not llm.handoff.journal.exists()
    llm._client.chat.assert_not_awaited()


async def test_upload_session_stalled_probe_refunds_live_claim(
    workflow_db, settings, tmp_path, managed
):
    w, scope = workflow_db
    llm, native, _, submissions, scheduler, _ = managed()
    await adopt(llm.handoff, ("fixture",))
    snapshot = frozen(settings, llm.handoff)
    image = await upload(w, scope, tmp_path, snapshot)
    reached = asyncio.Event()
    ready_for = submissions.ready_for

    async def stall(**kwargs):
        reached.set()
        await asyncio.Event().wait()

    async def ready_then_stall(candidate):
        ready = await ready_for(candidate)
        assert ready
        llm.resident_model_ids = stall
        return ready

    submissions.ready_for = ready_then_stall
    await scheduler.process_pending()

    assert reached.is_set()
    row = await assert_upload_held(w, scope, image, "model_handoff_residency_unknown")
    assert json.loads(row["effective_options"]) == snapshot
    assert native.events == [] and not llm.handoff.journal.exists()
    llm._client.chat.assert_not_awaited()
