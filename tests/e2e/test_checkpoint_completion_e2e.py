"""Real PostgreSQL publication and fencing of saved observations."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pytest
from PIL import Image
from test_gpu_handoff import Ocr
from test_handoff_binding import ResidentLlm, adopt
from test_observation_checkpoint import observation_payload

from vision_bill.config import OcrSettings
from vision_bill.model import workflow
from vision_bill.model.db.image import ImagePart
from vision_bill.provider.db import workflow_db as provider
from vision_bill.service.analysis_scheduler import AnalysisScheduler
from vision_bill.service.execution_policy import current_snapshot, semantic_policy_sha256
from vision_bill.service.gpu_handoff import HandoffOwner
from vision_bill.service.image_service import ImageService
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.service.submission_service import SubmissionService

from . import test_workflow_db_e2e as shared

postgres, db, workflow_db = shared.postgres, shared.db, shared.workflow_db
pytestmark = pytest.mark.e2e
ENDPOINT = "openai:" + "d" * 64


class NoInference:
    inference_identity = ENDPOINT
    handoff = None

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError(f"Unexpected provider operation: {name}")


async def produce_checkpoint(
    w,
    scope,
    settings,
    tmp_path,
    *,
    ocr=False,
    expired=False,
    managed=None,
    terminal=False,
    active=False,
    historical=False,
    contract_versions=None,
):
    images = ImageService(settings.images)
    if managed is None:
        settings.ocr = OcrSettings(enabled=ocr, manage_model_lifecycle=False)
        llm, native, handoff = NoInference(), None, None
    else:
        llm, native = managed
        settings.ocr = OcrSettings(enabled=True, coordination_path=tmp_path / "handoff.lock")
        handoff = None
    receipts = ReceiptService(settings.images, settings.pg, llm, native, settings.ocr)
    receipts._workflow_db = w
    service = SubmissionService(settings, receipts, images)
    if managed is not None:
        handoff = receipts.execution_handoff(settings.ocr)
        await adopt(handoff, ("test",))
    paths = [tmp_path / f"original-{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        Image.new("RGB", (120, 240), (i, 0, 0)).save(path)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    snapshot = current_snapshot(settings, inference_identity=ENDPOINT, handoff=handoff)
    if historical:
        snapshot.pop("inference_identity")
        snapshot["execution_policy_version"] = 3
    image = (
        await w.enqueue_submission(
            workflow.SubmissionCreate(
                id=uuid4(),
                parts=[ImagePart(image_path=str(path)) for path in paths],
                source_manifest=[
                    {"photo_index": i, "source_sha256": digest} for i, digest in enumerate(hashes)
                ],
                requested_options={"model_id": "test"},
                effective_options=snapshot,
                model_id="test",
                request_fingerprint=uuid4().hex,
            ),
            scope=scope,
            idempotency_key=None,
        )
    ).submission
    for _ in range(2):
        claim = await w.claim_next(image_id=image.id)
        await w.finish_attempt(
            claim,
            workflow.AttemptOutcome(kind="retry", result_code="timed_out", retry_delay_seconds=0),
        )
    claim = await w.claim_next(image_id=image.id)
    assert claim.attempt_number == 3
    parts = await w.get_source_parts(image.id, scope=scope)
    payload = observation_payload(parts=[part.model_dump() for part in parts], ocr=ocr)
    if contract_versions is not None:
        payload["execution_identity"].update(contract_versions)
    payload["producer_attempt_number"] = 3
    payload["execution_identity"]["semantic_policy_sha256"] = semantic_policy_sha256(snapshot)
    payload["envelope"]["classification_enrichment"]["missing_fields"] = ["category"]
    for path, view in zip(paths, payload["persisted_views"], strict=True):
        artifact = images._working_path(image.id, view["metadata"]["view_id"])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(path.read_bytes())
        view["artifact_path"] = str(artifact)
        view["metadata"]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    checkpoint = workflow.ObservationCheckpoint.model_validate(payload)
    assert await w.save_observation_checkpoint(claim, checkpoint)
    if expired:
        await expire(w, image.id)
        assert await w.reap_expired_claims() == 1
    elif not active:
        lost = await w.finish_attempt(
            claim,
            workflow.AttemptOutcome(
                kind="terminal" if terminal else "retry",
                result_code="source_integrity_failed" if terminal else "timed_out",
            ),
        )
        assert lost.submission.observations_available
    return claim, checkpoint, service, llm


async def expire(w, image_id):
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp() WHERE id=$1", image_id
        )


async def state(w, image_id):
    async with w.pool.acquire() as conn:
        producer = await conn.fetchrow(
            "SELECT * FROM receipt_extractions WHERE image_id=$1 AND attempt_number=3", image_id
        )
        count = await conn.fetchval("SELECT count(*) FROM receipts WHERE image_id=$1", image_id)
        events = await conn.fetchval("SELECT legacy_events FROM images WHERE id=$1", image_id)
        return dict(producer), count, json.loads(events)


@pytest.mark.parametrize("historical", [False, True], ids=["current", "historical"])
@pytest.mark.parametrize("ocr", [False, True], ids=["direct", "all-photo-ocr"])
async def test_last_attempt_completes_from_saved_observations(workflow_db, settings, tmp_path, ocr, historical):
    w, scope = workflow_db
    original, checkpoint, service, llm = await produce_checkpoint(
        w, scope, settings, tmp_path, ocr=ocr, expired=ocr, historical=historical,
    )
    image_id = original.image.id
    producer_before, _, _ = await state(w, image_id)
    assert await w.claim_next(image_id=image_id) is None
    assert not await w.due_candidates()
    scheduler = AnalysisScheduler(settings, llm, service.receipts, None, submission_service=service)
    results = await scheduler.process_pending(scope=scope)
    assert [result.status for result in results] == ["analyzed"]
    saved = await w.get_submission(image_id, scope=scope)
    assert (
        saved.attempt_count == 3 and not saved.observations_available and not saved.draft_available
    )
    assert llm.calls == []
    producer_after, count, events = await state(w, image_id)
    assert producer_after == producer_before and count == 1
    assert [
        event["checkpoint_id"] for event in events if event["event"] == "checkpoint_completed"
    ] == [str(checkpoint.checkpoint_id)]
    assert producer_before["outcome_kind"] == "retry"
    assert (
        json.loads(producer_before["envelope"])["classification_enrichment"]["state"] == "pending"
    )
    assert await w.get_submission(image_id, scope=workflow.AccessScope(user_id=uuid4())) is None
    async with w.pool.acquire() as conn:
        review = await conn.fetchval(
            "SELECT snapshot FROM receipt_review_revisions WHERE receipt_id=$1 AND operation_revision=1",
            saved.receipt_id,
        )
        assert json.loads(review)["provenance"] == checkpoint.envelope.provenance.model_dump(
            mode="json"
        )
    db = w._receipts
    db._pool = w.pool
    assert await db.get_receipt_with_details(saved.receipt_id, user_id=uuid4()) is None
    detail = await db.get_receipt_with_details(saved.receipt_id, user_id=scope.user_id)
    assert detail.receipt.extraction_id == producer_before["id"]
    assert detail.classification_enrichment.model_dump() == {
        "state": "unavailable",
        "reason_code": "recovered_without_classification",
        "missing_fields": ["category"],
    }


async def test_completion_race_lost_ack_and_replay_are_idempotent(
    workflow_db, settings, tmp_path, monkeypatch
):
    w, scope = workflow_db
    original, checkpoint, service, llm = await produce_checkpoint(w, scope, settings, tmp_path)
    image_id = original.image.id
    claims = await asyncio.gather(
        *(w.claim_next(image_id=image_id, purpose="completion") for _ in range(2))
    )
    (claim,) = [claim for claim in claims if claim is not None]
    assert (
        claim.claim_token != original.claim_token
        and claim.checkpoint_id == checkpoint.checkpoint_id
    )
    stale = claim.model_copy(update={"claim_token": uuid4()})
    assert (await service.complete_checkpoint(stale)).stale
    foreign = claim.model_copy(
        update={"image": claim.image.model_copy(update={"user_id": uuid4()})}
    )
    assert (await service.complete_checkpoint(foreign)).stale
    connection = provider.connection

    @asynccontextmanager
    async def lose_ack(*args, **kwargs):
        async with connection(*args, **kwargs) as conn:
            yield conn
        raise ConnectionError("completion acknowledgement lost")

    finish = w.finish_checkpoint_completion

    async def lose_finish_ack(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(provider, "connection", lose_ack)
            return await finish(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(w, "finish_checkpoint_completion", lose_finish_ack)
        with pytest.raises(ConnectionError, match="completion acknowledgement lost"):
            await service.complete_checkpoint(claim)
    result = await service.complete_checkpoint(claim)
    assert not result.stale and result.submission.receipt_id == result.receipt.id
    assert (await service.complete_checkpoint(stale)).stale
    assert await w.claim_next(image_id=image_id, purpose="completion") is None
    _, count, events = await state(w, image_id)
    assert count == 1
    assert sum(event["event"] == "checkpoint_completed" for event in events) == 1
    assert llm.calls == []


async def test_expired_completion_lease_keeps_producer_and_attempt_budget(
    workflow_db, settings, tmp_path
):
    w, scope = workflow_db
    original, _, service, _ = await produce_checkpoint(w, scope, settings, tmp_path)
    image_id = original.image.id
    first = await w.claim_next(image_id=image_id, purpose="completion")
    before = await state(w, image_id)
    await expire(w, image_id)
    assert (await service.complete_checkpoint(first)).stale
    assert await w.reap_expired_claims() == 1
    held = await w.get_submission(image_id, scope=scope)
    assert held.status == "pending" and held.attempt_count == 3
    assert held.wait_reason == "observation_recovery_required" and held.result_code == "timed_out"
    assert await state(w, image_id) == before
    second = await w.claim_next(image_id=image_id, purpose="completion")
    assert second.claim_token != first.claim_token
    assert (await service.complete_checkpoint(first)).stale
    assert (await service.complete_checkpoint(second)).submission.status == "analyzed"


@pytest.mark.parametrize("obstacle", ["original", "view", "policy", "human_edit"])
async def test_completion_blocks_if_current_evidence_or_ownership_changed(
    workflow_db, settings, tmp_path, obstacle
):
    w, scope = workflow_db
    original, checkpoint, service, _ = await produce_checkpoint(w, scope, settings, tmp_path)
    image_id = original.image.id
    claim = await w.claim_next(image_id=image_id, purpose="completion")
    if obstacle == "original":
        Path(checkpoint.source_parts[1].image_path).write_bytes(b"changed original")
    elif obstacle == "view":
        Path(checkpoint.persisted_views[1].artifact_path).write_bytes(b"changed working view")
    elif obstacle == "policy":
        settings.ocr = OcrSettings(enabled=True)
    else:
        async with w.pool.acquire() as conn:
            producer, _, _ = await state(w, image_id)
            await conn.execute(
                """INSERT INTO receipt_drafts(image_id,extraction_id,generation,document,
                amount_policy,findings,field_sources,human_edited)
                VALUES($1,$2,1,$3::jsonb,$4::jsonb,'[]','{}',TRUE)""",
                image_id,
                producer["id"],
                checkpoint.envelope.draft.model_dump_json(),
                checkpoint.envelope.amount_policy.model_dump_json(),
            )
    result = await service.complete_checkpoint(claim)
    assert result.submission.status == "pending" and result.submission.attempt_count == 3
    assert (
        result.submission.wait_reason
        == {
            "original": "source_integrity_failed",
            "view": "observation_checkpoint_invalid",
            "policy": "execution_policy_changed",
            "human_edit": "observation_checkpoint_invalid",
        }[obstacle]
    )
    assert await w.get_draft(image_id, scope=scope) is None
    assert (await state(w, image_id))[1] == 0


@pytest.mark.parametrize("blocker", ["lease", "deadline"])
async def test_mid_publication_fence_rolls_back_receipt_and_event(
    workflow_db, settings, tmp_path, monkeypatch, blocker
):
    w, scope = workflow_db
    producing, _, service, _ = await produce_checkpoint(w, scope, settings, tmp_path)
    image_id = producing.image.id
    claim = await w.claim_next(image_id=image_id, purpose="completion")
    before = await state(w, image_id)
    finish = w.finish_checkpoint_completion

    async def finish_before_deadline(*args, **kwargs):
        kwargs["deadline"] = asyncio.get_running_loop().time() + 1
        return await finish(*args, **kwargs)

    async with w.pool.acquire() as lock:
        await lock.execute("""CREATE FUNCTION delay_completion_snapshot() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_advisory_xact_lock(1120); RETURN NEW; END $$""")
        await lock.execute("""CREATE TRIGGER delay_completion_snapshot BEFORE INSERT ON receipt_review_revisions
            FOR EACH ROW EXECUTE FUNCTION delay_completion_snapshot()""")
        if blocker == "lease":
            await lock.execute(
                "UPDATE images SET lease_expires_at=clock_timestamp()+interval '1 second' WHERE id=$1",
                image_id,
            )
        else:
            monkeypatch.setattr(w, "finish_checkpoint_completion", finish_before_deadline)
        async with lock.transaction():
            await lock.execute("SELECT pg_advisory_xact_lock(1120)")
            worker = asyncio.create_task(service.complete_checkpoint(claim))
            async with w.pool.acquire() as observer, asyncio.timeout(3):
                while not await observer.fetchval("""SELECT EXISTS(SELECT 1 FROM pg_stat_activity
                    WHERE wait_event_type='Lock' AND query LIKE 'INSERT INTO receipt_review_revisions%')"""):
                    assert not worker.done(), "completion never reached initial receipt snapshot"
                    await asyncio.sleep(0.01)
            await lock.execute("SELECT pg_sleep(1)")
        result = await worker
    if blocker == "lease":
        assert result.stale
        assert await w.reap_expired_claims() == 1
    else:
        assert result.submission.wait_reason == "observation_recovery_required"
    assert await state(w, image_id) == before
    saved = await w.get_submission(image_id, scope=scope)
    assert saved.status == "pending" and saved.attempt_count == 3 and saved.observations_available


async def test_managed_checkpoint_waits_for_safe_owned_recovery(
    workflow_db, settings, tmp_path, monkeypatch
):
    from vision_bill.service import gpu_handoff

    class ManagedLlm(ResidentLlm):
        inference_identity = ENDPOINT
        handoff = None

        async def release_model(self, model):
            await super().release_model(model)
            self.resident -= {model}

    class ParkOnlyOcr(Ocr):
        managed_lifecycle = True

        async def get_capabilities(self):
            pytest.fail("Completion must not discover OCR capabilities")

    monkeypatch.setattr(gpu_handoff, "managed_ocr_enabled", lambda: True)
    monkeypatch.setattr("vision_bill.service.submission_service.managed_ocr_enabled", lambda: True)
    w, scope = workflow_db
    events = []
    llm = ManagedLlm(events)
    original, _, service, _ = await produce_checkpoint(
        w,
        scope,
        settings,
        tmp_path,
        ocr=True,
        managed=(llm, ParkOnlyOcr(events)),
    )
    image_id = original.image.id
    handoff = service.receipts.execution_handoff(settings.ocr)
    abandoned = HandoffOwner(handoff, monotonic() + 10)
    abandoned.models = ["test"]
    abandoned.unsettled = True
    abandoned.save()
    llm.resident = frozenset({"test"})
    events.clear()
    scheduler = AnalysisScheduler(settings, llm, service.receipts, None, submission_service=service)
    assert await scheduler.process_pending() == []
    assert events == []
    held = await w.get_submission(image_id, scope=scope)
    assert held.wait_reason == "model_request_unsettled" and held.attempt_count == 3

    # This deterministic fixture models outside confirmation that the request settled.
    abandoned.unsettled = False
    abandoned.save()
    llm.fail = True
    await scheduler.process_pending()
    assert (await state(w, image_id))[1] == 0
    assert handoff.read_journal().models == ["test"]
    assert "load_ocr" not in events

    llm.fail = False
    events.clear()
    results = await scheduler.process_pending()
    assert [result.status for result in results] == ["analyzed"]
    assert events == ["residency", "residency", "claim_park", ("unload", "test"), "residency"]
    assert not handoff.read_journal().models


@pytest.mark.parametrize("contract_versions", [
    {"prompt_version": "4"}, {"extraction_schema_version": "1"},
    {"prompt_version": "4", "extraction_schema_version": "1"},
    {"prompt_version": "5", "extraction_schema_version": "2"},
])
async def test_prior_formatter_checkpoint_holds_until_explicit_new_generation(
    workflow_db, settings, tmp_path, contract_versions,
):
    w, scope = workflow_db
    original, checkpoint, service, llm = await produce_checkpoint(
        w, scope, settings, tmp_path, contract_versions=contract_versions,
    )
    image_id = original.image.id
    scheduler = AnalysisScheduler(settings, llm, service.receipts, None, submission_service=service)
    await scheduler.process_pending(scope=scope)
    held = await w.get_submission(image_id, scope=scope)
    assert held.wait_reason == "execution_policy_changed" and held.observations_available
    assert held.generation == 1 and held.attempt_count == 3 and llm.calls == []
    retried = await w.retry_submission(
        image_id, workflow.RetryRequest(expected_generation=1, request_key=uuid4().hex, policy="current"),
        scope=scope, replacement_snapshot=service.current_snapshot(),
    )
    assert retried.generation == 2 and retried.attempt_count == 0 and not retried.observations_available
    _, count, events = await state(w, image_id)
    retired = next(event["checkpoint"] for event in events if event["event"] == "observation_checkpoint_retired")
    assert retired["execution_identity"] == checkpoint.execution_identity.model_dump()
    assert count == 0 and llm.calls == []


async def test_terminal_producer_can_complete_after_exact_original_restoration(
    workflow_db, settings, tmp_path
):
    w, scope = workflow_db
    original, checkpoint, service, llm = await produce_checkpoint(
        w, scope, settings, tmp_path, terminal=True
    )
    image_id = original.image.id
    path = Path(checkpoint.source_parts[1].image_path)
    previous = path.read_bytes()
    path.write_bytes(b"changed original")
    scheduler = AnalysisScheduler(settings, llm, service.receipts, None, submission_service=service)
    first = await scheduler.process_pending(scope=scope)
    assert [item.status for item in first] == ["pending"]
    held = await w.get_submission(image_id, scope=scope)
    assert held.wait_reason == "source_integrity_failed" and held.attempt_count == 3
    producer, count, _ = await state(w, image_id)
    assert producer["outcome_kind"] == "terminal" and count == 0
    path.write_bytes(previous)
    completed = await scheduler.process_pending(scope=scope)
    assert [item.status for item in completed] == ["analyzed"]
    final = await w.get_submission(image_id, scope=scope)
    assert final.generation == 1 and final.attempt_count == 3 and llm.calls == []
    after, count, _ = await state(w, image_id)
    assert after == producer and count == 1


async def test_working_view_corrupted_during_managed_release_blocks_publication(
    workflow_db, settings, tmp_path, monkeypatch
):
    from vision_bill.service import gpu_handoff

    class ManagedLlm(ResidentLlm):
        inference_identity = ENDPOINT
        handoff = None

        async def release_model(self, model):
            await super().release_model(model)
            self.resident -= {model}
            path.write_bytes(b"working image changed during release")

    class ParkOnlyOcr(Ocr):
        managed_lifecycle = True

    monkeypatch.setattr(gpu_handoff, "managed_ocr_enabled", lambda: True)
    monkeypatch.setattr("vision_bill.service.submission_service.managed_ocr_enabled", lambda: True)
    w, scope = workflow_db
    events = []
    llm = ManagedLlm(events)
    original, checkpoint, service, _ = await produce_checkpoint(
        w,
        scope,
        settings,
        tmp_path,
        ocr=True,
        managed=(llm, ParkOnlyOcr(events)),
    )
    path = Path(checkpoint.persisted_views[1].artifact_path)
    image_id = original.image.id
    handoff = service.receipts.execution_handoff(settings.ocr)
    abandoned = HandoffOwner(handoff, monotonic() + 10)
    abandoned.models = ["test"]
    abandoned.save()
    llm.resident = frozenset({"test"})
    scheduler = AnalysisScheduler(settings, llm, service.receipts, None, submission_service=service)
    results = await scheduler.process_pending(scope=scope)
    assert [item.status for item in results] == ["pending"]
    assert ("unload", "test") in events
    held = await w.get_submission(image_id, scope=scope)
    assert held.wait_reason == "observation_checkpoint_invalid" and held.attempt_count == 3
    _, count, history = await state(w, image_id)
    assert count == 0 and all(event["event"] != "checkpoint_completed" for event in history)


async def test_unmanaged_classification_requires_matching_operator_settlement(
    workflow_db, settings, tmp_path
):
    w, scope = workflow_db
    claim, checkpoint, service, llm = await produce_checkpoint(
        w, scope, settings, tmp_path, active=True
    )
    image_id, checkpoint_id = claim.image.id, checkpoint.checkpoint_id
    assert not await w.record_unmanaged_classification_event(claim, checkpoint_id, state="settled")
    assert not await w.record_unmanaged_classification_event(claim, uuid4(), state="started")
    assert await w.record_unmanaged_classification_event(claim, checkpoint_id, state="started")
    assert await w.record_unmanaged_classification_event(claim, checkpoint_id, state="started")
    assert not await w.reconcile_unmanaged_classification(
        image_id,
        checkpoint_id,
        expected_generation=1,
        inference_identity=ENDPOINT,
        request_settled=True,
    )
    failure = await w.finish_attempt(
        claim, workflow.AttemptOutcome(kind="retry", result_code="timed_out")
    )
    assert failure.submission.observations_available
    scheduler = AnalysisScheduler(settings, llm, service.receipts, None, submission_service=service)
    assert await scheduler.process_pending(scope=scope) == []
    held = await w.get_submission(image_id, scope=scope)
    assert held.wait_reason == "model_request_unsettled" and held.attempt_count == 3
    assert await w.claim_next(image_id=image_id, purpose="completion") is None
    status = await w.unmanaged_classification_status(image_id, inference_identity=ENDPOINT)
    assert status == {
        "state": "recovery_required",
        "image_id": str(image_id),
        "generation": 1,
        "checkpoint_id": str(checkpoint_id),
    }
    for target, generation, endpoint, attested in (
        (uuid4(), 1, ENDPOINT, True),
        (checkpoint_id, 2, ENDPOINT, True),
        (checkpoint_id, 1, "ollama:" + "e" * 64, True),
        (checkpoint_id, 1, ENDPOINT, False),
    ):
        assert not await w.reconcile_unmanaged_classification(
            image_id,
            target,
            expected_generation=generation,
            inference_identity=endpoint,
            request_settled=attested,
        )
    assert await w.reconcile_unmanaged_classification(
        image_id,
        checkpoint_id,
        expected_generation=1,
        inference_identity=ENDPOINT,
        request_settled=True,
    )
    assert await w.reconcile_unmanaged_classification(
        image_id,
        checkpoint_id,
        expected_generation=1,
        inference_identity=ENDPOINT,
        request_settled=True,
    )
    assert [item.status for item in await scheduler.process_pending(scope=scope)] == ["analyzed"]
    producer, count, history = await state(w, image_id)
    assert producer["result_code"] == "timed_out" and count == 1
    assert sum(event["event"] == "unmanaged_classification_started" for event in history) == 1
    assert sum(event.get("settlement") == "operator_attested" for event in history) == 1
    assert llm.calls == []


async def test_unmanaged_pending_marker_fences_normal_publication_and_explicit_retry(
    workflow_db, settings, tmp_path
):
    from vision_bill.model.extraction import InferenceEnvelope
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.service.receipt_review import evaluate_receipt

    w, scope = workflow_db
    claim, checkpoint, _, _ = await produce_checkpoint(w, scope, settings, tmp_path, active=True)
    assert await w.record_unmanaged_classification_event(
        claim, checkpoint.checkpoint_id, state="started"
    )
    envelope = InferenceEnvelope.model_validate(
        checkpoint.recovered_envelope().model_dump(exclude={"ocr_pages"})
    )
    originals = [
        OriginalAvailability(
            part_id=part.part_id, photo_index=part.photo_index, available=True, sha256_matches=True
        )
        for part in checkpoint.source_parts
    ]
    review = evaluate_receipt(
        envelope.receipt,
        envelope.amount_policy,
        envelope.field_sources,
        envelope.overlap_decisions,
        originals,
    )
    complete = workflow.AttemptOutcome(
        kind="complete",
        envelope=envelope,
        review=review,
        persisted_views=checkpoint.persisted_views,
    )
    result = await w.finish_attempt(claim, complete)
    assert result.stale
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM receipts WHERE image_id=$1", claim.image.id)
            == 0
        )
    assert await w.finish_attempt(
        claim, workflow.AttemptOutcome(kind="retry", result_code="timed_out")
    )
    with pytest.raises(ValueError, match="model_request_unsettled"):
        await w.retry_submission(
            claim.image.id,
            workflow.RetryRequest(
                expected_generation=1, request_key="no-generation-bypass", policy="current"
            ),
            scope=scope,
            replacement_snapshot=claim.image.effective_options,
        )
