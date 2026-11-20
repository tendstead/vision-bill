"""Exercise the awaited producer checkpoint and the real completion worker together."""

import hashlib
import json
from decimal import Decimal
from time import monotonic
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from ollama import ProcessResponse
from PIL import Image
from test_extraction_pipeline import sidecar
from test_gpu_handoff import enable_managed_ocr
from test_handoff_binding import adopt
from test_observation_enrichment import observation_provider

from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.workflow import AttemptOutcome, SubmissionCreate
from vision_bill.service.analysis_scheduler import AnalysisScheduler
from vision_bill.service.gpu_handoff import HandoffOwner
from vision_bill.service.image_service import ImageService
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.service.submission_service import SubmissionService

from . import test_workflow_db_e2e as shared

postgres, db, workflow_db = shared.postgres, shared.db, shared.workflow_db
pytestmark = pytest.mark.e2e


async def prepare_last_attempt(w, scope, receipts, settings, tmp_path):
    paths = [tmp_path / f"original-{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(path)
    image = (
        await w.enqueue_submission(
            SubmissionCreate(
                id=uuid4(),
                parts=[ImagePart(image_path=str(path)) for path in paths],
                source_manifest=[
                    {
                        "photo_index": i,
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for i, path in enumerate(paths)
                ],
                requested_options={"model_id": "fixture"},
                effective_options=receipts.current_execution_snapshot(settings),
                model_id="fixture",
                request_fingerprint=uuid4().hex,
            ),
            scope=scope,
            idempotency_key=None,
        )
    ).submission
    for _ in range(2):
        claim = await w.claim_next(image_id=image.id)
        assert claim is not None
        await w.finish_attempt(
            claim, AttemptOutcome(kind="retry", result_code="timed_out", retry_delay_seconds=0)
        )
    claim = await w.claim_next(image_id=image.id)
    assert claim is not None and claim.attempt_number == 3
    return image, claim


async def test_awaited_producer_checkpoint_resumes_after_uncertain_managed_classification(
    workflow_db,
    settings,
    tmp_path,
    monkeypatch,
):
    w, scope = workflow_db
    llm = observation_provider()
    enable_managed_ocr(monkeypatch)
    settings.ocr = OcrSettings(enabled=True, coordination_path=tmp_path / "owned-handoff.lock")
    native = sidecar()
    native.managed_lifecycle = True
    native.claim, native.park, native.load = AsyncMock(), AsyncMock(), AsyncMock()
    receipts = ReceiptService(settings.images, settings.pg, llm, native, settings.ocr)
    receipts._workflow_db = w
    receipts.list_tags = AsyncMock(return_value=[])
    submission = SubmissionService(settings, receipts, ImageService(settings.images))
    handoff = receipts.execution_handoff(settings.ocr)
    await adopt(handoff, ("fixture",))

    image, claim = await prepare_last_attempt(w, scope, receipts, settings, tmp_path)

    resident = False
    formatted = llm._client.chat.return_value
    checkpoint_ids = []

    async def chat(**request):
        nonlocal resident
        resident = True
        (message,) = request["messages"]
        assert "images" not in message
        if not message["content"].startswith("Classify these ordered"):
            assert "PHOTO 0" in message["content"] and "PHOTO 1" in message["content"]
            return formatted
        saved = await w.get_observation_checkpoint(claim)
        assert saved is not None and saved.envelope.receipt.total == Decimal("4.00")
        assert saved.envelope.classification_enrichment.state == "pending"
        assert len(saved.envelope.ocr_pages) == len(saved.persisted_views) == 2
        checkpoint_ids.append(saved.checkpoint_id)
        raise httpx.ReadTimeout("synthetic loss of classifier response")

    async def residency():
        return ProcessResponse(models=[{"model": "fixture:latest"}] if resident else [])

    async def release(**request):
        nonlocal resident
        assert request == {"model": "fixture", "keep_alive": 0}
        resident = False

    llm._client.chat.side_effect = chat
    llm._client.ps.side_effect = residency
    llm._client.generate = AsyncMock(side_effect=release)
    result = await submission.process_claim(claim)
    assert result.submission.status == "pending" and result.submission.attempt_count == 3
    (checkpoint_id,) = checkpoint_ids
    async with w.pool.acquire() as conn:
        producer = await conn.fetchrow(
            "SELECT * FROM receipt_extractions WHERE image_id=$1 AND attempt_number=3", image.id
        )
    assert producer["outcome_kind"] == "retry"
    journal = handoff.read_journal()
    assert journal.unsettled and journal.models == ["fixture"]
    counts = (
        llm._client.chat.await_count,
        llm.get_available_models.await_count,
        native.extract_page.await_count,
        native.get_capabilities.await_count,
        native.load.await_count,
        receipts.list_tags.await_count,
    )
    assert counts[0] == 2 and counts[2] == 2

    scheduler = AnalysisScheduler(settings, llm, receipts, None, submission_service=submission)
    assert await scheduler.process_pending() == []
    held = await w.get_submission(image.id, scope=scope)
    assert held.wait_reason == "model_request_unsettled" and held.receipt_id is None
    assert llm._client.generate.await_count == 0

    # The software double now represents external confirmation that the request settled.
    recovered = HandoffOwner(handoff, monotonic() + 10)
    recovered.models = journal.models
    recovered.generation = journal.generation
    recovered.ocr_pending = journal.ocr_pending
    recovered.save()
    finished = await scheduler.process_pending(scope=scope)
    assert [item.status for item in finished] == ["analyzed"]
    assert await scheduler.process_pending() == []
    assert counts == (
        llm._client.chat.await_count,
        llm.get_available_models.await_count,
        native.extract_page.await_count,
        native.get_capabilities.await_count,
        native.load.await_count,
        receipts.list_tags.await_count,
    )
    assert llm._client.generate.await_count == 1
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchrow("SELECT * FROM receipt_extractions WHERE id=$1", producer["id"])
            == producer
        )
        receipts = await conn.fetch(
            "SELECT total,extraction_id FROM receipts WHERE image_id=$1", image.id
        )
        assert [tuple(row) for row in receipts] == [(Decimal("4.00"), producer["id"])]
        events = json.loads(
            await conn.fetchval("SELECT legacy_events FROM images WHERE id=$1", image.id)
        )
        assert [
            event["checkpoint_id"] for event in events if event["event"] == "checkpoint_completed"
        ] == [str(checkpoint_id)]
    final = await w.get_submission(image.id, scope=scope)
    assert final.attempt_count == 3 and final.generation == 1 and not final.observations_available


@pytest.mark.parametrize(
    "uncertain", [True, False], ids=["unknown-settlement", "response-returned"]
)
async def test_unmanaged_producer_durably_fences_dispatch_and_recovers_once(
    workflow_db,
    settings,
    tmp_path,
    uncertain,
):
    from test_observation_enrichment import chat_reply

    w, scope = workflow_db
    llm = observation_provider()
    settings.ocr = OcrSettings(enabled=False, manage_model_lifecycle=False)
    receipts = ReceiptService(settings.images, settings.pg, llm, ocr_settings=settings.ocr)
    receipts._workflow_db = w
    receipts.list_tags = AsyncMock(return_value=[])
    submission = SubmissionService(settings, receipts, ImageService(settings.images))
    image, claim = await prepare_last_attempt(w, scope, receipts, settings, tmp_path)
    formatted = llm._client.chat.return_value
    checkpoint_ids = []

    async def chat(**request):
        (message,) = request["messages"]
        if not message["content"].startswith("Classify these ordered"):
            assert len(message["images"]) == 2
            return formatted
        assert "images" not in message
        checkpoint = await w.get_observation_checkpoint(claim)
        assert checkpoint is not None and len(checkpoint.persisted_views) == 2
        assert checkpoint.envelope.receipt.total == Decimal("4.00")
        checkpoint_ids.append(checkpoint.checkpoint_id)
        status = await w.unmanaged_classification_status(
            image.id, inference_identity=llm.inference_identity
        )
        assert status == {
            "state": "busy",
            "image_id": str(image.id),
            "checkpoint_id": str(checkpoint.checkpoint_id),
            "generation": 1,
        }
        if uncertain:
            raise TimeoutError("classification_request_unsettled")
        return chat_reply('{"category":"grocery","line_items":[{}]}')

    llm._client.chat.side_effect = chat
    produced = await submission.process_claim(claim)
    assert produced.submission.attempt_count == 3
    (checkpoint_id,) = checkpoint_ids
    scheduler = AnalysisScheduler(settings, llm, receipts, None, submission_service=submission)
    counts = (
        llm._client.chat.await_count,
        llm.get_available_models.await_count,
        receipts.list_tags.await_count,
    )
    assert counts[0] == 2
    if uncertain:
        assert (
            produced.submission.status == "pending"
            and produced.submission.result_code == "timed_out"
        )
        assert await scheduler.process_pending(scope=scope) == []
        held = await w.get_submission(image.id, scope=scope)
        assert held.wait_reason == "model_request_unsettled" and held.receipt_id is None
        assert await w.reconcile_unmanaged_classification(
            image.id,
            checkpoint_id,
            expected_generation=1,
            inference_identity=llm.inference_identity,
            request_settled=True,
        )
        assert [item.status for item in await scheduler.process_pending(scope=scope)] == [
            "analyzed"
        ]
    else:
        assert produced.submission.status == "analyzed"
    assert await scheduler.process_pending(scope=scope) == []
    assert counts == (
        llm._client.chat.await_count,
        llm.get_available_models.await_count,
        receipts.list_tags.await_count,
    )
    final = await w.get_submission(image.id, scope=scope)
    assert final.attempt_count == 3 and final.generation == 1 and not final.observations_available
    async with w.pool.acquire() as conn:
        saved = await conn.fetchrow("SELECT total FROM receipts WHERE image_id=$1", image.id)
        assert saved["total"] == Decimal("4.00")
        history = json.loads(
            await conn.fetchval("SELECT legacy_events FROM images WHERE id=$1", image.id)
        )
    assert [
        event["settlement"]
        for event in history
        if event["event"] == "unmanaged_classification_settled"
    ] == ["operator_attested" if uncertain else "provider_returned"]
    assert [
        event["checkpoint_id"] for event in history if event["event"] == "checkpoint_completed"
    ] == ([str(checkpoint_id)] if uncertain else [])
