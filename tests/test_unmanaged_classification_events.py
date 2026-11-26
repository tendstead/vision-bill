"""Observe the durable marker sequence through real upload extraction."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image
from test_experiment_scoring import receipt
from test_extraction_pipeline import formatter
from test_observation_enrichment import chat_reply, observation_provider
from test_submission_service import context

from vision_bill.model.experiment import ExperimentOptions
from vision_bill.service.experiment_execution import execute_submission
from vision_bill.service.receipt_service import ReceiptService


@pytest.mark.parametrize("scenario", [
    "malformed", "proxy_503", "classifier_timeout", "cancelled", "started_refused",
    "settled_refused", "started_ack_lost", "settled_ack_lost", "short_budget", "already_classified",
])
async def test_unmanaged_marker_fences_classifier_and_ambiguous_results(settings, tmp_path, scenario):
    service, _, db, claim, _ = context(settings, tmp_path)
    if scenario == "already_classified":
        raw = receipt().model_dump(mode="json")
        for row in raw["line_items"]:
            row["spending_category"] = "milk"
        llm = formatter(payload=raw)
    else:
        llm = observation_provider()
    receipts = ReceiptService(settings.images, settings.pg, llm)
    receipts.list_tags = AsyncMock(return_value=[])
    service.receipts = receipts
    claim.image.effective_options["inference_identity"] = receipts.inference_identity
    claim.image.model_id = "fixture"
    db.unreferenced_views = AsyncMock(return_value=[])
    if scenario in ("classifier_timeout", "short_budget"):
        seconds = 64 if scenario == "classifier_timeout" else 10
        service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": seconds})

    checkpoint_saved = []
    markers = []
    dispatched = asyncio.Event()
    formatted = llm._client.chat.return_value

    async def save_observations(claim_arg, checkpoint, *, deadline):
        assert claim_arg == claim
        checkpoint_saved.append((checkpoint.checkpoint_id, deadline))
        return True

    async def record(claim_arg, checkpoint_id, *, state, deadline):
        assert claim_arg == claim
        assert checkpoint_saved == [(checkpoint_id, deadline)]
        await asyncio.sleep(0)
        if scenario == f"{state}_refused":
            return False
        markers.append(state)
        if scenario == f"{state}_ack_lost":
            raise ConnectionError("marker acknowledgement lost")
        return True

    async def chat(**request):
        if request["messages"][0].get("images"):
            return formatted
        assert markers == ["started"]
        dispatched.set()
        if scenario == "proxy_503":
            proxy = httpx.Request("POST", "http://local/api/chat")
            raise httpx.HTTPStatusError(
                "ambiguous proxy response", request=proxy, response=httpx.Response(503, request=proxy),
            )
        if scenario in ("classifier_timeout", "cancelled"):
            await asyncio.Future()
        return chat_reply("{")

    db.save_observation_checkpoint = AsyncMock(side_effect=save_observations)
    db.record_unmanaged_classification_event = AsyncMock(side_effect=record)
    llm._client.chat.side_effect = chat

    if scenario == "cancelled":
        running = asyncio.create_task(service.process_claim(claim))
        try:
            async with asyncio.timeout(3):
                await dispatched.wait()
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        db.finish_attempt.assert_not_awaited()
        assert markers == ["started"]
        return

    result = await service.process_claim(claim)
    assert len(checkpoint_saved) == 1
    classifier_sent = scenario not in (
        "started_refused", "started_ack_lost", "short_budget", "already_classified",
    )
    assert dispatched.is_set() is classifier_sent
    assert llm._client.chat.await_count == (2 if classifier_sent else 1)
    if scenario in ("started_refused", "already_classified"):
        assert not markers
    elif scenario in ("malformed", "short_budget", "settled_ack_lost"):
        assert markers == ["started", "settled"]
    else:
        assert markers == ["started"]
    if scenario.endswith("_refused"):
        assert result.stale
        db.finish_attempt.assert_not_awaited()
    else:
        outcome = db.finish_attempt.await_args.args[1]
        states = {"malformed": "unavailable", "short_budget": "skipped", "already_classified": "completed"}
        if expected_state := states.get(scenario):
            assert outcome.envelope.classification_enrichment.state == expected_state
        else:
            assert outcome.kind == "retry" and outcome.envelope is None


async def test_benchmark_classification_uses_no_upload_markers(settings, tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    llm = observation_provider()
    llm._client.chat.side_effect = [llm._client.chat.return_value, chat_reply("{")]
    receipts = ReceiptService(settings.images, settings.pg, llm)
    record = AsyncMock(side_effect=AssertionError("benchmark must not mark an upload claim"))
    receipts.workflow_db.record_unmanaged_classification_event = record
    result = await execute_submission(receipts, "fixture", [path], ExperimentOptions(), [])
    assert result.metadata["classification_enrichment"]["state"] == "unavailable"
    assert llm._client.chat.await_count == 2
    record.assert_not_awaited()


@pytest.mark.parametrize("shared_gpu", [False, True])
async def test_managed_ocr_checkpoint_records_unmanaged_formatter_settlement(settings, tmp_path, monkeypatch, shared_gpu):
    import json
    from contextlib import asynccontextmanager
    from uuid import uuid4

    from vision_bill.provider.db import workflow_db
    from vision_bill.service.execution_policy import current_snapshot, semantic_policy_sha256

    _, _, _, claim, _ = context(settings, tmp_path)
    settings.ocr.enabled = True
    settings.ocr.shared_gpu = shared_gpu
    snapshot = current_snapshot(settings, inference_identity="ollama:" + "d" * 64)
    checkpoint_id = uuid4()
    row = {
        "id": claim.image.id, "generation": claim.image.generation, "attempt_count": claim.image.attempt_count,
        "receipt_id": None, "effective_options": snapshot, "legacy_events": [],
        "observation_checkpoint": {
            "checkpoint_id": str(checkpoint_id), "eligible_generation": claim.image.generation,
            "producer_generation": claim.image.generation, "producer_attempt_number": claim.image.attempt_count,
            "execution_identity": {"inference_identity": "openai:fixture",
                                   "semantic_policy_sha256": semantic_policy_sha256(snapshot)},
            "envelope": {"classification_enrichment": {"state": "pending"}},
        },
    }

    async def fetchval(_sql, image_id, events):
        assert image_id == row["id"]
        row["legacy_events"].extend(json.loads(events))
        return image_id

    @asynccontextmanager
    async def connection(*_args, **_kwargs):
        from types import SimpleNamespace

        yield SimpleNamespace(fetchval=fetchval)

    db = workflow_db.WorkflowDB(settings.pg)
    monkeypatch.setattr(db, "_pool", object())
    monkeypatch.setattr(db, "_lock_claim", AsyncMock(return_value=row))
    monkeypatch.setattr(workflow_db, "connection", connection)
    assert await db.record_unmanaged_classification_event(claim, checkpoint_id, state="started") is not shared_gpu
    assert await db.record_unmanaged_classification_event(claim, checkpoint_id, state="settled") is not shared_gpu
    events = [item["event"] for item in row["legacy_events"]]
    assert events == ([] if shared_gpu else ["unmanaged_classification_started", "unmanaged_classification_settled"])
