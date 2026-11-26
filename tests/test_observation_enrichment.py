"""All-photo checkpoints and optional labeling at the real provider boundaries."""

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from handoff_fixtures import install_owned_bindings
from ollama import ChatResponse, ProcessResponse
from openai import APIStatusError, APITimeoutError
from PIL import Image
from test_experiment_scoring import receipt
from test_extraction_pipeline import formatter, pipeline, sidecar
from test_gpu_handoff import enable_managed_ocr
from test_llamacpp_lifecycle import Ocr as RouterOcr
from test_llamacpp_lifecycle import configured, model_status
from test_submission_service import context

from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.experiment import ExperimentOptions
from vision_bill.model.workflow import PersistedSourcePart
from vision_bill.provider.llm.base import ModelInfo
from vision_bill.provider.llm.experiment import endpoint_fingerprint
from vision_bill.service.benchmark_scoring import PROMPT_VERSION, validation_schema_sha256
from vision_bill.service.benchmark_service import BenchmarkService
from vision_bill.service.experiment_execution import (
    ExperimentFailure,
    execute_submission,
    ocr_fingerprint,
    pipeline_identity,
)
from vision_bill.service.extraction_pipeline import ExtractionPipeline
from vision_bill.service.gpu_handoff import ModelHandoff, current_owner
from vision_bill.service.receipt_service import ReceiptService


def observation_provider():
    raw = receipt().model_dump(mode="json")
    del raw["category"]
    raw["line_items"][0]["spending_category"] = "milk"
    return formatter(payload=raw)


def chat_reply(content):
    return ChatResponse(message={"role": "assistant", "content": content})


@pytest.mark.parametrize("save_succeeds", [True, False])
@pytest.mark.parametrize("mode", ["direct_vision", "ocr"])
async def test_submission_checkpoints_all_views_before_text_labels(
    settings, tmp_path, save_succeeds, mode,
):
    service, _, db, claim, first = context(settings, tmp_path)
    second_path = tmp_path / "second.png"
    Image.new("RGB", (6, 7)).save(second_path)
    second = PersistedSourcePart(
        part_id=uuid4(), photo_index=1, image_path=str(second_path),
        source_sha256=hashlib.sha256(second_path.read_bytes()).hexdigest(),
    )
    claim.image.additional_images = [ImagePart(image_path=str(second_path))]
    db.bind_source_parts.return_value = [first, second]
    db.unreferenced_views = AsyncMock(return_value=[])
    llm = observation_provider()
    receipts = ReceiptService(settings.images, settings.pg, llm)
    if mode == "ocr":
        settings.ocr = OcrSettings(
            enabled=True, service_url="http://local-ocr", manage_model_lifecycle=False,
        )
        claim.image.effective_options["ocr"] = settings.ocr.model_dump(mode="json")
        receipts._ocr_settings = settings.ocr
        receipts._ocr_provider = sidecar()
    receipts.list_tags = AsyncMock(return_value=[])
    claim.image.user_id = uuid4()
    receipts.suggested_merchant_name = AsyncMock(return_value="Privater Hofladen")
    service.receipts = receipts
    claim.image.effective_options["inference_identity"] = receipts.inference_identity
    events = []

    async def save(claim_arg, checkpoint, *, deadline=None):
        await asyncio.sleep(0)
        assert events == []
        assert claim_arg == claim
        assert deadline > asyncio.get_running_loop().time()
        assert checkpoint.source_parts == [first, second]
        assert checkpoint.model_id == "fixture"
        assert checkpoint.envelope.receipt.total == Decimal("4.00")
        assert checkpoint.envelope.receipt.merchant_display_name is None
        assert checkpoint.envelope.draft.merchant_display_name is None
        assert checkpoint.envelope.draft.category is None
        assert checkpoint.envelope.classification_enrichment.state == "pending"
        assert checkpoint.envelope.provenance.used_pipeline == mode
        assert len(checkpoint.envelope.ocr_pages) == (2 if mode == "ocr" else 0)
        assert db.save_ocr_checkpoint.await_count == (2 if mode == "ocr" else 0)
        assert "png_base64" not in checkpoint.model_dump_json()
        assert [view.photo_index for view in checkpoint.persisted_views] == [0, 1]
        for view in checkpoint.persisted_views:
            assert Path(view.artifact_path).is_file()
            assert service.images.read_working_image(claim.image.id, view.artifact_path, view.metadata)
        if save_succeeds:
            events.append("checkpoint_committed")
        return save_succeeds

    db.save_observation_checkpoint = AsyncMock(side_effect=save)

    async def record(claim_arg, checkpoint_id, *, state, deadline=None):
        assert claim_arg == claim
        saved = db.save_observation_checkpoint.await_args
        assert checkpoint_id == saved.args[1].checkpoint_id
        assert deadline == saved.kwargs["deadline"]
        events.append(state)
        return True

    db.record_unmanaged_classification_event = AsyncMock(side_effect=record)
    formatted = llm._client.chat.return_value

    async def chat(**request):
        messages = request["messages"]
        prompt = messages[0]["content"]
        if not prompt.startswith("Classify these ordered"):
            if mode == "ocr":
                assert "PHOTO 0" in prompt and "PHOTO 1" in prompt
                assert all("images" not in message for message in messages)
            else:
                assert len(messages[0]["images"]) == 2
            return formatted
        assert "Privater Hofladen" in prompt
        events.append("classification_dispatched")
        assert all("images" not in message for message in messages)
        return chat_reply('{"category":"grocery","line_items":[{}]}')

    llm._client.chat.side_effect = chat
    claim.image.model_id = "fixture"
    result = await service.process_claim(claim)
    if save_succeeds:
        assert events == ["checkpoint_committed", "started", "classification_dispatched", "settled"]
        outcome = db.finish_attempt.call_args.args[1]
        assert outcome.envelope.receipt.total == Decimal("4.00")
        assert outcome.envelope.receipt.line_items[0].spending_category == "milk"
        assert outcome.envelope.classification_enrichment.state == "unavailable"
        assert len(outcome.persisted_views) == 2
        assert outcome.envelope.draft.merchant_display_name is None
        assert outcome.envelope.receipt.merchant_display_name == "Privater Hofladen"
        assert receipts.suggested_merchant_name.await_args.args == (claim.image.user_id, "Store", None)
    else:
        assert events == []
        assert result.stale
        receipts.suggested_merchant_name.assert_not_awaited()
        db.record_unmanaged_classification_event.assert_not_awaited()
        db.finish_attempt.assert_not_awaited()


def managed_pipeline(monkeypatch, tmp_path):
    llm = observation_provider()
    enable_managed_ocr(monkeypatch)
    install_owned_bindings(monkeypatch)
    ocr = sidecar()
    ocr.managed_lifecycle = True
    ocr.claim, ocr.park = AsyncMock(), AsyncMock()
    settings = OcrSettings(enabled=True, coordination_path=tmp_path / "handoff")
    return llm, ExtractionPipeline(llm, ocr, settings)


@pytest.mark.parametrize("release_succeeds", [True, False])
async def test_classification_local_timeout_requires_settlement_and_verified_release(
    monkeypatch, tmp_path, release_succeeds,
):
    llm, extract = managed_pipeline(monkeypatch, tmp_path)
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    entered = asyncio.Event()
    finish_response = asyncio.Event()
    events = []
    owners = []
    resident = False
    formatted = llm._client.chat.return_value
    record_upload_event = AsyncMock(side_effect=AssertionError("managed dispatch uses its journal"))

    async def checkpoint(observed):
        assert observed.input_views[0].photo_index == 0
        assert observed.receipt.total == Decimal("4.00")
        owners.append(current_owner())
        events.append("checkpoint_committed")

    async def chat(**request):
        nonlocal resident
        resident = True
        if request["messages"][0].get("images"):
            return formatted
        assert events == ["checkpoint_committed"]
        events.append("classification_dispatched")
        entered.set()
        await finish_response.wait()
        events.append("classification_settled")
        return chat_reply('{}')

    async def release(**request):
        nonlocal resident
        assert request == {"model": "fixture", "keep_alive": 0}
        assert events[-1] == "classification_settled"
        resident = not release_succeeds
        events.append("release_requested")

    async def residency():
        if "release_requested" in events:
            events.append("release_verified")
        return ProcessResponse(models=[{"model": "fixture:latest"}] if resident else [])

    llm._client.chat.side_effect = chat
    llm._client.generate = AsyncMock(side_effect=release)
    llm._client.ps.side_effect = residency
    deadline = monotonic() + 60.5

    async def finish_after_classification_deadline():
        await entered.wait()
        await asyncio.sleep(max(0, deadline - 60 - monotonic()) + 0.03)
        finish_response.set()

    async with asyncio.timeout(4), asyncio.TaskGroup() as tasks:
        responder = tasks.create_task(finish_after_classification_deadline())
        try:
            if release_succeeds:
                result = await extract.extract(
                    "fixture", [path], options=ExperimentOptions(pipeline="direct_vision"),
                    deadline=deadline, tags=[], observation_checkpoint=checkpoint,
                    unmanaged_classification_event=record_upload_event,
                )
                assert result.classification_enrichment.state == "unavailable"
                assert result.receipt.total == Decimal("4.00")
            else:
                with pytest.raises(RuntimeError, match="model_release_unconfirmed"):
                    await extract.extract(
                        "fixture", [path], options=ExperimentOptions(pipeline="direct_vision"),
                        deadline=deadline, tags=[], observation_checkpoint=checkpoint,
                        unmanaged_classification_event=record_upload_event,
                    )
        finally:
            responder.cancel()
    assert events == [
        "checkpoint_committed", "classification_dispatched", "classification_settled",
        "release_requested", "release_verified",
    ]
    assert owners[0] is not None and owners[0].request_sequence == 3
    record_upload_event.assert_not_awaited()


@pytest.mark.parametrize("failure", ["proxy", "transport_timeout", "cancel"])
async def test_ambiguous_classification_keeps_durable_fence(
    monkeypatch, tmp_path, failure,
):
    llm, extract = managed_pipeline(monkeypatch, tmp_path)
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    formatted = llm._client.chat.return_value
    entered = asyncio.Event()
    fail_response = asyncio.Event()
    checkpoints = []
    completed = []
    request = httpx.Request("POST", "http://local/api/chat")
    proxy_error = httpx.HTTPStatusError(
        "ambiguous proxy response", request=request, response=httpx.Response(503, request=request),
    )
    error = proxy_error if failure == "proxy" else httpx.ReadTimeout("in flight", request=request)

    async def checkpoint(observed):
        checkpoints.append(observed)

    async def chat(**request):
        if request["messages"][0].get("images"):
            return formatted
        assert checkpoints[0].classification_enrichment.state == "pending"
        entered.set()
        if failure == "cancel":
            await fail_response.wait()
        raise error

    async def run():
        completed.append(await extract.extract(
            "fixture", [path], options=ExperimentOptions(pipeline="direct_vision"),
            deadline=monotonic() + 90, tags=[], observation_checkpoint=checkpoint,
        ))

    llm._client.chat.side_effect = chat
    if failure == "cancel":
        task = asyncio.create_task(run())
        try:
            async with asyncio.timeout(3):
                await entered.wait()
                task.cancel()
                await asyncio.sleep(0)
                fail_response.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    else:
        with pytest.raises(type(error)):
            await run()
    journal = extract.coordinator().read_journal()
    assert journal.unsettled and journal.models == ["fixture"]
    assert len(checkpoints) == 1
    assert completed == []


async def test_benchmark_failure_metadata_retains_admitted_observations(settings, tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    llm = observation_provider()
    llm._client.chat.side_effect = [
        llm._client.chat.return_value, httpx.ReadTimeout("ambiguous classification"),
    ]
    service = ReceiptService(settings.images, settings.pg, llm)
    with pytest.raises(ExperimentFailure) as raised:
        await execute_submission(service, "fixture", [path], ExperimentOptions(), [])
    assert raised.value.metadata["draft"]["total"] == "4.00"
    assert raised.value.metadata["classification_enrichment"]["state"] == "pending"


async def test_short_work_budget_skips_label_dispatch(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    llm = observation_provider()
    result = await pipeline(llm).extract(
        "fixture", [path], options=ExperimentOptions(),
        deadline=monotonic() + 2, tags=[],
    )
    assert llm._client.chat.await_count == 1
    assert result.receipt.total == Decimal("4.00")
    assert result.classification_enrichment.state == "skipped"
    assert result.classification_enrichment.reason_code == "classification_budget_exhausted"


@pytest.mark.parametrize("mode", ["failed", "retry", "cancelled"])
async def test_benchmark_worker_preserves_observations_after_outer_release_failure(
    settings, monkeypatch, tmp_path, mode,
):
    llm, extract = managed_pipeline(monkeypatch, tmp_path)
    extract.ocr.load = AsyncMock()
    extract.ocr.get_capabilities.return_value.runtime_fingerprint = "runtime"
    extract.settings.required = True
    receipts = ReceiptService(
        settings.images, settings.pg, llm, ocr_provider=extract.ocr, ocr_settings=extract.settings,
    )
    worker = object.__new__(BenchmarkService)
    worker._provider, worker._receipts, worker._settings = llm, receipts, settings.worker
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    options = ExperimentOptions(pipeline="ocr")
    identity = await pipeline_identity(receipts, options)
    receipt_id = uuid4()
    manifest = {
        "identity": identity,
        "options": options.model_dump(mode="json"),
        "tags": [],
        "inputs": [{
            "receipt_id": str(receipt_id),
            "expected": receipt().model_dump(mode="json"),
            "images": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "pipeline": "ocr",
            "reference": {},
        }],
        "provider": type(llm).__name__,
        "endpoint_fingerprint": endpoint_fingerprint(llm),
        "ocr_fingerprint": ocr_fingerprint(extract.ocr),
        "validation_schema_sha256": validation_schema_sha256(),
        "prompt_hashes": {"1": hashlib.sha256(
            llm.for_experiment(options).build_prompt([], image_count=1).encode(),
        ).hexdigest()},
    }
    count = 2 if mode == "failed" else 0
    task = {
        "run_id": uuid4(), "receipt_id": receipt_id, "model_id": "fixture", "attempts": 0,
        "inference_started": False, "inference_attempt_count": count,
    }
    worker._db = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(model_ids=["fixture"], prompt_version=PROMPT_VERSION)),
        manifest=AsyncMock(return_value=manifest),
        begin_attempt=AsyncMock(return_value={
            "inference_started": True, "inference_attempt_count": count + 1,
        }),
        finish_task=AsyncMock(), retry_task=AsyncMock(),
    )
    worker._run_digests = AsyncMock(return_value={})
    worker._timeout = AsyncMock(return_value=600)
    llm._client.chat.side_effect = [llm._client.chat.return_value, chat_reply("{")]
    release_started, release_continue = asyncio.Event(), asyncio.Event()

    async def unload(**request):
        assert request == {"model": "fixture", "keep_alive": 0}
        assert llm._client.chat.await_count == 2
        release_started.set()
        if mode == "cancelled":
            await release_continue.wait()
        raise httpx.ReadTimeout("unload response lost")

    llm._client.generate = AsyncMock(side_effect=unload)
    if mode == "cancelled":
        running = asyncio.create_task(worker._execute(task))
        try:
            async with asyncio.timeout(3):
                await release_started.wait()
                running.cancel()
                await asyncio.sleep(0)
                release_continue.set()
                with pytest.raises(asyncio.CancelledError):
                    await running
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        worker._db.retry_task.assert_not_awaited()
        worker._db.finish_task.assert_not_awaited()
    else:
        await worker._execute(task)
        if mode == "retry":
            worker._db.retry_task.assert_awaited_once()
            worker._db.finish_task.assert_not_awaited()
        else:
            worker._db.retry_task.assert_not_awaited()
            failure = worker._db.finish_task.await_args.kwargs
            assert failure["success"] is False
            assert failure["result"]["full_receipt_correct"] is False
            assert "actual" not in failure["result"]
            metadata = failure["result"]["metadata"]
            assert metadata["draft"]["total"] == "4.00"
            assert metadata["ocr_pages"][0]["working_image"]["width"] == 3
            assert metadata["classification_enrichment"]["reason_code"] == "invalid_response"
            assert metadata["inference_attempt_count"] == 3
    journal = extract.coordinator().read_journal()
    assert release_started.is_set() and journal.unsettled and journal.models == ["fixture"]


@pytest.mark.parametrize("reply", ["empty_choices", "proxy_503", "read_timeout"])
async def test_openai_classifier_distinguishes_empty_choices_from_ambiguous_transport(
    monkeypatch, tmp_path, reply,
):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    raw = receipt().model_dump(mode="json")
    del raw["category"]
    raw["line_items"][0]["spending_category"] = "milk"
    raw["line_items"][0]["tags"] = ["observed"]
    events = []
    status = "unloaded"

    def handler(request):
        nonlocal status
        if request.url.path == "/llama/models":
            if "unload_requested" in events:
                events.append("unload_verified")
            return httpx.Response(200, json=model_status(status, foreign="unloaded"))
        if request.url.path == "/llama/models/unload":
            assert json.loads(request.content) == {"model": "own"}
            events.append("unload_requested")
            status = "unloaded"
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/llama/v1/chat/completions"
        status = "loaded"
        message = json.loads(request.content)["messages"][0]["content"]
        if isinstance(message, list):
            events.append("formatter")
            choices = [{
                "index": 0, "message": {"role": "assistant", "content": json.dumps(raw)},
                "finish_reason": "stop",
            }]
        else:
            assert message.startswith("Classify these ordered")
            assert events[-1] == "observations"
            events.append("classifier")
            if reply == "proxy_503":
                return httpx.Response(503, json={"error": "proxy unavailable"})
            if reply == "read_timeout":
                raise httpx.ReadTimeout("ambiguous inference", request=request)
            choices = []
        return httpx.Response(200, json={
            "id": "chat", "object": "chat.completion", "created": 0, "model": "own",
            "choices": choices,
        })

    async def checkpoint(observed):
        assert observed.receipt.total == Decimal("4.00")
        events.append("observations")

    install_owned_bindings(monkeypatch)
    async with configured(monkeypatch, handler) as llm:
        model = ModelInfo("own")
        model.capabilities = ["vision"]
        llm.get_available_models = AsyncMock(return_value=[model])
        llm.handoff = ModelHandoff(tmp_path / "handoff", llm, RouterOcr([]))
        extract = ExtractionPipeline(llm)

        async def run():
            return await extract.extract(
                "own", [path], options=ExperimentOptions(pipeline="direct_vision"),
                deadline=monotonic() + 90, tags=[], observation_checkpoint=checkpoint,
            )

        if reply == "empty_choices":
            result = await run()
            assert result.receipt.total == Decimal("4.00")
            assert result.receipt.line_items[0].spending_category == "milk"
            assert result.receipt.line_items[0].tags == ["observed"]
            assert result.classification_enrichment.state == "unavailable"
            assert result.classification_enrichment.reason_code == "invalid_response"
            assert result.telemetry.calls[-1]["stage"] == "classification"
            assert events == [
                "formatter", "observations", "classifier", "unload_requested", "unload_verified",
            ]
        else:
            error = APIStatusError if reply == "proxy_503" else APITimeoutError
            with pytest.raises(error):
                await run()
            assert events == ["formatter", "observations", "classifier"]
            journal = llm.handoff.read_journal()
            assert journal.unsettled and journal.models == ["own"]


@pytest.mark.parametrize("failure_at", ["formatter", "classifier"])
async def test_benchmark_worker_retry_only_before_admitted_observations(
    settings, tmp_path, failure_at,
):
    llm = observation_provider()
    failed = httpx.ReadTimeout("response status unknown")
    llm._client.chat.side_effect = (
        [failed] if failure_at == "formatter" else [llm._client.chat.return_value, failed]
    )
    receipts = ReceiptService(settings.images, settings.pg, llm)
    worker = object.__new__(BenchmarkService)
    worker._provider, worker._settings = llm, settings.worker
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    options = ExperimentOptions(pipeline="direct_vision")
    receipt_id = uuid4()
    manifest = {
        "options": options.model_dump(mode="json"),
        "inputs": [{
            "receipt_id": str(receipt_id),
            "expected": receipt().model_dump(mode="json"),
            "reference": {},
        }],
    }
    task = {
        "run_id": uuid4(), "receipt_id": receipt_id, "model_id": "fixture", "attempts": 0,
        "inference_started": False, "inference_attempt_count": 0,
    }
    worker._db = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(model_ids=["fixture"], prompt_version=PROMPT_VERSION)),
        manifest=AsyncMock(return_value=manifest),
        begin_attempt=AsyncMock(return_value={
            "inference_started": True, "inference_attempt_count": 1,
        }),
        finish_task=AsyncMock(), retry_task=AsyncMock(),
    )
    worker._run_digests = AsyncMock(return_value={})
    worker._timeout = AsyncMock(return_value=600)

    async def execute(task, _manifest, deadline, *, inference_deadline, managed_model_ids):
        assert managed_model_ids == ["fixture"]
        await worker._begin(task, deadline)
        await execute_submission(
            receipts, "fixture", [path], options, [], deadline=inference_deadline
        )

    worker._execute_experiment = execute
    await worker._execute(task)
    worker._db.begin_attempt.assert_awaited_once()
    if failure_at == "formatter":
        assert llm._client.chat.await_count == 1
        worker._db.retry_task.assert_awaited_once()
        worker._db.finish_task.assert_not_awaited()
    else:
        assert llm._client.chat.await_count == 2
        worker._db.retry_task.assert_not_awaited()
        worker._db.finish_task.assert_awaited_once()
        failure = worker._db.finish_task.await_args.kwargs
        assert failure["success"] is False
        result = failure["result"]
        assert result["full_receipt_correct"] is False
        assert result["metadata"]["draft"]["total"] == "4.00"
        assert result["metadata"]["classification_enrichment"]["state"] == "pending"
        assert result["metadata"]["inference_attempt_count"] == 1
