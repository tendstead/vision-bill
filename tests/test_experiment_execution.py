from unittest.mock import AsyncMock

import pytest

from vision_bill.model.experiment import ExperimentOptions
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.service.experiment_execution import compact_ocr_text


def test_empty_primary_text_uses_recognized_blocks():
    text = compact_ocr_text({"text": "", "blocks": [{"text": "Mate 12.90"}]})
    assert text == "Mate 12.90"


def test_layout_pixels_are_not_sent_to_llm():
    with pytest.raises(ValueError, match="recognized text"):
        compact_ocr_text({"layout_info": {"input_img": [[[255, 255, 255]]]}})


def test_experiment_provider_settings_do_not_mutate_live_provider():
    live = OllamaProvider("http://localhost:11434", temperature=0.7)
    probe = live.for_experiment(ExperimentOptions(temperature=0, include_classification=False))
    assert live._temperature == 0.7
    assert probe._temperature == 0
    prompt = probe.build_prompt(["deposit", "food"])
    assert "spending_category" not in prompt
    assert "Tag every deposit" not in prompt
    assert "spending_category" in live.build_prompt(["deposit", "food"])


async def test_thinking_override_is_sent_as_top_level_option():
    live = OllamaProvider("http://localhost:11434", temperature=0.7)
    probe = live.for_experiment(ExperimentOptions(think=False))
    response = AsyncMock()
    response.message.content = "{}"
    probe._client.chat = AsyncMock(return_value=response)
    assert await probe.send_message("model", [{"role": "user", "content": "read"}]) == "{}"
    kwargs = probe._client.chat.call_args.kwargs
    assert kwargs["think"] is False
    assert "think" not in kwargs["options"]


async def test_upload_execution_preserves_all_photo_paths_and_frozen_tags(settings, tmp_path):
    import json
    from ollama import ChatResponse
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.experiment_execution import execute_submission
    from test_experiment_scoring import receipt

    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    provider = OllamaProvider("http://localhost:11434", temperature=0.7)
    payload = receipt(("Milk",)).model_dump(mode="json")
    payload["line_items"][0]["source_image"] = 0
    payload["duplicate_candidates"] = []
    provider._client.chat = AsyncMock(return_value=ChatResponse(
        message={"role": "assistant", "content": json.dumps(payload)},
        eval_count=10, eval_duration=20_000_000))
    service = ReceiptService(settings.images, settings.pg, provider)
    result = await execute_submission(service, "model", [first, second],
                                      ExperimentOptions(temperature=0), ["frozen_tag"])
    assert result.receipt.line_items[0].description == "Milk"
    request = provider._client.chat.call_args.kwargs
    assert request["messages"][0]["images"] == [first, second]
    assert "frozen_tag" in request["messages"][0]["content"]
    assert request["options"]["temperature"] == 0
    assert provider._temperature == 0.7
    assert result.metadata["image_count"] == 2
    assert result.metadata["calls"][0]["eval_ms"] == 20


async def test_forced_ocr_is_not_silently_replaced_with_vision(settings, tmp_path):
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.experiment_execution import execute_submission
    path = tmp_path / "receipt.jpg"
    path.write_bytes(b"image")
    service = ReceiptService(settings.images, settings.pg,
                             OllamaProvider("http://localhost:11434"))
    with pytest.raises(ValueError, match="OCR extension"):
        await execute_submission(service, "model", [path], ExperimentOptions(pipeline="ocr"), [])


def test_ocr_fingerprint_detects_endpoint_and_timeout_changes():
    from types import SimpleNamespace
    from vision_bill.service.experiment_execution import ocr_fingerprint
    provider = SimpleNamespace(service_url="http://ocr-one:8080", timeout=300)
    original = ocr_fingerprint(provider)
    assert original == ocr_fingerprint(provider)
    provider.service_url = "http://ocr-two:8080"
    assert ocr_fingerprint(provider) != original
    provider.service_url = "http://ocr-one:8080"
    provider.timeout = 600
    assert ocr_fingerprint(provider) != original
    assert ocr_fingerprint(None) is None


@pytest.mark.parametrize("missing", [False, True])
async def test_reevaluation_rejects_changed_or_missing_model(missing):
    from types import SimpleNamespace
    from uuid import uuid4
    from vision_bill.service.benchmark_service import BenchmarkService
    run_id, receipt_id = uuid4(), uuid4()
    run = SimpleNamespace(receipt_ids=[receipt_id], model_ids=["model"])
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(run=AsyncMock(return_value=run),
                                  manifest=AsyncMock(return_value={"options": {}}))
    current = SimpleNamespace(id="model", digest="replacement")
    service._provider = SimpleNamespace(get_available_models=AsyncMock(return_value=[] if missing else [current]))
    service._run_digests = AsyncMock(return_value={"model": "original"})
    service._run_experiment = AsyncMock()
    with pytest.raises(ValueError, match="digest changed"):
        await service.reevaluate(run_id, receipt_id, "model")
    service._run_experiment.assert_not_awaited()


async def test_reevaluation_enforces_saved_run_deadline():
    import asyncio
    from types import SimpleNamespace
    from uuid import uuid4
    from vision_bill.service.benchmark_service import BenchmarkService
    run_id, receipt_id = uuid4(), uuid4()
    run = SimpleNamespace(receipt_ids=[receipt_id], model_ids=["model"])
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(run=AsyncMock(return_value=run),
                                  manifest=AsyncMock(return_value={"options": {}}))
    current = SimpleNamespace(id="model", digest="original")
    service._provider = SimpleNamespace(get_available_models=AsyncMock(return_value=[current]))
    service._run_digests = AsyncMock(return_value={"model": "original"})
    service._timeout = AsyncMock(return_value=0.01)

    async def slow(*args):
        await asyncio.Event().wait()

    service._run_experiment = slow
    with pytest.raises(TimeoutError):
        await service.reevaluate(run_id, receipt_id, "model")


@pytest.mark.parametrize("mode", ["success", "strict", "fallback"])
async def test_published_ocr_extension_contract(settings, tmp_path, mode):
    import json
    from types import SimpleNamespace
    from ollama import ChatResponse
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.experiment_execution import execute_submission, ExperimentFailure
    from test_experiment_scoring import receipt
    if not hasattr(ReceiptService, "analyse_receipt_with_ocr"):
        pytest.skip("Run after merging the optional OCR extension")
    path = tmp_path / "receipt.jpg"
    path.write_bytes(b"image")
    provider = OllamaProvider("http://localhost:11434")
    provider._client.chat = AsyncMock(return_value=ChatResponse(
        message={"role": "assistant", "content": json.dumps(receipt().model_dump(mode="json"))}))
    service = ReceiptService(settings.images, settings.pg, provider)
    result = {"text": "", "blocks": [{"text": "Milk 2.00"}], "layout_info": {"pixels": [255]}}
    service._ocr_provider = SimpleNamespace(extract_text=AsyncMock(
        return_value=result if mode == "success" else {"text": ""}))
    options = ExperimentOptions(pipeline="ocr", allow_ocr_fallback=mode == "fallback")
    if mode == "strict":
        with pytest.raises(ExperimentFailure, match="no recognized text"):
            await execute_submission(service, "model", [path], options, [])
        provider._client.chat.assert_not_awaited()
        return
    output = await execute_submission(service, "model", [path], options, [])
    assert output.metadata["fallback_used"] is (mode == "fallback")
    assert output.metadata["used_pipeline"] == ("direct_vision" if mode == "fallback" else "ocr")
    if mode == "success":
        content = provider._client.chat.call_args.kwargs["messages"][0]["content"]
        assert "Milk 2.00" in content
        assert "pixels" not in content
        assert output.metadata["ocr_characters"] == len("Milk 2.00")
