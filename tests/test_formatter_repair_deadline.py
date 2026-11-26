"""A stalled optional correction must not discard previously validated observations."""

import asyncio
import json
import logging
import time
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from ollama import ChatResponse, ResponseError
from test_experiment_scoring import receipt
from test_receipt_input_provider import ocr_input

from vision_bill.provider.llm import base
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.service.submission_service import SubmissionService


def uncorrected_receipt():
    data = receipt().model_dump(mode="json")
    data["merchant_name"] = "Private merchant sentinel"
    data["line_items"][0] = {"description": "Private product sentinel", "quantity": 0}
    return ChatResponse(
        done_reason="stop", eval_count=408, message={"role": "assistant", "content": json.dumps(data)},
    )


async def test_slow_field_repair_preserves_reviewable_draft(monkeypatch, caplog):
    monkeypatch.setattr(base, "FORMATTER_REPAIR_SECONDS", 0.03, raising=False)
    caplog.set_level(logging.INFO, logger=base.__name__)
    provider = OllamaProvider("http://local")
    correction_started = asyncio.Event()

    async def send(**request):
        if len(request["messages"]) == 1:
            return uncorrected_receipt()
        correction_started.set()
        await asyncio.Event().wait()

    provider._client.chat = AsyncMock(side_effect=send)
    output = await provider.extract_observations(
        "private-model-sentinel", ocr_input(), tags=[], deadline=time.monotonic() + 1,
    )

    assert correction_started.is_set()
    outcome = SubmissionService._outcome(output, [], [])
    assert outcome.kind == "partial"
    assert output.draft.merchant_name == "Private merchant sentinel"
    assert output.draft.total == Decimal("4.00")
    assert output.draft.line_items[0].description == "Private product sentinel"
    assert output.draft.line_items[0].quantity is None
    assert output.receipt is None
    assert output.telemetry.calls[0]["repair_fields"] == ["line_items.0.quantity"]
    assert "phase=field_correction" in caplog.text
    assert "state=timeout" in caplog.text
    assert "Private " not in caplog.text and "private-model-sentinel" not in caplog.text


async def test_repair_provider_error_preserves_reviewable_draft():
    provider = OllamaProvider("http://local")
    provider._client.chat = AsyncMock(side_effect=[
        uncorrected_receipt(), ResponseError("upstream temporarily unavailable", 503),
    ])

    output = await provider.extract_observations(
        "model", ocr_input(), tags=[], deadline=time.monotonic() + 2,
    )

    assert provider._client.chat.await_count == 2
    assert SubmissionService._outcome(output, [], []).kind == "partial"
    assert output.draft.merchant_name == "Private merchant sentinel"
    assert output.draft.line_items[0].quantity is None


async def test_formatter_call_log_records_safe_timing_tokens_and_reason(caplog):
    caplog.set_level(logging.INFO, logger=base.__name__)
    provider = OllamaProvider("http://local")
    provider._client.chat = AsyncMock(side_effect=[
        uncorrected_receipt(),
        ChatResponse(
            done_reason="stop", eval_count=17,
            message={"role": "assistant", "content": '{"corrections":{"line_items.0.quantity":1}}'},
        ),
    ])

    await provider.extract_observations(
        "private-model-sentinel", ocr_input(), tags=[], deadline=time.monotonic() + 2,
    )

    finished = [record.getMessage() for record in caplog.records if "state=finished" in record.getMessage()]
    assert len(finished) == 2
    assert "phase=formatter" in finished[0] and "output_tokens=408" in finished[0]
    assert "phase=field_correction" in finished[1] and "output_tokens=17" in finished[1]
    assert all("elapsed_ms=" in line and "finish_reason=stop" in line for line in finished)
    assert "repair_reason=invalid_fields" in caplog.text
    assert "Private " not in caplog.text and "private-model-sentinel" not in caplog.text
    limits = [request.kwargs["options"]["num_predict"] for request in provider._client.chat.await_args_list]
    assert limits == [32768, 2048]


async def test_slow_repair_of_malformed_json_spends_only_the_repair_budget(monkeypatch, caplog):
    monkeypatch.setattr(base, "FORMATTER_REPAIR_SECONDS", 0.03)
    caplog.set_level(logging.INFO, logger=base.__name__)
    provider = OllamaProvider("http://local")

    async def send(**request):
        if len(request["messages"]) == 1:
            return ChatResponse(
                done_reason="stop", message={"role": "assistant", "content": '{"merchant_name":'},
            )
        await asyncio.Event().wait()

    provider._client.chat = AsyncMock(side_effect=send)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(1):
            await provider.extract_observations(
                "private-model-sentinel", ocr_input(), tags=[], deadline=time.monotonic() + 10,
            )
    assert "phase=formatter_repair state=timeout" in caplog.text
    assert "repair_reason=invalid_json" in caplog.text
