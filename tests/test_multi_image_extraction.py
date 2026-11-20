import asyncio
import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APITimeoutError

from vision_bill.helper.duplicates import verify_duplicate
from vision_bill.model.receipt import LineItem
from vision_bill.model.receipt_extraction import MultiImageReceipt
from vision_bill.provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider
from vision_bill.service.receipt_service import ReceiptService


def item(name="Milk", quantity=1, total="2.50", source=0):
    return dict(
        description=name,
        quantity=quantity,
        unit_price=total,
        total_price=total,
        source_image=source,
    )


def extraction(items, pairs):
    return dict(
        confidence=95,
        merchant_name="Test",
        date="2026-09-27",
        subtotal="5.00",
        total="5.00",
        line_items=items,
        duplicate_candidates=[dict(first_index=a, second_index=b) for a, b in pairs],
    )


@pytest.mark.parametrize(
    "changes", [dict(description="Bread"), dict(quantity=2), dict(total_price="2.51")]
)
def test_duplicate_requires_name_count_and_total(changes):
    first = LineItem.model_validate(item())
    second = LineItem.model_validate({**item(), **changes})
    assert not verify_duplicate(first, second)


def test_duplicate_normalization_and_tunable_parameters():
    first = LineItem.model_validate(item("  Whole   MILK "))
    assert verify_duplicate(first, LineItem.model_validate(item("whole milk")))
    second = LineItem.model_validate(item("whole mil", quantity=1.01, total="2.51"))
    assert not verify_duplicate(first, second)
    assert verify_duplicate(
        first,
        second,
        name_similarity=0.9,
        quantity_tolerance=Decimal(".01"),
        total_tolerance=Decimal(".01"),
    )


def test_only_verified_boundary_overlap_is_removed():
    data = extraction([item(), item(), item(source=1), item("Bread", source=1)], [(1, 2)])
    receipt = MultiImageReceipt.model_validate(data).reconciled(2)
    assert [i.description for i in receipt.line_items] == ["Milk", "Milk", "Bread"]
    assert "source_image" not in receipt.line_items[0].model_dump()


def test_mismatching_overlap_is_preserved():
    data = extraction([item(), item(quantity=2, source=1)], [(0, 1)])
    assert len(MultiImageReceipt.model_validate(data).reconciled(2).line_items) == 2


@pytest.mark.parametrize("second_quantity", [None, 1])
def test_overlap_without_observed_quantity_is_not_verified(second_quantity):
    data = extraction([item(quantity=None), item(quantity=second_quantity, source=1)], [(0, 1)])
    receipt = MultiImageReceipt.model_validate(data).reconciled(2)
    assert [row.quantity for row in receipt.line_items] == [None, second_quantity]


@pytest.mark.parametrize(
    "items,pairs",
    [
        ([item(), item()], [(0, 1)]),  # same photo
        ([item(), item(source=2)], [(0, 1)]),  # non-adjacent photos
        ([item(), item("Bread"), item(source=1)], [(0, 2)]),  # not a suffix
        ([item(), item(source=1)], [(0, 3)]),  # invalid index
    ],
)
def test_invalid_overlap_cannot_delete_rows(items, pairs):
    with pytest.raises(ValueError):
        MultiImageReceipt.model_validate(extraction(items, pairs)).reconciled(3)


@pytest.mark.parametrize("kind", ["ollama", "openai"])
async def test_providers_send_ordered_photos_and_stop_on_unreadable(kind, tmp_path):
    paths = [tmp_path / "top.png", tmp_path / "bottom.png"]
    for index, path in enumerate(paths):
        path.write_bytes(str(index).encode())
    if kind == "ollama":
        provider = OllamaProvider("http://localhost")
    else:
        with patch("vision_bill.provider.llm.openai.AsyncOpenAI"):
            provider = OpenAIProvider("http://localhost", "test")
    provider.send_message = AsyncMock(
        return_value=json.dumps(
            {
                "error": {
                    "code": "unreadable",
                    "message": "Bottom photo is blurry; provide a clearer image",
                }
            }
        )
    )
    with pytest.raises(UnreadableReceiptError, match="Bottom photo"):
        await provider.analyse_receipt_from_model("test", paths)
    provider.send_message.assert_awaited_once()
    message = provider.send_message.call_args.args[1][0]
    if kind == "ollama":
        assert message["images"] == paths
    else:
        images = [part for part in message["content"] if part["type"] == "image_url"]
        assert len(images) == 2
        assert images[0]["image_url"]["url"].endswith("MA==")
        assert images[1]["image_url"]["url"].endswith("MQ==")


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("slow"),
        APITimeoutError(request=httpx.Request("POST", "http://local")),
    ],
)
async def test_sdk_timeouts_are_normalized(settings, error):
    provider = AsyncMock()
    provider.analysis_timeout_seconds = 1
    provider.analyse_receipt_from_model.side_effect = error
    service = ReceiptService(settings.images, settings.pg, provider)
    with pytest.raises(AnalysisTimeoutError):
        await service.analyse_receipt_from_path("test", Path("unused"))
    provider.analyse_receipt_from_model.assert_awaited_once()


async def test_overall_deadline_cancels_slow_extraction(settings):
    provider = AsyncMock()
    provider.analysis_timeout_seconds = 0.01
    cancelled = asyncio.Event()

    async def slow(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    provider.analyse_receipt_from_model.side_effect = slow
    service = ReceiptService(settings.images, settings.pg, provider)
    with pytest.raises(AnalysisTimeoutError):
        await service.analyse_receipt_from_path("test", Path("unused"))
    assert cancelled.is_set()
