"""Regression coverage for the OCR service's real response boundary."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider


@pytest.fixture
def ocr_server():
    path = Path(__file__).parents[1] / "paddleocr-service" / "server.py"
    spec = importlib.util.spec_from_file_location("paddle_server_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("native_blocks", [False, True])
def test_recognized_content_survives_without_image_pixels(ocr_server, native_blocks):
    blocks = [
        {"block_label": "text", "block_content": "SHOP", "block_bbox": [0, 0, 100, 20]},
        {"block_label": "text", "block_content": "TOTAL 12.34", "block_bbox": [0, 20, 100, 40]},
    ]
    if native_blocks:
        blocks = [
            SimpleNamespace(label=b["block_label"], content=b["block_content"], bbox=b["block_bbox"])
            for b in blocks
        ]
    # Actual Paddle results hold block objects; JSON exports hold dictionaries.
    # Image pixels must never cross this HTTP boundary, even when nested.
    result = ocr_server.process_ocrvl_result(
        {"parsing_res_list": blocks, "layout_det_res": {"input_img": object(), "boxes": []}}
    )
    assert result["text"] == "SHOP\nTOTAL 12.34"
    assert [b["text"] for b in result["blocks"]] == ["SHOP", "TOTAL 12.34"]
    assert result["blocks"][0]["bbox"] == [[0.0, 0.0], [100.0, 20.0]]
    assert "input_img" not in json.dumps(result)


def test_layout_boxes_are_not_a_transcription(ocr_server):
    with pytest.raises(ValueError, match="recognized text"):
        ocr_server.process_ocrvl_result(
            {"layout_det_res": {"boxes": [{"label": "text", "coordinate": [0, 0, 1, 1]}]}}
        )


@pytest.mark.parametrize("provider_class", [OllamaProvider, OpenAIProvider])
def test_empty_top_level_text_uses_recognized_blocks(provider_class):
    provider = object.__new__(provider_class)
    assert provider._ocr_result_to_text(
        {"text": "", "blocks": [{"text": "SHOP"}, {"text": "TOTAL 12.34"}]}
    ) == "SHOP\nTOTAL 12.34"


@pytest.mark.parametrize("provider_class", [OllamaProvider, OpenAIProvider])
@pytest.mark.parametrize("result", [{"text": "  ", "blocks": []}, {"layout_info": {"boxes": []}}])
@pytest.mark.asyncio
async def test_no_generation_from_missing_transcription(provider_class, result):
    provider = object.__new__(provider_class)
    provider.send_message = AsyncMock(side_effect=AssertionError("Must not generate without text"))
    with pytest.raises(ValueError, match="recognized text"):
        await provider.analyse_receipt_from_ocr_text("test-model", result)


@pytest.mark.asyncio
async def test_diagnostic_script_rejects_empty_http_success(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "test_ocr_service.py"
    spec = importlib.util.spec_from_file_location("ocr_diagnostic_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    (tmp_path / "receipt.png").write_bytes(b"image")
    provider = SimpleNamespace(
        check_connection=AsyncMock(return_value=True),
        extract_text=AsyncMock(return_value={"text": "", "blocks": []}),
    )
    monkeypatch.setattr(script, "get_ocr_provider", lambda settings: provider)
    args = SimpleNamespace(ocr_url=None, timeout=900, data_dir=str(tmp_path), quiet=True)
    results, exit_code = await script.run(args)
    assert exit_code == 1
    assert results[0]["ok"] is False
    assert "recognized text" in results[0]["error"]
