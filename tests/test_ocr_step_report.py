import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.ocr_harness.runner import write_reports


def test_html_shows_stages_and_escapes_untrusted_model_output(tmp_path):
    image = tmp_path / "receipt.png"
    Image.new("RGB", (2, 2), "white").save(image)
    ocr = tmp_path / "ocr.json"
    ocr.write_text(json.dumps({"text": "MILCH 1,29", "row_text": "MILCH  1,29", "blocks": []}))
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"total":"1.29"}')
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "messages": [{"role": "user", "content": "format receipt"}],
                        "response": '</pre><script>alert("unsafe")</script>',
                    },
                    {"response": '{"total":"1.29"}'},
                ]
            }
        )
    )
    row = {
        "input": str(image),
        "engine": "sample",
        "status": "ok",
        "artifact": str(ocr),
        "receipt_artifact": str(receipt),
        "trace_artifact": str(trace),
        "execution_path": "receipt_service",
        "llm_status": "ok",
        "cached": True,
    }
    write_reports(tmp_path, [row])
    text = (tmp_path / "report.html").read_text()
    assert "data:image/png;base64," in text
    assert "OCR output" in text and "Text sent to Gemma" in text
    assert "Attempt 1" in text and "Attempt 2" in text
    assert "ReceiptService return" in text
    assert '<script>alert("unsafe")</script>' not in text
    assert "&lt;script&gt;" in text
    assert "default-src 'none'" in text


def test_legacy_cache_does_not_invent_a_raw_response(tmp_path):
    image = tmp_path / "missing.png"
    row = {
        "input": str(image),
        "engine": "old",
        "status": "error",
        "cached": True,
        "artifact": str(tmp_path / "missing.json"),
        "error": "OCR failed",
    }
    write_reports(tmp_path, [row])
    text = (tmp_path / "report.html").read_text()
    assert "not captured" in text.lower()
    assert "OCR failed" in text


@pytest.mark.asyncio
async def test_capture_preserves_raw_attempts_without_thinking(tmp_path):
    from scripts.ocr_harness.extraction import capture_chat

    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="RAW JSON", thinking="private"))

    client = SimpleNamespace(chat=chat)
    trace = tmp_path / "trace.json"
    with capture_chat(client, trace):
        response = await client.chat(messages=[{"role": "user", "content": "OCR text"}])
    assert response.message.content == "RAW JSON"
    assert client.chat is chat
    saved = json.loads(trace.read_text())
    assert saved["attempts"][0]["response"] == "RAW JSON"
    assert "private" not in trace.read_text()
    assert len(calls) == 1
