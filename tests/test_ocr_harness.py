import json
import sys
import time
from decimal import Decimal

import pytest

from scripts.ocr_harness.common import normalize_result, row_text
from scripts.ocr_harness.runner import run_matrix


def engine(tmp_path, name="fake", body=None):
    worker = tmp_path / f"{name}.py"
    worker.write_text(
        body
        or (
            "import json,sys\nfrom pathlib import Path\n"
            "Path(sys.argv[2]).write_text(json.dumps({'text':Path(sys.argv[1]).read_text(),'blocks':[]}))"
        )
    )
    return {"id": name, "command": [sys.executable, str(worker), "{image}", "{result}"]}


def test_cache_tracks_image_and_engine_configuration(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    e = engine(tmp_path)
    out = tmp_path / "out"
    first = run_matrix([e], [image], out, timeout=5)
    assert first[0]["status"] == "ok"
    assert not first[0]["cached"]
    assert run_matrix([e], [image], out, timeout=5)[0]["cached"]
    image.write_text("TOTAL 42,00")
    assert not run_matrix([e], [image], out, timeout=5)[0]["cached"]
    e["options"] = {"variant": "other"}
    assert not run_matrix([e], [image], out, timeout=5)[0]["cached"]


def test_failure_does_not_stop_other_engines_and_is_not_cached(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    bad = engine(tmp_path, "bad", "raise RuntimeError('missing model')")
    rows = run_matrix([bad, engine(tmp_path)], [image], tmp_path / "out", timeout=5)
    assert [r["status"] for r in rows] == ["error", "ok"]
    assert rows[0]["error"]
    assert not run_matrix([bad], [image], tmp_path / "out", timeout=5)[0]["cached"]


def test_real_subprocess_timeout_is_reported(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("receipt")
    slow = engine(tmp_path, "slow", "import time; time.sleep(30)")
    start = time.monotonic()
    rows = run_matrix([slow], [image], tmp_path / "out", timeout=0.1)
    assert rows[0]["status"] == "timeout"
    assert time.monotonic() - start < 3


@pytest.mark.parametrize(
    "data",
    [
        {"text": " "},
        {"text": 7},
        {"text": "ok", "blocks": [{"text": "x", "bbox": [[float("nan"), 0]]}]},
    ],
)
def test_invalid_recognition_is_not_success(data):
    with pytest.raises(ValueError):
        normalize_result(data)


def test_row_reconstruction_preserves_price_pairing():
    blocks = [
        {"text": "1,29", "bbox": [[100, 11], [130, 21]]},
        {"text": "MILCH", "bbox": [[0, 10], [50, 20]]},
        {"text": "PFAND", "bbox": [[0, 35], [50, 45]]},
        {"text": "0,25", "bbox": [[100, 35], [130, 45]]},
    ]
    assert row_text(blocks) == "MILCH  1,29\nPFAND  0,25"


def test_native_nested_coordinates_are_json_serializable():
    from scripts.ocr_harness.worker import block

    class NativeNumber:
        def tolist(self):
            return 3

    result = block("MILCH", [[NativeNumber(), NativeNumber()], [10, 20]], 0.9)
    assert json.loads(json.dumps(result))["bbox"] == [[3, 3], [10, 20]]


def test_report_includes_failures_and_no_unearned_accuracy(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    out = tmp_path / "out"
    run_matrix([engine(tmp_path)], [image], out, timeout=5)
    report = json.loads((out / "report.json").read_text())
    assert report["results"][0]["accuracy"] is None
    assert (out / "report.csv").is_file()
    assert (out / "report.md").is_file()


def test_model_weights_invalidate_cache(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    out = tmp_path / "results"
    model = tmp_path / "models" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"weights-v1")
    e = engine(tmp_path)
    run_matrix([e], [image], out, timeout=5)
    assert run_matrix([e], [image], out, timeout=5)[0]["cached"]
    model.write_bytes(b"weights-v2")
    assert not run_matrix([e], [image], out, timeout=5)[0]["cached"]


def test_receipt_scoring_does_not_forgive_one_cent(tmp_path):
    from scripts.ocr_harness.extraction import score_expected
    from vision_bill.model.receipt import Receipt

    expected = Receipt(
        confidence=90,
        merchant_name="SHOP",
        date="2026-11-07",
        currency="EUR",
        line_items=[],
        subtotal="12.34",
        total="12.34",
    )
    path = tmp_path / "expected.json"
    path.write_text(expected.model_dump_json())
    actual = expected.model_copy(update={"total": Decimal("12.35")})
    score = score_expected(path, actual)
    assert any(d["path"] == "total" and not d["equivalent"] for d in score["differences"])


def test_custom_adapter_needs_no_docker_preparation(tmp_path):
    from scripts.ocr_harness.__main__ import prepare

    records = prepare([engine(tmp_path)], tmp_path, timeout=1)
    assert records[0]["status"] == "ok"


def test_local_llm_client_ignores_environment_proxy(monkeypatch):
    from scripts.ocr_harness.extraction import make_provider

    monkeypatch.setenv("ALL_PROXY", "http://untrusted.invalid:1234")
    provider = make_provider("http://127.0.0.1:11434")
    assert provider._client._client.trust_env is False
    assert provider._client._client.follow_redirects is False


def test_scores_for_different_expected_receipts_do_not_collide(tmp_path):
    from scripts.ocr_harness.extraction import score_path

    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text('{"total":"1.00"}')
    second.write_text('{"total":"2.00"}')
    assert score_path(tmp_path, first) != score_path(tmp_path, second)
