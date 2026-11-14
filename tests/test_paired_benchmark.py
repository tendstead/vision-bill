import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_experiment_scoring import receipt

from scripts.ocr_harness.comparison import (
    cached_ocr,
    capture_requests,
    compare_case,
    job_identity,
    load_cases,
    summary,
)
from scripts.ocr_harness.runner import digest


def case_files(tmp_path):
    first, second = tmp_path / "one.png", tmp_path / "two.png"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    expected = tmp_path / "expected.json"
    expected.write_text(receipt().model_dump_json())
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {"id": "sample", "images": ["one.png", "two.png"], "expected": "expected.json"}
                ]
            }
        )
    )
    return load_cases(manifest)[0]


def test_case_manifest_preserves_order_and_rejects_duplicate_ids(tmp_path):
    case = case_files(tmp_path)
    assert [p.name for p in case["images"]] == ["one.png", "two.png"]
    p = tmp_path / "dataset.json"
    data = json.loads(p.read_text())
    data["cases"] *= 2
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_cases(p)


def test_cached_ocr_requires_every_unchanged_photo(tmp_path):
    case = case_files(tmp_path)
    rows = []
    for i, image in enumerate(case["images"]):
        artifact = tmp_path / f"ocr-{i}.json"
        artifact.write_text(
            json.dumps({"text": f"photo {i}", "row_text": f"ordered {i}", "blocks": []})
        )
        rows.append(
            {
                "engine": "pp-medium",
                "input": str(image),
                "status": "ok",
                "identity": {"input_sha256": digest(image.read_bytes())},
                "artifact": str(artifact),
                "elapsed_ms": 10,
            }
        )
    text, elapsed = cached_ocr(case, rows, "pp-medium")
    assert text.index("ordered 0") < text.index("ordered 1") and elapsed == 20
    case["images"][1].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cached_ocr(case, rows, "pp-medium")
    with pytest.raises(ValueError, match="Missing"):
        cached_ocr(case, [], "pp-medium")


async def test_direct_service_gets_all_photos_and_ocr_gets_replayed_text(settings, tmp_path):
    from ollama import ChatResponse

    from vision_bill.provider.llm.ollama import OllamaProvider

    case = case_files(tmp_path)
    provider = OllamaProvider("http://localhost:11434")
    payload = receipt().model_dump(mode="json")
    for item in payload["line_items"]:
        item["source_image"] = 0
    payload["duplicate_candidates"] = []
    provider._client.chat = AsyncMock(
        return_value=ChatResponse(message={"role": "assistant", "content": json.dumps(payload)})
    )
    result = await compare_case(provider, "model", case, "direct_vision", None, tmp_path, 5)
    assert result.metadata["used_pipeline"] == "direct_vision"
    assert provider._client.chat.call_args.kwargs["messages"][0]["images"] == case["images"]
    result = await compare_case(provider, "model", case, "ocr", "both photos OCR", tmp_path, 5)
    request = provider._client.chat.call_args.kwargs["messages"][0]
    assert "both photos OCR" in request["content"] and not request.get("images")
    assert result.metadata["ocr_replayed"] and result.metadata["source_image_count"] == 2
    assert result.metadata["fallback_used"] is False
    with pytest.raises(ValueError, match="OCR text"):
        await compare_case(provider, "model", case, "ocr", "", tmp_path, 5)


def test_summary_excludes_payment_and_counts_failures():
    scores = {"header": 6 / 7, "normalized": 1 - 0.2 / 7}
    rows = [
        {
            "engine": "model/ocr",
            "llm_status": "ok",
            "score": {
                "scores": scores,
                "differences": [{"path": "payment_method", "equivalent": False}],
            },
            "total_exact": True,
            "item_count_exact": True,
        },
        {"engine": "model/ocr", "llm_status": "error"},
    ]
    result = summary(rows)[0]
    assert result["valid"] == 1 and result["cases"] == 2
    assert result["score_excluding_payment"] == pytest.approx(0.5)
    assert result["score_successes_excluding_payment"] == pytest.approx(1)


def test_identity_changes_with_models_references_photos_and_ocr(tmp_path):
    case = case_files(tmp_path)
    target = {"id": "test", "model": "m", "revision": "r1"}

    def key(text="ocr"):
        return job_identity(target, case, "ocr", text, 120, 4096)

    before = key()
    target["revision"] = "r2"
    assert key() != before
    before = key()
    case["expected"].write_text("{}")
    assert key() != before
    before = key()
    case["images"][1].write_bytes(b"other")
    assert key() != before
    assert key("changed text") != key()


@pytest.mark.parametrize("kind", ["ollama", "openai"])
async def test_trace_records_real_output_excludes_images_and_restores_client(tmp_path, kind):
    response = (
        SimpleNamespace(message=SimpleNamespace(content="raw JSON", thinking="hidden"))
        if kind == "ollama"
        else SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="raw JSON"))])
    )
    original = AsyncMock(return_value=response)
    endpoint = (
        SimpleNamespace(chat=original)
        if kind == "ollama"
        else SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=original)))
    )
    provider = SimpleNamespace(_client=endpoint)
    path = tmp_path / "trace.json"
    with capture_requests(provider, kind, path, 4096):
        call = endpoint.chat if kind == "ollama" else endpoint.chat.completions.create
        await call(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "read"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
                    ],
                    "images": [b"SECRET"],
                }
            ]
        )
    trace = json.loads(path.read_text())
    text = path.read_text()
    assert trace["attempts"][0]["response"] == "raw JSON"
    assert "SECRET" not in text and "hidden" not in text and "image omitted" in text
    assert (endpoint.chat if kind == "ollama" else endpoint.chat.completions.create) is original
    kw = original.call_args.kwargs
    assert (kw["options"]["num_predict"] if kind == "ollama" else kw["max_tokens"]) == 4096


async def test_runner_records_missing_ocr_and_continues_then_reuses_successes(
    tmp_path, monkeypatch
):
    from scripts.ocr_harness import comparison

    case = case_files(tmp_path)
    second = {**case, "id": "missing-ocr", "images": [tmp_path / "missing.png"]}
    second["images"][0].write_bytes(b"missing")
    cache = []
    for i, image in enumerate(case["images"]):
        artifact = tmp_path / f"cached-{i}.json"
        artifact.write_text(json.dumps({"text": str(i), "blocks": []}))
        cache.append(
            {
                "engine": "pp",
                "input": str(image),
                "identity": {"input_sha256": digest(image.read_bytes())},
                "artifact": str(artifact),
                "status": "ok",
            }
        )
    client = SimpleNamespace(
        chat=AsyncMock(),
        list=AsyncMock(
            return_value=SimpleNamespace(models=[SimpleNamespace(model="m", digest="r")])
        ),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )
    monkeypatch.setattr(
        comparison, "make_target_provider", lambda *args: SimpleNamespace(_client=client)
    )
    execute = AsyncMock(
        return_value=SimpleNamespace(receipt=receipt(), metadata={"fallback_used": False})
    )
    monkeypatch.setattr(comparison, "compare_case", execute)
    targets = [
        {"id": "target", "provider": "ollama", "host": "http://localhost:11434", "model": "m"}
    ]
    first = await comparison.run_comparison(
        [case, second], targets, cache, "pp", tmp_path / "results"
    )
    assert len(first) == 4 and sum(r["llm_status"] == "ok" for r in first) == 3
    assert first[-1]["llm_status"] == "error" and "Missing" in first[-1]["llm_error"]
    assert execute.await_count == 3
    second_run = await comparison.run_comparison(
        [case, second], targets, cache, "pp", tmp_path / "results"
    )
    assert execute.await_count == 3
    assert sum(r.get("llm_cached", False) for r in second_run) == 3


def test_report_labels_vision_path_and_ocr_replay(tmp_path):
    from scripts.ocr_harness.runner import write_reports

    image = tmp_path / "preview.png"
    image.write_bytes(b"preview")
    row = {
        "engine": "model/vision",
        "pipeline": "direct_vision",
        "input": str(image),
        "status": "ok",
        "llm_status": "error",
        "cached": False,
        "preview_note": "Overview only; two source photos were supplied.",
    }
    write_reports(tmp_path, [row])
    text = (tmp_path / "report.html").read_text()
    assert "OCR: not used" in text and "Direct vision: source images sent to the model" in text
    assert row["preview_note"] in text


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_ocr_prompt_explains_text_input_and_requires_receipt_instance(kind):
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.openai import OpenAIProvider

    provider = (
        OllamaProvider("http://localhost:11434")
        if kind == "ollama"
        else OpenAIProvider("http://localhost:8642/v1", "unused")
    )
    provider = provider.for_experiment(ExperimentOptions(include_classification=False))
    prompt = provider._build_ocr_formatting_prompt("Milk 2,00 EUR")
    assert "OCR TEXT, not an image" in prompt
    assert "Do NOT return a JSON schema" in prompt
    assert "Milk 2,00 EUR" in prompt and '"properties"' in prompt
    assert "from 1 ordered photos" not in prompt
    assert "spending_category" not in prompt


def test_direct_extraction_prompt_distinguishes_values_from_json_schema():
    from vision_bill.provider.llm.experiment import extraction_prompt

    assert "populated receipt object, not the JSON schema" in extraction_prompt(3)


def test_cli_isolates_configuration_from_live_app(tmp_path, monkeypatch):
    import logging
    import os
    import sys

    from scripts.ocr_harness import comparison

    case_files(tmp_path)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"targets": [{"id": "local"}]}))
    ocr = tmp_path / "ocr-report.json"
    ocr.write_text(json.dumps({"results": []}))
    live = tmp_path / "live-config.yaml"
    live.write_text("protected: app-settings")
    output = tmp_path / "results"
    monkeypatch.setenv("VISION_BILL_CONFIG_PATH", str(live))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparison",
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--targets",
            str(targets),
            "--ocr-report",
            str(ocr),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(comparison, "run_comparison", AsyncMock(return_value=[]))
    previous_umask = os.umask(0o077)
    os.umask(previous_umask)
    previous_logging = logging.root.manager.disable
    try:
        assert comparison.main() == 0
    finally:
        os.umask(previous_umask)
        logging.disable(previous_logging)
    isolated = Path(os.environ["VISION_BILL_CONFIG_PATH"])
    assert isolated != live and isolated.parent == output
    assert live.read_text() == "protected: app-settings"
    assert isolated.stat().st_mode & 0o077 == 0
    settings = json.loads(isolated.read_text())
    assert settings["pg"]["user"] == "harness" and settings["pg"]["password"] == "unused"


def test_bootstrap_settings_works_without_live_env_or_dotenv(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    code = 'from pathlib import Path; from scripts.ocr_harness.comparison import bootstrap_settings; import sys; bootstrap_settings(Path(sys.argv[1])); from vision_bill.config import settings; assert settings.pg.user == "harness"; assert settings.llm.model_name == "benchmark-unused"'
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "results")],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": str(root) + os.pathsep + str(root / "src")},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
