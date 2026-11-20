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
    from PIL import Image
    Image.new("RGB", (5, 7), "red").save(first)
    Image.new("RGB", (5, 7), "blue").save(second)
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
    rows = typed_rows(case, tmp_path)
    pages, elapsed = cached_ocr(case, rows, "native")
    assert [page["text"] for page in pages] == ["OBSERVED 0", "OBSERVED 1"] and elapsed == 20
    case["images"][1].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cached_ocr(case, rows, "native")
    with pytest.raises(ValueError, match="Missing"):
        cached_ocr(case, [], "native")


async def test_direct_service_gets_all_photos_and_ocr_gets_replayed_text(settings, tmp_path):
    from ollama import ChatResponse

    from vision_bill.provider.llm.ollama import OllamaProvider

    case = case_files(tmp_path)
    provider = OllamaProvider("http://localhost:11434")
    from vision_bill.provider.llm.base import ModelInfo
    available = ModelInfo("model")
    available.capabilities = ["vision", "completion"]
    provider.get_available_models = AsyncMock(return_value=[available])
    payload = receipt().model_dump(mode="json")
    for item in payload["line_items"]:
        item["source_image"] = 0
    payload["duplicate_candidates"] = []
    provider._client.chat = AsyncMock(
        return_value=ChatResponse(message={"role": "assistant", "content": json.dumps(payload)})
    )
    result = await compare_case(provider, "model", case, "direct_vision", None, tmp_path, 5)
    assert result.metadata["used_pipeline"] == "direct_vision"
    transported = provider._client.chat.call_args.kwargs["messages"][0]["images"]
    assert len(transported) == len(case["images"])
    assert "2 ordered photos" in provider._client.chat.call_args.kwargs["messages"][0]["content"]
    pages, _ = cached_ocr(case, typed_rows(case, tmp_path), "native")
    result = await compare_case(provider, "model", case, "ocr", pages, tmp_path, 5)
    request = provider._client.chat.call_args.kwargs["messages"][0]
    assert "OBSERVED 0" in request["content"] and "OBSERVED 1" in request["content"] and not request.get("images")
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
    cache = typed_rows(case, tmp_path)
    for row in cache:
        row["engine"] = "pp"
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
    Path(second_run[0]["trace_artifact"]).write_text("tampered raw response")
    third_run = await comparison.run_comparison([case, second], targets, cache, "pp", tmp_path / "results")
    assert execute.await_count == 4
    assert sum(r.get("llm_cached", False) for r in third_run) == 2


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


def typed_rows(case, tmp_path):
    from test_ocr_provider import wire_page
    rows = []
    for i, path in enumerate(case['images']):
        artifact = tmp_path / f'typed-{i}.json'
        artifact.write_text(json.dumps(wire_page(i, digest(path.read_bytes()), text=f'OBSERVED {i}')))
        rows.append({'engine': 'native', 'input': str(path), 'status': 'ok',
                     'identity': {'input_sha256': digest(path.read_bytes())},
                     'artifact': str(artifact), 'elapsed_ms': 10})
    return rows


async def test_typed_replay_routes_every_photo_without_reference_or_geometry_loss(tmp_path):
    from test_extraction_pipeline import formatter
    case = case_files(tmp_path)
    case['expected'].write_text('REFERENCE_SENTINEL')
    pages, elapsed = cached_ocr(case, typed_rows(case, tmp_path), 'native')
    assert isinstance(pages, list) and len(pages) == 2
    llm = formatter(False)
    result = await compare_case(llm, 'fixture', case, 'ocr', pages, tmp_path, 5)
    prompt = llm._client.chat.call_args.kwargs['messages'][0]['content']
    assert prompt.index('OBSERVED 0') < prompt.index('OBSERVED 1')
    assert 'REFERENCE_SENTINEL' not in prompt and 'png_base64' not in prompt
    assert result.metadata['source_image_count'] == 2
    assert len(result.metadata['ocr_pages']) == 2
    assert result.metadata['ocr_replayed'] and elapsed == 20


def test_replay_cache_binds_raw_profile_runtime_and_order(tmp_path):
    case = case_files(tmp_path)
    rows = typed_rows(case, tmp_path)
    pages, _ = cached_ocr(case, rows, 'native')
    key = job_identity({'model': 'a'}, case, 'ocr', pages, 600, 8192)
    for field in ('profile_id', 'runtime_fingerprint'):
        changed = json.loads(json.dumps(pages))
        changed[0][field] = 'different'
        assert job_identity({'model': 'a'}, case, 'ocr', changed, 600, 8192) != key
    reordered = {**case, 'images': list(reversed(case['images']))}
    assert job_identity({'model': 'a'}, reordered, 'ocr', pages, 600, 8192) != key
    raw = Path(rows[0]['artifact'])
    raw.write_text(raw.read_text() + '\n')
    changed, _ = cached_ocr(case, rows, 'native')
    assert job_identity({'model': 'a'}, case, 'ocr', changed, 600, 8192) != key


async def test_second_replay_page_failure_never_returns_successful_prefix(tmp_path):
    from test_extraction_pipeline import formatter
    from vision_bill.service.experiment_execution import ExperimentFailure
    case = case_files(tmp_path)
    pages, _ = cached_ocr(case, typed_rows(case, tmp_path), 'native')
    pages[1]['text'] = ' '
    llm = formatter(False)
    with pytest.raises(ExperimentFailure) as failure:
        await compare_case(llm, 'fixture', case, 'ocr', pages, tmp_path, 5)
    assert failure.value.metadata['source_image_count'] == 2
    assert len(failure.value.metadata['ocr_pages']) == 1
    llm._client.chat.assert_not_awaited()


def test_summary_keeps_scoring_errors_and_reports_exact_failure_denominators(tmp_path):
    result = summary([{'engine':'m','llm_status':'ok','score_artifact':str(tmp_path/'missing')},
                      {'engine':'m','llm_status':'error'}])[0]
    assert result['cases']==2 and result['extracted']==1 and result['scoring_failed']==1
    assert result['score_excluding_payment']==0
    assert result['full_receipt_correct_rate']==0


def test_complete_report_escapes_metadata_and_displays_fallback_repair_and_cache_modes(tmp_path):
    from scripts.ocr_harness.runner import write_reports
    path=tmp_path/'original.png';path.write_bytes(b'not-decoded-by-report')
    write_reports(tmp_path,[{'engine':'<script>alert(1)</script>','input':str(path),'status':'ok','cached':False,
        'cache_mode':'ocr_replay','warm_state':'unknown','metadata':{'used_pipeline':'direct_vision','fallback_used':True,
        'fallback_reason':'ocr_page_failed','telemetry':{'stage_ms':{'repair':123},'finish_reason':'length'},
        'field_sources':{'merchant_name':'<img onerror=evil>'}}}])
    html=(tmp_path/'report.html').read_text()
    assert '<script>' not in html and '<img onerror' not in html
    assert '&lt;script&gt;' in html and '&lt;img onerror=evil&gt;' in html
    assert all(value in html for value in ['ocr_replay','ocr_page_failed','repair','length','unknown','direct_vision'])


async def test_failed_dataset_scoring_still_records_both_extractions(tmp_path,monkeypatch):
    from scripts.ocr_harness import comparison
    case=case_files(tmp_path)
    case['expected'].write_text('{invalid reference')
    client=SimpleNamespace(chat=AsyncMock(),list=AsyncMock(return_value=SimpleNamespace(models=[SimpleNamespace(model='m',digest='d')])),_client=SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(comparison,'make_target_provider',lambda *args:SimpleNamespace(_client=client))
    execute=AsyncMock(return_value=SimpleNamespace(receipt=receipt(),metadata={}))
    monkeypatch.setattr(comparison,'compare_case',execute)
    rows=await comparison.run_comparison([case],[{'id':'m','provider':'ollama','host':'http://localhost','model':'m'}],typed_rows(case,tmp_path),'native',tmp_path/'output')
    assert len(rows)==2 and execute.await_count==2
    assert all(row['llm_status']=='ok' and row['score_status']=='error' for row in rows)
    assert all(s['cases']==1 and s['scoring_failed']==1 and s['score_excluding_payment']==0 for s in summary(rows))


async def test_openai_target_uses_sdk_compatible_local_client():
    from scripts.ocr_harness.comparison import make_target_provider

    provider = make_target_provider(
        {"provider": "openai", "host": "http://127.0.0.1:1/v1"}, 12
    )
    try:
        assert provider._client._client.trust_env is False
        assert provider._client._client.follow_redirects is False
        assert provider._client.timeout == 12
    finally:
        await provider._client.close()
