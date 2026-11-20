"""CPU contract tests for the test-only wire servers (no GPU/runtime shortcuts)."""

import base64
import hashlib
import importlib.util
import json
import sys
from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image, ImageOps

from vision_bill.model.extraction import ReceiptInput
from vision_bill.model.ocr import OcrResponse, SourceRef
from vision_bill.provider.llm.experiment import source_extraction_prompt
from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "e2e"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubConnection:
    """Run the real stdlib HTTP handler against an in-memory connection."""

    def __init__(self, request):
        self.request = BytesIO(request)
        self.response = BytesIO()

    def makefile(self, mode, buffering=None):
        return self.response if buffering is None else self.request

    def sendall(self, data):
        self.response.write(data)


def llm_request(server, path, *, model=None, authorization=None):
    headers = ["Host: llm-stub"]
    body = b""
    method = "GET"
    if model is not None:
        raw = (ROOT / "e2e/llm_stub/fixtures/a.png").read_bytes()
        image_url = "data:image/png;base64," + base64.b64encode(raw).decode()
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"content": [{"type": "image_url", "image_url": {"url": image_url}}]}
                ],
            }
        ).encode()
        headers.append(f"Content-Length: {len(body)}")
        method = "POST"
    if authorization is not None:
        headers.append(f"Authorization: {authorization}")
    wire_request = "\r\n".join([f"{method} {path} HTTP/1.0", *headers, "", ""]).encode()
    connection = StubConnection(wire_request + body)
    server.Handler(connection, ("127.0.0.1", 0), None)
    connection.response.seek(0)
    response = HTTPResponse(connection)
    response.begin()
    return response.status, json.load(response)


def test_llm_stub_accepts_only_opt_in_exact_model_without_credentials(monkeypatch):
    monkeypatch.setenv("VB_E2E_LLM_MODEL_ID", "Operator/team-receipt:smoke-v1")
    monkeypatch.delenv("VB_E2E_LLM_BEARER_TOKEN", raising=False)
    server = module("llm_stub_prepared_model", "e2e/llm_stub/server.py")
    status, discovery = llm_request(server, "/v1/models")
    assert status == 200
    assert [model["id"] for model in discovery["data"]] == ["Operator/team-receipt:smoke-v1"]
    assert "image" in discovery["data"][0]["architecture"]["input_modalities"]

    for invalid_model in ("operator/team-receipt:smoke-v1", "e2e-vision"):
        status, _ = llm_request(server, "/v1/chat/completions", model=invalid_model)
        assert status == 422
    assert not server.requests_log

    status, completion = llm_request(
        server,
        "/v1/chat/completions",
        model="Operator/team-receipt:smoke-v1",
        authorization="Bearer none",
    )
    assert status == 200
    assert completion["model"] == "Operator/team-receipt:smoke-v1"
    receipt = json.loads(completion["choices"][0]["message"]["content"])
    assert receipt["merchant_name"] == "Aldi Test"


def test_llm_stub_requires_opt_in_bearer_for_models_and_completions(monkeypatch, capsys):
    monkeypatch.delenv("VB_E2E_LLM_MODEL_ID", raising=False)
    token = "prepared-fixture-only-synthetic"
    monkeypatch.setenv("VB_E2E_LLM_BEARER_TOKEN", token)
    server = module("llm_stub_prepared_auth", "e2e/llm_stub/server.py")
    endpoints = [
        ("/v1/models", None),
        ("/v1/chat/completions", "e2e-vision"),
    ]

    for path, model in endpoints:
        for unauthorized in (None, "Bearer none", f"Basic {token}", f"Bearer {token}x"):
            status, body = llm_request(server, path, model=model, authorization=unauthorized)
            assert status == 401
            assert token not in json.dumps(body)
    assert not server.requests_log

    for path, model in endpoints:
        status, body = llm_request(server, path, model=model, authorization=f"Bearer {token}")
        assert status == 200
        assert token not in json.dumps(body)
    status, body = llm_request(server, "/__requests")
    assert status == 200
    assert len(body["requests"]) == 1
    assert token not in json.dumps(body)
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err


def test_llm_stub_stock_wire_behavior_without_opt_in(monkeypatch):
    monkeypatch.delenv("VB_E2E_LLM_MODEL_ID", raising=False)
    monkeypatch.delenv("VB_E2E_LLM_BEARER_TOKEN", raising=False)
    server = module("llm_stub_stock_wire", "e2e/llm_stub/server.py")
    status, discovery = llm_request(server, "/v1/models")
    assert status == 200
    assert [model["id"] for model in discovery["data"]] == ["e2e-vision", "e2e-text"]
    status, completion = llm_request(server, "/v1/chat/completions", model="e2e-text")
    assert status == 200
    assert completion["model"] == "e2e-vision"


def test_exact_ocr_v2_and_same_bytes_distinct_photo_plans():
    server = module("ocr_stub_test", "e2e/ocr_stub/server.py")
    raw = (ROOT / "e2e/llm_stub/fixtures/a.png").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    state = server.Scenario()
    state.configure({"plans": [{"source_sha256": digest, "photo_index": 1, "result": "503"}]})
    assert state.plan(digest, 0)["result"] == "success"
    assert state.plan(digest, 1)["result"] == "503"
    for index in (0, 1):
        request_id = uuid4()
        result = server.page(raw, str(request_id), index, digest, blank=False)
        parsed = PaddleOcrProvider._validated_response(
            json.dumps(result).encode(), request_id, index, digest, server.PROFILE_ID
        )
        assert parsed.blocks[0].id == f"photo-{index}-merchant"
        assert parsed.working_image.width == 320
        assert parsed.working_image.height == 480
    blank = OcrResponse.model_validate(server.page(raw, str(uuid4()), 1, digest, blank=True))
    assert blank.text == "" and blank.blocks == []
    assert base64.b64decode(blank.working_image.png_base64).startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="hash"):
        server.page(raw, str(uuid4()), 0, "0" * 64, blank=False)


def test_direct_alias_and_text_only_scenarios_use_actual_typed_sources(monkeypatch):
    monkeypatch.setenv("VB_E2E_FIXTURES_DIR", str(ROOT / "e2e/llm_stub/fixtures"))
    server = module("llm_stub_test", "e2e/llm_stub/server.py")
    raw = (ROOT / "e2e/llm_stub/fixtures/a.png").read_bytes()
    buffer = BytesIO()
    ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB").save(buffer, format="PNG")
    normalized = buffer.getvalue()
    assert (
        hashlib.sha256(normalized).hexdigest()
        == "659e82ae16064cd379be1a0780586aea7bf5d9f157b84ec8e3fcca14224a33a2"
    )
    ref = SourceRef(
        part_id=uuid4(),
        photo_index=0,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        coordinate_space="original",
    )
    prompt = source_extraction_prompt(
        ReceiptInput(mode="direct_vision", image_paths=["unused"], source_refs=[ref])
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(normalized).decode()
                        },
                    },
                ],
            }
        ]
    }
    server.configure_scenario({"scenario": "complete"})
    complete = json.loads(server._content_for(body))
    assert complete["merchant_name"] == "Aldi Test"
    assert complete["discount_total"] == "0"
    assert complete["amount_policy"]["tax_basis"] == "net"
    server.configure_scenario({"scenario": "partial"})
    partial = json.loads(server._content_for(body))
    assert partial["merchant_name"] == "Aldi Test"
    assert partial["discount_total"] is None
    ocr = module("ocr_stub_for_llm", "e2e/ocr_stub/server.py")
    page = OcrResponse.model_validate(
        ocr.page(raw, str(uuid4()), 0, ref.source_sha256, blank=False)
    )
    ref = ref.model_copy(
        update={"coordinate_space": "working", "view_id": page.working_image.view_id}
    )
    prompt = source_extraction_prompt(
        ReceiptInput(mode="ocr", image_paths=["unused"], source_refs=[ref], ocr_pages=[page])
    )
    formatted = json.loads(server._content_for({"messages": [{"role": "user", "content": prompt}]}))
    assert formatted["merchant_name"] == "Aldi Test" and formatted["discount_total"] is None
    assert formatted["field_sources"]["merchant_name"][0]["view_id"] == str(
        page.working_image.view_id
    )
    assert server.requests_log[-1]["kind"] == "ocr"
    assert server.requests_log[-1]["ordered_sources"] == [[ref.source_sha256, 0]]
    # Standard benchmark execution has validated OCR pages but no persisted source-part manifest.
    prompt = source_extraction_prompt(
        ReceiptInput(mode="ocr", image_paths=["unused"], ocr_pages=[page])
    )
    unbound = json.loads(server._content_for({"messages": [{"role": "user", "content": prompt}]}))
    assert unbound["merchant_name"] == "Aldi Test"
    assert unbound["field_sources"] == {}
    assert server.requests_log[-1]["ordered_sources"] == [[ref.source_sha256, 0]]
    with pytest.raises(ValueError, match="unregistered"):
        server._content_for(
            {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(b"unknown").decode()
                                },
                            }
                        ]
                    }
                ]
            }
        )


def test_ocr_http_multipart_barrier_blank_failure_and_timeout():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    server = module("ocr_http_test", "e2e/ocr_stub/server.py")
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    raw = (ROOT / "e2e/llm_stub/fixtures/a.png").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    url = f"http://127.0.0.1:{httpd.server_port}"

    def post(index, timeout=1000):
        return httpx.post(
            url + "/ocr",
            data={
                "request_id": str(uuid4()),
                "photo_index": str(index),
                "source_sha256": digest,
                "timeout_ms": str(timeout),
            },
            files={"image": ("a.png", raw, "image/png")},
            timeout=2,
            trust_env=False,
        )

    try:
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/info").json()["profile"]["max_new_tokens"] == 4096
            client.post(
                "/__scenario",
                json={
                    "barriers": ["second"],
                    "plans": [
                        {
                            "source_sha256": digest,
                            "photo_index": 1,
                            "result": "503",
                            "barrier": "second",
                        }
                    ],
                },
            ).raise_for_status()
            assert post(0).status_code == 200
            with ThreadPoolExecutor(1) as executor:
                future = executor.submit(post, 1)
                deadline = time.monotonic() + 2
                while len(client.get("/__requests").json()["requests"]) < 2:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                assert not future.done()
                client.post("/__release", json={"barrier": "second"}).raise_for_status()
                assert future.result().status_code == 503
            assert client.get("/ready").json()["ready"] is True
            client.post(
                "/__scenario",
                json={"plans": [{"source_sha256": digest, "photo_index": 1, "result": "empty"}]},
            )
            blank = OcrResponse.model_validate(post(1).json())
            assert not blank.text and not blank.blocks
            client.post(
                "/__scenario",
                json={"plans": [{"source_sha256": digest, "photo_index": 1, "result": "timeout"}]},
            )
            started = time.monotonic()
            assert post(1, timeout=50).status_code == 503
            assert time.monotonic() - started >= 0.05
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_transparent_generic_alias_matches_production_pixels_and_rejects_unknown(tmp_path, monkeypatch):
    from prepare_fixtures import prepare
    from vision_bill.provider.llm.ollama import OllamaProvider
    prepare(ROOT, tmp_path)
    path = tmp_path / 'fixtures/generic.png'
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    source = SourceRef(part_id=uuid4(), photo_index=0, source_sha256=digest, coordinate_space='original')
    input = ReceiptInput(mode='direct_vision', image_paths=[str(path)], source_refs=[source])
    _, _, views = OllamaProvider('http://unused-local')._prepare_receipt_input(input)
    normalized = base64.b64decode(views[0].working_image.png_base64)
    assert Image.open(BytesIO(normalized)).getpixel((0, 0)) == (255, 255, 255)
    aliases = json.loads((tmp_path / 'aliases.json').read_text())
    assert aliases['generic'] == {'source_sha256': digest, 'normalized_sha256': hashlib.sha256(normalized).hexdigest()}
    assert path.read_bytes() == original
    monkeypatch.setenv('VB_E2E_FIXTURES_DIR', str(tmp_path / 'fixtures'))
    monkeypatch.setenv('VB_E2E_REPLIES_DIR', str(tmp_path / 'replies'))
    monkeypatch.setenv('VB_E2E_ALIASES', str(tmp_path / 'aliases.json'))
    server = module('transparent_alias_test', 'e2e/llm_stub/server.py')
    server.configure_scenario({'scenario': 'complete'})
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(normalized).decode()}}
    body = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': source_extraction_prompt(input)}, image]}]}
    assert json.loads(server._content_for(body))['merchant_name'] == 'E2E Generic Store'
    image['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(b'unknown input').decode()
    with pytest.raises(ValueError, match='unregistered'):
        server._content_for(body)


@pytest.mark.parametrize(
    ("descriptions", "expected_tags"),
    [
        (["Milk", "Bread"], [["food"], ["food"]]),
        (["Unregistered product", "Milk"], [[], ["food"]]),
        (["Bread", "Unregistered product"], [["food"], []]),
    ],
)
def test_classification_stub_enriches_ordered_rows_without_changing_raw_observations(
    descriptions, expected_tags
):
    from vision_bill.model.extraction import InferenceEnvelope
    from vision_bill.provider.llm.ollama import OllamaProvider

    server = module("classification_stub_test", "e2e/llm_stub/server.py")
    server.configure_scenario({"scenario": "complete"})
    observed = json.loads((ROOT / "e2e/llm_stub/replies/a.complete.json").read_text())
    for row, description in zip(observed["line_items"], descriptions, strict=True):
        row["description"] = description
        assert "tags" not in row and "spending_category" not in row
    assert "category" not in observed
    original = json.dumps(observed)
    provider = OllamaProvider("http://unused-local")
    prototype = InferenceEnvelope.model_validate(
        {
            "provenance": {
                "requested_pipeline": "direct_vision",
                "used_pipeline": "direct_vision",
                "source_image_count": 1,
                "ordered_source_sha256": ["a" * 64],
                "model_id": "e2e-vision",
            },
            "telemetry": {"elapsed_ms": 0, "max_output_tokens": 8192},
        }
    )
    parsed = provider._parse_receipt_input(
        original,
        input=ReceiptInput(mode="direct_vision", image_paths=["unused"]),
        prototype=prototype,
    )
    raw = parsed.draft.model_dump()
    before = parsed.receipt.model_dump(exclude={"line_items": {"__all__": {"tags"}}})
    prompt = provider._classification_prompt(parsed, ["food"])
    reply = server._content_for({"messages": [{"role": "user", "content": prompt}]})
    enriched = provider._parse_classification(reply, parsed)
    assert [row.tags for row in enriched.receipt.line_items] == expected_tags
    assert enriched.receipt.model_dump(exclude={"line_items": {"__all__": {"tags"}}}) == before
    assert enriched.draft.model_dump() == raw == parsed.draft.model_dump()
    assert all(row.tags is None for row in enriched.draft.line_items)
    assert all(row.tags == [] for row in parsed.receipt.line_items)
    assert json.dumps(observed) == original
