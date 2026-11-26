"""Typed sidecar transport contract, exercised through real HTTP serialization."""

import base64
import hashlib
import time
from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from PIL import Image

from vision_bill.config import OcrSettings
from vision_bill.model.ocr_errors import OcrGpuOutOfMemoryError
from vision_bill.provider.factory import get_ocr_provider
from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider


def image_file(tmp_path, fmt="PNG"):
    path = tmp_path / "misleading.jpg"
    Image.new("RGB", (3, 4), "white").save(path, format=fmt)
    return path


def wire_page(index=0, digest="a" * 64, request_id=None, text="Milk 2.00"):
    data = BytesIO()
    Image.new("RGB", (3, 4)).save(data, format="PNG")
    raw = data.getvalue()
    return {
        "contract_version": 2,
        "request_id": str(request_id or uuid4()),
        "photo_index": index,
        "source_sha256": digest,
        "text": text,
        "blocks": [],
        "working_image": {
            "view_id": str(uuid4()),
            "width": 3,
            "height": 4,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "png_base64": base64.b64encode(raw).decode(),
        },
        "profile_id": "fixed-profile",
        "runtime_fingerprint": "runtime",
        "telemetry": {},
    }


def http_client(monkeypatch, handler):
    original = httpx.AsyncClient
    calls = []

    def make(**kwargs):
        calls.append(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", make)
    return calls


def test_disabled_ocr_constructs_no_client():
    assert get_ocr_provider(OcrSettings()) is None


async def test_readiness_requires_info_and_ready(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if request.url.path == "/info":
            return httpx.Response(
                200, json={"contract_version": 2, "profile": {"profile_id": "fixed-profile"}}
            )
        return httpx.Response(
            503,
            json={
                "ready": False,
                "generation": 1,
                "profile_id": "fixed-profile",
                "reason_code": "ocr_starting",
            },
        )

    calls = http_client(monkeypatch, handler)
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    assert hasattr(provider, "get_capabilities"), "typed readiness is missing"
    capabilities = await provider.get_capabilities()
    assert not capabilities.ready and capabilities.reachable
    assert capabilities.reason_code == "ocr_starting"
    assert requests == ["/info", "/ready"]
    assert all(call["trust_env"] is False for call in calls)


@pytest.mark.parametrize("status,ready", [(200, False), (200, True), (503, True), (503, "false")])
async def test_readiness_ignores_oom_with_bad_status_or_ready_value(monkeypatch, status, ready):
    def handler(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={"contract_version": 2, "profile": {}})
        return httpx.Response(status, json={
            "ready": ready,
            "generation": 1,
            "profile_id": "fixed-profile",
            "reason_code": "ocr_gpu_out_of_memory",
        })

    http_client(monkeypatch, handler)
    capabilities = await PaddleOcrProvider("http://local").get_capabilities()
    assert not capabilities.ready and not capabilities.can_load and capabilities.reachable
    assert capabilities.reason_code == "ocr_unavailable"


async def test_original_png_bytes_mime_hash_and_remaining_deadline(monkeypatch, tmp_path):
    path = image_file(tmp_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    def handler(request):
        body = request.content
        assert b"Content-Type: image/png" in body
        assert raw in body and digest.encode() in body
        import re

        request_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', body)[1].decode()
        timeout = int(re.search(rb'name="timeout_ms"\r\n\r\n([^\r]+)', body)[1])
        assert 0 < timeout <= 2000
        return httpx.Response(200, json=wire_page(digest=digest, request_id=request_id))

    calls = http_client(monkeypatch, handler)
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    assert hasattr(provider, "extract_page"), "typed page extraction is missing"
    page = await provider.extract_page(
        path,
        photo_index=0,
        source_sha256=digest,
        deadline=time.monotonic() + 2,
        expected_profile_id="fixed-profile",
    )
    assert page.source_sha256 == digest
    assert calls[0]["trust_env"] is False


@pytest.mark.parametrize(
    "change", ["photo_index", "source_sha256", "profile_id", "request_id", "text", "png_base64"]
)
async def test_rejects_malformed_or_reordered_result(monkeypatch, tmp_path, change):
    path = image_file(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def handler(request):
        import re

        request_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', request.content)[1].decode()
        page = wire_page(digest=digest, request_id=request_id)
        if change == "png_base64":
            page["working_image"][change] = "not png"
        else:
            page[change] = {
                "photo_index": 1,
                "source_sha256": "b" * 64,
                "profile_id": "other",
                "request_id": str(uuid4()),
                "text": " ",
            }[change]
        return httpx.Response(200, json=page)

    http_client(monkeypatch, handler)
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    assert hasattr(provider, "extract_page"), "typed page extraction is missing"
    with pytest.raises(ValueError):
        await provider.extract_page(
            path,
            photo_index=0,
            source_sha256=digest,
            deadline=time.monotonic() + 2,
            expected_profile_id="fixed-profile",
        )


async def test_paddleocr_extract_text_http_failure(monkeypatch, tmp_path):
    path = image_file(tmp_path)
    http_client(monkeypatch, lambda request: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        await PaddleOcrProvider("http://local", managed_lifecycle=False).extract_text(path)


@pytest.mark.parametrize("action", ["load", "inference", "claim", "park"])
@pytest.mark.parametrize(
    "status,detail,is_oom",
    [
        (503, b'{"detail":"ocr_gpu_out_of_memory"}', True),
        (500, b'{"detail":"ocr_gpu_out_of_memory"}', False),
        (503, b'{"detail":"ocr_gpu_out_of_memory: private backend message"}', False),
        (503, b'{"detail":{"code":"ocr_gpu_out_of_memory"}}', False),
        (503, b'{"detail":"ocr_gpu_out_of_memory","extra":true}', False),
        (503, b'{"detail":"ocr_gpu_out_of_memory"', False),
        (503, b'{"detail":"ocr_generic","detail":"ocr_gpu_out_of_memory"}', False),
        (503, b'{"detail":"ocr_gpu_out_of_memory"}' + b' ' * 4096, False),
    ],
)
async def test_ocr_http_failure_recognizes_only_bounded_native_oom(
    monkeypatch, tmp_path, action, status, detail, is_oom
):
    class ErrorBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield detail[:20]
            yield detail[20:]

    http_client(monkeypatch, lambda request: httpx.Response(status, stream=ErrorBody()))
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    if is_oom and action in {"load", "inference"}:
        error_type = OcrGpuOutOfMemoryError
    else:
        error_type = httpx.HTTPStatusError
    with pytest.raises(error_type) as error:
        if action == "inference":
            await provider.extract_text(image_file(tmp_path))
        else:
            await getattr(provider, action)("1:test-owner", time.monotonic() + 2)
    if is_oom and action in {"load", "inference"}:
        assert str(error.value) == "ocr_gpu_out_of_memory"


@pytest.mark.parametrize("body", [[], {"contract_version": 2, "profile": {}, "worker": "bad"}])
async def test_malformed_info_is_unready_instead_of_crashing(monkeypatch, body):
    def handler(request):
        return httpx.Response(
            200,
            json=body
            if request.url.path == "/info"
            else {"ready": True, "generation": 1, "profile_id": "profile"},
        )

    http_client(monkeypatch, handler)
    result = await PaddleOcrProvider("http://local", managed_lifecycle=False).get_capabilities()
    assert not result.ready and result.reason_code == "ocr_unavailable"


async def test_ocr_limit_comes_from_matching_ready_profile(monkeypatch):
    def handler(request):
        if request.url.path == "/info":
            return httpx.Response(
                200,
                json={
                    "contract_version": 2,
                    "profile": {"max_new_tokens": 4096},
                    "worker": {"profile_id": "fixed-profile"},
                },
            )
        return httpx.Response(
            200, json={"ready": True, "generation": 1, "profile_id": "fixed-profile"}
        )

    http_client(monkeypatch, handler)
    capabilities = await PaddleOcrProvider("http://local", managed_lifecycle=False).get_capabilities()
    assert capabilities.ready
    assert getattr(capabilities, "ocr_max_new_tokens", None) == 4096


async def test_original_byte_preparation_is_bounded_before_http(monkeypatch, tmp_path):
    import asyncio
    import threading
    from pathlib import Path

    path = image_file(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = Path.open
    gate, entered, finished = threading.Event(), threading.Event(), threading.Event()

    def read(selected, *args, **kwargs):
        if selected != path:
            return original(selected, *args, **kwargs)
        entered.set()
        gate.wait(0.5)
        try:
            return original(selected, *args, **kwargs)
        finally:
            finished.set()

    calls = http_client(monkeypatch, lambda request: httpx.Response(500))
    monkeypatch.setattr(Path, "open", read)
    timer = threading.Timer(0.5, gate.set)
    timer.start()
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await PaddleOcrProvider("http://local", managed_lifecycle=False).extract_page(
                path, photo_index=0, source_sha256=digest, deadline=began + 0.08
            )
        assert entered.is_set()
        assert time.monotonic() - began < 0.3
    finally:
        gate.set()
        timer.cancel()
    assert await asyncio.to_thread(finished.wait, 0.5)
    assert calls == []


async def test_capabilities_exposes_qualified_live_runtime(monkeypatch):
    def handler(request):
        ready = {
            "ready": True,
            "generation": 2,
            "profile_id": "fixed-profile",
            "runtime_fingerprint": "qualified-runtime",
        }
        return httpx.Response(
            200,
            json=(
                {"contract_version": 2, "profile": {}, "worker": ready}
                if request.url.path == "/info"
                else ready
            ),
        )

    http_client(monkeypatch, handler)
    assert (
        await PaddleOcrProvider("http://local", managed_lifecycle=False).get_capabilities()
    ).runtime_fingerprint == "qualified-runtime"


@pytest.mark.parametrize("worker_ready", [False, True], ids=["parked", "ready"])
@pytest.mark.parametrize("managed_lifecycle", [False, True], ids=["unmanaged", "managed"])
@pytest.mark.parametrize(
    "mode_fields,managed_compatible,unmanaged_compatible",
    [
        pytest.param({}, True, True, id="legacy-missing"),
        pytest.param({"service_mode": "managed"}, True, False, id="managed-mode"),
        pytest.param({"service_mode": "standalone"}, False, True, id="standalone-mode"),
        pytest.param({"service_mode": None}, False, False, id="null-mode"),
        pytest.param({"service_mode": "unknown"}, False, False, id="unknown-mode"),
        pytest.param({"service_mode": ""}, False, False, id="empty-mode"),
        pytest.param({"service_mode": False}, False, False, id="boolean-mode"),
        pytest.param({"service_mode": 1}, False, False, id="numeric-mode"),
        pytest.param({"service_mode": ["managed"]}, False, False, id="list-mode"),
        pytest.param({"service_mode": {"mode": "managed"}}, False, False, id="object-mode"),
    ],
)
async def test_capabilities_requires_compatible_explicit_service_mode(
    monkeypatch, worker_ready, managed_lifecycle, mode_fields,
    managed_compatible, unmanaged_compatible,
):
    requests = []
    worker = {
        "ready": worker_ready,
        "generation": 1,
        "profile_id": "fixed-profile" if worker_ready else None,
        "reason_code": None if worker_ready else "ocr_parked",
    }

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "contract_version": 2,
                "profile": {"max_new_tokens": 4096},
                "worker": worker,
                **mode_fields,
            })
        assert request.url.path == "/ready"
        return httpx.Response(200 if worker_ready else 503, json=worker)

    http_client(monkeypatch, handler)
    provider = PaddleOcrProvider("http://local", managed_lifecycle=managed_lifecycle)
    capabilities = await provider.get_capabilities()
    compatible = managed_compatible if managed_lifecycle else unmanaged_compatible
    assert capabilities.enabled and capabilities.configured and capabilities.reachable
    if compatible:
        assert capabilities.ready is worker_ready
        assert capabilities.can_load is not worker_ready
        assert capabilities.reason_code == worker["reason_code"]
        assert requests == [("GET", "/info"), ("GET", "/ready")]
    else:
        assert not capabilities.ready
        assert not capabilities.can_load
        assert capabilities.reason_code == (
            "ocr_mode_mismatch" if mode_fields.get("service_mode") in ("managed", "standalone")
            else "ocr_unavailable"
        )
        assert requests == [("GET", "/info")]


async def test_managed_extract_text_checks_selected_model_before_native_effects(monkeypatch, tmp_path):
    import json
    from unittest.mock import AsyncMock

    from test_handoff_binding import ResidentLlm

    from vision_bill.config import settings
    from vision_bill.distribution.handoff import adopt_binding
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service.gpu_handoff import ModelHandoff

    path = image_file(tmp_path)
    events = []
    provider = PaddleOcrProvider("http://owned-native.local")
    provider.claim = AsyncMock(side_effect=lambda *args: events.append("claim"))
    provider.park = AsyncMock(side_effect=lambda *args: events.append("park"))
    provider.load = AsyncMock(side_effect=lambda *args: events.append("load"))
    provider.extract_page = AsyncMock()
    llm = ResidentLlm(events)
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, provider)
    await adopt_binding(handoff, ["own"], dedicated_coordinated_requests=True,
                        external_eager_loading_disabled=True, deadline=time.monotonic() + 1)
    state = {"version": 2, "coordinator": "vision-bill-model-handoff-v2", "resources": handoff.resources.model_dump(),
             "models": ["own"], "generation": 7, "unsettled": False, "ocr_pending": False}
    handoff.journal.write_text(json.dumps(state))
    before = handoff.journal.read_bytes()
    events.clear()
    monkeypatch.setattr(settings.ocr, "enabled", True)
    monkeypatch.setattr(settings.ocr, "manage_model_lifecycle", True)
    monkeypatch.setattr(settings.ocr, "coordination_path", handoff.path)
    monkeypatch.setattr(settings.llm, "model_name", "unadopted")
    monkeypatch.setattr("vision_bill.provider.factory.get_llm_provider", lambda config: llm)
    with pytest.raises(HandoffWait) as caught:
        await provider.extract_text(path)
    assert caught.value.status.reason_code == "model_handoff_model_not_adopted"
    assert events == []
    provider.claim.assert_not_awaited()
    provider.load.assert_not_awaited()
    provider.park.assert_not_awaited()
    provider.extract_page.assert_not_awaited()
    assert handoff.journal.read_bytes() == before
    assert json.loads(handoff.journal.read_bytes())["generation"] == 7


async def test_app_ocr_provider_stops_at_oversized_chunked_body(monkeypatch, tmp_path):
    from ocr_contract import limits

    class OversizedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"01234567"
            yield b"8"
            pytest.fail("provider read another chunk after crossing the response cap")

    monkeypatch.setattr(limits, "MAX_OCR_RESPONSE_BYTES", 8)
    path = image_file(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    http_client(monkeypatch, lambda _: httpx.Response(200, stream=OversizedBody()))
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    with pytest.raises(ValueError, match="ocr_response_too_large"):
        await provider.extract_page(
            path, photo_index=0, source_sha256=digest, deadline=time.monotonic() + 2
        )


def test_app_provider_rejects_crc_correct_invalid_deflate(tmp_path):
    from test_ocr_worker_lifecycle import request, response, with_invalid_deflate

    result = with_invalid_deflate(response(request(tmp_path)))
    with pytest.raises(ValueError, match="ocr_invalid_working_image"):
        PaddleOcrProvider._validated_response(
            result.model_dump_json().encode(), result.request_id, result.photo_index,
            result.source_sha256, result.profile_id,
        )


@pytest.mark.parametrize("action,status,code", [
    ("load", 503, "ocr_model_load_failed"),
    ("load", 503, "ocr_runtime_identity_mismatch"),
    ("inference", 502, "ocr_invalid_result"),
    ("inference", 502, "ocr_inference_failed"),
    ("inference", 503, "ocr_worker_crashed"),
    ("inference", 504, "ocr_deadline_exceeded"),
])
async def test_typed_native_reason_matches_http_contract(monkeypatch, tmp_path, action, status, code):
    http_client(monkeypatch, lambda _: httpx.Response(status, json={"detail": code}))
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    with pytest.raises(RuntimeError, match=f"^{code}$"):
        if action == "load":
            await provider.load("1:owner", time.monotonic() + 2)
        else:
            await provider.extract_text(image_file(tmp_path))


async def test_parked_prepared_identity_is_used_before_first_receipt(monkeypatch):
    parked = {"ready": False, "generation": 1, "reason_code": "ocr_parked"}
    def handler(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "contract_version": 2, "service_mode": "managed", "profile": {}, "worker": parked,
                "runtime_identity": {"profile_id": "profile", "runtime_fingerprint": "prepared"},
                "last_qualified_identity": {"profile_id": "old", "runtime_fingerprint": "stale"},
            })
        return httpx.Response(503, json=parked)
    http_client(monkeypatch, handler)
    caps = await PaddleOcrProvider("http://local").get_capabilities()
    assert caps.can_load and not caps.ready
    assert caps.profile_id == "profile" and caps.runtime_fingerprint == "prepared"


@pytest.mark.parametrize("code,expected_identity", [
    ("ocr_startup_timeout", "prepared"),
    ("ocr_cancelled", "prepared"),
    ("ocr_runtime_identity_mismatch", None),
])
async def test_reclaimed_load_preserves_usable_cache_identity_only(monkeypatch, code, expected_identity):
    failed = {"ready": False, "generation": 2, "reason_code": code, "can_claim_after_failed_load": True}
    def handler(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "contract_version": 2, "service_mode": "managed", "profile": {"max_new_tokens": 4096},
                "worker": failed, "runtime_identity": {"profile_id": "fixed", "runtime_fingerprint": "prepared"},
            })
        return httpx.Response(503, json=failed)
    http_client(monkeypatch, handler)
    caps = await PaddleOcrProvider("http://local").get_capabilities()
    assert not caps.ready and caps.can_load and caps.reason_code == code
    assert caps.runtime_fingerprint == expected_identity


async def test_identity_invalidation_between_info_and_parked_ready_cannot_reuse_cache(monkeypatch):
    parked = {"ready": False, "generation": 1, "reason_code": "ocr_parked"}
    def handler(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "contract_version": 2, "service_mode": "managed", "profile": {}, "worker": parked,
                "runtime_identity": {"profile_id": "p", "runtime_fingerprint": "stale"},
            })
        return httpx.Response(503, json={**parked, "runtime_identity": {}})
    http_client(monkeypatch, handler)
    caps = await PaddleOcrProvider("http://local").get_capabilities()
    assert caps.can_load and caps.runtime_fingerprint is None and caps.profile_id is None
