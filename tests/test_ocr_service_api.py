import asyncio
import hashlib
import threading
import time
from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_ocr_worker_lifecycle import FakeContext, manager


def png():
    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, "PNG")
    return output.getvalue()


def fields(raw):
    return {
        "request_id": str(uuid4()),
        "photo_index": "0",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "timeout_ms": "500",
    }


def test_feature_requires_lifecycle_api():
    import importlib.util

    assert importlib.util.find_spec("ocr_service.api") is not None, "lifecycle API missing"


@pytest.mark.parametrize("operation", ["load", "page"])
@pytest.mark.parametrize("interrupt", ["disconnect", "cancel", "deadline"])
@pytest.mark.asyncio
async def test_interrupted_managed_http_operation_reaps_before_cleanup(operation, interrupt):
    from ocr_service.api import create_app
    from ocr_service.worker import OcrWorkerManager

    context = FakeContext("hang", init_blocked=operation == "load")
    worker = OcrWorkerManager(mp_context=context, startup_timeout=0.5)
    app = create_app(mode="managed", worker_factory=lambda: worker)
    disconnected = asyncio.Event()

    async def connection(scope, receive, send):
        body_finished = False

        async def receive_after_upload():
            nonlocal body_finished
            if body_finished:
                await disconnected.wait()
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body"):
                # The next read follows multipart parsing; disconnect independently
                # of the response completing, as a real network connection can.
                body_finished = True
            return message

        await app(scope, receive_after_upload, send)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=connection), base_url="http://ocr"
        ) as client,
    ):
        owner = {"owner_token": "1:managed"}
        async with asyncio.timeout(2):
            while (await client.get("/ready")).json().get("reason_code") == "ocr_starting":
                await asyncio.sleep(0)
            assert (await client.post("/lifecycle/claim", json=owner)).status_code == 200
            timeout_ms = 150 if interrupt == "deadline" else 4000
            if operation == "load":
                task = asyncio.create_task(client.post(
                    "/lifecycle/load", json=owner | {"timeout_ms": timeout_ms},
                ))
                assert await asyncio.to_thread(context.started.wait, 1)
            else:
                assert (await client.post("/lifecycle/load", json=owner)).status_code == 200
                raw = png()
                task = asyncio.create_task(client.post(
                    "/ocr", files={"image": ("receipt.png", raw, "image/png")},
                    data=fields(raw) | owner | {"timeout_ms": str(timeout_ms)},
                ))
                assert await asyncio.to_thread(context.entered.wait, 1)
            if interrupt == "disconnect":
                disconnected.set()
                reply = await task
                assert (reply.status_code, reply.json()["detail"]) == (499, "ocr_cancelled")
            elif interrupt == "deadline":
                reply = await task
                assert (reply.status_code, reply.json()["detail"]) == (504, "ocr_deadline_exceeded")
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert worker.readiness().reason_code == (
            "ocr_deadline_exceeded" if interrupt == "deadline" else "ocr_cancelled"
        )
        assert worker.readiness().can_claim_after_failed_load
        assert len(context.processes) == 1
        assert not context.processes[0].is_alive()
        if operation == "page":
            assert not context.requests[0].original_path.exists()
        else:
            assert not context.requests


def test_park_fences_delayed_load_http_until_new_claim_without_starting_child():
    from ocr_service.api import create_app
    from ocr_service.worker import OcrWorkerManager

    context = FakeContext()
    worker = OcrWorkerManager(mp_context=context, startup_timeout=.2)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        owner = {"owner_token": "1:first"}
        assert client.post("/lifecycle/claim", json=owner).status_code == 200
        assert client.post("/lifecycle/load", json=owner).status_code == 200
        assert client.post("/lifecycle/load", json=owner).status_code == 200
        assert len(context.processes) == 1
        assert client.post("/lifecycle/park", json=owner).status_code == 200
        assert not context.processes[0].is_alive()
        assert client.post("/lifecycle/park", json=owner).status_code == 200
        delayed = client.post("/lifecycle/load", json=owner)
        assert delayed.status_code == 409
        assert delayed.json()["detail"] == "ocr_owner_phase_closed"
        assert len(context.processes) == 1
        assert client.get("/ready").json()["reason_code"] == "ocr_parked"
        successor = {"owner_token": "2:second"}
        assert client.post("/lifecycle/claim", json=successor).status_code == 200
        assert client.post("/lifecycle/load", json=successor).status_code == 200
        assert len(context.processes) == 2


def test_park_and_fresh_claim_preserve_identity_invalidation_until_successful_load():
    from ocr_service.api import create_app
    from ocr_service.worker import OcrWorkerManager

    context = FakeContext()
    worker = OcrWorkerManager(mp_context=context, startup_timeout=.2)
    first, second, third = ({"owner_token": f"{i}:receipt"} for i in (1, 2, 3))
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        assert client.post("/lifecycle/claim", json=first).status_code == 200
        assert client.post("/lifecycle/load", json=first).status_code == 200
        verified = client.get("/info").json()["runtime_identity"]
        assert verified["runtime_fingerprint"] == "runtime"
        assert client.post("/lifecycle/park", json=first).status_code == 200
        assert client.post("/lifecycle/claim", json=second).status_code == 200
        context.init_fail = "identity"
        failure = client.post("/lifecycle/load", json=second)
        assert failure.status_code == 503 and failure.json()["detail"] == "ocr_runtime_identity_mismatch"
        assert client.get("/info").json()["runtime_identity"] == {}
        assert client.post("/lifecycle/park", json=second).status_code == 200
        assert client.get("/info").json()["runtime_identity"] == {}
        # A concurrent capabilities caller may already hold the first /info.
        assert client.get("/ready").json()["runtime_identity"] == {}
        assert client.post("/lifecycle/claim", json=third).status_code == 200
        assert client.get("/info").json()["runtime_identity"] == {}
        context.init_fail = False
        assert client.post("/lifecycle/load", json=third).status_code == 200
        assert client.get("/info").json()["runtime_identity"] == verified


def test_health_does_not_imply_readiness_and_metadata_is_safe():
    from ocr_service.api import create_app
    from ocr_service.worker import OcrWorkerManager

    context = FakeContext(init_fail=True, init_blocked=True)
    worker = OcrWorkerManager(mp_context=context, startup_timeout=2)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        deadline = time.monotonic() + 1
        while client.get("/ready").json()["reason_code"] == "ocr_starting":
            assert time.monotonic() < deadline
        assert not context.started.is_set()
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["reason_code"] == "ocr_parked"
        assert client.post("/lifecycle/claim", json={"owner_token": "1:test"}).status_code == 200
        context.init_release.set()
        assert client.post("/lifecycle/load", json={"owner_token": "1:test"}).status_code == 503
        ready = client.get("/ready")
        assert client.get("/health").status_code == 200
        assert ready.status_code == 503
        assert ready.json()["reason_code"] == "ocr_init_failed"
        info = client.get("/info").json()
        assert info["contract_version"] == 2
        assert info["worker"]["reason_code"] == "ocr_init_failed"
        assert "model_dirs" not in str(info)
        assert "exception" not in str(info)


def test_valid_png_returns_contract_v2_and_cleans_file():
    from ocr_service.api import create_app

    context = FakeContext()
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        raw = png()
        data = fields(raw)
        result = client.post("/ocr", files={"image": ("receipt.png", raw, "image/png")}, data=data)
        assert result.status_code == 200, result.text
        assert result.json()["contract_version"] == 2
        assert result.json()["request_id"] == data["request_id"]
        assert not context.requests[0].original_path.exists()


@pytest.mark.parametrize(
    "case,status",
    [("hash", 422), ("bytes", 415), ("jpeg", 415), ("oversize", 413), ("timeout", 422)],
)
def test_malformed_or_oversized_payload_rejected_before_inference(case, status):
    from ocr_service.api import create_app

    context = FakeContext()
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker, max_upload_bytes=100)) as client:
        raw = png()
        data = fields(raw)
        media_type = "image/png"
        if case == "hash":
            data["source_sha256"] = "0" * 64
        if case == "bytes":
            raw = b"not png"
            data = fields(raw)
        if case == "jpeg":
            media_type = "image/jpeg"
        if case == "oversize":
            raw = b"x" * 101
        if case == "timeout":
            data["timeout_ms"] = "600001"
        result = client.post("/ocr", files={"image": ("x.png", raw, media_type)}, data=data)
        assert result.status_code == status, result.text
        assert not context.requests


def test_timeout_terminates_before_http_removes_input():
    from ocr_service.api import create_app

    context = FakeContext("hang")
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        raw = png()
        data = fields(raw) | {"timeout_ms": "25"}
        result = client.post("/ocr", files={"image": ("x.png", raw, "image/png")}, data=data)
        assert result.status_code == 504
        assert result.json()["detail"] == "ocr_deadline_exceeded"
        assert not context.requests[0].original_path.exists()


def test_response_encoding_obeys_deadline_without_reaping_a_finished_page(monkeypatch):
    from types import SimpleNamespace

    from ocr_service import api

    worker = manager(FakeContext())
    offset = 0
    validate = api.validate_result

    def validate_after_budget(result):
        nonlocal offset
        validate(result)
        offset = 30

    monkeypatch.setattr(api, "validate_result", validate_after_budget)
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset))
    with TestClient(api.create_app(worker_factory=lambda: worker)) as client:
        raw = png()
        response = client.post("/ocr", files={"image": ("x.png", raw, "image/png")}, data=fields(raw))
        assert (response.status_code, response.json()["detail"]) == (504, "ocr_deadline_exceeded")
        assert worker.readiness().ready


def test_shutdown_does_not_block_event_loop():
    import asyncio

    from ocr_service.api import create_app

    from vision_bill.model.ocr import WorkerReadiness

    entered, release = threading.Event(), threading.Event()

    class SlowStop:
        def start(self):
            pass

        def stop(self):
            entered.set()
            assert release.wait(1)

        def readiness(self):
            return WorkerReadiness(ready=False, generation=1, reason_code="ocr_stopped")

    async def scenario():
        app = create_app(worker_factory=SlowStop)

        async def heartbeat():
            await asyncio.to_thread(entered.wait, 1)
            release.set()

        task = asyncio.create_task(heartbeat())
        async with app.router.lifespan_context(app):
            pass
        await task

    asyncio.run(scenario())


def test_shutdown_during_delayed_factory_never_starts_worker():
    import asyncio

    from ocr_service.api import create_app

    entered, release = threading.Event(), threading.Event()
    events = []

    class Worker:
        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    def factory():
        entered.set()
        assert release.wait(1)
        return Worker()

    async def scenario():
        app = create_app(worker_factory=factory)
        async with app.router.lifespan_context(app):
            assert await asyncio.to_thread(entered.wait, 1)
            asyncio.get_running_loop().call_soon(release.set)

    asyncio.run(scenario())
    assert "start" not in events
    assert "stop" in events


def test_busy_worker_keeps_http_health_and_ready_responsive():
    from concurrent.futures import ThreadPoolExecutor

    from ocr_service.api import create_app

    context = FakeContext("hang")
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        raw = png()
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(
                client.post, "/ocr", files={"image": ("x.png", raw, "image/png")}, data=fields(raw)
            )
            assert context.entered.wait(1)
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
            context.release.set()
            assert job.result(1).status_code == 200


@pytest.mark.parametrize("change", ["hash", "size", "base64"])
def test_working_png_integrity_must_match_wire_metadata(change, tmp_path):
    from fastapi import HTTPException
    from ocr_service.api import validate_result
    from test_ocr_worker_lifecycle import request, response

    result = response(request(tmp_path))
    if change == "hash":
        result.working_image.sha256 = "0" * 64
    elif change == "size":
        result.working_image.width = 3
    else:
        result.working_image.png_base64 = "invalid-base64"
    with pytest.raises(HTTPException) as error:
        validate_result(result)
    assert error.value.status_code == 502
    assert error.value.detail == "ocr_invalid_result"


def test_explicit_runtime_startup_inputs_are_consumed(tmp_path, monkeypatch):
    from ocr_service.api import configured_worker
    from ocr_service.profile import RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="rocm",
        gpu_index=0,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "sealed.json",
        expected_packages={},
        expected_paddle_commit="abcdef1",
        image_identity="test-image",
    )
    settings = tmp_path / "runtime.json"
    settings.write_text(runtime.model_dump_json())
    monkeypatch.setenv("OCR_RUNTIME_SETTINGS_PATH", str(settings))
    monkeypatch.delenv("OCR_QUALIFICATION_IMAGE_PATH", raising=False)
    monkeypatch.setenv("OCR_STARTUP_TIMEOUT_SECONDS", "17")
    worker = configured_worker()
    assert worker._runtime == runtime
    assert worker._startup_timeout == 17
    worker.stop()


def test_prepared_worker_reports_parked_identity_without_loading(tmp_path, monkeypatch):
    import json

    from ocr_service.api import configured_worker, create_app
    from ocr_service.profile import MODEL_NAMES, PreparedRuntimeSettings
    from pydantic import ValidationError

    runtime = {
        "schema_version": 2,
        "accelerator": "cuda",
        "gpu_index": 0,
        "cache_dir": str(tmp_path / "cache"),
        "model_dirs": {name: str(tmp_path / name) for name in MODEL_NAMES},
        "model_content": {name: "a" * 64 for name in MODEL_NAMES},
        "device_identity": {"accelerator": "cuda", "gpu_index": 0, "uuid": "1" * 32},
        "expected_packages": {"paddleocr": "3.7.0"},
        "expected_paddle_commit": "a" * 40,
        "image_identity": "sha256:" + "b" * 64,
    }
    settings = tmp_path / "runtime.json"
    settings.write_text(json.dumps(runtime))
    monkeypatch.setenv("OCR_RUNTIME_SETTINGS_PATH", str(settings))
    monkeypatch.delenv("OCR_QUALIFICATION_IMAGE_PATH", raising=False)
    worker = configured_worker()
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        for _ in range(2):
            assert client.get("/ready").json()["reason_code"] == "ocr_parked"
            identity = client.get("/info").json()["runtime_identity"]
            assert len(identity["profile_id"]) == len(identity["runtime_fingerprint"]) == 64
        assert worker._proc is None

    runtime["model_content"].pop("PP-DocLayoutV3")
    with pytest.raises(ValidationError):
        PreparedRuntimeSettings.model_validate(runtime)


@pytest.mark.parametrize(
    "source_format,media_type",
    [("JPEG", "image/jpeg"), ("HEIF", "image/heic"), ("GIF", "image/gif")],
)
def test_original_ingress_preserves_raw_identity_and_real_normalization(source_format, media_type):
    from ocr_service.api import create_app
    from ocr_service.images import normalize_source
    from test_ocr_worker_lifecycle import response

    from vision_bill.model.ocr import WorkerReadiness

    original = BytesIO()
    image = Image.new("RGB", (9, 5), "red")
    if source_format == "JPEG":
        exif = Image.Exif()
        exif[274] = 6
        image.save(original, "JPEG", exif=exif)
    elif source_format == "HEIF":
        from pillow_heif import from_pillow

        from_pillow(image).save(original)
    else:
        image.save(original, "GIF", save_all=True, append_images=[Image.new("RGB", (9, 5), "blue")])
    raw = original.getvalue()
    seen = []

    class NormalizingWorker:
        def start(self):
            pass

        def stop(self):
            pass

        def readiness(self):
            return WorkerReadiness(ready=True, generation=1)

        def infer(self, request, *, deadline_monotonic, owner=None, cancel_event=None):
            seen.append(request.original_path.read_bytes())
            with normalize_source(request.original_path, request.source_sha256) as normalized:
                derived = normalized.path.read_bytes()
                with Image.open(normalized.path) as decoded:
                    size = decoded.size
                    assert decoded.mode == "RGB"
                result = response(request)
                import base64

                result.working_image.png_base64 = base64.b64encode(derived).decode()
                result.working_image.sha256 = hashlib.sha256(derived).hexdigest()
                result.working_image.width, result.working_image.height = size
                result.working_image.transforms = normalized.transforms
                result.telemetry = {"findings": normalized.findings, "note": "grüße \n"}
                return result

    with TestClient(create_app(worker_factory=NormalizingWorker)) as client:
        result = client.post(
            "/ocr", files={"image": ("original", raw, media_type)}, data=fields(raw)
        )
    assert result.status_code == 200, result.text
    from ocr_contract import OcrResponse

    typed = OcrResponse.parse_wire(result.content)
    assert result.content == typed.model_dump_json().encode("utf-8")
    assert b"gr\xc3\xbc\xc3\x9fe" in result.content
    page = result.json()
    assert seen == [raw]
    assert page["source_sha256"] == hashlib.sha256(raw).hexdigest()
    transform = page["working_image"]["transforms"][0]
    assert transform["source_format"] == source_format
    assert transform["selected_frame"] == 0
    if source_format == "JPEG":
        assert transform["exif_orientation"] == 6
        assert (page["working_image"]["width"], page["working_image"]["height"]) == (5, 9)
    if source_format == "GIF":
        assert transform["frame_count"] == 2
        assert page["telemetry"]["findings"] == ["multiframe_first_frame_only"]


def test_ingress_rejects_unsupported_actual_format_and_forged_hash():
    from ocr_service.api import create_app

    context = FakeContext()
    worker = manager(context)
    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, "TIFF")
    raw = output.getvalue()
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        result = client.post(
            "/ocr", files={"image": ("original", raw, "image/jpeg")}, data=fields(raw)
        )
        assert result.status_code == 415
        jpeg = BytesIO()
        Image.new("RGB", (2, 2)).save(jpeg, "JPEG")
        raw = jpeg.getvalue()
        data = fields(raw) | {"source_sha256": "0" * 64}
        result = client.post("/ocr", files={"image": ("original", raw, "image/jpeg")}, data=data)
        assert result.status_code == 422
        assert not context.requests


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_native_api_caps_canonical_utf8_metadata_without_counting_png(tmp_path, monkeypatch, delta):
    from fastapi import HTTPException
    from ocr_contract import limits
    from ocr_service.api import validate_result
    from test_ocr_worker_lifecycle import request, response

    result = response(request(tmp_path))
    result.telemetry = {"label": "grüße \\ ＼n"}  # Multibyte and escaped wire characters.
    empty_png_wire = result.model_copy(
        update={"working_image": result.working_image.model_copy(update={"png_base64": ""})}
    ).model_dump_json().encode()
    monkeypatch.setattr(limits, "MAX_RESPONSE_METADATA_BYTES", len(empty_png_wire) - delta)
    if delta <= 0:
        validate_result(result)
    else:
        with pytest.raises(HTTPException) as error:
            validate_result(result)
        assert error.value.status_code == 502
        assert error.value.detail == "ocr_invalid_result"


@pytest.mark.parametrize("size", [200_000, 200_001])
def test_native_api_retains_compact_ocr_text_cap(tmp_path, size):
    from fastapi import HTTPException
    from ocr_service.api import validate_result
    from test_ocr_worker_lifecycle import request, response

    result = response(request(tmp_path))
    result.text = "  " + "é" * size + "  "
    if size == 200_000:
        validate_result(result)
    else:
        with pytest.raises(HTTPException) as error:
            validate_result(result)
        assert error.value.status_code == 502
        assert error.value.detail == "ocr_invalid_result"


def test_native_result_rejects_crc_correct_invalid_deflate(tmp_path):
    from fastapi import HTTPException
    from ocr_service.api import validate_result
    from test_ocr_worker_lifecycle import request, response, with_invalid_deflate

    with pytest.raises(HTTPException) as error:
        validate_result(with_invalid_deflate(response(request(tmp_path))))
    assert error.value.status_code == 502


def test_failed_source_write_removes_partial_private_upload(tmp_path, monkeypatch):
    from ocr_service import api

    original = api.tempfile.NamedTemporaryFile
    context = FakeContext()
    worker = manager(context)

    def partial_write(**kwargs):
        source = original(dir=tmp_path, **kwargs)
        write = source.write

        def fail(raw):
            write(raw[:5])
            raise OSError("no space left")

        source.write = fail
        return source

    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", partial_write)
    raw = png()
    with TestClient(api.create_app(worker_factory=lambda: worker)) as client:
        with pytest.raises(OSError, match="no space left"):
            client.post("/ocr", files={"image": ("x.png", raw, "image/png")}, data=fields(raw))
    assert not list(tmp_path.iterdir())
    assert not context.requests
