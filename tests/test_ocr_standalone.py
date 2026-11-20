"""Standalone request ownership, with deterministic native process simulations."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from ocr_service.api import create_app
from ocr_service.worker import OcrWorkerError, OcrWorkerManager
from test_ocr_service_api import fields, png
from test_ocr_worker_lifecycle import FakeContext, request


def cold(context):
    return OcrWorkerManager(mp_context=context, startup_timeout=0.2)


def post(client, *, timeout_ms=500, owner=None, raw=None):
    raw = png() if raw is None else raw
    data = fields(raw) | {"timeout_ms": str(timeout_ms)}
    if owner is not None:
        data["owner_token"] = owner
    return client.post("/ocr", files={"image": ("receipt.png", raw, "image/png")}, data=data)


def test_standalone_cold_requests_each_claim_load_infer_and_confirm_park():
    context = FakeContext()
    worker = cold(context)
    with TestClient(create_app(mode="standalone", worker_factory=lambda: worker)) as client:
        for generation in (1, 2):
            result = post(client)
            assert result.status_code == 200, result.text
            assert result.json()["text"] == "SHOP"
            assert worker._owner_generation == generation
            assert not worker._owner_load_allowed
            assert len(context.processes) == generation
            assert all(not process.is_alive() for process in context.processes)
            assert not context.requests[-1].original_path.exists()
        assert client.get("/info").json()["service_mode"] == "standalone"


def test_standalone_serializes_entire_lifecycle():
    context = FakeContext("hang")
    worker = cold(context)
    with TestClient(create_app(mode="standalone", worker_factory=lambda: worker)) as client:
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(post, client)
            assert context.entered.wait(1)
            second = pool.submit(post, client)
            time.sleep(0.03)
            assert len(context.processes) == 1
            assert len(context.requests) == 1
            context.release.set()
            assert first.result(2).status_code == 200
            assert second.result(2).status_code == 200
        assert len(context.processes) == 2
        assert all(not process.is_alive() for process in context.processes)


@pytest.mark.parametrize("case,status", [("bytes", 415), ("hash", 422), ("owner", 403)])
def test_standalone_rejects_bad_input_and_caller_owners_before_loading(case, status):
    context = FakeContext()
    worker = cold(context)
    with TestClient(create_app(mode="standalone", worker_factory=lambda: worker)) as client:
        raw = b"bad" if case == "bytes" else png()
        data = fields(raw)
        if case == "hash":
            data["source_sha256"] = "0" * 64
        if case == "owner":
            data["owner_token"] = "1:caller"
        result = client.post("/ocr", files={"image": ("x", raw, "image/png")}, data=data)
        assert result.status_code == status
        assert not context.processes


@pytest.mark.parametrize("action", ["claim", "load", "park"])
def test_standalone_forbids_external_lifecycle(action):
    context = FakeContext()
    worker = cold(context)
    with TestClient(create_app(mode="standalone", worker_factory=lambda: worker)) as client:
        result = client.post(f"/lifecycle/{action}", json={"owner_token": "1:external"})
        assert result.status_code == 403
        assert not context.processes


def test_standalone_deadline_includes_cold_start_and_parks():
    context = FakeContext(init_blocked=True)
    worker = cold(context)
    with TestClient(create_app(mode="standalone", worker_factory=lambda: worker)) as client:
        started = time.monotonic()
        result = post(client, timeout_ms=25)
        assert result.status_code == 504
        assert time.monotonic() - started < 0.18
        assert not worker._owner_load_allowed
        assert all(not process.is_alive() for process in context.processes)


def test_standalone_failed_inference_parks_and_never_reports_success():
    context = FakeContext("hang")
    worker = cold(context)
    with TestClient(create_app(mode="standalone", worker_factory=lambda: worker)) as client:
        result = post(client, timeout_ms=25)
        assert result.status_code == 504
        assert not worker._owner_load_allowed
        assert all(not process.is_alive() for process in context.processes)
        assert not context.requests[0].original_path.exists()


def test_standalone_failed_park_preserves_original_until_confirmed_death(monkeypatch):
    context = FakeContext()
    worker = cold(context)
    real_park = worker.park

    def failed_park(owner):
        raise OcrWorkerError("ocr_worker_unstoppable", 503, cleanup_safe=False)

    monkeypatch.setattr(worker, "park", failed_park)
    with TestClient(create_app(mode="standalone", worker_factory=lambda: worker)) as client:
        result = post(client)
        assert result.status_code == 503
        path = context.requests[0].original_path
        assert path.exists()
        real_park(worker._owner)
        path.unlink()


def test_standalone_waiting_deadline_does_not_stop_active_request(tmp_path):
    context = FakeContext("hang")
    worker = cold(context)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(
            worker.infer_standalone, request(tmp_path), deadline_monotonic=time.monotonic() + 1
        )
        assert context.entered.wait(1)
        with pytest.raises(OcrWorkerError, match="ocr_deadline_exceeded"):
            worker.infer_standalone(request(tmp_path), deadline_monotonic=time.monotonic() + 0.02)
        assert context.processes[0].is_alive()
        context.release.set()
        assert first.result(1).text == "SHOP"
    worker.stop()


def test_cancelled_standalone_http_request_confirms_park_before_finishing():
    context = FakeContext("hang")
    worker = cold(context)

    async def scenario():
        app = create_app(mode="standalone", worker_factory=lambda: worker)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://ocr"
            ) as client,
        ):
            async with asyncio.timeout(1):
                while (await client.get("/ready")).json()["reason_code"] == "ocr_starting":
                    await asyncio.sleep(0)
            raw = png()
            task = asyncio.create_task(
                client.post("/ocr", files={"image": ("x.png", raw, "image/png")}, data=fields(raw))
            )
            assert await asyncio.to_thread(context.entered.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not worker._owner_load_allowed
            assert all(not process.is_alive() for process in context.processes)
            assert not context.requests[0].original_path.exists()

    asyncio.run(scenario())


def test_invalid_service_mode_is_configuration_error():
    with pytest.raises(ValueError, match="mode"):
        create_app(mode="typo")


@pytest.mark.parametrize("already_queued", [False, True])
def test_standalone_readiness_crash_cannot_restart_between_inference_and_park(
    tmp_path, monkeypatch, already_queued
):
    context = FakeContext("hang")
    worker = cold(context)
    real_infer = worker.infer
    real_thread = threading.Thread
    queued = []
    poll_completed = threading.Event()

    class QueuedRecovery:
        def __init__(self, target, args):
            self.target, self.args = target, args

        def start(self):
            queued.append((self.target, self.args))

    def schedule_thread(*args, **kwargs):
        if kwargs.get("name") == "ocr-idle-recovery":
            return QueuedRecovery(kwargs["target"], kwargs["args"])
        return real_thread(*args, **kwargs)

    def infer_then_run_queued_recovery(*args, **kwargs):
        try:
            return real_infer(*args, **kwargs)
        finally:
            assert poll_completed.wait(1)
            # Deliberately give readiness recovery the _active lock after
            # inference releases it, before the standalone finally can park.
            for target, recovery_args in queued:
                target(*recovery_args)

    monkeypatch.setattr(worker, "infer", infer_then_run_queued_recovery)
    try:
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(
                worker.infer_standalone,
                request(tmp_path),
                deadline_monotonic=time.monotonic() + 1,
            )
            assert context.entered.wait(1)
            monkeypatch.setattr(threading, "Thread", schedule_thread)
            with worker._control:
                if already_queued:
                    # Also exercise the synchronized policy check for a callback
                    # queued before standalone admission disabled recovery.
                    queued.append((worker._recover_idle, (worker._generation,)))
                context.processes[0].dead.set()
                observation = worker.readiness()
                assert observation.reason_code == "ocr_worker_crashed"
            poll_completed.set()
            with pytest.raises(OcrWorkerError, match="ocr_worker_crashed"):
                job.result(2)
        assert len(context.processes) == 1
        assert all(not process.is_alive() for process in context.processes)
        assert not worker._owner_load_allowed
        if not already_queued:
            assert queued == []
    finally:
        poll_completed.set()
        worker.stop()


def test_disconnected_standalone_http_request_reaps_without_success():
    context = FakeContext("hang")
    worker = cold(context)

    async def scenario():
        app = create_app(mode="standalone", worker_factory=lambda: worker)
        raw = png()
        built = httpx.Request(
            "POST", "http://ocr/ocr", files={"image": ("x.png", raw, "image/png")}, data=fields(raw)
        )
        body = built.read()
        delivered = False
        replies = []

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.to_thread(context.entered.wait, 1)
            return {"type": "http.disconnect"}

        async def send(message):
            replies.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/ocr",
            "raw_path": b"/ocr",
            "query_string": b"",
            "headers": [(key.lower(), value) for key, value in built.headers.raw],
            "client": ("127.0.0.1", 1234),
            "server": ("ocr", 80),
            "root_path": "",
        }
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.01)
            started = time.monotonic()
            await app(scope, receive, send)
            assert time.monotonic() - started < 0.3
            assert all(message.get("status") != 200 for message in replies)
            assert not worker._owner_load_allowed
            assert context.requests, replies
            assert not context.requests[0].original_path.exists()
            assert all(not process.is_alive() for process in context.processes)

    asyncio.run(scenario())
