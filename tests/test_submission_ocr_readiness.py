"""The actual preclaim path only admits OCR that the pipeline can dispatch."""

import asyncio
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from ocr_service.api import create_app
from ocr_service.worker import OcrWorkerManager
from test_ocr_worker_lifecycle import FakeContext

from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImageRow
from vision_bill.provider.llm.base import ModelInfo
from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider
from vision_bill.service.analysis_scheduler import AnalysisScheduler
from vision_bill.service.execution_policy import current_snapshot
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.service.submission_service import SubmissionService


async def submission_context(settings, monkeypatch, tmp_path, *, managed):
    from vision_bill import config

    settings.ocr = OcrSettings(
        enabled=True,
        required=True,
        manage_model_lifecycle=managed,
        service_url="http://ocr",
        coordination_path=tmp_path / "handoff.lock",
    )
    monkeypatch.setattr(config, "settings", settings)
    image = ImageRow(
        id=uuid4(), model_id=settings.llm.model_name, effective_options=current_snapshot(settings, inference_identity="ollama:" + "d" * 64)
    )
    model = ModelInfo(settings.llm.model_name)
    model.capabilities = ["completion"]
    llm = SimpleNamespace(
        inference_identity="ollama:" + "d" * 64,
        supports_model_release=True,
        lifecycle_identity="cpu-fixture:formatter",
        check_connection=AsyncMock(return_value=True),
        resident_model_ids=AsyncMock(return_value=frozenset()),
        get_available_models=AsyncMock(return_value=[model]),
    )
    receipts = ReceiptService(settings.images, settings.pg, llm)
    if managed:
        from test_handoff_binding import adopt

        await adopt(receipts.execution_handoff(settings.ocr), (image.model_id,))
        image.effective_options = receipts.current_execution_snapshot(settings)
    db = SimpleNamespace(
        is_ready=True,
        reap_expired_claims=AsyncMock(return_value=0),
        due_candidates=AsyncMock(return_value=[image]),
    )
    waits = []

    async def persist_wait(observed, reason):
        assert observed is image
        image.wait_reason = reason
        waits.append(reason)
        return True

    claims = []

    async def claim_next(*, lease_seconds, image_id):
        claims.append((lease_seconds, image_id))
        # This is the only operation that spends the fake durable row's budget.
        image.attempt_count += 1

    db.set_wait_reason = persist_wait
    db.claim_next = claim_next
    receipts._workflow_db = db
    service = SubmissionService(settings, receipts, receipts._image_service)
    return service, receipts, llm, image, waits, claims


@pytest.mark.parametrize("managed", [False, True])
async def test_real_parked_service_only_admitted_with_managed_load_path(
    settings, monkeypatch, tmp_path, managed
):
    service, receipts, llm, image, waits, claims = await submission_context(
        settings, monkeypatch, tmp_path, managed=managed
    )
    context = FakeContext()
    worker = OcrWorkerManager(mp_context=context, startup_timeout=0.2)
    app = create_app(mode="managed" if managed else "standalone", worker_factory=lambda: worker)
    original_client = httpx.AsyncClient
    requests = []

    async def record(request):
        requests.append((request.method, request.url.path))

    def client(**kwargs):
        return original_client(
            **kwargs, transport=httpx.ASGITransport(app=app), event_hooks={"request": [record]}
        )

    monkeypatch.setattr(httpx, "AsyncClient", client)
    async with app.router.lifespan_context(app):
        # Factory publication runs in a background task, just as on real startup.
        end = monotonic() + 1
        while True:
            capabilities = await receipts.ocr_capabilities(settings.ocr)
            if capabilities.can_load:
                break
            assert monotonic() < end
            await asyncio.sleep(0.005)
        assert not capabilities.ready and capabilities.reason_code == "ocr_parked"
        assert capabilities.profile_id is None
        if managed:
            assert await service.ready_for(image)
            assert waits == [None]
            llm.get_available_models.assert_awaited_once_with(capability="text")
        else:
            scheduler = AnalysisScheduler(settings, llm, receipts, None, submission_service=service)
            for _ in range(4):
                assert await scheduler.process_pending() == []
                assert image.attempt_count == 0
                assert image.status == "pending" and image.wait_reason == "ocr_unavailable"
            assert waits == ["ocr_unavailable"] * 4
            llm.get_available_models.assert_not_awaited()
        assert claims == []
        assert not context.processes
        assert requests and set(requests) == {("GET", "/info"), ("GET", "/ready")}


@pytest.mark.parametrize("service_mode", [None, "standalone"])
async def test_ready_unmanaged_service_including_legacy_missing_mode_is_eligible(
    settings, monkeypatch, tmp_path, service_mode
):
    service, _receipts, llm, image, waits, claims = await submission_context(
        settings, monkeypatch, tmp_path, managed=False
    )
    ready = {
        "ready": True,
        "generation": 1,
        "profile_id": "qualified-cpu-profile",
        "runtime_fingerprint": "qualified-cpu-runtime",
    }
    info = {"contract_version": 2, "profile": {"max_new_tokens": 4096}, "worker": ready}
    if service_mode is not None:
        info["service_mode"] = service_mode
    original_client = httpx.AsyncClient

    def server(request):
        assert request.method == "GET" and request.url.path in ("/info", "/ready")
        return httpx.Response(200, json=info if request.url.path == "/info" else ready)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(server)),
    )
    capabilities = await PaddleOcrProvider("http://ocr", managed_lifecycle=False).get_capabilities()
    assert capabilities.ready and capabilities.profile_id == "qualified-cpu-profile"
    assert await service.ready_for(image)
    assert waits == [None] and claims == [] and image.attempt_count == 0
    llm.get_available_models.assert_awaited_once_with(capability="text")


@pytest.mark.parametrize("code", ["ocr_gpu_out_of_memory", "ocr_model_load_failed", "ocr_runtime_identity_mismatch"])
async def test_readiness_exposes_typed_native_error_without_claiming_receipt(settings, monkeypatch, tmp_path, code):
    from vision_bill.provider.db.workflow_db import READINESS_WAIT_REASONS

    service, _receipts, llm, image, waits, claims = await submission_context(
        settings, monkeypatch, tmp_path, managed=False
    )
    original_client = httpx.AsyncClient

    def server(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={"contract_version": 2, "profile": {}})
        assert request.url.path == "/ready"
        return httpx.Response(503, json={
            "ready": False, "generation": 1, "reason_code": code,
        })

    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(server)),
    )
    assert not await service.ready_for(image)
    assert waits == [code] and waits[0] in READINESS_WAIT_REASONS
    assert claims == [] and image.attempt_count == 0
    llm.get_available_models.assert_not_awaited()
