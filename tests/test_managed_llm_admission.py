"""Actual provider compatibility gates durable claims without loading resources."""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImageRow
from vision_bill.model.workflow import CommitResult
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider
from vision_bill.service.analysis_scheduler import AnalysisScheduler
from vision_bill.service.execution_policy import current_snapshot
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.service.submission_service import SubmissionService


@pytest.fixture
async def admission_context(settings, monkeypatch, tmp_path):
    from vision_bill import config
    from vision_bill.provider.llm import openai

    requests = []
    clients = []
    worker_ready = False
    original_client = httpx.AsyncClient

    def server(request):
        requests.append((request.method, request.url.host, request.url.path))
        if request.url.host == "ocr":
            assert request.method == "GET" and request.url.path in ("/info", "/ready")
            worker = {
                "ready": worker_ready,
                "generation": 1,
                "profile_id": "cpu-profile" if worker_ready else None,
                "runtime_fingerprint": "cpu-runtime" if worker_ready else None,
                "reason_code": None if worker_ready else "ocr_parked",
            }
            if request.url.path == "/info":
                return httpx.Response(
                    200,
                    json={
                        "contract_version": 2,
                        "profile": {"max_new_tokens": 4096},
                        "service_mode": "managed"
                        if settings.ocr.manage_model_lifecycle
                        else "standalone",
                        "worker": worker,
                    },
                )
            return httpx.Response(200 if worker_ready else 503, json=worker)
        if request.url.path == "/v1/models":
            assert request.method == "GET"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "formatter", "capabilities": ["completion", "vision"]},
                    ]
                },
            )
        if request.url.path == "/api/tags":
            assert request.method == "GET"
            return httpx.Response(200, json={"models": [{"model": "formatter"}]})
        assert request.method == "POST" and request.url.path == "/api/show"
        return httpx.Response(
            200, json={"capabilities": ["completion", "vision"], "model_info": {}}
        )

    class Client(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(server))
            clients.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", Client)
    monkeypatch.setattr(config, "settings", settings)

    def provider(kind):
        if kind == "ollama":
            return OllamaProvider("http://llm")
        return OpenAIProvider(
            "http://llm/v1",
            "test-only",
            model_lifecycle="llamacpp" if kind == "llamacpp" else "none",
        )

    async def build(
        kind="generic", *, ready=False, managed=True, enabled=True, attempts=0, adopt=True
    ):
        nonlocal worker_ready
        worker_ready = ready
        settings.ocr = OcrSettings(
            enabled=enabled,
            required=enabled,
            manage_model_lifecycle=managed,
            service_url="http://ocr",
            coordination_path=tmp_path / "coordination" / "lock",
        )
        llm = provider(kind)
        image = ImageRow(
            id=uuid4(),
            model_id="formatter",
            attempt_count=attempts,
            effective_options=current_snapshot(settings, inference_identity=llm.inference_identity),
        )
        receipts = ReceiptService(settings.images, settings.pg, llm)
        if adopt and managed and enabled and llm.supports_model_release:
            from test_handoff_binding import adopt as adopt_dedication

            llm.resident_model_ids = AsyncMock(return_value=frozenset())
            await adopt_dedication(llm.handoff, ("formatter",))
            image.effective_options = receipts.current_execution_snapshot(settings)

        async def persist_wait(observed, reason):
            assert observed is image
            image.wait_reason = reason
            return True

        async def claim_next(*, image_id, lease_seconds):
            assert image_id == image.id
            image.attempt_count += 1
            image.status = "analyzing"
            return SimpleNamespace(image=image, generation=image.generation)

        db = SimpleNamespace(
            is_ready=True,
            reap_expired_claims=AsyncMock(return_value=0),
            due_candidates=AsyncMock(return_value=[image]),
            set_wait_reason=AsyncMock(side_effect=persist_wait),
            claim_next=AsyncMock(side_effect=claim_next),
        )
        receipts._workflow_db = db
        receipts.extract_receipt = AsyncMock()
        service = SubmissionService(settings, receipts, receipts._image_service)
        service.process_claim = AsyncMock(return_value=CommitResult(stale=False))
        scheduler = AnalysisScheduler(settings, llm, receipts, None, submission_service=service)
        return SimpleNamespace(
            scheduler=scheduler,
            service=service,
            receipts=receipts,
            llm=llm,
            image=image,
            db=db,
            requests=requests,
            provider=provider,
            coordination=settings.ocr.coordination_path.parent,
        )

    yield build
    for session in clients:
        await session.aclose()


@pytest.mark.parametrize("ready", [False, True], ids=["parked", "warm"])
@pytest.mark.parametrize("attempts", [0, 2], ids=["new", "last-attempt"])
@pytest.mark.parametrize(
    "identity",
    ["generic", None, "", " \t", "unsupported", 123, "release-unsupported"],
)
async def test_incompatible_actual_llm_waits_without_claim_or_resource_side_effects(
    admission_context,
    ready,
    attempts,
    identity,
):
    ctx = await admission_context(ready=ready, attempts=attempts)
    if identity != "generic":
        if identity == "release-unsupported":
            ctx.llm.lifecycle_identity = "test:explicit-stable-identity"
        else:
            ctx.llm.supports_model_release = True
            ctx.llm.lifecycle_identity = identity
    # Establish that reachability and actual text discovery both succeed.
    assert await ctx.receipts.check_connection()
    assert [model.id for model in await ctx.receipts.get_available_models(capability="text")] == [
        "formatter"
    ]
    capabilities = await ctx.receipts.ocr_capabilities(ctx.service.settings.ocr)
    assert capabilities.ready if ready else capabilities.can_load
    snapshot = deepcopy(ctx.image.effective_options)
    generation = ctx.image.generation
    observed_requests = list(ctx.requests)
    for _ in range(ctx.image.max_attempts + 1):
        assert await ctx.scheduler.process_pending() == []
    ctx.db.claim_next.assert_not_awaited()
    ctx.service.process_claim.assert_not_awaited()
    ctx.receipts.extract_receipt.assert_not_awaited()
    assert ctx.image.attempt_count == attempts
    assert ctx.image.status == "pending" and ctx.image.wait_reason == "model_release_unsupported"
    assert ctx.image.generation == generation and ctx.image.effective_options == snapshot
    assert ctx.db.set_wait_reason.await_count == ctx.image.max_attempts + 1
    assert all(
        call.args == (ctx.image, "model_release_unsupported")
        for call in ctx.db.set_wait_reason.await_args_list
    )
    assert all(method == "GET" for method, _, _ in ctx.requests)
    assert ctx.requests == observed_requests  # Compatibility itself needs no network.
    error = (
        "model_release_unsupported"
        if identity == "release-unsupported"
        else "model_handoff_llm_identity_required"
    )
    with pytest.raises(ValueError, match=error):
        async with ctx.llm.handoff.session(monotonic() + 1):
            pytest.fail("dispatch guard admitted an incompatible provider")
    assert ctx.requests == observed_requests
    assert not ctx.coordination.exists()


@pytest.mark.parametrize("kind", ["ollama", "llamacpp"])
@pytest.mark.parametrize("ready", [False, True], ids=["parked", "warm"])
async def test_actual_managed_providers_are_admitted_without_loading_resources(
    admission_context,
    kind,
    ready,
):
    ctx = await admission_context(kind, ready=ready)
    assert await ctx.receipts.get_available_models(capability="text")
    assert await ctx.service._readiness_reason(ctx.image) is None, ctx.requests
    assert await ctx.scheduler.process_pending() == []
    ctx.db.claim_next.assert_awaited_once()
    ctx.service.process_claim.assert_awaited_once()
    assert ctx.image.attempt_count == 1 and ctx.image.wait_reason is None
    assert ctx.llm.handoff.binding_path.exists()
    assert not ctx.llm.handoff.journal.exists()
    assert {(method, path) for method, _, path in ctx.requests} <= {
        ("GET", "/v1/models"),
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
        ("GET", "/info"),
        ("GET", "/ready"),
    }


@pytest.mark.parametrize("enabled", [False, True], ids=["direct", "unmanaged-ocr"])
async def test_generic_openai_without_managed_ocr_remains_eligible(admission_context, enabled):
    ctx = await admission_context(ready=True, managed=False, enabled=enabled)
    assert await ctx.service._readiness_reason(ctx.image) is None, ctx.requests
    assert await ctx.scheduler.process_pending() == []
    ctx.db.claim_next.assert_awaited_once()
    ctx.service.process_claim.assert_awaited_once()
    assert ctx.image.attempt_count == 1 and ctx.image.wait_reason is None
    assert not ctx.llm.supports_model_release and ctx.llm.lifecycle_identity == "unsupported"
    assert not ctx.coordination.exists()


@pytest.mark.parametrize("kind", ["ollama", "llamacpp"])
async def test_restoring_compatible_provider_requires_explicit_policy_adoption(
    admission_context,
    kind,
):
    ctx = await admission_context(attempts=2)
    snapshot = deepcopy(ctx.image.effective_options)
    generation = ctx.image.generation
    for _ in range(2):
        assert await ctx.scheduler.process_pending() == []
    ctx.db.claim_next.assert_not_awaited()
    assert ctx.image.status == "pending" and ctx.image.attempt_count == 2
    from test_handoff_binding import adopt

    replacement = ctx.provider(kind)
    replacement.resident_model_ids = AsyncMock(return_value=frozenset())
    await adopt(replacement.handoff, ("formatter",))
    ctx.receipts._provider = replacement
    assert await ctx.scheduler.process_pending() == []
    ctx.db.claim_next.assert_not_awaited()
    ctx.service.process_claim.assert_not_awaited()
    assert ctx.image.attempt_count == 2 and ctx.image.wait_reason == "execution_policy_changed"
    assert ctx.image.generation == generation and ctx.image.effective_options == snapshot
    assert not replacement.handoff.journal.exists()


async def test_missing_dedication_is_a_readiness_hold(admission_context):
    ctx = await admission_context("ollama", adopt=False)
    assert await ctx.scheduler.process_pending() == []
    ctx.db.claim_next.assert_not_awaited()
    assert ctx.image.wait_reason == "model_handoff_binding_required"
    assert not ctx.llm.handoff.journal.exists()
