"""Wire contracts for OpenAI inference and opt-in llama.cpp router handoff."""

import json
from contextlib import asynccontextmanager
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from vision_bill.config import LLMSettings
from vision_bill.provider.factory import get_llm_provider
from vision_bill.provider.llm import llamacpp as lifecycle_module
from vision_bill.provider.llm.openai import OpenAIProvider
from vision_bill.service.gpu_handoff import ModelHandoff

REAL_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def owned_fixture_bindings(monkeypatch):
    from handoff_fixtures import install_owned_bindings
    install_owned_bindings(monkeypatch)


def model_status(value, foreign="loaded"):
    return {
        "object": "list",
        "data": [
            {"id": "own", "aliases": [], "status": {"value": value}},
            {"id": "foreign", "aliases": [], "status": {"value": foreign}},
        ],
    }


@asynccontextmanager
async def configured(monkeypatch, handler, host="http://llama.local/llama/v1"):
    config = LLMSettings(
        provider="openai",
        host=host,
        api_key="test-key",
        model_name="own",
        temperature=0.0,
        model_lifecycle="llamacpp",
    )
    provider = get_llm_provider(config)
    try:
        assert provider.supports_model_release, "explicit llama.cpp lifecycle was ignored"
        await provider._client.close()
        transport = httpx.MockTransport(handler)
        provider._client = AsyncOpenAI(
            base_url=host,
            api_key="test-key",
            max_retries=0,
            http_client=REAL_CLIENT(transport=transport, trust_env=False),
        )

        def control_client(**kwargs):
            assert kwargs.get("trust_env") is False
            assert kwargs.get("follow_redirects") is False
            return REAL_CLIENT(transport=transport, **kwargs)

        monkeypatch.setattr(lifecycle_module, "httpx", SimpleNamespace(
            AsyncClient=control_client, Timeout=httpx.Timeout,
        ))
        yield provider
    finally:
        await provider._client.close()


@pytest.mark.asyncio
async def test_openai_wire_does_not_force_optional_reasoning_extension():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "own"
        assert "reasoning_effort" not in body
        return httpx.Response(200, json={
            "id": "chat", "object": "chat.completion", "created": 0,
            "model": "own",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    provider = OpenAIProvider("http://llama.local/v1", "test-key")
    await provider._client.close()
    provider._client = AsyncOpenAI(
        base_url="http://llama.local/v1", api_key="test-key", max_retries=0,
        http_client=REAL_CLIENT(transport=httpx.MockTransport(handler), trust_env=False),
    )
    try:
        assert await provider.send_message("own", [{"role": "user", "content": "hi"}]) == "ok"
    finally:
        await provider._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["loaded", "loading", "sleeping"])
async def test_router_release_waits_for_exact_owned_model_exit(monkeypatch, initial):
    requests = []
    statuses = iter([initial, "loading", "unloaded"])

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            assert request.url.path == "/llama/models/unload"
            assert json.loads(request.content) == {"model": "own"}
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/llama/models"
        return httpx.Response(200, json=model_status(next(statuses)))

    async with configured(monkeypatch, handler) as provider:
        await provider.release_model("own")
    assert requests == [
        ("GET", "/llama/models"), ("POST", "/llama/models/unload"),
        ("GET", "/llama/models"), ("GET", "/llama/models"),
    ]


@pytest.mark.parametrize("path", ["//", "/v1//", "/proxy/v1///"])
def test_router_rejects_ambiguous_trailing_slashes(path):
    with pytest.raises(ValueError, match="unambiguous"):
        OpenAIProvider(f"http://llama.local{path}", "test-key", model_lifecycle="llamacpp")


@pytest.mark.asyncio
@pytest.mark.parametrize("host,path", [
    ("http://llama.local", "/models"),
    ("http://llama.local/v1/", "/models"),
    ("http://llama.local/proxy/v1", "/proxy/models"),
])
async def test_already_unloaded_router_model_never_posts(monkeypatch, host, path):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == path
        return httpx.Response(200, json=model_status("unloaded"))

    async with configured(monkeypatch, handler, host) as provider:
        await provider.release_model("own")


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"object": "list", "data": []},
    {"object": "list", "data": [{"id": "alias", "aliases": ["own"],
                                "status": {"value": "unloaded"}}]},
    {"object": "list", "data": [{"id": "own"}]},
    model_status("unknown"),
    {"object": "list", "data": [
        {"id": "own", "status": {"value": "unloaded"}},
        {"id": "own", "status": {"value": "loaded"}},
    ]},
])
async def test_unknown_router_state_blocks_ocr_and_keeps_owned_journal(
    monkeypatch, tmp_path, body
):
    events = []

    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json=body)

    async with configured(monkeypatch, handler) as provider:
        handoff = ModelHandoff(tmp_path / "handoff", provider, Ocr(events))
        handoff.journal.write_text(json.dumps({
            "version": 2, "coordinator": "vision-bill-model-handoff-v2",
            "resources": handoff.resources.model_dump(), "models": ["own"],
            "generation": 1, "unsettled": False, "ocr_pending": False,
        }))
        with pytest.raises(RuntimeError, match="model_"):
            async with handoff.session(monotonic() + 2) as owner:
                owner.track("own")
                await owner.ocr_phase()
        state = json.loads((tmp_path / "handoff.json").read_text())
        assert state["models"] == ["own"]
        assert "load_ocr" not in events


@pytest.mark.asyncio
@pytest.mark.parametrize("code,body", [
    (401, {"error": "unauthorized"}),
    (302, {}),
    (200, {"success": False}),
])
async def test_failed_unload_is_not_accepted(monkeypatch, code, body):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=model_status("loaded"))
        return httpx.Response(
            code, json=body, headers={"location": "http://other.local/unload"}
        )

    async with configured(monkeypatch, handler) as provider:
        with pytest.raises((httpx.HTTPError, RuntimeError)):
            await provider.release_model("own")


class Ocr:
    lifecycle_identity = "ocr:llamacpp-fixture"

    def __init__(self, events):
        self.events = events
        self.active = False

    async def claim(self, token, deadline):
        self.active = False
        self.events.append("claim_park")

    async def load(self, token, deadline):
        self.active = True
        self.events.append("load_ocr")

    async def park(self, token, deadline):
        self.active = False
        self.events.append("park_ocr")


@pytest.mark.asyncio
async def test_openai_direct_and_ocr_calls_use_same_handoff(monkeypatch, tmp_path):
    events = []
    ocr = Ocr(events)
    state = "unloaded"

    def handler(request):
        nonlocal state
        if request.url.path == "/llama/models":
            return httpx.Response(200, json=model_status(state, foreign="unloaded"))
        if request.url.path == "/llama/models/unload":
            assert json.loads(request.content) == {"model": "own"}
            events.append("unload_llm")
            state = "unloaded"
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/llama/v1/chat/completions"
        assert not ocr.active, "LLM request overlapped native OCR"
        assert json.loads(request.content)["model"] == "own"
        events.append("chat")
        state = "loaded"
        return httpx.Response(200, json={
            "id": "chat", "object": "chat.completion", "created": 0,
            "model": "own",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    async with configured(monkeypatch, handler) as provider:
        provider.handoff = ModelHandoff(tmp_path / "handoff", provider, ocr)
        messages = [{"role": "user", "content": "hi"}]
        assert await provider.send_message("own", messages) == "ok"
        assert state == "unloaded"
        async with provider.handoff.session(monotonic() + 2) as owner:
            owner.track("own")
            await owner.ocr_phase()
            assert await provider.send_message("own", messages) == "ok"
        assert events == [
            "claim_park", "chat", "park_ocr", "unload_llm",
            "claim_park", "load_ocr", "park_ocr", "chat", "park_ocr", "unload_llm",
        ]


@pytest.mark.asyncio
async def test_router_never_unloads_blocks_successor_ocr(monkeypatch, tmp_path):
    events = []
    reads = 0

    def handler(request):
        nonlocal reads
        if request.method == "POST":
            return httpx.Response(200, json={"success": True})
        reads += 1
        return httpx.Response(200, json=model_status("unloaded" if reads == 1 else "loaded", foreign="unloaded"))

    async with configured(monkeypatch, handler) as provider:
        handoff = ModelHandoff(tmp_path / "handoff", provider, Ocr(events))
        with pytest.raises(TimeoutError):
            async with handoff.session(monotonic() + 0.03) as owner:
                owner.track("own")
                await owner.ocr_phase()
        with pytest.raises(RuntimeError, match="model_request_unsettled"):
            async with handoff.session(monotonic() + 1):
                pytest.fail("unconfirmed release admitted a successor")
        assert "load_ocr" not in events
        assert json.loads((tmp_path / "handoff.json").read_text())["models"] == ["own"]


def test_llamacpp_option_does_not_break_saved_provider_switch(settings, tmp_path, monkeypatch):
    import yaml

    from vision_bill.config import CONFIG_PATH_ENV, LLMProviderEnum, Settings, write_config_file

    settings.llm.provider = LLMProviderEnum.OPENAI
    settings.llm.model_lifecycle = "llamacpp"
    candidate = settings.model_copy(deep=True)
    # The settings API assigns editable fields before persisting the full object.
    candidate.llm.provider = LLMProviderEnum.OLLAMA
    path = tmp_path / "config.yaml"
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
    write_config_file(candidate)
    restored = Settings(**yaml.safe_load(path.read_text()))
    assert restored.llm.provider == LLMProviderEnum.OLLAMA
    assert restored.llm.model_lifecycle == "llamacpp"


@pytest.mark.asyncio
async def test_managed_openai_client_does_not_follow_inference_redirect(monkeypatch):
    import httpx2
    from openai import APIStatusError

    from vision_bill.provider.llm import openai as provider_module

    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "llama.local":
            return httpx2.Response(307, headers={
                "location": "http://other.local/v1/chat/completions",
            })
        return httpx2.Response(200, json={
            "id": "chat", "object": "chat.completion", "created": 0,
            "model": "own",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "wrong"},
                         "finish_reason": "stop"}],
        })

    sdk_client_class = provider_module.DefaultAsyncHttpxClient

    def client_factory(**kwargs):
        return sdk_client_class(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(provider_module, "DefaultAsyncHttpxClient", client_factory)
    provider = OpenAIProvider(
        "http://llama.local/v1", "test-key", model_lifecycle="llamacpp"
    )
    try:
        with pytest.raises(APIStatusError):
            await provider.send_message("own", [{"role": "user", "content": "hi"}])
        assert hosts == ["llama.local"]
    finally:
        await provider._client.close()


@pytest.mark.parametrize("mode", ["legacy", "none", "bearer"])
async def test_authentication_applies_to_inference_and_router_controls(monkeypatch, mode):
    import httpx2

    from vision_bill.provider.llm import openai as provider_module

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "ambient-admin-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "aUtHoRiZaTiOn: Bearer ambient-header\nHost: other.local")
    seen = []

    def inference(request):
        assert request.headers["Host"] == ("other.local" if mode == "legacy" else "llama.local")
        seen.append((request.method, request.url.path, request.headers.get_list("authorization")))
        return httpx2.Response(200, json={
            "id": "chat", "object": "chat.completion", "created": 0, "model": "own",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    def control(request):
        seen.append((request.method, request.url.path, request.headers.get_list("authorization")))
        return httpx.Response(200, json=model_status("unloaded", "unloaded"))

    sdk_client = provider_module.DefaultAsyncHttpxClient
    monkeypatch.setattr(provider_module, "DefaultAsyncHttpxClient", lambda **kwargs: sdk_client(
        transport=httpx2.MockTransport(inference), **kwargs,
    ))
    monkeypatch.setattr(lifecycle_module, "httpx", SimpleNamespace(
        AsyncClient=lambda **kwargs: REAL_CLIENT(transport=httpx.MockTransport(control), **kwargs),
        Timeout=httpx.Timeout,
    ))
    authentication = {"mode": "none"} if mode == "none" else {
        "mode": "bearer", "origin": "http://llama.local:80", "api_key": "selected-key",
    }
    settings = LLMSettings.model_validate({
        "provider": "openai", "host": "http://llama.local/llama/v1", "api_key": "deployment-key",
        "model_name": "own", "temperature": 0, "model_lifecycle": "llamacpp",
        "authentication": authentication if mode != "legacy" else None,
    })
    provider = get_llm_provider(settings)
    try:
        assert await provider.send_message("own", [{"role": "user", "content": "hi"}]) == "ok"
        assert await provider.resident_model_ids(deadline=monotonic() + 1) == frozenset()
    finally:
        await provider._client.close()
    wire_headers = {
        "legacy": (["Bearer deployment-key", "Bearer ambient-header"], ["Bearer deployment-key"]),
        "none": ([], []),
        "bearer": (["Bearer selected-key"], ["Bearer selected-key"]),
    }
    inference_headers, control_headers = wire_headers[mode]
    assert seen == [("POST", "/llama/v1/chat/completions", inference_headers), ("GET", "/llama/models", control_headers)]
