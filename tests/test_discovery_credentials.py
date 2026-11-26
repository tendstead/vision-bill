import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vision_bill.provider.llm.base import ModelDiscoveryError
from vision_bill.provider.llm.connection import _model_ids
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider


@pytest.mark.parametrize("native", [True, False])
def test_saved_catalog_short_key_allows_unrelated_id(native):
    def discover(model):
        body = {"models": [{"model": model}]} if native else {"data": [{"id": model}]}
        return _model_ids(bytearray(json.dumps(body).encode()), native, {"a"})

    assert discover("llama").models == ["llama"]
    assert discover("a").status == "invalid_response"


@pytest.mark.parametrize("kind", ["openai", "ollama"])
@pytest.mark.parametrize("field", ["id", "digest", "capability", "parameter_size"])
async def test_active_catalog_rejects_credential_in_published_field(kind, field):
    key = "synthetic-marker"
    values = {"id": "example", "digest": "digest", "capability": "text", "parameter_size": "7B"}
    values[field] = "sample-" + key
    capabilities = ["vision", values["capability"]]
    details = {"parameter_size": values["parameter_size"]}
    if kind == "openai":
        provider = OpenAIProvider("http://catalog.invalid/v1", key)
        entry = {"model": values["id"], "capabilities": capabilities, "details": details, "digest": values["digest"]}
        raw = SimpleNamespace(content=json.dumps({"models": [entry]}).encode())
        provider._client.models.with_raw_response.list = AsyncMock(return_value=SimpleNamespace(http_response=raw))
    else:
        provider = OllamaProvider("http://catalog.invalid", api_key=key)
        entry = SimpleNamespace(model=values["id"], digest=values["digest"], details=SimpleNamespace(**details))
        provider._client.list = AsyncMock(return_value=SimpleNamespace(models=[entry]))
        provider._client.show = AsyncMock(return_value=SimpleNamespace(capabilities=capabilities, modelinfo={}))

    with pytest.raises(ModelDiscoveryError, match="^Invalid formatter model list$"):
        await provider.get_available_models()


@pytest.mark.parametrize(
    ("environment", "setting"),
    [
        ("OPENAI_ORG_ID", "synthetic-header"),
        ("OPENAI_PROJECT_ID", "synthetic-header"),
        ("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer synthetic-header"),
        ("OPENAI_CUSTOM_HEADERS", "X-Deployment: synthetic-header"),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
async def test_openai_catalog_uses_connection_header_snapshot(monkeypatch, environment, setting, explicit):
    monkeypatch.setenv(environment, setting)
    provider = OpenAIProvider("http://catalog.invalid/v1", "none", explicit_authentication=explicit)
    monkeypatch.delenv(environment)
    raw = SimpleNamespace(content=b'{"data":[{"id":"synthetic-header"}]}')
    provider._client.models.with_raw_response.list = AsyncMock(return_value=SimpleNamespace(http_response=raw))
    if explicit:
        assert [model.id for model in await provider.get_available_models(capability="text")] == ["synthetic-header"]
    else:
        with pytest.raises(ModelDiscoveryError, match="^Invalid formatter model list$"):
            await provider.get_available_models(capability="text")


@pytest.mark.parametrize("explicit", [True, False])
async def test_openai_catalog_distinguishes_explicit_bearer_from_legacy_sentinel(explicit):
    provider = OpenAIProvider("http://catalog.invalid/v1", "none", explicit_authentication=explicit)
    raw = SimpleNamespace(content=b'{"data":[{"id":"none"}]}')
    provider._client.models.with_raw_response.list = AsyncMock(return_value=SimpleNamespace(http_response=raw))
    if explicit:
        with pytest.raises(ModelDiscoveryError, match="^Invalid formatter model list$"):
            await provider.get_available_models(capability="text")
    else:
        assert [model.id for model in await provider.get_available_models(capability="text")] == ["none"]


@pytest.mark.parametrize("kind", ["openai", "ollama"])
@pytest.mark.parametrize("key", ["32768", "2768", "a"])
async def test_active_catalog_matches_numeric_capacity_exactly(kind, key):
    capacity = 32768
    if kind == "openai":
        provider = OpenAIProvider("http://catalog.invalid/v1", key)
        entry = {"id": "llama", "context_length": capacity}
        raw = SimpleNamespace(content=json.dumps({"data": [entry]}).encode())
        provider._client.models.with_raw_response.list = AsyncMock(return_value=SimpleNamespace(http_response=raw))
    else:
        provider = OllamaProvider("http://catalog.invalid", api_key=key)
        entry = SimpleNamespace(model="llama", digest=None, details=None)
        provider._client.list = AsyncMock(return_value=SimpleNamespace(models=[entry]))
        details = SimpleNamespace(capabilities=["text"], modelinfo={"llama.context_length": capacity})
        provider._client.show = AsyncMock(return_value=details)

    if key == str(capacity):
        with pytest.raises(ModelDiscoveryError, match="^Invalid formatter model list$"):
            await provider.get_available_models(capability="text")
    else:
        models = await provider.get_available_models(capability="text")
        assert [(model.id, model.context_window_tokens) for model in models] == [("llama", capacity)]
