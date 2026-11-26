from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vision_bill.provider.llm.base import ModelDiscoveryError
from vision_bill.provider.llm.openai import OpenAIProvider


@pytest.fixture
def mock_client():
    """Fixture to mock the openai.AsyncOpenAI client."""
    with patch("vision_bill.provider.llm.openai.AsyncOpenAI") as mocked_client:
        client = AsyncMock()
        mocked_client.return_value = client
        yield client


@pytest.mark.asyncio
async def test_check_connection_success(mock_client):
    mock_client.models.list.return_value = MagicMock()

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    assert await provider.check_connection() is True
    mock_client.models.list.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_connection_connection_error(mock_client):
    from openai import APIConnectionError

    mock_client.models.list.side_effect = APIConnectionError(request=MagicMock())

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    assert await provider.check_connection() is False
    mock_client.models.list.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_connection_status_error(mock_client):
    from openai import APIStatusError

    mock_client.models.list.side_effect = APIStatusError(
        message="server error", response=MagicMock(status_code=500), body=None
    )

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    assert await provider.check_connection() is False
    mock_client.models.list.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_connection_unexpected_error_propagates(mock_client):
    # Only openai connection/status errors map to "unreachable" (the SDK
    # wraps every transport failure into those); anything else is a bug
    # and must surface.
    mock_client.models.list.side_effect = RuntimeError("boom")

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with pytest.raises(RuntimeError, match="boom"):
        await provider.check_connection()
    mock_client.models.list.assert_awaited_once()


@pytest.mark.asyncio
async def test_constructor_passes_host_verbatim():
    with patch("vision_bill.provider.llm.openai.AsyncOpenAI") as mocked_client:
        OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
        kwargs = mocked_client.call_args.kwargs
        assert kwargs["base_url"] == "http://localhost:8642/v1"
        assert kwargs["api_key"] == "none"


import json as _json


def _hybrid_body(models_entries, data_entries):
    """Build a realistic llama.cpp hybrid /v1/models body."""
    return _json.dumps(
        {
            "models": models_entries,
            "object": "list",
            "data": data_entries,
        }
    ).encode()


def _raw_response(body_bytes):
    """Build a stand-in for `await client.models.with_raw_response.list()`."""
    http_response = MagicMock()
    http_response.content = body_bytes
    return MagicMock(http_response=http_response)


def _ollama_entry(model, capabilities, digest="", parameter_size=""):
    return {
        "name": model,
        "model": model,
        "digest": digest,
        "tags": [],
        "capabilities": capabilities,
        "details": {"parameter_size": parameter_size, "format": "gguf", "quantization_level": ""},
    }


def _openai_entry(model_id, n_params):
    return {
        "id": model_id,
        "aliases": [model_id],
        "tags": [],
        "object": "model",
        "created": 1788563881,
        "owned_by": "llamacpp",
        "meta": {
            "n_params": n_params,
            "size": 13135396864,
            "ftype": "Q3_K - Large",
            "n_ctx": 32000,
        },
    }


@pytest.mark.asyncio
async def test_get_available_models_filters_to_vision(mock_client):
    body = _hybrid_body(
        [
            _ollama_entry("qwen-vision:27b", ["completion", "multimodal"], digest="sha256:abc123"),
            _ollama_entry("textonly:7b", ["completion"]),
        ],
        [
            _openai_entry("qwen-vision:27b", 27320697856),
            _openai_entry("textonly:7b", 7000000000),
        ],
    )
    mock_client.models.with_raw_response.list.return_value = _raw_response(body)

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    models = await provider.get_available_models()

    assert [m.id for m in models] == ["qwen-vision:27b"]
    assert models[0].capabilities == ["completion", "multimodal"]
    assert models[0].digest == "sha256:abc123"
    # details.parameter_size is empty on this build -> formatted from meta.n_params
    assert models[0].parameter_size == "27.3B"


@pytest.mark.asyncio
async def test_get_available_models_prefers_details_parameter_size(mock_client):
    body = _hybrid_body(
        [
            _ollama_entry(
                "gemma:7b", ["completion", "vision"], digest="sha256:xyz", parameter_size="7B"
            )
        ],
        [_openai_entry("gemma:7b", 7000000000)],
    )
    mock_client.models.with_raw_response.list.return_value = _raw_response(body)

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    models = await provider.get_available_models()

    assert [m.id for m in models] == ["gemma:7b"]
    assert models[0].parameter_size == "7B"
    # "vision" is an accepted capability marker too
    assert models[0].capabilities == ["completion", "vision"]


@pytest.mark.asyncio
async def test_get_available_models_falls_back_to_data_entries(mock_client):
    # Build without the ollama-style "models" array (other OpenAI-compatible
    # servers): ids come from data[].id, vision from architecture.input_modalities.
    body = _json.dumps(
        {
            "object": "list",
            "data": [
                {
                    "id": "vl-model",
                    "object": "model",
                    "created": 1,
                    "owned_by": "x",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                },
                {
                    "id": "text-model",
                    "object": "model",
                    "created": 1,
                    "owned_by": "x",
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                },
            ],
        }
    ).encode()
    mock_client.models.with_raw_response.list.return_value = _raw_response(body)

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    models = await provider.get_available_models()

    assert [m.id for m in models] == ["vl-model"]
    assert models[0].digest is None
    assert models[0].parameter_size is None


@pytest.mark.asyncio
async def test_get_available_models_rejects_empty_ids(mock_client):
    entry = _ollama_entry("", ["completion", "multimodal"])
    body = _hybrid_body([entry], [_openai_entry("", 1000000)])
    mock_client.models.with_raw_response.list.return_value = _raw_response(body)

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with pytest.raises(ModelDiscoveryError, match="Invalid formatter model list"):
        await provider.get_available_models()


def _chat_response(content: str | None):
    """Build a stand-in for a chat.completions.create() response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    return MagicMock(choices=[choice])


@pytest.mark.asyncio
async def test_send_message_returns_content(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response("hello there")

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    result = await provider.send_message("model-x", [{"role": "user", "content": "hi"}])

    assert result == "hello there"
    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_message_empty_content(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response(None)

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    result = await provider.send_message("model-x", [{"role": "user", "content": "hi"}])

    assert result == ""


@pytest.mark.asyncio
async def test_send_message_sends_temperature_without_optional_reasoning_extension(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response("ok")

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none", temperature=0.3)
    await provider.send_message("model-x", [{"role": "user", "content": "hi"}])

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "model-x"
    assert kwargs["temperature"] == 0.3
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_update_runtime_settings_changes_temperature(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response("ok")

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none", temperature=0.1)
    provider.update_runtime_settings(temperature=0.9)
    await provider.send_message("model-x", [{"role": "user", "content": "hi"}])

    assert mock_client.chat.completions.create.call_args.kwargs["temperature"] == 0.9


def test_build_image_messages_data_uri():
    import base64

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    image_path = Path("fake.jpg")

    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch(
            "vision_bill.provider.llm.openai.Path.read_bytes",
            return_value=b"\xff\xd8\xff fake jpeg bytes",
        ),
    ):
        messages = provider._build_image_messages(image_path, tags=["coffee"])

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    parts = messages[0]["content"]
    assert parts[0]["type"] == "text"
    assert "Prefer tags from this list: coffee" in parts[0]["text"]
    assert parts[1]["type"] == "image_url"
    expected_b64 = base64.b64encode(b"\xff\xd8\xff fake jpeg bytes").decode()
    assert parts[1]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


def test_build_image_messages_png_mime():
    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")

    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch("vision_bill.provider.llm.openai.Path.read_bytes", return_value=b"\x89PNG fake"),
    ):
        messages = provider._build_image_messages(Path("fake.png"))

    parts = messages[0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_image_messages_missing_file_raises():
    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")

    with patch("vision_bill.provider.llm.openai.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            provider._build_image_messages(Path("nope.jpg"))


VALID_RECEIPT_JSON = (
    '{"confidence": 95, "merchant_name": "Test Shop", "merchant_address": "123 Main St", '
    '"receipt_number": "REC001", "date": "2024-08-06", "time": "14:00", "currency": "USD", '
    '"line_items": [{"description": "Coffee", "quantity": 1, "unit_price": 5.00, '
    '"total_price": 5.00}], '
    '"taxes": [{"name": "VAT", "rate": 0.20, "amount": 1.00}], "subtotal": 5.00, '
    '"discount_total": 0.00, "tax_total": 1.00, "tip": 0.50, "total": 6.50, '
    '"payment_method": "credit_card"}'
)


@pytest.mark.asyncio
async def test_analyse_receipt_success(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response(VALID_RECEIPT_JSON)

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch("vision_bill.provider.llm.openai.Path.read_bytes", return_value=b"fake image bytes"),
    ):
        result = await provider.analyse_receipt_from_model("model-x", Path("fake.jpg"))

    assert result.merchant_name == "Test Shop"
    assert float(result.total) == 6.50
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_analyse_receipt_retry_on_empty(mock_client):
    mock_client.chat.completions.create.side_effect = [
        _chat_response(None),
        _chat_response(VALID_RECEIPT_JSON),
    ]

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch("vision_bill.provider.llm.openai.Path.read_bytes", return_value=b"fake image bytes"),
    ):
        result = await provider.analyse_receipt_from_model("model-x", Path("fake.jpg"))

    assert result.merchant_name == "Test Shop"
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_analyse_receipt_repair_loop(mock_client):
    bad_content = '{"merchant": "Bad JSON"'  # Missing closing brace
    mock_client.chat.completions.create.side_effect = [
        _chat_response(bad_content),
        _chat_response(VALID_RECEIPT_JSON),
    ]

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch("vision_bill.provider.llm.openai.Path.read_bytes", return_value=b"fake image bytes"),
    ):
        result = await provider.analyse_receipt_from_model("model-x", Path("fake.jpg"))

    assert result.merchant_name == "Test Shop"
    assert mock_client.chat.completions.create.call_count == 2

    # The second call must contain the repair feedback in its message history.
    second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    flat = _flatten_message_contents(second_call_messages)
    assert any("Please respond again with ONLY corrected JSON" in part for part in flat)
    assert any(bad_content in part for part in flat)


@pytest.mark.asyncio
async def test_analyse_receipt_no_vision_message(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response(
        "I cannot provide an analysis because I cannot process the image."
    )

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch("vision_bill.provider.llm.openai.Path.read_bytes", return_value=b"fake image bytes"),
        pytest.raises(ValueError, match=".*supports vision capabilities.*"),
    ):
        await provider.analyse_receipt_from_model("model-x", Path("fake.jpg"))


@pytest.mark.asyncio
async def test_analyse_receipt_fails_after_retries(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response('{"merchant": "Broken"')

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch("vision_bill.provider.llm.openai.Path.read_bytes", return_value=b"fake image bytes"),
        pytest.raises(ValueError, match=r".*Failed to get a valid response.*3 attempts.*"),
    ):
        await provider.analyse_receipt_from_model("model-x", Path("fake.jpg"))

    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_analyse_receipt_with_metadata_returns_telemetry(mock_client):
    mock_client.chat.completions.create.return_value = _chat_response(VALID_RECEIPT_JSON)

    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    with (
        patch("vision_bill.provider.llm.openai.Path.exists", return_value=True),
        patch("vision_bill.provider.llm.openai.Path.read_bytes", return_value=b"fake image bytes"),
    ):
        result = await provider.analyse_receipt_with_metadata("model-x", Path("fake.jpg"))

    assert result.receipt.merchant_name == "Test Shop"
    assert result.attempts == 1
    assert result.elapsed_ms >= 0


def _flatten_message_contents(messages):
    """Yield text parts out of OpenAI-style messages (str or content-part lists)."""
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
    return parts


@pytest.mark.asyncio
@pytest.mark.parametrize("hybrid", [False, True])
async def test_discovery_distinguishes_image_input_from_unrelated_modalities(mock_client, hybrid):
    entries = [
        {"id": "output-only", "input_modalities": ["text"], "meta": {"output_modalities": ["image"]}},
        {"id": "explicit-text", "meta": {"input_modalities": ["text"]}, "capabilities": ["vision"]},
        {"id": "architecture-input", "architecture": {"input_modalities": ["image", "text"]}},
        {"id": "meta-input", "meta": {"input_modalities": ["image", "text"]}},
        {"id": "top-input", "input_modalities": ["image", "text"]},
        {"id": "declared", "meta": {"output_modalities": ["image"]}},
        {"id": "vision-name-without-input-evidence", "meta": {"other": ["image"]}},
        {"id": "text-capabilities", "capabilities": ["completion"]},
    ]
    hybrid_entries = [
        _ollama_entry(model["id"], model.get("capabilities", [])) for model in entries
    ] if hybrid else []
    mock_client.models.with_raw_response.list.return_value = _raw_response(
        _json.dumps({"models": hybrid_entries, "data": entries}).encode()
    )
    provider = OpenAIProvider(
        host="http://localhost:8642/v1",
        api_key="none",
        vision_model_ids=["output-only", "explicit-text", "declared", "text-capabilities"],
    )

    expected = {"architecture-input", "meta-input", "top-input", "declared"}
    assert {model.id for model in await provider.get_available_models()} == expected
    all_models = await provider.get_available_models(capability="all")
    assert {model.id for model in all_models} == {entry["id"] for entry in entries}
    assert {
        model.id for model in all_models if {"vision", "multimodal"}.intersection(model.capabilities)
    } == expected


@pytest.mark.asyncio
async def test_hybrid_model_list_keeps_data_only_ids_without_assuming_vision(mock_client):
    body = _hybrid_body(
        [{}, _ollama_entry("native", ["vision"])],
        [{"id": "native"}, {"id": "chat"}, {"id": "extra", "input_modalities": ["image", "text"]}],
    )
    mock_client.models.with_raw_response.list.return_value = _raw_response(body)
    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    assert [model.id for model in await provider.get_available_models()] == ["native", "extra"]
    assert [model.id for model in await provider.get_available_models(capability="text")] == ["native", "chat", "extra"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hybrid", [False, True])
async def test_text_discovery_requires_compatible_advertised_modalities(mock_client, hybrid):
    entries = [
        {"id": "opaque"},
        {"id": "formatter", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
        {"id": "audio", "meta": {"input_modalities": ["audio"]}},
        {"id": "embedding", "input_modalities": ["text"], "output_modalities": ["embedding"]},
        {"id": "image-input", "input_modalities": ["image"], "capabilities": ["completion"]},
        {"id": "mixed", "input_modalities": ["text", "image"], "output_modalities": ["audio", "text"]},
    ]
    native = [_ollama_entry("audio", ["completion"]), _ollama_entry("embedding", ["chat"])] if hybrid else []
    mock_client.models.with_raw_response.list.return_value = _raw_response(_hybrid_body(native, entries))
    provider = OpenAIProvider(host="http://localhost:8642/v1", api_key="none")
    assert {m.id for m in await provider.get_available_models(capability="text")} == {"opaque", "formatter", "mixed"}
    assert {m.id for m in await provider.get_available_models(capability="vision")} == {"mixed", "image-input"}
