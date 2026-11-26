from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vision_bill.provider.llm.ollama import OllamaProvider


@pytest.fixture
def mock_client():
    """Fixture to mock the ollama.AsyncClient."""
    with patch("vision_bill.provider.llm.ollama.AsyncClient") as mocked_client:
        # Create an AsyncMock instance that will be returned by the constructor
        client = AsyncMock()
        client._client.base_url = 'http://localhost:11434'
        mocked_client.return_value = client
        yield client


@pytest.fixture
def settings(tmp_path: Path):
    """Fixture to provide a valid Settings object for tests."""
    from vision_bill.config import Settings

    return Settings(
        api={
            "port": 8080,
            "log_level": "INFO",
            "save_dir": str(tmp_path / "uploads"),
            "tmp_dir": str(tmp_path / "uploads_tmp"),
        }
    )


@pytest.mark.asyncio
async def test_get_available_models(mock_client):
    # Arrange
    mock_list_response = MagicMock()
    mock_list_response.models = [
        MagicMock(model="gemma4:vision", details={"parameter_size": "7B"}),
        MagicMock(model="llama3"),
        MagicMock(model="bad_model", details={}),
    ]
    # The list() method is awaited in production code, so it must be an AsyncMock or a mock returning an awaitable
    mock_client.list.return_value = MagicMock(
        models=[
            MagicMock(model="gemma4:vision", details=MagicMock(parameter_size="7B")),
            MagicMock(model="llama3"),
            MagicMock(model="bad_model", details=None),
        ]
    )

    mock_show_success = MagicMock()
    mock_show_success.capabilities = {"vision": True, "structured": True}

    # Side effect to return different show results based on model name
    async def side_effect(model_id):
        if model_id == "gemma4:vision":
            return mock_show_success
        elif model_id == "llama3":
            mock_res = MagicMock()
            mock_res.capabilities = {"text": True}
            return mock_res
        else:
            return MagicMock(capabilities={})

    # The show() method is also awaited in production code
    mock_client.show.side_effect = side_effect

    # Act
    provider = OllamaProvider(host="http://localhost:11434")
    models = await provider.get_available_models()

    # Assert
    assert len(models) == 1
    assert models[0].id == "gemma4:vision"
    assert models[0].capabilities["vision"] is True


@pytest.mark.asyncio
async def test_analyse_receipt_from_model_success(mock_client):
    # Arrange
    provider = OllamaProvider(host="http://localhost:11434")
    content = '{"confidence": 95, "merchant_name": "Test Shop", "merchant_address": "123 Main St", "receipt_number": "REC001", "date": "2024-08-06", "time": "14:00", "currency": "USD", "line_items": [{"description": "Coffee", "quantity": 1, "unit_price": 5.00, "total_price": 5.00}], "taxes": [{"name": "VAT", "rate": 0.20, "amount": 1.00}], "subtotal": 5.00, "discount_total": 0.00, "tax_total": 1.00, "tip": 0.50, "total": 6.50, "payment_method": "credit_card"}'

    # The chat() method is awaited in production code
    mock_client.chat.return_value = MagicMock(message=MagicMock(content=content))

    # Mock a file that exists
    with patch("vision_bill.provider.llm.ollama.Path.exists", return_value=True):
        result = await provider.analyse_receipt_from_model("gemma4:vision", Path("fake.png"))

    # Assert
    assert result.merchant_name == "Test Shop"
    assert float(result.total) == 6.50


@pytest.mark.asyncio
async def test_analyse_receipt_from_model_retry_empty(mock_client):
    # Arrange
    provider = OllamaProvider(host="http://localhost:11434")

    # Return empty content first, then success
    mock_client.chat.side_effect = [
        MagicMock(message=MagicMock(content="")),
        MagicMock(
            message=MagicMock(
                content='{"confidence": 95, "merchant_name": "Retry Shop", "merchant_address": "123 Retry St", "receipt_number": "REC002", "date": "2024-08-06", "time": "15:00", "currency": "USD", "line_items": [{"description": "Retry Item", "quantity": 1, "unit_price": 1.0, "total_price": 1.0}], "taxes": [], "subtotal": 1.0, "discount_total": 0.0, "tax_total": 0.0, "tip": 0.0, "total": 1.0, "payment_method": "credit_card"}'
            )
        ),
    ]

    with patch("vision_bill.provider.llm.ollama.Path.exists", return_value=True):
        result = await provider.analyse_receipt_from_model("gemma4:vision", Path("fake.png"))

    # Assert
    assert result.merchant_name == "Retry Shop"
    # Verify it called chat twice
    assert mock_client.chat.call_count == 2


@pytest.mark.asyncio
async def test_analyse_receipt_from_model_failure_no_vision(mock_client):
    # Arrange
    provider = OllamaProvider(host="http://localhost:11434")
    mock_client.chat.return_value = MagicMock(message=MagicMock(content="I cannot process images."))

    with (
        patch("vision_bill.provider.llm.ollama.Path.exists", return_value=True),
        pytest.raises(ValueError, match=r".*Failed to get a valid response.*"),
    ):
        await provider.analyse_receipt_from_model("gemma4:vision", Path("fake.png"))


@pytest.mark.asyncio
async def test_analyse_receipt_from_model_repair(mock_client):
    # Arrange
    provider = OllamaProvider(host="http://localhost:11434")

    bad_content = '{"merchant": "Bad JSON"'  # Missing closing brace
    good_content = '{"confidence": 95, "merchant_name": "Fixed Shop", "merchant_address": "123 Fix St", "receipt_number": "REC002", "date": "2024-08-06", "time": "15:00", "currency": "USD", "line_items": [{"description": "Repair Item", "quantity": 1, "unit_price": 5.00, "total_price": 5.00}], "taxes": [], "subtotal": 5.00, "discount_total": 0.00, "tax_total": 0.00, "tip": 0.00, "total": 5.00, "payment_method": "credit_card"}'

    mock_client.chat.side_effect = [
        MagicMock(message=MagicMock(content=bad_content)),
        MagicMock(message=MagicMock(content=good_content)),
    ]

    with patch("vision_bill.provider.llm.ollama.Path.exists", return_value=True):
        result = await provider.analyse_receipt_from_model("gemma4:vision", Path("fake.png"))

    # Assert
    assert result.merchant_name == "Fixed Shop"
    assert float(result.total) == 5.0

    # Verify that the second call contained a repair message in the context
    last_call = mock_client.chat.call_args[1]
    messages = last_call["messages"]
    # The last message should be the user's repair instruction
    assert any("Please respond again with ONLY corrected JSON" in m["content"] for m in messages)


@pytest.mark.asyncio
async def test_check_connection_success(mock_client):
    mock_client.list.return_value = MagicMock(models=[])

    provider = OllamaProvider(host="http://localhost:11434")
    assert await provider.check_connection() is True
    mock_client.list.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_connection_failure(mock_client):
    mock_client.list.side_effect = ConnectionError("connection refused")

    provider = OllamaProvider(host="http://localhost:11434")
    assert await provider.check_connection() is False
    mock_client.list.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_connection_unexpected_error_propagates(mock_client):
    # Only known failure modes (ConnectionError, ResponseError, httpx errors)
    # map to "unreachable"; anything else is a bug and must surface.
    mock_client.list.side_effect = RuntimeError("boom")

    provider = OllamaProvider(host="http://localhost:11434")
    with pytest.raises(RuntimeError, match="boom"):
        await provider.check_connection()
    mock_client.list.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyse_receipt_embeds_tag_vocabulary_in_prompt(mock_client):
    """When a tag vocabulary is supplied it must appear in the prompt content."""
    provider = OllamaProvider(host="http://localhost:11434")
    content = '{"confidence": 95, "merchant_name": "Shop", "date": "2024-08-06", "line_items": [], "subtotal": 0.00, "total": 0.00}'
    mock_client.chat.return_value = MagicMock(message=MagicMock(content=content))

    with patch("vision_bill.provider.llm.ollama.Path.exists", return_value=True):
        await provider.analyse_receipt_from_model(
            "gemma4:vision", Path("fake.png"), tags=["coffee", "food"]
        )

    messages = mock_client.chat.call_args[1]["messages"]
    user_prompt = messages[0]["content"]
    assert "Prefer tags from this list: coffee, food" in user_prompt
    # The merchant-name guidance (no guessing from logos) ships with the prompt.
    assert "Do not guess or infer the company name from a logo" in user_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selection", "expected"),
    [(None, "native-deployment"), ("none", None), ("bearer", "native-selected"), ("foreign", None)],
)
async def test_native_client_authentication_and_redirect_policy(monkeypatch, settings, selection, expected):
    from pydantic import SecretStr

    from vision_bill.config import LlmAuthentication, LLMProviderEnum
    from vision_bill.provider.factory import get_llm_provider

    monkeypatch.setenv("OLLAMA_API_KEY", "native-deployment")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Private: openai-only")
    llm = settings.llm
    llm.host = "https://proxy.example/native"
    llm.api_key = "openai-only"
    if selection == "none":
        llm.authentication = LlmAuthentication(provider=LLMProviderEnum.OLLAMA, mode="none")
    elif selection:
        llm.authentication = LlmAuthentication(
            provider=LLMProviderEnum.OPENAI if selection == "foreign" else LLMProviderEnum.OLLAMA,
            mode="bearer", api_key=SecretStr("native-selected"), origin="https://proxy.example:443",
        )
    provider = get_llm_provider(llm)
    client = provider._client._client
    try:
        assert client.headers.get("authorization") == (f"Bearer {expected}" if expected else None)
        assert "x-private" not in client.headers
        assert client.follow_redirects is False
    finally:
        await client.aclose()
