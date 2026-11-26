from unittest.mock import MagicMock

import pytest

from vision_bill.config import LLMProviderEnum
from vision_bill.provider.factory import get_llm_provider
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider


@pytest.mark.parametrize(
    "provider_enum, expected_class",
    [
        (LLMProviderEnum.OLLAMA, OllamaProvider),
        (LLMProviderEnum.OPENAI, OpenAIProvider),
    ],
)
async def test_get_llm_provider_success(settings, provider_enum, expected_class):
    """Test that the correct provider is returned for each implemented provider."""
    settings.llm.provider = provider_enum
    provider = get_llm_provider(settings.llm)
    assert isinstance(provider, expected_class)


def test_get_llm_provider_failure():
    """Test that a ValueError is raised for unsupported providers."""
    # Mock settings object specifically to trigger the failure path in factory.py:8-15
    mock_settings = MagicMock()
    mock_settings.provider = "non_existent_provider"
    mock_settings.host = "http://localhost:11434"

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_provider(mock_settings)


def test_unsupported_existing_provider_has_actionable_diagnostic():
    from pydantic import ValidationError

    from vision_bill.config import LLMSettings

    with pytest.raises(ValidationError, match="(?i)unsupported.*ollama.*openai"):
        LLMSettings(
            provider="anthropic",
            host="http://external",
            api_key="private-key",
            model_name="selected",
            temperature=0,
        )


@pytest.mark.parametrize("field", ["host", "model_name"])
@pytest.mark.parametrize("value", ["", " "])
def test_provider_configuration_requires_explicit_nonblank_identity(field, value):
    from pydantic import ValidationError

    from vision_bill.config import LLMSettings

    values = {"host": "http://external", "api_key": "", "model_name": "chosen", "temperature": 0}
    values[field] = value
    with pytest.raises(ValidationError, match=field):
        LLMSettings(**values)
