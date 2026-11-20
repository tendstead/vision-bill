# llm/factory.py
from ..config import LLMProviderEnum, LLMSettings, OcrSettings
from .llm.base import LLMProvider
from .ocr.base import OcrProvider


def get_llm_provider(llm_settings: LLMSettings) -> LLMProvider:
    provider = llm_settings.provider

    if provider == LLMProviderEnum.OLLAMA:
        from .llm.ollama import OllamaProvider

        return OllamaProvider(
            host=llm_settings.host,
            temperature=llm_settings.temperature,
            analysis_timeout_seconds=llm_settings.analysis_timeout_seconds,
            max_output_tokens=llm_settings.max_output_tokens,
        )

    if provider == LLMProviderEnum.OPENAI:
        from .llm.openai import OpenAIProvider

        return OpenAIProvider(
            host=llm_settings.host,
            api_key=llm_settings.request_api_key(),
            explicit_authentication=llm_settings.authentication is not None,
            vision_model_ids=llm_settings.vision_model_ids,
            model_lifecycle=llm_settings.model_lifecycle,
            temperature=llm_settings.temperature,
            analysis_timeout_seconds=llm_settings.analysis_timeout_seconds,
            max_output_tokens=llm_settings.max_output_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_ocr_provider(ocr_settings: OcrSettings) -> OcrProvider | None:
    """Create an OCR provider based on settings."""
    if not ocr_settings.enabled:
        return None
    if ocr_settings.provider == "paddleocr":
        from .ocr.paddleocr import PaddleOcrProvider

        return PaddleOcrProvider(
            service_url=ocr_settings.service_url,
            timeout=ocr_settings.timeout_seconds,
            managed_lifecycle=ocr_settings.manage_model_lifecycle,
        )

    raise ValueError(f"Unsupported OCR provider: {ocr_settings.provider}")
