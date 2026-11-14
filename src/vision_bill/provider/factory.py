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
        )

    if provider == LLMProviderEnum.OPENAI:
        from .llm.openai import OpenAIProvider

        return OpenAIProvider(
            host=llm_settings.host,
            api_key=llm_settings.api_key,
            temperature=llm_settings.temperature,
            analysis_timeout_seconds=llm_settings.analysis_timeout_seconds,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_ocr_provider(ocr_settings: OcrSettings) -> OcrProvider:
    """Create an OCR provider based on settings."""
    if ocr_settings.provider == "paddleocr":
        from .ocr.paddleocr import PaddleOcrProvider

        return PaddleOcrProvider(
            service_url=ocr_settings.service_url,
            timeout=ocr_settings.timeout_seconds,
        )

    raise ValueError(f"Unsupported OCR provider: {ocr_settings.provider}")
