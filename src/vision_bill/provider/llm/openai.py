import base64
import json
import logging
import mimetypes
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

from ...helper.logging_config import setup_logging

setup_logging()

from ...model.receipt import Receipt
from ...provider.llm.base import AnalysisResult, LLMProvider, ModelInfo
from ..ocr.base import compact_ocr_text
from .experiment import chat_telemetry

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for OpenAIProvider. Please install it with 'uv add openai'."
    )


logger = logging.getLogger(__name__)

RETRY_LIMIT = 3
# Typed as a literal so it matches the SDK's reasoning_effort parameter type.
REASONING_EFFORT: Literal["low"] = "low"
VISION_CAPABILITY_FIELDS = ("multimodal", "vision")
DEFAULT_IMAGE_MIME = "image/jpeg"


def _format_parameter_size(n_params: int) -> str:
    """Format a raw parameter count like 27320697856 as '27.3B'."""
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n_params >= threshold:
            value = n_params / threshold
            return f"{value:.1f}{suffix}".replace(".0", "")
    return str(n_params)


def _is_vision_capable(entry: dict[str, Any], capabilities: list[str] | None) -> bool:
    """True when a model entry advertises image/vision input capability."""
    if capabilities and any(field in capabilities for field in VISION_CAPABILITY_FIELDS):
        return True
    architecture = entry.get("architecture")
    if isinstance(architecture, dict):
        input_modalities = architecture.get("input_modalities")
        if isinstance(input_modalities, list) and "image" in input_modalities:
            return True
    meta = entry.get("meta")
    if isinstance(meta, dict):
        for value in meta.values():
            if isinstance(value, list) and "image" in value:
                return True
    return False


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        host: str,
        api_key: str,
        temperature: float = 0.0,
        analysis_timeout_seconds: float = 600,
    ):
        self.analysis_timeout_seconds = analysis_timeout_seconds
        self._client = AsyncOpenAI(
            base_url=host, api_key=api_key, timeout=analysis_timeout_seconds, max_retries=0
        )
        self._temperature = temperature

    def update_runtime_settings(self, *, temperature: float) -> None:
        self._temperature = temperature

    async def check_connection(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except (APIConnectionError, APIStatusError) as e:
            # The openai SDK wraps every transport failure into
            # APIConnectionError (incl. APITimeoutError for timeouts) once
            # retries are exhausted, and HTTP error responses into
            # APIStatusError — so these two cover all "unreachable" modes.
            logger.warning("OpenAI-compatible connection check failed: %s", e)
            return False

    async def get_available_models(self) -> list[ModelInfo]:
        response = await self._client.models.with_raw_response.list()
        # llama.cpp serves a hybrid body: an Ollama-style "models" array
        # (with capabilities/details) alongside the OpenAI-style "data"
        # array. Parse the raw JSON so both are available; fall back to
        # "data" entries alone for other OpenAI-compatible servers.
        body = json.loads(response.http_response.content)
        data_entries: list[dict[str, Any]] = list(body.get("data") or [])
        ollama_entries: list[dict[str, Any]] = list(body.get("models") or [])
        data_by_id: dict[str, dict[str, Any]] = {
            str(d.get("id")): d for d in data_entries if isinstance(d, dict)
        }

        result: list[ModelInfo] = []
        for entry in ollama_entries:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("model") or entry.get("name") or "")
            if not model_id:
                continue
            capabilities = entry.get("capabilities")
            if not isinstance(capabilities, list):
                capabilities = []
            if not _is_vision_capable(data_by_id.get(model_id, {}), capabilities):
                continue

            details = entry.get("details")
            parameter_size = ""
            if isinstance(details, dict):
                parameter_size = str(details.get("parameter_size") or "")

            meta = (data_by_id.get(model_id) or {}).get("meta")
            if not parameter_size and isinstance(meta, dict):
                n_params = meta.get("n_params")
                if isinstance(n_params, int) and n_params > 0:
                    parameter_size = _format_parameter_size(n_params)

            mi = ModelInfo(id=model_id)
            mi.capabilities = capabilities
            digest = str(entry.get("digest") or "")
            mi.digest = digest or None
            mi.parameter_size = parameter_size or None
            result.append(mi)

        # Servers without the ollama-style array: use data entries directly.
        if not ollama_entries:
            for entry in data_entries:
                if not isinstance(entry, dict):
                    continue
                model_id = str(entry.get("id") or "")
                if not model_id:
                    continue
                if not _is_vision_capable(entry, None):
                    continue
                mi = ModelInfo(id=model_id)
                capabilities = entry.get("capabilities")
                mi.capabilities = capabilities if isinstance(capabilities, list) else []
                digest = str(entry.get("digest") or "")
                mi.digest = digest or None
                mi.parameter_size = None
                result.append(mi)

        return result

    async def send_message(self, model_id: str, messages: Sequence[Mapping[str, Any]]) -> str:
        # Plain dicts are exactly what the SDK serializes (verified against
        # the live server); cast() satisfies the TypedDict-parameter overloads.
        started = perf_counter()
        response = await self._client.chat.completions.create(
            model=model_id,
            messages=cast(Any, messages),
            temperature=self._temperature,
            reasoning_effort=REASONING_EFFORT,
        )
        if self.experiment_trace is not None:
            self.experiment_trace.append(chat_telemetry(response, messages, (perf_counter() - started) * 1000))
        return response.choices[0].message.content or ""

    def _build_image_messages(
        self, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        images = [image] if isinstance(image, Path) else list(image)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": self.build_prompt(tags, image_count=len(images))}
        ]
        for path in images:
            if not path.exists():
                raise FileNotFoundError(f"Image not found at: {path}")
            mime = mimetypes.guess_type(path.name)[0] or DEFAULT_IMAGE_MIME
            image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
            )
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    async def analyse_receipt_from_model(
        self, model_id: str, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> Receipt:
        return (await self.analyse_receipt_with_metadata(model_id, image, tags=tags)).receipt

    async def analyse_receipt_from_ocr_text(
        self, model_id: str, ocr_result: dict[str, Any], tags: Sequence[str] | None = None
    ) -> Receipt:
        """Send OCR text to the LLM for formatting into a Receipt."""
        logger.info("Sending OCR text to model %s for formatting", model_id)

        # Convert OCR result to text format for the LLM
        ocr_text = self._ocr_result_to_text(ocr_result)
        logger.debug("OCR text length: %d characters", len(ocr_text))
        if len(ocr_text) > 200:
            logger.debug("OCR text (first 200 chars): %s...", ocr_text[:200])
        else:
            logger.debug("OCR text: %s", ocr_text)

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._build_ocr_formatting_prompt(ocr_text, tags),
            }
        ]

        last_error: Exception | None = None
        for attempt in range(1, RETRY_LIMIT + 1):
            logger.info("LLM attempt %d/%d for OCR text formatting", attempt, RETRY_LIMIT)
            content = await self.send_message(model_id, messages)

            if not content:
                logger.warning("Attempt %d/%d: model returned no content", attempt, RETRY_LIMIT)
                last_error = ValueError("Empty response from model")
                continue

            logger.debug("LLM response length: %d characters", len(content))
            try:
                return self.parse_llm_response(content)
            except ValueError as e:
                logger.warning(
                    "Attempt %d/%d: failed to parse LLM response: %s",
                    attempt,
                    RETRY_LIMIT,
                    e,
                )
                last_error = e
                messages = self._append_repair_message(messages, content, e)

        raise ValueError(
            f"Failed to get a valid response from model '{model_id}' after {RETRY_LIMIT} attempts"
        ) from last_error

    def _ocr_result_to_text(self, ocr_result: dict[str, Any]) -> str:
        """Require recognized text before invoking the formatting model."""
        return compact_ocr_text(ocr_result)

    def _build_ocr_formatting_prompt(self, ocr_text: str, tags: Sequence[str] | None = None) -> str:
        """Build a prompt for formatting OCR text into a Receipt."""
        base_prompt = self.build_prompt(tags)
        return f"""{base_prompt}

Here is the OCR-extracted text from the receipt:

---
{ocr_text}
---

Parse this text and return the structured JSON receipt."""

    async def analyse_receipt_with_metadata(
        self, model_id: str, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> AnalysisResult:
        from time import perf_counter

        started = perf_counter()
        messages = self._build_image_messages(image, tags)

        last_error: Exception | None = None
        for attempt in range(1, RETRY_LIMIT + 1):
            content = await self.send_message(model_id, messages)

            if not content:
                logger.warning("Attempt %d/%d: model returned no content", attempt, RETRY_LIMIT)
                last_error = ValueError("Empty response from model")
                continue

            logger.warning(f"Attempt {attempt}/{RETRY_LIMIT}: model returned content: {content}")

            if not content.lstrip().startswith(("{", "```")) and re.search(
                r"provide.*image", content, re.IGNORECASE | re.DOTALL
            ):
                raise ValueError(
                    f"Model '{model_id}' returned a message indicating it cannot process images. "
                    "Please ensure the model supports vision capabilities."
                )

            try:
                return AnalysisResult(
                    receipt=self.parse_llm_response(
                        content, image_count=1 if isinstance(image, Path) else len(image)
                    ),
                    attempts=attempt,
                    elapsed_ms=(perf_counter() - started) * 1000,
                )
            except ValueError as e:
                logger.warning(
                    "Attempt %d/%d: failed to parse LLM response: %s",
                    attempt,
                    RETRY_LIMIT,
                    e,
                )
                last_error = e
                # Feed the parse error back so the retry has a chance to self-correct
                messages = self._append_repair_message(messages, content, e)

        raise ValueError(
            f"Failed to get a valid response from model '{model_id}' after {RETRY_LIMIT} attempts"
        ) from last_error

    def _append_repair_message(
        self,
        messages: list[dict[str, Any]],
        bad_output: str,
        error: ValueError,
    ) -> list[dict[str, Any]]:
        """Append the failed output + error so the model can self-correct on retry."""
        return [
            *messages,
            {"role": "assistant", "content": bad_output},
            {
                "role": "user",
                "content": (
                    "That response was not valid JSON matching the required schema. "
                    f"Error: {error}\n\nPlease respond again with ONLY corrected JSON."
                ),
            },
        ]
