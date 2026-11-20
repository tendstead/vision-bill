import base64
import json
import logging
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Literal, cast

import httpx2

from ...helper.logging_config import setup_logging

setup_logging()

from ...model.receipt import Receipt
from ...provider.llm.base import (
    TEXT_CAPABILITIES,
    AnalysisResult,
    LLMProvider,
    ModelInfo,
    inference_endpoint_identity,
)
from ...service.activation_admission import direct_inference_call, inference_admission
from . import OPENAI_CREDENTIALLESS_API_KEY
from .experiment import chat_telemetry
from .llamacpp import LlamaCppLifecycle

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, DefaultAsyncHttpxClient
except ImportError:
    raise ImportError(
        "The 'openai' package is required for OpenAIProvider. Please install it with 'uv add openai'."
    )


logger = logging.getLogger(__name__)

RETRY_LIMIT = 3
VISION_CAPABILITY_FIELDS = ("multimodal", "vision")
DEFAULT_IMAGE_MIME = "image/jpeg"


def _format_parameter_size(n_params: int) -> str:
    """Format a raw parameter count like 27320697856 as '27.3B'."""
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n_params >= threshold:
            value = n_params / threshold
            return f"{value:.1f}{suffix}".replace(".0", "")
    return str(n_params)


def _model_capabilities(*entries: dict[str, Any], declared: bool) -> list[str]:
    """Read image input support without inferring it from outputs or names."""
    capabilities: dict[str, None] = {}
    input_modalities = []
    for entry in entries:
        advertised = entry.get("capabilities")
        if isinstance(advertised, list):
            capabilities.update({capability: None for capability in advertised if isinstance(capability, str)})
        for descriptor in (entry, entry.get("architecture"), entry.get("meta")):
            if isinstance(descriptor, dict) and isinstance(descriptor.get("input_modalities"), list):
                input_modalities.append(descriptor["input_modalities"])

    has_vision_marker = any(capability in capabilities for capability in VISION_CAPABILITY_FIELDS)
    if input_modalities:
        # Explicit input constraints win over general capability labels.
        vision = all("image" in modalities for modalities in input_modalities)
    else:
        vision = has_vision_marker or (declared and not capabilities)
    if not vision:
        return [capability for capability in capabilities if capability not in VISION_CAPABILITY_FIELDS]
    if not has_vision_marker:
        capabilities["vision"] = None
    return list(capabilities)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        host: str,
        api_key: str | None,
        temperature: float = 0.0,
        analysis_timeout_seconds: float = 600,
        max_output_tokens: int = 8192,
        vision_model_ids: Sequence[str] = (),
        model_lifecycle: Literal["none", "llamacpp"] = "none",
        *,
        explicit_authentication: bool = False,
    ):
        if model_lifecycle not in ("none", "llamacpp"):
            raise ValueError("Unsupported OpenAI-compatible model lifecycle")
        # The SDK requires a key internally even when the wire uses no Authorization.
        sdk_api_key = api_key or OPENAI_CREDENTIALLESS_API_KEY
        effective_api_key = sdk_api_key if api_key is not None else None
        self._lifecycle = LlamaCppLifecycle(host, effective_api_key) if model_lifecycle == "llamacpp" else None
        self.supports_model_release = self._lifecycle is not None
        self.lifecycle_identity = self._lifecycle.identity if self._lifecycle else "unsupported"
        self.inference_identity = inference_endpoint_identity("openai", host)
        self._vision_model_ids = frozenset(vision_model_ids)
        self.analysis_timeout_seconds = analysis_timeout_seconds
        self.max_output_tokens = max_output_tokens
        enforce_authentication = explicit_authentication or api_key is None
        url = httpx2.URL(host)
        origin = (url.scheme, url.raw_host, url.port)

        async def enforce_authorization(request: httpx2.Request) -> None:
            url = request.url
            if effective_api_key is None or (url.scheme, url.raw_host, url.port) != origin:
                request.headers.pop("Authorization", None)
            else:
                request.headers["Authorization"] = f"Bearer {effective_api_key}"

        self._client = AsyncOpenAI(
            base_url=host,
            api_key=sdk_api_key,
            timeout=analysis_timeout_seconds,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(
                trust_env=False, follow_redirects=model_lifecycle == "none",
                event_hooks={"request": [enforce_authorization]} if enforce_authentication else None,
            ),
        )
        if enforce_authentication:
            # The SDK merges OPENAI_CUSTOM_HEADERS even with an explicit constructor mapping.
            self._client._custom_headers = {}
            self._client.organization = None
            self._client.project = None
        self._temperature = temperature
        from ...service.gpu_handoff import configure_handoff
        configure_handoff(self)

    async def release_model(self, model_id: str) -> None:
        if self._lifecycle is None:
            raise ValueError("model_release_unsupported")
        await self._lifecycle.release_model(model_id)

    async def resident_model_ids(self, *, deadline: float) -> frozenset[str]:
        if self._lifecycle is None:
            raise ValueError("model_residency_unsupported")
        return await self._lifecycle.resident_model_ids(deadline=deadline)

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
            logger.warning("OpenAI-compatible connection check failed: %s", type(e).__name__)
            return False

    async def get_available_models(self, *, capability: str = "vision") -> list[ModelInfo]:
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
            data_entry = data_by_id.get(model_id, {})
            capabilities = _model_capabilities(
                entry, data_entry, declared=model_id in self._vision_model_ids,
            )
            if capability == "vision" and not any(c in VISION_CAPABILITY_FIELDS for c in capabilities):
                continue

            if (
                capability == "text"
                and capabilities
                and not any(c in capabilities for c in TEXT_CAPABILITIES)
            ):
                continue

            details = entry.get("details")
            parameter_size = ""
            if isinstance(details, dict):
                parameter_size = str(details.get("parameter_size") or "")

            meta = data_entry.get("meta")
            if not parameter_size and isinstance(meta, dict):
                n_params = meta.get("n_params")
                if isinstance(n_params, int) and n_params > 0:
                    parameter_size = _format_parameter_size(n_params)

            mi = ModelInfo(id=model_id)
            capacity = data_entry.get("context_length")
            mi.context_window_tokens = capacity if type(capacity) is int and capacity > 0 else None
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
                capabilities = _model_capabilities(
                    entry, declared=model_id in self._vision_model_ids,
                )
                if (
                    capability == "text"
                    and capabilities
                    and not any(c in capabilities for c in TEXT_CAPABILITIES)
                ):
                    continue
                if capability == "vision" and not any(
                    c in VISION_CAPABILITY_FIELDS for c in capabilities
                ):
                    continue
                mi = ModelInfo(id=model_id)
                capacity = entry.get("context_length")
                mi.context_window_tokens = (
                    capacity if type(capacity) is int and capacity > 0 else None
                )
                mi.capabilities = capabilities
                digest = str(entry.get("digest") or "")
                mi.digest = digest or None
                mi.parameter_size = None
                result.append(mi)

        return result

    async def send_message(self, model_id: str, messages: Sequence[Mapping[str, Any]]) -> str:
        from ...service.gpu_handoff import current_owner

        with inference_admission(self):
            if self.handoff is not None and not self.supports_model_release:
                raise ValueError("model_release_unsupported: managed OCR requires an unload-capable provider")
            owner = current_owner()
            if owner is not None:
                if owner.coordinator.shared_gpu and owner.resources.llm != self.lifecycle_identity:
                    raise ValueError("model_handoff_identity_changed")
                await owner.llm_phase(model_id)
            elif self.handoff is not None:
                async with self.handoff.session(monotonic() + self.analysis_timeout_seconds, model_id=model_id) as owner:
                    await owner.llm_phase(model_id)
                    return await self._send_message(model_id, messages)
            return await self._send_message(model_id, messages)

    async def _send_message(self, model_id: str, messages: Sequence[Mapping[str, Any]]) -> str:
        from ...service.gpu_handoff import current_llm_owner

        # Plain dicts are exactly what the SDK serializes (verified against
        # the live server); cast() satisfies the TypedDict-parameter overloads.
        started = perf_counter()
        owner = current_llm_owner()
        with direct_inference_call(coordinated=owner is not None):
            call = self._client.chat.completions.create(
                model=model_id,
                messages=cast(Any, messages),
                temperature=self._temperature,
                max_tokens=self.max_output_tokens,
            )
            response = await owner.model_call(call) if owner is not None else await call
        if self.experiment_trace is not None:
            self.experiment_trace.append(
                chat_telemetry(response, messages, (perf_counter() - started) * 1000)
            )
        if not response.choices:
            return ""
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
        """Explicit legacy/replay adapter for historical untyped OCR transcripts."""
        from ..ocr.base import compact_ocr_text

        messages = [
            {"role": "user", "content": self.build_ocr_prompt(compact_ocr_text(ocr_result), tags)}
        ]
        return (await self._legacy_analysis(model_id, messages)).receipt

    async def analyse_receipt_with_metadata(
        self, model_id: str, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> AnalysisResult:
        return await self._legacy_analysis(
            model_id,
            self._build_image_messages(image, tags),
            image_count=1 if isinstance(image, Path) else len(image),
        )
