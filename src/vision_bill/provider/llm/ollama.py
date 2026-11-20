import asyncio
import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import httpx

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
from .experiment import chat_telemetry

try:
    from ollama import AsyncClient, ListResponse, ResponseError
except ImportError:
    raise ImportError(
        "The 'ollama' package is required for OllamaProvider. Please install it with 'uv add ollama'."
    )


logger = logging.getLogger(__name__)


VISION_CAPABILITY_FIELD = "vision"
STRUCTURED_CAPABILITY_FIELD = "structured"
RETRY_LIMIT = 3


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        host: str,
        temperature: float = 0.0,
        analysis_timeout_seconds: float = 600,
        max_output_tokens: int = 8192,
    ):
        self.analysis_timeout_seconds = analysis_timeout_seconds
        self.max_output_tokens = max_output_tokens
        self._client = AsyncClient(
            host=host, timeout=httpx.Timeout(analysis_timeout_seconds, connect=5), trust_env=False
        )
        self._temperature = temperature
        self.supports_thinking_control = True
        self.supports_model_release = True
        self.lifecycle_identity = "ollama:" + hashlib.sha256(host.rstrip("/").encode()).hexdigest()
        # The SDK resolves a bare hostname to port 11434, while an explicit
        # http:// base without a port reaches the standard HTTP port.
        self.inference_identity = inference_endpoint_identity("ollama", str(self._client._client.base_url))
        from ...service.gpu_handoff import configure_handoff
        configure_handoff(self)

    async def release_model(self, model_id: str) -> None:
        """Unload precisely the requested ID and verify absence, preserving others."""
        from ...service.gpu_handoff import current_owner
        owner = current_owner()
        if owner is not None:
            if owner.resources.llm != self.lifecycle_identity:
                raise ValueError("model_handoff_identity_changed")
            owner.require_release(model_id)
        call = self._release_model(model_id, owner.deadline if owner else monotonic() + 30)
        if owner is not None:
            await owner.model_call(call)
        else:
            await call

    async def _release_model(self, model_id: str, deadline: float) -> None:
        await self._client.generate(model=model_id, keep_alive=0)
        running = await self.resident_model_ids(deadline=deadline)
        if not running.isdisjoint(self.residency_ids_for(model_id)):
            raise RuntimeError("model_release_unconfirmed")

    def residency_ids_for(self, model_id: str) -> tuple[str, ...]:
        """Ollama reports untagged requests with the implicit latest tag."""
        if ":" in model_id.rsplit("/", 1)[-1]:
            return (model_id,)
        return (model_id, model_id + ":latest")

    async def resident_model_ids(self, *, deadline: float) -> frozenset[str]:
        async with asyncio.timeout(max(0, deadline - monotonic())):
            running = await self._client.ps()
        if "models" not in running.model_fields_set:
            raise RuntimeError("model_residency_unknown")
        models: list[str] = []
        for item in running.models:
            model = item.model
            if not isinstance(model, str) or not model.strip() or model != model.strip() or model in models:
                raise RuntimeError("model_residency_unknown")
            models.append(model)
        return frozenset(models)

    def update_runtime_settings(self, *, temperature: float) -> None:
        self._temperature = temperature

    async def check_connection(self) -> bool:
        try:
            await self._client.list()
            return True
        except (ConnectionError, ResponseError, httpx.HTTPError) as e:
            # The SDK raises ConnectionError when Ollama is unreachable,
            # ResponseError on HTTP error responses, and raw httpx errors
            # (e.g. timeouts) otherwise.
            logger.warning("Ollama connection check failed: %s", type(e).__name__)
            return False

    async def get_available_models(self, *, capability: str = "vision") -> list[ModelInfo]:
        models: ListResponse = await self._client.list()
        result = []
        for model in models.models:
            # model.model is the name of the model, e.g., "gemma4:e4b", we need this for all further calls.
            if not model.model:
                continue

            details = await self._client.show(model.model)
            # If the model does not support vision, we skip it.
            if capability == "vision" and (
                not details.capabilities or VISION_CAPABILITY_FIELD not in details.capabilities
            ):
                continue

            if capability == "text" and not any(
                c in (details.capabilities or []) for c in TEXT_CAPABILITIES
            ):
                continue

            mi = ModelInfo(id=model.model)
            mi.digest = getattr(model, "digest", None)
            if model.details:
                mi.parameter_size = model.details.parameter_size

            mi.capabilities = details.capabilities
            info = details.modelinfo if isinstance(details.modelinfo, dict) else {}
            capacities = [
                value
                for key, value in info.items()
                if key.endswith("context_length") and type(value) is int and value > 0
            ]
            mi.context_window_tokens = min(capacities) if capacities else None

            result.append(mi)

        return result

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

    async def send_message(self, model_id: str, messages: Sequence[Mapping[str, Any]]) -> str:
        from ...service.gpu_handoff import current_owner
        with inference_admission(self):
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
        started = perf_counter()
        extra: dict[str, Any] = {}
        if self.experiment_options and self.experiment_options.think is not None:
            extra["think"] = self.experiment_options.think
        owner = current_llm_owner()
        with direct_inference_call(coordinated=owner is not None):
            call = self._client.chat(
                model=model_id,
                # format=Receipt.model_json_schema(),
                messages=messages,
                options={"temperature": self._temperature, "num_predict": self.max_output_tokens},
                **extra,
            )
            response = await owner.model_call(call) if owner is not None else await call
        if self.experiment_trace is not None:
            self.experiment_trace.append(
                chat_telemetry(response, messages, (perf_counter() - started) * 1000)
            )
        return response.message.content or ""

    def _build_image_messages(
        self, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        images = [image] if isinstance(image, Path) else list(image)
        for path in images:
            if not path.exists():
                raise FileNotFoundError(f"Image not found at: {path}")
        return [
            {
                "role": "user",
                "content": self.build_prompt(tags, image_count=len(images)),
                "images": images,
            }
        ]
