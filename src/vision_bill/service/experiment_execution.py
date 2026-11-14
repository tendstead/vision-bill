"""Benchmark execution through the same public receipt-service paths as uploads.

The optional OCR extension is supplied by the existing OCR branch. This module
neither creates another OCR service nor changes the application's provider.
"""
import asyncio
import hashlib
import json
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from ..model.experiment import ExperimentOptions, ExtractionPipeline
from ..model.receipt import Receipt
from ..provider.ocr.base import compact_ocr_text

if TYPE_CHECKING:
    from .receipt_service import ReceiptService


@dataclass
class ExperimentResult:
    receipt: Receipt
    attempts: int
    elapsed_ms: float
    metadata: dict[str, Any]


class ExperimentFailure(ValueError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


class StrictOcrFailure(RuntimeError):
    """Not swallowed by the OCR extension's normal fallback handler."""


def ocr_fingerprint(provider: Any) -> str | None:
    """Identify the published OCR extension's client without exposing its URL.

    Remote model weights/server settings cannot be verified through its current
    API; changing those in place still requires a new experiment run.
    """
    if provider is None:
        return None
    identity = {"provider": type(provider).__name__,
                "endpoint": getattr(provider, "service_url", None),
                "timeout": getattr(provider, "timeout", None)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class ObservedOcr:
    def __init__(self, original: Any, metadata: dict[str, Any], allow_fallback: bool):
        self.original = original
        self.metadata = metadata
        self.allow_fallback = allow_fallback

    async def extract_text(self, image: Path) -> dict[str, Any]:
        started = perf_counter()
        try:
            result = await self.original.extract_text(image)
            text = compact_ocr_text(result)
            self.metadata["ocr_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            self.metadata["ocr_characters"] = len(text)
            return {"text": text}
        except Exception as exc:
            self.metadata["ocr_error"] = type(exc).__name__
            self.metadata["fallback_used"] = self.allow_fallback
            if not self.allow_fallback:
                raise StrictOcrFailure(str(exc)) from exc
            # Normalize an extension failure to its existing fallback contract.
            raise ValueError(str(exc)) from exc
        finally:
            self.metadata["ocr_ms"] = (perf_counter() - started) * 1000


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


async def execute_submission(service: "ReceiptService", model_id: str, paths: list[Path],
                             options: ExperimentOptions, tags: list[str]) -> ExperimentResult:
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("Receipt image or additional photo is unavailable")
    started = perf_counter()
    probe = copy(service)
    provider = service._provider.for_experiment(options)
    probe._provider = provider
    probe._experiment_tags = list(tags)
    metadata: dict[str, Any] = {"requested_pipeline": options.pipeline.value,
                               "fallback_used": False, "image_count": len(paths),
                               "provider": type(provider).__name__,
                               "options": provider.experiment_options.model_dump(mode="json")
                               if provider.experiment_options else options.model_dump(mode="json")}
    original_ocr = getattr(service, "_ocr_provider", None)
    use_ocr = options.pipeline != ExtractionPipeline.DIRECT and original_ocr is not None and len(paths) == 1
    if options.pipeline == ExtractionPipeline.OCR and not use_ocr:
        raise ValueError("OCR mode requires the OCR extension and one source image")
    try:
        async with asyncio.timeout(provider.analysis_timeout_seconds):
            if use_ocr:
                extension = cast(Any, probe)
                extension._ocr_provider = ObservedOcr(original_ocr, metadata, options.allow_ocr_fallback)
                receipt = await extension.analyse_receipt_with_ocr(model_id, paths[0])
            else:
                receipt = await probe.analyse_receipt_from_path(model_id, paths[0] if len(paths) == 1 else paths)
    except Exception as exc:
        metadata["calls"] = provider.experiment_trace or []
        metadata["elapsed_ms"] = (perf_counter() - started) * 1000
        raise ExperimentFailure(f"{type(exc).__name__}: {exc}", metadata) from exc
    metadata["used_pipeline"] = "ocr" if use_ocr and not metadata["fallback_used"] else "direct_vision"
    metadata["calls"] = provider.experiment_trace or []
    metadata["llm_ms"] = sum(call["elapsed_ms"] for call in metadata["calls"])
    elapsed = (perf_counter() - started) * 1000
    metadata["elapsed_ms"] = elapsed
    return ExperimentResult(receipt, len(metadata["calls"]) or 1, elapsed, metadata)
