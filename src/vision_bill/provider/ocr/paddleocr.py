"""Local v2 OCR transport. Original bytes and identities cross this boundary unchanged."""

import asyncio
import hashlib
import json
from io import BytesIO
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import httpx
from ocr_contract import limits
from ocr_contract.document_image import decode_working_png, validate_working_png
from PIL import Image
from pydantic import ValidationError

from ...helper.deadline import check_deadline, run_sync_before
from ...model.ocr import OcrCapabilities, OcrResponse, WorkerReadiness
from ...model.ocr_errors import (
    OCR_GPU_OUT_OF_MEMORY,
    OCR_HTTP_ERRORS,
    OCR_RESULT_CODES,
    OcrGpuOutOfMemoryError,
    OcrWorkerFailure,
)
from .base import OcrProvider

MAX_ERROR_BODY_BYTES = 4096


async def _raise_native_error(response: httpx.Response) -> None:
    """Only recognize the native detail within a short, bounded identity response."""
    allowed = OCR_HTTP_ERRORS.get(response.status_code)
    if allowed is None:
        return
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        return
    body = bytearray()
    try:
        async with asyncio.timeout(1):
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > MAX_ERROR_BODY_BYTES:
                    return
                body.extend(chunk)
        # Keep object pairs to reject duplicate keys and any additional members.
        detail: object = json.loads(body, object_pairs_hook=list)
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError):
        return
    for code in allowed:
        if detail == [("detail", code)]:
            if code == OCR_GPU_OUT_OF_MEMORY:
                raise OcrGpuOutOfMemoryError
            raise OcrWorkerFailure(code)


class PaddleOcrProvider(OcrProvider):
    managed_lifecycle = True

    def __init__(self, service_url: str, timeout: int = 300, *, managed_lifecycle: bool = True):
        self.service_url = service_url.rstrip("/")
        self.lifecycle_identity = "paddleocr:" + hashlib.sha256(self.service_url.encode()).hexdigest()
        self.timeout = timeout
        self.managed_lifecycle = managed_lifecycle

    async def _lifecycle(self, action: str, token: str, deadline: float) -> None:
        async with (
            httpx.AsyncClient(timeout=max(.001, deadline - monotonic()), trust_env=False) as client,
            client.stream(
                "POST", f"{self.service_url}/lifecycle/{action}",
                headers={"Accept-Encoding": "identity"}, json={"owner_token": token},
            ) as response,
        ):
            if action == "load":
                await _raise_native_error(response)
            response.raise_for_status()
            body = await response.aread()

        error = f"ocr_invalid_lifecycle_acknowledgement:{action}"
        if response.status_code != 200:
            raise httpx.RemoteProtocolError(error, request=response.request)
        try:
            ready = WorkerReadiness.model_validate_json(body, strict=True)
        except ValidationError:
            raise httpx.RemoteProtocolError(error, request=response.request) from None

        if action == "load":
            confirmed = ready.ready and ready.reason_code is None and bool(ready.profile_id)
        else:
            confirmed = (
                not ready.ready and ready.reason_code == "ocr_parked"
                and ready.profile_id is None and ready.runtime_fingerprint is None
            )
        if not confirmed:
            raise httpx.RemoteProtocolError(error, request=response.request)

    async def claim(self, token: str, deadline: float) -> None:
        await self._lifecycle("claim", token, deadline)

    async def load(self, token: str, deadline: float) -> None:
        await self._lifecycle("load", token, deadline)

    async def park(self, token: str, deadline: float) -> None:
        await self._lifecycle("park", token, deadline)

    async def get_capabilities(self) -> OcrCapabilities:
        reachable = False
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                info_response = await client.get(f"{self.service_url}/info")
                info_response.raise_for_status()
                reachable = True
                info = await asyncio.to_thread(info_response.json)
                if (
                    not isinstance(info, dict)
                    or info.get("contract_version") != 2
                    or not isinstance(info.get("profile"), dict)
                    or not isinstance(info.get("worker", {}), dict)
                ):
                    raise ValueError("ocr_contract_mismatch")
                # Older v2 services omit mode; explicit ownership must match this client.
                expected_mode = "managed" if self.managed_lifecycle else "standalone"
                if "service_mode" in info and info["service_mode"] not in ("managed", "standalone"):
                    raise ValueError("ocr_contract_mismatch")
                if info.get("service_mode", expected_mode) != expected_mode:
                    return OcrCapabilities(
                        enabled=True, configured=True, reachable=True, ready=False,
                        reason_code="ocr_mode_mismatch",
                    )
                response = await client.get(f"{self.service_url}/ready")
                if response.status_code not in (200, 503):
                    response.raise_for_status()
                ready = await asyncio.to_thread(
                    WorkerReadiness.model_validate_json, response.content
                )
                if ready.reason_code in OCR_RESULT_CODES or ready.reason_code == "ocr_parked":
                    if response.status_code != 503 or ready.ready:
                        raise ValueError("ocr_invalid_readiness")
                    # An affirmative native state requires a JSON boolean, not a coerced value.
                    await asyncio.to_thread(
                        WorkerReadiness.model_validate_json, response.content, strict=True,
                    )
                info_profile = info.get("worker", {}).get("profile_id")
                if ready.ready and (
                    response.status_code != 200
                    or not ready.profile_id
                    or (info_profile is not None and ready.profile_id != info_profile)
                ):
                    raise ValueError("ocr_profile_mismatch")
                reported_limit = info["profile"].get("max_new_tokens")
                parked = ready.reason_code == "ocr_parked"
                identity = info.get("runtime_identity", info.get("last_qualified_identity", {})) if parked else {}
                if not isinstance(identity, dict):
                    raise TypeError("ocr_contract_mismatch")
                ocr_limit = (
                    reported_limit
                    if (parked or (ready.ready and info_profile == ready.profile_id))
                    and type(reported_limit) is int
                    and reported_limit > 0
                    else None
                )
                return OcrCapabilities(
                    enabled=True,
                    configured=True,
                    reachable=True,
                    ready=ready.ready,
                    can_load=parked,
                    profile_id=ready.profile_id or identity.get("profile_id"),
                    reason_code=ready.reason_code,
                    runtime_fingerprint=ready.runtime_fingerprint if ready.ready else identity.get("runtime_fingerprint"),
                    ocr_max_new_tokens=ocr_limit,
                )
        except (httpx.HTTPError, OSError, ValueError, TypeError):
            return OcrCapabilities(
                enabled=True,
                configured=True,
                reachable=reachable,
                ready=False,
                reason_code="ocr_unavailable",
            )

    async def extract_page(
        self,
        image: Path,
        *,
        photo_index: int,
        source_sha256: str,
        deadline: float,
        expected_profile_id: str | None = None,
    ) -> OcrResponse:
        raw, media_type = await run_sync_before(
            deadline, self._source_payload, image, source_sha256
        )
        remaining = min(float(self.timeout), deadline - monotonic())
        if remaining <= 0:
            raise TimeoutError("ocr_deadline")
        request_id = uuid4()
        from ...service.gpu_handoff import current_owner
        owner = current_owner()
        if owner is None and self.managed_lifecycle:
            raise ValueError("ocr_managed_session_required")
        async with asyncio.timeout(max(0, deadline - monotonic())):
            async with httpx.AsyncClient(timeout=remaining, trust_env=False) as client:
                async with client.stream(
                    "POST",
                    f"{self.service_url}/ocr",
                    headers={"Accept-Encoding": "identity"},
                    files={"image": (image.name, raw, media_type)},
                    data={
                        "request_id": str(request_id),
                        "photo_index": str(photo_index),
                        "source_sha256": source_sha256,
                        "timeout_ms": str(max(1, min(600000, int(remaining * 1000)))),
                        "owner_token": owner.token if owner is not None else "",
                    },
                ) as response:
                    await _raise_native_error(response)
                    response.raise_for_status()
                    limits.check_ocr_response_headers(
                        response.headers.get("Content-Length"),
                        response.headers.get("Content-Encoding", ""),
                    )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        limits.append_ocr_response_chunk(body, chunk)
        check_deadline(deadline)
        return await run_sync_before(
            deadline,
            self._validated_response,
            body,
            request_id,
            photo_index,
            source_sha256,
            expected_profile_id,
        )

    @staticmethod
    def _read_original(image: Path) -> bytes:
        with image.open("rb") as source:
            raw = source.read(limits.APP_SOURCE_MAX_BYTES + 1)
        if len(raw) > limits.APP_SOURCE_MAX_BYTES:
            raise ValueError("ocr_source_too_large")
        return raw

    @staticmethod
    def _source_payload(image: Path, source_sha256: str) -> tuple[bytes, str]:
        """Pure original-byte/MIME preparation; no caller objects are mutated."""
        raw = PaddleOcrProvider._read_original(image)
        if hashlib.sha256(raw).hexdigest() != source_sha256:
            raise ValueError("ocr_source_hash_mismatch")
        from pillow_heif.as_plugin import register_heif_opener

        register_heif_opener()
        with Image.open(BytesIO(raw)) as decoded:
            media_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "HEIF": "image/heif",
                "GIF": "image/gif",
            }.get(decoded.format or "")
        if media_type is None:
            raise ValueError("ocr_unsupported_media_type")
        return raw, media_type

    @staticmethod
    def _validated_response(
        content: bytes | bytearray,
        request_id: object,
        photo_index: int,
        source_sha256: str,
        expected_profile_id: str | None,
    ) -> OcrResponse:
        """Decode/validate PNG and typed page in an isolated synchronous worker."""
        page = OcrResponse.parse_wire(content)
        if (
            page.request_id != request_id
            or page.photo_index != photo_index
            or page.source_sha256 != source_sha256
            or (expected_profile_id is not None and page.profile_id != expected_profile_id)
        ):
            raise ValueError("ocr_identity_mismatch")
        view = page.working_image
        try:
            validate_working_png(decode_working_png(view.png_base64), view)
        except ValueError as exc:
            raise ValueError("ocr_invalid_working_image") from exc
        return page

    async def extract_text(self, image: Path) -> dict[str, Any]:
        """Compatibility API; live sidecars still require a validated v2 response."""
        from ...service.activation_admission import inference_admission

        deadline = monotonic() + self.timeout
        digest = await run_sync_before(
            deadline, lambda: hashlib.sha256(self._read_original(image)).hexdigest()
        )
        if not self.managed_lifecycle:
            with inference_admission(uses_ocr=True):
                page = await self.extract_page(image, photo_index=0, source_sha256=digest, deadline=deadline)
            return await run_sync_before(deadline, page.model_dump, mode="json")
        from ...config import settings
        from ...service.gpu_handoff import ModelHandoff, admission_deadline
        from ..factory import get_llm_provider
        provider = get_llm_provider(settings.llm)
        model_id = settings.llm.model_name
        handoff = ModelHandoff(settings.ocr.coordination_path, provider, self, shared_gpu=settings.ocr.shared_gpu)
        with inference_admission(provider, handoff=handoff, uses_ocr=True):
            async with handoff.session(deadline, model_id=model_id, lock_deadline=admission_deadline(deadline)) as owner:
                if handoff.shared_gpu:
                    owner.track(model_id)
                await owner.ocr_phase()
                page = await self.extract_page(
                    image, photo_index=0, source_sha256=digest, deadline=deadline
                )
        return await run_sync_before(deadline, page.model_dump, mode="json")

    async def check_connection(self) -> bool:
        capabilities = await self.get_capabilities()
        return capabilities.ready or capabilities.can_load
