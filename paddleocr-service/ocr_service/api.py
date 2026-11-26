"""Original-image HTTP ingress, validated working PNGs, and worker lifecycle."""

import asyncio
import hashlib
import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from io import BytesIO
from pathlib import Path
from typing import Annotated, TypeVar
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from ocr_contract import OcrResponse
from ocr_contract.document_image import decode_working_png, validate_working_png
from ocr_contract.limits import MAX_IMAGE_PIXELS, NATIVE_SOURCE_MAX_BYTES
from PIL import Image
from pydantic import BaseModel, Field, TypeAdapter

from .profile import PreparedRuntimeSettings, RuntimeSettings, load_profile
from .worker import OcrRequest, OcrWorkerError, OcrWorkerManager

logger = logging.getLogger(__name__)

SOURCE_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/heic": "HEIF",
    "image/heif": "HEIF",
    "image/gif": "GIF",
}


class LifecycleRequest(BaseModel):
    timeout_ms: int | None = Field(default=None, gt=0, le=600000)
    owner_token: str = Field(pattern=r"^[1-9][0-9]*:[a-zA-Z0-9-]+$", max_length=100)


def configured_worker() -> OcrWorkerManager:
    runtime_path = os.environ.get("OCR_RUNTIME_SETTINGS_PATH")
    if not runtime_path:
        # Missing configuration is surfaced by the child's stable init failure.
        return OcrWorkerManager()
    return OcrWorkerManager(
        runtime=TypeAdapter(PreparedRuntimeSettings | RuntimeSettings).validate_json(
            Path(runtime_path).read_bytes()
        ),
        startup_timeout=float(os.environ.get("OCR_STARTUP_TIMEOUT_SECONDS", "600")),
    )


def validate_source(raw: bytes, media_type: str | None, *, limit: int) -> None:
    if len(raw) > limit:
        raise HTTPException(413, "ocr_payload_too_large")
    if media_type not in SOURCE_FORMATS:
        raise HTTPException(415, "ocr_unsupported_media_type")
    from pillow_heif.as_plugin import register_heif_opener

    register_heif_opener()
    try:
        with Image.open(BytesIO(raw)) as decoded:
            if (
                decoded.format != SOURCE_FORMATS[media_type]
                or decoded.width * decoded.height > MAX_IMAGE_PIXELS
            ):
                raise ValueError("Invalid source format or dimensions")
            decoded.verify()
        with Image.open(BytesIO(raw)) as decoded:
            decoded.seek(0)
            decoded.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(415, "ocr_invalid_image") from None


def validate_result(result: OcrResponse) -> None:
    try:
        result.check_wire_bounds()
        view = result.working_image
        validate_working_png(decode_working_png(view.png_base64), view)
    except (TypeError, ValueError):
        raise HTTPException(502, "ocr_invalid_result") from None


T = TypeVar("T")


async def run_until_disconnect(
    http_request: Request, cancel_event: threading.Event, operation: Callable[[], T],
) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))

    async def observe_disconnect() -> None:
        # FastAPI has consumed the body before the operation starts.
        # A disconnect therefore belongs to this admitted request only.
        while True:
            message = await http_request.receive()
            if message["type"] == "http.disconnect":
                cancel_event.set()
                return

    observer = asyncio.create_task(observe_disconnect())
    try:
        result = await asyncio.shield(task)
        if cancel_event.is_set():
            raise HTTPException(499, "ocr_cancelled")
        return result
    except asyncio.CancelledError:
        cancel_event.set()
        # Drain native ownership before closing an upload or ending lifespan.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - drain failure while preserving cancellation
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise
    finally:
        observer.cancel()
        with suppress(asyncio.CancelledError):
            await observer


def create_app(
    *,
    mode: str = "managed",
    worker_factory: Callable[[], OcrWorkerManager] = configured_worker,
    max_upload_bytes: int = NATIVE_SOURCE_MAX_BYTES,
) -> FastAPI:
    if mode not in ("managed", "standalone"):
        raise ValueError("OCR service mode must be managed or standalone")
    worker: OcrWorkerManager | None = None
    startup_failed = False
    closing = False

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal closing

        async def initialize() -> None:
            nonlocal worker, startup_failed
            try:
                worker = await asyncio.to_thread(worker_factory)
                if closing:
                    await asyncio.to_thread(worker.stop)
                    return
            except Exception:  # noqa: BLE001 - safe lifecycle failure boundary
                startup_failed = True

        initialization = asyncio.create_task(initialize())
        try:
            yield
        finally:
            closing = True
            # The factory only parses local settings; awaiting it prevents late
            # publication of an orphan manager. Native initialization observes stop.
            if worker is not None:
                await asyncio.to_thread(worker.stop)
            await initialization
            if worker is not None:
                await asyncio.to_thread(worker.stop)

    app = FastAPI(title="PaddleOCR-VL-1.6 Service", version="2.0.0", lifespan=lifespan)

    def ready_body() -> dict[str, object]:
        if worker is None:
            return {
                "ready": False,
                "generation": 1,
                "profile_id": None,
                "reason_code": "ocr_init_failed" if startup_failed else "ocr_starting",
            }
        return worker.readiness().model_dump()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        body = ready_body()
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    @app.get("/info")
    async def info() -> dict[str, object]:
        if worker is None:
            body = ready_body()
            identity: dict[str, str | None] = {}
        else:
            state, identity = worker.readiness_with_identity()
            body = state.model_dump(exclude={"runtime_identity"})
        return {
            "contract_version": 2,
            "service_mode": mode,
            "profile": load_profile(Path(__file__).parents[1] / "profile.json").model_dump(),
            "worker": body,
            "runtime_identity": identity,
            "max_upload_bytes": max_upload_bytes,
            "max_pending": 4,
            "max_timeout_ms": 600000,
        }

    def process_request(
        image: UploadFile,
        request_id: UUID,
        photo_index: int,
        source_sha256: str,
        deadline: float,
        owner_token: str | None,
        cancel_event: threading.Event,
    ) -> Response:
        if mode == "standalone" and owner_token is not None:
            raise HTTPException(403, "ocr_standalone_lifecycle_owned")
        if worker is None:
            raise HTTPException(503, "ocr_init_failed" if startup_failed else "ocr_starting")
        raw = image.file.read(max_upload_bytes + 1)
        validate_source(raw, image.content_type, limit=max_upload_bytes)
        if hashlib.sha256(raw).hexdigest() != source_sha256:
            raise HTTPException(422, "ocr_source_hash_mismatch")
        path: Path | None = None
        cleanup_safe = True
        try:
            with tempfile.NamedTemporaryFile(
                prefix="vision-bill-upload-", suffix=".source", delete=False
            ) as source:
                path = Path(source.name)
                source.write(raw)
            request = OcrRequest(request_id, path, photo_index, source_sha256)
            if mode == "standalone":
                result = worker.infer_standalone(
                    request,
                    deadline_monotonic=deadline,
                    cancel_event=cancel_event,
                )
            else:
                result = worker.infer(
                    request,
                    deadline_monotonic=deadline,
                    owner=owner_token,
                    cancel_event=cancel_event,
                )
            encoding_started = time.monotonic()
            validate_result(result)
            response = Response(result.dump_checked_wire(), media_type="application/json")
            if time.monotonic() >= deadline:
                # The request no longer owns a predictor. A later page may already
                # be active, so rejecting this late response must never reap it.
                logger.warning("ocr_worker_request phase=response_encode state=failed "
                               "reason=ocr_deadline_exceeded request_id=%s elapsed_ms=%d",
                               request_id, round((time.monotonic() - encoding_started) * 1000))
                raise HTTPException(504, "ocr_deadline_exceeded")
            return response
        except OcrWorkerError as exc:
            cleanup_safe = exc.cleanup_safe
            raise HTTPException(exc.status_code, exc.code) from None
        finally:
            if cleanup_safe and path is not None:
                path.unlink(missing_ok=True)

    @app.post("/ocr", response_model=OcrResponse)
    async def ocr(
        http_request: Request,
        image: Annotated[UploadFile, File()],
        request_id: Annotated[UUID, Form()],
        photo_index: Annotated[int, Form(ge=0)],
        source_sha256: Annotated[str, Form(pattern=r"^[0-9a-f]{64}$")],
        timeout_ms: Annotated[int, Form(gt=0, le=600000)],
        owner_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        deadline = time.monotonic() + timeout_ms / 1000
        cancel_event = threading.Event()
        return await run_until_disconnect(
            http_request, cancel_event,
            lambda: process_request(image, request_id, photo_index, source_sha256,
                                    deadline, owner_token, cancel_event),
        )

    @app.post("/lifecycle/{action}")
    async def lifecycle(action: str, body: LifecycleRequest, http_request: Request) -> dict[str, object]:
        if mode == "standalone":
            raise HTTPException(403, "ocr_standalone_lifecycle_owned")
        if worker is None or closing:
            raise HTTPException(503, "ocr_unavailable")
        if action not in ("claim", "load", "park"):
            raise HTTPException(404, "ocr_unknown_lifecycle_action")
        try:
            if action == "load":
                manager = worker
                cancel_event = threading.Event()
                deadline = time.monotonic() + body.timeout_ms / 1000 if body.timeout_ms else None
                await run_until_disconnect(
                    http_request, cancel_event,
                    lambda: manager.load(body.owner_token, deadline_monotonic=deadline,
                                         cancel_event=cancel_event),
                )
            else:
                await asyncio.to_thread(getattr(worker, action), body.owner_token)
        except OcrWorkerError as exc:
            raise HTTPException(exc.status_code, exc.code) from None
        return worker.readiness().model_dump(exclude={"runtime_identity"})

    return app
