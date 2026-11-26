import json
import logging
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from ocr_contract.limits import APP_SOURCE_MAX_BYTES

from ..config import settings
from ..model.db.image import ImagePart, ImageRow
from ..model.handoff import HandoffWait
from ..model.workflow import AccessScope, SubmissionCreate, SubmissionStatus
from ..security.dependencies import get_current_user
from ..security.models import User
from ..service.activation_admission import policy_capture
from ..service.analysis_scheduler import AnalysisScheduler
from ..service.image_service import ImageService, UnsupportedImageTypeError
from ..service.receipt_service import ReceiptService
from .helper.helper import get_analysis_scheduler, get_image_service, get_receipt_service

router = APIRouter(dependencies=[Depends(get_current_user)])
logger = logging.getLogger(__name__)


def _discard_unreferenced(parts: list[ImagePart], image_service: ImageService) -> None:
    """Caller must prove these uniquely allocated candidate files unreferenced."""
    for part in parts:
        try:
            image_service.delete_image(Path(part.image_path))
        except OSError:
            # Cleanup may leave an orphan, but cannot change committed acceptance.
            logger.warning("Could not remove unreferenced upload candidate")


def _location(image_id: UUID) -> str:
    """Absolute path to a single-image resource for the Location header."""
    return f"/api/v1/images/{image_id}"


@router.post("", status_code=202, response_model=SubmissionStatus)
async def upload_image(
    receipt: list[UploadFile] = File(...),  # noqa: B008
    model_id: str | None = None,
    bypass_review: bool | None = Query(None),
    allow_ocr_fallback: bool | None = Query(None),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    image_service: ImageService = Depends(get_image_service),  # noqa: B008
    analysis_scheduler: AnalysisScheduler = Depends(get_analysis_scheduler),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> JSONResponse:
    """Accept ordered originals durably; extraction belongs to the worker."""
    if not receipt_service.db_ready or not receipt_service.workflow_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    if idempotency_key is not None and not idempotency_key.strip():
        raise HTTPException(status_code=422, detail="Idempotency-Key must not be empty")
    if not 1 <= len(receipt) <= 10:
        raise HTTPException(status_code=422, detail="Upload between 1 and 10 photos of one receipt")

    submission_id = uuid4()
    parts: list[ImagePart] = []
    # Before enqueue begins these paths cannot be referenced by a DB transaction.
    try:
        for index, upload in enumerate(receipt):
            content = await upload.read(APP_SOURCE_MAX_BYTES + 1)
            if len(content) > APP_SOURCE_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Each photo must be at most 20 MiB")
            try:
                info = image_service.validate_and_inspect(content)
            except UnsupportedImageTypeError as exc:
                raise HTTPException(status_code=415, detail="Unsupported image type") from exc
            part = image_service.store_original(content, submission_id, index, info.media_type)
            parts.append(part)
            part.original_filename = upload.filename
            thumb = image_service.generate_thumbnail(Path(part.image_path))
            part.thumbnail_path = str(thumb) if thumb else None
        with policy_capture(settings.ocr):
            effective_options = receipt_service.current_execution_snapshot(settings)
    except HandoffWait as exc:
        _discard_unreferenced(parts, image_service)
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except BaseException:
        _discard_unreferenced(parts, image_service)
        raise

    requested: dict[str, object] = {}
    if model_id is not None:
        requested["model_id"] = model_id
    if bypass_review is not None:
        requested["bypass_review"] = bypass_review
    if allow_ocr_fallback is not None:
        requested["allow_ocr_fallback"] = allow_ocr_fallback
    manifest = [
        {"photo_index": index, "source_sha256": part.source_sha256}
        for index, part in enumerate(parts)
    ]
    fingerprint = sha256(
        json.dumps(
            {"sources": manifest, "requested_options": requested},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    submission = SubmissionCreate(
        id=submission_id,
        parts=parts,
        source_manifest=manifest,
        requested_options=requested,
        request_fingerprint=fingerprint,
        effective_options=effective_options,
        model_id=model_id if model_id is not None else settings.llm.model_name,
        bypass_review=(
            settings.images.bypass_review_default if bypass_review is None else bypass_review
        ),
    )
    try:
        result = await receipt_service.enqueue_submission(
            submission, scope=AccessScope(user_id=current_user.id), idempotency_key=idempotency_key
        )
    except ValueError as exc:
        if str(exc) == "upload idempotency payload conflict":
            # Task 7 raises only after rejecting identity before any INSERT;
            # its transaction has rolled back. These unique paths never entered
            # the database. All ambiguous outcomes retain their candidate files.
            _discard_unreferenced(parts, image_service)
            raise HTTPException(
                status_code=409, detail="Idempotency-Key conflicts with upload"
            ) from exc
        raise
    if not result.created and result.submission.id != submission_id:
        # A completed replay returned another UUID; this unique candidate was
        # never inserted and cannot contain any winner's referenced originals.
        _discard_unreferenced(parts, image_service)
    analysis_scheduler.trigger()
    return JSONResponse(
        status_code=202,
        headers={"Location": _location(result.submission.id)},
        content=result.submission.model_dump(mode="json"),
    )


@router.get("")
async def list_images(
    status: str | None = Query(
        None,
        description="Comma-separated statuses to filter by, e.g. pending,failed",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> list[ImageRow]:
    """List image resources, newest first. Filter with ``?status=pending,failed``."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    statuses = [s.strip() for s in status.split(",") if s.strip()] if status else None
    return await receipt_service.list_images(
        status=statuses,
        limit=limit,
        offset=offset,
        user_id=current_user.id,
        can_see_all=current_user.can_see_all,
    )


@router.post("/analyze")
async def analyze_pending(
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    analysis_scheduler: AnalysisScheduler = Depends(get_analysis_scheduler),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, object]:
    """Manually trigger one analysis cycle over the pending queue."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    results = await analysis_scheduler.process_pending(
        scope=AccessScope(user_id=current_user.id, can_see_all=current_user.can_see_all)
    )
    return {"results": results}


@router.get("/{image_id}", response_model=SubmissionStatus)
async def get_image(
    image_id: UUID,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> SubmissionStatus:
    """Read the persistent public status, including immediately after acceptance."""
    if not receipt_service.db_ready or not receipt_service.workflow_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    submission = await receipt_service.get_submission(
        image_id, scope=AccessScope(user_id=current_user.id, can_see_all=current_user.can_see_all)
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return submission


@router.get("/{image_id}/file")
async def get_image_file(
    image_id: UUID,
    part: int = Query(0, ge=0),
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> FileResponse:
    """Stream the stored image file behind an image resource."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    image = await receipt_service.get_image_by_id(
        image_id, user_id=current_user.id, can_see_all=current_user.can_see_all
    )
    if image is None or not image.image_path:
        raise HTTPException(status_code=404, detail="Image file not found")
    parts = image.parts()
    if part >= len(parts):
        raise HTTPException(status_code=404, detail="Image part not found")
    source = parts[part]
    path = Path(source.image_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(
        path,
        media_type=source.media_type or "application/octet-stream",
        headers={"Cache-Control": "private, no-store"},
    )


@router.get("/{image_id}/thumb")
async def get_image_thumb(
    image_id: UUID,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> FileResponse:
    """Serve the stored thumbnail for an image (404 when there is none).

    Dumb file server: returns the thumb or 404. It never falls back to the full
    image — missing-thumb handling is the frontend's decision.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    image = await receipt_service.get_image_by_id(
        image_id, user_id=current_user.id, can_see_all=current_user.can_see_all
    )
    if image is None or not image.thumbnail_path:
        raise HTTPException(status_code=404, detail="Image thumbnail not found")
    path = Path(image.thumbnail_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image thumbnail not found")
    return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "private, no-store"})


@router.delete("/{image_id}")
async def delete_image(
    image_id: UUID,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    image_service: ImageService = Depends(get_image_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, object]:
    """Delete an owned pending, needs-review, failed, timed-out or unreadable submission.

    Unfinalized drafts and owned files are removed through durable cleanup;
    processing submissions and submissions linked to receipts conflict.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        deleted = await receipt_service.delete_resource(image_id, receipt=False,
            scope=AccessScope(user_id=current_user.id, can_see_all=current_user.can_see_all), images=image_service)
    except ValueError:
        raise HTTPException(status_code=409, detail="Cannot delete an active or referenced submission") from None
    if not deleted:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"deleted": image_id}
