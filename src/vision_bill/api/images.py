import asyncio
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config import settings
from ..model.db.image import ImagePart, ImageRow
from ..provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError
from ..security.dependencies import get_current_user
from ..security.models import User
from ..service.analysis_scheduler import AnalysisScheduler
from ..service.image_service import ImageService, UnsupportedImageTypeError
from ..service.receipt_service import ReceiptService
from .helper.helper import get_analysis_scheduler, get_image_service, get_receipt_service

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])

PENDING_QUEUE_WARNING = "LLM provider not available – image queued for background analysis"


def _location(image_id: UUID) -> str:
    """Absolute path to a single-image resource for the Location header."""
    return f"/api/v1/images/{image_id}"


@router.post("", status_code=201, response_model=None)
async def upload_image(
    receipt: list[UploadFile] = File(...),  # noqa: B008
    model_id: str | None = None,
    bypass_review: bool | None = Query(None),
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    image_service: ImageService = Depends(get_image_service),  # noqa: B008
    analysis_scheduler: AnalysisScheduler = Depends(get_analysis_scheduler),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> JSONResponse:
    """Create a receipt submission from one or more ordered photos.

    Multiple photos return 202 immediately after storage and wake the scheduler.
    Their image UUID identifies the durable submission and all its source photos.

    For one photo, when a model is reachable the image is analysed synchronously, a
    receipt is persisted and a ``201`` is returned. When no model is reachable
    the image is queued as ``pending`` and a ``202`` is returned; the background
    scheduler picks it up later. Both responses carry a ``Location`` header so a
    client can poll the image and, once analysed, follow ``receipt_id`` to the
    receipt. The tmp image file is always retained so a later verify step (or a
    background retry) can still find it.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")

    effective_bypass_review = (
        settings.images.bypass_review_default if bypass_review is None else bypass_review
    )

    if not 1 <= len(receipt) <= 10:
        raise HTTPException(status_code=422, detail="Upload between 1 and 10 photos of one receipt")
    parts: list[ImagePart] = []
    try:
        for upload in receipt:
            content = await upload.read(20 * 1024 * 1024 + 1)
            if len(content) > 20 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Each photo must be at most 20 MB")
            try:
                info = image_service.validate_and_inspect(content)
            except UnsupportedImageTypeError as exc:
                raise HTTPException(status_code=415, detail="Unsupported image type") from exc
            path = image_service.store_tmp_image(content)
            part = ImagePart(
                image_path=str(path),
                original_filename=upload.filename,
                media_type=info.media_type,
                size_bytes=info.size_bytes,
            )
            parts.append(part)
            thumb = image_service.generate_thumbnail(path)
            part.thumbnail_path = str(thumb) if thumb else None
    except BaseException:
        for part in parts:
            image_service.delete_image(Path(part.image_path))
        raise

    first = parts[0]
    tmp_path = Path(first.image_path)
    models = []
    if len(parts) == 1:
        try:
            async with asyncio.timeout(5):
                if await receipt_service.check_connection():
                    models = await receipt_service.get_available_models()
        except Exception:  # noqa: BLE001 - discovery failure leaves the persisted upload queued
            logger.warning("LLM provider unreachable - queueing image for background analysis")
    provider_available = bool(models)

    try:
        image_row = await receipt_service.store_image(
            image_path=first.image_path,
            original_filename=first.original_filename,
            media_type=first.media_type,
            size_bytes=first.size_bytes,
            status="pending",
            user_id=current_user.id,
            bypass_review=effective_bypass_review,
            thumbnail_path=first.thumbnail_path,
            additional_images=parts[1:],
            model_id=model_id,
        )
    except BaseException:
        for part in parts:
            image_service.delete_image(Path(part.image_path))
        raise

    if len(parts) > 1:
        analysis_scheduler.trigger()
        return JSONResponse(
            status_code=202,
            headers={"Location": _location(image_row.id)},
            content={"image_id": str(image_row.id), "status": "pending"},
        )

    if not provider_available:
        return JSONResponse(
            status_code=202,
            headers={"Location": _location(image_row.id)},
            content={
                "image_id": str(image_row.id),
                "status": "pending",
                "warning": PENDING_QUEUE_WARNING,
            },
        )

    claimed = await receipt_service.claim_image_for_analysis(image_row.id)
    if claimed is None:
        logger.info("Image %s was claimed by the background worker", image_row.id)
        return JSONResponse(
            status_code=202,
            headers={"Location": _location(image_row.id)},
            content={
                "image_id": str(image_row.id),
                "status": "pending",
                "warning": PENDING_QUEUE_WARNING,
            },
        )
    image_row = claimed

    available_ids = {m.id for m in models}
    preferred_model = settings.llm.model_name
    chosen_model = (
        model_id
        if model_id and model_id in available_ids
        else preferred_model
        if preferred_model in available_ids
        else models[0].id
    )
    try:
        llm_response = await receipt_service.analyse_receipt_with_ocr(chosen_model, tmp_path)
    except (AnalysisTimeoutError, UnreadableReceiptError) as exc:
        status = "timed_out" if isinstance(exc, AnalysisTimeoutError) else "unreadable"
        await receipt_service.mark_image_terminal(image_row.id, status, str(exc))
        return JSONResponse(
            status_code=504 if status == "timed_out" else 422,
            headers={"Location": _location(image_row.id)},
            content={"detail": str(exc), "image_id": str(image_row.id), "status": status},
        )
    except Exception as exc:
        await receipt_service.mark_image_failed(image_row.id, str(exc))
        raise

    if effective_bypass_review:
        row = await receipt_service.persist_receipt(
            llm_response,
            image_id=image_row.id,
            status="verified",
            verified=True,
            user_id=current_user.id,
        )
        await receipt_service.mark_image_analyzed(image_row.id, row.id)
        try:
            perm_path, perm_thumb = image_service.store_perm_image(tmp_path, row.id)
            if perm_path is not None:
                await receipt_service.update_image_path(image_row.id, str(perm_path))
            if perm_thumb is not None:
                await receipt_service.update_image_thumbnail_path(image_row.id, str(perm_thumb))
        except Exception:
            logger.exception(
                "Failed to move bypass-reviewed image %s to permanent storage", image_row.id
            )
    else:
        row = await receipt_service.persist_receipt(
            llm_response, image_id=image_row.id, status="unverified", user_id=current_user.id
        )
        await receipt_service.mark_image_analyzed(image_row.id, row.id)

    return JSONResponse(
        status_code=201,
        headers={"Location": _location(image_row.id)},
        content={
            "image_id": str(image_row.id),
            "status": "analyzed",
            "receipt_id": str(row.id),
            "original_filename": image_row.original_filename,
            "media_type": first.media_type,
            "size_bytes": first.size_bytes,
            "image_path": image_row.image_path,
        },
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
) -> dict[str, object]:
    """Manually trigger one analysis cycle over the pending queue."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    results = await analysis_scheduler.process_pending()
    return {"results": results}


@router.get("/{image_id}")
async def get_image(
    image_id: UUID,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ImageRow:
    """Fetch a single image resource by id."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    image = await receipt_service.get_image_by_id(
        image_id, user_id=current_user.id, can_see_all=current_user.can_see_all
    )
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return image


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
    return FileResponse(path, media_type=source.media_type or "application/octet-stream")


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
    return FileResponse(path, media_type="image/webp")


@router.delete("/{image_id}")
async def delete_image(
    image_id: UUID,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    image_service: ImageService = Depends(get_image_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, object]:
    """Remove a queued (pending or failed) image resource and its on-disk file."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")

    image = await receipt_service.get_image_by_id(
        image_id, user_id=current_user.id, can_see_all=current_user.can_see_all
    )
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.status not in ("pending", "failed", "timed_out", "unreadable"):
        raise HTTPException(
            status_code=409, detail="Cannot delete an active or analyzed submission"
        )

    await receipt_service.delete_image_row(image_id)
    for part in image.parts():
        if part.image_path:
            image_service.delete_image(Path(part.image_path))
    return {"deleted": image_id}
