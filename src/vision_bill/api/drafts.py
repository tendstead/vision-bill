"""Owner-scoped review drafts, retries, immutable extraction snapshots and PNG views."""

from time import monotonic
from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import ValidationError

from ..helper.deadline import run_sync_before
from ..model.db.receipt import ReceiptRow
from ..model.extraction import DraftView
from ..model.handoff import HandoffWait
from ..model.workflow import AccessScope, DraftFinalize, DraftPatch, RetryRequest, SubmissionStatus
from ..security.dependencies import get_current_user
from ..security.models import User
from ..service.activation_admission import policy_capture
from ..service.analysis_scheduler import AnalysisScheduler
from ..service.submission_service import SubmissionService
from .helper.helper import get_analysis_scheduler
from .helper.receipt_json import ReceiptWriteRoute

router = APIRouter(dependencies=[Depends(get_current_user)], route_class=ReceiptWriteRoute)


def get_submission_service(request: Request) -> SubmissionService:
    service = cast(SubmissionService | None, getattr(request.app.state, "submission_service", None))
    if service is None or not service.db.is_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    return service


def access_scope(user: User = Depends(get_current_user)) -> AccessScope:  # noqa: B008
    return AccessScope(user_id=user.id, can_see_all=user.can_see_all)


def mutation_error(error: ValueError) -> HTTPException:
    conflict = str(error) in {
        "draft_revision_conflict",
        "draft_already_finalized",
        "draft_processing_conflict",
        "draft_state_conflict",
    }
    return HTTPException(
        status_code=409 if conflict else 422,
        detail="Draft conflict" if conflict else "Draft cannot be finalized or updated",
    )


@router.get("/{image_id}/draft")
async def get_draft(
    image_id: UUID,
    service: SubmissionService = Depends(get_submission_service),  # noqa: B008
    scope: AccessScope = Depends(access_scope),  # noqa: B008
) -> DraftView:
    draft = await service.get_draft(image_id, scope=scope)
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft


@router.patch("/{image_id}/draft")
async def patch_draft(
    image_id: UUID,
    patch: DraftPatch,
    service: SubmissionService = Depends(get_submission_service),  # noqa: B008
    scope: AccessScope = Depends(access_scope),  # noqa: B008
) -> DraftView:
    try:
        draft = await service.patch_draft(image_id, patch, scope=scope)
    except (ValueError, ValidationError) as exc:
        raise mutation_error(exc) from None
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft


@router.post("/{image_id}/draft/finalize")
async def finalize_draft(
    image_id: UUID,
    request: DraftFinalize,
    service: SubmissionService = Depends(get_submission_service),  # noqa: B008
    scope: AccessScope = Depends(access_scope),  # noqa: B008
) -> ReceiptRow:
    try:
        receipt = await service.finalize_draft(image_id, request, scope=scope)
    except ValueError as exc:
        raise mutation_error(exc) from None
    if receipt is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    return receipt


@router.post("/{image_id}/retry", status_code=202)
async def retry(
    image_id: UUID,
    request: RetryRequest,
    service: SubmissionService = Depends(get_submission_service),  # noqa: B008
    scope: AccessScope = Depends(access_scope),  # noqa: B008
    scheduler: AnalysisScheduler = Depends(get_analysis_scheduler),  # noqa: B008
) -> SubmissionStatus:
    try:
        replacement_snapshot = None
        if request.policy == "current":
            with policy_capture(service.settings.ocr):
                replacement_snapshot = service.current_snapshot()
        status = await service.db.retry_submission(
            image_id, request, scope=scope,
            replacement_snapshot=replacement_snapshot,
        )
    except HandoffWait as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError:
        raise HTTPException(status_code=409, detail="Retry conflict") from None
    if status is None:
        raise HTTPException(status_code=404, detail="Image not found")
    scheduler.trigger()
    return status


@router.get("/{image_id}/extractions/{extraction_id}")
async def get_extraction(
    image_id: UUID,
    extraction_id: UUID,
    service: SubmissionService = Depends(get_submission_service),  # noqa: B008
    scope: AccessScope = Depends(access_scope),  # noqa: B008
) -> dict[str, Any]:
    extraction = await service.db.get_extraction(image_id, extraction_id, scope=scope)
    if extraction is None:
        raise HTTPException(status_code=404, detail="Extraction not found")
    return extraction


@router.get("/{image_id}/views/{view_id}")
async def get_view(
    image_id: UUID,
    view_id: UUID,
    service: SubmissionService = Depends(get_submission_service),  # noqa: B008
    scope: AccessScope = Depends(access_scope),  # noqa: B008
) -> Response:
    view = await service.db.get_working_view(image_id, view_id, scope=scope)
    if view is None:
        raise HTTPException(status_code=404, detail="View not found")
    try:
        # Return the exact validated bytes, avoiding a second file open after validation.
        png = await run_sync_before(
            monotonic() + 30,
            service.images.read_working_image,
            image_id,
            view.artifact_path,
            view.metadata,
        )
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="View not available") from None
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, no-store"})
