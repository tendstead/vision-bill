import logging
from datetime import date
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import Field, model_validator

from ..model.db.receipt import ReceiptRow, ReceiptWithDetails
from ..model.item_category import CATEGORY_CODE_SET
from ..model.ocr import SourceRef
from ..model.receipt import LineItem, Receipt
from ..model.review import AmountPolicy
from ..model.workflow import AccessScope, ReviewPreview, ReviewRequest
from ..security.dependencies import get_current_user
from ..security.models import User
from ..service.image_service import ImageService
from ..service.receipt_service import ReceiptReferencedError, ReceiptService
from .helper.helper import get_image_service, get_receipt_service
from .helper.receipt_json import ReceiptWriteRoute

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)], route_class=ReceiptWriteRoute)


class LineItemUpdate(LineItem):
    id: UUID | None = None
    remember_category: bool = False

    @model_validator(mode="before")
    @classmethod
    def _check_category(cls, value: object) -> object:
        if isinstance(value, dict):
            category = value.get("spending_category", "unknown")
            if not isinstance(category, str) or category not in CATEGORY_CODE_SET:
                raise ValueError("Unsupported spending category")
        return value


class ReceiptUpdate(Receipt):
    """PUT body: the receipt fields plus the collection membership to set.

    ``collection_ids`` distinguishes "omitted" (None -> leave memberships
    untouched) from an explicit empty list (clear all memberships).
    """

    expected_operation_revision: int | None = Field(default=None, ge=1)
    amount_policy: AmountPolicy | None = None
    field_sources: dict[str, list[SourceRef]] | None = None
    overlap_decisions: list[dict[str, object]] | None = None
    collection_ids: list[UUID] | None = None
    discount_total: Decimal | None = Field(
        default=None, ge=0, description="Total discounts applied if printed"
    )
    line_items: list[LineItemUpdate]  # type: ignore[assignment]  # Pydantic parses PUT-only fields


@router.get("")
async def list_receipts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: str | None = Query(
        None, description="Comma-separated statuses, e.g. unverified,verified"
    ),
    date_from: date | None = Query(  # noqa: B008
        None, description="Only receipts dated on or after this date"
    ),
    date_to: date | None = Query(  # noqa: B008
        None, description="Only receipts dated on or before this date"
    ),
    search: str | None = Query(
        None, description="Case-insensitive match on merchant name or receipt number"
    ),
    collection_id: UUID | None = Query(  # noqa: B008
        None, description="Only receipts in this collection"
    ),
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> list[ReceiptRow]:
    """List persisted receipts (the collection resource).

    Optional filters: ``status`` (comma-separated), an inclusive
    ``date_from``/``date_to`` range, ``search`` over merchant name or
    receipt number, and ``collection_id`` to scope to one collection.
    Non-see-all users only list their own receipts.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    statuses = [s.strip() for s in status.split(",") if s.strip()] if status else None
    return await receipt_service.list_receipts(
        limit=limit,
        offset=offset,
        status=statuses,
        date_from=date_from,
        date_to=date_to,
        search=search,
        collection_id=collection_id,
        user_id=current_user.id,
        can_see_all=current_user.can_see_all,
    )


@router.get("/{receipt_id}")
async def get_receipt(
    receipt_id: UUID,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ReceiptWithDetails:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    details = await receipt_service.get_receipt_with_details(
        receipt_id, user_id=current_user.id, can_see_all=current_user.can_see_all
    )
    if details is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return details


@router.put("/{receipt_id}")
async def update_receipt(
    receipt_id: UUID,
    receipt: ReceiptUpdate,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ReceiptRow:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        row = await receipt_service.update_receipt(
            receipt_id, receipt, user_id=current_user.id, can_see_all=current_user.can_see_all
        )
    except ValueError as exc:
        code = str(exc)
        status = 404 if code == "collection_not_found" else 409 if "revision" in code else 422
        raise HTTPException(status_code=status, detail=code) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return row


@router.get("/{receipt_id}/verify")
async def preview_human_review(
    receipt_id: UUID,
    response: Response,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ReviewPreview:
    """Preview current findings for initial human verification or an eligible re-review."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        preview = await receipt_service.preview_human_review(
            receipt_id,
            scope=AccessScope(user_id=current_user.id, can_see_all=current_user.can_see_all),
        )
    except ValueError as exc:
        code = str(exc)
        raise HTTPException(
            status_code=409 if code == "human_rereview_not_available" else 422, detail=code
        ) from exc
    if preview is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    response.headers["Cache-Control"] = "no-store"
    return preview


@router.post("/{receipt_id}/verify")
async def verify_receipt(
    receipt_id: UUID,
    request: ReviewRequest | None = None,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ReceiptRow:
    """Verify against locked current observations and preserve original file paths."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        verified = await receipt_service.verify_receipt(
            receipt_id,
            request or ReviewRequest(),
            scope=AccessScope(user_id=current_user.id, can_see_all=current_user.can_see_all),
        )
    except ValueError as exc:
        code = str(exc)
        status = 409 if "revision" in code or code == "receipt_already_verified" else 422
        detail = "Receipt already verified" if code == "receipt_already_verified" else code
        raise HTTPException(status_code=status, detail=detail) from exc
    if verified is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return verified


@router.delete("/{receipt_id}")
async def delete_receipt(
    receipt_id: UUID,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    image_service: ImageService = Depends(get_image_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, object]:
    """Delete a receipt (its line items and taxes cascade) plus its stored image.

    Non-see-all users may only delete their own receipts.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        deleted = await receipt_service.delete_resource(receipt_id, receipt=True,
            scope=AccessScope(user_id=current_user.id, can_see_all=current_user.can_see_all), images=image_service)
    except ReceiptReferencedError:
        raise HTTPException(status_code=409, detail="Receipt is referenced by a benchmark run and cannot be deleted") from None
    except ValueError:
        raise HTTPException(status_code=409, detail="Receipt evidence is still referenced or has invalid ownership") from None
    if not deleted:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return {"deleted": receipt_id}
