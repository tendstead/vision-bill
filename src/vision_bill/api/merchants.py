from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..model.db.receipt import MerchantOptions
from ..model.workflow import AccessScope
from ..security.dependencies import get_current_user
from ..security.models import User
from ..service.receipt_service import ReceiptService
from .helper.helper import get_receipt_service

router = APIRouter()


class MerchantQuery(BaseModel):
    merchant_name: str = Field("", max_length=4096)
    merchant_address: str | None = Field(None, max_length=4096)
    q: str = Field("", max_length=200)
    receipt_id: UUID | None = None
    image_id: UUID | None = None


@router.post("/suggestions", response_model=MerchantOptions)
async def merchant_options(
    query: MerchantQuery,
    current_user: User = Depends(get_current_user),  # noqa: B008
    service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
) -> MerchantOptions:
    if query.receipt_id and query.image_id:
        raise HTTPException(422, "Specify a receipt or an image")
    if not service.db_ready:
        raise HTTPException(503, "Database not available")
    result = await service.merchant_options(
        AccessScope(user_id=current_user.id, can_see_all=current_user.can_see_all),
        query.merchant_name, query.merchant_address, query.q, query.receipt_id, query.image_id,
    )
    if result is None:
        raise HTTPException(404, "Receipt or image not found")
    return result
