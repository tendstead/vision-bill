import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..model.item_category import CATEGORY_CODES
from ..security.dependencies import get_current_user
from ..security.models import User
from ..service.receipt_service import ReceiptService
from .helper.helper import get_receipt_service

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])


class TagCreate(BaseModel):
    """Body for creating (or confirming) a tag in the vocabulary."""

    name: str = Field(min_length=1, max_length=100)


@router.get("/categories")
async def list_categories() -> list[dict[str, str]]:
    """Stable codes for one spending category per line item."""
    return [{"code": code} for code in CATEGORY_CODES]


class CategoryQuery(BaseModel):
    merchant_name: str = Field(min_length=1)
    merchant_display_name: str | None = Field(None, max_length=200)
    merchant_address: str | None = None
    description: str = Field(min_length=1)
    receipt_id: UUID | None = None


@router.post("/category-suggestion")
async def editor_category_suggestion(
    query: CategoryQuery,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, str | None]:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    category = await receipt_service.get_category_suggestion(
        current_user.id, query.merchant_name, query.description,
        display_name=query.merchant_display_name, address=query.merchant_address,
        receipt_id=query.receipt_id,
    )
    return {"category": category, "source": "history" if category else None}


@router.get("/category-suggestion")
async def category_suggestion(
    merchant_name: str = Query(min_length=1),
    description: str = Query(min_length=1),
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, str | None]:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    category = await receipt_service.get_category_suggestion(
        current_user.id, merchant_name, description
    )
    return {"category": category, "source": "history" if category else None}


@router.get("")
async def list_tags(
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
) -> list[str]:
    """Return the allowed line-item tag vocabulary, ordered by name.

    This is the source of truth behind the tag <select> in the UI.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    return await receipt_service.list_tags()


@router.post("", response_model=None)
async def create_tag(
    body: TagCreate,
    receipt_service: ReceiptService = Depends(get_receipt_service),  # noqa: B008
) -> JSONResponse:
    """Create a tag, idempotently.

    Names are normalized (trimmed, lower-cased). A tag that already exists is
    returned as-is (200); a newly created one returns 201. This lets the UI
    "promote" an LLM-suggested tag without racing a duplicate insert.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        name, created = await receipt_service.create_tag(body.name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return JSONResponse(
        status_code=201 if created else 200,
        content={"name": name, "created": created},
    )
