import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from ..model.collection import (
    Collection,
    CollectionCreate,
    CollectionDetail,
    CollectionSummary,
    CollectionUpdate,
)
from ..security.dependencies import get_current_user
from ..security.models import User
from ..service.collection_service import (
    CollectionService,
    NameConflictError,
    NotFoundError,
)
from .helper.helper import get_collection_service

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[CollectionSummary])
async def list_collections(
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> list[CollectionSummary]:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    return await receipt_service.list_summaries(current_user.id, current_user.can_see_all)


@router.post("", response_model=Collection, status_code=201)
async def create_collection(
    body: CollectionCreate,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> Collection:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        return await receipt_service.create(body, current_user.id)
    except NameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/active", response_model=Collection)
async def get_active_collection(
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> Collection | Response:
    """The current user's active (auto-capturing) collection; 204 when none.

    Declared before the ``/{collection_id}`` route so the literal path wins.
    """
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    coll = await receipt_service.active_collection(current_user.id, can_see_all=False)
    if coll is None:
        # A bare Response renders an empty body for a 204. JSONResponse(None)
        # would render the 4-byte body "null", which h11 rejects for a 204
        # ("Too much data for declared Content-Length") and aborts the
        # keep-alive connection.
        return Response(status_code=204)
    return coll


@router.get("/{collection_id}", response_model=CollectionDetail)
async def get_collection(
    collection_id: UUID,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> CollectionDetail:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    detail = await receipt_service.get_detail(
        collection_id, current_user.id, current_user.can_see_all
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return detail


@router.patch("/{collection_id}", response_model=Collection)
async def update_collection(
    collection_id: UUID,
    body: CollectionUpdate,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> Collection:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        coll = await receipt_service.update(
            collection_id, body, current_user.id, current_user.can_see_all
        )
    except NameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if coll is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return coll


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: UUID,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> None:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    if not await receipt_service.delete(collection_id, current_user.id, current_user.can_see_all):
        raise HTTPException(status_code=404, detail="Collection not found")


@router.post("/{collection_id}/receipts/{receipt_id}", status_code=200)
async def assign_receipt(
    collection_id: UUID,
    receipt_id: UUID,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, bool]:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        await receipt_service.assign(
            collection_id, receipt_id, current_user.id, current_user.can_see_all
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Collection or receipt not found") from None
    return {"assigned": True}


@router.delete("/{collection_id}/receipts/{receipt_id}", status_code=204)
async def unassign_receipt(
    collection_id: UUID,
    receipt_id: UUID,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> None:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        if not await receipt_service.unassign(
            collection_id, receipt_id, current_user.id, current_user.can_see_all
        ):
            raise HTTPException(status_code=404, detail="Not assigned")
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Collection not found") from None


@router.post("/{collection_id}/activate", response_model=Collection)
async def activate_collection(
    collection_id: UUID,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> Collection:
    """Activate auto-capture for this collection (deactivates the user's other active one)."""
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        coll = await receipt_service.activate(
            collection_id, current_user.id, current_user.can_see_all
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Collection not found") from None
    if coll is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return coll


@router.post("/{collection_id}/deactivate", response_model=Collection)
async def deactivate_collection(
    collection_id: UUID,
    receipt_service: CollectionService = Depends(get_collection_service),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> Collection:
    if not receipt_service.db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        coll = await receipt_service.deactivate(
            collection_id, current_user.id, current_user.can_see_all
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Collection not found") from None
    if coll is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return coll
