"""Private, admin-scoped photo cases and human benchmark expectations."""

import logging
from pathlib import Path
from typing import Annotated, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from ocr_contract.limits import APP_SOURCE_MAX_BYTES

from ..model.benchmark_case import (
    BenchmarkCase,
    BenchmarkCaseSummary,
    BenchmarkCaseUpdate,
    CaseName,
)
from ..model.workflow import AccessScope
from ..provider.db.benchmark_cases import BenchmarkCaseDB, CaseNotFound, CaseRevisionConflict
from ..security.dependencies import require_admin
from ..security.models import User
from ..service.image_service import ImageService, UnsupportedImageTypeError
from .helper.helper import get_image_service

router = APIRouter()
logger = logging.getLogger(__name__)


def case_store(request: Request) -> BenchmarkCaseDB:
    store = getattr(request.app.state, "benchmark_cases", None)
    if store is None:
        raise HTTPException(503, "Database is unavailable")
    return cast(BenchmarkCaseDB, store)


def case_scope(user: Annotated[User, Depends(require_admin)]) -> AccessScope:
    return AccessScope(user_id=user.id, can_see_all=user.can_see_all)


Scope = Annotated[AccessScope, Depends(case_scope)]
Store = Annotated[BenchmarkCaseDB, Depends(case_store)]


@router.post("", response_model=BenchmarkCase, status_code=201)
async def upload_case(
    response: Response,
    name: Annotated[CaseName, Form()],
    photos: Annotated[list[UploadFile], File()],
    scope: Scope,
    store: Store,
    image_service: Annotated[ImageService, Depends(get_image_service)],
) -> BenchmarkCase:
    if not 1 <= len(photos) <= 10:
        raise HTTPException(422, "Upload between 1 and 10 photos of one case")
    case_id = uuid4()
    sources = []
    try:
        for index, upload in enumerate(photos):
            content = await upload.read(APP_SOURCE_MAX_BYTES + 1)
            if len(content) > APP_SOURCE_MAX_BYTES:
                raise HTTPException(413, "Each photo must be at most 20 MiB")
            try:
                info = image_service.validate_and_inspect(content)
            except UnsupportedImageTypeError as exc:
                raise HTTPException(415, "Unsupported image type") from exc
            part = image_service.store_original(
                content, case_id, index, info.media_type, namespace="benchmark_cases"
            )
            sources.append(
                {
                    "path": part.image_path,
                    "media_type": part.media_type,
                    "sha256": part.source_sha256,
                    "photo_index": index,
                }
            )
    except BaseException:
        for source in sources:
            try:
                Path(str(source["path"])).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove unreferenced benchmark case photo")
        raise
    # A connection failure after dispatch can have an ambiguous commit outcome.
    # At that point the newly allocated originals must be retained.
    result = await store.create(case_id, name, sources, scope=scope)
    response.headers["Location"] = f"/api/v1/benchmarks/cases/{case_id}"
    return result


@router.get("", response_model=list[BenchmarkCaseSummary])
async def list_cases(scope: Scope, store: Store) -> list[BenchmarkCaseSummary]:
    return await store.list_cases(scope=scope)


@router.get("/{case_id}", response_model=BenchmarkCase)
async def get_case(case_id: UUID, scope: Scope, store: Store) -> BenchmarkCase:
    try:
        return await store.get(case_id, scope=scope)
    except CaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.patch("/{case_id}", response_model=BenchmarkCase)
async def update_case(
    case_id: UUID, update: BenchmarkCaseUpdate, scope: Scope, store: Store
) -> BenchmarkCase:
    try:
        return await store.update(case_id, update, scope=scope)
    except CaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except CaseRevisionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{case_id}/photos/{index}", response_class=Response)
async def get_photo(case_id: UUID, index: int, scope: Scope, store: Store) -> Response:
    try:
        content, media_type = await store.photo(case_id, index, scope=scope)
    except CaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(
        content,
        media_type=media_type,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )
