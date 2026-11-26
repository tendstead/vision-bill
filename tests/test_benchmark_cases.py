"""Case API boundary, ordered originals and sparse human confirmation."""

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from vision_bill.api.benchmark_cases import case_store, router
from vision_bill.api.helper.helper import get_image_service
from vision_bill.model.benchmark_case import BenchmarkCaseUpdate
from vision_bill.provider.db.benchmark_cases import scored_observations
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User
from vision_bill.service.image_service import ImageService


def test_confirmation_requires_printed_observations_and_exact_fractions():
    assert scored_observations({"tax_total": "0", "tip": None, "taxes": []}) == [
        "tax_total",
        "taxes",
    ]
    assert scored_observations({"line_items": [{"quantity": "1.5"}]}) == ["line_items"]
    for invalid in (
        {},
        {"total": None},
        {"merchant_name": " "},
        {"line_items": [{}]},
        {"taxes": [{"rate": "19"}]},
        {"line_items": [{"tags": ["private"]}]},
    ):
        with pytest.raises(ValueError):
            scored_observations(invalid)
    for oversized in ({"taxes": [{}] * 501}, {"merchant_name": "x" * 200_000}):
        with pytest.raises(ValidationError):
            BenchmarkCaseUpdate(expected=oversized, expected_revision=1)


@pytest.fixture
async def case_api(settings):
    app = FastAPI()
    app.include_router(router, prefix="/cases")
    user = User(id=uuid4(), username="admin", is_admin=True)
    store = AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[case_store] = lambda: store
    images = ImageService(settings.images)
    app.dependency_overrides[get_image_service] = lambda: images
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, store, user, Path(settings.images.save_dir)


async def test_upload_rejects_invalid_names_and_counts_without_writing(case_api):
    client, store, _, directory = case_api
    photo = ("photos", ("one.png", b"invalid", "image/png"))
    for count, name in ((1, "  "), (11, "ten max"), (1, "x" * 161)):
        result = await client.post("/cases", files=[photo] * count, data={"name": name})
        assert result.status_code == 422
    store.create.assert_not_awaited()
    assert not list(directory.iterdir())


async def test_failed_later_photo_discards_earlier_original(case_api, monkeypatch):
    client, store, _, directory = case_api
    first = (
        "photos",
        (
            "../../outside.png",
            (Path(__file__).resolve().parents[1] / "e2e/llm_stub/fixtures/a.png").read_bytes(),
            "image/png",
        ),
    )
    response = await client.post(
        "/cases", files=[first, ("photos", ("bad", b"text"))], data={"name": "case"}
    )
    assert response.status_code == 415
    store.create.assert_not_awaited()
    assert not [p for p in directory.rglob("*") if p.is_file()]
    monkeypatch.setattr("vision_bill.api.benchmark_cases.APP_SOURCE_MAX_BYTES", 4)
    response = await client.post("/cases", files=[first], data={"name": "case"})
    assert response.status_code == 413
    store.create.assert_not_awaited()


async def test_non_admin_cannot_read_case_or_fetch_photo(case_api):
    client, store, user, _ = case_api
    user.is_admin = False
    for url in ("/cases", f"/cases/{uuid4()}", f"/cases/{uuid4()}/photos/0"):
        assert (await client.get(url)).status_code == 403
    assert not store.method_calls
