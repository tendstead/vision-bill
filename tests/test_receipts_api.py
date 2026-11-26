"""Endpoint wiring tests: real FastAPI app + TestClient, mocked DB and provider."""

import base64
import json
from collections.abc import Generator
from datetime import UTC, datetime
from datetime import date as Date
from datetime import time as Time
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import asyncpg
import pytest
from fastapi.testclient import TestClient

import vision_bill.main as main_module
from vision_bill.api import images as images_api_module
from vision_bill.api.system import main as system_api_module
from vision_bill.config import Settings
from vision_bill.model.receipt import LineItem, Receipt
from vision_bill.model.review import CHECKS_VERSION, AmountPolicy
from vision_bill.provider.db import image_db as image_db_module
from vision_bill.provider.db import receipt_db as receipt_db_module
from vision_bill.provider.db import user_db as user_db_module
from vision_bill.provider.db import workflow_db as workflow_db_module
from vision_bill.provider.llm.base import LLMProvider, ModelInfo
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User

# A tiny synthetic 1x1 JPEG (Pillow-generated, no PII). The upload endpoints
# validate the file type with python-magic, so the bytes must be a real JPEG,
# but the LLM is mocked here, so the actual content is irrelevant. Keeping the
# bytes in-code instead of reading tests/data/ (gitignored local fixtures)
# keeps the unit suite runnable on a clean checkout.
JPEG_BYTES = base64.b64decode(
    b"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQ"
    b"DQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQU"
    b"FBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAHwAA"
    b"AQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1Fh"
    b"ByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZn"
    b"aGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19j"
    b"Z2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAg"
    b"ECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJy"
    b"gpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpa"
    b"anqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPw"
    b"DwKiiivzI/uM//2Q=="
)
RECEIPTS_URL = "/api/v1/receipts"
SEARCH_URL = "/api/v1/search"
IMAGES_URL = "/api/v1/images"
USER_ID = UUID("00000000-0000-4000-8000-000000000001")
RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000002")
OTHER_RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000003")
IMAGE_ID = UUID("00000000-0000-4000-8000-000000000004")
OTHER_IMAGE_ID = UUID("00000000-0000-4000-8000-000000000005")


class ApiContext(NamedTuple):
    client: TestClient
    conn: AsyncMock
    provider: MagicMock


def _make_pool(conn: AsyncMock) -> MagicMock:
    conn.transaction = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    pool.close = AsyncMock()
    return pool


def _make_provider() -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.inference_identity = "ollama:" + "d" * 64
    provider.analysis_timeout_seconds = 600
    provider.check_connection = AsyncMock(return_value=True)
    provider.get_available_models = AsyncMock(return_value=[ModelInfo(id="test-model")])
    provider.analyse_receipt_from_model = AsyncMock(return_value=_make_receipt())
    return provider


def _receipt_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": RECEIPT_ID,
        "confidence": 88,
        "merchant_name": "Bauhaus",
        "merchant_address": "Main St 1",
        "receipt_number": "1001",
        "date": Date(2024, 1, 15),
        "time": Time(14, 30),
        "currency": "EUR",
        "category": "other",
        "subtotal": Decimal("19.98"),
        "discount_total": Decimal("0.00"),
        "tax_total": Decimal("3.80"),
        "tip": None,
        "total": Decimal("23.78"),
        "payment_method": "card",
        "status": "unverified",
        "image_id": None,
        "verified": False,
        "created_at": datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return base


def _image_row(id: UUID = IMAGE_ID, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": id,
        "original_filename": "upload.jpeg",
        "media_type": "image/jpeg",
        "size_bytes": 1234,
        "image_path": None,
        "status": "pending",
        "error": None,
        "receipt_id": None,
        "bypass_review": False,
        "created_at": datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        "analyzed_at": None,
    }
    base.update(overrides)
    return base


def _make_receipt() -> Receipt:
    return Receipt(
        receipt_number="R-2024-0001",
        currency="EUR",
        merchant_name="ACME Corp",
        date=Date(2024, 1, 15),
        line_items=[
            LineItem(
                description="Widget",
                quantity=2,
                unit_price=Decimal("10.00"),
                total_price=Decimal("20.00"),
            )
        ],
        subtotal=Decimal("20.00"),
        total=Decimal("20.00"),
        confidence=95,
    )


def _patch_app(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    conn: AsyncMock,
    provider: MagicMock,
    db_down: bool = False,
) -> None:
    """Patch settings, provider factory, and asyncpg used by the DB providers."""
    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(images_api_module, "settings", settings)
    monkeypatch.setattr(system_api_module, "settings", settings)
    monkeypatch.setattr(main_module, "get_llm_provider", lambda cfg: provider)
    monkeypatch.setattr(main_module, "bootstrap_admin", AsyncMock())
    monkeypatch.setattr(main_module.AnalysisScheduler, "start", AsyncMock())
    monkeypatch.setattr(main_module.BenchmarkService, "start", AsyncMock())

    fake_asyncpg = MagicMock()
    if db_down:
        fake_asyncpg.create_pool = AsyncMock(side_effect=RuntimeError("database unavailable"))
    else:
        fake_asyncpg.create_pool = AsyncMock(return_value=_make_pool(conn))

    monkeypatch.setattr(receipt_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(image_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(user_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(workflow_db_module, "asyncpg", fake_asyncpg)


def _admin_user() -> User:
    # can_see_all=True keeps these (pre-scoping) assertions on the base SQL.
    return User(id=USER_ID, username="tester", is_admin=True, can_see_all=True)


@pytest.fixture
def api_context(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Generator[ApiContext, None, None]:
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"missing": False})
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    provider = _make_provider()
    _patch_app(monkeypatch, settings, conn, provider)
    main_module.app.dependency_overrides[get_current_user] = _admin_user

    with TestClient(main_module.app) as client:
        yield ApiContext(client=client, conn=conn, provider=provider)
    main_module.app.dependency_overrides.clear()


@pytest.fixture
def broken_context(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Generator[ApiContext, None, None]:
    """DB pool unavailable -> endpoints return 503."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"missing": False})
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    provider = _make_provider()
    provider.check_connection = AsyncMock(return_value=False)
    _patch_app(monkeypatch, settings, conn, provider, db_down=True)
    main_module.app.dependency_overrides[get_current_user] = _admin_user

    with TestClient(main_module.app) as client:
        yield ApiContext(client=client, conn=conn, provider=provider)
    main_module.app.dependency_overrides.clear()


def _upload_jpeg() -> tuple[str, bytes, str]:
    return ("bauhaus.jpeg", JPEG_BYTES, "image/jpeg")


# ── Image upload (POST /images) ────────────────────────────────────────


@pytest.mark.parametrize(
    "explicit,default,expected",
    [(None, False, False), (None, True, True), (False, True, False), (True, False, True)],
)
def test_upload_stores_resolved_review_snapshot(api_context, settings, explicit, default, expected):
    settings.images.bypass_review_default = default
    ctx = api_context
    ctx.conn.fetchrow.return_value = _image_row(bypass_review=expected)
    params = {"bypass_review": explicit} if explicit is not None else {}
    response = ctx.client.post(IMAGES_URL, params=params, files={"receipt": _upload_jpeg()})
    assert response.status_code == 202
    assert response.json()["bypass_review"] is expected
    assert response.json()["receipt_id"] is None
    assert ctx.conn.fetchrow.call_args.args[8] is expected
    ctx.provider.analyse_receipt_from_model.assert_not_awaited()
    ctx.provider.get_available_models.assert_not_awaited()


def test_ui_config_exposes_bypass_review_default(
    api_context: ApiContext, settings: Settings
) -> None:
    settings.images.bypass_review_default = True

    response = api_context.client.get("/api/v1/system/ui-config")

    assert response.status_code == 200
    assert response.json() == {
        "bypass_review_default": True,
        "registration_open": True,
        "ocr_enabled": False,
        "ocr_required": False,
        "ocr_allow_vision_fallback": False,
        "formatter_max_output_tokens": settings.llm.max_output_tokens,
    }


def test_upload_image_rejects_non_image(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_receipt_row())

    response = ctx.client.post(
        IMAGES_URL,
        params={"model_id": "test-model"},
        files={"receipt": ("notes.txt", b"plain text, not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported image type" in response.json()["detail"]
    ctx.conn.fetchrow.assert_not_called()


def test_upload_image_returns_202_when_no_models(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.provider.get_available_models = AsyncMock(return_value=[])
    ctx.conn.fetchrow = AsyncMock(return_value=_image_row(id=IMAGE_ID))

    response = ctx.client.post(
        IMAGES_URL,
        params={"model_id": "test-model"},
        files={"receipt": _upload_jpeg()},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["id"] == str(IMAGE_ID)
    assert "warning" not in body
    assert response.headers["Location"] == f"{IMAGES_URL}/{IMAGE_ID}"
    ctx.provider.analyse_receipt_from_model.assert_not_awaited()


def test_upload_image_returns_202_when_provider_unreachable(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.provider.check_connection = AsyncMock(return_value=False)
    ctx.provider.get_available_models = AsyncMock(return_value=[])
    ctx.conn.fetchrow = AsyncMock(return_value=_image_row(id=IMAGE_ID))

    response = ctx.client.post(
        IMAGES_URL,
        params={"model_id": "test-model"},
        files={"receipt": _upload_jpeg()},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["id"] == str(IMAGE_ID)
    assert "warning" not in body
    ctx.provider.get_available_models.assert_not_awaited()
    ctx.provider.analyse_receipt_from_model.assert_not_awaited()


def test_upload_cannot_freeze_a_stale_dedicated_ocr_topology(api_context, monkeypatch, tmp_path):
    from ocr_contract.activation import (
        ActivationPhase,
        ActivationStatus,
        ActivationStore,
        MaintenanceIntent,
    )

    from vision_bill.config import settings
    from vision_bill.service import activation_admission

    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    monkeypatch.setattr(activation_admission, "ActivationStore", lambda: store)
    monkeypatch.setenv("VISION_BILL_OCR_ACTIVATION", "true")
    monkeypatch.setattr(settings.ocr, "enabled", True)
    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    store.set_shared_gpu(False)
    assert store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_image_row(id=IMAGE_ID))
    response = ctx.client.post(IMAGES_URL, files={"receipt": _upload_jpeg()})
    assert response.status_code == 409 and response.json()["detail"] == "execution_policy_changed"
    ctx.conn.fetchrow.assert_not_called()


# ── Image collection (GET /images) ─────────────────────────────────────


def test_list_images(api_context: ApiContext) -> None:
    """GET /images lists every image row (newest first)."""
    ctx = api_context
    ctx.conn.fetch = AsyncMock(
        return_value=[
            _image_row(id=OTHER_IMAGE_ID, status="analyzed"),
            _image_row(id=IMAGE_ID, status="pending"),
        ]
    )

    response = ctx.client.get(IMAGES_URL)

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body] == [str(OTHER_IMAGE_ID), str(IMAGE_ID)]


def test_list_images_filtered(api_context: ApiContext) -> None:
    """GET /images?status=pending,failed is the queued-image view."""
    ctx = api_context
    ctx.conn.fetch = AsyncMock(
        return_value=[
            _image_row(id=IMAGE_ID, status="pending"),
            _image_row(id=OTHER_IMAGE_ID, status="failed"),
        ]
    )

    response = ctx.client.get(IMAGES_URL, params={"status": "pending,failed"})

    assert response.status_code == 200
    body = response.json()
    assert [(item["id"], item["status"]) for item in body] == [
        (str(IMAGE_ID), "pending"),
        (str(OTHER_IMAGE_ID), "failed"),
    ]


def test_get_image_by_id(api_context: ApiContext) -> None:
    """GET /images/{id} returns a single image row."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(
        return_value=_image_row(id=IMAGE_ID, status="analyzed", receipt_id=RECEIPT_ID)
    )

    response = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(IMAGE_ID)
    assert body["status"] == "analyzed"
    assert body["receipt_id"] == str(RECEIPT_ID)


def test_get_image_by_id_not_found(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=None)

    response = ctx.client.get(f"{IMAGES_URL}/{OTHER_IMAGE_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


# ── Image file (GET /images/{id}/file) ─────────────────────────────────


def test_get_image_file_streams_bytes(api_context: ApiContext, tmp_path: Path) -> None:
    """GET /images/{id}/file streams the stored file with its media type."""
    ctx = api_context
    image_file = tmp_path / "stored.png"
    image_file.write_bytes(b"image-bytes")
    ctx.conn.fetchrow = AsyncMock(
        return_value=_image_row(
            id=IMAGE_ID, status="analyzed", image_path=str(image_file), media_type="image/png"
        )
    )

    response = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file")

    assert response.status_code == 200
    assert response.content == b"image-bytes"
    assert response.headers["content-type"] == "image/png"


def test_get_image_file_not_found_when_row_missing(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=None)

    response = ctx.client.get(f"{IMAGES_URL}/{OTHER_IMAGE_ID}/file")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image file not found"


def test_get_image_file_not_found_when_path_missing(api_context: ApiContext) -> None:
    """Row exists but carries no stored path -> 404."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_image_row(id=IMAGE_ID, image_path=None))

    response = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image file not found"


def test_get_image_file_not_found_when_file_missing(api_context: ApiContext) -> None:
    """Row points at a path that no longer exists on disk -> 404."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(
        return_value=_image_row(id=IMAGE_ID, image_path="/nonexistent/receipt.png")
    )

    response = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image file not found"


# ── Image thumbnail (GET /images/{id}/thumb) ──────────────────────────


def test_get_image_thumb_streams_webp(api_context: ApiContext, tmp_path: Path) -> None:
    ctx = api_context
    thumb_file = tmp_path / "stored.thumb.webp"
    thumb_file.write_bytes(b"thumb-bytes")
    ctx.conn.fetchrow = AsyncMock(
        return_value=_image_row(id=IMAGE_ID, status="analyzed", thumbnail_path=str(thumb_file))
    )

    response = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}/thumb")

    assert response.status_code == 200
    assert response.content == b"thumb-bytes"
    assert response.headers["content-type"] == "image/webp"


def test_get_image_thumb_404_when_no_thumb(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(
        return_value=_image_row(id=IMAGE_ID, status="analyzed", thumbnail_path=None)
    )

    response = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}/thumb")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image thumbnail not found"


def test_get_image_thumb_404_when_file_missing(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(
        return_value=_image_row(id=IMAGE_ID, thumbnail_path="/nonexistent/x.thumb.webp")
    )

    response = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}/thumb")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image thumbnail not found"


def test_get_image_thumb_503_when_db_down(broken_context: ApiContext) -> None:
    response = broken_context.client.get(f"{IMAGES_URL}/{IMAGE_ID}/thumb")
    assert response.status_code == 503


# ── Image delete (DELETE /images/{id}) ─────────────────────────────────


def test_delete_image(api_context: ApiContext, settings: Settings) -> None:
    """DELETE /images/{id} removes the row and the on-disk file."""
    ctx = api_context
    tmp_file = Path(settings.images.tmp_dir) / "x.png"
    tmp_file.write_bytes(b"queued-image-bytes")
    async def rows(query, *args):
        if "SELECT files FROM resource_deletions" in query:
            insert = next(call for call in ctx.conn.execute.call_args_list if "INSERT INTO resource_deletions" in call.args[0])
            return {"files": insert.args[4]}
        if "resource_deletions" in query:
            return None
        return _image_row(id=IMAGE_ID, user_id=USER_ID, status="pending", image_path=str(tmp_file))
    ctx.conn.fetchrow = AsyncMock(side_effect=rows)
    ctx.conn.execute = AsyncMock()

    response = ctx.client.delete(f"{IMAGES_URL}/{IMAGE_ID}")

    assert response.status_code == 200
    assert response.json()["deleted"] == str(IMAGE_ID)
    assert any("DELETE FROM images" in call.args[0] for call in ctx.conn.execute.call_args_list)
    assert not tmp_file.exists()


def test_delete_analyzed_image_conflict(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(side_effect=[None, _image_row(id=IMAGE_ID, status="analyzed")])

    response = ctx.client.delete(f"{IMAGES_URL}/{IMAGE_ID}")

    assert response.status_code == 409
    ctx.conn.execute.assert_not_awaited()


def test_delete_image_not_found(api_context: ApiContext) -> None:
    """DELETE /images/{id} -> 404 when no such image row exists."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=None)

    response = ctx.client.delete(f"{IMAGES_URL}/{OTHER_IMAGE_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"
    ctx.conn.execute.assert_not_awaited()


# ── Manual analysis trigger (POST /images/analyze) ─────────────────────


def test_analyze_images_manual_trigger(api_context: ApiContext) -> None:
    """POST /images/analyze runs one scheduler cycle; down provider -> no results."""
    ctx = api_context
    ctx.provider.get_available_models = AsyncMock(side_effect=RuntimeError("down"))

    response = ctx.client.post(f"{IMAGES_URL}/analyze")

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_analyze_images_success(api_context: ApiContext, tmp_path: Path) -> None:
    """The manual route runs the same fenced worker as the scheduled path."""
    from types import SimpleNamespace

    from vision_bill.model.db.image import ImageRow
    from vision_bill.model.workflow import CommitResult, SubmissionStatus

    ctx = api_context
    queued_file = tmp_path / "queued.png"
    queued_file.write_bytes(b"queued-image-bytes")
    image = ImageRow(id=IMAGE_ID, image_path=str(queued_file), user_id=USER_ID)
    claim = SimpleNamespace(image=image)
    service = ctx.client.app.state.submission_service
    service.db.due_candidates = AsyncMock(return_value=[image])
    service.db.claim_next = AsyncMock(return_value=claim)
    service.ready_for = AsyncMock(return_value=True)
    service.process_claim = AsyncMock(
        return_value=CommitResult(
            stale=False,
            submission=SubmissionStatus(id=IMAGE_ID, status="analyzed", receipt_id=RECEIPT_ID),
        )
    )
    response = ctx.client.post(f"{IMAGES_URL}/analyze")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["image_id"] == str(IMAGE_ID)
    assert results[0]["status"] == "analyzed"
    assert results[0]["receipt_id"] == str(RECEIPT_ID)
    service.db.claim_next.assert_awaited_once_with(image_id=IMAGE_ID, lease_seconds=900)
    service.process_claim.assert_awaited_once_with(claim)
    assert queued_file.read_bytes() == b"queued-image-bytes"


# ── Tags (GET /tags, POST /tags) ──────────────────────────────────────

TAGS_URL = "/api/v1/tags"


def test_list_tags_endpoint(api_context: ApiContext) -> None:
    """GET /tags returns the vocabulary, ordered by name."""
    ctx = api_context
    ctx.conn.fetch = AsyncMock(return_value=[{"name": "coffee"}, {"name": "food"}])

    response = ctx.client.get(TAGS_URL)

    assert response.status_code == 200
    assert response.json() == ["coffee", "food"]


def test_spending_categories_require_auth(api_context: ApiContext) -> None:
    response = api_context.client.get(f"{TAGS_URL}/categories")
    assert response.status_code == 200
    assert {"code": "milk"} in response.json()
    main_module.app.state.receipt_service.user_db._pool = _make_pool(api_context.conn)
    main_module.app.dependency_overrides.clear()
    assert api_context.client.get(f"{TAGS_URL}/categories").status_code == 401


def test_category_suggestion_scopes_exact_history_to_user(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchval = AsyncMock(return_value=None)
    ctx.conn.fetchrow = AsyncMock(return_value={"category": "milk"})
    response = ctx.client.get(
        f"{TAGS_URL}/category-suggestion",
        params={"merchant_name": "  Rewe  ", "description": " H-MILCH   1,5% "},
    )
    assert response.status_code == 200
    assert response.json() == {"category": "milk", "source": "history"}
    sql, user, merchant, description = ctx.conn.fetchrow.await_args.args
    assert "item_category_aliases" in sql
    assert user == USER_ID
    assert (merchant, description) == ("rewe", "h-milch 1,5%")

    ctx.conn.fetchrow = AsyncMock(return_value=None)
    missing = ctx.client.get(
        f"{TAGS_URL}/category-suggestion",
        params={"merchant_name": "Rewe", "description": "H-MILCH 3,5%"},
    )
    assert missing.json() == {"category": None, "source": None}


def test_create_tag_endpoint_new(api_context: ApiContext) -> None:
    """POST /tags creates a tag (201) and returns the normalized name."""
    ctx = api_context
    # create_tag issues INSERT ... RETURNING name via fetchrow.
    ctx.conn.fetchrow = AsyncMock(return_value={"name": "hot drink"})

    response = ctx.client.post(
        TAGS_URL, json={"name": "  Hot   Drink "}, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 201
    assert response.json() == {"name": "hot drink", "created": True}
    insert_call = ctx.conn.fetchrow.await_args
    assert insert_call is not None
    assert insert_call.args[0] == receipt_db_module.INSERT_TAG_SQL
    assert insert_call.args[1] == "hot drink"


def test_create_tag_endpoint_existing_is_idempotent(api_context: ApiContext) -> None:
    """POST /tags preserves the spelling of a known tag and returns 200."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=None)

    ctx.conn.fetchval = AsyncMock(return_value="Coffee")
    response = ctx.client.post(TAGS_URL, json={"name": "coffee"})

    assert response.status_code == 200
    assert response.json() == {"name": "Coffee", "created": False}


def test_create_tag_endpoint_blank_rejected(api_context: ApiContext) -> None:
    """A blank name is a 422 and never reaches the DB."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock()

    response = ctx.client.post(TAGS_URL, json={"name": "   "})

    assert response.status_code == 422
    assert "must not be blank" in response.json()["detail"]
    ctx.conn.fetchrow.assert_not_awaited()


def test_update_receipt_accepts_suggested_tags(api_context: ApiContext) -> None:
    """PUT /receipts/{id} stores tags that are not yet in the vocabulary.

    LLM-suggested tags must persist (the UI marks them for review); only the
    model normalizes them, it does not reject them.
    """
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_receipt_row(id=RECEIPT_ID, merchant_name="ACME"))
    ctx.conn.execute = AsyncMock()

    body = {
        "confidence": 95,
        "merchant_name": "ACME",
        "date": "2024-01-15",
        "line_items": [
            {"description": "Widget", "quantity": 1, "unit_price": "10.00", "total_price": "10.00"}
        ],
        "subtotal": "10.00",
        "total": "10.00",
    }
    body["expected_operation_revision"] = 1
    body["line_items"][0]["tags"] = ["brunch", "  Brunch  ", ""]

    response = ctx.client.put(
        f"{RECEIPTS_URL}/{RECEIPT_ID}",
        json=body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    insert_calls = [
        call
        for call in ctx.conn.execute.call_args_list
        if call.args and call.args[0] == receipt_db_module.INSERT_LINE_ITEM_SQL
    ]
    assert len(insert_calls) == 1
    # The tags array remains normalized and deduped.
    assert insert_calls[0].args[6] == ["brunch"]


@pytest.mark.parametrize("discount", [{}, {"discount_total": None}, {"discount_total": "0"}])
def test_update_receipt_distinguishes_unobserved_discount_from_printed_zero(api_context, discount):
    ctx = api_context
    current = _receipt_row(discount_total=None)
    expected = Decimal(0) if discount.get("discount_total") == "0" else None

    def rows(query, *args):
        if query.startswith(receipt_db_module.UPDATE_RECEIPT_SQL):
            current["discount_total"] = args[9]
        return current

    ctx.conn.fetchrow = AsyncMock(side_effect=rows)
    body = _make_receipt().model_dump(mode="json", exclude={"discount_total"})
    body.update(discount, expected_operation_revision=1)

    response = ctx.client.put(f"{RECEIPTS_URL}/{RECEIPT_ID}", json=body)

    assert response.status_code == 200
    assert current["discount_total"] == expected
    assert response.json()["discount_total"] == discount.get("discount_total")


@pytest.mark.parametrize("invalid", [["milk"], {"code": "milk"}, "fake_code"])
def test_update_rejects_invalid_spending_category(api_context: ApiContext, invalid: object) -> None:
    body = {
        "confidence": 95,
        "merchant_name": "ACME",
        "date": "2024-01-15",
        "line_items": [
            {
                "description": "Widget",
                "quantity": 1,
                "unit_price": "10.00",
                "total_price": "10.00",
                "spending_category": invalid,
            }
        ],
        "subtotal": "10.00",
        "total": "10.00",
    }
    response = api_context.client.put(f"{RECEIPTS_URL}/{RECEIPT_ID}", json=body)
    assert response.status_code == 422


# ── Receipts (GET /receipts, GET /receipts/{id}, verify) ───────────────


def test_get_receipt_not_found(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=None)

    response = ctx.client.get(f"{RECEIPTS_URL}/{OTHER_RECEIPT_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Receipt not found"


def test_list_receipts(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetch = AsyncMock(
        return_value=[
            _receipt_row(id=OTHER_RECEIPT_ID, merchant_name="Store B"),
            _receipt_row(id=RECEIPT_ID, merchant_name="Store A"),
        ]
    )

    response = ctx.client.get(RECEIPTS_URL, params={"limit": 10, "offset": 0})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["merchant_name"] == "Store B"
    assert body[1]["merchant_name"] == "Store A"


def test_search_products_returns_verified_purchase_history(api_context: ApiContext) -> None:
    """GET /search searches line-item descriptions and returns price summary data."""
    ctx = api_context
    ctx.conn.fetch = AsyncMock(
        return_value=[
            {
                "receipt_id": RECEIPT_ID,
                "description": "Gouda Mittelalt",
                "quantity": Decimal("1.0000"),
                "unit_price": Decimal("3.20"),
                "merchant_name": "Store A",
                "date": Date(2024, 3, 1),
                "time": Time(18, 0),
                "currency": "EUR",
            },
            {
                "receipt_id": OTHER_RECEIPT_ID,
                "description": "Gouda",
                "quantity": Decimal("1.0000"),
                "unit_price": Decimal("2.80"),
                "merchant_name": "Store B",
                "date": Date(2024, 2, 1),
                "time": Time(12, 0),
                "currency": "EUR",
            },
        ]
    )

    response = ctx.client.get(SEARCH_URL, params={"query": "gouda"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "gouda"
    assert body["latest_price"] == "3.20"
    assert body["cheapest_price"] == "2.80"
    assert body["average_price"] == "3.00"
    assert body["currency"] == "EUR"
    assert body["purchases"][0]["description"] == "Gouda Mittelalt"
    assert body["purchases"][0]["merchant_name"] == "Store A"
    assert body["purchases"][0]["date"] == "2024-03-01"
    assert body["purchases"][0]["unit_price"] == "3.20"
    ctx.provider.analyse_receipt_from_model.assert_not_awaited()


def test_list_receipts_without_filters_keeps_base_sql(api_context: ApiContext) -> None:
    """No filters -> plain paged query, no WHERE clause."""
    ctx = api_context
    ctx.conn.fetch = AsyncMock(return_value=[])

    response = ctx.client.get(RECEIPTS_URL)

    assert response.status_code == 200
    fetch_call = ctx.conn.fetch.await_args
    assert fetch_call is not None
    sql = fetch_call.args[0]
    # Base SQL now includes the thumbnail_path correlated subquery;
    # no filter WHERE clause should be appended when no filters are given.
    expected = (
        "SELECT r.*, (SELECT i.thumbnail_path FROM images i WHERE i.id = r.image_id) "
        "AS thumbnail_path FROM receipts r ORDER BY r.date DESC, r.id LIMIT $1 OFFSET $2"
    )
    assert sql == expected
    assert fetch_call.args[1] == 50
    assert fetch_call.args[2] == 0


def test_list_receipts_passes_filters_to_db(api_context: ApiContext) -> None:
    """Status, date range and search are forwarded to the DB query."""
    ctx = api_context
    ctx.conn.fetch = AsyncMock(return_value=[])

    response = ctx.client.get(
        RECEIPTS_URL,
        params={
            "status": "verified",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "search": "acme",
            "limit": 10,
            "offset": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == []

    fetch_call = ctx.conn.fetch.await_args
    assert fetch_call is not None
    sql = fetch_call.args[0]
    assert "status = ANY($1)" in sql
    assert "date >= $2" in sql
    assert "date <= $3" in sql
    assert "merchant_name ILIKE $4" in sql
    assert "receipt_number ILIKE $4" in sql
    assert "ORDER BY r.date DESC, r.id LIMIT $5 OFFSET $6" in sql
    args = fetch_call.args[1:]
    assert args[0] == ["verified"]
    assert args[1] == Date(2024, 1, 1)
    assert args[2] == Date(2024, 12, 31)
    assert args[3] == "%acme%"
    assert args[4] == 10
    assert args[5] == 5


def test_verify_receipt_preserves_original_path(api_context, settings):
    """Review changes receipt metadata; original/legacy paths stay immutable."""
    ctx = api_context
    original = Path(settings.images.tmp_dir) / "legacy.jpeg"
    original.write_bytes(JPEG_BYTES)
    from vision_bill.model.db.image import ImageRow
    from vision_bill.model.db.receipt import ReceiptRow

    service = ctx.client.app.state.receipt_service
    service.get_receipt_by_id = AsyncMock(
        return_value=ReceiptRow(
            **_receipt_row(image_id=IMAGE_ID, time=None, created_at=Date(2024, 1, 15))
        )
    )
    service.get_image_by_id = AsyncMock(
        return_value=ImageRow(**_image_row(image_path=str(original)))
    )
    service.verify_receipt = AsyncMock(
        return_value=ReceiptRow(
            **_receipt_row(
                status="verified", verified=True, time=None, created_at=Date(2024, 1, 15)
            )
        )
    )
    move = MagicMock(side_effect=AssertionError("verification moved original"))
    ctx.client.app.state.image_service.store_perm_image = move
    response = ctx.client.post(f"{RECEIPTS_URL}/{RECEIPT_ID}/verify")
    assert response.status_code == 200
    assert response.json()["verified"] is True
    assert original.read_bytes() == JPEG_BYTES
    move.assert_not_called()


def test_verify_receipt_already_verified_conflict(api_context: ApiContext) -> None:
    """An already-human review returns 409 on duplicate verification."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(
        return_value=_receipt_row(
            id=RECEIPT_ID,
            image_id=IMAGE_ID,
            status="verified",
            verified=True,
            verification_source="human",
            checks_version=CHECKS_VERSION,
        )
    )

    response = ctx.client.post(f"{RECEIPTS_URL}/{RECEIPT_ID}/verify")

    assert response.status_code == 409
    assert response.json()["detail"] == "Receipt already verified"
    ctx.conn.execute.assert_not_awaited()


def test_human_review_under_old_checks_requires_current_findings_and_revisions(api_context, tmp_path):
    ctx = api_context
    original = tmp_path / "original.jpeg"
    original.write_bytes(JPEG_BYTES)
    policy = AmountPolicy(tax_basis="gross", subtotal_origin="printed", tax_information="explicit_zero")
    old_ack = {"finding_id": "old-checks-finding", "comment": "Printed total checked"}
    current = _receipt_row(
        image_id=IMAGE_ID, status="verified", verified=True, verification_source="human",
        checks_version="2", subtotal=Decimal(0), tax_total=Decimal(0), total=Decimal(1),
        payment_method="cash", amount_policy=policy.model_dump(), acknowledgements=[old_ack],
    )
    image = _image_row(
        image_path=str(original), source_manifest=[], requested_options={}, effective_options={},
        request_fingerprint=None, idempotency_key=None, ocr_checkpoint=None, attempt_count=0,
        generation=1,
    )

    def rows(query, *args):
        if "FROM images" in query:
            return image
        if "UPDATE receipts SET status='verified'" in query:
            current.update(findings=json.loads(args[1]), checks_version=args[2],
                           acknowledgements=json.loads(args[3]), operation_revision=2)
        return current

    ctx.conn.fetchrow = AsyncMock(side_effect=rows)
    ctx.conn.fetchval = AsyncMock(return_value=False)
    receipt_url = f"{RECEIPTS_URL}/{RECEIPT_ID}"
    verify_url = f"{receipt_url}/verify"

    assert ctx.client.get(receipt_url).json()["can_rereview"] is True
    assert ctx.client.post(verify_url).json()["detail"] == "review_revision_required"
    response = ctx.client.get(verify_url)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert [finding["code"] for finding in preview["findings"]] == ["amount_discrepancy"]
    revisions = {"expected_operation_revision": preview["operation_revision"],
                 "expected_review_revision": preview["review_revision"]}
    stale = ctx.client.post(verify_url, json={**revisions, "acknowledgements": [old_ack]})
    assert stale.status_code == 422 and stale.json()["detail"] == "review_acknowledgements_invalid"
    accepted = ctx.client.post(verify_url, json={**revisions, "acknowledgements": [
        {**old_ack, "finding_id": preview["findings"][0]["id"]},
    ]})
    assert accepted.status_code == 200 and accepted.json()["checks_version"] == CHECKS_VERSION
    assert ctx.client.get(receipt_url).json()["can_rereview"] is False
    assert ctx.client.get(verify_url).status_code == 409


def test_bypass_rereview_requires_explicit_revisions(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(
        return_value=_receipt_row(
            image_id=IMAGE_ID,
            status="verified",
            verified=True,
            verification_source="automatic_bypass",
        )
    )

    response = ctx.client.post(f"{RECEIPTS_URL}/{RECEIPT_ID}/verify")

    assert response.status_code == 409
    assert response.json()["detail"] == "review_revision_required"
    ctx.conn.execute.assert_not_awaited()


def test_endpoints_return_503_when_db_not_ready(broken_context: ApiContext) -> None:
    ctx = broken_context

    upload = ctx.client.post(
        IMAGES_URL,
        params={"model_id": "test-model"},
        files={"receipt": _upload_jpeg()},
    )
    receipt_listing = ctx.client.get(RECEIPTS_URL)
    image_listing = ctx.client.get(IMAGES_URL)
    detail = ctx.client.get(f"{RECEIPTS_URL}/{OTHER_RECEIPT_ID}")
    image_file = ctx.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file")
    tag_listing = ctx.client.get(TAGS_URL)
    tag_create = ctx.client.post(TAGS_URL, json={"name": "coffee"})

    assert upload.status_code == 503
    assert receipt_listing.status_code == 503
    assert image_listing.status_code == 503
    assert detail.status_code == 503
    assert image_file.status_code == 503
    assert tag_listing.status_code == 503
    assert tag_create.status_code == 503
    assert "Database not available" in upload.json()["detail"]


# ── Receipt delete (DELETE /receipts/{id}) ────────────────────────────


def test_delete_receipt(api_context: ApiContext, settings: Settings) -> None:
    """DELETE /receipts/{id} removes the row, its image row and the on-disk file."""
    ctx = api_context
    tmp_file = Path(settings.images.tmp_dir) / "img.png"
    tmp_file.write_bytes(b"receipt-image-bytes")
    async def rows(query, *args):
        if "SELECT files FROM resource_deletions" in query:
            insert = next(call for call in ctx.conn.execute.call_args_list if "INSERT INTO resource_deletions" in call.args[0])
            return {"files": insert.args[4]}
        if "resource_deletions" in query:
            return None
        if "FROM receipts" in query:
            return _receipt_row(id=RECEIPT_ID, user_id=USER_ID, image_id=IMAGE_ID, merchant_name="Doomed Store")
        return _image_row(id=IMAGE_ID, status="analyzed", receipt_id=RECEIPT_ID, image_path=str(tmp_file))
    ctx.conn.fetchrow = AsyncMock(side_effect=rows)
    ctx.conn.execute = AsyncMock()

    response = ctx.client.delete(f"{RECEIPTS_URL}/{RECEIPT_ID}")

    assert response.status_code == 200
    assert response.json()["deleted"] == str(RECEIPT_ID)
    executed = [call.args[0] for call in ctx.conn.execute.call_args_list]
    assert any("DELETE FROM images" in query for query in executed)
    assert not tmp_file.exists()


def test_delete_receipt_not_found(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=None)

    response = ctx.client.delete(f"{RECEIPTS_URL}/{OTHER_RECEIPT_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Receipt not found"
    ctx.conn.execute.assert_not_awaited()


def test_delete_receipt_referenced_conflict(api_context: ApiContext) -> None:
    """A receipt still referenced by a benchmark run cannot be deleted -> 409."""
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(side_effect=asyncpg.ForeignKeyViolationError("fk"))

    response = ctx.client.delete(f"{RECEIPTS_URL}/{RECEIPT_ID}")

    assert response.status_code == 409
    assert "benchmark" in response.json()["detail"]
    ctx.conn.execute.assert_not_awaited()


def test_update_receipt_sets_collections(api_context: ApiContext) -> None:
    """PUT /receipts/{id} with collection_ids reconciles the join rows exactly once."""
    coll_svc = MagicMock()
    coll_svc.db_ready = True
    coll_svc.set_receipt_collections = AsyncMock()
    main_module.app.state.collection_service = coll_svc

    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_receipt_row(id=RECEIPT_ID))

    ctx.conn.fetch = AsyncMock(side_effect=lambda sql, *args: [{"id": UUID("00000000-0000-4000-8000-0000000000c1")}] if "SELECT id FROM collections" in sql else [])

    body = {
        "expected_operation_revision": 1,
        "confidence": 90,
        "merchant_name": "M",
        "date": "2026-06-05",
        "time": None,
        "currency": "EUR",
        "category": "other",
        "line_items": [],
        "subtotal": "10",
        "discount_total": "0",
        "tax_total": "0",
        "total": "10",
        "payment_method": "unknown",
        "collection_ids": ["00000000-0000-4000-8000-0000000000c1"],
    }
    r = ctx.client.put(f"{RECEIPTS_URL}/{RECEIPT_ID}", json=body)
    assert r.status_code == 200
    calls = [c for c in ctx.conn.execute.call_args_list if "INSERT INTO receipt_collections" in c.args[0]]
    assert len(calls) == 1 and calls[0].args[1:] == (RECEIPT_ID, UUID("00000000-0000-4000-8000-0000000000c1"))
    coll_svc.set_receipt_collections.assert_not_awaited()


def test_update_receipt_empty_collection_ids_clears_all(api_context: ApiContext) -> None:
    """PUT /receipts/{id} with an explicit empty collection_ids list clears all links."""
    coll_svc = MagicMock()
    coll_svc.db_ready = True
    coll_svc.set_receipt_collections = AsyncMock()
    main_module.app.state.collection_service = coll_svc

    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_receipt_row(id=RECEIPT_ID))

    body = {
        "expected_operation_revision": 1,
        "confidence": 90,
        "merchant_name": "M",
        "date": "2026-06-05",
        "time": None,
        "currency": "EUR",
        "category": "other",
        "line_items": [],
        "subtotal": "10",
        "discount_total": "0",
        "tax_total": "0",
        "total": "10",
        "payment_method": "unknown",
        "collection_ids": [],
    }
    r = ctx.client.put(f"{RECEIPTS_URL}/{RECEIPT_ID}", json=body)
    assert r.status_code == 200
    assert any("DELETE FROM receipt_collections" in c.args[0] for c in ctx.conn.execute.call_args_list)
    assert not any("INSERT INTO receipt_collections" in c.args[0] for c in ctx.conn.execute.call_args_list)
    coll_svc.set_receipt_collections.assert_not_awaited()


def test_update_receipt_omitted_collection_ids_noop(api_context: ApiContext) -> None:
    """PUT /receipts/{id} without collection_ids leaves the membership untouched."""
    coll_svc = MagicMock()
    coll_svc.db_ready = True
    coll_svc.set_receipt_collections = AsyncMock()
    main_module.app.state.collection_service = coll_svc

    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_receipt_row(id=RECEIPT_ID))

    body = {
        "expected_operation_revision": 1,
        "confidence": 90,
        "merchant_name": "M",
        "date": "2026-06-05",
        "time": None,
        "currency": "EUR",
        "category": "other",
        "line_items": [],
        "subtotal": "10",
        "discount_total": "0",
        "tax_total": "0",
        "total": "10",
        "payment_method": "unknown",
    }
    r = ctx.client.put(f"{RECEIPTS_URL}/{RECEIPT_ID}", json=body)
    assert r.status_code == 200
    coll_svc.set_receipt_collections.assert_not_awaited()


@pytest.mark.parametrize(
    "viewer,admin,see_all,expected",
    [
        ("first", False, False, ["first"]),
        ("second", False, False, ["second"]),
        ("admin", True, False, []),
        ("admin", True, True, ["first", "second"]),
    ],
)
def test_manual_analysis_results_follow_owner_and_admin_scope(
    api_context, viewer, admin, see_all, expected
):
    from types import SimpleNamespace
    from uuid import uuid4

    from vision_bill.model.db.image import ImageRow
    from vision_bill.model.workflow import CommitResult, SubmissionStatus

    owners = {name: uuid4() for name in ["first", "second", "admin"]}
    images = {name: ImageRow(id=uuid4(), user_id=owners[name]) for name in ["first", "second"]}
    receipts = {name: uuid4() for name in images}
    service = api_context.client.app.state.submission_service
    service.db.due_candidates = AsyncMock(return_value=list(images.values()))
    service.ready_for = AsyncMock(return_value=True)

    async def claim(*, image_id, lease_seconds):
        return SimpleNamespace(image=next(i for i in images.values() if i.id == image_id))

    async def process(claim):
        name = next(name for name, image in images.items() if image.id == claim.image.id)
        return CommitResult(
            stale=False,
            submission=SubmissionStatus(
                id=claim.image.id,
                user_id=claim.image.user_id,
                status="analyzed",
                receipt_id=receipts[name],
                result_code=f"{name}_code",
            ),
        )

    service.db.claim_next = AsyncMock(side_effect=claim)
    service.process_claim = AsyncMock(side_effect=process)
    user = User(id=owners[viewer], username=viewer, is_admin=admin, can_see_all=see_all)
    api_context.client.app.dependency_overrides[get_current_user] = lambda: user
    response = api_context.client.post(f"{IMAGES_URL}/analyze")
    assert response.status_code == 200
    results = response.json()["results"]
    assert {r["image_id"] for r in results} == {str(images[name].id) for name in expected}
    for name in set(images) - set(expected):
        assert str(receipts[name]) not in response.text
        assert f"{name}_code" not in response.text


@pytest.mark.parametrize("error,status", [("operation_revision_conflict",409), ("review_acknowledgements_invalid",422), ("receipt_integrity_failed",422)])
def test_locked_review_gate_errors_are_http_responses(api_context, error, status):
    service = api_context.client.app.state.receipt_service
    service.get_receipt_by_id = AsyncMock(return_value=__import__("types").SimpleNamespace(status="unverified"))
    service.verify_receipt = AsyncMock(side_effect=ValueError(error))
    response = api_context.client.post(f"{RECEIPTS_URL}/{RECEIPT_ID}/verify", json={"expected_operation_revision":2,"expected_review_revision":2,"acknowledgements":[]})
    assert response.status_code == status
    assert response.json()["detail"] == error


def test_put_decodes_unquoted_decimal_observations_exactly(api_context: ApiContext) -> None:
    ctx = api_context
    ctx.conn.fetchrow = AsyncMock(return_value=_receipt_row(id=RECEIPT_ID))
    response = ctx.client.put(
        f"{RECEIPTS_URL}/{RECEIPT_ID}",
        headers={"Content-Type": "application/json"},
        content="""{"confidence": 95, "merchant_name": "Shop", "date": "2024-01-15",
            "line_items": [{"description": "Item", "quantity": 1.0000000000000001,
                            "unit_price": 10, "total_price": 10}],
            "taxes": [{"name": "VAT", "rate": 0.20000000000000001, "amount": 0}],
            "subtotal": 10, "total": 10, "expected_operation_revision": 1}""",
    )
    assert response.status_code == 200, response.text
    writes = {call.args[0]: call.args for call in ctx.conn.execute.call_args_list}
    assert writes[receipt_db_module.INSERT_LINE_ITEM_SQL][3] == Decimal("1.0000000000000001")
    assert writes[receipt_db_module.INSERT_TAX_SQL][3] == Decimal("0.20000000000000001")


def test_current_formatter_model_listing_hides_upstream_authentication_error(api_context: ApiContext) -> None:
    from ollama import ResponseError

    api_context.provider.get_available_models.side_effect = ResponseError("private upstream response", 401)
    response = api_context.client.get("/api/v1/llm/models?capability=text")
    assert response.status_code == 503
    assert response.json() == {"detail": "Formatter authentication failed"}
    api_context.provider.get_available_models.assert_any_await(capability="text")
