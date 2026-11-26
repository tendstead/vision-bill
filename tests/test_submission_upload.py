"""Durable upload acceptance: real files, route, and service; isolated queue boundary."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from test_receipts_api import JPEG_BYTES

from vision_bill.api import images
from vision_bill.api.helper.helper import (
    get_analysis_scheduler,
    get_image_service,
    get_receipt_service,
)
from vision_bill.model.db.image import ImageRow
from vision_bill.model.workflow import EnqueueResult, SubmissionStatus
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User
from vision_bill.service.image_service import ImageService
from vision_bill.service.receipt_service import ReceiptService


class Queue:
    is_ready = True

    def __init__(self):
        self.rows = {}
        self.keys = {}
        self.creates = []
        self.lock = asyncio.Lock()
        self.lose_response = False

    async def enqueue_submission(self, submission, *, scope, idempotency_key):
        async with self.lock:
            key = (scope.user_id, idempotency_key)
            if idempotency_key and key in self.keys:
                old = self.creates[self.keys[key]]
                if old.request_fingerprint != submission.request_fingerprint:
                    raise ValueError("upload idempotency payload conflict")
                return EnqueueResult(submission=self.rows[old.id], created=False)
            if idempotency_key:
                self.keys[key] = len(self.creates)
            self.creates.append(submission)
            self.rows[submission.id] = SubmissionStatus(
                id=submission.id,
                user_id=scope.user_id,
                image_path=submission.parts[0].image_path,
                additional_images=submission.parts[1:],
                model_id=submission.model_id,
                bypass_review=submission.bypass_review,
            )
            if self.lose_response:
                self.lose_response = False
                raise ConnectionError("commit response lost")
            return EnqueueResult(submission=self.rows[submission.id], created=True)

    async def get_submission(self, image_id, *, scope):
        return self.rows.get(image_id)


@pytest.fixture
def upload_context(settings, monkeypatch):
    monkeypatch.setattr(images, "settings", settings)
    provider = MagicMock()
    provider.inference_identity = "ollama:" + "d" * 64
    provider.check_connection = AsyncMock(side_effect=AssertionError("upload discovered models"))
    provider.get_available_models = AsyncMock(
        side_effect=AssertionError("upload discovered models")
    )
    provider.analyse_receipt_from_model = AsyncMock(side_effect=AssertionError("upload inferred"))
    service = ReceiptService(settings.images, settings.pg, provider)
    service._db._pool = MagicMock()
    service._image_db.store_image = AsyncMock(return_value=ImageRow(id=uuid4()))
    queue = Queue()
    service._workflow_db = queue
    image_service = ImageService(settings.images)
    app = FastAPI()
    app.include_router(images.router, prefix="/api/v1/images")
    # Stable owner across requests.
    owner = User(id=uuid4(), username="owner")
    app.dependency_overrides[get_current_user] = lambda: owner
    app.dependency_overrides[get_receipt_service] = lambda: service
    app.dependency_overrides[get_image_service] = lambda: image_service
    scheduler = MagicMock()
    app.dependency_overrides[get_analysis_scheduler] = lambda: scheduler
    return app, queue, image_service, scheduler, provider


def photos(count=1):
    return [("receipt", (f"{i}.png", JPEG_BYTES + bytes([i]), "image/png")) for i in range(count)]


@pytest.mark.parametrize("count", [1, 2])
async def test_one_and_many_accept_same_durable_pollable_contract(upload_context, count):
    app, queue, _, scheduler, provider = upload_context
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/api/v1/images", files=photos(count))
        assert response.status_code == 202
        assert len(queue.creates) == 1, "upload bypassed durable workflow acceptance"
        status = response.json()
        assert status["id"] == str(queue.creates[0].id)
        assert "claim_token" not in status and "lease_expires_at" not in status
        polled = await client.get(response.headers["location"])
        assert polled.status_code == 200
        assert polled.json() == status
    assert len(queue.creates) == 1
    assert len(queue.creates[0].parts) == count
    for index, part in enumerate(queue.creates[0].parts):
        assert Path(part.image_path).read_bytes() == JPEG_BYTES + bytes([index])
        assert Path(part.image_path).name == f"{index}.jpg"
    scheduler.trigger.assert_called_once()
    provider.check_connection.assert_not_awaited()
    provider.get_available_models.assert_not_awaited()
    provider.analyse_receipt_from_model.assert_not_awaited()


async def test_concurrent_replays_freeze_defaults_and_clean_only_loser(upload_context, settings):
    app, queue, _, _, provider = upload_context
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:

        async def send():
            return await client.post(
                "/api/v1/images", files=photos(2), headers={"Idempotency-Key": "key"}
            )

        first, second = await asyncio.gather(send(), send())
        assert first.status_code == second.status_code == 202
        assert first.json() == second.json()
        first_endpoint = provider.inference_identity
        provider.inference_identity = "openai:" + "a" * 64
        settings.llm.model_name = "changed-default"
        settings.images.bypass_review_default = True
        third = await send()
        assert third.json() == first.json()
        for query in ["?model_id=explicit", "?bypass_review=false"]:
            conflict = await client.post(
                "/api/v1/images" + query, files=photos(2), headers={"Idempotency-Key": "key"}
            )
            assert conflict.status_code == 409
        reordered = await client.post(
            "/api/v1/images", files=list(reversed(photos(2))), headers={"Idempotency-Key": "key"}
        )
        assert reordered.status_code == 409
    assert len(queue.creates) == 1
    for part in queue.creates[0].parts:
        assert Path(part.image_path).is_file()
    assert len(list(Path(settings.images.save_dir).glob("submissions/*/*.jpg"))) == 2
    assert queue.creates[0].model_id != "changed-default"
    assert queue.creates[0].effective_options["inference_identity"] == first_endpoint


async def test_ambiguous_commit_and_thumbnail_failure_keep_winning_originals(
    upload_context, monkeypatch
):
    app, queue, image_service, _, _ = upload_context
    monkeypatch.setattr(image_service, "generate_thumbnail", lambda path: None)
    queue.lose_response = True
    async with AsyncClient(
        transport=ASGITransport(app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        first = await client.post(
            "/api/v1/images", files=photos(2), headers={"Idempotency-Key": "lost"}
        )
        assert first.status_code == 500
        for part in queue.creates[0].parts:
            assert Path(part.image_path).is_file()
        replay = await client.post(
            "/api/v1/images", files=photos(2), headers={"Idempotency-Key": "lost"}
        )
        assert replay.status_code == 202
        assert replay.json()["id"] == str(queue.creates[0].id)
    for part in queue.creates[0].parts:
        assert Path(part.image_path).is_file()


async def test_replay_cleanup_failure_does_not_hide_committed_acceptance(
    upload_context, monkeypatch
):
    app, queue, image_service, _, _ = upload_context
    async with AsyncClient(
        transport=ASGITransport(app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        first = await client.post(
            "/api/v1/images", files=photos(), headers={"Idempotency-Key": "cleanup"}
        )

        def cannot_delete(path):
            raise OSError("filesystem unavailable")

        monkeypatch.setattr(image_service, "delete_image", cannot_delete)
        replay = await client.post(
            "/api/v1/images", files=photos(), headers={"Idempotency-Key": "cleanup"}
        )
        assert replay.status_code == 202
        assert replay.json() == first.json()
        assert Path(queue.creates[0].parts[0].image_path).is_file()


@pytest.mark.parametrize("case,expected", [("count", 422), ("size", 413), ("mime", 415)])
async def test_invalid_uploads_never_enqueue_and_remove_unreferenced_originals(
    upload_context, settings, case, expected
):
    app, queue, _, _, _ = upload_context
    payload = photos(11) if case == "count" else photos()
    if case == "size":
        payload.append(("receipt", ("large.jpg", b"x" * (20 * 1024 * 1024 + 1), "image/jpeg")))
    elif case == "mime":
        payload.append(("receipt", ("invalid.txt", b"not an image", "text/plain")))
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/api/v1/images", files=payload)
        assert response.status_code == expected
    assert not queue.creates
    assert not list(Path(settings.images.save_dir).glob("submissions/*/*.jpg"))


async def test_explicit_model_presence_is_identity_and_contents_are_not_logged(
    upload_context, settings, caplog
):
    app, queue, _, _, _ = upload_context
    marker = b"private-OCR-receipt-marker"
    payload = [("receipt", ("private.jpg", JPEG_BYTES + marker, "image/jpeg"))]
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        first = await client.post(
            "/api/v1/images", files=payload, headers={"Idempotency-Key": "intent"}
        )
        assert first.status_code == 202
        explicit = await client.post(
            "/api/v1/images",
            params={"model_id": settings.llm.model_name},
            files=payload,
            headers={"Idempotency-Key": "intent"},
        )
        assert explicit.status_code == 409
        changed = await client.post(
            "/api/v1/images", files=photos(), headers={"Idempotency-Key": "intent"}
        )
        assert changed.status_code == 409
        blank = await client.post("/api/v1/images", files=payload, headers={"Idempotency-Key": " "})
        assert blank.status_code == 422
    assert len(queue.creates) == 1
    assert marker.decode() not in caplog.text


@pytest.mark.parametrize("permission", [None, False, True])
async def test_fallback_permission_is_separate_frozen_intent(upload_context, settings, permission):
    app, queue, *_ = upload_context
    settings.ocr.enabled = True
    settings.ocr.allow_vision_fallback = True
    query = "" if permission is None else "?allow_ocr_fallback=" + str(permission).lower()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/images" + query, files=photos(), headers={"Idempotency-Key": "fallback"}
        )
        assert response.status_code == 202
        saved = queue.creates[0]
        assert saved.effective_options["execution_policy_version"] == 4
        assert saved.effective_options["inference_identity"] == "ollama:" + "d" * 64
        assert saved.effective_options["managed_binding_sha256"] is None
        assert saved.effective_options["allow_ocr_fallback"] is False
        assert ("allow_ocr_fallback" in saved.requested_options) is (permission is not None)
        settings.ocr.allow_vision_fallback = False
        replay = await client.post(
            "/api/v1/images" + query, files=photos(), headers={"Idempotency-Key": "fallback"}
        )
        assert replay.json()["id"] == response.json()["id"]
        changed = "?allow_ocr_fallback=" + str(permission is not True).lower()
        conflict = await client.post(
            "/api/v1/images" + changed, files=photos(), headers={"Idempotency-Key": "fallback"}
        )
        assert conflict.status_code == 409
        assert saved.effective_options["ocr"]["allow_vision_fallback"] is False
