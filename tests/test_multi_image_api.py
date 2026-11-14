import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_receipts_api import (
    IMAGE_ID,
    IMAGES_URL,
    JPEG_BYTES,
    _image_row,
    _upload_jpeg,
    api_context,
)
from test_analysis_scheduler import scheduler_context, _pending_image, _make_receipt, _receipt_row
from vision_bill.model.db.image import ImagePart
from vision_bill.provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError
from vision_bill.provider.db.image_db import LIST_PENDING_IMAGES_SQL


def test_multi_upload_persists_order_and_wakes_worker_without_calling_llm(api_context, monkeypatch):
    ctx = api_context
    wake = MagicMock()
    monkeypatch.setattr(ctx.client.app.state.analysis_scheduler, "trigger", wake)
    ctx.conn.fetchrow.return_value = _image_row()
    response = ctx.client.post(
        IMAGES_URL,
        files=[
            ("receipt", ("top.jpg", JPEG_BYTES, "image/jpeg")),
            ("receipt", ("bottom.jpg", JPEG_BYTES, "image/jpeg")),
        ],
    )
    assert response.status_code == 202
    assert response.json()["image_id"] == str(IMAGE_ID)
    wake.assert_called_once()
    ctx.provider.analyse_receipt_from_model.assert_not_awaited()
    args = ctx.conn.fetchrow.call_args.args
    assert args[1] == "top.jpg"
    assert json.loads(args[9])[0]["original_filename"] == "bottom.jpg"


@pytest.mark.parametrize(
    "error,status,http_status",
    [
        (AnalysisTimeoutError("Timed out"), "timed_out", 504),
        (UnreadableReceiptError("Blurry"), "unreadable", 422),
    ],
)
def test_single_upload_records_terminal_error(api_context, error, status, http_status):
    ctx = api_context
    ctx.provider.analysis_timeout_seconds = 1
    ctx.provider.analyse_receipt_from_model.side_effect = error
    ctx.conn.fetchrow.return_value = _image_row()
    response = ctx.client.post(IMAGES_URL, files={"receipt": _upload_jpeg()})
    assert response.status_code == http_status
    assert response.json()["status"] == status
    assert response.json()["image_id"] == str(IMAGE_ID)
    assert ctx.conn.execute.call_args.args[2] == status
    assert status not in LIST_PENDING_IMAGES_SQL


def test_invalid_second_photo_cleans_up_first(api_context, settings):
    response = api_context.client.post(
        IMAGES_URL,
        files=[
            ("receipt", _upload_jpeg()),
            ("receipt", ("bad.txt", b"text", "text/plain")),
        ],
    )
    assert response.status_code == 415
    assert not list(Path(settings.images.tmp_dir).glob("*.png"))


def test_additional_photo_can_be_viewed(api_context, tmp_path):
    photo = tmp_path / "second.jpg"
    photo.write_bytes(JPEG_BYTES)
    api_context.conn.fetchrow.return_value = _image_row(
        image_path="first.jpg",
        additional_images=[dict(image_path=str(photo), media_type="image/jpeg")],
    )
    response = api_context.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file?part=1")
    assert response.status_code == 200
    assert response.content == JPEG_BYTES
    assert api_context.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file?part=2").status_code == 404


async def test_worker_trigger_does_not_wait_for_interval(scheduler_context):
    scheduler, *_ = scheduler_context
    ran = asyncio.Event()

    async def process():
        ran.set()
        return []

    scheduler.process_pending = AsyncMock(side_effect=process)
    await scheduler.start()
    try:
        await asyncio.wait_for(ran.wait(), 1)
        ran.clear()
        scheduler.trigger()
        await asyncio.wait_for(ran.wait(), 1)
    finally:
        await scheduler.stop()


@pytest.mark.parametrize(
    "error,status",
    [
        (AnalysisTimeoutError("Timed out"), "timed_out"),
        (UnreadableReceiptError("Blurry"), "unreadable"),
    ],
)
async def test_worker_records_terminal_errors(scheduler_context, tmp_path, error, status):
    scheduler, _, service, db = scheduler_context
    photo = tmp_path / "top.jpg"
    photo.write_bytes(JPEG_BYTES)
    db.mark_terminal = AsyncMock()
    service.analyse_receipt_with_ocr.side_effect = error
    result = await scheduler._analyze_one(_pending_image(IMAGE_ID, photo), "test")
    assert result.status == status
    db.mark_terminal.assert_awaited_once_with(IMAGE_ID, status, str(error))
    db.mark_failed.assert_not_awaited()


async def test_worker_passes_all_photos_in_order(scheduler_context, tmp_path):
    scheduler, _, service, db = scheduler_context
    photo = tmp_path / "top.jpg"
    photo.write_bytes(JPEG_BYTES)
    image = _pending_image(IMAGE_ID, photo)
    image.additional_images = [ImagePart(image_path="bottom.jpg")]
    service.analyse_receipt_from_path.return_value = _make_receipt()
    service.persist_receipt.return_value = _receipt_row()
    result = await scheduler._analyze_one(image, "test")
    assert result.status == "analyzed"
    service.analyse_receipt_from_path.assert_awaited_once_with("test", [photo, Path("bottom.jpg")])
