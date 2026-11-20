import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from test_analysis_scheduler import context
from test_receipts_api import (
    IMAGE_ID,
    IMAGES_URL,
    JPEG_BYTES,
    _image_row,
    _upload_jpeg,
)
from test_receipts_api import (
    api_context as api_context,  # noqa: PLC0414 - re-export imported pytest fixture
)


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
    assert response.json()["id"] == str(IMAGE_ID)
    wake.assert_called_once()
    ctx.provider.analyse_receipt_from_model.assert_not_awaited()
    ctx.provider.get_available_models.assert_not_awaited()
    args = ctx.conn.fetchrow.call_args.args
    assert args[2] == "top.jpg"
    assert json.loads(args[9])[0]["original_filename"] == "bottom.jpg"


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
        additional_images=[{"image_path": str(photo), "media_type": "image/jpeg"}],
    )
    response = api_context.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file?part=1")
    assert response.status_code == 200
    assert response.content == JPEG_BYTES
    assert api_context.client.get(f"{IMAGES_URL}/{IMAGE_ID}/file?part=2").status_code == 404


async def test_worker_trigger_does_not_wait_for_interval(settings):
    scheduler, *_ = context(settings)
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
