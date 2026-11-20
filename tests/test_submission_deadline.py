"""Normal upload deadlines exercise installed asyncpg, simulating only wire waits."""

import asyncio
from unittest.mock import AsyncMock

import asyncpg
import pytest
from test_benchmark_db_deadline import WireConnection
from test_submission_service import context

from vision_bill.model.workflow import AttemptOutcome
from vision_bill.provider.db.workflow_db import WorkflowDB


class UploadWire(WireConnection):
    async def fetchrow(self, sql, *args, **kwargs):
        self.fetch_count += 1
        if self.phase == "rollback":
            await self.pause(10)
        if self.phase == "error":
            raise ValueError("original wire failure")
        if self.phase == "release":
            await self.pause(.08)

    async def fetch(self, sql, *args, **kwargs):
        return []


async def wire_pool(phase):
    made = []

    async def connect(*args, **kwargs):
        raw = UploadWire(phase)
        made.append(raw)
        return raw

    async def setup(conn):
        if phase == "acquire_setup":
            await conn.pause(10)

    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect,
                                    connection_class=UploadWire, setup=setup)
    return pool, made


@pytest.mark.parametrize("attempt", [1, 3])
@pytest.mark.parametrize("phase", ["rollback", "release", "acquire_setup"])
async def test_upload_finish_bounds_real_driver_cleanup(settings, tmp_path, phase, attempt):
    service, receipts, db, claim, _ = context(settings, tmp_path)
    claim = claim.model_copy(update={"attempt_number": attempt})
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": .1})
    receipts.extract_receipt.side_effect = TimeoutError()
    pool, made = await wire_pool(phase)
    w = WorkflowDB(settings.pg)
    w._pool = pool
    db.finish_attempt = w.finish_attempt
    started = asyncio.get_running_loop().time()
    try:
        assert (await service.process_claim(claim)).stale
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < .2, f"budget=.1s, elapsed={elapsed:.3f}s"
        assert made[0].terminated and made[0].active_io == 0
        assert made[0].commit_at is None or made[0].commit_at < started + .1
        assert len(made) == 1, "no fresh failure bookkeeping after deadline"
        pool._holders[0]._setup = None
        async with asyncio.timeout(.2), pool.acquire() as replacement:
            assert replacement._con is not made[0]
            replacement.terminate()
    finally:
        pool.terminate()


@pytest.mark.parametrize("operation", ["bind", "snapshot", "get", "clear", "save", "finish", "references", "tags"])
async def test_each_acquisition_uses_absolute_deadline(settings, tmp_path, operation):
    from vision_bill.provider.db.receipt_db import ReceiptDB
    from vision_bill.service.receipt_service import ReceiptService

    _, _, _, claim, _ = context(settings, tmp_path)
    pool, made = await wire_pool("acquire_setup")
    w = WorkflowDB(settings.pg)
    w._pool = pool
    receipts = ReceiptService(settings.images, settings.pg, AsyncMock())
    receipts._db = ReceiptDB(settings.pg)
    receipts._db._pool = pool
    deadline = asyncio.get_running_loop().time() + .05
    calls = {
        "bind": lambda: w.bind_source_parts(claim, [], deadline=deadline),
        "snapshot": lambda: w.ensure_execution_snapshot(claim, model_id="m", effective_options={}, deadline=deadline),
        "get": lambda: w.get_ocr_checkpoint(claim, deadline=deadline),
        "clear": lambda: w.clear_ocr_checkpoint(claim, deadline=deadline),
        "save": lambda: w.save_ocr_checkpoint(claim, None, deadline=deadline),
        "finish": lambda: w.finish_attempt(claim, AttemptOutcome(kind="retry", result_code="timed_out"), deadline=deadline),
        "references": lambda: w.unreferenced_views(claim.image.id, [], deadline=deadline),
        "tags": lambda: receipts.list_tags(deadline=deadline),
    }
    try:
        with pytest.raises(TimeoutError):
            await calls[operation]()
        assert asyncio.get_running_loop().time() < deadline + .1
        assert made[0].terminated and made[0].active_io == 0
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_offset_clocks_keep_distinct_work_and_final_db_deadlines(settings, tmp_path, monkeypatch):
    import vision_bill.helper.deadline as deadlines
    import vision_bill.service.submission_service as submissions

    service, receipts, db, claim, _ = context(settings, tmp_path)
    monkeypatch.setattr(submissions, "monotonic", lambda: 100.0)
    monkeypatch.setattr(deadlines, "monotonic", lambda: 100.0)
    started = asyncio.get_running_loop().time()
    receipts.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    assert receipts.extract_receipt.call_args.kwargs["deadline"] == 695.0
    values = [method.call_args.kwargs["deadline"] for method in
              (db.bind_source_parts, db.get_ocr_checkpoint, receipts.list_tags)]
    assert len(set(values)) == 1
    final_deadline = db.finish_attempt.call_args.kwargs["deadline"]
    assert started + 599 < final_deadline <= asyncio.get_running_loop().time() + 600
    assert final_deadline - values[0] == 5.0
    assert values[0] != 695.0


async def test_queued_removal_checks_deadline_immediately_before_unlink(settings, tmp_path, monkeypatch):
    import threading
    from pathlib import Path
    from time import monotonic

    from test_ocr_provider import wire_page

    from vision_bill.model.ocr import WorkingImage, WorkingImageMetadata

    service, _, _, claim, _ = context(settings, tmp_path)
    working = WorkingImage.model_validate(wire_page()["working_image"])
    path = await service.images.save_working_image(claim.image.id, working, deadline=monotonic()+2)
    metadata = WorkingImageMetadata.model_validate(working.model_dump())
    entered, resume = threading.Event(), threading.Event()
    real_check = Path.is_symlink

    def barrier(path):
        if path == target:
            entered.set()
            assert resume.wait(2)
        return real_check(path)

    target = path
    monkeypatch.setattr(Path, "is_symlink", barrier)
    deadline = monotonic() + .1
    task = asyncio.create_task(asyncio.to_thread(service.images.remove_unreferenced_working_image,
                                                claim.image.id, str(path), metadata, deadline=deadline))
    try:
        assert await asyncio.to_thread(entered.wait, .5), "remover must reach path validation"
        await asyncio.sleep(max(0, deadline-monotonic())+.01)
        resume.set()
        with pytest.raises(TimeoutError):
            await task
        assert path.exists()
        assert Path(claim.image.image_path).exists()
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["rollback", "release"])
async def test_uncertain_reference_cleanup_retains_new_artifact(settings, tmp_path, phase):
    from pathlib import Path

    from test_ocr_provider import wire_page

    from vision_bill.model.ocr import OcrResponse

    service, receipts, db, claim, part = context(settings, tmp_path)
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": .3})
    db.save_ocr_checkpoint.return_value = False
    pool, made = await wire_pool(phase)
    w = WorkflowDB(settings.pg)
    w._pool = pool
    db.unreferenced_views = w.unreferenced_views

    async def extract(*args, **kwargs):
        await kwargs["checkpoint"](OcrResponse.model_validate(wire_page(0, part.source_sha256)))
        raise AssertionError("stale checkpoint cannot format")

    receipts.extract_receipt.side_effect = extract
    started = asyncio.get_running_loop().time()
    try:
        assert (await service.process_claim(claim)).stale
        assert asyncio.get_running_loop().time() - started < .4
        assert made[0].terminated and made[0].active_io == 0
        assert pool._queue.qsize() == 1
        assert len(list((tmp_path / "uploads" / "working").glob("*/*.png"))) == 1
        assert Path(claim.image.image_path).exists()
        db.finish_attempt.assert_not_awaited()
    finally:
        pool.terminate()


async def test_upload_external_cancellation_preserved_through_driver_rollback(settings, tmp_path):
    service, receipts, db, claim, _ = context(settings, tmp_path)
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": .1})
    receipts.extract_receipt.side_effect = TimeoutError()
    pool, made = await wire_pool("rollback")
    w = WorkflowDB(settings.pg)
    w._pool = pool
    db.finish_attempt = w.finish_attempt
    task = asyncio.create_task(service.process_claim(claim))
    started = asyncio.get_running_loop().time()
    try:
        async with asyncio.timeout(.2):
            while not made or not made[0].fetch_count:
                await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert asyncio.get_running_loop().time() - started < .2
        assert made[0].active_io == 0 and made[0].terminated
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_finish_preserves_original_body_error_when_rollback_expires(settings, tmp_path):
    _, _, _, claim, _ = context(settings, tmp_path)
    pool, made = await wire_pool("error")
    w = WorkflowDB(settings.pg)
    w._pool = pool
    deadline = asyncio.get_running_loop().time() + .05
    try:
        with pytest.raises(ValueError, match="original wire failure"):
            await w.finish_attempt(claim, AttemptOutcome(kind="retry", result_code="timed_out"), deadline=deadline)
        assert asyncio.get_running_loop().time() < deadline + .1
        assert made[0].active_io == 0 and made[0].terminated
        assert made[0].commands == ["BEGIN;", "ROLLBACK;"]
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_tag_read_release_is_bounded_and_ordinary_forwarding_stays_compatible(settings):
    from vision_bill.service.receipt_service import ReceiptService

    receipts = ReceiptService(settings.images, settings.pg, AsyncMock())
    receipts._db.list_tags = AsyncMock(return_value=["food"])
    assert await receipts.list_tags() == ["food"]
    receipts._db.list_tags.assert_awaited_once_with()
    del receipts._db.list_tags
    pool, made = await wire_pool("read_release")
    receipts._db._pool = pool
    deadline = asyncio.get_running_loop().time() + .05
    try:
        with pytest.raises(TimeoutError):
            await receipts.list_tags(deadline=deadline)
        assert asyncio.get_running_loop().time() < deadline + .1
        assert made[0].active_io == 0 and made[0].terminated
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_submission_queue_timeout_leaves_unrelated_borrower_usable(settings, tmp_path):
    service, receipts, _, claim, _ = context(settings, tmp_path)
    pool, made = await wire_pool("responsive")
    w = WorkflowDB(settings.pg)
    w._pool = pool
    service.db = w
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": .05})
    peer = await pool.acquire()
    try:
        assert (await service.process_claim(claim)).stale
        assert not peer.is_closed() and not made[0].terminated
        assert await peer.fetch("SELECT harmless") == []
        assert pool._queue.qsize() == 0
        receipts.extract_receipt.assert_not_awaited()
        await pool.release(peer)
        async with pool.acquire() as reused:
            assert reused._con is made[0]
    finally:
        pool.terminate()


async def test_file_cleanup_uses_one_cap_for_all_owned_views(settings, tmp_path, monkeypatch):
    from test_ocr_provider import wire_page

    import vision_bill.helper.deadline as deadlines
    import vision_bill.service.submission_service as submissions
    from vision_bill.model.ocr import OcrResponse

    service, receipts, db, claim, part = context(settings, tmp_path)
    from time import monotonic

    started = monotonic()
    clock = [started]
    monkeypatch.setattr(submissions, "monotonic", lambda: clock[0])
    monkeypatch.setattr(deadlines, "monotonic", lambda: clock[0])
    db.unreferenced_views = AsyncMock(side_effect=lambda image_id, views, **kwargs: views)

    async def extract(*args, **kwargs):
        for _ in range(2):
            await kwargs["checkpoint"](OcrResponse.model_validate(wire_page(0, part.source_sha256)))
        raise submissions.StaleClaimError

    receipts.extract_receipt.side_effect = extract
    remove = service.images.remove_unreferenced_working_image
    removals = []

    def expire_after_first(*args, **kwargs):
        removals.append(kwargs["deadline"])
        remove(*args, **kwargs)
        clock[0] = started + 6

    service.images.remove_unreferenced_working_image = expire_after_first
    assert (await service.process_claim(claim)).stale
    assert removals == [started + 5]
    assert len(list((tmp_path / "uploads" / "working").glob("*/*.png"))) == 1
    db.finish_attempt.assert_not_awaited()
