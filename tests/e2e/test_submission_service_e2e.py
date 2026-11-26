"""Actual PostgreSQL transactions, actual PNGs, typed pipeline and deterministic inference."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image
from test_extraction_pipeline import formatter, sidecar

from vision_bill.api.drafts import router
from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.workflow import SubmissionCreate
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User
from vision_bill.service.execution_policy import current_snapshot
from vision_bill.service.extraction_pipeline import ExtractionPipeline
from vision_bill.service.image_service import ImageService
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.service.submission_service import SubmissionService

from .test_workflow_db_e2e import db as db  # noqa: PLC0414 - pytest fixture re-export
from .test_workflow_db_e2e import migrate
from .test_workflow_db_e2e import postgres as postgres  # noqa: PLC0414 - pytest fixture re-export
from .test_workflow_db_e2e import (
    workflow_db as workflow_db,  # noqa: PLC0414 - pytest fixture re-export
)

pytestmark = pytest.mark.e2e


async def setup(w, scope, settings, tmp_path, *, ocr=True):
    paths = [tmp_path / f"{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        Image.new("RGB", (4, 5), (i, 0, 0)).save(path)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    config = OcrSettings(enabled=ocr, required=ocr, manage_model_lifecycle=False)
    settings.ocr = config
    settings.llm.temperature = 0.13
    settings.llm.max_output_tokens = 1234
    llm, native = formatter(not ocr), sidecar()
    upload = SubmissionCreate(
        id=uuid4(),
        parts=[ImagePart(image_path=str(p)) for p in paths],
        source_manifest=[{"photo_index": i, "source_sha256": h} for i, h in enumerate(hashes)],
        requested_options={},
        effective_options=current_snapshot(settings, inference_identity=llm.inference_identity),
        request_fingerprint=uuid4().hex,
        model_id="fixture",
        bypass_review=True,
    )
    image = (await w.enqueue_submission(upload, scope=scope, idempotency_key=None)).submission
    native.get_capabilities.return_value.runtime_fingerprint = "runtime"
    receipts = ReceiptService(settings.images, settings.pg, llm)
    receipts._workflow_db = w
    receipts._db._pool = w.pool
    receipts.ocr_capabilities = native.get_capabilities

    async def extract(model, paths, *, ocr_settings, **kwargs):
        return await ExtractionPipeline(llm, native if ocr else None, ocr_settings).extract(
            model, paths, **kwargs
        )

    receipts.extract_receipt = extract
    images = ImageService(settings.images)
    service = SubmissionService(settings, receipts, images)
    return image, paths, llm, native, service


@pytest.mark.parametrize(
    "change",
    ["none", "runtime", "profile", "corrupt", "source", "race", "malformed_page", "malformed_manifest"],
)
async def test_checkpoint_reload_reuses_only_exact_prefix(workflow_db, settings, tmp_path, change):
    w, scope = workflow_db
    image, paths, _llm, native, service = await setup(w, scope, settings, tmp_path)
    original_extract = native.extract_page.side_effect

    async def fail_second(*args, **kwargs):
        if kwargs["photo_index"] == 1:
            raise TimeoutError()
        return await original_extract(*args, **kwargs)

    native.extract_page.side_effect = fail_second
    first = await service.process_claim(await w.claim_next())
    assert first.submission.status == "pending" and not first.submission.terminal
    async with w.pool.acquire() as conn:
        checkpoint = await conn.fetchval("SELECT ocr_checkpoint FROM images WHERE id=$1", image.id)
        assert "png_base64" not in checkpoint
        await conn.execute("UPDATE images SET retry_at=clock_timestamp() WHERE id=$1", image.id)
    claim = await w.claim_next()
    saved = await w.get_ocr_checkpoint(claim)
    if change == "runtime":
        native.get_capabilities.return_value.runtime_fingerprint = "new-runtime"
    if change == "profile":
        native.get_capabilities.return_value.profile_id = "rotation-only-profile"
    if change == "race":
        old = native.get_capabilities.return_value.model_copy()
        new = old.model_copy(update={"runtime_fingerprint": "new-runtime"})
        native.get_capabilities.side_effect = [old, new, new]
    if change == "corrupt":
        Path(saved.persisted_views[0].artifact_path).write_bytes(b"corrupt")
    if change == "source":
        paths[0].write_bytes(b"changed original")
    if change in {"malformed_page", "malformed_manifest"}:
        payload = json.loads(checkpoint)
        if change == "malformed_page":
            payload["pages"][0]["working_image"]["width"] = "invalid"
        else:
            payload["persisted_views"][0]["part_id"] = "invalid"
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE images SET ocr_checkpoint=$2::jsonb WHERE id=$1",
                image.id,
                json.dumps(payload),
            )
        original_load = w.get_ocr_checkpoint
        stale = claim.model_copy(update={"claim_token": uuid4()})
        assert await original_load(stale) is None
        async with w.pool.acquire() as conn:
            assert (
                json.loads(
                    await conn.fetchval("SELECT ocr_checkpoint FROM images WHERE id=$1", image.id)
                )
                == payload
            )

        async def load(claim, *, deadline=None):
            result = await original_load(claim, deadline=deadline)
            assert result is None
            async with w.pool.acquire() as conn:
                assert (
                    await conn.fetchval("SELECT ocr_checkpoint FROM images WHERE id=$1", image.id)
                    is None
                )
                assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 1
                assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == 1
            assert Path(saved.persisted_views[0].artifact_path).exists()
            return result

        w.get_ocr_checkpoint = load

    async def page(*args, **kwargs):
        result = await original_extract(*args, **kwargs)
        if change in {"runtime", "race"}:
            result.runtime_fingerprint = "new-runtime"
        if change == "profile":
            result.profile_id = "rotation-only-profile"
        return result

    native.extract_page = AsyncMock(side_effect=page)
    original_attempt = service._attempt
    deadlines = []

    async def attempt(claim, deadline, owned, *, db_deadline):
        deadlines.append(deadline)
        return await original_attempt(claim, deadline, owned, db_deadline=db_deadline)

    service._attempt = attempt
    second = await service.process_claim(claim)
    assert len(deadlines) == 1
    for call in native.extract_page.await_args_list:
        assert call.kwargs["deadline"] == deadlines[0]
    if change == "source":
        assert second.submission.status == "failed"
        assert second.submission.result_code == "source_integrity_failed"
        native.extract_page.assert_not_awaited()
        return
    assert second.submission.status == "analyzed"
    assert not second.receipt.verified
    assert native.extract_page.await_count == (1 if change == "none" else 2)
    assert all(p.exists() for p in paths)
    status = await w.get_submission(image.id, scope=scope)
    extraction = await w.get_extraction(image.id, status.latest_extraction_id, scope=scope)
    assert len(extraction["envelope"]["ocr_pages"]) == 2
    if change == "profile":
        assert all(page["profile_id"] == "rotation-only-profile" for page in extraction["envelope"]["ocr_pages"])
    assert "artifact_path" not in str(extraction) and "png_base64" not in str(extraction)
    assert status.source_views
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 2
        assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == (
            2 if change == "none" else 3
        )


async def test_direct_source_routes_owner_admin_integrity_and_exact_bytes(
    workflow_db, settings, tmp_path
):
    w, scope = workflow_db
    image, _paths, _llm, _native, service = await setup(w, scope, settings, tmp_path, ocr=False)
    result = await service.process_claim(await w.claim_next())
    assert result.receipt and not result.receipt.verified
    status = await w.get_submission(image.id, scope=scope)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/images")
    app.state.submission_service = service
    user = User(id=scope.user_id, username="owner")
    app.dependency_overrides[get_current_user] = lambda: user
    extraction_url = f"/api/v1/images/{image.id}/extractions/{status.latest_extraction_id}"
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.get(extraction_url)
        assert response.status_code == 200 and "artifact_path" not in response.text
        view = status.source_views[0]
        manifest = await w.get_working_view(image.id, view.view_id, scope=scope)
        png = await client.get(view.url)
        assert png.content == Path(manifest.artifact_path).read_bytes()
        assert png.headers["content-type"] == "image/png"
        user = User(id=uuid4(), username="other")
        assert (await client.get(view.url)).status_code == 404
        assert (await client.get(extraction_url)).status_code == 404
        user.can_see_all = True
        assert (await client.get(view.url)).status_code == 200
        Path(manifest.artifact_path).write_bytes(b"tampered")
        assert (await client.get(view.url)).status_code == 404


async def test_cancelled_formatter_retains_checkpoint_and_no_receipt(
    workflow_db, settings, tmp_path
):
    w, scope = workflow_db
    _image, _paths, llm, _native, service = await setup(w, scope, settings, tmp_path)
    entered = asyncio.Event()

    async def hang(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    llm._client.chat.side_effect = hang
    claim = await w.claim_next()
    task = asyncio.create_task(service.process_claim(claim))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len((await w.get_ocr_checkpoint(claim)).pages) == 2
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 0


@pytest.mark.parametrize("old_status", ["pending", "failed"])
async def test_migrated_unknown_destination_requires_explicit_adoption(db, settings, tmp_path, old_status):
    from vision_bill.provider.db.workflow_db import WorkflowDB

    migrate(db, "0010")
    path = tmp_path / "legacy.png"
    Image.new("RGB", (4, 5)).save(path)
    conn = await asyncpg.connect(db.pg_dsn)
    image_id = await conn.fetchval(
        "INSERT INTO images(status,image_path) VALUES($1,$2) RETURNING id", old_status, str(path)
    )
    await conn.close()
    migrate(db)
    w = WorkflowDB(db)
    await w.init_db()
    try:
        llm = formatter()
        settings.llm.model_name = "fixture"
        receipts = ReceiptService(settings.images, db, llm)
        receipts._workflow_db = w
        receipts.list_tags = AsyncMock(return_value=[])
        service = SubmissionService(settings, receipts, ImageService(settings.images))
        claim = await w.claim_next()
        assert not claim.image.effective_options
        result = await service.process_claim(claim)
        assert result.submission.id == image_id and path.exists()
        assert result.submission.wait_reason == "execution_policy_changed"
        assert result.submission.attempt_count == 0
        async with w.pool.acquire() as conn:
            saved = await conn.fetchval("SELECT effective_options FROM images WHERE id=$1", image_id)
        assert json.loads(saved) == {}
        llm._client.chat.assert_not_awaited()
    finally:
        await w.destroy_db()


async def test_delete_owned_unfinalized_review_draft(workflow_db, settings, tmp_path):
    from fastapi import HTTPException

    from vision_bill.api.images import delete_image

    from .test_workflow_db_e2e import partial_draft

    w, scope = workflow_db
    path = tmp_path / "original.png"
    image = await partial_draft(w, scope, path)
    receipts = ReceiptService(settings.images, settings.pg, formatter())
    receipts._workflow_db = w
    receipts._db._pool = w.pool
    images = ImageService(settings.images)
    owner = User(id=scope.user_id, username="owner")
    with pytest.raises(HTTPException) as error:
        await delete_image(image.id, receipts, images, User(id=uuid4(), username="outsider"))
    assert error.value.status_code == 404 and path.exists()
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipt_drafts WHERE image_id=$1", image.id) == 1
    assert await delete_image(image.id, receipts, images, owner) == {"deleted": image.id}
    assert not path.exists()
    async with w.pool.acquire() as conn:
        for table in ("images", "receipt_drafts", "receipt_extractions", "image_source_parts"):
            column = "id" if table == "images" else "image_id"
            assert await conn.fetchval(f"SELECT count(*) FROM {table} WHERE {column}=$1", image.id) == 0
        assert await conn.fetchval("SELECT count(*) FROM resource_deletions") == 0


@pytest.mark.parametrize("linked", [False, True])
async def test_delete_active_or_receipt_linked_submission_conflicts(workflow_db, settings, tmp_path, linked):
    from fastapi import HTTPException

    from vision_bill.api.images import delete_image

    w, scope = workflow_db
    image, paths, _llm, _native, service = await setup(w, scope, settings, tmp_path)
    claim = await w.claim_next()
    if linked:
        result = await service.process_claim(claim)
        assert result.receipt is not None
    service.receipts._db._pool = w.pool
    with pytest.raises(HTTPException) as error:
        await delete_image(image.id, service.receipts, service.images, User(id=scope.user_id, username="owner"))
    assert error.value.status_code == 409 and all(path.exists() for path in paths)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM images WHERE id=$1", image.id) == 1
        assert await conn.fetchval("SELECT count(*) FROM resource_deletions") == 0
        if linked:
            assert await conn.fetchval("SELECT count(*) FROM receipts WHERE image_id=$1", image.id) == 1


@pytest.mark.parametrize('completed',[False,True])
async def test_committed_resource_deletion_removes_owned_originals_and_working_files(workflow_db,settings,tmp_path,completed):
    from vision_bill.api.images import delete_image
    from vision_bill.api.receipts import delete_receipt
    w,scope=workflow_db
    image,paths,llm,native,service=await setup(w,scope,settings,tmp_path)
    if not completed:
        llm._client.chat.side_effect=ValueError('private failure')
    result=await service.process_claim(await w.claim_next())
    async with w.pool.acquire() as conn:
        working=[Path(r['artifact_path']) for r in await conn.fetch('SELECT artifact_path FROM extraction_working_views WHERE image_id=$1',image.id)]
        if not completed:
            await conn.execute("UPDATE images SET status='failed' WHERE id=$1",image.id)
    assert working and all(p.exists() for p in working)
    receipts=service.receipts
    receipts._db._pool=w.pool
    receipts._image_db._pool=w.pool
    user=User(id=scope.user_id,username='owner')
    if completed:
        assert result.receipt is not None
        await delete_receipt(result.receipt.id,receipts,service.images,user)
    else:
        await delete_image(image.id,receipts,service.images,user)
    assert not any(p.exists() for p in [*paths,*working])
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM images WHERE id=$1',image.id)==0


async def test_delete_unlink_failure_retains_owner_scoped_retry_manifest(workflow_db,settings,tmp_path,monkeypatch):
    from vision_bill.api.receipts import delete_receipt
    from fastapi import HTTPException
    w,scope=workflow_db
    image,paths,llm,native,service=await setup(w,scope,settings,tmp_path)
    result=await service.process_claim(await w.claim_next())
    receipts=service.receipts;receipts._db._pool=w.pool
    user=User(id=scope.user_id,username='owner')
    original=service.images.remove_deleted_files
    monkeypatch.setattr(service.images,'remove_deleted_files',lambda _: (_ for _ in ()).throw(OSError('private path')))
    with pytest.raises(OSError): await delete_receipt(result.receipt.id,receipts,service.images,user)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==1
        assert await conn.fetchval('SELECT count(*) FROM images WHERE id=$1',image.id)==0
        assert await conn.fetchval('SELECT count(*) FROM receipts WHERE id=$1',result.receipt.id)==0
    assert all(path.exists() for path in paths)
    monkeypatch.setattr(service.images,'remove_deleted_files',original)
    with pytest.raises(HTTPException) as error:
        await delete_receipt(result.receipt.id,receipts,service.images,User(id=uuid4(),username='outsider'))
    assert error.value.status_code==404 and all(path.exists() for path in paths)
    await delete_receipt(result.receipt.id,receipts,service.images,user)
    assert not any(path.exists() for path in paths)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==0


@pytest.mark.parametrize("failure", ["open", "fsync"])
async def test_owned_deletion_sync_failure_retains_manifest_until_durable_retry(
    workflow_db, settings, tmp_path, monkeypatch, failure
):
    w, scope = workflow_db
    image, originals, _, _, service = await setup(w, scope, settings, tmp_path)
    result = await service.process_claim(await w.claim_next())
    assert result.receipt is not None
    thumbnails = [service.images.generate_thumbnail(path) for path in originals]
    assert all(path is not None for path in thumbnails)
    async with w.pool.acquire() as conn:
        working = [
            Path(row["artifact_path"]) for row in await conn.fetch(
                "SELECT artifact_path FROM extraction_working_views WHERE image_id=$1", image.id
            )
        ]
    assert working
    owned = [*originals, *thumbnails, *working]
    parents = {path.parent for path in owned}
    assert len(parents) == 3
    foreign = tmp_path / "foreign" / "foreign.png"
    foreign.parent.mkdir()
    foreign.write_bytes(b"foreign")

    events = []
    real_fsync = os.fsync
    real_open = os.open
    real_execute = asyncpg.Connection.execute

    def fail_directory_open(path, flags, *args, **kwargs):
        if flags & os.O_DIRECTORY and Path(path) in parents:
            raise OSError("directory open failed")
        return real_open(path, flags, *args, **kwargs)

    def fail_fsync(_fd):
        raise OSError("directory fsync failed")

    def record_fsync(fd):
        assert not any(path.exists() for path in owned)
        real_fsync(fd)
        events.append(("fsync_parent", os.fstat(fd).st_ino))

    async def execute(conn, query, *args, **kwargs):
        if query.startswith("DELETE FROM resource_deletions"):
            events.append(("delete_cleanup_manifest", None))
        return await real_execute(conn, query, *args, **kwargs)

    async def delete():
        return await service.receipts.delete_resource(
            result.receipt.id, receipt=True, scope=scope, images=service.images
        )

    monkeypatch.setattr(asyncpg.Connection, "execute", execute)
    with monkeypatch.context() as failing:
        if failure == "open":
            failing.setattr(os, "open", fail_directory_open)
        else:
            failing.setattr(os, "fsync", fail_fsync)
        with pytest.raises(OSError, match=f"directory {failure} failed"):
            await delete()

    assert events == []
    assert not any(path.exists() for path in owned)
    assert foreign.read_bytes() == b"foreign"
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM resource_deletions") == 1
        assert await conn.fetchval("SELECT count(*) FROM images") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0

    monkeypatch.setattr(os, "fsync", record_fsync)
    assert await delete()
    assert sorted(events[:-1]) == sorted(
        ("fsync_parent", path.stat().st_ino) for path in parents
    )
    assert events[-1] == ("delete_cleanup_manifest", None)
    assert foreign.read_bytes() == b"foreign"
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM resource_deletions") == 0


async def test_provider_wait_visible_and_stale_probe_cannot_touch_claim(workflow_db,settings,tmp_path):
    w,scope=workflow_db
    image,paths,llm,native,service=await setup(w,scope,settings,tmp_path)
    service.receipts.check_connection=AsyncMock(return_value=False)
    from vision_bill.model.db.image import ImageRow
    candidate=(await w.due_candidates())[0]
    assert not await service.ready_for(candidate)
    status=await w.get_submission(image.id,scope=scope)
    assert status.wait_reason=='provider_unavailable' and status.attempt_count==0
    assert (await w.due_candidates())[0].wait_reason=='provider_unavailable'
    service.receipts.check_connection.return_value=True
    service.receipts.get_available_models=llm.get_available_models
    assert await service.ready_for(candidate)
    assert (await w.get_submission(image.id,scope=scope)).wait_reason is None
    claim=await w.claim_next()
    assert not await w.set_wait_reason(candidate,'provider_unavailable')
    assert not await w.set_wait_reason(claim.image.model_copy(update={'generation':2}),'provider_unavailable')
    current=await w.get_submission(image.id,scope=scope)
    assert current.wait_reason is None and current.attempt_count==1


async def test_benchmark_blocked_delete_rolls_back_receipt_image_and_cleanup_manifest(workflow_db,settings,tmp_path):
    from vision_bill.api.receipts import delete_receipt
    from fastapi import HTTPException
    w,scope=workflow_db
    image,paths,llm,native,service=await setup(w,scope,settings,tmp_path)
    result=await service.process_claim(await w.claim_next())
    service.receipts._db._pool=w.pool
    async with w.pool.acquire() as conn:
        run=await conn.fetchval("INSERT INTO benchmark_runs(model_ids,receipt_ids,dataset_fingerprint,prompt_version,scoring_version) VALUES('[]','[]','f','4','4') RETURNING id")
        await conn.execute("INSERT INTO benchmark_tasks(run_id,model_id,receipt_id) VALUES($1,'fixture',$2)",run,result.receipt.id)
        working=[Path(r['artifact_path']) for r in await conn.fetch('SELECT artifact_path FROM extraction_working_views WHERE image_id=$1',image.id)]
    with pytest.raises(HTTPException) as error:
        await delete_receipt(result.receipt.id,service.receipts,service.images,User(id=scope.user_id,username='owner'))
    assert error.value.status_code==409 and 'benchmark' in error.value.detail
    assert all(p.exists() for p in [*paths,*working])
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==0
        assert await conn.fetchval('SELECT count(*) FROM images WHERE id=$1',image.id)==1
        assert await conn.fetchval('SELECT count(*) FROM receipts WHERE id=$1',result.receipt.id)==1


@pytest.mark.parametrize('interruption',['cancel','uncertain_commit'])
async def test_deletion_interruption_never_unlinks_until_confirmed_retry(workflow_db,settings,tmp_path,monkeypatch,interruption):
    w,scope=workflow_db
    image,paths,_,_,service=await setup(w,scope,settings,tmp_path)
    result=await service.process_claim(await w.claim_next())
    service.receipts._db._pool=w.pool
    remove=service.images.remove_deleted_files
    calls=[]
    monkeypatch.setattr(service.images,'remove_deleted_files',lambda deleted:(calls.append(deleted),remove(deleted)))
    if interruption=='cancel':
        async with w.pool.acquire() as blocker,blocker.transaction():
            await blocker.fetchrow('SELECT id FROM images WHERE id=$1 FOR UPDATE',image.id)
            task=asyncio.create_task(service.receipts.delete_resource(result.receipt.id,receipt=True,scope=scope,images=service.images))
            try:
                async with asyncio.timeout(3):
                    async with w.pool.acquire() as observer:
                        while not await observer.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock')"):
                            await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError): await task
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True)
        async with w.pool.acquire() as conn:
            assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==0
            assert await conn.fetchval('SELECT count(*) FROM images WHERE id=$1',image.id)==1
    else:
        delete=w.delete_resource
        async def lost_ack(*args,**kwargs):
            await delete(*args,**kwargs)
            raise ConnectionError('commit acknowledgement lost')
        monkeypatch.setattr(w,'delete_resource',lost_ack)
        with pytest.raises(ConnectionError):
            await service.receipts.delete_resource(result.receipt.id,receipt=True,scope=scope,images=service.images)
        monkeypatch.setattr(w,'delete_resource',delete)
        async with w.pool.acquire() as conn:
            assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==1
    assert calls==[] and all(path.exists() for path in paths)
    assert await service.receipts.delete_resource(result.receipt.id,receipt=True,scope=scope,images=service.images)
    assert len(calls)==1 and not any(path.exists() for path in paths)


@pytest.mark.parametrize('changed',['original','working','symlink','live_reference'])
async def test_cleanup_retry_retains_changed_or_live_evidence(workflow_db,settings,tmp_path,changed):
    w,scope=workflow_db
    image,paths,_,_,service=await setup(w,scope,settings,tmp_path)
    result=await service.process_claim(await w.claim_next())
    deleted=await w.delete_resource(result.receipt.id,receipt=True,scope=scope,
        validate_files=lambda d:service.images.validate_deleted_files(d,initial=True))
    working=Path(deleted.working_views[0].artifact_path)
    if changed=='original': paths[0].write_bytes(b'changed original')
    if changed=='working': working.write_bytes(b'changed derived bytes')
    if changed=='symlink':
        paths[0].unlink();paths[0].symlink_to(paths[1])
    if changed=='live_reference':
        async with w.pool.acquire() as conn:
            await conn.execute("INSERT INTO images(image_path,user_id,status) VALUES($1,$2,'pending')",str(paths[0]),scope.user_id)
    with pytest.raises(ValueError):
        await w.cleanup_deleted_resource(result.receipt.id,receipt=True,scope=scope,remove_files=service.images.remove_deleted_files)
    assert all(p.exists() for p in paths) and working.exists()
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==1


async def test_cleanup_admin_visibility_and_duplicate_retries_are_serialized(workflow_db,settings,tmp_path):
    from vision_bill.model.workflow import AccessScope
    w,scope=workflow_db
    image,paths,_,_,service=await setup(w,scope,settings,tmp_path)
    result=await service.process_claim(await w.claim_next())
    await w.delete_resource(result.receipt.id,receipt=True,scope=scope,
        validate_files=lambda d:service.images.validate_deleted_files(d,initial=True))
    admin_id=uuid4()
    assert await w.delete_resource(result.receipt.id,receipt=True,scope=AccessScope(user_id=admin_id),validate_files=lambda _:None) is None
    admin=AccessScope(user_id=admin_id,can_see_all=True)
    assert await w.delete_resource(result.receipt.id,receipt=True,scope=admin,validate_files=lambda _:None)
    calls=[]
    def remove(deleted):
        calls.append(deleted);service.images.remove_deleted_files(deleted)
    results=await asyncio.gather(*(w.cleanup_deleted_resource(result.receipt.id,receipt=True,scope=admin,remove_files=remove) for _ in range(2)))
    assert sorted(results)==[False,True] and len(calls)==1
    assert not any(path.exists() for path in paths)


async def test_deletion_preserves_shared_original_and_rejects_changed_durable_source(workflow_db,settings,tmp_path):
    w,scope=workflow_db
    image,paths,_,_,service=await setup(w,scope,settings,tmp_path)
    result=await service.process_claim(await w.claim_next())
    original=paths[0].read_bytes()
    paths[0].write_bytes(b'changed source')
    with pytest.raises(ValueError,match='identity_changed'):
        await w.delete_resource(result.receipt.id,receipt=True,scope=scope,
            validate_files=lambda d:service.images.validate_deleted_files(d,initial=True))
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM receipts WHERE id=$1',result.receipt.id)==1
        assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==0
        shared=await conn.fetchval("INSERT INTO images(image_path,user_id,status) VALUES($1,$2,'pending') RETURNING id",str(paths[0]),scope.user_id)
    paths[0].write_bytes(original)
    assert await service.receipts.delete_resource(result.receipt.id,receipt=True,scope=scope,images=service.images)
    assert paths[0].read_bytes()==original and not paths[1].exists()
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM images WHERE id=$1',shared)==1
        assert await conn.fetchval('SELECT count(*) FROM resource_deletions')==0


@pytest.mark.parametrize('legacy', [True, False])
@pytest.mark.parametrize('discovery', ['selected', 'alternative', 'text-only'])
async def test_scheduler_initial_legacy_model_resolution_is_frozen_once(workflow_db,settings,tmp_path,legacy,discovery):
    from vision_bill.service.analysis_scheduler import AnalysisScheduler
    from vision_bill.provider.llm.base import ModelInfo
    w,scope=workflow_db
    image,paths,llm,_,service=await setup(w,scope,settings,tmp_path,ocr=False)
    preferred = 'fixture' if discovery == 'selected' else 'missing-preferred'
    settings.llm.model_name=preferred
    settings.ocr=OcrSettings(enabled=False)
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE images SET model_id=$2 WHERE id=$1",image.id, preferred)
        if legacy:
            await conn.execute("UPDATE images SET effective_options='{}', request_fingerprint=NULL WHERE id=$1",image.id)
    candidate=(await w.due_candidates())[0]
    frozen=dict(candidate.effective_options)
    service.receipts.check_connection=AsyncMock(return_value=True)
    if discovery == 'text-only':
        text=ModelInfo('text-only');text.capabilities=['completion']
        llm.get_available_models.return_value=[text]
    service.receipts.get_available_models=llm.get_available_models
    scheduler=AnalysisScheduler(settings,llm,service.receipts,service.receipts._image_db,service.images,submission_service=service)
    await scheduler.process_pending()
    async with w.pool.acquire() as conn:
        saved=await conn.fetchrow('SELECT * FROM images WHERE id=$1',image.id)
    if legacy:
        assert saved['wait_reason'] == 'execution_policy_changed'
        assert saved['attempt_count'] == 0 and saved['receipt_id'] is None
        assert json.loads(saved['effective_options']) == frozen
        llm._client.chat.assert_not_awaited()
        return
    if discovery == 'selected':
        assert saved['attempt_count']==1 and saved['receipt_id'] is not None, (saved['status'],saved['result_code'],saved['model_id'])
        assert saved['model_id']=='fixture'
        snapshot=json.loads(saved['effective_options'])
        assert snapshot == frozen
        assert snapshot['allow_ocr_fallback'] is False
        assert llm._client.chat.call_args.kwargs['model']=='fixture'
        assert await w.claim_next() is None
    else:
        assert saved['attempt_count']==0 and saved['receipt_id'] is None
        assert saved['model_id']==preferred
        assert json.loads(saved['effective_options'])==frozen
        assert saved['wait_reason']=='vision_model_unavailable'
        llm._client.chat.assert_not_awaited()


async def test_bound_direct_model_vanishing_before_dispatch_refunds_reservation(
    workflow_db, settings, tmp_path,
):
    w, scope = workflow_db
    image, _, llm, _, service = await setup(w, scope, settings, tmp_path, ocr=False)
    selected = llm.get_available_models.return_value
    llm.get_available_models.side_effect = [selected, [], selected, selected]
    service.receipts.check_connection = AsyncMock(return_value=True)
    before = (await w.due_candidates())[0]
    assert await service.ready_for(before)
    claim = await w.claim_next(image_id=image.id)
    assert claim is not None and claim.attempt_number == 1
    await service.process_claim(claim)

    waiting = await w.get_submission(image.id, scope=scope)
    assert waiting.status == "pending" and waiting.wait_reason == "vision_model_unavailable"
    assert waiting.attempt_count == 0 and waiting.receipt_id is None
    llm._client.chat.assert_not_awaited()
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions WHERE image_id=$1", image.id) == 0

    assert await service.ready_for((await w.due_candidates())[0])
    resumed = await w.claim_next(image_id=image.id)
    assert resumed is not None and resumed.attempt_number == 1
    result = await service.process_claim(resumed)
    assert result.receipt is not None and result.submission.attempt_count == 1
    assert llm._client.chat.await_args.kwargs["model"] == "fixture"


@pytest.mark.parametrize("ocr", [False, True], ids=["direct", "ocr_formatter"])
async def test_provider_change_requires_explicit_policy_adoption(workflow_db, settings, tmp_path, ocr):
    from vision_bill.model.workflow import RetryRequest
    from vision_bill.provider.llm.openai import OpenAIProvider

    w, scope = workflow_db
    image, paths, original, _, service = await setup(w, scope, settings, tmp_path, ocr=ocr)
    changed = OpenAIProvider("http://other.invalid/v1", "fixture-secret")
    try:
        restarted = ReceiptService(settings.images, settings.pg, changed)
        restarted._workflow_db = w
        restarted.check_connection = AsyncMock(side_effect=AssertionError("held job probed new provider"))
        worker = SubmissionService(settings, restarted, service.images)
        candidate, = await w.due_candidates()
        assert not await worker.ready_for(candidate)
        held = await w.get_submission(image.id, scope=scope)
        assert held.wait_reason == "execution_policy_changed" and held.attempt_count == 0
        restarted.check_connection.assert_not_awaited()
        original._client.chat.assert_not_awaited()

        request = RetryRequest(expected_generation=1, request_key="accept-new-provider", policy="current")
        adopted = await w.retry_submission(
            image.id, request, scope=scope, replacement_snapshot=worker.current_snapshot(),
        )
        assert adopted.generation == 2 and adopted.attempt_count == 0
        assert await w.retry_submission(image.id, request, scope=scope) == adopted
        updated, = await w.due_candidates()
        assert updated.effective_options["inference_identity"] == changed.inference_identity
        async with w.pool.acquire() as conn:
            history = json.loads(await conn.fetchval("SELECT legacy_events FROM images WHERE id=$1", image.id))
        assert history[-1]["effective_options"]["inference_identity"] == original.inference_identity
        assert all(path.exists() for path in paths)
    finally:
        await changed._client.close()
        await original._client._client.aclose()
