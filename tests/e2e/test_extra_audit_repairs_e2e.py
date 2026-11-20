"""Focused owned PostgreSQL seams; no app image, browser, model or operator data."""
import asyncio
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from .test_workflow_db_e2e import db as db, postgres as postgres, workflow_db as workflow_db
from vision_bill.provider.db.collection_db import CollectionDB
from vision_bill.provider.db.receipt_db import MEMBERSHIP_LOCK_SQL
from vision_bill.model.db.image import ImagePart
from vision_bill.model.workflow import SubmissionCreate
from vision_bill.service.image_service import ImageService
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.security.models import User
from vision_bill.security.dependencies import get_current_user
from vision_bill.api.helper.helper import get_receipt_service, get_image_service

pytestmark = pytest.mark.e2e


async def owner(pool):
    value = uuid4()
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO users(id,username,hashed_password) VALUES($1,$2,'test')", value, str(value))
    return value


async def active_ids(pool):
    async with pool.acquire() as conn:
        return {r['user_id']: r['id'] for r in await conn.fetch('SELECT user_id,id FROM collections WHERE active')}


async def test_admin_activation_uses_target_owner_and_preserves_other_owners(workflow_db):
    w, scope = workflow_db
    db = CollectionDB(w.pool)
    other = await owner(w.pool)
    a = await db.create(scope.user_id, 'own A', None, None, None, True)
    b = await db.create(scope.user_id, 'own B', None, None, None)
    c = await db.create(other, 'other A', None, None, None, True)
    d = await db.create(other, 'other B', None, None, None)
    assert await db.activate(d.id, scope.user_id, False) is None
    assert await active_ids(w.pool) == {scope.user_id: a.id, other: c.id}
    assert (await db.activate(b.id, scope.user_id, True)).active
    assert (await db.activate(d.id, scope.user_id, True)).active
    assert await active_ids(w.pool) == {scope.user_id: b.id, other: d.id}


async def test_create_active_and_activate_share_capture_lock_order(workflow_db):
    w, scope = workflow_db
    db = CollectionDB(w.pool)
    first = await db.create(scope.user_id, 'first', None, None, None, True)
    target = await db.create(scope.user_id, 'target', None, None, None)
    async with w.pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        await conn.execute(MEMBERSHIP_LOCK_SQL)
        jobs = [asyncio.create_task(db.create(scope.user_id, 'created', None, None, None, True)), asyncio.create_task(db.activate(target.id, scope.user_id, False))]
        try:
            await asyncio.sleep(.05)
            assert not any(job.done() for job in jobs), 'transition bypassed capture ownership lock'
            assert await active_ids(w.pool) == {scope.user_id: first.id}
        finally:
            await transaction.rollback()
            results = await asyncio.gather(*jobs, return_exceptions=True)
    assert all(not isinstance(result, BaseException) for result in results)
    current = await active_ids(w.pool)
    assert len(current) == 1 and current[scope.user_id] in {result.id for result in results}


@pytest.mark.parametrize('operation', ['activate', 'create'])
async def test_cancelled_active_transition_rolls_back_sibling_deactivation(workflow_db, operation):
    w, scope = workflow_db
    db = CollectionDB(w.pool)
    first = await db.create(scope.user_id, 'first', None, None, None, True)
    target = await db.create(scope.user_id, 'target', None, None, None)
    key = 92341711
    async with w.pool.acquire() as conn:
        await conn.execute(f"""CREATE FUNCTION pause_active() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
            IF NEW.active THEN PERFORM pg_advisory_xact_lock({key}); END IF; RETURN NEW; END $$;
            CREATE TRIGGER pause_active BEFORE INSERT OR UPDATE ON collections FOR EACH ROW EXECUTE FUNCTION pause_active();""")
        await conn.execute('SELECT pg_advisory_lock($1)', key)
        task = asyncio.create_task(db.activate(target.id, scope.user_id, False) if operation == 'activate' else db.create(scope.user_id, 'cancelled', None, None, None, True))
        try:
            async with asyncio.timeout(2):
                while not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND objid=$1 AND NOT granted)", key):
                    await asyncio.sleep(.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await conn.execute('SELECT pg_advisory_unlock($1)', key)
            if not task.done(): task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert await active_ids(w.pool) == {scope.user_id: first.id}


@pytest.mark.parametrize('interrupted', [False, True])
async def test_relative_original_thumbnail_api_delete_and_durable_retry(workflow_db, settings, tmp_path, monkeypatch, interrupted):
    from vision_bill.api.images import router
    w, scope = workflow_db
    monkeypatch.chdir(tmp_path)
    settings.images.save_dir = 'uploads'
    settings.images.tmp_dir = 'tmp'
    images = ImageService(settings.images)
    original = Path('uploads/original.png')
    Image.new('RGB', (12, 12), 'white').save(original)
    thumb = images.generate_thumbnail(original)
    assert thumb is not None and not thumb.is_absolute()
    ident = uuid4()
    upload = SubmissionCreate(id=ident, parts=[ImagePart(image_path=str(original), thumbnail_path=str(thumb))], source_manifest=[{'photo_index': 0, 'source_sha256': sha256(original.read_bytes()).hexdigest()}], requested_options={}, effective_options={}, request_fingerprint=uuid4().hex)
    await w.enqueue_submission(upload, scope=scope, idempotency_key=None)
    receipts = ReceiptService(settings.images, settings.pg, MagicMock())
    receipts._workflow_db = w
    receipts._db._pool = w.pool
    app = FastAPI(); app.include_router(router, prefix='/api/v1/images')
    app.dependency_overrides[get_receipt_service] = lambda: receipts
    app.dependency_overrides[get_image_service] = lambda: images
    app.dependency_overrides[get_current_user] = lambda: User(id=uuid4(), username='outsider')
    unlink = Path.unlink
    failed = False
    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if interrupted and path == thumb and not failed:
            failed = True
            raise OSError('synthetic thumbnail interruption')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail_once)
    async with AsyncClient(transport=ASGITransport(app, raise_app_exceptions=False), base_url='http://test') as client:
        assert (await client.delete(f'/api/v1/images/{ident}')).status_code == 404
        assert original.exists() and thumb.exists()
        app.dependency_overrides[get_current_user] = lambda: User(id=scope.user_id, username='owner')
        response = await client.delete(f'/api/v1/images/{ident}')
        if interrupted:
            assert response.status_code == 500
            assert not original.exists() and thumb.exists()
            async with w.pool.acquire() as conn:
                assert await conn.fetchval('SELECT count(*) FROM resource_deletions') == 1
            response = await client.delete(f'/api/v1/images/{ident}')
        assert response.status_code == 200, response.text
    assert not original.exists() and not thumb.exists()
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM images') == 0
        assert await conn.fetchval('SELECT count(*) FROM resource_deletions') == 0
