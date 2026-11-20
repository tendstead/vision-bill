"""Upload route + original files + isolated PostgreSQL acceptance and restart."""

import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from test_submission_upload import photos
from test_submission_upload import (
    upload_context as upload_context,  # noqa: PLC0414 - re-export imported pytest fixture
)

from vision_bill.api.helper.helper import get_receipt_service
from vision_bill.provider.db.workflow_db import WorkflowDB
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User

from .test_workflow_db_e2e import db as db  # noqa: PLC0414 - re-export imported pytest fixture
from .test_workflow_db_e2e import (
    postgres as postgres,  # noqa: PLC0414 - re-export imported pytest fixture
)
from .test_workflow_db_e2e import (
    workflow_db as workflow_db,  # noqa: PLC0414 - re-export imported pytest fixture
)

pytestmark = pytest.mark.e2e


async def test_real_upload_concurrent_lost_response_and_restart(
    upload_context, workflow_db, settings
):
    app, _, _, _, provider = upload_context
    database, scope = workflow_db
    service = app.dependency_overrides[get_receipt_service]()
    service._workflow_db = database
    app.dependency_overrides[get_current_user] = lambda: User(id=scope.user_id, username="owner")
    headers = {"Idempotency-Key": "concurrent"}
    async with AsyncClient(
        transport=ASGITransport(app, raise_app_exceptions=False), base_url="http://test"
    ) as client:

        async def send():
            return await client.post("/api/v1/images", files=photos(2), headers=headers)

        first, second = await asyncio.gather(send(), send())
        assert first.status_code == second.status_code == 202
        assert first.json() == second.json()
        image_id = first.json()["id"]
        async with database.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM images WHERE id=$1", first.json()["id"])
            assert await conn.fetchval("SELECT count(*) FROM images") == 1
            assert await conn.fetchval("SELECT count(*) FROM image_source_parts") == 2
        sources = await database.get_source_parts(row["id"], scope=scope)
        assert len({part.part_id for part in sources}) == 2
        assert [part.photo_index for part in sources] == [0, 1]
        originals = [Path(part.image_path) for part in sources]
        assert all(path.is_file() for path in originals)
        assert len(list((Path(settings.images.save_dir) / "submissions").glob("*/*.jpg"))) == 2
        old_model = settings.llm.model_name
        settings.llm.model_name = "new-default"
        settings.images.bypass_review_default = True
        replay = await send()
        assert replay.json() == first.json()
        async with database.pool.acquire() as conn:
            stored = await conn.fetchrow("SELECT * FROM images WHERE id=$1", row["id"])
            assert stored["model_id"] == old_model
            assert stored["effective_options"] == row["effective_options"]
        # The exact same committed transaction can lose its response to the caller.
        enqueue = database.enqueue_submission

        async def lose_response(*args, **kwargs):
            await enqueue(*args, **kwargs)
            raise ConnectionError("COMMIT response lost")

        database.enqueue_submission = lose_response
        lost = await client.post(
            "/api/v1/images", files=photos(), headers={"Idempotency-Key": "lost"}
        )
        assert lost.status_code == 500
        database.enqueue_submission = enqueue
        recovered = await client.post(
            "/api/v1/images", files=photos(), headers={"Idempotency-Key": "lost"}
        )
        assert recovered.status_code == 202
        recovered_id = recovered.json()["id"]
        # A fresh DB pool reads committed status without any process-local cache.
        restarted = WorkflowDB(database._settings)
        await restarted.init_db()
        try:
            service._workflow_db = restarted
            polled = await client.get(first.headers["location"])
            assert polled.status_code == 200
            assert polled.json()["id"] == image_id
            assert (await client.get(recovered.headers["location"])).json()["id"] == recovered_id
        finally:
            service._workflow_db = database
            await restarted.destroy_db()
        assert all(path.is_file() for path in originals)
        # An explicit intent change conflicts; original files still survive.
        changed = await client.post(
            "/api/v1/images?bypass_review=false", files=photos(2), headers=headers
        )
        assert changed.status_code == 409
        assert all(path.is_file() for path in originals)
    provider.analyse_receipt_from_model.assert_not_awaited()
