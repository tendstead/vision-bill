"""Recover submission leases and persisted retry budgets on private PostgreSQL."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest

from vision_bill.config import WorkerSettings
from vision_bill.model.workflow import AccessScope, AttemptOutcome, RetryRequest
from vision_bill.provider.db.workflow_db import WorkflowDB
from vision_bill.service.analysis_scheduler import AnalysisScheduler
from vision_bill.service.submission_service import SubmissionService

from .test_submission_service_e2e import setup
from .test_workflow_db_e2e import complete, enqueue
from .test_workflow_db_e2e import db as db  # noqa: PLC0414 - pytest fixture re-export
from .test_workflow_db_e2e import postgres as postgres  # noqa: PLC0414 - pytest fixture re-export
from .test_workflow_db_e2e import (
    workflow_db as workflow_db,  # noqa: PLC0414 - pytest fixture re-export
)

pytestmark = pytest.mark.e2e


async def claim_attempt(workflow, image_id, attempt_number):
    for _ in range(attempt_number - 1):
        claim = await workflow.claim_next(image_id=image_id)
        await workflow.finish_attempt(claim, AttemptOutcome(kind="retry", retry_delay_seconds=0))
    claim = await workflow.claim_next(image_id=image_id)
    assert claim.attempt_number == attempt_number
    return claim


async def expire(workflow, *image_ids):
    async with workflow.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=ANY($1)",
            list(image_ids),
        )
        return await conn.fetchval("SELECT clock_timestamp()")


async def make_due(workflow, image_id):
    async with workflow.pool.acquire() as conn:
        await conn.execute("UPDATE images SET retry_at=clock_timestamp() WHERE id=$1", image_id)


async def test_scheduler_recovers_abandoned_work_while_provider_is_unavailable(
    workflow_db,
    settings,
    tmp_path,
):
    workflow, owner = workflow_db
    deletable, paths, provider, _, service = await setup(
        workflow, owner, settings, tmp_path, ocr=False
    )
    provider.check_connection = AsyncMock(return_value=False)
    scheduler = AnalysisScheduler(
        settings, provider, service.receipts, None, submission_service=service
    )

    snapshot = service.current_snapshot()
    remaining = await enqueue(workflow, owner, effective_options=snapshot)
    exhausted = await enqueue(workflow, owner, effective_options=snapshot)
    live = await enqueue(workflow, owner, effective_options=snapshot)
    await claim_attempt(workflow, remaining.id, 1)
    await claim_attempt(workflow, exhausted.id, 3)
    await claim_attempt(workflow, deletable.id, 3)
    live_claim = await claim_attempt(workflow, live.id, 1)
    before = await expire(workflow, remaining.id, exhausted.id, deletable.id)

    assert await scheduler.process_pending() == []
    async with workflow.pool.acquire() as conn:
        after = await conn.fetchval("SELECT clock_timestamp()")
        rows = {
            row["id"]: row
            for row in await conn.fetch("""SELECT id,status,claim_token,lease_expires_at,processing_at,
                generation,attempt_count,max_attempts,result_code,retry_at FROM images""")
        }
        abandonment = await conn.fetch("""SELECT image_id,attempt_number,outcome_kind
            FROM receipt_extractions WHERE result_code='lease_expired'""")
    row = rows[remaining.id]
    assert (row["status"], row["attempt_count"], row["max_attempts"], row["generation"]) == (
        "pending",
        1,
        3,
        1,
    )
    assert before <= row["retry_at"] - timedelta(seconds=30) <= after
    for image_id in (exhausted.id, deletable.id):
        assert (rows[image_id]["status"], rows[image_id]["attempt_count"]) == ("failed", 3)
        assert (
            rows[image_id]["result_code"] == "attempt_limit" and rows[image_id]["retry_at"] is None
        )
    for image_id in (remaining.id, exhausted.id, deletable.id):
        dead = rows[image_id]
        assert dead["claim_token"] is dead["lease_expires_at"] is dead["processing_at"] is None
    assert {tuple(record.values()) for record in abandonment} == {
        (remaining.id, 1, "retry"),
        (exhausted.id, 3, "terminal"),
        (deletable.id, 3, "terminal"),
    }
    assert rows[live.id]["status"] == "processing" and rows[live.id]["attempt_count"] == 1
    assert rows[live.id]["claim_token"] == live_claim.claim_token
    assert rows[live.id]["lease_expires_at"] == live_claim.lease_expires_at
    provider.check_connection.assert_not_awaited()

    await make_due(workflow, remaining.id)
    for _ in range(4):
        assert await scheduler.process_pending() == []
    pending = await workflow.get_submission(remaining.id, scope=owner)
    assert pending.status == "pending" and pending.attempt_count == 1
    assert pending.wait_reason == "provider_unavailable"
    assert provider.check_connection.await_count == 4
    assert remaining.id in scheduler._deferred_image_ids
    assert (
        await workflow.next_wake_delay(
            300,
            deferred_image_ids=tuple(scheduler._deferred_image_ids),
        )
        > 0
    )

    request = RetryRequest(expected_generation=1, request_key="offline-retry")
    outsider = AccessScope(user_id=uuid4())
    assert await workflow.retry_submission(exhausted.id, request, scope=outsider) is None
    retried = await workflow.retry_submission(exhausted.id, request, scope=owner)
    assert (retried.status, retried.generation, retried.attempt_count) == ("pending", 2, 0)
    assert not await service.receipts.delete_resource(
        deletable.id, receipt=False, scope=outsider, images=service.images
    )
    assert await service.receipts.delete_resource(
        deletable.id, receipt=False, scope=owner, images=service.images
    )
    assert all(not path.exists() for path in paths)


async def test_two_reapers_and_late_completion_seal_one_abandonment(workflow_db):
    workflow, owner = workflow_db
    image = await enqueue(workflow, owner)
    dead = await workflow.claim_next(image_id=image.id)
    await expire(workflow, image.id)
    assert await workflow.claim_next(image_id=image.id) is None

    first, second, late = await asyncio.gather(
        workflow.reap_expired_claims(deadline=asyncio.get_running_loop().time() + 5),
        workflow.reap_expired_claims(),
        workflow.finish_attempt(dead, complete()),
    )
    assert late.stale
    # Both concurrent sweeps may skip the row while the stale finisher holds its lock.
    follow_up = await workflow.reap_expired_claims()
    assert first + second + follow_up == 1
    assert await workflow.reap_expired_claims() == 0
    assert await workflow.claim_next(image_id=image.id) is None
    await make_due(workflow, image.id)
    successor = await workflow.claim_next(image_id=image.id)
    assert successor.generation == dead.generation and successor.attempt_number == 2
    assert successor.claim_token != dead.claim_token
    assert (await workflow.finish_attempt(dead, complete())).stale
    async with workflow.pool.acquire() as conn:
        history = await conn.fetch(
            """SELECT generation,attempt_number,outcome_kind,result_code
            FROM receipt_extractions WHERE image_id=$1""",
            image.id,
        )
        assert [tuple(row.values()) for row in history] == [(1, 1, "retry", "lease_expired")]
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0


async def test_contended_expired_lease_defers_until_post_lock_state_is_visible(workflow_db):
    workflow, owner = workflow_db
    image = await enqueue(workflow, owner)
    claim = await workflow.claim_next(image_id=image.id)
    await expire(workflow, image.id)
    async with workflow.pool.acquire() as blocker, blocker.transaction():
        await blocker.fetchval("SELECT id FROM images WHERE id=$1 FOR UPDATE", image.id)
        assert (
            await workflow.reap_expired_claims(deadline=asyncio.get_running_loop().time() + 5) == 0
        )
        assert 0 < await workflow.next_wake_delay(300) < 300
        await blocker.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp()+interval '30 seconds' WHERE id=$1",
            image.id,
        )
    assert await workflow.reap_expired_claims() == 0
    assert (await workflow.get_submission(image.id, scope=owner)).attempt_count == 1
    assert not (await workflow.finish_attempt(claim, complete())).stale


async def test_abandonment_failure_rolls_back_claim_clearance(workflow_db):
    workflow, owner = workflow_db
    image = await enqueue(workflow, owner)
    claim = await workflow.claim_next(image_id=image.id)
    await expire(workflow, image.id)
    async with workflow.pool.acquire() as conn:
        await conn.execute("""CREATE FUNCTION reject_abandonment() RETURNS trigger LANGUAGE plpgsql
            AS $$ BEGIN RAISE EXCEPTION 'injected abandonment failure'; END $$""")
        await conn.execute("""CREATE TRIGGER reject_abandonment BEFORE INSERT ON receipt_extractions
            FOR EACH ROW EXECUTE FUNCTION reject_abandonment()""")
    with pytest.raises(asyncpg.RaiseError, match="injected abandonment failure"):
        await workflow.reap_expired_claims()
    async with workflow.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT status, claim_token FROM images WHERE id=$1", image.id)
        assert tuple(row.values()) == ("processing", claim.claim_token)
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 0
        await conn.execute("DROP TRIGGER reject_abandonment ON receipt_extractions")
    assert await workflow.reap_expired_claims() == 1


@pytest.mark.parametrize(
    "current_max,delays,cap,attempt_number,expected_delay",
    [(1, (), 300, 1, 30), (2, (30,), 300, 2, 60), (2, (30,), 35, 2, 35)],
)
async def test_restart_keeps_accepted_budget_with_shorter_current_backoff(
    workflow_db,
    settings,
    tmp_path,
    current_max,
    delays,
    cap,
    attempt_number,
    expected_delay,
):
    workflow, owner = workflow_db
    image, _, _, _, service = await setup(workflow, owner, settings, tmp_path, ocr=False)
    claim = await claim_attempt(workflow, image.id, attempt_number)
    settings.worker = WorkerSettings(
        max_attempts=current_max,
        retry_backoff_seconds=delays,
        retry_backoff_cap_seconds=cap,
    )
    restarted = WorkflowDB(workflow._settings, worker_settings=settings.worker)
    restarted._pool = workflow.pool
    service.receipts._workflow_db = restarted
    service = SubmissionService(settings, service.receipts, service.images)
    service.receipts.extract_receipt = AsyncMock(side_effect=TimeoutError)
    async with workflow.pool.acquire() as conn:
        before = await conn.fetchval("SELECT clock_timestamp()")
    outcome = await service.process_claim(claim)
    async with workflow.pool.acquire() as conn:
        after = await conn.fetchval("SELECT clock_timestamp()")
        row = await conn.fetchrow(
            "SELECT claim_token, lease_expires_at FROM images WHERE id=$1", image.id
        )
    assert not outcome.stale
    saved = outcome.submission
    assert saved.status == "pending" and saved.attempt_count == attempt_number
    assert saved.result_code == "timed_out" and saved.max_attempts == 3
    assert before <= saved.retry_at - timedelta(seconds=expected_delay) <= after
    assert row["claim_token"] is row["lease_expires_at"] is None
    assert await restarted.claim_next(image_id=image.id) is None
    await make_due(restarted, image.id)
    successor = await restarted.claim_next(image_id=image.id)
    assert successor.attempt_number == attempt_number + 1 and successor.image.max_attempts == 3
