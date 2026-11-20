"""Benchmark reservations, actual attempts and fenced completion on private PostgreSQL."""

import asyncio
from uuid import uuid4

import asyncpg
import pytest

from tests.e2e import test_benchmark_references_e2e as references
from vision_bill.config import WorkerSettings
from vision_bill.model.benchmark import CouncilPolicy
from vision_bill.provider.db.benchmark_db import BenchmarkDB

postgres, db, workflow_db = references.postgres, references.db, references.workflow_db
pytestmark = pytest.mark.e2e


async def queue(w, scope, tmp_path, *, models=("m",), settings=None, version="2"):
    row, _, path = await references.human(w, scope, tmp_path)
    b = BenchmarkDB(w.pool, w._receipts, worker=settings) if settings else BenchmarkDB(w.pool, w._receipts)
    run = await b.create_run(
        model_ids=list(models), receipt_ids=[row.id], fingerprint="fixture",
        model_digests={}, timeout=600, policy=CouncilPolicy.ALL, absolute=None,
        relative=None, apply_flags=False, prompt_version="4", scoring_version=version,
    )
    return b, run, row, path


async def due(w, run_id, *, expire=False):
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE benchmark_tasks SET retry_at=clock_timestamp()-interval '1 second'"
            + (", leased_until=clock_timestamp()-interval '1 second'" if expire else "")
            + " WHERE run_id=$1", run_id,
        )


async def saved(w, run_id):
    async with w.pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM benchmark_tasks WHERE run_id=$1", run_id)


async def test_reservations_do_not_spend_attempts_and_begin_is_once_per_fence(workflow_db, tmp_path):
    w, scope = workflow_db
    b, run, _, _ = await queue(w, scope, tmp_path)
    for _ in range(5):
        task = await b.lease_task()
        assert task["status"] == "running" and task["claim_token"] is not None
        assert task["attempts"] == task["inference_attempt_count"] == 0
        assert task["inference_started"] is False
        assert await b.retry_task(task, waiting=True)
        assert (await b.summaries(run.id))[0].completed == 0
        await due(w, run.id)
    task = await b.lease_task()
    begun = await b.begin_attempt(task)
    assert begun["inference_attempt_count"] == 1 and begun["inference_started"]
    assert await b.begin_attempt(task) is None
    assert (await saved(w, run.id))["inference_attempt_count"] == 1


@pytest.mark.parametrize("delays,cap,want", [((30, 60), 300, (30, 60)), ((70, 90), 45, (45, 45))])
async def test_actual_retry_schedule_and_budget(workflow_db, tmp_path, delays, cap, want):
    w, scope = workflow_db
    settings = WorkerSettings(retry_backoff_seconds=delays, retry_backoff_cap_seconds=cap)
    b, run, _, _ = await queue(w, scope, tmp_path, settings=settings)
    for count in (1, 2, 3):
        task = await b.begin_attempt(await b.lease_task())
        assert task["inference_attempt_count"] == count
        if count == 3:
            assert not await b.retry_task(task)
            assert await b.finish_task(task, success=False)
            break
        async with w.pool.acquire() as conn:
            before = await conn.fetchval("SELECT clock_timestamp()")
        assert await b.retry_task(task)
        current = await saved(w, run.id)
        assert current["claim_token"] is None and current["leased_until"] is None
        assert abs((current["retry_at"] - before).total_seconds() - want[count-1]) < 1
        assert (await b.summaries(run.id))[0].completed == 0
        await due(w, run.id)
    assert await b.lease_task() is None
    assert (await b.summaries(run.id))[0].failed == 1


@pytest.mark.parametrize("stale", ["wrong_token", "expired", "reclaimed", "released"])
async def test_stale_begin_retry_finish_leave_current_claim_untouched(workflow_db, tmp_path, stale):
    w, scope = workflow_db
    b, run, _, _ = await queue(w, scope, tmp_path)
    task = dict(await b.lease_task())
    if stale == "wrong_token":
        task["claim_token"] = uuid4()
    elif stale in {"expired", "reclaimed"}:
        await due(w, run.id, expire=True)
        if stale == "reclaimed":
            new = await b.lease_task()
            assert new["claim_token"] != task["claim_token"]
    else:
        assert await b.retry_task(task, waiting=True)
    before = await saved(w, run.id)
    assert await b.begin_attempt(task) is None
    assert not await b.retry_task(task)
    assert not await b.finish_task(task, success=True, score=1, result={"sentinel": True})
    assert await saved(w, run.id) == before
    assert (await b.summaries(run.id))[0].completed == 0
    assert await b.results(run.id) == []


async def wait_blocked(pool, count):
    # Observe actual server lock contention; no timing sleeps masquerading as overlap.
    async with asyncio.timeout(5):
        while True:
            async with pool.acquire() as conn:
                n = await conn.fetchval("""SELECT count(*) FROM pg_stat_activity
                    WHERE datname=current_database() AND wait_event_type='Lock'""")
            if n >= count:
                return
            await asyncio.sleep(0)


@pytest.mark.parametrize("operation", ["begin", "retry", "finish"])
async def test_fence_uses_current_clock_after_waiting_for_lock(workflow_db, tmp_path, operation):
    w, scope = workflow_db
    b, run, _, _ = await queue(w, scope, tmp_path)
    task = await b.lease_task()
    blocker = await w.pool.acquire()
    transaction = blocker.transaction()
    await transaction.start()
    await blocker.fetchrow("SELECT * FROM benchmark_tasks WHERE run_id=$1 FOR UPDATE", run.id)
    async def mutate():
        if operation == "begin":
            return await b.begin_attempt(task)
        if operation == "retry":
            return await b.retry_task(task)
        return await b.finish_task(task, success=True)
    work = asyncio.create_task(mutate())
    try:
        await wait_blocked(w.pool, 1)
        await blocker.execute("UPDATE benchmark_tasks SET leased_until=clock_timestamp()-interval '1 second' WHERE run_id=$1", run.id)
        await transaction.commit()
        assert not await work
        assert (await saved(w, run.id))["inference_attempt_count"] == 0
        assert (await b.summaries(run.id))[0].completed == 0
    finally:
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
        if blocker.is_in_transaction():
            await transaction.rollback()
        await w.pool.release(blocker)


@pytest.mark.parametrize("different_models", [False, True])
async def test_concurrent_finish_counts_once_and_completes_last_models(workflow_db, tmp_path, different_models):
    w, scope = workflow_db
    b, run, _, _ = await queue(w, scope, tmp_path, models=("m", "n") if different_models else ("m",))
    first = await b.begin_attempt(await b.lease_task())
    second = await b.begin_attempt(await b.lease_task()) if different_models else first
    works = []
    try:
        async with w.pool.acquire() as blocker, blocker.transaction():
            await blocker.fetchval("SELECT id FROM benchmark_runs WHERE id=$1 FOR UPDATE", run.id)
            works = [asyncio.create_task(b.finish_task(t, success=True, score=1, result={"version": "old"})) for t in (first, second)]
            await wait_blocked(w.pool, 2)
        assert sum(await asyncio.gather(*works)) == (2 if different_models else 1)
    finally:
        for work in works:
            if not work.done():
                work.cancel()
        await asyncio.gather(*works, return_exceptions=True)
    assert (await b.run(run.id)).status == "completed"
    assert sum(s.completed for s in await b.summaries(run.id)) == (2 if different_models else 1)
    assert len(await b.results(run.id)) == (2 if different_models else 1)


@pytest.mark.parametrize("failure_table", ["benchmark_results", "benchmark_summaries"])
async def test_finish_rollback_preserves_claim_and_no_partial_counts(workflow_db, tmp_path, failure_table):
    w, scope = workflow_db
    b, run, _, _ = await queue(w, scope, tmp_path)
    task = await b.begin_attempt(await b.lease_task())
    async with w.pool.acquire() as conn:
        await conn.execute("CREATE FUNCTION refuse_finish() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected failure'; END $$")
        await conn.execute(f"CREATE TRIGGER refuse_finish BEFORE INSERT OR UPDATE ON {failure_table} FOR EACH ROW EXECUTE FUNCTION refuse_finish()")
    with pytest.raises(asyncpg.RaiseError, match="injected failure"):
        await b.finish_task(task, success=True, result={"version": "old"})
    assert dict(await saved(w, run.id)) == dict(task)
    assert await b.results(run.id) == []
    assert (await b.summaries(run.id))[0].completed == 0
    async with w.pool.acquire() as conn:
        await conn.execute(f"DROP TRIGGER refuse_finish ON {failure_table}")
    assert await b.finish_task(task, success=True, result={"version": "old"})
    assert not await b.finish_task(task, success=True, result={"version": "old"})


async def test_lease_begin_retry_do_not_invert_finisher_run_task_lock_order(workflow_db, tmp_path):
    w, scope = workflow_db
    b, run, _, _ = await queue(w, scope, tmp_path, models=("m", "n"))
    first = await b.begin_attempt(await b.lease_task())
    finish = None
    try:
        async with w.pool.acquire() as blocker, blocker.transaction():
            await blocker.fetchval("SELECT 1 FROM benchmark_tasks WHERE run_id=$1 AND model_id=$2 FOR UPDATE", run.id, first["model_id"])
            finish = asyncio.create_task(b.finish_task(first, success=True))
            await wait_blocked(w.pool, 1)  # finisher owns run and waits on first task
            async with asyncio.timeout(2):
                second = await b.lease_task()  # skip locked first, never wait on run
                assert second["model_id"] != first["model_id"]
                begun = await b.begin_attempt(second)
                assert begun["inference_attempt_count"] == 1
                assert await b.retry_task(begun)
        assert await finish
    finally:
        if finish is not None:
            if not finish.done():
                finish.cancel()
            await asyncio.gather(finish, return_exceptions=True)
    await due(w, run.id)
    assert await b.finish_task(await b.lease_task(), success=False)
    assert (await b.run(run.id)).status == "completed"


async def test_expired_reservation_is_free_but_expired_begun_attempt_is_spent(workflow_db, tmp_path):
    w, scope = workflow_db
    b, run, _, _ = await queue(w, scope, tmp_path, settings=WorkerSettings(claim_lease_seconds=1200))
    first = await b.lease_task()
    async with w.pool.acquire() as conn:
        remaining = await conn.fetchval("SELECT extract(epoch FROM leased_until-clock_timestamp()) FROM benchmark_tasks WHERE run_id=$1", run.id)
    assert 1195 < remaining <= 1200
    await due(w, run.id, expire=True)
    second = await b.lease_task()
    assert second["inference_attempt_count"] == 0 and second["claim_token"] != first["claim_token"]
    await b.begin_attempt(second)
    await due(w, run.id, expire=True)
    third = await b.lease_task()
    assert third["inference_attempt_count"] == 1 and not third["inference_started"]
    assert (await b.begin_attempt(third))["inference_attempt_count"] == 2
