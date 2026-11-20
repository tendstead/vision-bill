"""Deadline discard returns pool capacity and retains atomic claim recovery on PostgreSQL."""

import asyncio

import asyncpg
import pytest

from tests.e2e import test_benchmark_worker_e2e as worker_tests
from vision_bill.config import WorkerSettings
from vision_bill.provider.db.benchmark_db import BenchmarkDB

postgres, db, workflow_db = worker_tests.postgres, worker_tests.db, worker_tests.workflow_db
pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("phase", ["rollback", "release"])
async def test_deadline_discards_only_owned_connection_and_recovers_real_claim(workflow_db, tmp_path, phase):
    w, scope = workflow_db
    service, run, _, _, inference, _ = await worker_tests.prepare(
        w, scope, tmp_path, kind="manifest", worker=WorkerSettings(attempt_timeout_seconds=1))
    slow_reset = phase == "release"
    reset_active = 0
    rollback_started = 0

    class SlowRollbackConnection(asyncpg.Connection):
        async def execute(self, query, *args, **kwargs):
            nonlocal rollback_started
            if query == "ROLLBACK;":
                rollback_started += 1
                await asyncio.sleep(.3)
            return await super().execute(query, *args, **kwargs)

    async def reset(conn):
        nonlocal reset_active
        if slow_reset:
            reset_active += 1
            try:
                await asyncio.sleep(1.3)
            finally:
                reset_active -= 1

    pool = await asyncpg.create_pool(w._settings.pg_dsn, min_size=0, max_size=2,
                                    connection_class=SlowRollbackConnection, reset=reset)
    peer = await pool.acquire()
    peer_pid = await peer.fetchval("SELECT pg_backend_pid()")
    service._db = BenchmarkDB(pool, w._receipts, worker=service._settings)
    # A separate pool/borrower holds the contested lock. No operator resources.
    blocker = await w.pool.acquire()
    tx = blocker.transaction()
    await tx.start()
    try:
        if phase == "rollback":
            for _ in range(3):
                await service._db.begin_attempt(await service._db.lease_task())
                await worker_tests.claims.due(w, run.id, expire=True)
            await blocker.fetchval("SELECT id FROM benchmark_runs WHERE id=$1 FOR UPDATE", run.id)
        claim = dict(await service._db.lease_task())
        started = asyncio.get_running_loop().time()
        work = asyncio.create_task(service._execute(claim))
        if phase == "rollback":
            await worker_tests.claims.wait_blocked(w.pool, 1)
        await asyncio.wait_for(work, 1.2)
        assert asyncio.get_running_loop().time() - started < 1.15
        assert reset_active == 0
        slow_reset = False
        assert rollback_started == 0  # expiry skips graceful rollback and aborts instead
        assert await peer.fetchval("SELECT pg_backend_pid()") == peer_pid
        assert pool._queue.qsize() == 1  # peer owns the other slot
        current = await worker_tests.claims.saved(w, run.id)
        assert current["status"] == "running"
        assert current["inference_attempt_count"] == (3 if phase == "rollback" else 0)
        assert (await service._db.summaries(run.id))[0].completed == 0
        await tx.rollback()
        slow_reset = False
        await worker_tests.claims.due(w, run.id, expire=True)
        recovered = dict(await service._db.lease_task())
        await service._execute(recovered)
        await service._execute(recovered)  # no duplicate terminal count
        summary = (await service._db.summaries(run.id))[0]
        assert summary.completed == 1
        assert summary.failed == (phase == "rollback")
        assert inference.await_count == (0 if phase == "rollback" else 1)
        assert len(await service._db.results(run.id)) == 1
        assert pool._queue.qsize() == 1
    finally:
        if blocker.is_in_transaction():
            await tx.rollback()
        await w.pool.release(blocker)
        slow_reset = False
        await pool.release(peer)
        await pool.close()
