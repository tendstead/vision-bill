"""Installed asyncpg transaction/pool lifecycle with only wire I/O simulated."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest

from vision_bill.config import WorkerSettings
from vision_bill.provider.db.benchmark_db import BenchmarkDB
from vision_bill.service.benchmark_service import BenchmarkService


class WireConnection(asyncpg.Connection):
    """Real Transaction and PoolConnectionHolder; controllable network waits."""

    def __init__(self, phase):
        self.phase = phase
        self._loop = asyncio.get_running_loop()
        self._pool_release_ctr = 0
        self._top_xact = None
        self._proxy = None
        self._listeners = {}
        self._log_listeners = set()
        self._protocol = SimpleNamespace(is_in_transaction=lambda: False, queries_count=0,
                                         _is_cancelling=lambda: False)
        self.closed = False
        self.commands = []
        self.active_io = 0
        self.terminated = False
        self.fetch_count = 0
        self.commit_at = None

    async def pause(self, seconds):
        self.active_io += 1
        try:
            await asyncio.sleep(seconds)
        finally:
            self.active_io -= 1

    def is_closed(self):
        return self.closed

    def terminate(self):
        self.closed = self.terminated = True
        if self._proxy is not None:
            self._proxy._holder._release_on_close()

    async def close(self, *, timeout=None):
        # Connection close after a failed acquire/setup is network I/O too.
        if self.phase in {"acquire_setup", "acquire_init"}:
            await self.pause(.3)
        self.terminate()

    async def _reset(self):
        pass

    async def reset(self, *, timeout=None):
        if self.phase in {"release", "read_release"}:
            await self.pause(.3)

    async def execute(self, sql, *args, **kwargs):
        self.commands.append(sql)
        if sql == "COMMIT;":
            self.commit_at = self._loop.time()
        if sql == "ROLLBACK;":
            await self.pause(.3)
        return "DELETE 1" if sql.startswith("DELETE") else "UPDATE 1"

    async def fetchrow(self, sql, *args, **kwargs):
        self.fetch_count += 1
        if self.phase == "read_release":
            await self.pause(.02)
            return None
        if self.phase == "rollback":
            await self.pause(10)  # outer deadline cancels the exhausted-run lock
        elif self.phase == "release" and self.fetch_count == 1:
            await self.pause(.9)  # release starts just before the deadline
        return {"inference_attempt_count": 1}

    async def fetchval(self, sql, *args, **kwargs):
        await self.pause(.02)
        return 600 if "request_timeout_seconds" in sql else None


@pytest.mark.parametrize("phase", ["rollback", "release", "acquire_setup", "acquire_init"])
async def test_exhausted_worker_bounds_actual_asyncpg_cleanup_and_preserves_pool_capacity(phase):
    made = []
    async def connect(*args, **kwargs):
        conn = WireConnection(phase)
        made.append(conn)
        return conn
    async def setup(conn):
        if phase == "acquire_setup":
            await conn.pause(10)
    async def init(conn):
        if phase == "acquire_init":
            await conn.pause(10)
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect,
                                    connection_class=WireConnection, setup=setup, init=init)
    service = object.__new__(BenchmarkService)
    service._settings = WorkerSettings(attempt_timeout_seconds=1)
    service._receipts = SimpleNamespace(pool=pool)
    service._db = BenchmarkDB(pool)
    service._db.run = AsyncMock(return_value=SimpleNamespace(id=uuid4()))
    service._db.manifest = AsyncMock(return_value=None)
    service._timeout = AsyncMock(return_value=600)
    task = {"run_id": uuid4(), "model_id": "m", "receipt_id": uuid4(), "claim_token": uuid4(),
            "inference_attempt_count": 3, "inference_started": False, "attempts": 0}
    started = asyncio.get_running_loop().time()
    try:
        await service._execute(task)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 1.15, f"budget=1s, elapsed={elapsed:.3f}s, phase={phase}"
        assert made[0].terminated
        assert made[0].active_io == 0, "no shielded background cleanup may escape"
        assert made[0].commit_at is None or made[0].commit_at < started + 1
        # A discarded borrowed connection returns its slot, not the whole pool.
        async with asyncio.timeout(.2):
            pool._setup = None
            pool._init = None
            pool._holders[0]._setup = None
            conn = await pool.acquire()
            assert conn._con is not made[0]
            conn.terminate()
    finally:
        pool.terminate()


@pytest.mark.parametrize("phase", ["read", "body", "start", "rollback", "commit", "setup", "release",
                                   "read_return", "setup_return", "release_return"])
@pytest.mark.parametrize("successor_phase", ["public", "connect", "setup"])
async def test_disconnect_cleanup_never_touches_same_holder_successor(phase, successor_phase):
    from vision_bill.provider.db.deadline import connection
    made = []
    disconnected, unwind, successor_ready, finish_successor = (asyncio.Event() for _ in range(4))

    async def lose_connection(raw):
        # Installed asyncpg's disconnect callback releases the holder before
        # the interrupted operation delivers its error to the old borrower.
        raw.terminate()
        disconnected.set()
        await unwind.wait()
        if not phase.endswith("_return"):
            raise RuntimeError("original connection lost")

    class DisconnectConnection(WireConnection):
        async def reset(self, *, timeout=None):
            if self is made[0] and phase in {"release", "release_return"}:
                await lose_connection(self)
            await super().reset(timeout=timeout)

        async def execute(self, sql, *args, **kwargs):
            target = {"start": "BEGIN;", "rollback": "ROLLBACK;", "commit": "COMMIT;"}.get(phase)
            if self is made[0] and sql == target:
                await lose_connection(self)
            return await super().execute(sql, *args, **kwargs)

    async def connect(*args, **kwargs):
        raw = DisconnectConnection("responsive")
        made.append(raw)
        if len(made) == 2 and successor_phase == "connect":
            successor_ready.set()
            await finish_successor.wait()
        return raw

    async def setup(proxy):
        if proxy._con is made[0] and phase in {"setup", "setup_return"}:
            await lose_connection(proxy._con)
        elif proxy._con is made[-1] and len(made) == 2 and successor_phase == "setup":
            successor_ready.set()
            await finish_successor.wait()

    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect,
                                    connection_class=DisconnectConnection, setup=setup)
    deadline = asyncio.get_running_loop().time() + 2

    async def original():
        async with connection(pool, deadline=deadline, transaction=phase not in {"read", "setup", "read_return", "setup_return"}) as proxy:
            if phase in {"read", "body", "read_return"}:
                await lose_connection(proxy._con)
            elif phase == "rollback":
                raise ValueError("original body failure")

    async def successor():
        context = pool.acquire() if successor_phase == "public" else connection(pool, deadline=deadline)
        async with context as proxy:
            successor_ready.set()
            await finish_successor.wait()
            assert not proxy.is_closed()

    first = asyncio.create_task(original())
    second = None
    try:
        async with asyncio.timeout(2):
            await disconnected.wait()
            original_holder = pool._holders[0]
            second = asyncio.create_task(successor())
            await successor_ready.wait()
            successor_ownership = original_holder._in_use
            assert successor_ownership is not None and pool._queue.qsize() == 0
            unwind.set()
            if phase in {"read_return", "release_return"}:
                await first
            elif phase == "setup_return":
                with pytest.raises(asyncpg.ConnectionDoesNotExistError, match="connection lost"):
                    await first
            else:
                expected = ValueError if phase == "rollback" else RuntimeError
                with pytest.raises(expected, match="original"):
                    await first
            assert not made[1].terminated, "stale cleanup terminated the successor connection"
            assert original_holder._in_use is successor_ownership, "stale cleanup returned the successor slot"
            assert pool._queue.qsize() == 0
            finish_successor.set()
            await second
            assert pool._queue.qsize() == 1
            async with pool.acquire() as reused:
                assert reused._con is made[1]
    finally:
        unwind.set()
        finish_successor.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
        pool.terminate()
        for raw in made:
            raw.terminate()


@pytest.mark.parametrize("operation", ["run", "manifest", "timeout", "digests", "legacy_receipt"])
async def test_read_preflight_release_uses_its_explicit_deadline(operation, tmp_path):
    from vision_bill.config import ImageSettings, PGSettings
    from vision_bill.service.receipt_service import ReceiptService
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("read_release"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path), tmp_dir=str(tmp_path)), PGSettings(user="test", password="test"), AsyncMock())
    receipts._db._pool = pool
    service = BenchmarkService(AsyncMock(), receipts)
    run = SimpleNamespace(id=uuid4())
    started = asyncio.get_running_loop().time()
    deadline = started + .1
    calls = {
        "run": lambda: service._db.run(run.id, deadline=deadline),
        "manifest": lambda: service._db.manifest(run.id, deadline=deadline),
        "timeout": lambda: service._timeout(run, deadline=deadline),
        "digests": lambda: service._run_digests(run, deadline=deadline),
        "legacy_receipt": lambda: service._expected_receipt_and_path(run.id, deadline=deadline),
    }
    try:
        with pytest.raises(TimeoutError):
            await calls[operation]()
        assert asyncio.get_running_loop().time() - started < .2
        assert made[0].terminated and made[0].active_io == 0
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_bounded_normal_release_reuses_connection_and_detaches_old_proxy():
    from vision_bill.provider.db.deadline import connection
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    try:
        async with connection(pool, deadline=asyncio.get_running_loop().time()+1, transaction=True) as first:
            raw = first._con
            await first.execute("UPDATE harmless")
        assert first._con is None and not raw.terminated
        async with connection(pool, deadline=asyncio.get_running_loop().time()+1) as second:
            assert second._con is raw and second is not first
        assert second._con is None and len(made) == 1 and pool._queue.qsize() == 1
    finally:
        pool.terminate()


@pytest.mark.parametrize("reason", ["query_limit", "expired_generation"])
async def test_bounded_release_retires_connection_at_driver_reuse_limits(reason):
    from vision_bill.provider.db.deadline import connection
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, max_queries=1,
                                    connect=connect, connection_class=WireConnection)
    try:
        async with connection(pool, deadline=asyncio.get_running_loop().time()+1) as proxy:
            if reason == "query_limit":
                proxy._con._protocol.queries_count = 1
            else:
                await pool.expire_connections()
        assert made[0].terminated and proxy._con is None and pool._queue.qsize() == 1
        async with connection(pool, deadline=asyncio.get_running_loop().time()+1) as replacement:
            assert replacement._con is not made[0]
        assert len(made) == 2 and pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_slow_rollback_preserves_original_error_and_leaves_no_cleanup_task():
    from vision_bill.provider.db.deadline import connection
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    started = asyncio.get_running_loop().time()
    try:
        with pytest.raises(ValueError, match="original failure"):
            async with connection(pool, deadline=started+.1, transaction=True):
                raise ValueError("original failure")
        assert asyncio.get_running_loop().time() - started < .2
        assert made[0].terminated and made[0].active_io == 0
        assert "ROLLBACK;" in made[0].commands and "COMMIT;" not in made[0].commands
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_external_cancellation_is_preserved_during_bounded_rollback():
    from vision_bill.provider.db.deadline import connection
    made = []
    entered = asyncio.Event()
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    started = asyncio.get_running_loop().time()
    async def work():
        async with connection(pool, deadline=started+.1, transaction=True):
            entered.set()
            await asyncio.Event().wait()
    task = asyncio.create_task(work())
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and asyncio.get_running_loop().time() - started < .2
        assert made[0].terminated and made[0].active_io == 0
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_unsupported_pool_fails_closed_without_public_cleanup_fallback():
    from vision_bill.provider.db.deadline import connection
    class UnsupportedPool:
        def acquire(self):
            raise AssertionError("unbounded fallback must never run")
    with pytest.raises(RuntimeError, match="Unsupported asyncpg deadline adapter"):
        async with connection(UnsupportedPool(), deadline=asyncio.get_running_loop().time()+1):
            pytest.fail("unsupported pool cannot be borrowed")


async def test_cancelled_pool_queue_wait_does_not_disrupt_another_borrower():
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect,
                                    connection_class=WireConnection)
    borrowed = await pool.acquire()
    service = object.__new__(BenchmarkService)
    service._settings = WorkerSettings(attempt_timeout_seconds=1)
    service._db = BenchmarkDB(pool)
    # Exercise the actual first run-read acquisition, before inference or reserve.
    started = asyncio.get_running_loop().time()
    try:
        await service._execute({"run_id": uuid4()})
        assert asyncio.get_running_loop().time() - started < 1.15
        assert not borrowed.is_closed() and not made[0].terminated
        await pool.release(borrowed)
        async with asyncio.timeout(.2):
            reused = await pool.acquire()
            assert reused._con is made[0]
            await pool.release(reused)
    finally:
        pool.terminate()


async def test_benchmark_reads_include_current_receipt_scope():
    from contextlib import asynccontextmanager

    from vision_bill.model.workflow import AccessScope

    conn = SimpleNamespace(fetchrow=AsyncMock(return_value=None), fetch=AsyncMock(return_value=[]))

    @asynccontextmanager
    async def acquire():
        yield conn

    database = BenchmarkDB(SimpleNamespace(acquire=acquire))
    run_id, owner = uuid4(), uuid4()
    scope = AccessScope(user_id=owner)
    assert await database.run(run_id, scope=scope) is None
    assert await database.runs(scope=scope) == []
    status_sql, *status_args = conn.fetchrow.await_args.args
    list_sql, *list_args = conn.fetch.await_args.args
    for sql in (status_sql, list_sql):
        assert "jsonb_array_elements_text" in sql
        assert "FROM receipts" in sql and "r.user_id" in sql
    assert status_args == [run_id, False, owner]
    assert list_args == [False, owner]
