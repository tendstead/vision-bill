"""Attempt-local asyncpg lifecycle without unbounded or shielded cleanup.

Only callers supplying an absolute loop-clock deadline use this adapter. The
ordinary API path keeps asyncpg's public contexts. Private pool coupling lives
here because public acquire can await close after setup failure, and public
release shields a new task. Neither can enforce cancellation during cleanup.
The adapter is checked against the locked asyncpg version and fails closed on
an upgrade until its lifecycle tests have been reviewed.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any, TypeVar

import asyncpg

T = TypeVar("T")


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError("database_attempt_deadline")
    return remaining


async def _within(deadline: float, operation: Callable[[], Awaitable[T]]) -> T:
    _remaining(deadline)  # Do not even start COMMIT/ROLLBACK after expiry.
    async with asyncio.timeout_at(deadline):
        return await operation()


def _discard(holder: Any, ownership: asyncio.Future[None]) -> None:
    # Connection.terminate() aborts the transport synchronously. Its normal
    # callback detaches the proxy and returns the slot. Explicit release is
    # idempotent, also covering connect/init failures before a proxy exists.
    if holder._in_use is not ownership:
        return  # Disconnect already returned this lease; a successor may own it.
    if holder._con is not None:
        with suppress(Exception):
            holder._con.terminate()
    holder._release_on_close()


async def _acquire(pool: asyncpg.Pool, deadline: float) -> tuple[Any, asyncio.Future[None]]:
    if asyncpg.__version__ != "0.31.0" or not isinstance(pool, asyncpg.Pool):
        raise RuntimeError("Unsupported asyncpg deadline adapter; review locked lifecycle")
    pool._check_init()
    if pool._closing:
        raise asyncpg.InterfaceError("pool is closing")
    holder = await _within(deadline, pool._queue.get)
    if not isinstance(holder, asyncpg.pool.PoolConnectionHolder):
        pool._queue.put_nowait(holder)
        raise TypeError("Unsupported asyncpg pool holder")
    # Own the slot during connect/init/setup as well as during queries. No
    # other borrower or whole-pool state is terminated on this task's timeout.
    ownership: asyncio.Future[None] = pool._loop.create_future()
    holder._in_use = ownership
    holder._maybe_cancel_inactive_callback()
    try:
        async with asyncio.timeout_at(deadline):
            _remaining(deadline)
            if holder._con is None or holder._con.is_closed() or holder._generation != pool._generation:
                if holder._con is not None:
                    holder._con.terminate()
                    holder._con = None
                kwargs = dict(pool._connect_kwargs)
                kwargs["timeout"] = min(kwargs.get("timeout") or 60, _remaining(deadline))
                holder._con = await pool._connect(
                    *pool._connect_args, loop=pool._loop,
                    connection_class=pool._connection_class, record_class=pool._record_class,
                    **kwargs,
                )
                if not isinstance(holder._con, pool._connection_class):
                    raise TypeError("Unexpected asyncpg connection class")
                holder._generation = pool._generation
                # Retain the connection BEFORE init so cancellation discards it
                # instead of entering asyncpg's unbounded graceful-close handler.
                if pool._init is not None:
                    await pool._init(holder._con)
            holder._proxy = asyncpg.pool.PoolConnectionProxy(holder, holder._con)
            if holder._setup is not None:
                await holder._setup(holder._proxy)
            _remaining(deadline)
            if holder._in_use is not ownership:
                raise asyncpg.ConnectionDoesNotExistError("connection lost during acquisition")
            return holder, ownership
    except BaseException:
        _discard(holder, ownership)
        raise


async def _release(holder: Any, ownership: asyncio.Future[None], raw: Any, deadline: float) -> None:
    # PoolConnectionHolder.release reads mutable holder._con again after its
    # awaits, including in its error handler. A disconnect can return the slot
    # during those awaits. Keep every I/O operation on the original raw connection
    # and touch holder state only while the captured lease is still current.
    holder._timeout = None
    if raw.is_closed():
        _discard(holder, ownership)
        return
    if raw._protocol.queries_count >= holder._max_queries or holder._generation != holder._pool._generation:
        await raw.close(timeout=_remaining(deadline))
        return
    if raw._protocol._is_cancelling():
        await raw._protocol._wait_for_cancellation()
    if holder._in_use is not ownership:
        return
    if holder._pool._reset is not None:
        await raw._reset()
        if holder._in_use is not ownership:
            return
        await holder._pool._reset(raw)
    else:
        await raw.reset(timeout=_remaining(deadline))
    if holder._in_use is ownership:
        holder._release()
        holder._setup_inactive_callback()


@asynccontextmanager
async def connection(
    pool: asyncpg.Pool, *, deadline: float | None = None, transaction: bool = False,
) -> AsyncIterator[asyncpg.Connection]:
    if deadline is None:
        async with pool.acquire() as conn:
            if transaction:
                async with conn.transaction():
                    yield conn
            else:
                yield conn
        return

    holder, ownership = await _acquire(pool, deadline)
    conn = holder._proxy
    failed = False
    try:
        if transaction:
            tx = conn.transaction()
            try:
                await _within(deadline, tx.start)
            except BaseException:
                _discard(holder, ownership)
                raise
            try:
                async with asyncio.timeout_at(deadline):
                    yield conn
            except BaseException:
                try:
                    if holder._in_use is ownership:
                        await _within(deadline, tx.rollback)
                except BaseException:  # noqa: BLE001 - discard, preserve original body error
                    _discard(holder, ownership)
                raise  # Keep the original cancellation/error, never a closed-tx error.
            else:
                try:
                    await _within(deadline, tx.commit)
                except BaseException:
                    _discard(holder, ownership)
                    raise
        else:
            async with asyncio.timeout_at(deadline):
                yield conn
    except BaseException:
        failed = True
        raise
    finally:
        if holder._in_use is ownership:
            try:
                _remaining(deadline)
                conn._con._on_release()
                # No shielded task; all release I/O stays on the original lease.
                await _within(deadline, lambda: _release(holder, ownership, conn._con, deadline))
            except BaseException:
                _discard(holder, ownership)
                if not failed:
                    raise
