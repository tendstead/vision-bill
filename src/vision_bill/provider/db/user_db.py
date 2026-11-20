import logging
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import asyncpg

from ...config import AuthSettings, PGSettings
from ...security.models import User
from ...security.password import verify_password

# ── SQL (DML; DDL lives in alembic/versions/0001_initial_schema.py) ──

CREATE_USER_SQL = """
    INSERT INTO users (username, hashed_password, is_admin)
    VALUES ($1, $2, $3)
    RETURNING *
"""

GET_USER_BY_USERNAME_SQL = "SELECT * FROM users WHERE username = $1"
GET_USER_BY_ID_SQL = "SELECT * FROM users WHERE id = $1"
COUNT_USERS_SQL = "SELECT count(*) AS count FROM users"
SET_ORPHAN_OWNER_RECEIPTS_SQL = "UPDATE receipts SET user_id = $1 WHERE user_id IS NULL"
SET_ORPHAN_OWNER_IMAGES_SQL = "UPDATE images SET user_id = $1 WHERE user_id IS NULL"


logger = logging.getLogger(__name__)


class UserDB:
    """Owns the asyncpg pool and all SQL for the users table and row ownership.

    Mirrors :class:`ReceiptDB`'s pool lifecycle. The effective ``can_see_all``
    privilege is deliberately *not* resolved here — it is derived by the API
    layer (``security.dependencies``) from the global settings flag.
    """

    def __init__(self, settings: PGSettings):
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    # ── Connection pool lifecycle ────────────────────────────────────
    async def init_db(self) -> None:
        """Create the connection pool."""
        if self._pool is not None:
            logger.warning("User database pool already initialised - skipping")
            return

        dsn = self._settings.pg_dsn
        logger.info("Creating asyncpg user connection pool")
        self._pool = await asyncpg.create_pool(dsn=dsn)

    async def destroy_db(self) -> None:
        """Close the connection pool and release all resources."""
        if self._pool is not None:
            logger.info("Closing user database connection pool")
            await self._pool.close()
            self._pool = None

    @property
    def is_ready(self) -> bool:
        """Whether the connection pool has been initialised."""
        return self._pool is not None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("User database pool not initialised. Call init_db() first.")
        return self._pool

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _user_from_record(row: Mapping[str, Any]) -> User:
        """Map a raw users row to a ``User`` (dropping the password hash)."""
        d = dict(row)
        return User(
            id=d["id"],
            username=d["username"],
            is_admin=bool(d["is_admin"]),
            can_see_all=False,  # resolved by the caller from the settings flag
        )

    # ── DML ──────────────────────────────────────────────────────────

    async def create_user(
        self, username: str, hashed_password: str, is_admin: bool = False
    ) -> User:
        """Insert a new user and return it."""
        logger.info("Creating user '%s' (is_admin=%s)", username, is_admin)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(CREATE_USER_SQL, username, hashed_password, is_admin)
        return self._user_from_record(row)

    async def get_user_by_username(self, username: str) -> User | None:
        """Fetch a user by username, or ``None`` when absent."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(GET_USER_BY_USERNAME_SQL, username)
        if row is None:
            return None
        return self._user_from_record(row)

    async def get_user_by_id(self, user_id: UUID) -> User | None:
        """Fetch a user by primary key, or ``None`` when absent."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(GET_USER_BY_ID_SQL, user_id)
        if row is None:
            return None
        return self._user_from_record(row)

    async def verify_credentials(
        self, username: str, password: str, auth: AuthSettings
    ) -> User | None:
        """Return the user when ``username``/``password`` are valid, else ``None``.

        The peppered Argon2id hash is compared internally and never leaves the
        DB layer; only the verified :class:`User` is returned.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(GET_USER_BY_USERNAME_SQL, username)
        if row is None:
            return None
        if not verify_password(row["hashed_password"], password, auth):
            return None
        return self._user_from_record(row)

    async def count_users(self) -> int:
        """Return the total number of user rows."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(COUNT_USERS_SQL)
        return int(row["count"]) if row is not None else 0

    async def set_owner_of_orphan_rows(self, user_id: UUID) -> None:
        """Assign legacy receipts/images with no owner to ``user_id`` (idempotent)."""
        logger.info("Backfilling orphan receipts/images to user %s", user_id)
        async with self.pool.acquire() as conn:
            await conn.execute(SET_ORPHAN_OWNER_RECEIPTS_SQL, user_id)
            await conn.execute(SET_ORPHAN_OWNER_IMAGES_SQL, user_id)
