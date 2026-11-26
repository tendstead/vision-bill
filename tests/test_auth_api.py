"""Auth endpoint tests: real FastAPI app + TestClient, mocked DB and provider."""

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import vision_bill.main as main_module
import vision_bill.security.dependencies as deps_module
from vision_bill.api import auth as auth_module
from vision_bill.config import Settings
from vision_bill.provider.db import image_db as image_db_module
from vision_bill.provider.db import receipt_db as receipt_db_module
from vision_bill.provider.db import user_db as user_db_module
from vision_bill.provider.db import workflow_db as workflow_db_module
from vision_bill.provider.llm.base import LLMProvider
from vision_bill.security import create_token, hash_password
from vision_bill.security.models import User

AUTH_URL = "/api/v1/auth"
USER_ID = UUID("00000000-0000-4000-8000-000000000001")


def _make_pool(conn: AsyncMock) -> MagicMock:
    conn.transaction = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    pool.close = AsyncMock()
    return pool


def _make_provider() -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.check_connection = AsyncMock(return_value=False)
    provider.get_available_models = AsyncMock(return_value=[])
    return provider


def _user_row(
    id: UUID = USER_ID,
    username: str = "alice",
    is_admin: bool = False,
    hashed: str | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "username": username,
        "hashed_password": hashed if hashed is not None else "$argon2id$placeholder",
        "is_admin": is_admin,
        "created_at": datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
    }


class AuthContext(NamedTuple):
    client: TestClient
    conn: AsyncMock
    provider: MagicMock


@pytest.fixture
def auth_context(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Generator[AuthContext, None, None]:
    conn = AsyncMock()
    # init_db checks to_regclass (missing); the startup bootstrap reads count.
    conn.fetchrow = AsyncMock(return_value={"missing": False, "count": 0})
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()
    provider = _make_provider()

    fake_asyncpg = MagicMock()
    fake_asyncpg.create_pool = AsyncMock(return_value=_make_pool(conn))

    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(auth_module, "settings", settings)
    monkeypatch.setattr(deps_module, "settings", settings)
    monkeypatch.setattr(main_module, "get_llm_provider", lambda cfg: provider)
    monkeypatch.setattr(receipt_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(image_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(user_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(workflow_db_module, "asyncpg", fake_asyncpg)
    monkeypatch.setattr(main_module.BenchmarkService, "start", AsyncMock())
    monkeypatch.setattr(main_module.AnalysisScheduler, "start", AsyncMock())

    with TestClient(main_module.app) as client:
        yield AuthContext(client=client, conn=conn, provider=provider)


def _creds(username: str, password: str) -> dict[str, str]:
    return {"username": username, "password": password}


def test_register_sets_cookie(auth_context: AuthContext) -> None:
    ctx = auth_context
    ctx.conn.fetchrow = AsyncMock(side_effect=[None, _user_row(username="alice")])
    response = ctx.client.post(f"{AUTH_URL}/register", json=_creds("alice", "s3cret"))
    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "alice"
    assert body["is_admin"] is False
    assert ctx.client.cookies.get("vb_session") is not None


def test_register_duplicate_conflict(auth_context: AuthContext) -> None:
    ctx = auth_context
    ctx.conn.fetchrow = AsyncMock(return_value=_user_row(username="alice"))
    response = ctx.client.post(f"{AUTH_URL}/register", json=_creds("alice", "s3cret"))
    assert response.status_code == 409
    assert "already taken" in response.json()["detail"]


def test_register_disabled_forbidden(settings: Settings, auth_context: AuthContext) -> None:
    settings.auth.allow_registration = False
    response = auth_context.client.post(f"{AUTH_URL}/register", json=_creds("bob", "pw"))
    assert response.status_code == 403


def test_login_success_sets_cookie(auth_context: AuthContext, settings: Settings) -> None:
    ctx = auth_context
    hashed = hash_password("s3cret", settings.auth)
    ctx.conn.fetchrow = AsyncMock(return_value=_user_row(username="alice", hashed=hashed))
    response = ctx.client.post(f"{AUTH_URL}/login", json=_creds("alice", "s3cret"))
    assert response.status_code == 200
    assert response.json()["username"] == "alice"
    assert ctx.client.cookies.get("vb_session") is not None


def test_login_bad_password_401(auth_context: AuthContext, settings: Settings) -> None:
    ctx = auth_context
    hashed = hash_password("s3cret", settings.auth)
    ctx.conn.fetchrow = AsyncMock(return_value=_user_row(username="alice", hashed=hashed))
    response = ctx.client.post(f"{AUTH_URL}/login", json=_creds("alice", "wrong"))
    assert response.status_code == 401
    assert ctx.client.cookies.get("vb_session") is None


def test_login_unknown_user_401(auth_context: AuthContext) -> None:
    ctx = auth_context
    ctx.conn.fetchrow = AsyncMock(return_value=None)
    response = ctx.client.post(f"{AUTH_URL}/login", json=_creds("nobody", "pw"))
    assert response.status_code == 401


def test_me_with_valid_cookie(auth_context: AuthContext, settings: Settings) -> None:
    ctx = auth_context
    ctx.conn.fetchrow = AsyncMock(return_value=_user_row(username="alice", is_admin=True))
    token = create_token(USER_ID, settings.auth.session_max_age_seconds, settings.auth.secret_key)
    ctx.client.cookies.set(settings.auth.session_cookie_name, token)
    response = ctx.client.get(f"{AUTH_URL}/me")
    assert response.status_code == 200
    assert response.json()["username"] == "alice"
    assert response.json()["is_admin"] is True


def test_me_without_cookie_401(auth_context: AuthContext) -> None:
    response = auth_context.client.get(f"{AUTH_URL}/me")
    assert response.status_code == 401


def test_logout_clears_cookie(auth_context: AuthContext, settings: Settings) -> None:
    ctx = auth_context
    response = ctx.client.post(f"{AUTH_URL}/logout")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    # The endpoint instructs the client to drop the session cookie (expired).
    set_cookie = response.headers.get("set-cookie", "")
    assert settings.auth.session_cookie_name in set_cookie
    assert "Max-Age=0" in set_cookie


@pytest.mark.asyncio
async def test_bootstrap_creates_one_admin_then_noop(settings: Settings) -> None:
    settings.auth.bootstrap_username = "admin"
    settings.auth.bootstrap_password = "admin-pass"

    # First boot: empty DB -> exactly one admin created, orphans backfilled.
    empty_db = MagicMock()
    empty_db.count_users = AsyncMock(return_value=0)
    empty_db.create_user = AsyncMock(
        return_value=User(id=USER_ID, username="admin", is_admin=True)
    )
    empty_db.set_owner_of_orphan_rows = AsyncMock()
    await main_module.bootstrap_admin(empty_db, settings)
    empty_db.create_user.assert_awaited_once()
    args, kwargs = empty_db.create_user.await_args
    assert args[0] == "admin"
    assert kwargs["is_admin"] is True
    empty_db.set_owner_of_orphan_rows.assert_awaited_once_with(USER_ID)

    # Next boot: a user already exists -> no create, no backfill.
    populated_db = MagicMock()
    populated_db.count_users = AsyncMock(return_value=1)
    populated_db.create_user = AsyncMock()
    populated_db.set_owner_of_orphan_rows = AsyncMock()
    await main_module.bootstrap_admin(populated_db, settings)
    populated_db.create_user.assert_not_awaited()
    populated_db.set_owner_of_orphan_rows.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_skips_without_credentials(settings: Settings) -> None:
    empty_db = MagicMock()
    empty_db.count_users = AsyncMock(return_value=0)
    empty_db.create_user = AsyncMock()
    await main_module.bootstrap_admin(empty_db, settings)
    empty_db.create_user.assert_not_awaited()


EXPECTED_ACTOR_HEADER = "X-Vision-Bill-Expected-User"
OTHER_USER_ID = UUID("00000000-0000-4000-8000-000000000002")


def _authenticate_as(ctx: AuthContext, settings: Settings, *, admin: bool = False) -> None:
    ctx.conn.fetchrow = AsyncMock(return_value=_user_row(is_admin=admin))
    ctx.client.cookies.set(
        settings.auth.session_cookie_name,
        create_token(USER_ID, settings.auth.session_max_age_seconds, settings.auth.secret_key),
    )


@pytest.mark.parametrize("admin", [False, True])
def test_actor_intent_checks_cookie_before_private_reads(
    auth_context: AuthContext, settings: Settings, monkeypatch: pytest.MonkeyPatch, admin: bool
) -> None:
    ctx = auth_context
    settings.auth.admin_can_see_all = True
    _authenticate_as(ctx, settings, admin=admin)
    list_images = AsyncMock(return_value=[])
    monkeypatch.setattr(main_module.app.state.receipt_service, "list_images", list_images)

    stale = ctx.client.get("/api/v1/images", headers={EXPECTED_ACTOR_HEADER: str(OTHER_USER_ID)})

    assert stale.status_code == 409
    assert stale.json() == {"detail": "session_actor_changed"}
    list_images.assert_not_awaited()

    # Both deliberately unbound API clients and correctly bound requests use
    # the cookie user's normal ownership scope, including the admin setting.
    assert ctx.client.get("/api/v1/images").status_code == 200
    assert ctx.client.get("/api/v1/images", headers={EXPECTED_ACTOR_HEADER: str(USER_ID)}).status_code == 200
    assert list_images.await_count == 2
    for call in list_images.await_args_list:
        assert call.kwargs["user_id"] == USER_ID
        assert call.kwargs["can_see_all"] is admin


def test_actor_mismatch_blocks_upload_and_benchmark_mutations(
    auth_context: AuthContext, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = auth_context
    _authenticate_as(ctx, settings, admin=True)
    stale = {EXPECTED_ACTOR_HEADER: str(OTHER_USER_ID)}
    store = MagicMock(side_effect=AssertionError("upload reached file storage"))
    monkeypatch.setattr(main_module.app.state.image_service, "store_original", store)
    create = AsyncMock(side_effect=ValueError("run creation reached service"))
    qualify = AsyncMock(side_effect=ValueError("reference qualification reached service"))
    service = main_module.app.state.benchmark_service
    monkeypatch.setattr(service, "create_run", create)
    monkeypatch.setattr(service, "qualify_reference", qualify)

    responses = [
        ctx.client.post(
            "/api/v1/images", headers=stale, files={"receipt": ("a.png", b"bad image", "image/png")}
        ),
        ctx.client.post("/api/v1/benchmarks", headers=stale, json={"model_ids": ["stub"]}),
        ctx.client.post(
            f"/api/v1/benchmarks/references/{USER_ID}",
            headers=stale,
            json={"expected_review_revision": 1, "expected_document_sha256": "a" * 64},
        ),
    ]
    for response in responses:
        assert response.status_code == 409
        assert response.json() == {"detail": "session_actor_changed"}
    assert (store.call_count, create.await_count, qualify.await_count) == (0, 0, 0)


@pytest.mark.parametrize(
    ("header", "query", "status"),
    [
        (None, str(OTHER_USER_ID), 409),
        (str(USER_ID), str(OTHER_USER_ID), 400),
        ("not-a-uuid", str(USER_ID), 400),
        (str(USER_ID), "", 400),
        (f"{USER_ID},{USER_ID}", None, 400),
        (None, f"{USER_ID}&vb_expected_user={USER_ID}", 400),
    ],
)
def test_invalid_or_stale_media_actor_is_rejected_before_image_lookup(
    auth_context: AuthContext, settings: Settings, monkeypatch: pytest.MonkeyPatch,
    header: str | None, query: str | None, status: int
) -> None:
    ctx = auth_context
    _authenticate_as(ctx, settings, admin=True)
    lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(main_module.app.state.receipt_service, "get_image_by_id", lookup)

    response = ctx.client.get(
        f"/api/v1/images/{USER_ID}/file" + (f"?vb_expected_user={query}" if query is not None else ""),
        headers={EXPECTED_ACTOR_HEADER: header} if header is not None else {},
    )

    assert response.status_code == status
    if status == 409:
        assert response.json() == {"detail": "session_actor_changed"}
    lookup.assert_not_awaited()


def test_stale_browser_logout_cannot_clear_the_successor_cookie(
    auth_context: AuthContext, settings: Settings
) -> None:
    successor = UUID("00000000-0000-4000-8000-000000000002")
    token = create_token(successor, settings.auth.session_max_age_seconds, settings.auth.secret_key)
    auth_context.client.cookies.set(settings.auth.session_cookie_name, token)
    response = auth_context.client.post(
        f"{AUTH_URL}/logout", headers={"X-Vision-Bill-Expected-User": str(USER_ID)}
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "session_actor_changed"}
    assert "set-cookie" not in response.headers
    assert auth_context.client.cookies.get(settings.auth.session_cookie_name) == token


@pytest.mark.parametrize("max_age", [300, -300])
def test_expected_actor_logout_clears_matching_or_expired_cookie(
    auth_context: AuthContext, settings: Settings, max_age: int
) -> None:
    token = create_token(USER_ID, max_age, settings.auth.secret_key)
    auth_context.client.cookies.set(
        settings.auth.session_cookie_name, token, domain="testserver.local", path="/"
    )
    response = auth_context.client.post(
        f"{AUTH_URL}/logout", headers={"X-Vision-Bill-Expected-User": str(USER_ID)}
    )
    assert response.status_code == 200
    assert settings.auth.session_cookie_name not in auth_context.client.cookies
