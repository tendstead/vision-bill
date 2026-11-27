"""Adversarial input + security-boundary tests against the real e2e stack.

Complements the unit tests (pure logic) with real-stack pinning: the HTTP
boundary must never 500 on hostile input, the hand-rolled HMAC session must
reject forgeries, hostile collection names round-trip through normalization,
and free text is stored verbatim in SQL.
"""

import base64
import hashlib
import hmac
import time
import uuid

import asyncpg
import httpx
import pytest

from e2e.helpers import API, register_user

pytestmark = pytest.mark.e2e

APP_BASE = "http://localhost:53625"
SECRET_KEY = "e2e-secret-do-not-use"  # docker-compose.e2e.yml
PG = dict(
    user="vision_bill", password="vision_bill", host="localhost", port=5433, database="vision_bill"
)
# 1x1 PNG (same bytes as web/src/lib/e2e/uploadPng.ts)
GENERIC_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="
)


@pytest.fixture
async def pg() -> asyncpg.Connection:
    conn = await asyncpg.connect(**PG)
    try:
        yield conn
    finally:
        await conn.close()


def _admin_client(http_factory):
    client = http_factory()
    response = client.post(
        f"{API}/auth/login", json={"username": "admin", "password": "admin-e2e-pass"}
    )
    assert response.status_code == 200
    return client


def _forged_client(cookie: str) -> httpx.Client:
    return httpx.Client(base_url=APP_BASE, timeout=30.0, cookies={"vb_session": cookie})


# --- malformed ids: the framework boundary must answer, never 500 ----------


def test_malformed_ids_never_500(http_factory) -> None:
    register_user(http_factory, "sec")  # any session; auth must succeed first
    client = http_factory()
    # Fresh client WITHOUT a session: path validation (422) or auth (401/404)
    # — a 500 would mean the boundary leaked.
    for url in (
        f"{API}/receipts/not-a-uuid",
        f"{API}/images/not-a-uuid",
        f"{API}/collections/not-a-uuid",
    ):
        response = client.get(url)
        assert response.status_code in (401, 404, 422), (url, response.status_code)

    admin = _admin_client(http_factory)
    response = admin.get(f"{API}/benchmarks/not-a-uuid")
    assert response.status_code in (404, 422)


# --- hand-rolled HMAC session: every forgery variant must 401 ---------------


def test_tampered_session_signature_rejected(http_factory) -> None:
    client, _ = register_user(http_factory, "sec")
    token = client.cookies.get("vb_session")
    assert token
    tampered = token[:-1] + ("0" if token[-1] != "0" else "1")
    response = _forged_client(tampered).get(f"{API}/auth/me")
    assert response.status_code == 401


def test_garbage_session_cookie_rejected(http_factory) -> None:
    response = _forged_client("garbage").get(f"{API}/auth/me")
    assert response.status_code == 401


def test_expired_session_cookie_rejected(http_factory) -> None:
    client, user = register_user(http_factory, "sec")
    exp = int(time.time()) - 60
    message = f"{user['id']}.{exp}"
    sig = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).hexdigest()
    response = _forged_client(f"{message}.{sig}").get(f"{API}/auth/me")
    assert response.status_code == 401


# --- auth input boundaries ---------------------------------------------------


def test_username_length_boundaries(http_factory) -> None:
    client = http_factory()
    ok = client.post(
        f"{API}/auth/register",
        json={"username": "u" * 100, "password": "e2e-pw-12345"},
    )
    assert ok.status_code == 201, ok.text
    too_long = client.post(
        f"{API}/auth/register",
        json={"username": "u" * 101, "password": "e2e-pw-12345"},
    )
    assert too_long.status_code == 422


def test_password_length_boundary(http_factory) -> None:
    client = http_factory()
    response = client.post(
        f"{API}/auth/register",
        json={"username": f"e2e-sec-{uuid.uuid4().hex[:8]}", "password": "p" * 201},
    )
    assert response.status_code == 422


async def test_sql_metacharacter_username_roundtrip(http_factory, pg) -> None:
    username = "weird';--name\"; DROP TABLE users;--"
    client = http_factory()
    registered = client.post(
        f"{API}/auth/register", json={"username": username, "password": "e2e-pw-12345"}
    )
    assert registered.status_code == 201, registered.text
    login = client.post(
        f"{API}/auth/login", json={"username": username, "password": "e2e-pw-12345"}
    )
    assert login.status_code == 200
    # The table is intact and the payload was stored verbatim (parameterized
    # SQL: no injection, no mangling).
    assert await pg.fetchval("SELECT count(*) FROM users") >= 1
    stored = await pg.fetchval("SELECT username FROM users WHERE username = $1", username)
    assert stored == username


# --- free text round-trips verbatim ------------------------------------------


def test_hostile_collection_name_roundtrips_normalized(http_factory) -> None:
    client, _ = register_user(http_factory, "sec")
    raw_name = 'Script <img src=x onerror="alert(1)"> & "quotes"\nnewline \U0001f680'
    # Collection names are normalized (whitespace collapsed) on storage.
    expected_name = " ".join(raw_name.split()).strip()
    created = client.post(f"{API}/collections", json={"name": raw_name, "color": "#112233"})
    assert created.status_code == 201, created.text
    listed = client.get(f"{API}/collections").json()
    assert any(c["name"] == expected_name for c in listed)


def test_tag_length_boundary(http_factory) -> None:
    client, _ = register_user(http_factory, "sec")
    ok = client.post(f"{API}/tags", json={"name": "t" * 100})
    assert ok.status_code in (200, 201)
    too_long = client.post(f"{API}/tags", json={"name": "t" * 101})
    assert too_long.status_code == 422


# --- upload filename hygiene --------------------------------------------------


def test_upload_path_traversal_filename(http_factory) -> None:
    client, _ = register_user(http_factory, "sec")
    png = base64.b64decode(GENERIC_PNG_B64)
    response = client.post(f"{API}/images", files={"receipt": ("../../evil.png", png, "image/png")})
    assert response.status_code == 201, response.text
    images = client.get(f"{API}/images", params={"limit": 5}).json()
    uploaded = next(i for i in images if i["original_filename"] == "../../evil.png")
    # The stored file path must never contain traversal sequences.
    assert ".." not in uploaded["image_path"]


# --- settings API --------------------------------------------------------------


def test_settings_temperature_out_of_range(http_factory) -> None:
    """EXPECTED RED (#5): out-of-range temperature must be rejected with 422.

    The product does not do this: ``LLMSettingsUpdate.temperature`` is a bare
    ``float`` with no range constraint, so 99 is accepted (200) and persisted.
    This test pins the correct boundary behaviour and stays red until the
    product adds range validation (e.g. ``ge=0, le=2``).
    """
    admin = _admin_client(http_factory)
    view = admin.get(f"{API}/system/settings").json()
    response = admin.put(
        f"{API}/system/settings",
        json={
            "llm": {**view["llm"], "temperature": 99},
            "allow_registration": view["allow_registration"],
        },
    )
    assert response.status_code == 422, (
        f"Expected 422 (out-of-range temperature rejected) but got "
        f"{response.status_code}: {response.text}"
    )


def test_settings_non_admin_forbidden(http_factory) -> None:
    client, _ = register_user(http_factory, "sec")
    response = client.put(
        f"{API}/system/settings",
        json={
            "llm": {"provider": "openai", "host": "x", "model_name": "y", "temperature": 0.1},
            "allow_registration": True,
        },
    )
    assert response.status_code == 403


# --- targeted fill-ins (cases no UI can reach) --------------------------------


def test_image_delete_after_analysis_conflict(http_factory) -> None:
    client, _ = register_user(http_factory, "sec")
    png = base64.b64decode(GENERIC_PNG_B64)
    response = client.post(f"{API}/images", files={"receipt": ("x.png", png, "image/png")})
    assert response.status_code == 201, response.text
    image_id = response.json()["image_id"]
    deleted = client.delete(f"{API}/images/{image_id}")
    assert deleted.status_code == 409


def test_benchmark_reevaluate_transient(http_factory) -> None:
    admin = _admin_client(http_factory)
    # The admin's OWN verified receipt is the ground truth (admin has no
    # can_see_all in the e2e stack).
    png = base64.b64decode(GENERIC_PNG_B64)
    uploaded = admin.post(f"{API}/images", files={"receipt": ("admin.png", png, "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    receipt_id = uploaded.json()["receipt_id"]
    verified = admin.post(f"{API}/receipts/{receipt_id}/verify")
    assert verified.status_code == 200

    run = admin.post(
        f"{API}/benchmarks",
        json={"model_ids": ["e2e-vision"], "receipt_ids": [receipt_id]},
    )
    assert run.status_code == 202, run.text
    run_id = run.json()["id"]

    result = admin.post(
        f"{API}/benchmarks/{run_id}/reevaluate",
        json={"receipt_id": receipt_id, "model_id": "e2e-vision"},
    )
    assert result.status_code == 200, result.text
    body = result.json()
    # The response carries `scores` (field-level) and `receipt_id`, not
    # singular `score`/`receipt`.
    assert "scores" in body or "receipt_id" in body

    missing = admin.post(
        f"{API}/benchmarks/{uuid.uuid4()}/reevaluate",
        json={"receipt_id": receipt_id, "model_id": "e2e-vision"},
    )
    assert missing.status_code == 404


def test_collections_scoping(http_factory) -> None:
    client_a, _ = register_user(http_factory, "seca")
    client_b, _ = register_user(http_factory, "secb")
    created = client_a.post(f"{API}/collections", json={"name": "A only"})
    assert created.status_code == 201
    collection_id = created.json()["id"]

    listed_b = client_b.get(f"{API}/collections").json()
    assert collection_id not in [c["id"] for c in listed_b]
    direct = client_b.get(f"{API}/collections/{collection_id}")
    assert direct.status_code == 404
