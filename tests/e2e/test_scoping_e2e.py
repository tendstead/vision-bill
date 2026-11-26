"""Multi-user isolation and admin see-all over HTTP and in-process ASGI."""

from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI

from e2e.discovery import first_case
from e2e.helpers import (
    API,
    register_user,
    wait_until,
)
from tests.e2e import test_workflow_db_e2e as pg_fixtures
from vision_bill.api.collections import router as collections_router
from vision_bill.model.collection import CollectionCreate
from vision_bill.provider.llm.base import LLMProvider
from vision_bill.security.dependencies import get_current_user
from vision_bill.service.collection_service import CollectionService
from vision_bill.service.receipt_service import ReceiptService

pytestmark = pytest.mark.e2e
db, postgres = pg_fixtures.db, pg_fixtures.postgres


def _receipt_id_for(client) -> str:
    case = first_case()
    assert case is not None, "no e2e fixtures registered"
    with case.image_path.open("rb") as fh:
        response = client.post(
            f"{API}/images", files={"receipt": (case.image_path.name, fh, "image/png")}
        )
    assert response.status_code == 202
    image_id = response.json()["id"]

    def check():
        row = client.get(f"{API}/images/{image_id}").json()
        return row["receipt_id"] if row["status"] == "analyzed" else None

    return wait_until(check, timeout_s=60.0)


def test_users_cannot_see_each_others_data(http_factory) -> None:
    alice, _ = register_user(http_factory, "scope-a")
    bob, _ = register_user(http_factory, "scope-b")
    alice_receipt = _receipt_id_for(alice)

    # Bob cannot read or list Alice's receipt.
    assert bob.get(f"{API}/receipts/{alice_receipt}").status_code == 404
    bob_list = bob.get(f"{API}/receipts").json()
    assert all(r["id"] != alice_receipt for r in bob_list)

    # Bob cannot verify or delete Alice's receipt.
    assert bob.post(f"{API}/receipts/{alice_receipt}/verify").status_code == 404
    assert bob.delete(f"{API}/receipts/{alice_receipt}").status_code == 404

    # Alice still owns it.
    assert alice.get(f"{API}/receipts/{alice_receipt}").status_code == 200


def test_admin_sees_all(http_factory) -> None:
    carol, _ = register_user(http_factory, "scope-c")
    carol_receipt = _receipt_id_for(carol)

    admin = http_factory()
    login = admin.post(
        f"{API}/auth/login", json={"username": "admin", "password": "admin-e2e-pass"}
    )
    assert login.status_code == 200
    admin_list = admin.get(f"{API}/receipts").json()
    assert any(r["id"] == carol_receipt for r in admin_list)
    assert admin.get(f"{API}/receipts/{carol_receipt}").status_code == 200


async def test_active_collection_belongs_to_current_user_even_for_see_all_admin(
    db, settings
) -> None:
    pg_fixtures.migrate(db)
    receipts = ReceiptService(settings.images, db, Mock(spec=LLMProvider))
    await receipts.init_db()
    try:
        owner = await receipts.user_db.create_user("owner", "fixture")
        admin = await receipts.user_db.create_user("admin", "fixture", is_admin=True)
        admin.can_see_all = True
        collections = CollectionService(receipts)
        foreign_active = await collections.create(
            CollectionCreate(name="Owner's current", active=True), owner.id
        )
        foreign_next = await collections.create(CollectionCreate(name="Owner's next"), owner.id)
        admin_collection = await collections.create(CollectionCreate(name="Admin's own"), admin.id)

        prefix = "/api/v1/collections"
        app = FastAPI()
        app.include_router(collections_router, prefix=prefix)
        app.state.collection_service = collections
        app.dependency_overrides[get_current_user] = lambda: admin
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            admin_active_response = await client.get(f"{prefix}/active")
            assert admin_active_response.status_code == 204
            assert admin_active_response.content == b""

            listed = await client.get(prefix)
            assert listed.status_code == 200
            assert {row["id"] for row in listed.json()} == {
                str(foreign_active.id),
                str(foreign_next.id),
                str(admin_collection.id),
            }

            for collection_id in (admin_collection.id, foreign_next.id):
                activated = await client.post(f"{prefix}/{collection_id}/activate")
                assert activated.status_code == 200
                assert activated.json()["active"] is True

            admin_active_response = await client.get(f"{prefix}/active")
            assert admin_active_response.status_code == 200
            assert admin_active_response.json()["id"] == str(admin_collection.id)

            app.dependency_overrides[get_current_user] = lambda: owner
            owner_active_response = await client.get(f"{prefix}/active")
            assert owner_active_response.status_code == 200
            assert owner_active_response.json()["id"] == str(foreign_next.id)
    finally:
        await receipts.destroy_db()
