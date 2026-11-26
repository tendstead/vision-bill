"""Receipt CRUD over live HTTP and collection filtering on isolated PostgreSQL."""

from decimal import Decimal
from uuid import uuid4

import pytest

from e2e.discovery import first_case
from e2e.helpers import (
    API,
    receipt_write,
    register_user,
    wait_until,
)
from tests.e2e import test_workflow_db_e2e as pg_fixtures
from vision_bill.model.workflow import AccessScope
from vision_bill.provider.db.receipt_db import ReceiptDB

pytestmark = pytest.mark.e2e
db, postgres = pg_fixtures.db, pg_fixtures.postgres


def _make_receipt(client) -> str:
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


def test_receipt_crud(http_factory) -> None:
    client, _ = register_user(http_factory, "crud")
    receipt_id = _make_receipt(client)

    # GET detail
    detail = client.get(f"{API}/receipts/{receipt_id}")
    assert detail.status_code == 200

    # PUT update (change merchant_name, keep required fields valid)
    current = detail.json()
    update = receipt_write(current, merchant_name="Renamed Merchant")
    put = client.put(f"{API}/receipts/{receipt_id}", json=update)
    assert put.status_code == 200
    assert put.json()["merchant_name"] == "Renamed Merchant"

    # DELETE
    delete = client.delete(f"{API}/receipts/{receipt_id}")
    assert delete.status_code == 200
    assert delete.json() == {"deleted": receipt_id}
    assert client.get(f"{API}/receipts/{receipt_id}").status_code == 404
    assert client.delete(f"{API}/receipts/{receipt_id}").status_code == 404


async def test_receipt_collection_filter_respects_membership_and_owner(db) -> None:
    pg_fixtures.migrate(db)
    receipts = ReceiptDB(db)
    await receipts.init_db()
    try:
        owner, foreign, admin = uuid4(), uuid4(), uuid4()
        owned_collection, foreign_collection, empty_collection = uuid4(), uuid4(), uuid4()
        selected_receipt, unassigned_receipt, foreign_receipt = uuid4(), uuid4(), uuid4()
        async with receipts.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO users(id, username, hashed_password, is_admin) "
                "VALUES($1, $2, 'fixture', $3)",
                [(owner, "owner", False), (foreign, "foreign", False), (admin, "admin", True)],
            )
            await conn.executemany(
                "INSERT INTO collections(id, user_id, name) VALUES($1, $2, $3)",
                [
                    (owned_collection, owner, "Owned"),
                    (foreign_collection, foreign, "Foreign"),
                    (empty_collection, owner, "Empty"),
                ],
            )
            await conn.executemany(
                "INSERT INTO receipts(id, user_id, merchant_name, date, total, currency, "
                "status, verified) VALUES($1, $2, 'Fixture', CURRENT_DATE, $3, "
                "'USD', 'verified', TRUE)",
                [
                    (selected_receipt, owner, Decimal("3.25")),
                    (unassigned_receipt, owner, Decimal("9.50")),
                    (foreign_receipt, foreign, Decimal("7.10")),
                ],
            )
            await conn.executemany(
                "INSERT INTO receipt_collections(receipt_id, collection_id) VALUES($1, $2)",
                [(selected_receipt, owned_collection), (foreign_receipt, foreign_collection)],
            )

        owner_scope = AccessScope(user_id=owner)
        admin_scope = AccessScope(user_id=admin, can_see_all=True)
        for scope, collection_id, expected_ids in (
            (owner_scope, owned_collection, [selected_receipt]),
            (owner_scope, foreign_collection, []),
            (owner_scope, empty_collection, []),
            (admin_scope, owned_collection, [selected_receipt]),
            (admin_scope, foreign_collection, [foreign_receipt]),
            (admin_scope, empty_collection, []),
        ):
            rows = await receipts.list_receipts(
                collection_id=collection_id,
                status=["verified"],
                user_id=scope.user_id,
                can_see_all=scope.can_see_all,
            )
            assert [row.id for row in rows] == expected_ids

        statistics = await receipts.get_statistics(user_id=owner, collection_id=owned_collection)
        assert statistics.verified_receipt_count == 1
        assert [(row.currency, row.total) for row in statistics.currencies] == [
            ("USD", Decimal("3.25"))
        ]
    finally:
        await receipts.destroy_db()


async def test_collection_totals_match_visible_receipts_with_mixed_membership(db) -> None:
    from vision_bill.provider.db.collection_db import CollectionDB

    pg_fixtures.migrate(db)
    receipts = ReceiptDB(db)
    await receipts.init_db()
    try:
        owner, foreign, admin = uuid4(), uuid4(), uuid4()
        collection, foreign_only = uuid4(), uuid4()
        own_receipt, foreign_receipt = uuid4(), uuid4()
        async with receipts.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO users(id, username, hashed_password, is_admin) "
                "VALUES($1, $2, 'fixture', $3)",
                [(owner, "owner", False), (foreign, "foreign", False), (admin, "admin", True)],
            )
            await conn.executemany(
                "INSERT INTO collections(id, user_id, name) VALUES($1, $2, $3)",
                [(collection, owner, "Mixed"), (foreign_only, owner, "Other")],
            )
            await conn.executemany(
                "INSERT INTO receipts(id, user_id, merchant_name, date, total, currency, "
                "status, verified) VALUES($1, $2, 'Fixture', CURRENT_DATE, $3, $4, 'verified', TRUE)",
                [
                    (own_receipt, owner, Decimal("3.25"), "EUR"),
                    (foreign_receipt, foreign, Decimal("7.10"), "CHF"),
                ],
            )

        collections = CollectionDB(receipts.pool)
        await collections.assign(collection, own_receipt, user_id=owner)
        await collections.assign(collection, foreign_receipt, user_id=admin, can_see_all=True)
        await collections.assign(foreign_only, foreign_receipt, user_id=admin, can_see_all=True)

        for actor, see_all, expected in (
            (owner, False, {own_receipt: ("EUR", Decimal("3.25"))}),
            (admin, True, {
                own_receipt: ("EUR", Decimal("3.25")),
                foreign_receipt: ("CHF", Decimal("7.10")),
            }),
        ):
            summaries = {summary.id: summary for summary in await collections.list_with_totals(actor, see_all)}
            for ident, members in (
                (collection, expected),
                (foreign_only, {rid: total for rid, total in expected.items() if rid == foreign_receipt}),
            ):
                summary = summaries[ident]
                detail = await collections.get_detail(ident, actor, see_all)
                assert detail is not None
                assert summary.receipt_count == len(members)
                assert {row.id for row in detail.receipts} == members.keys()
                assert {(total.currency, total.total) for total in summary.totals} == set(members.values())
                assert detail.totals == summary.totals
    finally:
        await receipts.destroy_db()
