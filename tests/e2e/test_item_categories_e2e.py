"""Confirmed history and category accounting against the real Docker stack."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from e2e.discovery import first_case
from e2e.helpers import (
    API,
    receipt_write,
    register_user,
    wait_until,
)

pytestmark = pytest.mark.e2e


def upload(client):
    case = first_case()
    assert case is not None
    with case.image_path.open("rb") as image:
        response = client.post(f"{API}/images", files={"receipt": (case.image_path.name, image)})
    assert response.status_code == 202, response.text
    image_id = response.json()["id"]

    def ready():
        row = client.get(f"{API}/images/{image_id}").json()
        return row.get("receipt_id") if row["status"] == "analyzed" else None

    receipt_id = wait_until(ready, timeout_s=60)
    return client.get(f"{API}/receipts/{receipt_id}").json()


def save(client, details, lines, **updates):
    body = receipt_write(details, line_items=lines, **updates)
    response = client.put(f"{API}/receipts/{details['receipt']['id']}", json=body)
    assert response.status_code == 200, response.text
    return client.get(f"{API}/receipts/{details['receipt']['id']}").json()


def test_confirmed_history_survives_rename_and_is_private(http_factory):
    alice, _ = register_user(http_factory, "category-alice")
    bob, _ = register_user(http_factory, "category-bob")
    details = upload(alice)
    original = details["line_items"][0]["description"]
    merchant = details["receipt"]["merchant_name"]
    lines = details["line_items"]
    lines[0].update(
        description="My breakfast item", spending_category="cereals", remember_category=True
    )
    saved = save(alice, details, lines)
    assert saved["line_items"][0]["original_description"] == original
    assert saved["line_items"][0]["category_source"] == "user"
    for description in (original, "My breakfast item"):
        params = {"merchant_name": merchant, "description": description}
        own = alice.get(f"{API}/tags/category-suggestion", params=params)
        assert own.status_code == 200, own.text
        assert own.json() == {"category": "cereals", "source": "history"}
        assert bob.get(f"{API}/tags/category-suggestion", params=params).json()["category"] is None
    repeated = upload(alice)
    assert repeated["line_items"][0]["spending_category"] == "cereals"
    assert repeated["line_items"][0]["category_source"] == "history"
    # A normal save does not turn another unconfirmed category into a mapping.
    save(alice, repeated, [{**repeated["line_items"][0], "spending_category": "milk"}])
    assert (
        alice.get(
            f"{API}/tags/category-suggestion",
            params={
                "merchant_name": merchant,
                "description": original,
            },
        ).json()["category"]
        == "cereals"
    )
    stranger = upload(bob)
    assert stranger["line_items"][0]["category_source"] != "history"


def test_item_statistics_scoping_dates_deposits_and_collections(http_factory):
    alice, _ = register_user(http_factory, "item-stats")
    bob, _ = register_user(http_factory, "item-stats-other")
    today = datetime.now(UTC).date()
    details = upload(alice)
    lines = [
        {
            "description": "Milk",
            "quantity": 1,
            "unit_price": "6",
            "total_price": "6",
            "spending_category": "milk",
        },
        {
            "description": "Unclear",
            "quantity": 1,
            "unit_price": "4",
            "total_price": "4",
            "spending_category": "unknown",
        },
        {
            "description": "Pfand",
            "quantity": 1,
            "unit_price": "0.25",
            "total_price": "0.25",
            "spending_category": "deposit",
        },
    ]
    details = save(
        alice,
        details,
        lines,
        date=today.isoformat(),
        total="9.25",
        subtotal="10.25",
        discount_total="1",
        tax_total="0",
        taxes=[],
        amount_policy={
            "tax_basis": "unknown",
            "subtotal_origin": "unknown",
            "tax_information": "missing",
            "discount_treatment": "unknown",
            "tip_treatment": "none",
        },
    )
    receipt_id = details["receipt"]["id"]
    assert alice.get(f"{API}/statistics/items").json()["categories"] == []
    row = details["receipt"]
    assert {f["code"] for f in row["findings"]} == {
        "unknown_tax_basis",
        "unknown_subtotal_origin",
        "missing_tax_information",
        "unknown_discount_treatment",
    }
    assert all(f["severity"] == "review" for f in row["findings"])
    verified = alice.post(
        f"{API}/receipts/{receipt_id}/verify",
        json={
            "expected_operation_revision": row["operation_revision"],
            "expected_review_revision": row["review_revision"],
            "acknowledgements": [
                {"finding_id": f["id"], "comment": "Test fixture printed values confirmed"}
                for f in row["findings"]
                if f["severity"] == "review"
            ],
        },
    )
    assert verified.status_code == 200, verified.text
    details = alice.get(f"{API}/receipts/{receipt_id}").json()
    report = alice.get(f"{API}/statistics/items", params={"months": 3})
    assert report.status_code == 200, report.text
    report = report.json()
    categories = {row["category"]: row for row in report["categories"]}
    assert Decimal(categories["milk"]["share"]) == 60
    assert Decimal(categories["unknown"]["share"]) == 40
    assert "deposit" not in categories
    assert len(report["monthly"]) == 6
    amounts = report["reconciliation"][0]
    assert Decimal(amounts["deposit_total"]) == Decimal("0.25")
    assert Decimal(amounts["unallocated_total"]) == -1
    assert bob.get(f"{API}/statistics/items").json()["categories"] == []
    collection = alice.post(f"{API}/collections", json={"name": "Category test"}).json()
    assert (
        alice.get(f"{API}/statistics/items", params={"collection_id": collection["id"]}).json()[
            "categories"
        ]
        == []
    )
    details = save(alice, details, details["line_items"], collection_ids=[collection["id"]])
    scoped = alice.get(f"{API}/statistics/items", params={"collection_id": collection["id"]}).json()
    assert scoped["categories"] == report["categories"]
    assert (
        bob.get(f"{API}/statistics/items", params={"collection_id": collection["id"]}).json()[
            "categories"
        ]
        == []
    )
    # Future-dated receipts do not inflate a partial current month.
    save(alice, details, details["line_items"], date=(today + timedelta(days=1)).isoformat())
    assert alice.get(f"{API}/statistics/items").json()["categories"] == []


def test_category_endpoints_require_authentication(http_factory):
    client = http_factory()
    for path in (
        "/tags/categories",
        "/tags/category-suggestion?merchant_name=X&description=Milk",
        "/statistics/items",
    ):
        assert client.get(API + path).status_code == 401
