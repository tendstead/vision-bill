"""Search + statistics over verified receipts (first registered fixture)."""

from decimal import Decimal

import pytest

from e2e.discovery import first_case
from e2e.helpers import (
    API,
    register_user,
    verify_receipt,
    wait_until,
)

pytestmark = pytest.mark.e2e


def _verified_receipt(client) -> dict:
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
        return row if row["status"] == "analyzed" else None

    row = wait_until(check, timeout_s=60.0)
    verify = verify_receipt(client, row["receipt_id"])
    assert verify.status_code == 200
    return client.get(f"{API}/receipts/{row['receipt_id']}").json()


def test_search_finds_verified_line_item(http_factory) -> None:
    client, _ = register_user(http_factory, "search")
    _verified_receipt(client)
    case = first_case()
    assert case is not None
    description = case.expected_receipt.line_items[0].description
    # Search a distinctive substring of the first line item.
    term = description.split()[0]
    response = client.get(f"{API}/search", params={"query": term})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == term
    assert any(p["description"] == description for p in body["purchases"])
    assert body["latest_price"] is not None


def test_statistics_reflect_verified_receipt(http_factory) -> None:
    client, _ = register_user(http_factory, "stats")
    _verified_receipt(client)
    case = first_case()
    assert case is not None
    expected_total = case.expected_receipt.total
    expected_currency = case.expected_receipt.currency

    response = client.get(f"{API}/statistics")
    assert response.status_code == 200
    stats = response.json()
    assert stats["verified_receipt_count"] == 1
    cur = next(c for c in stats["currencies"] if c["currency"] == expected_currency)
    assert cur["receipt_count"] == 1
    assert Decimal(cur["total"]) == expected_total
    merchants = {m["name"] for m in stats["merchants"]}
    assert case.expected_receipt.merchant_name in merchants
