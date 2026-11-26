"""Data-driven core pipeline: upload -> LLM stub -> receipt -> verify.

One test case per registered fixture pair (tests/e2e/data + fixtures.toml).
With no fixtures registered the parametrized test is auto-skipped.
"""

from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from e2e.discovery import FixtureCase, load_fixture_cases
from e2e.helpers import (
    API,
    await_submission,
    register_user,
    verify_receipt,
    wait_until,
)

pytestmark = pytest.mark.e2e

CASES: list[FixtureCase] = load_fixture_cases()


def _upload(client: httpx.Client, image_path: Path) -> httpx.Response:
    with image_path.open("rb") as fh:
        response = client.post(
            f"{API}/images",
            files={"receipt": (image_path.name, fh, "application/octet-stream")},
        )
    assert response.status_code == 202, response.text
    return response


def _wait_analyzed(client: httpx.Client, image_id: str) -> dict:
    def check():
        # Drive one analysis cycle, then read the image row back.
        row = client.get(f"{API}/images/{image_id}").json()
        return row if row["status"] == "analyzed" else None

    return wait_until(check, timeout_s=60.0)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_pipeline_per_fixture(http_factory, case: FixtureCase) -> None:
    client, _ = register_user(http_factory, "pipe")
    response = _upload(client, case.image_path)
    image_id = response.json()["id"]
    row = _wait_analyzed(client, image_id)
    assert row["receipt_id"] is not None

    receipt = client.get(f"{API}/receipts/{row['receipt_id']}").json()
    expected = case.expected_receipt
    assert receipt["receipt"]["merchant_name"] == expected.merchant_name
    assert receipt["receipt"]["date"] == expected.date.isoformat()
    assert receipt["receipt"]["currency"] == expected.currency
    assert Decimal(receipt["receipt"]["total"]) == expected.total
    assert Decimal(receipt["receipt"]["subtotal"]) == expected.subtotal
    got_items = {
        (
            item["description"],
            Decimal(item["quantity"]) if item["quantity"] is not None else None,
            Decimal(item["unit_price"]),
            Decimal(item["total_price"]),
        )
        for item in receipt["line_items"]
    }
    want_items = {
        (item.description, item.quantity, item.unit_price, item.total_price)
        for item in expected.line_items
    }
    assert got_items == want_items

    # Verify marks the receipt and moves the image to permanent storage.
    verify = verify_receipt(client, row["receipt_id"])
    assert verify.status_code == 200
    assert verify.json()["status"] == "verified"
    assert client.post(f"{API}/receipts/{row['receipt_id']}/verify").status_code == 409

    # The stored file still serves the original uploaded bytes.
    file_response = client.get(f"{API}/images/{image_id}/file")
    assert file_response.status_code == 200
    assert file_response.content == case.image_path.read_bytes()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_bypass_review_verifies_immediately(http_factory, case: FixtureCase) -> None:
    """bypass_review=true persists the receipt as verified in one step."""
    client, _ = register_user(http_factory, "bypass")
    with case.image_path.open("rb") as fh:
        response = client.post(
            f"{API}/images?bypass_review=true",
            files={"receipt": (case.image_path.name, fh, "application/octet-stream")},
        )
    assert response.status_code == 202, response.text
    receipt_id = await_submission(client, response.json()["id"])["receipt_id"]

    detail = client.get(f"{API}/receipts/{receipt_id}").json()
    assert detail["receipt"]["status"] == "verified"
    assert detail["receipt"]["verified"] is True
    # Already verified -> a second verify is a conflict.
    assert client.post(f"{API}/receipts/{receipt_id}/verify").status_code == 409
