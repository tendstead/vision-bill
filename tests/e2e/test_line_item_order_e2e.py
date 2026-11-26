"""E2E: line items are persisted and fetched in their original (LLM) order.

Regression test for the line-item ordering fix (migration 0006 adds a
``position`` column; rows are listed ``ORDER BY position``). Before the fix,
line items were listed ``ORDER BY id`` on random UUIDs and reshuffled on every
save. Uses the deterministic LLM stub fixture 'a' (Milk, Bread) to verify the
DB returns the rows in that order with sequential positions.
"""

from pathlib import Path

import pytest

from e2e.helpers import (
    API,
    receipt_write,
    register_user,
    wait_until,
)

pytestmark = pytest.mark.e2e

A_PNG = Path(__file__).resolve().parents[2] / "e2e" / "llm_stub" / "fixtures" / "a.png"


def test_line_items_fetched_in_original_order(http_factory) -> None:
    client, _ = register_user(http_factory, "order")
    with A_PNG.open("rb") as fh:
        response = client.post(f"{API}/images", files={"receipt": ("a.png", fh, "image/png")})
    assert response.status_code == 202, response.text
    image_id = response.json()["id"]

    def check():
        row = client.get(f"{API}/images/{image_id}").json()
        return row if row["status"] == "analyzed" else None

    row = wait_until(check, timeout_s=60.0)
    receipt = client.get(f"{API}/receipts/{row['receipt_id']}").json()

    descriptions = [li["description"] for li in receipt["line_items"]]
    positions = [li["position"] for li in receipt["line_items"]]
    # Fixture 'a' is (Milk, Bread); the DB must return them in that order.
    assert descriptions == ["Milk", "Bread"], f"line items reshuffled: {descriptions}"
    assert positions == [0, 1], f"positions not sequential: {positions}"

    reordered = list(reversed(receipt["line_items"]))
    reordered[0]["description"] = reordered[1]["description"] = "Repeated item"
    body = receipt_write(receipt, line_items=reordered)
    saved = client.put(f"{API}/receipts/{row['receipt_id']}", json=body)
    assert saved.status_code == 200, saved.text
    fetched = client.get(f"{API}/receipts/{row['receipt_id']}").json()["line_items"]
    assert [item["id"] for item in fetched] == [item["id"] for item in reordered]
    assert [item["position"] for item in fetched] == [0, 1]
    assert [item["description"] for item in fetched] == ["Repeated item", "Repeated item"]
