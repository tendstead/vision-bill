"""Real PostgreSQL/provider round trips for durable multi-photo submissions."""

import httpx
import pytest

from e2e.helpers import (
    API,
    STUB_BASE,
    register_user,
    wait_until,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("bypass", [False, True])
def test_multi_photo_immediate_processing_review_and_cleanup(http_factory, stub, bypass):
    stub.set_scenario("policy_unknown")
    client, _ = register_user(http_factory, "multi")
    other, _ = register_user(http_factory, "other-multi")
    photo = httpx.get(f"{STUB_BASE}/__fixtures/a").content
    truth = httpx.get(f"{STUB_BASE}/__fixtures/a.json").json()
    response = client.post(
        f"{API}/images",
        params={"bypass_review": str(bypass).lower()},
        files=[
            ("receipt", ("top.png", photo, "image/png")),
            ("receipt", ("bottom.png", photo, "image/png")),
        ],
    )
    assert response.status_code == 202, response.text
    image_url = f"{API}/images/{response.json()['id']}"

    def completed():
        row = client.get(image_url).json()
        return row if row["status"] not in ("pending", "processing") else None

    # No /analyze call: the upload itself must wake the worker, not the 300s timer.
    job = wait_until(completed, timeout_s=20)
    assert job["status"] == "analyzed", job
    assert len(job["additional_images"]) == 1
    assert other.get(image_url).status_code == 404
    assert other.get(f"{image_url}/file?part=1").status_code == 404
    receipt_url = f"{API}/receipts/{job['receipt_id']}"
    receipt = client.get(receipt_url).json()
    assert len(receipt["line_items"]) == len(truth["line_items"])
    row = receipt["receipt"]
    # This fixture supplies no amount policy: bypass cannot approve these unknowns.
    assert {f["code"] for f in row["findings"]} == {
        "unknown_tax_basis",
        "unknown_subtotal_origin",
        "missing_tax_information",
    }
    assert all(f["severity"] == "review" for f in row["findings"])
    assert row["status"] == "unverified" and row["verified"] is False
    verified = client.post(
        f"{receipt_url}/verify",
        json={
            "expected_operation_revision": row["operation_revision"],
            "expected_review_revision": row["review_revision"],
            "acknowledgements": [
                {
                    "finding_id": f["id"],
                    "comment": "Fixture policy unknowns reviewed against originals",
                }
                for f in row["findings"]
            ],
        },
    )
    assert verified.status_code == 200, verified.text
    assert client.get(f"{image_url}/file?part=1").content == photo
    assert client.delete(receipt_url).status_code == 200
    assert client.get(image_url).status_code == 404


@pytest.mark.parametrize("count", [1, 2])
def test_unreadable_is_terminal_and_survives_reload(http_factory, stub, count):
    client, _ = register_user(http_factory, "blur")
    photo = httpx.get(f"{STUB_BASE}/__fixtures/a").content
    stub.set_mode("unreadable")
    response = client.post(
        f"{API}/images",
        files=[("receipt", (f"part-{i}.png", photo, "image/png")) for i in range(count)],
    )
    assert response.status_code == 202, response.text
    image_url = f"{API}/images/{response.json()['id']}"

    def unreadable():
        row = client.get(image_url).json()
        return row if row["status"] == "unreadable" else None

    row = wait_until(unreadable, timeout_s=20)
    assert row["receipt_id"] is None
    stub.set_mode("ok")
    client.post(f"{API}/images/analyze")
    assert client.get(image_url).json()["status"] == "unreadable"
    assert client.delete(image_url).status_code == 200
