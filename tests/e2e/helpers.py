"""Shared helpers for the e2e suite (requires `make e2e-up`)."""

import os
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx

API = os.environ.get("VB_E2E_BASE_URL", "http://127.0.0.1:1") + "/api/v1"
STUB_BASE = os.environ.get("VB_E2E_STUB_URL", "http://127.0.0.1:1")


def register_user(
    http_factory: Callable[[], httpx.Client], prefix: str
) -> tuple[httpx.Client, dict[str, Any]]:
    """Register a fresh, isolated user; returns (logged-in client, user JSON)."""
    client = http_factory()
    username = f"e2e-{prefix}-{uuid.uuid4().hex[:8]}"
    response = client.post(
        f"{API}/auth/register", json={"username": username, "password": "e2e-pw-12345"}
    )
    assert response.status_code == 201, response.text
    return client, response.json()


class Stub:
    """Thin client for the LLM stub control API (host port 9123)."""

    def __init__(self) -> None:
        self._http = httpx.Client(base_url=STUB_BASE, timeout=10.0)

    def set_mode(self, mode: str) -> None:
        response = self._http.post("/__mode", json={"mode": mode})
        assert response.status_code == 200, response.text

    def set_scenario(self, scenario: str) -> None:
        response = self._http.post("/__scenario", json={"scenario": scenario})
        assert response.status_code == 200, response.text

    def get_requests(self) -> list[dict[str, Any]]:
        response = self._http.get("/__requests")
        assert response.status_code == 200, response.text
        return response.json()["requests"]

    def close(self) -> None:
        self._http.close()


def wait_until(fn: Callable[[], Any], timeout_s: float = 30.0, interval_s: float = 0.5) -> Any:
    """Poll fn() until it returns a truthy value; returns that value or raises."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = fn()
        if result:
            return result
        if time.monotonic() > deadline:
            raise AssertionError(f"wait_until timed out after {timeout_s}s")
        time.sleep(interval_s)


def accept_upload(client, photos, *, key=None, **options):
    """Return durable acceptance only; callers choose whether/when to poll."""
    response = client.post(
        f"{API}/images",
        params=options,
        headers={"Idempotency-Key": key or str(uuid.uuid4())},
        files=[("receipt", photo) for photo in photos],
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["id"] and response.headers["Location"].endswith(body["id"])
    return body


def await_submission(client, image_id, *, timeout=120, expected="analyzed"):
    latest = {}

    def completed():
        nonlocal latest
        response = client.get(f"{API}/images/{image_id}")
        assert response.status_code == 200, response.text
        latest = response.json()
        return latest if latest["terminal"] else None

    try:
        result = wait_until(completed, timeout_s=timeout, interval_s=0.2)
    except AssertionError as exc:
        raise AssertionError(f"submission {image_id}: {latest}") from exc
    if expected is not None:
        assert result["status"] == expected, result
    return result


def verify_receipt(client, receipt_id, expected=()):
    """Exact current review intent; never acknowledge unknown findings automatically."""
    response = client.get(f"{API}/receipts/{receipt_id}")
    assert response.status_code == 200, response.text
    row = response.json()["receipt"]
    assert not [f for f in row["findings"] if f["severity"] == "integrity"]
    findings = [f for f in row["findings"] if f["severity"] == "review"]
    assert sorted(f["code"] for f in findings) == sorted(item["code"] for item in expected)
    acknowledgements = []
    for item in expected:
        assert item["comment"].strip()
        index = next(
            i
            for i, f in enumerate(findings)
            if f["code"] == item["code"] and ("id" not in item or f["id"] == item["id"])
        )
        acknowledgements.append(
            {"finding_id": findings.pop(index)["id"], "comment": item["comment"]}
        )
    response = client.post(
        f"{API}/receipts/{receipt_id}/verify",
        json={
            "expected_operation_revision": row["operation_revision"],
            "expected_review_revision": row["review_revision"],
            "acknowledgements": acknowledgements,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["verified"] is True
    return response


def qualify_gold(client, receipt_id):
    """Explicit test intent to freeze this ordinary-human-verified reference."""
    response = client.get(f"{API}/benchmarks/references/{receipt_id}")
    assert response.status_code == 200, response.text
    preview = response.json()
    response = client.post(
        f"{API}/benchmarks/references/{receipt_id}",
        json={
            "expected_review_revision": preview["review_revision"],
            "expected_document_sha256": preview["document_sha256"],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def receipt_write(detail, **changes):
    """Convert the read DTO to the existing versioned write contract, retaining child IDs."""
    observed_time = detail["receipt"].get("time")
    if observed_time and re.fullmatch(r"\d{2}:\d{2}:00(?:\.0+)?", observed_time):
        observed_time = observed_time[:5]
    return {
        **detail["receipt"],
        "time": observed_time,
        "line_items": detail["line_items"],
        "taxes": detail["taxes"],
        "collection_ids": detail.get("collection_ids", []),
        "expected_operation_revision": detail["receipt"]["operation_revision"],
        **changes,
    }
