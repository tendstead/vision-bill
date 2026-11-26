"""Real deployed HTTP boundaries, independent of injected provider/ASGI tests."""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import Barrier

import httpx
import pytest

from e2e.helpers import API, STUB_BASE, accept_upload, await_submission, register_user

pytestmark = pytest.mark.e2e


def test_same_pixels_distinct_jobs_and_owner_views(http_factory, stub):
    owner, _ = register_user(http_factory, "same-pixels")
    other, _ = register_user(http_factory, "same-pixels-other")
    raw = httpx.get(STUB_BASE + "/__fixtures/a").content
    first = await_submission(owner, accept_upload(owner, [("a.png", raw, "image/png")])["id"])
    second = await_submission(owner, accept_upload(owner, [("a.png", raw, "image/png")])["id"])
    foreign = await_submission(other, accept_upload(other, [("a.png", raw, "image/png")])["id"])
    assert len({first["id"], second["id"], foreign["id"]}) == 3
    views = [job["source_views"][0] for job in (first, second, foreign)]
    assert len({view["view_id"] for view in views}) == 3
    assert len({view["sha256"] for view in views}) == 1
    for view in views[:2]:
        png = owner.get(view["url"])
        assert png.status_code == 200
        assert hashlib.sha256(png.content).hexdigest() == view["sha256"]
        assert other.get(view["url"]).status_code == 404
    assert owner.get(views[2]["url"]).status_code == 404


def test_concurrent_acceptance_draft_cas_finalize_and_cross_owner(http_factory, stub):
    stub.set_scenario("partial")
    owner, _ = register_user(http_factory, "wire-cas")
    other, _ = register_user(http_factory, "wire-cas-other")
    raw = httpx.get(STUB_BASE + "/__fixtures/a").content
    barrier = Barrier(2)

    def upload():
        barrier.wait(timeout=5)
        return accept_upload(
            owner,
            [("first.png", raw, "image/png"), ("second.png", raw, "image/png")],
            key="same-intent",
        )

    with ThreadPoolExecutor(2) as executor:
        first, second = list(executor.map(lambda _: upload(), range(2)))
    assert first["id"] == second["id"]
    job = await_submission(owner, first["id"], expected="needs_review")
    image_url = f"{API}/images/{job['id']}"
    original = owner.get(image_url + "/file").content
    draft = owner.get(image_url + "/draft").json()
    for suffix in (
        "",
        "/draft",
        f"/extractions/{job['latest_extraction_id']}",
        "/file",
        "/file?part=1",
    ):
        assert other.get(image_url + suffix).status_code == 404
    for view in job["source_views"]:
        assert other.get(view["url"]).status_code == 404
    assert (
        other.patch(
            image_url + "/draft",
            json={"expected_revision": draft["revision"], "changes": {"merchant_name": "intruder"}},
        ).status_code
        == 404
    )
    assert (
        other.post(
            image_url + "/draft/finalize",
            json={"expected_revision": draft["revision"], "request_key": "foreign"},
        ).status_code
        == 404
    )
    assert (
        other.post(
            image_url + "/retry",
            json={"expected_generation": job["generation"], "request_key": "foreign"},
        ).status_code
        == 404
    )
    assert owner.get(image_url + "/draft").json()["revision"] == draft["revision"]
    assert owner.get(image_url + "/file").content == original == raw
    changed = owner.post(
        API + "/images?allow_ocr_fallback=false",
        headers={"Idempotency-Key": "same-intent"},
        files=[
            ("receipt", ("first.png", raw, "image/png")),
            ("receipt", ("second.png", raw, "image/png")),
        ],
    )
    assert changed.status_code == 409
    barrier = Barrier(2)

    def patch():
        barrier.wait(timeout=5)
        return owner.patch(
            image_url + "/draft",
            json={
                "expected_revision": draft["revision"],
                "changes": {"merchant_name": "CAS winner"},
            },
        )

    with ThreadPoolExecutor(2) as executor:
        responses = list(executor.map(lambda _: patch(), range(2)))
    assert sorted(r.status_code for r in responses) == [200, 409]
    current = owner.get(image_url + "/draft").json()
    assert current["document"]["discount_total"] is None
    assert current["document"]["tax_total"] is None
    tax_total = sum(Decimal(tax["amount"]) for tax in current["document"]["taxes"])
    completed = owner.patch(
        image_url + "/draft",
        json={
            "expected_revision": current["revision"],
            "changes": {
                "tax_total": str(tax_total),
                "line_items": [{**line, "tags": []} for line in current["document"]["line_items"]],
            },
        },
    )
    assert completed.status_code == 200, completed.text
    barrier = Barrier(2)

    def finalize():
        barrier.wait(timeout=5)
        return owner.post(
            image_url + "/draft/finalize",
            json={
                "expected_revision": completed.json()["revision"],
                "request_key": "same-finalize",
            },
        )

    with ThreadPoolExecutor(2) as executor:
        responses = list(executor.map(lambda _: finalize(), range(2)))
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert responses[0].json()["id"] == responses[1].json()["id"]
    assert responses[0].json()["discount_total"] is None
    assert (
        owner.get(image_url + "/draft").json()["finalized_receipt_id"] == responses[0].json()["id"]
    )


def test_real_list_contract_for_waiting_and_terminal_rows(http_factory, stub):
    client, _ = register_user(http_factory, "list-shape")
    raw = httpx.get(STUB_BASE + "/__fixtures/a").content
    stub.set_mode("down")
    accepted = accept_upload(client, [("waiting.png", raw, "image/png")])
    row = next(r for r in client.get(API + "/images").json() if r["id"] == accepted["id"])
    assert row["status"] == "pending" and row["attempt_count"] == 0
    assert {
        "terminal",
        "latest_extraction_id",
        "source_views",
        "claim_token",
        "lease_expires_at",
        "effective_options",
        "artifact_path",
        "png_base64",
    }.isdisjoint(row)
    stub.set_mode("ok")
    client.post(API + "/images/analyze")
    done = await_submission(client, row["id"])
    row = next(r for r in client.get(API + "/images").json() if r["id"] == accepted["id"])
    assert row["status"] == "analyzed" and row["receipt_id"] == done["receipt_id"]
    assert "terminal" not in row and done["terminal"] is True
    if os.environ["VB_E2E_POLICY"] == "off":
        assert httpx.get(os.environ["VB_E2E_OCR_URL"] + "/__requests").json()["requests"] == []


def test_retry_generation_and_request_replay_are_fenced(http_factory, stub):
    stub.set_scenario("partial")
    owner, _ = register_user(http_factory, "wire-retry")
    raw = httpx.get(STUB_BASE + "/__fixtures/a").content
    job = await_submission(
        owner,
        accept_upload(owner, [("retry.png", raw, "image/png")])["id"],
        expected="needs_review",
    )
    old_view = job["source_views"][0]
    old_png = owner.get(old_view["url"]).content
    url = f"{API}/images/{job['id']}/retry"
    request = {"expected_generation": job["generation"], "request_key": "retry-once"}
    first = owner.post(url, json=request)
    assert first.status_code == 202, first.text
    assert first.json()["generation"] == job["generation"] + 1
    replay = owner.post(url, json=request)
    assert replay.status_code == 202 and replay.json() == first.json()
    assert (
        owner.post(
            url, json={**request, "expected_generation": first.json()["generation"]}
        ).status_code
        == 409
    )
    assert owner.post(url, json={**request, "request_key": "stale-new-key"}).status_code == 409
    current = await_submission(owner, job["id"], expected="needs_review")
    assert current["generation"] == first.json()["generation"]
    assert current["latest_extraction_id"] != job["latest_extraction_id"]
    assert len(current["source_views"]) == 2
    assert len({view["view_id"] for view in current["source_views"]}) == 2
    assert {view["sha256"] for view in current["source_views"]} == {old_view["sha256"]}
    assert owner.get(old_view["url"]).content == old_png
    draft = owner.get(f"{API}/images/{job['id']}/draft").json()
    edited = owner.patch(
        f"{API}/images/{job['id']}/draft",
        json={
            "expected_revision": draft["revision"],
            "changes": {"merchant_name": "Preserve human work"},
        },
    )
    assert edited.status_code == 200
    assert (
        owner.post(
            url, json={"expected_generation": current["generation"], "request_key": "edited"}
        ).status_code
        == 409
    )
