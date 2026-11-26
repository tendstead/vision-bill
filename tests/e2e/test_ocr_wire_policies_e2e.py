"""Each policy runs the deployed image in a separate positively owned CPU Compose project."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=["enabled", "required", "fallback"])
def policy_stack(request, tmp_path_factory):
    sys.path.insert(0, str(ROOT / "e2e"))
    spec = importlib.util.spec_from_file_location("owned_wire_runner", ROOT / "e2e/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parent = json.loads(Path(os.environ["VB_E2E_STATE"]).read_text())
    # Only the image from the positively identified parent test project may be reused.
    image = runner.call(parent, "images", "-q", "app", capture=True).stdout.strip()
    assert image and "\n" not in image
    state = runner.create_state(tmp_path_factory.mktemp("ocr-policy-" + request.param))
    state.update(policy=request.param, image=image, admin_can_see_all=False)
    try:
        runner.call(state, "up", "--detach", "--no-build")
        endpoints = runner.endpoints(state)
        subprocess.run(
            [sys.executable, str(ROOT / "e2e/wait_ready.py")],
            env={**os.environ, **endpoints},
            check=True,
        )
        yield endpoints
    finally:
        runner.call(state, "logs", "--no-color", "--tail", "60")
        runner.call(state, "down", "--volumes", "--remove-orphans")


@pytest.fixture
def wire(policy_stack):
    urls = policy_stack
    with (
        httpx.Client(base_url=urls["VB_E2E_BASE_URL"], timeout=15, trust_env=False) as client,
        httpx.Client(base_url=urls["VB_E2E_OCR_URL"], timeout=5, trust_env=False) as ocr,
        httpx.Client(base_url=urls["VB_E2E_STUB_URL"], timeout=5, trust_env=False) as llm,
    ):
        assert (
            client.post(
                "/api/v1/auth/register",
                json={"username": "wire-" + str(time.time_ns()), "password": "e2e-password"},
            ).status_code
            == 201
        )
        llm.post("/__mode", json={"mode": "ok"}).raise_for_status()
        llm.post("/__scenario", json={"scenario": "complete"}).raise_for_status()
        ocr.post("/__scenario", json={}).raise_for_status()
        yield client, ocr, llm, urls["VB_E2E_POLICY"]
        print("OCR audit", ocr.get("/__requests").json())
        print("LLM audit", llm.get("/__requests").json())
        ocr.post("/__scenario", json={}).raise_for_status()
        llm.post("/__mode", json={"mode": "ok"}).raise_for_status()


def upload(client, raw, *, count=1, **options):
    response = client.post(
        "/api/v1/images",
        params=options,
        files=[("receipt", (f"photo-{i}.png", raw, "image/png")) for i in range(count)],
    )
    assert response.status_code == 202, response.text
    return response.json()["id"]


def poll(client, image_id, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        row = client.get("/api/v1/images/" + image_id).json()
        if predicate(row):
            return row
        assert time.monotonic() < deadline, json.dumps(row)
        time.sleep(0.1)


@pytest.mark.parametrize("model_id", ["e2e-vision", "e2e-text"])
def test_enabled_exact_v2_saved_working_png_and_text_only_format(wire, model_id):
    client, ocr, llm, _ = wire
    raw = llm.get("/__fixtures/a").content
    id = upload(client, raw, count=2, model_id=model_id)
    job = poll(client, id, lambda row: row["terminal"])
    assert job["status"] == "analyzed", job
    assert [r["photo_index"] for r in ocr.get("/__requests").json()["requests"]] == [0, 1]
    calls = llm.get("/__requests").json()["requests"]
    assert calls and all(call["image_sha256"] is None for call in calls)
    formatted = [r for r in calls if r["kind"] != "classification"]
    assert len(formatted) == 1 and formatted[0]["kind"] == "ocr"
    assert formatted[0]["image_sha256"] is None
    assert formatted[0]["ordered_sources"] == [[hashlib.sha256(raw).hexdigest(), i] for i in (0, 1)]
    assert len(job["source_views"]) == 2
    for view in job["source_views"]:
        response = client.get(view["url"])
        assert response.status_code == 200
        assert hashlib.sha256(response.content).hexdigest() == view["sha256"]
        assert (view["width"], view["height"]) == (320, 480)
    receipt = client.get("/api/v1/receipts/" + job["receipt_id"]).json()
    assert receipt["receipt"]["merchant_name"] == "Aldi Test"
    assert [line["description"] for line in receipt["line_items"]] == ["Milk", "Bread"]


@pytest.mark.parametrize("result", ["empty", "503", "timeout"])
@pytest.mark.parametrize("permission", [None, False, True])
def test_legacy_fallback_grants_cannot_enable_image_inference(wire, result, permission):
    client, ocr, llm, _ = wire
    raw = llm.get("/__fixtures/a").content
    digest = hashlib.sha256(raw).hexdigest()
    ocr.post(
        "/__scenario",
        json={"plans": [{"source_sha256": digest, "photo_index": 0, "result": result}]},
    ).raise_for_status()
    id = upload(
        client,
        raw,
        **({} if permission is None else {"allow_ocr_fallback": str(permission).lower()}),
    )
    job = poll(
        client,
        id,
        lambda row: row["attempt_count"] >= 1 and row["status"] == "pending",
    )
    assert llm.get("/__requests").json()["requests"] == []
    assert job["receipt_id"] is None and job["result_code"] == "ocr_page_failed"
    assert job["attempt_count"] == 1 and not job["terminal"]
    # Delete only this receipt-free test job before its automatic retry can cross scenarios.
    if job["receipt_id"] is None:
        assert client.delete("/api/v1/images/" + id).status_code == 200


@pytest.mark.parametrize("result", ["503", "timeout"])
def test_later_photo_failure_keeps_exact_prefix_and_reuses_it_on_recovery(wire, result):
    client, ocr, llm, _ = wire
    raw = llm.get("/__fixtures/a").content
    digest = hashlib.sha256(raw).hexdigest()
    ocr.post(
        "/__scenario",
        json={
            "barriers": ["second"],
            "plans": [
                {"source_sha256": digest, "photo_index": 1, "result": result, "barrier": "second"}
            ],
        },
    ).raise_for_status()
    id = upload(client, raw, count=2)
    deadline = time.monotonic() + 10
    while len(ocr.get("/__requests").json()["requests"]) < 2:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    during = client.get("/api/v1/images/" + id).json()
    assert during["status"] == "processing" and during["receipt_id"] is None
    assert len(during["source_views"]) == 1
    view = during["source_views"][0]
    prefix = client.get(view["url"]).content
    ocr.post("/__release", json={"barrier": "second"}).raise_for_status()
    failed = poll(client, id, lambda row: row["status"] == "pending" and row["attempt_count"] == 1)
    assert failed["receipt_id"] is None
    assert failed["source_views"][0]["view_id"] == view["view_id"]
    assert client.get(view["url"]).content == prefix
    # Explicit recovery keeps profile/runtime stable; no UUID or global call-count routing.
    ocr.post("/__scenario", json={}).raise_for_status()
    recovered = poll(client, id, lambda row: row["terminal"], timeout=45)
    assert recovered["status"] == "analyzed", recovered
    assert [r["photo_index"] for r in ocr.get("/__requests").json()["requests"]] == [1]
    assert recovered["source_views"][0]["view_id"] == view["view_id"]


def test_text_only_model_never_receives_vision_fallback(wire):
    client, ocr, llm, _ = wire
    raw = llm.get("/__fixtures/a").content
    ocr.post(
        "/__scenario",
        json={
            "plans": [
                {
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "photo_index": 0,
                    "result": "503",
                }
            ]
        },
    )
    id = upload(client, raw, model_id="e2e-text", allow_ocr_fallback="true")
    job = poll(client, id, lambda row: row["attempt_count"] == 1 and row["status"] == "pending")
    assert job["receipt_id"] is None
    assert llm.get("/__requests").json()["requests"] == []
    assert client.delete("/api/v1/images/" + id).status_code == 200


def test_admin_without_see_all_cannot_access_another_users_submission(wire):
    owner, _, llm, _ = wire
    raw = llm.get("/__fixtures/a").content
    image_id = upload(owner, raw)
    job = poll(owner, image_id, lambda row: row["terminal"])
    with httpx.Client(base_url=owner.base_url, trust_env=False) as admin:
        assert (
            admin.post(
                "/api/v1/auth/login", json={"username": "admin", "password": "admin-e2e-pass"}
            ).status_code
            == 200
        )
        for suffix in ("", "/file", "/draft", f"/extractions/{job['latest_extraction_id']}"):
            assert admin.get("/api/v1/images/" + image_id + suffix).status_code == 404
        for view in job["source_views"]:
            assert admin.get(view["url"]).status_code == 404
        assert admin.get("/api/v1/receipts/" + job["receipt_id"]).status_code == 404
        assert (
            admin.post(
                "/api/v1/images/" + image_id + "/retry",
                json={"expected_generation": job["generation"], "request_key": "forbidden"},
            ).status_code
            == 404
        )
