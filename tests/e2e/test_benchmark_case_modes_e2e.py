"""Exercise private benchmark uploads and the model/arm matrix through HTTP."""

import os
from collections import Counter
from pathlib import Path

import httpx
import pytest

from e2e.helpers import API, wait_until
from tests.e2e.test_benchmark_experiments_e2e import admin

pytestmark = pytest.mark.e2e
FIXTURES = Path(__file__).resolve().parents[2] / "e2e/llm_stub/fixtures"


def create_case(client, *names):
    files = [
        ("photos", (f"{name}.png", (FIXTURES / f"{name}.png").read_bytes(), "image/png"))
        for name in names
    ]
    response = client.post(f"{API}/benchmarks/cases", data={"name": "Counter receipt"}, files=files)
    assert response.status_code == 201, response.text
    case = response.json()
    assert response.headers["Location"].endswith(f"/cases/{case['id']}")
    assert case["photo_count"] == len(names) and not case["confirmed"]
    return case


def read_json(client, url):
    response = client.get(url)
    assert response.status_code == 200, response.text
    return response.json()


def ocr_calls():
    with httpx.Client(base_url=os.environ["VB_E2E_OCR_URL"]) as ocr:
        return read_json(ocr, "/__requests")["requests"]


def test_case_upload_preview_and_confirmation_leave_ledger_queue_and_models_idle(
    http_factory, registered_user, stub,
):
    client = admin(http_factory)
    ordinary, _ = registered_user
    resources = [f"{API}/{name}" for name in ("images", "receipts", "statistics", "benchmarks")]
    before = [read_json(client, url) for url in resources]
    counts = len(stub.get_requests()), len(ocr_calls())

    case = create_case(client, "b", "a")
    url = f"{API}/benchmarks/cases/{case['id']}"
    for index, name in enumerate(("b", "a")):
        photo = client.get(f"{url}/photos/{index}")
        assert photo.status_code == 200
        assert photo.content == (FIXTURES / f"{name}.png").read_bytes()
        assert ordinary.get(f"{url}/photos/{index}").status_code == 403
    assert ordinary.get(url).status_code == 403
    selection = {"model_ids": ["e2e-vision"], "case_ids": [case["id"]]}
    assert client.post(f"{API}/benchmarks", json=selection).status_code == 422
    draft = client.patch(url, json={"expected_revision": case["revision"], "expected": {"total": "3.86"}})
    assert draft.status_code == 200, draft.text
    revision = draft.json()["revision"]
    assert client.patch(url, json={"expected_revision": case["revision"], "confirm": True}).status_code == 409
    saved = client.patch(url, json={"expected_revision": revision, "confirm": True})
    assert saved.status_code == 200, saved.text
    assert read_json(client, url)["confirmed"] is True
    assert [read_json(client, url) for url in resources] == before
    assert (len(stub.get_requests()), len(ocr_calls())) == counts


@pytest.mark.parametrize(
    ("pipeline", "comparison", "models", "calls"),
    [
        ("direct_vision", False, ["e2e-vision"], [("e2e-vision", "direct")]),
        ("ocr", False, ["e2e-text"], [("e2e-text", "ocr")]),
        (
            "direct_vision", True, ["e2e-text", "e2e-vision"],
            [("e2e-vision", "direct"), ("e2e-vision", "ocr"), ("e2e-text", "ocr")],
        ),
    ],
)
def test_only_requested_model_and_pipeline_combinations_execute(
    http_factory, stub, pipeline, comparison, models, calls,
):
    client = admin(http_factory)
    case = create_case(client, "a")
    base = f"{API}/benchmarks"
    confirmed = client.patch(
        f"{base}/cases/{case['id']}",
        json={"expected_revision": case["revision"], "expected": {"total": "3.86"}, "confirm": True},
    )
    assert confirmed.status_code == 200, confirmed.text
    before = len(ocr_calls())
    response = client.post(
        base,
        json={
            "model_ids": models, "case_ids": [case["id"]], "comparison": comparison,
            "experiment": {"pipeline": pipeline, "include_classification": False},
        },
    )
    capabilities = read_json(client, base + "/capabilities")
    if (pipeline == "ocr" or comparison) and not capabilities["ocr_available"]:
        assert response.status_code == 422, response.text
        assert not stub.get_requests() and len(ocr_calls()) == before
        return
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["scoring_version"] == "5"
    assert response.headers["Location"].endswith(run["id"])
    runs = [run]
    if comparison:
        assert run["comparison_id"]
        runs = read_json(client, f"{base}?comparison_id={run['comparison_id']}")
        assert {r["comparison_arm"] for r in runs} == {"direct_vision", "ocr"} and len(runs) == 2
        assert {r["dataset_fingerprint"] for r in runs} == {run["dataset_fingerprint"]}
    else:
        assert run["comparison_id"] is None
    for arm in runs:
        url = f"{base}/{arm['id']}"
        count = len(arm["model_ids"])
        wait_until(
            lambda url=url, count=count: read_json(client, url)["terminal"] == count,
            timeout_s=120,
        )
        results = read_json(client, url + "/results")
        assert {r["model_id"] for r in results} == set(arm["model_ids"])
        for result in results:
            assert result["success"], result
            assert result["scores"]["normalized"] == 1, result["scores"]
            assert result["expected"]["total"] == "3.86"
    requests = stub.get_requests()
    assert Counter((r["model"], r["kind"]) for r in requests) == Counter(calls)
    assert all((r["image_sha256"] is None) == (r["kind"] == "ocr") for r in requests)
    assert len(ocr_calls()) - before == sum(kind == "ocr" for _, kind in calls)
