"""Durable benchmark results exercise the real migration, API and worker."""
from pathlib import Path

import pytest

from e2e.discovery import load_fixture_cases
from e2e.helpers import API, wait_until

pytestmark = pytest.mark.e2e


def admin(http_factory):
    client = http_factory()
    response = client.post(f"{API}/auth/login", json={"username": "admin", "password": "admin-e2e-pass"})
    assert response.status_code == 200, response.text
    return client


def verified_receipt(client):
    case = load_fixture_cases()[0]
    with Path(case.image_path).open("rb") as stream:
        response = client.post(f"{API}/images?bypass_review=true", files={"receipt": ("receipt.png", stream, "application/octet-stream")})
    assert response.status_code == 201, response.text
    return response.json()["receipt_id"], case


def test_experiment_persists_scores_and_uses_frozen_verified_data(http_factory):
    client = admin(http_factory)
    receipt_id, case = verified_receipt(client)
    response = client.post(f"{API}/benchmarks", json={"model_ids": ["e2e-vision"], "receipt_ids": [receipt_id], "experiment": {"temperature": 0, "include_classification": False}})
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["scoring_version"] == "3"
    assert run["experiment"]["options"]["temperature"] == 0
    run_id = run["id"]
    wait_until(lambda: client.get(f"{API}/benchmarks/{run_id}").json()["terminal"] == 1)
    results = client.get(f"{API}/benchmarks/{run_id}/results").json()
    assert len(results) == 1
    result = results[0]
    assert result["success"], result
    assert result["scores"]["normalized"] == 1, result["differences"]
    assert result["scores"]["classification"] is None
    assert result["metadata"]["image_count"] == 1
    assert result["metadata"]["used_pipeline"] == "direct_vision"
    assert result["metadata"]["calls"][0]["prompt_sha256"]
    assert result["expected"]["merchant_name"] == case.expected_receipt.merchant_name
    # Editing the source receipt later must not silently replace the run's target.
    edited = case.expected_receipt.model_dump(mode="json")
    edited["merchant_name"] = "Edited after benchmark creation"
    update = client.put(f"{API}/receipts/{receipt_id}", json=edited)
    assert update.status_code == 200, update.text
    reevaluate = client.post(f"{API}/benchmarks/{run_id}/reevaluate", json={"receipt_id": receipt_id, "model_id": "e2e-vision"})
    assert reevaluate.status_code == 200, reevaluate.text
    assert reevaluate.json()["expected"]["merchant_name"] == case.expected_receipt.merchant_name


def test_forced_ocr_rejected_without_extension_and_results_are_admin_only(http_factory):
    client = admin(http_factory)
    capabilities = client.get(f"{API}/benchmarks/capabilities").json()
    assert capabilities == {"ocr_available": False, "thinking_control": False}
    receipt_id, _ = verified_receipt(client)
    response = client.post(f"{API}/benchmarks", json={"model_ids": ["e2e-vision"], "receipt_ids": [receipt_id], "experiment": {"pipeline": "ocr"}})
    assert response.status_code == 422
    response = client.post(f"{API}/benchmarks", json={"model_ids": ["e2e-vision"], "receipt_ids": [receipt_id], "experiment": {"think": False}})
    assert response.status_code == 422
    anonymous = http_factory()
    assert anonymous.get(f"{API}/benchmarks/capabilities").status_code == 401
