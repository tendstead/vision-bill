"""Readiness proves authenticated database use and both healthy test dependencies."""

import os
import time

import httpx


def check() -> None:
    base = os.environ["VB_E2E_BASE_URL"]
    with httpx.Client(base_url=base, timeout=3, trust_env=False) as client:
        response = client.post(
            "/api/v1/auth/login", json={"username": "admin", "password": "admin-e2e-pass"}
        )
        response.raise_for_status()
        client.get("/api/v1/images").raise_for_status()
        models = client.get("/api/v1/llm/models")
        models.raise_for_status()
        assert models.json(), "no discovered model"
        client.get(os.environ["VB_E2E_STUB_URL"] + "/v1/models").raise_for_status()
        ready = client.get(os.environ["VB_E2E_OCR_URL"] + "/ready")
        info = client.get(os.environ["VB_E2E_OCR_URL"] + "/info")
        ready.raise_for_status()
        info.raise_for_status()
        assert ready.json()["ready"] and info.json()["contract_version"] == 2
        assert ready.json()["profile_id"] == info.json()["worker"]["profile_id"]


def main() -> int:
    deadline = time.monotonic() + 180
    failure = "not started"
    while time.monotonic() < deadline:
        try:
            check()
            print("Owned E2E app, database, LLM and OCR are ready")
            return 0
        except (httpx.HTTPError, ValueError, AssertionError) as exc:
            failure = type(exc).__name__ + ": " + str(exc)
            time.sleep(1)
    print("E2E readiness failed: " + failure)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
