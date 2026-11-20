"""Error paths: unsupported type, provider down, LLM self-correction retry."""

from pathlib import Path

import pytest

from e2e.discovery import first_case
from e2e.helpers import (
    API,
    register_user,
    wait_until,
)

pytestmark = pytest.mark.e2e


def test_unsupported_image_type_415(http_factory, tmp_path: Path) -> None:
    client, _ = register_user(http_factory, "err415")
    not_image = tmp_path / "not-an-image.txt"
    not_image.write_text("hello", encoding="utf-8")
    with not_image.open("rb") as fh:
        response = client.post(
            f"{API}/images", files={"receipt": ("not-an-image.txt", fh, "text/plain")}
        )
    assert response.status_code == 415
    assert "Unsupported image type" in response.json()["detail"]


def test_provider_down_keeps_image_pending(http_factory, stub) -> None:
    case = first_case()
    if case is None:
        pytest.skip("no e2e fixtures registered")
    client, _ = register_user(http_factory, "errdown")
    stub.set_mode("down")

    # Upload a real (valid) image; with the provider down it must queue (202).
    with case.image_path.open("rb") as fh:
        response = client.post(
            f"{API}/images", files={"receipt": (case.image_path.name, fh, "image/png")}
        )
    assert response.status_code == 202
    image_id = response.json()["id"]

    # Trigger a cycle: with the provider down the image must stay pending
    # (no terminal failure state).
    client.post(f"{API}/images/analyze")
    row = client.get(f"{API}/images/{image_id}").json()
    assert row["status"] == "pending"
    assert row["receipt_id"] is None


def test_self_correction_retry_recovers(http_factory, stub) -> None:
    """repair_first mode: the stub's first answer is broken JSON; the
    provider's retry loop must feed the error back and get valid JSON."""
    case = first_case()
    if case is None:
        pytest.skip("no e2e fixtures registered")

    client, _ = register_user(http_factory, "repair")
    stub.set_mode("repair_first")
    with case.image_path.open("rb") as fh:
        response = client.post(
            f"{API}/images", files={"receipt": (case.image_path.name, fh, "image/png")}
        )
    assert response.status_code == 202
    image_id = response.json()["id"]

    def check():
        row = client.get(f"{API}/images/{image_id}").json()
        return row if row["status"] in ("analyzed", "failed") else None

    row = wait_until(check, timeout_s=60.0)
    assert row["status"] == "analyzed", row

    served = [r for r in stub.get_requests() if r["kind"] in ("direct", "ocr")]
    broken = [r for r in served if r["served"] == "broken"]
    good = [r for r in served if r["served"] == "fixture"]
    assert len(broken) >= 1, "expected at least one broken first attempt"
    assert len(good) >= 1, "expected a corrected retry"
