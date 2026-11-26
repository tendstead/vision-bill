import pytest
from pydantic import ValidationError

from vision_bill.config import LLMSettings, OcrSettings, WorkerSettings


def test_required_ocr_requires_enabled():
    with pytest.raises(ValidationError):
        OcrSettings(enabled=False, required=True)
    assert OcrSettings(enabled=True, required=True).required


def test_no_implicit_fallback():
    assert OcrSettings().enabled is False
    assert OcrSettings().allow_vision_fallback is False


def test_worker_attempt_budget_fits_claim_lease():
    worker = WorkerSettings()
    assert worker.attempt_timeout_seconds == 600
    assert worker.max_attempts == 3
    assert worker.retry_backoff_seconds == (30, 60)
    assert worker.retry_backoff_cap_seconds == 300
    assert worker.claim_lease_seconds == 900
    with pytest.raises(ValidationError):
        WorkerSettings(attempt_timeout_seconds=601)
    with pytest.raises(ValidationError):
        WorkerSettings(claim_lease_seconds=600)
    for change in (
        {"max_attempts": 0},
        {"max_attempts": 4},
        {"retry_backoff_seconds": [0, 60]},
        {"retry_backoff_cap_seconds": 0},
    ):
        with pytest.raises(ValidationError):
            WorkerSettings(**change)


def test_formatter_limit_is_separate_from_context_and_ocr_limits():
    llm = LLMSettings(host="local", api_key="none", model_name="model", temperature=0)
    assert llm.max_output_tokens == 32768
    for explicit_limit in (8192, 65536):
        overridden = LLMSettings.model_validate({**llm.model_dump(), "max_output_tokens": explicit_limit})
        assert overridden.max_output_tokens == explicit_limit
    with pytest.raises(ValidationError):
        LLMSettings(
            host="local", api_key="none", model_name="model", temperature=0, max_output_tokens=0
        )


@pytest.mark.parametrize(
    "maximum,delays,cap,attempt,expected",
    [
        (1, (), 300, 1, 30),
        (1, (), 300, 2, 60),
        (2, (20,), 300, 1, 20),
        (2, (20,), 45, 2, 45),
        (3, (50, 150), 40, 1, 40),
        (3, (30, 60), 300, 2, 60),
    ],
)
def test_retry_delay_uses_current_entries_or_defaults_for_persisted_attempts(
    maximum, delays, cap, attempt, expected,
):
    worker = WorkerSettings(
        max_attempts=maximum, retry_backoff_seconds=delays, retry_backoff_cap_seconds=cap,
    )
    assert worker.retry_delay(attempt) == expected
