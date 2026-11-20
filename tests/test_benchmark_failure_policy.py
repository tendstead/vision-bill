"""Transport/availability retries must never turn source loss into automatic inference."""

import httpx
import pytest
from ollama import ResponseError
from openai import APIConnectionError

from vision_bill.model.ocr_errors import OcrGpuOutOfMemoryError, OcrWorkerFailure
from vision_bill.provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError
from vision_bill.service.benchmark_service import BenchmarkService
from vision_bill.service.experiment_execution import ExperimentFailure
from vision_bill.service.extraction_pipeline import (
    OcrRuntimeChangedError,
    OriginalSourceChangedError,
)


@pytest.mark.parametrize("failure,retryable", [
    (AnalysisTimeoutError("deadline"), True),
    (httpx.ConnectError("connection"), True),
    (APIConnectionError(request=httpx.Request("POST", "http://local")), True),
    (ResponseError("busy", 503), True),
    (ResponseError("limited", 429), True),
    (ResponseError("invalid", 400), False),
    (OcrRuntimeChangedError("generation changed"), True),
    (ValueError("ocr_unavailable"), True),
    (OcrGpuOutOfMemoryError(), True),
    (ValueError("ocr_gpu_out_of_memory"), True),
    (OcrWorkerFailure("ocr_model_load_failed"), True),
    (ValueError("ocr_worker_crashed"), True),
    (ValueError("ocr_deadline_exceeded"), True),
    # Historical page_failed includes blank/invalid pages; retries remain bounded.
    (ValueError("ocr_page_failed"), True),
    (ValueError("formatter_model_unavailable"), True),
    (ValueError("vision_model_unavailable"), True),
    (ValueError("vision_fallback_unavailable"), True),
    (FileNotFoundError("source"), False),
    (OriginalSourceChangedError("source"), False),
    (UnreadableReceiptError("unreadable"), False),
    (ValueError("receipt_incomplete"), False),
    (ValueError("schema mismatch"), False),
])
def test_failure_classification_preserves_wrapped_causes(failure, retryable):
    assert BenchmarkService._retryable(failure) is retryable
    wrapped = ExperimentFailure(str(failure), {"calls": []})
    wrapped.__cause__ = failure
    assert BenchmarkService._retryable(wrapped) is retryable
