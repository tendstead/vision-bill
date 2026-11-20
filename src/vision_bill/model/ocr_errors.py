"""Safe application errors reported by the native OCR worker."""

OCR_GPU_OUT_OF_MEMORY = "ocr_gpu_out_of_memory"


# Only these exact native status/detail pairs can enter durable job diagnostics.
OCR_HTTP_ERRORS = {
    502: {"ocr_inference_failed", "ocr_invalid_result"},
    503: {
        OCR_GPU_OUT_OF_MEMORY, "ocr_worker_crashed", "ocr_model_load_failed",
        "ocr_runtime_identity_mismatch", "ocr_startup_timeout", "ocr_init_failed",
    },
    504: {"ocr_deadline_exceeded"},
}
OCR_RESULT_STATUS = {code: status for status, codes in OCR_HTTP_ERRORS.items() for code in codes}
OCR_RESULT_CODES = frozenset(OCR_RESULT_STATUS)


class OcrWorkerFailure(RuntimeError):
    """A bounded, recognized failure from the load or inference endpoint."""


class OcrGpuOutOfMemoryError(OcrWorkerFailure):
    """The native worker reported a device allocation failure."""

    def __init__(self) -> None:
        super().__init__(OCR_GPU_OUT_OF_MEMORY)


def raise_for_worker_reason(reason: str | None) -> None:
    """Preserve known native failure reasons observed during readiness checks."""
    if reason == OCR_GPU_OUT_OF_MEMORY:
        raise OcrGpuOutOfMemoryError
    if reason is not None and reason in OCR_RESULT_CODES:
        raise OcrWorkerFailure(reason)
