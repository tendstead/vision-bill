"""Compatibility exports for the independent OCR wire contract."""

from ocr_contract import (
    CONTRACT_VERSION,
    BoundingBox,
    CoordinateSpace,
    GpuMemorySnapshot,
    OcrBlock,
    OcrCapabilities,
    OcrPageMetadata,
    OcrResponse,
    Point,
    Sha256,
    SourceRef,
    WorkerReadiness,
    WorkingImage,
    WorkingImageMetadata,
)

__all__ = [
    "CONTRACT_VERSION",
    "BoundingBox",
    "CoordinateSpace",
    "GpuMemorySnapshot",
    "OcrBlock",
    "OcrCapabilities",
    "OcrPageMetadata",
    "OcrResponse",
    "Point",
    "Sha256",
    "SourceRef",
    "WorkerReadiness",
    "WorkingImage",
    "WorkingImageMetadata",
]
