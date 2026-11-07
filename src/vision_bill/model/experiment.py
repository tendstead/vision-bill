"""Request-scoped options for receipt experiments."""
from enum import Enum

from pydantic import BaseModel, Field


class ExtractionPipeline(str, Enum):
    UPLOAD = "upload"
    DIRECT = "direct_vision"
    OCR = "ocr"


class ExperimentOptions(BaseModel):
    pipeline: ExtractionPipeline = ExtractionPipeline.UPLOAD
    temperature: float | None = Field(default=None, ge=0, le=2)
    think: bool | None = None
    include_classification: bool = True
    allow_ocr_fallback: bool = False
