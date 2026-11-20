"""Versioned OCR wire types, portable to the sidecar without application imports."""

from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import limits
from .wire_json import check_encoded_metadata, preflight_wire_metadata

CONTRACT_VERSION: Literal[2] = 2

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
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CoordinateSpace = Literal["original", "working", "unknown"]
Point = Annotated[
    list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=2, max_length=2)
]
BoundingBox = Annotated[list[Point], Field(min_length=3)]


class SourceRef(BaseModel):
    part_id: UUID
    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    view_id: UUID | None = None
    coordinate_space: CoordinateSpace = "unknown"
    block_id: str | None = None
    bbox: BoundingBox | None = None
    text_span: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] | None = None

    @field_validator("text_span")
    @classmethod
    def ordered_span(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and value[1] < value[0]:
            raise ValueError("text span end must not precede start")
        return value


class OcrBlock(BaseModel):
    id: str
    label: str
    text: str
    bbox: BoundingBox


class WorkingImageMetadata(BaseModel):
    """Persisted view identity and geometry; safe for formatter and public metadata."""

    view_id: UUID
    media_type: Literal["image/png"] = "image/png"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: Sha256
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    inverse_mapping: None = None


class WorkingImage(WorkingImageMetadata):
    png_base64: str = Field(repr=False)


class OcrPageMetadata(BaseModel):
    """Validated OCR page after the working PNG has been persisted separately."""

    contract_version: Literal[2] = CONTRACT_VERSION
    request_id: UUID
    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    text: str
    blocks: list[OcrBlock]
    working_image: WorkingImageMetadata
    profile_id: str
    runtime_fingerprint: str
    telemetry: dict[str, Any] = Field(default_factory=dict)


class OcrResponse(OcrPageMetadata):
    model_config = ConfigDict(extra="forbid")
    working_image: WorkingImage

    def check_wire_bounds(self) -> None:
        """Bound the transcription and evidence without rewriting either."""
        view = self.working_image
        limits.check_working_image_dimensions(view.width, view.height)
        limits.check_working_png_base64(view.png_base64)
        # The bounded measuring traversal leaves the wire PNG framing intact
        # while excluding its contents. All remaining fields count.
        check_encoded_metadata(
            self.model_copy(update={"working_image": view.model_copy(update={"png_base64": ""})})
        )
        text = self.text.strip()
        if not text or len(text) > limits.MAX_OCR_TEXT_CHARS:
            raise ValueError("ocr_invalid_recognized_text")
        if any(
            not (0 <= x <= view.width and 0 <= y <= view.height)
            for block in self.blocks
            for x, y in block.bbox
        ):
            raise ValueError("ocr_invalid_block_geometry")

    def dump_checked_wire(self) -> bytes:
        """Emit compact UTF-8 JSON after the producer has checked the page."""
        payload = self.__pydantic_serializer__.to_json(self)
        limits.check_ocr_response_size(len(payload))
        return payload

    @classmethod
    def parse_wire(cls, payload: bytes | bytearray) -> "OcrResponse":
        limits.check_ocr_response_size(len(payload))
        preflight_wire_metadata(payload)
        # Unknown nested wire fields would otherwise disappear before the
        # canonical measurement. Persisted metadata keeps its own validation.
        result = cls.model_validate_json(payload, extra="forbid")
        result.check_wire_bounds()
        return result


class OcrCapabilities(BaseModel):
    enabled: bool
    configured: bool
    reachable: bool
    ready: bool
    can_load: bool = False
    profile_id: str | None = None
    runtime_fingerprint: str | None = None
    reason_code: str | None = None
    ocr_max_new_tokens: int | None = Field(default=None, gt=0)


class GpuMemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    accelerator: Literal["cuda", "rocm"]
    gpu_index: int = Field(ge=0, le=1024)
    free_bytes: int = Field(ge=0)
    total_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_free(self) -> Self:
        if self.free_bytes > self.total_bytes:
            raise ValueError("Reported free GPU memory exceeds total GPU memory")
        return self


class WorkerReadiness(BaseModel):
    runtime_fingerprint: str | None = None
    ready: bool
    generation: int = Field(ge=1)
    profile_id: str | None = None
    reason_code: str | None = None
    preload_memory: GpuMemorySnapshot | None = None
