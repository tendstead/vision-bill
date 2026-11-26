"""Business review contracts independent of persistence and I/O."""

from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from .ocr import Sha256, SourceRef

CHECKS_VERSION = "3"


class ReviewFinding(BaseModel):
    id: str
    code: str
    severity: Literal["integrity", "review", "info"]
    field_path: str
    source_refs: list[SourceRef] = Field(default_factory=list)
    comparison: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("comparison", mode="before")
    @classmethod
    def decimal_comparison(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: str(item) if isinstance(item, Decimal) else item for key, item in value.items()
            }
        return value


class FindingAcknowledgement(BaseModel):
    finding_id: str
    comment: str = Field(min_length=1)


class AmountPolicy(BaseModel):
    tax_basis: Literal["gross", "net", "unknown"] = "unknown"
    discount_treatment: Literal["included_in_lines", "subtract", "unknown"] = "unknown"
    tip_treatment: Literal["included", "add", "outside_receipt", "none", "unknown"] = "unknown"
    # Existing "printed" records use the line-item tax basis. "printed_net" is an
    # explicitly printed net subtotal for the whole transaction, across tax rates.
    subtotal_origin: Literal["printed", "printed_net", "derived_net", "derived_gross", "unknown"] = "unknown"
    tax_information: Literal["present", "explicit_zero", "missing"] = "missing"


class SourceView(BaseModel):
    view_id: UUID
    photo_index: int = Field(ge=0)
    coordinate_space: Literal["original", "working"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: Sha256
    url: str
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    inverse_mapping: None = None


class OriginalAvailability(BaseModel):
    # None is an honest legacy path observation, never a fabricated persisted UUID.
    part_id: UUID | None
    photo_index: int | None = Field(default=None, ge=0)
    available: bool
    sha256_matches: bool

    @model_validator(mode="after")
    def legacy_index(self) -> "OriginalAvailability":
        if self.part_id is None and self.photo_index is None:
            raise ValueError("legacy original availability requires photo_index")
        return self


class ReviewEvaluation(BaseModel):
    findings: list[ReviewFinding] = Field(default_factory=list)
    can_finalize: bool
    can_auto_verify: bool
    checks_version: str = CHECKS_VERSION
