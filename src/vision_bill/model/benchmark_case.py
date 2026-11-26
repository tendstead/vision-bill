"""Public benchmark case contracts. Source paths and hashes stay server-side."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .extraction import ExtractionDraft

CaseName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class BenchmarkCaseSummary(BaseModel):
    id: UUID
    owner_id: UUID
    name: str
    revision: int
    confirmed: bool
    photo_count: int
    created_at: datetime
    updated_at: datetime
    confirmed_by: UUID | None
    confirmed_at: datetime | None


class BenchmarkCase(BenchmarkCaseSummary):
    expected: ExtractionDraft


class BenchmarkCaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1)
    name: CaseName | None = None
    expected: ExtractionDraft | None = None
    confirm: bool = Field(False, strict=True)

    @field_validator("expected")
    @classmethod
    def bounded_expected(cls, draft: ExtractionDraft | None) -> ExtractionDraft | None:
        if draft is not None and (
            len(draft.line_items or []) > 500
            or len(draft.taxes or []) > 500
            or len(draft.model_dump_json(exclude_none=True).encode()) > 200_000
        ):
            raise ValueError(
                "Expected values exceed the case limit of 500 rows per group or 200 KB"
            )
        return draft
