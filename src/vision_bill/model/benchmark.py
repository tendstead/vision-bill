from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .experiment import ExperimentOptions


class CouncilPolicy(str, Enum):
    ALL = "all"
    MATERIAL = "material"
    CUSTOM = "custom"


def require_selected_models(model_ids: list[str] | None) -> list[str]:
    if not isinstance(model_ids, list) or not model_ids or any(
        not isinstance(model_id, str) or not model_id.strip() for model_id in model_ids
    ):
        raise ValueError("Select at least one explicit benchmark model")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Selected benchmark models must be unique")
    return model_ids


class BenchmarkCreate(BaseModel):
    """A durable, local-only benchmark request."""

    experiment: ExperimentOptions = Field(default_factory=ExperimentOptions)
    comparison: bool = False
    model_ids: list[str] = Field(min_length=1)
    receipt_ids: list[UUID] | None = None
    case_ids: list[UUID] | None = None
    category: str | None = None
    max_source_confidence: int | None = Field(default=None, ge=0, le=100)
    limit: int | None = Field(default=None, ge=1)
    request_timeout_seconds: int = Field(default=600, ge=1, le=600)
    council_policy: CouncilPolicy = CouncilPolicy.ALL
    council_absolute_threshold: Decimal | None = Field(default=None, ge=0)
    council_relative_threshold: Decimal | None = Field(default=None, ge=0)
    apply_council_flags: bool = False

    @field_validator("receipt_ids", "case_ids")
    @classmethod
    def unique_receipt_selection(cls, receipt_ids: list[UUID] | None) -> list[UUID] | None:
        return list(dict.fromkeys(receipt_ids)) if receipt_ids is not None else None

    @field_validator("model_ids")
    @classmethod
    def explicit_model_selection(cls, model_ids: list[str]) -> list[str]:
        return require_selected_models(model_ids)


class BenchmarkRun(BaseModel):
    id: UUID
    comparison_id: UUID | None = None
    comparison_arm: Literal["direct_vision", "ocr"] | None = None
    status: str
    model_ids: list[str]
    receipt_ids: list[UUID]
    case_ids: list[UUID] = Field(default_factory=list)
    case_names: dict[UUID, str] = Field(default_factory=dict)
    dataset_fingerprint: str
    prompt_version: str
    scoring_version: str
    council_policy: CouncilPolicy
    apply_council_flags: bool
    created_at: datetime
    completed_at: datetime | None = None
    experiment: dict[str, Any] | None = None


class BenchmarkSummary(BaseModel):
    model_id: str
    model_digest: str | None = None
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    average_score: float | None = None
    all_case_average_score: float | None = None
    full_receipt_correct_rate: float | None = None
    average_confidence: float | None = None
    average_attempts: float | None = None
    average_latency_ms: float | None = None
    council_candidates: int = 0
    council_findings: int = 0


class BenchmarkStatus(BaseModel):
    run: BenchmarkRun
    queued: int
    running: int
    waiting_for_model: int
    retrying: int
    terminal: int
    summaries: list[BenchmarkSummary]


class ReferenceConfirmation(BaseModel):
    expected_review_revision: int = Field(ge=1)
    expected_document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ReferencePreview(BaseModel):
    currently_qualified: bool = False
    receipt_identity: UUID
    owner_id: UUID | None
    review_revision: int
    document_sha256: str
    expected_sha256: str
    expected: dict[str, Any]
    scored_fields: list[str]
    sources: list[dict[str, Any]]
    schema_version: str
    scoring_version: str
    findings: list[dict[str, Any]]
    acknowledgements: list[dict[str, Any]]


class GoldSnapshot(ReferencePreview):
    id: UUID
    receipt_id: UUID | None
    qualified_by: UUID | None
    qualified_at: datetime
    owner_identity: UUID | None
    actor_identity: UUID
