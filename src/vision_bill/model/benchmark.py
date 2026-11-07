from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .experiment import ExperimentOptions


class CouncilPolicy(str, Enum):
    ALL = "all"
    MATERIAL = "material"
    CUSTOM = "custom"


class BenchmarkCreate(BaseModel):
    """A durable, local-only benchmark request."""

    experiment: ExperimentOptions = Field(default_factory=ExperimentOptions)
    model_ids: list[str] | None = None
    receipt_ids: list[UUID] | None = None
    category: str | None = None
    max_source_confidence: int | None = Field(default=None, ge=0, le=100)
    limit: int | None = Field(default=None, ge=1)
    request_timeout_seconds: int = Field(default=300, ge=1)
    council_policy: CouncilPolicy = CouncilPolicy.ALL
    council_absolute_threshold: Decimal | None = Field(default=None, ge=0)
    council_relative_threshold: Decimal | None = Field(default=None, ge=0)
    apply_council_flags: bool = False


class BenchmarkRun(BaseModel):
    id: UUID
    status: str
    model_ids: list[str]
    receipt_ids: list[UUID]
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
