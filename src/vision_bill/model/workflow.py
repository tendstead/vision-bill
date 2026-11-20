"""Submission and mutation contracts. Internal claims never serve as response DTOs."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .db.image import ClaimPurpose, ImagePart, ImageRow
from .db.receipt import ReceiptRow
from .extraction import (
    ClassificationEnrichment,
    InferenceEnvelope,
    PersistedInferenceEnvelope,
    assert_metadata_only,
)
from .ocr import OcrPageMetadata, OcrResponse, Sha256, SourceRef, WorkingImageMetadata
from .review import FindingAcknowledgement, ReviewEvaluation, ReviewFinding, SourceView

TERMINAL_STATES = frozenset({"analyzed", "needs_review", "failed", "timed_out", "unreadable"})


class SourcePartBinding(BaseModel):
    """Original-file hash attested by the application, in exact capture order."""

    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    image_path: str = Field(min_length=1)


class PersistedSourcePart(SourcePartBinding):
    part_id: UUID


class PersistedWorkingView(BaseModel):
    """Caller attests that the exact validated PNG was saved before DB commit.

    DB owns identity/metadata checks, never file or codec I/O. Artifact creation
    and cleanup of unreferenced files belong to the caller/FileProvider.
    """

    model_config = ConfigDict(extra="forbid")
    part_id: UUID
    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    artifact_path: str = Field(min_length=1)
    metadata: WorkingImageMetadata

    @model_validator(mode="before")
    @classmethod
    def no_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            assert_metadata_only(value)
        return value


class AccessScope(BaseModel):
    user_id: UUID
    can_see_all: bool = False


class SubmissionStatus(BaseModel):
    """Explicit public projection; extra DB fields are intentionally discarded."""

    id: UUID
    original_filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    image_path: str | None = None
    thumbnail_path: str | None = None
    status: str = "pending"
    error: str | None = None
    receipt_id: UUID | None = None
    bypass_review: bool = False
    user_id: UUID | None = None
    created_at: datetime | None = None
    analyzed_at: datetime | None = None
    processing_at: datetime | None = None
    additional_images: list[ImagePart] = Field(default_factory=list)
    model_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=3)
    retry_at: datetime | None = None
    result_code: str | None = None
    wait_reason: str | None = None
    draft_available: bool = False
    observations_available: bool = False
    latest_extraction_id: UUID | None = None
    source_views: list[SourceView] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATES


class AnalysisClaim(BaseModel):
    image: ImageRow
    generation: int = Field(ge=1)
    attempt_number: int = Field(ge=1)
    claim_token: UUID = Field(repr=False)
    lease_expires_at: datetime
    purpose: ClaimPurpose = "inference"
    checkpoint_id: UUID | None = None


class RetryRequest(BaseModel):
    expected_generation: int = Field(ge=1)
    request_key: str = Field(min_length=1)
    policy: Literal["preserve", "current"] = "preserve"


class DraftPatch(BaseModel):
    expected_revision: int = Field(ge=1)
    changes: dict[str, Any]


class DraftFinalize(BaseModel):
    expected_revision: int = Field(ge=1)
    request_key: str = Field(min_length=1)


class ReviewPreview(BaseModel):
    """Current diagnostics and compare-and-swap revisions for explicit human review."""

    operation_revision: int = Field(ge=1)
    review_revision: int = Field(ge=1)
    findings: list[ReviewFinding]


class ReviewRequest(BaseModel):
    expected_operation_revision: int | None = Field(default=None, ge=1)
    expected_review_revision: int | None = Field(default=None, ge=1)
    acknowledgements: list[FindingAcknowledgement] = Field(default_factory=list)


class SubmissionCreate(BaseModel):
    """Upload identity and the first resolved execution snapshot.

    The caller fingerprints ordered ORIGINAL hashes plus analysis-relevant
    explicitly supplied options/model selection. requested_options contains
    those explicit values (including an explicit bypass flag); effective_options,
    model_id and bypass_review preserve the first resolution of server defaults.
    Generated paths, thumbnails and database IDs are never request identity.
    """

    id: UUID
    parts: list[ImagePart] = Field(min_length=1)
    source_manifest: list[dict[str, Any]]
    requested_options: dict[str, Any]
    effective_options: dict[str, Any]
    request_fingerprint: str
    model_id: str | None = None
    bypass_review: bool = False

    @model_validator(mode="after")
    def metadata_only(self) -> "SubmissionCreate":
        assert_metadata_only(self.model_dump())
        return self


class EnqueueResult(BaseModel):
    submission: SubmissionStatus
    created: bool


class OcrCheckpoint(BaseModel):
    ordered_source_sha256: list[Sha256]
    profile_id: str
    runtime_fingerprint: str
    pages: list[dict[str, Any]]
    persisted_views: list[PersistedWorkingView] = Field(default_factory=list)

    @field_validator("pages", mode="before")
    @classmethod
    def metadata_only(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [
                OcrPageMetadata.model_validate(
                    page.model_dump() if isinstance(page, OcrResponse) else page
                ).model_dump(mode="json")
                for page in value
            ]
        return value


class CheckpointExecutionIdentity(BaseModel):
    """Credential-free execution identity as observed by the producing attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    inference_identity: str = Field(pattern=r"^(ollama|openai):[0-9a-f]{64}$")
    semantic_policy_sha256: Sha256
    model_id: str = Field(min_length=1)
    model_digest: str | None
    prompt_version: str = Field(min_length=1)
    extraction_schema_version: str = Field(min_length=1)
    source_projection_version: str = Field(min_length=1)
    enrichment_policy_version: Literal["1"] = "1"


class ObservationCheckpoint(BaseModel):
    """All-photo observations and artifact identities, before optional enrichment."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    version: Literal[1] = 1
    checkpoint_id: UUID
    producer_generation: int = Field(ge=1)
    producer_attempt_number: int = Field(ge=1)
    eligible_generation: int | None = Field(ge=1)
    source_parts: list[PersistedSourcePart] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    execution_identity: CheckpointExecutionIdentity
    envelope: PersistedInferenceEnvelope
    persisted_views: list[PersistedWorkingView]

    @model_validator(mode="before")
    @classmethod
    def metadata_only(cls, value: Any) -> Any:
        assert_metadata_only(value)
        return value

    @model_validator(mode="after")
    def all_sources_bound(self) -> "ObservationCheckpoint":
        if self.eligible_generation not in (None, self.producer_generation):
            raise ValueError("checkpoint can only be eligible in its producer generation")
        if self.envelope.draft is None or self.envelope.receipt is None:
            raise ValueError("checkpoint requires complete observations")
        provenance = self.envelope.provenance
        identity = self.execution_identity
        if (
            self.model_id != identity.model_id or self.model_id != provenance.model_id
            or identity.model_digest != provenance.model_digest
            or identity.source_projection_version != self.envelope.source_projection_version
        ):
            raise ValueError("checkpoint execution identity mismatch")

        sources = self.source_parts
        if (
            provenance.source_image_count != len(sources)
            or provenance.ordered_source_sha256 != [part.source_sha256 for part in sources]
            or [part.photo_index for part in sources] != list(range(len(sources)))
            or len({part.part_id for part in sources}) != len(sources)
        ):
            raise ValueError("checkpoint ordered source manifest mismatch")
        if self.envelope.field_sources.get("input") != [
            SourceRef(
                part_id=part.part_id, photo_index=part.photo_index,
                source_sha256=part.source_sha256, coordinate_space="original",
            )
            for part in sources
        ]:
            raise ValueError("checkpoint requires all ordered original source references")

        views = self.persisted_views
        if (
            len(views) != len(sources)
            or len({view.metadata.view_id for view in views}) != len(views)
            or any(
                view.part_id != part.part_id or view.photo_index != part.photo_index
                or view.source_sha256 != part.source_sha256
                for part, view in zip(sources, views, strict=True)
            )
        ):
            raise ValueError("checkpoint requires an ordered working view for each source")
        pages = self.envelope.ocr_pages
        if provenance.used_pipeline == "ocr" and not provenance.fallback_used:
            if (
                not provenance.profile_id or not provenance.runtime_fingerprint
                or len(pages) != len(sources)
                or any(
                    page.photo_index != view.photo_index or page.source_sha256 != view.source_sha256
                    or page.working_image != view.metadata or not page.text.strip()
                    or page.profile_id != provenance.profile_id
                    or page.runtime_fingerprint != provenance.runtime_fingerprint
                    for page, view in zip(pages, views, strict=True)
                )
            ):
                raise ValueError("checkpoint requires complete matching views for all OCR sources")
        elif provenance.used_pipeline != "direct_vision" or provenance.fallback_used or pages:
            raise ValueError("checkpoint pipeline mismatch")
        return self

    def recovered_envelope(self) -> PersistedInferenceEnvelope:
        enrichment = self.envelope.classification_enrichment
        if enrichment.state != "pending":
            return self.envelope
        return self.envelope.model_copy(update={
            "classification_enrichment": ClassificationEnrichment(
                state="unavailable", reason_code="recovered_without_classification",
                missing_fields=enrichment.missing_fields,
            ),
        })


class CheckpointCompletionOutcome(BaseModel):
    """Final observation projection after a separate recovery reservation."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    envelope: PersistedInferenceEnvelope
    review: ReviewEvaluation
    persisted_views: list[PersistedWorkingView]

    @model_validator(mode="after")
    def require_complete(self) -> "CheckpointCompletionOutcome":
        if (
            self.envelope.receipt is None or self.envelope.draft is None
            or self.envelope.classification_enrichment.state == "pending"
            or not self.review.can_finalize
        ):
            raise ValueError("checkpoint completion requires a finalizable receipt")
        return self


class AttemptOutcome(BaseModel):
    kind: Literal["complete", "partial", "retry", "terminal"]
    envelope: InferenceEnvelope | None = None
    result_code: str | None = None
    retry_delay_seconds: int | None = Field(default=None, ge=0)
    review: ReviewEvaluation | None = None
    persisted_views: list[PersistedWorkingView] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_result(self) -> "AttemptOutcome":
        if self.kind == "complete" and (
            self.envelope is None
            or self.envelope.receipt is None
            or self.review is None
            or not self.review.can_finalize
        ):
            raise ValueError("complete requires a receipt and finalizable review")
        if self.kind == "partial" and (self.envelope is None or self.envelope.draft is None):
            raise ValueError("partial requires an extraction draft")
        return self


class CommitResult(BaseModel):
    stale: bool
    submission: SubmissionStatus | None = None
    receipt: ReceiptRow | None = None


class DeletedResource(BaseModel):
    """Owned file identities returned only after the authorized DB deletion commits."""
    image_id: UUID | None = None
    receipt_id: UUID | None = None
    parts: list[ImagePart] = Field(default_factory=list)
    working_views: list[PersistedWorkingView] = Field(default_factory=list)
    thumbnail_sha256: dict[str, str | None] = Field(default_factory=dict)
