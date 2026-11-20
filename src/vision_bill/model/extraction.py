"""Nullable extraction values and inference metadata, without database dependencies."""

from datetime import date as Date
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .item_category import SpendingCategory, llm_category
from .ocr import OcrPageMetadata, OcrResponse, Sha256, SourceRef, WorkingImage
from .receipt import Category, Currency, LineItem, Receipt, ReceiptTime
from .review import AmountPolicy, ReviewFinding, SourceView

EXTRACTION_SCHEMA_VERSION = "1"
SOURCE_PROJECTION_VERSION = "2"
PaymentMethod = Literal[
    "cash", "credit_card", "debit_card", "mobile_payment", "check", "other", "unknown"
]
EnrichmentReason = Literal[
    "invalid_response",
    "request_rejected",
    "not_dispatched",
    "classification_budget_exhausted",
    "classification_disabled",
    "incomplete_observations",
    "recovered_without_classification",
]


class ClassificationEnrichment(BaseModel):
    """Safe status and field paths for optional receipt and row labels."""

    model_config = ConfigDict(extra="forbid")
    state: Literal["not_recorded", "pending", "completed", "unavailable", "skipped"] = "not_recorded"
    reason_code: EnrichmentReason | None = None
    missing_fields: list[str] = Field(default_factory=list)


def assert_metadata_only(value: Any) -> None:
    """Reject transport image payloads from arbitrary persisted metadata."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("image bytes belong in saved artifacts")  # noqa: TRY004
    if isinstance(value, BaseModel):
        assert_metadata_only(value.model_dump(serialize_as_any=True))
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_metadata_only(key)
            if str(key).lower() in {"png_base64", "image_base64", "image_bytes"}:
                raise ValueError("raw image payload belongs in saved artifacts")
            assert_metadata_only(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            assert_metadata_only(item)


class DraftLineItem(BaseModel):
    description: str | None = None
    spending_category: SpendingCategory | None = None
    quantity: Annotated[Decimal, Field(gt=0)] | None = None
    unit_price: Decimal | None = None
    total_price: Decimal | None = None
    tags: list[str] | None = None

    @field_validator("spending_category", mode="before")
    @classmethod
    def category(cls, value: object) -> str | None:
        return None if value is None else llm_category(value)

    @field_validator("tags")
    @classmethod
    def tags_normalized(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else LineItem._normalize_tags(value)


class DraftTaxLine(BaseModel):
    name: str | None = None
    rate: Decimal | None = None
    amount: Annotated[Decimal, Field(ge=0)] | None = None


class ExtractionDraft(BaseModel):
    """Missing observations stay null, including fields with final Receipt defaults."""

    confidence: Annotated[int, Field(ge=0, le=100)] | None = None
    merchant_name: str | None = None
    merchant_address: str | None = None
    receipt_number: str | None = None
    date: Date | None = None
    time: ReceiptTime | None = None
    currency: Currency | None = None
    category: Category | None = None
    line_items: list[DraftLineItem] | None = None
    taxes: list[DraftTaxLine] | None = None
    subtotal: Decimal | None = None
    discount_total: Annotated[Decimal, Field(ge=0)] | None = None
    tax_total: Annotated[Decimal, Field(ge=0)] | None = None
    tip: Annotated[Decimal, Field(ge=0)] | None = None
    total: Decimal | None = None
    payment_method: PaymentMethod | None = None


class ReceiptInput(BaseModel):
    """Formatter input contains observations, never expected/reference receipts or PNG bytes."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["direct_vision", "ocr"]
    image_paths: list[str] = Field(min_length=1)
    ocr_pages: list[OcrPageMetadata] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)

    @field_validator("ocr_pages", mode="before")
    @classmethod
    def metadata_only(cls, value: Any) -> Any:
        if isinstance(value, list):
            # Revalidation into the metadata type removes the wire-only PNG field.
            return [page.model_dump() if isinstance(page, OcrResponse) else page for page in value]
        return value


class PipelineProvenance(BaseModel):
    requested_pipeline: str
    used_pipeline: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    source_image_count: int = Field(ge=1)
    ordered_source_sha256: list[Sha256]
    model_id: str
    requested_model_id: str | None = None  # Unknown for historical saved envelopes.
    model_digest: str | None = None
    profile_id: str | None = None
    runtime_fingerprint: str | None = None


class InferenceTelemetry(BaseModel):
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    stage_ms: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )
    calls: list[dict[str, Any]] = Field(default_factory=list)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    max_output_tokens: int = Field(gt=0)
    context_window_tokens: int | None = Field(default=None, gt=0)
    ocr_max_new_tokens: int | None = Field(default=None, gt=0)


class DerivedInputView(BaseModel):
    """Internal Direct/fallback transport derivative; persist bytes before exposing metadata."""

    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    working_image: WorkingImage


class _InferenceMetadata(BaseModel):
    source_projection_version: str = "1"  # Missing identity in historical envelopes stays legacy.
    raw_field_sources: dict[str, list[SourceRef]] = Field(default_factory=dict)
    draft: ExtractionDraft | None = None
    receipt: Receipt | None = None
    amount_policy: AmountPolicy = Field(default_factory=AmountPolicy)
    field_sources: dict[str, list[SourceRef]] = Field(default_factory=dict)
    overlap_decisions: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    provenance: PipelineProvenance
    telemetry: InferenceTelemetry
    classification_enrichment: ClassificationEnrichment = Field(
        default_factory=ClassificationEnrichment
    )


class InferenceEnvelope(_InferenceMetadata):
    ocr_pages: list[OcrResponse] = Field(default_factory=list, repr=False)
    input_views: list[DerivedInputView] = Field(default_factory=list, repr=False, exclude=True)


class PersistedInferenceEnvelope(_InferenceMetadata):
    """Validated observation and attempt metadata. Saved view manifests live separately."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    ocr_pages: list[OcrPageMetadata] = Field(default_factory=list, repr=False)

    @model_validator(mode="before")
    @classmethod
    def metadata_only(cls, value: Any) -> Any:
        assert_metadata_only(value)
        return value


def pack_inference_metadata(envelope: InferenceEnvelope) -> PersistedInferenceEnvelope:
    """Project wire OCR pages and exclude independently stored input derivatives."""
    payload = envelope.model_dump(exclude={"ocr_pages", "input_views"})
    payload["ocr_pages"] = [
        page.model_dump(exclude={"working_image": {"png_base64"}}) for page in envelope.ocr_pages
    ]
    return PersistedInferenceEnvelope.model_validate(payload)


class DraftView(BaseModel):
    image_id: UUID
    revision: int = Field(ge=1)
    document: ExtractionDraft
    amount_policy: AmountPolicy = Field(default_factory=AmountPolicy)
    findings: list[ReviewFinding] = Field(default_factory=list)
    source_views: list[SourceView] = Field(default_factory=list)
    finalized_receipt_id: UUID | None = None
    field_sources: dict[str, list[SourceRef]] = Field(default_factory=dict)
    overlap_decisions: list[dict[str, Any]] = Field(default_factory=list)
