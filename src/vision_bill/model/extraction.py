"""Nullable extraction values and inference metadata, without database dependencies."""

from datetime import date as Date
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .item_category import SpendingCategory, llm_category
from .ocr import OcrPageMetadata, OcrResponse, Sha256, SourceRef, WorkingImage
from .receipt import Category, Currency, LineItem, MerchantDisplayName, Receipt, ReceiptTime
from .review import AmountPolicy, ReviewFinding, SourceView

# Completion-only recovery uses saved observations and never reruns optional classification.
OBSERVATION_PROMPT_VERSION = "6"
# Version 3 captures printed per-position VAT evidence for deterministic amount derivation.
EXTRACTION_SCHEMA_VERSION = "3"
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
    description: str | None = Field(
        None,
        description=(
            "Printed product or service name, including distinguishing size, colour or model "
            "from its detail lines. Omit price and VAT percentage. Use a SKU as the name "
            "only when it is the sole printed item identifier."
        ),
    )
    spending_category: SpendingCategory | None = None
    quantity: Decimal | None = Field(
        None, gt=0,
        description=(
            "Printed count or billed weight/volume. A package size in the name is not the "
            "quantity purchased. Leave null if unstated."
        ),
    )
    unit_price: Decimal | None = Field(
        None,
        description=(
            "Printed price per unit of the billed quantity, e.g. price per kg for weighed "
            "goods. Leave null if the receipt gives only a row amount."
        ),
    )
    total_price: Decimal | None = Field(
        None,
        description=(
            "Signed amount charged for the whole row. Preserve printed signs on returns, "
            "deposits and credits."
        ),
    )
    tax_code: str | None = Field(
        None, max_length=32, description="Tax band printed on this position, e.g. A or B."
    )
    tax_rate: Decimal | None = Field(
        None, ge=0, le=1,
        description="Rate printed on this position, as a fraction. A letter alone is not a rate.",
    )
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
    name: str | None = Field(
        None,
        description=(
            "Printed tax name or band label from this tax-summary row, e.g. VAT, MwSt or A. "
            "Keep different bands separate."
        ),
    )
    rate: Decimal | None = Field(
        None,
        description=(
            "Fraction of the taxable base, from 0 to 1: 19% = 0.19, 7% = 0.07, 1% = 0.01. "
            "Use 0 only for an explicitly zero or exempt rate. Use null for an unknown rate."
        ),
    )
    amount: Decimal | None = Field(
        None, ge=0,
        description=(
            "Money in this row's tax/VAT amount column. The net taxable base and the gross "
            "amount including tax are separate columns."
        ),
    )


class ExtractionDraft(BaseModel):
    """Missing observations stay null, including fields with final Receipt defaults."""

    confidence: int | None = Field(
        None, ge=0, le=100,
        description=(
            "Your confidence in the extraction as a whole, as an integer percent from 0 to "
            "100."
        ),
    )
    merchant_name: str | None = Field(
        None,
        description=(
            "Seller's trading or legal name from printed receipt text. A logo alone or a "
            "payment processor name does not identify the seller."
        ),
    )
    merchant_display_name: MerchantDisplayName = None
    merchant_address: str | None = Field(
        None,
        description=(
            "Seller's printed address, with the street, postal code and locality that are "
            "visible."
        ),
    )
    receipt_number: str | None = Field(
        None,
        description=(
            "Merchant's receipt or invoice number as printed, including leading zeros. "
            "Exclude terminal IDs, card authorizations and payment references."
        ),
    )
    date: Date | None = Field(
        None,
        description=(
            "Purchase date in YYYY-MM-DD. Resolve local date order from evidence. Leave "
            "ambiguous dates null."
        ),
    )
    time: ReceiptTime | None = Field(
        None,
        description=(
            "Local purchase time in HH:MM or HH:MM:SS, preserving printed fractional seconds "
            "up to six places."
        ),
    )
    currency: Currency | None = Field(
        None,
        description=(
            "Transaction currency as a three-letter uppercase ISO code supported by the "
            "printed code or an unambiguous currency symbol."
        ),
    )
    category: Category | None = None
    line_items: list[DraftLineItem] | None = Field(
        None,
        description=(
            "Ordered logical purchase, return, deposit and credit entries. Combine a name "
            "and its detail/price lines into one entry. Exclude section headings, totals, "
            "tax summaries and payment rows."
        ),
    )
    taxes: list[DraftTaxLine] | None = Field(
        None,
        description=(
            "Tax-summary rows in printed order. Preserve separately stated rates. Leave null "
            "when tax information is missing."
        ),
    )
    subtotal: Decimal | None = Field(
        None,
        description=(
            "Receipt-wide subtotal interpreted with subtotal_origin. Prefer a printed subtotal. "
            "Derive a net subtotal only from complete line and tax evidence."
        ),
    )
    discount_total: Decimal | None = Field(
        None, ge=0,
        description=(
            "Total discount magnitude when evidenced. Use discount_treatment to identify any "
            "amount already included in the line prices."
        ),
    )
    tax_total: Decimal | None = Field(
        None, ge=0,
        description=(
            "Total tax charged across tax bands. Use a printed tax total or the sum of a "
            "complete set of printed tax amounts, counting each once. Missing tax stays "
            "null."
        ),
    )
    tip: Decimal | None = Field(
        None, ge=0,
        description=(
            "Gratuity actually paid as evidenced on the receipt. Suggested tip amounts are "
            "not payments."
        ),
    )
    total: Decimal | None = Field(
        None,
        description=(
            "Final transaction amount charged or refunded. Exclude cash tendered, change, "
            "card balance and payment subtotals."
        ),
    )
    payment_method: PaymentMethod | None = Field(
        None,
        description=(
            "Payment type supported by the tender entry. Use unknown for an unidentified "
            "type, e.g. card with no debit/credit distinction. Leave missing evidence null."
        ),
    )


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
