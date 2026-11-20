import json
from datetime import date as Date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, computed_field, field_validator

from ..extraction import ClassificationEnrichment
from ..ocr import SourceRef
from ..review import CHECKS_VERSION, AmountPolicy, FindingAcknowledgement, ReviewFinding


class ReceiptRow(BaseModel):
    """Represents a single row in the `receipts` table."""

    id: UUID
    confidence: int
    merchant_name: str
    merchant_address: str | None = None
    receipt_number: str | None = None
    date: Date
    time: str | None = None
    currency: str = "USD"
    category: str = "other"
    subtotal: Decimal
    discount_total: Decimal | None = Decimal(0)
    tax_total: Decimal = Decimal(0)
    tip: Decimal | None = None
    total: Decimal
    payment_method: str = "unknown"
    created_at: Date | None = None
    status: str = "unverified"
    image_id: UUID | None = None
    thumbnail_path: str | None = None
    verified: bool = False
    user_id: UUID | None = None
    extraction_id: UUID | None = None
    operation_revision: int = Field(default=1, ge=1)
    review_revision: int = Field(default=1, ge=1)
    amount_policy: AmountPolicy = Field(default_factory=AmountPolicy)
    findings: list[ReviewFinding] = Field(default_factory=list)
    verification_source: str | None = None
    checks_version: str | None = None
    field_sources: dict[str, list[SourceRef]] = Field(default_factory=dict)
    overlap_decisions: list[dict[str, object]] = Field(default_factory=list)
    acknowledgements: list[FindingAcknowledgement] = Field(default_factory=list)

    @field_validator(
        "amount_policy",
        "findings",
        "field_sources",
        "overlap_decisions",
        "acknowledgements",
        mode="before",
    )
    @classmethod
    def decode_review_json(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value

    def can_rereview(self) -> bool:
        return (
            self.verified
            and self.status == "verified"
            and (self.verification_source != "human" or self.checks_version != CHECKS_VERSION)
        )


class LineItemRow(BaseModel):
    """Represents a single row in the `line_items` table."""

    id: UUID
    receipt_id: UUID
    description: str
    original_description: str | None = None
    spending_category: str = "unknown"
    category_source: str = "unknown"
    quantity: Decimal | None
    unit_price: Decimal | None
    total_price: Decimal
    tags: list[str] = Field(default_factory=list)
    position: int = 0


class TaxLineRow(BaseModel):
    """Represents a single row in the `taxes` table."""

    id: UUID
    receipt_id: UUID
    name: str
    rate: Decimal | None = None
    amount: Decimal


class ReceiptWithDetails(BaseModel):
    """A receipt row enriched with its line items and taxes.

    ``image_path`` is resolved from the joined ``images`` row so frontends
    can keep displaying the image location without a second lookup.
    """

    receipt: ReceiptRow
    line_items: list[LineItemRow]
    taxes: list[TaxLineRow]
    image_path: str | None = None
    collection_ids: list[UUID] = Field(default_factory=list)
    classification_enrichment: ClassificationEnrichment = Field(
        default_factory=ClassificationEnrichment
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def can_rereview(self) -> bool:
        return self.receipt.can_rereview()
