from datetime import date as Date
from datetime import time as Time
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator

from .item_category import SpendingCategory, llm_category

Category = Literal[
    "grocery",
    "electronics",
    "clothing",
    "restaurant",
    "fuel",
    "pharmacy",
    "entertainment",
    "other",
]


MAX_TAG_LENGTH = 100
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


def _canonical_local_time(value: str) -> str:
    # Match stored review material before hashing; parsing follows strict validation
    # so unsupported ISO forms and precision beyond PostgreSQL TIME cannot be lost.
    parsed = Time.fromisoformat(value)
    return (
        parsed.isoformat(timespec="minutes")
        if not parsed.second and not parsed.microsecond
        else parsed.isoformat()
    )


ReceiptTime = Annotated[
    str,
    Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](?:\.[0-9]{1,6})?)?$"),
    AfterValidator(_canonical_local_time),
]


class LineItem(BaseModel):
    description: str = Field(description="Name of the purchased item")
    spending_category: SpendingCategory = Field(
        default="unknown", description="One spending category code"
    )
    quantity: Decimal | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Quantity purchased if printed"
    )
    unit_price: Decimal | None = Field(default=None, description="Price per unit if printed")
    total_price: Decimal = Field(description="Total price of the purchased line")
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Short tags for the item. Prefer the tag list from the prompt; a single new "
            "suggested tag is allowed when nothing fits."
        ),
    )

    @field_validator("spending_category", mode="before")
    @classmethod
    def _validate_llm_category(cls, value: object) -> str:
        return llm_category(value)

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, v: list[str]) -> list[str]:
        """Clean LLM-produced tags: trim, dedupe case-insensitively, cap length.

        Suggestions (tags not yet in the vocabulary) are kept — the UI marks
        them for human review — but obviously broken values are dropped.
        """
        seen: set[str] = set()
        clean: list[str] = []
        for tag in v:
            name = " ".join(tag.split()).strip()
            if not name or len(name) > MAX_TAG_LENGTH:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            clean.append(name)
        return clean


class TaxLine(BaseModel):
    name: str = Field(description="e.g. 'VAT', 'Sales Tax', 'GST'")
    rate: Decimal | None = Field(default=None, description="Tax rate as a decimal, e.g. 0.19 for 19%")
    amount: Decimal = Field(ge=0, description="Tax amount charged")


class Receipt(BaseModel):
    confidence: int = Field(
        ge=0,
        le=100,
        description="from 0 - 100 as percentage, how sure are you in the presented data",
    )
    merchant_name: str = Field(
        description=(
            "Name of the store or vendor, taken from printed text on the receipt "
            "(header, footer, address, website) — never guessed from a logo"
        )
    )
    merchant_address: str | None = Field(
        default=None, description="Store address if printed on receipt"
    )
    receipt_number: str | None = Field(default=None, description="Transaction/receipt ID")
    date: Date = Field(description="Date of purchase, ISO format YYYY-MM-DD")
    time: ReceiptTime | None = Field(
        default=None,
        description="Local time of purchase: HH:MM or HH:MM:SS, optionally with 1–6 fractional second digits",
    )

    currency: Currency = Field(default="EUR", description="ISO 4217 currency code, e.g. USD, EUR")
    category: Category = Field(
        default="other", description="Best-guess category for the whole purchase"
    )
    line_items: list[LineItem] = Field(description="All purchased items")
    taxes: list[TaxLine] = Field(default_factory=list, description="Tax lines applied")

    subtotal: Decimal = Field(description="Sum of line items before tax/discounts")
    discount_total: Decimal | None = Field(
        default=Decimal(0), ge=0, description="Total discounts applied if printed"
    )
    tax_total: Decimal = Field(default=Decimal(0), ge=0, description="Sum of all taxes")
    tip: Decimal | None = Field(default=None, ge=0, description="Tip/gratuity if applicable")
    total: Decimal = Field(description="Final amount charged")

    payment_method: Literal[
        "cash", "credit_card", "debit_card", "mobile_payment", "check", "other", "unknown"
    ] = Field(default="unknown")
