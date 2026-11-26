from datetime import date as Date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel


class ProductPurchase(BaseModel):
    """One verified line-item purchase matching a product search."""

    receipt_id: UUID
    description: str
    merchant_name: str
    merchant_display_name: str | None = None
    date: Date
    time: str | None = None
    quantity: Decimal | None
    unit_price: Decimal | None
    currency: str


class ProductSearchResponse(BaseModel):
    """Purchase history and unit-price summary for one product query."""

    query: str
    purchases: list[ProductPurchase]
    latest_price: Decimal | None = None
    cheapest_price: Decimal | None = None
    average_price: Decimal | None = None
    currency: str | None = None
