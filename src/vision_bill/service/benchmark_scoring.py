"""Pure benchmark scoring helpers; no extraction data is persisted here."""

import json
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from typing import Any

from ..model.receipt import Receipt

MONEY_TOLERANCE = Decimal("0.01")
# Bumped to 2: scoring output includes a separate spending-category component.
SCORING_VERSION = "2"
# Version 7 captures per-position VAT and completeness evidence for amount derivation.
PROMPT_VERSION = "7"
COMPONENT_WEIGHTS = {"header": 0.20, "totals": 0.35, "line_items": 0.30, "taxes": 0.15}


def normalize_money(value: Decimal | float | str | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value)).quantize(MONEY_TOLERANCE, rounding=ROUND_HALF_UP)


def money_equal(expected: Any, actual: Any) -> bool:
    left, right = normalize_money(expected), normalize_money(actual)
    return left is not None and right is not None and abs(left - right) <= MONEY_TOLERANCE


def _fraction(matches: int, total: int) -> float:
    return 1.0 if total == 0 else matches / total


def score_receipts(expected: Receipt, actual: Receipt) -> dict[str, float]:
    header_fields = (
        "merchant_name",
        "merchant_address",
        "receipt_number",
        "date",
        "time",
        "currency",
        "category",
        "payment_method",
    )
    header = _fraction(
        sum(getattr(expected, f) == getattr(actual, f) for f in header_fields), len(header_fields)
    )
    total_fields = ("subtotal", "discount_total", "tax_total", "tip", "total")
    totals = _fraction(
        sum(money_equal(getattr(expected, f), getattr(actual, f)) for f in total_fields),
        len(total_fields),
    )
    expected_items, actual_items = expected.line_items, actual.line_items
    item_matches = sum(
        e.description == a.description and money_equal(e.total_price, a.total_price)
        for e, a in zip(expected_items, actual_items, strict=False)
    )
    line_items = _fraction(item_matches, max(len(expected_items), len(actual_items)))
    labelled_items = [
        (index, item)
        for index, item in enumerate(expected_items)
        if item.spending_category != "unknown"
    ]
    if labelled_items:
        category_matches = sum(
            index < len(actual_items)
            and item.description == actual_items[index].description
            and item.spending_category == actual_items[index].spending_category
            for index, item in labelled_items
        )
        spending_categories = _fraction(category_matches, len(labelled_items))
    expected_taxes, actual_taxes = expected.taxes, actual.taxes
    tax_matches = sum(
        e.name == a.name and money_equal(e.amount, a.amount)
        for e, a in zip(expected_taxes, actual_taxes, strict=False)
    )
    taxes = _fraction(tax_matches, max(len(expected_taxes), len(actual_taxes)))
    components = {"header": header, "totals": totals, "line_items": line_items, "taxes": taxes}
    if labelled_items:
        components["spending_categories"] = spending_categories
    components["overall"] = sum(
        components[name] * weight for name, weight in COMPONENT_WEIGHTS.items()
    )
    return components


def validation_schema_sha256() -> str:
    """Hash the receipt fields a benchmark model can produce and validate."""
    schema = Receipt.model_json_schema()
    schema["properties"].pop("merchant_display_name")
    return sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def dataset_fingerprint(receipts: list[Receipt]) -> str:
    """Stable fingerprint of the expected, selected data set."""
    rows = [r.model_dump_json(exclude={"confidence", "merchant_display_name"}) for r in receipts]
    return sha256("\n".join(sorted(rows)).encode()).hexdigest()
