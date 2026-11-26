from decimal import Decimal
from uuid import UUID

from vision_bill.model.extraction import ExtractionDraft
from vision_bill.model.ocr import SourceRef
from vision_bill.model.receipt import Receipt
from vision_bill.service.receipt_grouping import group_purchases, match_grouped_reference


def test_repeated_purchases_sum_quantities_and_keep_every_source():
    document = ExtractionDraft.model_validate(
        {
            "line_items": [
                {"description": "Energy Editions", "total_price": "0.29"},
                {"description": "Pfand", "total_price": "0.25"},
                {"description": " energy  editions ", "total_price": "0.29"},
                {
                    "description": "Pfand",
                    "quantity": 2,
                    "unit_price": "0.25",
                    "total_price": "0.50",
                },
            ]
        }
    )
    first = SourceRef(part_id=UUID(int=1), photo_index=0, source_sha256="a" * 64)
    second = SourceRef(part_id=UUID(int=2), photo_index=1, source_sha256="b" * 64)
    grouped, sources = group_purchases(
        document, {"line_items.0": [first], "line_items.2": [second]}
    )
    assert grouped.line_items is not None and document.line_items is not None
    assert [
        (row.description, row.quantity, row.unit_price, row.total_price)
        for row in grouped.line_items
    ] == [
        ("Energy Editions", 2, Decimal("0.29"), Decimal("0.58")),
        ("Pfand", 3, Decimal("0.25"), Decimal("0.75")),
    ]
    assert len(document.line_items) == 4
    assert sources["line_items.0"] == [first, second]
    assert sources["line_items.0.quantity"] == [first, second]


def test_distinct_prices_tax_bands_returns_and_unknown_quantities_stay_separate():
    document = ExtractionDraft.model_validate(
        {
            "line_items": [
                {"description": "Item", "total_price": "3.00"},
                {"description": "Item", "total_price": "-3.00"},
                {"description": "Item", "total_price": "4.00"},
                {"description": "Item", "total_price": "3.00", "quantity": 2},
                {"description": "Item", "total_price": "3.00"},
                {"description": "Item", "total_price": None},
            ]
        }
    )
    grouped, _ = group_purchases(document, {}, tax_keys=["A", "A", "A", "A", "B", "B"])
    assert grouped is document


def test_grouped_benchmark_reference_accepts_exact_purchases_and_keeps_label_positions():
    header = {
        "confidence": 90,
        "merchant_name": "Store",
        "date": "2026-01-01",
        "currency": "EUR",
        "subtotal": "0.50",
        "total": "0.50",
    }
    actual = Receipt.model_validate(
        header
        | {
            "line_items": [
                {"description": "Pfand", "total_price": "0.25"},
                {"description": "Pfand", "total_price": "0.25"},
            ]
        }
    )
    reference = Receipt.model_validate(
        header
        | {
            "line_items": [
                {
                    "description": "Pfand",
                    "total_price": "0.50",
                    "quantity": 2,
                    "unit_price": "0.25",
                },
            ]
        }
    )
    metadata = {"classification_enrichment": {"missing_fields": ["line_items.1.tags"]}}
    compact = match_grouped_reference(reference, actual, metadata)
    assert len(compact.line_items) == 1 and compact.line_items[0].quantity == 2
    assert metadata["classification_enrichment"]["missing_fields"] == ["line_items.0.tags"]
    reference.line_items[0].description = "Other"
    assert match_grouped_reference(reference, actual, {}) is actual
    assert match_grouped_reference(actual, actual, {}) is actual
