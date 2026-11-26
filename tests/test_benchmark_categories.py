from copy import deepcopy
from datetime import date
from decimal import Decimal

from vision_bill.model.extraction import EXTRACTION_SCHEMA_VERSION, OBSERVATION_PROMPT_VERSION
from vision_bill.model.receipt import Receipt
from vision_bill.service.benchmark_scoring import (
    PROMPT_VERSION,
    SCORING_VERSION,
    dataset_fingerprint,
    score_receipts,
    validation_schema_sha256,
)


def _receipt(items: list[tuple[str, str]]) -> Receipt:
    return Receipt(
        confidence=90,
        merchant_name="Test Store",
        date=date(2025, 1, 1),
        line_items=[
            {
                "description": description,
                "spending_category": category,
                "quantity": 1,
                "unit_price": Decimal("2.00"),
                "total_price": Decimal("2.00"),
            }
            for description, category in items
        ],
        subtotal=Decimal("2.00"),
        total=Decimal("2.00"),
    )


def test_wrong_category_is_scored_separately_from_correct_item_and_price() -> None:
    expected = _receipt([("Milk", "milk")])
    actual = _receipt([("Milk", "cheese")])

    scores = score_receipts(expected, actual)

    assert scores["line_items"] == 1.0
    assert scores["spending_categories"] == 0.0
    assert scores["overall"] == score_receipts(expected, expected)["overall"]


def test_unknown_expected_categories_are_excluded() -> None:
    expected = _receipt([("Milk", "milk"), ("Mystery item", "unknown")])
    actual = _receipt([("Milk", "milk"), ("Mystery item", "cheese")])

    assert score_receipts(expected, actual)["spending_categories"] == 1.0


def test_missing_actual_item_counts_as_category_mismatch() -> None:
    expected = _receipt([("Milk", "milk"), ("Bread", "bread_bakery")])
    actual = _receipt([("Milk", "milk")])

    assert score_receipts(expected, actual)["spending_categories"] == 0.5


def test_category_on_wrong_description_does_not_match() -> None:
    expected = _receipt([("Milk", "milk")])
    actual = _receipt([("Bread", "milk")])

    assert score_receipts(expected, actual)["spending_categories"] == 0.0


def test_category_component_is_omitted_without_expected_labels() -> None:
    scores = score_receipts(_receipt([("Mystery item", "unknown")]), _receipt([]))

    assert "spending_categories" not in scores


def test_logical_item_and_tax_contract_versions() -> None:
    assert PROMPT_VERSION == "7"
    assert OBSERVATION_PROMPT_VERSION == "6"
    assert EXTRACTION_SCHEMA_VERSION == "3"
    assert SCORING_VERSION == "2"


def test_display_name_does_not_change_historical_dataset_fingerprint() -> None:
    printed = _receipt([("Milk", "milk")])
    # Digest of the version-5 fixture's original JSON before display names were introduced.
    historical = "e22530828eeaff01baee70ef8b7b83336e674c5eda0c346229aa0ab6c2fdc997"
    assert dataset_fingerprint([printed]) == historical
    assert dataset_fingerprint([printed.model_copy(update={"merchant_display_name": "HIT"})]) == historical
    assert dataset_fingerprint([printed.model_copy(update={"merchant_name": "HIT"})]) != historical


def test_benchmark_schema_hash_tracks_printed_merchant_contract_only(monkeypatch) -> None:
    original = validation_schema_sha256()
    schema = Receipt.model_json_schema()
    monkeypatch.setattr(Receipt, "model_json_schema", lambda: deepcopy(schema))
    schema["properties"]["merchant_display_name"]["description"] = "Revised display-only guidance"
    assert validation_schema_sha256() == original
    schema["properties"]["merchant_name"]["description"] = "Changed printed receipt requirement"
    assert validation_schema_sha256() != original
