from decimal import Decimal

import pytest
from test_experiment_scoring import receipt

from vision_bill.service import experiment_scoring


def evaluate(gold, actual):
    return experiment_scoring.masked_reference_score(gold, actual, list(gold), classification=False)


def test_only_confirmed_values_contribute_to_case_score():
    actual = receipt()
    gold = {"total": "4.000"}
    scored = evaluate(gold, actual)
    assert scored["scores"]["normalized"] == 1
    assert scored["full_receipt_correct"] and scored["exact_amounts"]
    assert scored["item_tp"] is None and scored["exact_item_count"] is None
    wrong = evaluate({"total": "4.01"}, actual)
    assert wrong["scores"]["normalized"] == 0
    assert [d["path"] for d in wrong["differences"]] == ["total"]


def test_masked_rows_keep_order_occurrences_and_unknown_cells():
    actual = receipt(("Cap", "Cap", "Bag"))
    gold = {"line_items": [{"description": "Cap", "quantity": None, "total_price": "2"}] * 2}
    scored = evaluate(gold, actual)
    assert scored["item_matches"] == [[0, 0], [1, 1]]
    assert (scored["item_tp"], scored["item_fp"], scored["item_fn"]) == (2, 1, 0)
    assert scored["scores"]["normalized"] == pytest.approx(.8)
    assert not scored["exact_item_count"] and not scored["full_receipt_correct"]
    assert scored["differences"][0]["path"] == "line_items[extra:2]"
    reversed_order = evaluate({"line_items": [{"description": "Bag"}, {"description": "Cap"}]}, actual)
    assert reversed_order["item_tp"] == 1 and not reversed_order["exact_item_order"]


def test_failed_and_explicit_zero_remain_distinct_from_unassessed():
    failed = evaluate({"total": "0.00"}, None)
    assert failed["extraction_failed"] and failed["scores"]["normalized"] == 0
    assert failed["item_fp"] is None and not failed["full_receipt_correct"]
    empty_taxes = evaluate({"taxes": []}, receipt())
    assert empty_taxes["scores"]["normalized"] == 1 and empty_taxes["item_matches"] == []
    actual = receipt().model_copy(update={"total": Decimal("0")})
    assert evaluate({"total": "0"}, actual)["full_receipt_correct"]


def test_missing_labels_never_score_as_observed_defaults():
    from uuid import uuid4

    from vision_bill.model.extraction import ExtractionDraft
    from vision_bill.service.benchmark_service import BenchmarkService

    case_id = uuid4()
    expected = {"category": "other", "line_items": [{"description": "Cap", "tags": []}]}
    manifest = {"scoring_version": "5", "options": {"include_classification": True},
                "inputs": [{"receipt_id": str(case_id), "expected": expected, "scored_fields": ["line_items"]}]}
    actual = receipt(("Bag", "Cap"))
    for missing, classification in [
        (["category", "line_items.1.tags"], 0), (["category"], .5), ([], 1),
    ]:
        metadata = {"classification_enrichment": {"missing_fields": missing}}
        detail = BenchmarkService._score(manifest, case_id, ExtractionDraft.model_validate(expected), actual, metadata)
        assert detail["scores"]["classification"] == classification
        assert detail["scores"]["normalized"] == pytest.approx(2 / 3)
