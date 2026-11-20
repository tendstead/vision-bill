from datetime import date
from decimal import Decimal

from vision_bill.model.receipt import Receipt
from vision_bill.service.experiment_scoring import detailed_score


def receipt(names=("Milk", "Bread")):
    return Receipt(confidence=90, merchant_name="Store", date=date(2026, 1, 1),
                   line_items=[dict(description=n, quantity=1, unit_price="2.00",
                                    total_price="2.00") for n in names],
                   subtotal="4.00", total="4.00")


def test_one_cent_difference_is_not_exact():
    expected = receipt()
    actual = expected.model_copy(update={"total": Decimal("4.01")})
    result = detailed_score(expected, actual)
    assert result["scores"]["strict"] < 1
    assert result["scores"]["normalized"] < 1
    assert any(d["path"] == "total" for d in result["differences"])


def test_quantities_and_unit_prices_are_scored():
    expected = receipt()
    actual = expected.model_copy(deep=True)
    actual.line_items[0].quantity = Decimal(2)
    actual.line_items[0].unit_price = Decimal("1.00")
    result = detailed_score(expected, actual)
    paths = {d["path"] for d in result["differences"]}
    assert {"line_items[0].quantity", "line_items[0].unit_price"} <= paths


def test_missing_first_row_does_not_shift_every_following_row():
    result = detailed_score(receipt(), receipt(("Bread",)))
    assert result["rows"]["missing"] == 1
    assert result["rows"]["extra"] == 0
    assert result["scores"]["line_items"] == 0.5
    assert result["rows"]["matched"] == 1


def test_duplicate_purchases_are_not_collapsed():
    result = detailed_score(receipt(("Milk", "Milk")), receipt(("Milk",)))
    assert result["rows"]["missing"] == 1
    assert result["scores"]["line_items"] == 0.5


def test_classification_is_separate_from_extraction():
    expected = receipt()
    actual = expected.model_copy(deep=True)
    actual.category = "electronics"
    actual.line_items[0].spending_category = "cheese"
    result = detailed_score(expected, actual)
    assert result["scores"]["strict"] == 1
    assert result["scores"]["ordered_items"] == 1
    assert result["scores"]["classification"] < 1


def test_normalization_preserves_wrong_product_names():
    expected = receipt(("M.Mio Mate",))
    actual = receipt(("Pilsner",))
    result = detailed_score(expected, actual)
    assert result["scores"]["line_items"] < 1
    assert result["differences"][0]["equivalent"] is False


def test_whitespace_is_reported_as_representation_difference():
    expected = receipt(("M.Mio Mate",))
    actual = receipt(("M.Mio  Mate",))
    result = detailed_score(expected, actual)
    assert result["scores"]["normalized"] == 1
    assert result["scores"]["strict"] < 1


def test_absent_optional_values_match():
    result = detailed_score(receipt(), receipt())
    assert result["scores"]["strict"] == 1
    assert result["differences"] == []


def test_exact_scoring_distinguishes_unprinted_values_from_numbers():
    expected = receipt()
    expected.discount_total = None
    expected.line_items[0].quantity = None
    expected.line_items[0].unit_price = None

    perfect = detailed_score(expected, expected)
    assert perfect["item_tp"] == 2
    assert perfect["exact_amounts"] and perfect["full_receipt_correct"]
    assert perfect["checks"]["schema_net_plus_tax_matches_total"]

    inferred = detailed_score(expected, receipt())
    assert inferred["item_tp"] == 1
    assert not inferred["exact_amounts"] and not inferred["full_receipt_correct"]
    assert {d["path"] for d in inferred["differences"]} == {
        "discount_total", "line_items[0].quantity", "line_items[0].unit_price",
    }


def test_zero_seconds_time_representation_is_equivalent():
    expected = receipt().model_copy(update={"time": "12:30:00"})
    actual = receipt().model_copy(update={"time": "12:30"})
    result = detailed_score(expected, actual)
    assert result["scores"]["normalized"] == 1
    assert result["scores"]["strict"] < 1
    assert result["differences"][0]["equivalent"] is True


def test_nonzero_time_seconds_are_not_ignored():
    expected = receipt().model_copy(update={"time": "12:30:45"})
    actual = receipt().model_copy(update={"time": "12:30"})
    assert detailed_score(expected, actual)["scores"]["normalized"] < 1


def test_exact_item_metrics_use_amounts_and_count_every_occurrence():
    expected = receipt()
    actual = expected.model_copy(deep=True)
    actual.line_items[1].total_price = Decimal('2.01')
    score = detailed_score(expected, actual)
    assert score['version'] == '4'
    assert (score['item_tp'], score['item_fp'], score['item_fn']) == (1, 1, 1)
    assert score['precision'] == score['recall'] == .5
    assert score['full_receipt_correct'] is False
    missing = detailed_score(expected, receipt(('Milk',)))
    assert (missing['item_tp'], missing['item_fp'], missing['item_fn']) == (1, 0, 1)
    extra = detailed_score(receipt(('Milk',)), receipt(('Milk', 'Milk')))
    assert (extra['item_tp'], extra['item_fp'], extra['item_fn']) == (1, 1, 0)


def test_exact_lcs_uses_nfkc_casefold_and_earlier_index_ties():
    score = detailed_score(receipt(('ＭＩＬＫ', 'Bread', 'Milk')),
                           receipt((' milk ', 'Milk', 'BREAD')))
    assert score['item_matches'] == [[0, 0], [1, 2]]
    assert (score['item_tp'], score['item_fp'], score['item_fn']) == (2, 1, 1)
    assert score['exact_item_order'] is False
    assert score['full_receipt_correct'] is False
    assert detailed_score(receipt(('ＭＩＬＫ',)), receipt(('milk',)))['full_receipt_correct']


def test_frozen_observable_mask_excludes_nonobservable_defaults_and_labels():
    expected = receipt()
    actual = expected.model_copy(deep=True)
    actual.payment_method = 'cash'
    actual.category = 'electronics'
    actual.confidence = 3
    actual.line_items[0].tags = ['anything']
    assert detailed_score(expected, actual, scored_fields=['merchant_name', 'line_items', 'total'])['full_receipt_correct']
    assert not detailed_score(expected, actual)['full_receipt_correct']


def test_exact_total_tax_rate_and_failed_receipt():
    from vision_bill.model.receipt import TaxLine
    expected = receipt()
    expected.taxes = [TaxLine(name='VAT', rate=19, amount='1.00')]
    actual = expected.model_copy(deep=True)
    actual.taxes[0].rate = Decimal(7)
    assert not detailed_score(expected, actual)['full_receipt_correct']
    perfect = detailed_score(expected, expected)
    assert perfect['full_receipt_correct']
    assert perfect['exact_amounts'] and perfect['exact_item_count'] and perfect['exact_item_order']
    failed = detailed_score(expected, None)
    assert not failed['full_receipt_correct']
    assert (failed['item_tp'], failed['item_fp'], failed['item_fn']) == (0, 0, 2)
    assert failed['precision'] == failed['recall'] == 0
    assert failed['scores']['normalized'] == 0
    assert failed['extraction_failed']


def test_frozen_numeric_strings_are_scored_without_float_roundtrip():
    expected = receipt(('A',)).model_dump(mode='json')
    expected['line_items'][0]['quantity'] = '1.00000000000000000001'
    result = detailed_score(expected, receipt(('A',)))
    assert result['item_tp']==0 and result['item_fn']==1 and result['item_fp']==1
    assert not result['full_receipt_correct']


def test_historical_factor_scores_ignore_scale_and_keep_decimal_differences():
    from vision_bill.model.receipt import TaxLine

    expected = receipt()
    expected.taxes = [TaxLine(name="VAT", rate=Decimal("0.20"), amount=Decimal(0))]
    actual = expected.model_copy(deep=True)
    actual.line_items[0].quantity = Decimal("1.00")
    actual.taxes[0].rate = Decimal("0.2")
    assert detailed_score(expected, actual)["scores"]["strict"] == 1
    actual.line_items[0].quantity = Decimal("1.0000000000000001")
    assert detailed_score(expected, actual)["scores"]["strict"] < 1
