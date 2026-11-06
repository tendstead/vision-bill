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
    actual.line_items[0].quantity = 2
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
