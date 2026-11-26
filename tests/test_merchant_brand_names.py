"""Consumer-facing German trading names and multi-brand operators."""

import pytest

from vision_bill.model.merchant_catalog import merchant_category, suggest_merchant


@pytest.mark.parametrize(
    ("printed", "display", "category"),
    [
        ("k kiosk", "k kiosk", "grocery"),
        ("Kiosk", "Kiosk", "grocery"),
        ("star Tankstelle", "star Tankstelle", "fuel"),
        ("ORLEN", "ORLEN", "fuel"),
        ("OIL! Tankstellen GmbH", "OIL! Tankstelle", "fuel"),
        ("BayWa Tankstelle", "BayWa Tankstelle", "fuel"),
        ("Q1 Energie AG", "Q1", "fuel"),
        ("ARLT Computer", "ARLT Computer", "electronics"),
        ("notebooksbilliger.de", "notebooksbilliger.de", "electronics"),
    ],
)
def test_printed_consumer_brand_and_category(printed, display, category):
    assert suggest_merchant(printed) == display
    assert merchant_category(display) == category


@pytest.mark.parametrize("printed", ["Valora Holding Germany GmbH", "Star Friseur", "Oil Studio", "BayWa Baustoffe"])
def test_operator_and_unrelated_businesses_do_not_identify_a_retail_brand(printed):
    assert suggest_merchant(printed) is None
