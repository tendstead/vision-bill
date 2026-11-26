from test_experiment_scoring import receipt

from vision_bill.model.extraction import ExtractionDraft, InferenceEnvelope
from vision_bill.service.receipt_review import (
    editable_draft,
    material_document,
    suggest_product_labels,
)


def make_envelope(draft, canonical=None, *, state="completed", missing=(), decisions=()):
    return InferenceEnvelope(
        draft=draft,
        receipt=canonical,
        classification_enrichment={"state": state, "missing_fields": list(missing)},
        overlap_decisions=list(decisions),
        telemetry={"elapsed_ms": 0, "max_output_tokens": 100},
        provenance={
            "requested_pipeline": "direct_vision", "used_pipeline": "direct_vision",
            "source_image_count": 1, "ordered_source_sha256": ["a" * 64], "model_id": "model",
        },
    )


def test_unpaid_draft_gets_product_labels_without_fabricating_payment_or_changing_evidence():
    raw = ExtractionDraft.model_validate({
        "merchant_name": "HIT 003 St. Augustin", "line_items": [
            {"description": "P&W Remoulade 250 ml"},
            {"description": "Bio Möhren"},
            {"description": "Karottenkuchen"},
            {"description": "Shorts Blau", "tags": [], "spending_category": "other"},
        ],
    })
    envelope = make_envelope(raw, state="skipped")
    document = editable_draft(envelope)
    suggest_product_labels(document)
    assert document.payment_method is None
    assert [(row.tags, row.spending_category) for row in document.line_items] == [
        (["sauces"], "sauces_spices"), (["carrots", "vegetables", "organic"], "vegetables"),
        (None, None), ([], "other"),
    ]
    assert raw.line_items[0].tags is None
    document.merchant_display_name = "HIT"
    assert material_document(document) == material_document(raw)
    assert "merchant_display_name" not in material_document(document)


def test_classifier_projects_only_the_supported_fields_onto_the_correct_raw_rows():
    canonical = receipt(("Product A", "Product B"))
    canonical.category = "grocery"
    canonical.line_items[0].spending_category = "meat"
    canonical.line_items[1].tags = ["coffee"]
    raw = ExtractionDraft.model_validate({
        "line_items": [
            {"description": "Product A"}, {"description": "Product A"}, {"description": "Product B"},
        ]
    })
    document = editable_draft(make_envelope(
        raw, canonical, state="unavailable", missing=["line_items.1.spending_category"],
        decisions=[{"verified": True, "removed_index": 1}],
    ))
    assert document.category == "grocery"
    assert [(row.tags, row.spending_category) for row in document.line_items] == [
        ([], "meat"), (None, None), (["coffee"], None),
    ]
    assert raw.category is None
    assert all(row.tags is None for row in raw.line_items)


def test_unsettled_draft_labels_recognizable_products_using_only_their_own_rows():
    names = [
        "Hackfl. gem. XXL", "Grillfackeln", "HFS S-Nackensteak", "Hähn. Brustf. XXL",
        "Vitalgebäck", "Energy Editions 250 ml", "Einwegpfand", "Damen Pyjama",
        "Kinder T-Shirt blau", "Aufkleber Set", "Bio Schweinefleisch 500 g",
        "Energy Riegel", "Pfandkuchen", "Artikel 25671", "Shorts Aufkleber",
    ]
    raw = ExtractionDraft.model_validate({"line_items": [{"description": name} for name in names]})
    document = editable_draft(make_envelope(raw, state="skipped"))
    suggest_product_labels(document)
    assert [(row.spending_category, row.tags) for row in document.line_items] == [
        ("meat", ["meat"]), ("meat", ["meat"]), ("meat", ["pork", "meat"]),
        ("meat", ["poultry", "meat"]), ("bread_bakery", ["pastries"]),
        ("soft_drinks", ["soft_drinks", "beverage"]), ("deposit", ["deposit"]),
        ("clothing", ["clothing"]), ("clothing", ["clothing"]), ("other", ["stationery"]),
        ("meat", ["pork", "meat", "organic"]), (None, None), (None, None),
        (None, None), (None, None),
    ]
    assert material_document(document) == material_document(raw)


def test_unresolved_receipt_category_and_human_label_are_preserved():
    canonical = receipt(("Möhren",))
    raw = ExtractionDraft.model_validate({"line_items": [{"description": "Möhren", "tags": []}]})
    document = editable_draft(make_envelope(
        raw, canonical, state="unavailable", missing=["category", "line_items.0.spending_category"],
    ))
    suggest_product_labels(document)
    assert document.category is None
    assert document.line_items[0].tags == []
    assert document.line_items[0].spending_category == "vegetables"


def test_display_name_is_nullable_and_normalized_on_draft_and_receipt():
    assert ExtractionDraft(merchant_display_name=" ").merchant_display_name is None
    assert ExtractionDraft(merchant_display_name=" H & M  ").merchant_display_name == "H & M"
    assert receipt().merchant_display_name is None
