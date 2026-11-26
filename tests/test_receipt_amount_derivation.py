import json
from decimal import Decimal

import pytest

from vision_bill.model.extraction import ExtractionDraft, InferenceEnvelope, ReceiptInput
from vision_bill.model.review import AmountPolicy
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.service.receipt_amounts import derive_missing_amounts


def document(**extra):
    return ExtractionDraft.model_validate(
        {
            "merchant_name": "ALDI SÜD",
            "merchant_address": "Beispielstraße 1, 50667 Köln",
            "date": "2025-04-10",
            "currency": "EUR",
            "line_items": [
                {"description": "Brot", "total_price": "10.70", "tax_code": "A"},
                {"description": "Artikel", "total_price": "11.90", "tax_code": "B"},
            ],
            **extra,
        }
    )


def derive(draft, **flags):
    policy = AmountPolicy.model_validate({"tax_basis": "gross", **flags})
    findings = derive_missing_amounts(draft, policy, list(range(len(draft.line_items))), {})
    return policy, {f.field_path: f for f in findings}


def test_missing_aldi_vat_is_included_in_gross_and_records_its_evidence():
    draft = document()
    policy, findings = derive(draft)
    assert [tax.amount for tax in draft.taxes] == [Decimal("0.70"), Decimal("1.90")]
    assert (
        draft.subtotal == 20
        and draft.tax_total == Decimal("2.60")
        and draft.total == Decimal("22.60")
    )
    assert policy.subtotal_origin == "derived_net"
    assert findings["taxes.0.amount"].comparison["inputs"].split(", ") == [
        "line_items.0.total_price",
        "line_items.0.tax_code",
        "merchant_name",
        "merchant_address",
        "date",
        "currency",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        {"date": "2020-09-10"},
        {"merchant_name": "Another retailer"},
        {"merchant_address": "Example road, Italy"},
        {"currency": "CHF"},
    ],
)
def test_bare_tax_letters_outside_the_evidenced_convention_remain_unknown(extra):
    draft = document(**extra)
    policy, _ = derive(draft)
    assert draft.total == Decimal("22.60")
    assert draft.subtotal == Decimal("22.60") and policy.subtotal_origin == "derived_gross"
    assert draft.tax_total is draft.taxes is None


def test_printed_rates_and_amounts_win_and_tax_rounds_once_per_band():
    draft = document(
        merchant_name="Store",
        line_items=[{"description": "Item", "total_price": "0.04", "tax_code": "A"}] * 3,
        taxes=[{"name": "A", "rate": "0.19"}],
    )
    derive(draft)
    assert draft.taxes[0].amount == Decimal("0.02")
    original = document(
        taxes=[
            {"name": "A", "rate": 0, "amount": 0},
            {"name": "B", "rate": "0.19", "amount": "1.89"},
        ]
    )
    derive(original)
    assert [t.amount for t in original.taxes] == [0, Decimal("1.89")]


def test_separate_discount_with_no_band_allocation_recovers_gross_amounts():
    draft = document(discount_total="2.64")
    policy, _ = derive(draft, discount_treatment="subtract")
    assert draft.total == Decimal("19.96")
    assert draft.subtotal == Decimal("22.60") and policy.subtotal_origin == "derived_gross"
    assert draft.tax_total is None
    assert [(t.rate, t.amount) for t in draft.taxes] == [
        (Decimal("0.07"), None),
        (Decimal("0.19"), None),
    ]


def test_unreadable_position_or_contradictory_posted_total_does_not_generate_amounts():
    draft = document(total="52.73")
    _, findings = derive(draft)
    assert not findings and draft.total == Decimal("52.73") and draft.tax_total is None
    draft = document()
    draft.line_items[0].total_price = None
    assert not derive(draft)[1] and draft.total is None


def parse(payload, count=1):
    provider = OllamaProvider("http://local")
    inputs = ReceiptInput(mode="direct_vision", image_paths=["unused"] * count)
    prototype = InferenceEnvelope.model_validate(
        {
            "provenance": {
                "requested_pipeline": "direct_vision",
                "used_pipeline": "direct_vision",
                "source_image_count": count,
                "ordered_source_sha256": [],
                "model_id": "test",
            },
            "telemetry": {"elapsed_ms": 0, "max_output_tokens": 1024},
        }
    )
    return provider._parse_receipt_input(json.dumps(payload), input=inputs, prototype=prototype)


def test_provider_derives_after_verified_overlap_and_keeps_unresolved_duplicate_tax_bands():
    payload = document().model_dump(mode="json")
    first, second = payload["line_items"]
    payload.update(
        confidence=90,
        payment_method="cash",
        line_items_complete=True,
        amount_policy={"tax_basis": "gross"},
        duplicate_candidates=[{"first_index": 0, "second_index": 1}],
    )
    payload["line_items"] = [
        dict(first, quantity=1, source_image=0),
        dict(first, quantity=1, source_image=1),
        dict(second, source_image=1),
    ]
    result = parse(payload, 2)
    assert result.receipt.total == Decimal("22.60") and len(result.receipt.line_items) == 2
    assert len(result.draft.line_items) == 3 and result.overlap_decisions[0]["verified"]
    assert any("line_items.2.total_price" in f.comparison["inputs"] for f in result.findings)
    payload["line_items"][1]["tax_code"] = "B"
    payload.update(subtotal="28.70", tax_total="4.60", taxes=[], total="33.30")
    result = parse(payload, 2)
    assert len(result.receipt.line_items) == 3 and not result.overlap_decisions[0]["verified"]
    assert not result.findings


def test_provider_leaves_unconfirmed_partial_crop_for_existing_repair_and_review():
    from vision_bill.provider.llm.base import _PartialDraftError

    payload = document().model_dump(mode="json")
    payload.update(confidence=95, payment_method="cash", amount_policy={"tax_basis": "gross"})
    with pytest.raises(_PartialDraftError) as partial:
        parse(payload)
    assert (
        partial.value.envelope.draft.tax_total is None
        and partial.value.envelope.draft.total is None
    )


@pytest.mark.parametrize(
    "overlap", [{}, {"duplicate_candidates": [], "overlap_decisions": [{"verified": None}]}]
)
def test_multi_photo_without_established_overlap_does_not_invent_totals(overlap):
    from vision_bill.provider.llm.base import _PartialDraftError

    payload = document().model_dump(mode="json")
    for i, item in enumerate(payload["line_items"]):
        item["source_image"] = i
    payload.update(
        confidence=90,
        payment_method="cash",
        line_items_complete=True,
        amount_policy={"tax_basis": "gross"},
        **overlap,
    )
    with pytest.raises(_PartialDraftError) as partial:
        parse(payload, 2)
    assert partial.value.envelope.draft.total is None


def test_quantization_uses_the_currency_and_signed_positions():
    draft = document(
        merchant_name="Store",
        currency="KWD",
        line_items=[
            {"description": "Purchase", "tax_rate": "0.05", "total_price": "1.010"},
            {"description": "Return", "tax_rate": "0.05", "total_price": "-0.200"},
        ],
    )
    derive(draft, tax_basis="net")
    assert draft.tax_total == Decimal("0.041") and draft.subtotal == Decimal("0.810")
    assert draft.total == Decimal("0.851")


def test_explicit_nonzero_position_rate_precludes_fabricating_zero_from_conflicting_policy():
    draft = document(
        line_items=[{"description": "Item", "total_price": "11.90", "tax_rate": "0.19"}]
    )
    policy, _ = derive(draft, tax_information="explicit_zero")
    assert draft.tax_total is None and draft.taxes is None
    assert draft.subtotal == Decimal("11.90") and policy.subtotal_origin == "derived_gross"


def test_single_band_discount_and_complete_printed_tax_table():
    draft = document(
        line_items=[{"description": "Item", "total_price": "23.80", "tax_code": "B"}],
        discount_total="11.90",
    )
    derive(draft, discount_treatment="subtract")
    assert draft.total == Decimal("11.90") and draft.tax_total == Decimal("1.90")

    payload = document(
        line_items=[{"description": "Item", "total_price": "5.35"}],
        taxes=[{"name": "7%", "rate": "0.07", "amount": "0.35"}],
    ).model_dump(mode="json")
    payload.update(
        confidence=90,
        payment_method="cash",
        line_items_complete=True,
        tax_summary_complete=True,
        amount_policy={"tax_basis": "gross", "tax_information": "present"},
    )
    receipt = parse(payload).receipt
    assert (
        receipt.total == Decimal("5.35")
        and receipt.subtotal == 5
        and receipt.tax_total == Decimal("0.35")
    )
