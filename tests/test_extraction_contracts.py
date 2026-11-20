"""Boundary validation and safe wire projections for the extraction workflow."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vision_bill.model.db.image import ImageRow
from vision_bill.model.extraction import DraftLineItem, DraftTaxLine, ExtractionDraft, ReceiptInput
from vision_bill.model.ocr import OcrResponse, WorkingImage
from vision_bill.model.review import AmountPolicy, ReviewFinding
from vision_bill.model.workflow import AnalysisClaim, DraftPatch, RetryRequest, SubmissionStatus


def test_nullable_draft_retains_constraints():
    assert ExtractionDraft(confidence=None).confidence is None
    for change in (
        {"confidence": 101},
        {"confidence": -1},
        {"discount_total": -1},
        {"tax_total": -1},
        {"tip": -1},
        {"date": "2025-02-30"},
        {"time": "24:00"},
        {"time": "9:00"},
        {"currency": "EURO"},
    ):
        with pytest.raises(ValidationError):
            ExtractionDraft(**change)


def test_draft_does_not_invent_final_receipt_defaults():
    draft = ExtractionDraft()
    assert draft.currency is None
    assert draft.discount_total is None
    assert draft.taxes is None
    assert draft.line_items is None
    zero = ExtractionDraft(tax_total="0", taxes=[])
    assert zero.tax_total == Decimal(0)
    assert zero.taxes == []
    assert AmountPolicy().tax_information == "missing"
    assert AmountPolicy(tax_information="explicit_zero").tax_information == "explicit_zero"
    assert AmountPolicy(tax_information="present").tax_information == "present"


def test_nested_draft_constraints_and_negative_refunds():
    assert DraftLineItem().quantity is None
    with pytest.raises(ValidationError):
        DraftLineItem(quantity=0)
    with pytest.raises(ValidationError):
        DraftTaxLine(amount=-1)
    item = DraftLineItem(unit_price="-0.25", total_price="-0.75", tags=[" A ", "a"])
    assert item.model_dump(mode="json")["unit_price"] == "-0.25"
    assert item.tags == ["A"]
    assert DraftLineItem(tags=None).tags is None


def test_comparison_money_stays_decimal_text():
    finding = ReviewFinding(
        id="total",
        code="total_mismatch",
        severity="review",
        field_path="total",
        comparison={"expected": Decimal("1.20"), "actual": None},
    )
    assert finding.model_dump(mode="json")["comparison"] == {"expected": "1.20", "actual": None}
    with pytest.raises(ValidationError):
        ReviewFinding(
            id="x", code="x", severity="info", field_path="total", comparison={"actual": 0.1}
        )


def test_safe_status_omits_claim_fields_and_starts_at_generation_one():
    image = ImageRow(id=uuid4(), claim_token=uuid4(), lease_expires_at=datetime.now(UTC))
    claim = AnalysisClaim(
        image=image,
        generation=1,
        attempt_number=1,
        claim_token=uuid4(),
        lease_expires_at=datetime.now(UTC),
    )
    status = SubmissionStatus.model_validate(claim.image.model_dump())
    assert status.generation == 1
    assert status.terminal is False
    assert "claim_token" not in status.model_dump(mode="json")
    assert "lease_expires_at" not in status.model_dump(mode="json")
    assert SubmissionStatus(id=image.id, status="needs_review").terminal is True
    with pytest.raises(ValidationError):
        SubmissionStatus(id=image.id, generation=0)


def test_working_png_cannot_leak_through_status_or_formatter_input():
    working = WorkingImage(
        view_id=uuid4(),
        media_type="image/png",
        width=2,
        height=2,
        sha256="a" * 64,
        png_base64="private-png",
        transforms=[],
    )
    page = OcrResponse(
        request_id=uuid4(),
        photo_index=0,
        source_sha256="b" * 64,
        text="hello",
        blocks=[],
        working_image=working,
        profile_id="v1.6",
        runtime_fingerprint="cuda",
        telemetry={},
    )
    formatter = ReceiptInput(
        mode="ocr", image_paths=["original.png"], ocr_pages=[page], source_refs=[]
    )
    assert "private-png" not in formatter.model_dump_json()
    assert "png_base64" not in formatter.model_dump_json()
    assert formatter.ocr_pages[0].working_image.view_id == working.view_id
    assert page.model_dump()["contract_version"] == 2
    with pytest.raises(ValidationError):
        OcrResponse(**(page.model_dump() | {"contract_version": 1}))
    with pytest.raises(ValidationError):
        ReceiptInput(mode="ocr", image_paths=[], expected_receipt={})


def test_mutation_revisions_are_positive_and_independent():
    with pytest.raises(ValidationError):
        RetryRequest(expected_generation=0, request_key="retry")
    with pytest.raises(ValidationError):
        DraftPatch(expected_revision=0, changes={})


def test_source_spans_cannot_be_reversed():
    from vision_bill.model.ocr import SourceRef

    with pytest.raises(ValidationError):
        SourceRef(part_id=uuid4(), photo_index=0, source_sha256="a" * 64, text_span=(10, 2))


def test_receipt_row_loads_json_review_values_and_keeps_revisions_separate():
    from vision_bill.model.db.receipt import ReceiptRow

    row = ReceiptRow(
        id=uuid4(),
        confidence=80,
        merchant_name="Shop",
        date="2025-01-01",
        subtotal="1.20",
        total="1.20",
        operation_revision=3,
        review_revision=2,
        amount_policy='{"tax_basis":"gross","tax_information":"explicit_zero"}',
        findings="[]",
    )
    assert row.operation_revision == 3
    assert row.review_revision == 2
    assert row.amount_policy.tax_information == "explicit_zero"
    assert row.findings == []


def test_ocr_checkpoint_drops_wire_payload_before_persistence():
    from vision_bill.model.workflow import OcrCheckpoint

    page = OcrResponse(
        request_id=uuid4(),
        photo_index=0,
        source_sha256="a" * 64,
        text="text",
        blocks=[],
        profile_id="profile",
        runtime_fingerprint="fp",
        working_image={
            "view_id": uuid4(),
            "width": 1,
            "height": 1,
            "sha256": "b" * 64,
            "png_base64": "secret",
        },
    )
    checkpoint = OcrCheckpoint(
        ordered_source_sha256=["a" * 64],
        profile_id="profile",
        runtime_fingerprint="fp",
        pages=[page.model_dump()],
    )
    assert "png_base64" not in checkpoint.model_dump_json()
    assert checkpoint.pages[0]["working_image"]["view_id"] == str(page.working_image.view_id)
    assert checkpoint.pages[0]["working_image"]["sha256"] == "b" * 64


def test_public_schema_contains_no_internal_lease_or_working_payload():
    from vision_bill.model.extraction import DraftView

    status_schema = str(SubmissionStatus.model_json_schema())
    assert "claim_token" not in status_schema
    assert "lease_expires_at" not in status_schema
    assert "png_base64" not in str(DraftView.model_json_schema())
    assert "png_base64" not in str(ReceiptInput.model_json_schema())


def test_ocr_wire_module_loads_standalone_without_application_config():
    import runpy
    from pathlib import Path

    from vision_bill.model import ocr

    copied_module = runpy.run_path(str(Path(ocr.__file__)))
    assert (
        copied_module["OcrResponse"].model_json_schema()["properties"]["contract_version"]["const"]
        == 2
    )


def test_existing_image_row_responses_cannot_serialize_new_internal_claim_fields():
    image = ImageRow(id=uuid4(), claim_token=uuid4(), lease_expires_at=datetime.now(UTC))
    assert "claim_token" not in image.model_dump(mode="json")
    assert "lease_expires_at" not in image.model_dump(mode="json")


def test_direct_input_views_are_internal_and_cannot_dump_png():
    from test_ocr_provider import wire_page

    from vision_bill.model import extraction
    from vision_bill.model.ocr import OcrResponse
    assert hasattr(extraction, "DerivedInputView"), "direct derivative contract missing"
    page = OcrResponse.model_validate(wire_page())
    view = extraction.DerivedInputView(photo_index=0, source_sha256=page.source_sha256,
                                       working_image=page.working_image)
    envelope = extraction.InferenceEnvelope(provenance=extraction.PipelineProvenance(
        requested_pipeline="direct_vision", used_pipeline="direct_vision", source_image_count=1,
        ordered_source_sha256=[page.source_sha256], model_id="model"),
        telemetry=extraction.InferenceTelemetry(elapsed_ms=0, max_output_tokens=8192), input_views=[view])
    assert envelope.input_views[0].working_image.png_base64
    assert "input_views" not in envelope.model_dump()
    assert "png_base64" not in envelope.model_dump_json()


def test_historical_enrichment_defaults_to_not_recorded_in_receipt_details():
    from test_experiment_scoring import receipt

    from vision_bill.model.db.receipt import ReceiptWithDetails
    from vision_bill.model.extraction import InferenceEnvelope

    history = {
        "provenance": {
            "requested_pipeline": "ocr", "used_pipeline": "ocr", "source_image_count": 1,
            "ordered_source_sha256": ["a" * 64], "model_id": "private-model",
        },
        "telemetry": {"elapsed_ms": 1, "max_output_tokens": 8192},
    }
    envelope = InferenceEnvelope.model_validate(history)
    assert envelope.classification_enrichment.state == "not_recorded"
    details = ReceiptWithDetails.model_validate({
        "receipt": receipt().model_dump() | {"id": uuid4()}, "line_items": [], "taxes": [],
    })
    assert details.classification_enrichment.state == "not_recorded"
    assert "private-model" not in details.model_dump_json()


def test_inference_metadata_projection_keeps_ocr_identity_without_transport_payloads():
    from test_ocr_provider import wire_page

    from vision_bill.model.extraction import (
        DerivedInputView,
        InferenceEnvelope,
        PersistedInferenceEnvelope,
        pack_inference_metadata,
    )
    from vision_bill.model.ocr import OcrPageMetadata

    page = OcrResponse.model_validate(wire_page())
    envelope = InferenceEnvelope.model_validate({
        "provenance": {
            "requested_pipeline": "ocr", "used_pipeline": "ocr", "source_image_count": 1,
            "ordered_source_sha256": [page.source_sha256], "model_id": "model",
        },
        "telemetry": {"elapsed_ms": 1, "max_output_tokens": 8192},
        "ocr_pages": [page],
        "input_views": [DerivedInputView(photo_index=0, source_sha256=page.source_sha256,
                                        working_image=page.working_image)],
    })
    persisted = pack_inference_metadata(envelope)
    assert type(persisted.ocr_pages[0]) is OcrPageMetadata
    assert persisted.ocr_pages[0].working_image.view_id == page.working_image.view_id
    assert persisted.ocr_pages[0].text == page.text
    assert "input_views" not in persisted.model_dump()
    assert "png_base64" not in persisted.model_dump_json()
    assert envelope.ocr_pages[0].working_image.png_base64
    with pytest.raises(ValidationError):
        PersistedInferenceEnvelope.model_validate(envelope.model_dump())
    with pytest.raises(ValidationError):
        PersistedInferenceEnvelope.model_validate(persisted.model_dump() | {"input_views": []})
    persisted.telemetry.calls.append({"image_bytes": "embedded"})
    with pytest.raises(ValidationError):
        PersistedInferenceEnvelope.model_validate(persisted)
    for transport in ({"x": [b"image"]}, {"x": [{"png_base64": "image"}]}):
        envelope.telemetry.calls = [transport]
        with pytest.raises(ValueError):
            pack_inference_metadata(envelope)

    envelope.telemetry.calls = []
    page.working_image.transforms = [{"image_bytes": "embedded"}]
    with pytest.raises(ValueError):
        pack_inference_metadata(envelope)
