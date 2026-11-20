"""Public contract and artifact boundary tests (no database needed)."""

import importlib.util
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vision_bill.model import workflow


def test_transaction_provider_exists():
    assert importlib.util.find_spec("vision_bill.provider.db.workflow_db") is not None


def test_artifact_boundary_requires_saved_path():
    assert hasattr(workflow, "PersistedWorkingView"), "typed saved artifact contract missing"
    with pytest.raises(ValidationError):
        workflow.PersistedWorkingView(  # type: ignore[attr-defined]
            part_id=uuid4(),
            photo_index=0,
            source_sha256="a" * 64,
            metadata={
                "view_id": str(uuid4()),
                "media_type": "image/png",
                "width": 1,
                "height": 1,
                "sha256": "b" * 64,
            },
        )


def test_outcome_cannot_complete_without_receipt_and_review():
    with pytest.raises(ValidationError):
        workflow.AttemptOutcome(kind="complete")


def test_raw_png_in_generic_submission_metadata_is_rejected():
    with pytest.raises(ValidationError):
        workflow.SubmissionCreate(
            id=uuid4(),
            parts=[{"image_path": "/tmp/original"}],
            source_manifest=[{"photo_index": 0, "source_sha256": "a" * 64}],
            requested_options={"png_base64": "secret"},
            effective_options={},
            request_fingerprint="fingerprint",
        )


def test_reference_collection_does_not_skip_nested_reference_objects():
    from vision_bill.provider.db.workflow_db import _nested_source_refs

    first, nested = uuid4(), uuid4()
    value = {
        "part_id": str(first),
        "photo_index": 0,
        "source_sha256": "a" * 64,
        "nested": {
            "source_refs": [{"part_id": str(nested), "photo_index": 1, "source_sha256": "b" * 64}]
        },
    }
    assert {ref.part_id for ref in _nested_source_refs(value)} == {first, nested}
    assert _nested_source_refs({"elapsed_ms": 1, "stages": ["ocr", "format"]}) == []


def test_standalone_saved_view_references_are_collected_recursively():
    from vision_bill.provider.db import workflow_db

    assert hasattr(workflow_db, "_nested_view_ids")
    view = uuid4()
    assert workflow_db._nested_view_ids({"telemetry": {"view_id": str(view)}}) == [view]


def test_retry_policy_is_explicit_and_strict():
    assert workflow.RetryRequest(expected_generation=1, request_key="r").policy == "preserve"
    assert workflow.RetryRequest(expected_generation=1, request_key="r", policy="current").policy == "current"
    with pytest.raises(ValidationError):
        workflow.RetryRequest(expected_generation=1, request_key="r", policy="unknown")


@pytest.mark.asyncio
async def test_active_legacy_draft_hides_obsolete_optional_field_findings(monkeypatch):
    from vision_bill.provider.db.workflow_db import WorkflowDB

    db = object.__new__(WorkflowDB)
    monkeypatch.setattr(db, "_source_views", AsyncMock(return_value=[]))
    fields = ("discount_total", "line_items.0.quantity", "line_items.0.unit_price", "total")
    row = {
        "image_id": uuid4(), "revision": 2, "document": {}, "amount_policy": {},
        "findings": [
            {"id": path, "code": "missing_mandatory_field", "severity": "integrity", "field_path": path}
            for path in fields
        ],
        "field_sources": {}, "overlap_decisions": [], "finalized_receipt_id": None,
    }
    view = await db._draft_view(None, row)
    assert [finding.field_path for finding in view.findings] == ["total"]
    assert len(row["findings"]) == 4
