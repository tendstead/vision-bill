"""The persisted observation boundary admits complete metadata for all photos."""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vision_bill.model import workflow
from vision_bill.model.extraction import EXTRACTION_SCHEMA_VERSION, SOURCE_PROJECTION_VERSION
from vision_bill.service.benchmark_scoring import PROMPT_VERSION


def observation_payload(
    *,
    parts: list[dict[str, Any]] | None = None,
    ocr: bool = False,
) -> dict[str, Any]:
    if parts is None:
        parts = [
            {
                "part_id": uuid4(),
                "photo_index": index,
                "source_sha256": letter * 64,
                "image_path": f"/private/original-{index}.png",
            }
            for index, letter in enumerate(("a", "b"))
        ]
    digest = "sha256:recorded-model-digest"
    mode = "ocr" if ocr else "direct_vision"
    observations = {
        "confidence": 90,
        "merchant_name": "Merchant",
        "date": "2026-01-01",
        "currency": "EUR",
        "line_items": [],
        "taxes": [],
        "subtotal": "90071992547.91",
        "discount_total": "0.00",
        "tax_total": "0.00",
        "total": "90071992547.91",
        "payment_method": "unknown",
    }
    sources = [
        {key: part[key] for key in ("part_id", "photo_index", "source_sha256")}
        | {"coordinate_space": "original"}
        for part in parts
    ]
    views = [
        {key: part[key] for key in ("part_id", "photo_index", "source_sha256")}
        | {
            "artifact_path": f"/private/view-{part['photo_index']}.png",
            "metadata": {"view_id": uuid4(), "width": 120, "height": 240, "sha256": "c" * 64},
        }
        for part in parts
    ]
    return {
        "checkpoint_id": uuid4(),
        "producer_generation": 1,
        "producer_attempt_number": 1,
        "eligible_generation": 1,
        "source_parts": parts,
        "model_id": "test",
        "execution_identity": {
            "inference_identity": "openai:" + "d" * 64,
            "semantic_policy_sha256": "e" * 64,
            "model_id": "test",
            "model_digest": digest,
            "prompt_version": PROMPT_VERSION,
            "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
            "source_projection_version": SOURCE_PROJECTION_VERSION,
        },
        "envelope": {
            "draft": observations.copy(),
            "receipt": observations.copy(),
            "field_sources": {"input": sources},
            "source_projection_version": SOURCE_PROJECTION_VERSION,
            "provenance": {
                "requested_pipeline": "auto",
                "used_pipeline": mode,
                "source_image_count": len(parts),
                "ordered_source_sha256": [part["source_sha256"] for part in parts],
                "model_id": "test",
                "model_digest": digest,
                "profile_id": "profile" if ocr else None,
                "runtime_fingerprint": "recorded-runtime" if ocr else None,
            },
            "telemetry": {"elapsed_ms": 1, "max_output_tokens": 8192},
            "classification_enrichment": {"state": "pending"},
            "ocr_pages": [
                {
                    "request_id": uuid4(),
                    "photo_index": view["photo_index"],
                    "source_sha256": view["source_sha256"],
                    "profile_id": "profile",
                    "runtime_fingerprint": "recorded-runtime",
                    "working_image": view["metadata"],
                    "text": "Merchant",
                    "blocks": [],
                }
                for view in views
            ]
            if ocr
            else [],
        },
        "persisted_views": views,
    }


@pytest.mark.parametrize("ocr", [False, True], ids=["direct", "ocr"])
def test_checkpoint_preserves_complete_observations_without_bytes(ocr):
    checkpoint = workflow.ObservationCheckpoint.model_validate(observation_payload(ocr=ocr))

    serialized = checkpoint.model_dump_json()
    assert '"total":"90071992547.91"' in serialized
    assert "png_base64" not in serialized
    assert checkpoint.envelope.provenance.runtime_fingerprint == (
        "recorded-runtime" if ocr else None
    )
    assert workflow.ObservationCheckpoint.model_validate_json(serialized) == checkpoint


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("payload", "belongs in saved artifacts"),
        ("prefix", "all OCR sources"),
        ("view_prefix", "working view"),
        ("unexpected_model_digest", "execution identity"),
        ("unbound_generation", "producer generation"),
        ("input_prefix", "ordered original source references"),
    ],
)
def test_checkpoint_rejects_unbound_or_incomplete_observations(corruption, message):
    payload = observation_payload(ocr=True)
    envelope = payload["envelope"]
    if corruption == "payload":
        envelope["ocr_pages"][0]["working_image"]["png_base64"] = "Ynl0ZXM="
    if corruption == "prefix":
        envelope["ocr_pages"].pop()
    if corruption == "view_prefix":
        payload["persisted_views"].pop()
    if corruption == "unexpected_model_digest":
        envelope["provenance"]["model_digest"] = "a-different-model"
    if corruption == "unbound_generation":
        payload["eligible_generation"] = 2
    if corruption == "input_prefix":
        envelope["field_sources"]["input"].pop()

    with pytest.raises(ValidationError, match=message):
        workflow.ObservationCheckpoint.model_validate(payload)
