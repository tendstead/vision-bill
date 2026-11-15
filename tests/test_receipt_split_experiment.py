"""Safety checks for the standalone experiment; no production integration."""

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from vision_bill.model.receipt import Receipt

spec = importlib.util.spec_from_file_location(
    "receipt_split_experiment", Path(__file__).parents[1] / "scripts/receipt_split_experiment.py"
)
assert spec and spec.loader
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def part(names, **fields):
    return experiment.PartialReceipt(
        line_items=[
            {"description": name, "quantity": 1, "unit_price": "1.00", "total_price": "1.00"}
            for name in names
        ],
        **fields,
    )


def names(payload):
    return [item["description"] for item in payload["line_items"]]


def test_retains_repeated_purchases_inside_a_photo_and_ambiguous_single_row():
    payload, review = experiment.merge_parts([part(["A", "A"]), part(["A", "B"])])
    assert names(payload) == ["A", "A", "A", "B"]
    assert review["boundaries"][1]["ambiguous_singleton_retained"]
    assert review["requires_review"]


def test_removes_only_ordered_multi_row_boundary_matches_and_flags_review():
    payload, review = experiment.merge_parts([part(["A", "B", "C"]), part(["B", "C", "D", "A"])])
    assert names(payload) == ["A", "B", "C", "D", "A"]
    assert review["boundaries"][1]["removed_prefix"] == 2
    assert review["boundaries"][1]["requires_review"]


def test_same_names_with_different_unit_prices_are_not_removed():
    first, second = part(["A", "B"]), part(["A", "B"])
    second.line_items[0].unit_price = Decimal("2.00")
    payload, _ = experiment.merge_parts([first, second])
    assert names(payload) == ["A", "B", "A", "B"]


def test_conflicting_headers_and_totals_remain_visible_without_summing():
    payload, review = experiment.merge_parts(
        [
            part(["A"], merchant_name="First", total="10.00"),
            part(["B"], merchant_name="Second", total="11.00"),
        ]
    )
    assert payload["merchant_name"] == "First"
    assert str(payload["total"]) == "11.00"
    assert {c["field"] for c in review["conflicts"]} == {"merchant_name", "total"}


def test_missing_header_is_not_fabricated_to_satisfy_receipt_schema():
    payload, _ = experiment.merge_parts([part(["A"])])
    with pytest.raises(ValidationError):
        Receipt.model_validate(payload)
    assert "merchant_name" not in payload
    assert "date" not in payload
    assert "total" not in payload


def test_crops_cover_whole_image_in_order_with_overlap(tmp_path):
    original = tmp_path / "photo.png"
    Image.new("RGB", (100, 2000), "white").save(original)
    crops = experiment.source_parts(
        {"id": "synthetic", "images": [original]}, "vision_tiles", tmp_path, 1000
    )
    assert [c["crop"] for c in crops] == [
        [0, 0, 100, 1000],
        [0, 850, 100, 1850],
        [0, 1700, 100, 2000],
    ]
    for crop in crops:
        with Image.open(crop["path"]) as image:
            assert image.size == (100, crop["crop"][3] - crop["crop"][1])


@pytest.mark.asyncio
async def test_control_repairs_invalid_overlap_metadata_before_accepting_reply(tmp_path):
    import json
    from types import SimpleNamespace

    from vision_bill.model.receipt_extraction import MultiImageReceipt

    def payload(sources):
        return {
            "confidence": 90,
            "merchant_name": "Synthetic",
            "date": "2026-01-01",
            "subtotal": "1",
            "total": "1",
            "line_items": [
                {
                    "description": "Same printed row",
                    "quantity": 1,
                    "unit_price": "1",
                    "total_price": "1",
                    "source_image": source,
                }
                for source in sources
            ],
            "duplicate_candidates": [{"first_index": 0, "second_index": 1}],
        }

    class Client:
        def __init__(self):
            self.responses = iter([payload([0, 0]), payload([0, 1])])

        async def chat(self, **kwargs):
            return SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(next(self.responses)))
            )

    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in paths:
        path.write_bytes(b"synthetic image bytes; fake transport")
    value, trace, _, _ = await experiment.request(
        Client(),
        "synthetic",
        "revision",
        "extract",
        paths,
        MultiImageReceipt,
        tmp_path,
        1,
        {},
    )
    assert len(value.reconciled(2).line_items) == 1
    assert "adjacent photos" in trace["attempts"][0]["validation_error"]
    assert len(trace["attempts"]) == 2


def test_text_chunks_preserve_ordered_source_lines_and_charge_ocr_once():
    original = [f"printed row {i}" for i in range(40)]
    chunks = experiment.chunk_ocr_parts(
        [{"text": "\n".join(original), "ocr_ms": 100, "artifact": "synthetic"}], 16
    )
    assert [c["row_range"] for c in chunks] == [[0, 16], [13, 29], [26, 40]]
    reconstructed = chunks[0]["text"].splitlines()
    for chunk in chunks[1:]:
        reconstructed.extend(chunk["text"].splitlines()[3:])
    assert reconstructed == original
    assert sum(c["ocr_ms"] for c in chunks) == 100
