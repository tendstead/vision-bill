"""Small helpers for request telemetry and extraction-only prompts."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ...model.extraction import ReceiptInput
from ...model.ocr import SourceRef
from ...model.receipt import Receipt
from ...model.receipt_extraction import MultiImageReceipt


def extraction_prompt(image_count: int, *, ocr: bool = False) -> str:
    schema = (MultiImageReceipt if image_count > 1 else Receipt).model_json_schema()
    for obj in [schema, *schema.get("$defs", {}).values()]:
        props = obj.get("properties", {})
        for name in ("category", "spending_category", "tags"):
            props.pop(name, None)
            if name in obj.get("required", []):
                obj["required"].remove(name)
    overlap = (
        "Extract each printed row from each photo, including overlaps. Set source_image to its "
        "zero-based photo index. Group rows by photo and then top-to-bottom. In duplicate_candidates "
        "identify the complete ordered overlap using zero-based line_items indices. Do not remove "
        "duplicates yourself. Totals must come from printed totals, not duplicated rows."
        if image_count > 1
        else "Keep line_items in printed top-to-bottom order."
    )
    source = "the supplied OCR text" if ocr else f"{image_count} ordered photos"
    return (
        f"Extract the literal contents of ONE receipt from {source}. "
        "Return a populated receipt object, not the JSON schema. "
        f"{overlap} "
        "Preserve genuinely repeated purchases. Copy product names and abbreviations as printed; "
        "do not expand, translate, categorize or tag them. Read merchant_name from printed text, "
        "never guess from a logo. Positive prices are purchases; refunds and returned deposits "
        "have negative prices. Do not invent missing values. If unreadable, return only "
        '{"error":{"code":"unreadable","message":"Which part needs a clearer photo"}}. '
        "Return only JSON matching this schema: " + json.dumps(schema)
    )


def chat_telemetry(
    response: Any, messages: Sequence[Mapping[str, Any]], elapsed_ms: float
) -> dict[str, Any]:
    prompt = [{"role": m.get("role"), "content": m.get("content")} for m in messages]
    result: dict[str, Any] = {
        "elapsed_ms": elapsed_ms,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    for field in ("load_duration", "prompt_eval_duration", "eval_duration"):
        value = getattr(response, field, None)
        if isinstance(value, (int, float)):
            result[field.removesuffix("_duration") + "_ms"] = value / 1_000_000
    for field in ("prompt_eval_count", "eval_count"):
        value = getattr(response, field, None)
        if isinstance(value, int):
            result[field] = value
    model = getattr(response, "model", None)
    if isinstance(model, str):
        result["actual_model"] = model
    reason = getattr(response, "done_reason", None)
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        reason = getattr(choices[0], "finish_reason", reason)
    if isinstance(reason, str):
        result["finish_reason"] = reason
    thinking = getattr(getattr(response, "message", None), "thinking", None)
    if isinstance(thinking, str):
        result["thinking_characters"] = len(thinking)
    usage = getattr(response, "usage", None)
    for field in ("prompt_tokens", "completion_tokens"):
        value = getattr(usage, field, None)
        if isinstance(value, int):
            result[field] = value
    return result


def endpoint_fingerprint(provider: Any) -> str | None:
    client = getattr(provider, "_client", None)
    url = getattr(client, "base_url", None)
    if url is None:
        url = getattr(getattr(client, "_client", None), "base_url", None)
    return hashlib.sha256(str(url).encode()).hexdigest() if url is not None else None


def source_extraction_prompt(
    input: ReceiptInput, *, source_refs: Sequence[SourceRef] | None = None
) -> str:
    """Classification-free observations with explicit photo/source boundaries, never PNG data."""
    from ...model.extraction import ExtractionDraft
    from ...model.review import AmountPolicy

    schema = ExtractionDraft.model_json_schema()
    schema.setdefault("$defs", {})["AmountPolicy"] = AmountPolicy.model_json_schema()
    schema["properties"]["amount_policy"] = {"$ref": "#/$defs/AmountPolicy"}
    schema.setdefault("required", []).append("amount_policy")
    for obj in [schema, *schema.get("$defs", {}).values()]:
        props = obj.get("properties", {})
        for name in ("category", "spending_category", "tags"):
            props.pop(name, None)
    line_schema = schema.get("$defs", {}).get("DraftLineItem", {})
    line_schema.setdefault("properties", {})["source_image"] = {"type": "integer", "minimum": 0}
    observed = (
        "\n".join(
            f"PHOTO {page.photo_index}\n"
            + page.model_dump_json(exclude={"telemetry": True, "working_image": {"transforms"}})
            for page in input.ocr_pages
        )
        if input.mode == "ocr"
        else f"{len(input.image_paths)} ordered photos"
    )
    return (
        "Extract literal observations for ONE complete receipt from ALL ordered input photos. "
        "Keep missing or unreadable values null; never invent mandatory values or correct numbers. "
        "Do not classify, categorize or tag. Preserve every original printed row in photo order, "
        "then top-to-bottom, including overlapping rows and genuinely repeated purchases. "
        "Each line_item must include source_image (zero-based photo index). "
        "Return a flat populated receipt object following the nullable schema below. "
        "Include amount_policy based only on printed evidence: tax_basis gross means line "
        "amounts include tax, net means tax is additive, unknown means evidence is insufficient. "
        "discount_treatment included_in_lines means already applied, subtract means a separate "
        "discount, otherwise unknown. tip_treatment included means already in line amounts, "
        "add means added to the receipt total, outside_receipt means paid separately, none means "
        "no tip, otherwise unknown. subtotal_origin printed means the observed printed subtotal; "
        "derived_net means line amounts converted to net using explicit known tax evidence "
        "before separately applied discounts/tips; derive only with sufficient inputs, otherwise "
        "leave subtotal null and origin unknown. tax_information present means observed tax "
        "information, explicit_zero requires printed zero/exempt evidence, missing means absent "
        "or unreadable and must never be inferred as zero. Do not invent financial defaults. "
        "Also include duplicate_candidates as pairs {first_index,second_index} only for the "
        "complete ordered suffix/prefix overlap between adjacent photos. Never search globally "
        "for duplicate products. Include overlap_decisions for all overlap proposals. "
        "Optional field_sources map field paths to supplied SourceRef objects; use only supplied "
        "identities and block coordinates in the indicated coordinate_space. "
        "Source refs: "
        + json.dumps(
            [
                ref.model_dump(mode="json")
                for ref in (input.source_refs if source_refs is None else source_refs)
            ]
        )
        + "\nNullable schema: "
        + json.dumps(schema)
        + "\nObservations:\n"
        + observed
    )
