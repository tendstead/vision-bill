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
        for name in ("category", "spending_category", "tags", "merchant_display_name"):
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
        for name in ("category", "spending_category", "tags", "merchant_display_name"):
            props.pop(name, None)
    schema["properties"]["line_items_complete"] = {
        "type": "boolean",
        "description": "True only if all charged and credited positions and any separate "
                       "discounts, fees and tips are legible and accounted for across all photos.",
    }
    schema["properties"]["tax_summary_complete"] = {
        "type": "boolean", "description": "True only when every printed tax band and amount is legible.",
    }
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
        "Keep printed merchant names and product descriptions in their original language. "
        "Keep missing or unreadable values null. Preserve the original figures even if they "
        "disagree. Convert a decimal comma to a decimal point for JSON: 1,25 becomes 1.25. "
        "Use headings, currency symbols, multiplication marks and row alignment to identify columns. "
        "Do not classify, categorize or tag. Group physical text lines into logical purchases. "
        "A product name or department with a tax code may introduce a following SKU, size, "
        "colour and price. Illustrative example only: the two lines\nMug 19%\nSKU 481 / Blue   4,50\n"
        'describe one item with description "Mug Blue" and total_price 4.50. Do not create an '
        "item for a department heading, continuation or tax code. If OCR split the name and price, "
        "cite both blocks for the item when available. Keep a clearly separate purchase with an "
        "unreadable price as its own item with total_price null. Keep each independently charged "
        "occurrence of an identical product. Preserve item order within and across photos, "
        "including items visible again in adjacent overlapping photos. "
        "Each line_item must include source_image (zero-based photo index). "
        "Return only one flat JSON object using the field meanings in the nullable schema below. "
        "Include amount_policy based only on printed evidence: tax_basis gross means line "
        "amounts already include tax (do not add the VAT again), net means tax is additive, "
        "unknown means evidence is insufficient. "
        "discount_treatment included_in_lines means already applied, subtract means a separate "
        "discount, otherwise unknown. tip_treatment included means already in line amounts, "
        "add means added to the receipt total, outside_receipt means paid separately, none means "
        "no tip, otherwise unknown. Set subtotal_origin to printed for a printed subtotal on "
        "the same basis as the items, or printed_net for an explicitly printed receipt-wide "
        "net subtotal, including a net grand total in a tax table. The net base of one tax band "
        "alone does not establish a printed receipt-wide subtotal. Leave missing receipt totals, "
        "subtotal and VAT amounts null for deterministic calculation. Set line_items_complete "
        "false if positions or adjustments may be cut off, obscured or missing. Record a printed "
        "tax band in each position's tax_code and a directly printed percentage in tax_rate. "
        "For a tax legend such as A 7% and B 19%, keep separate taxes rows with the printed "
        "band as name and fractional rate. Letters vary by receipt. A missing or unreadable "
        "percentage is null, never 0. Preserve tax-coded discounts and refunds as signed "
        "positions if already counted in the line amounts. Tax_information present means "
        "observed tax information, explicit_zero requires printed zero/exempt evidence, missing means absent "
        "or unreadable and must never be inferred as zero. Do not invent financial defaults. "
        "Also include duplicate_candidates as pairs {first_index,second_index} only for the "
        "complete ordered suffix/prefix overlap between adjacent photos. Never search globally "
        "for duplicate products. Include overlap_decisions for all overlap proposals. "
        "Use field_sources to cite the evidence for individual fields when identifiable, "
        "especially each tax rate and amount, subtotal and total. For example, taxes.0.rate "
        "cites the printed percentage even though the JSON value is a fraction. Use only "
        "supplied source identities and exact block coordinates in their coordinate_space. "
        "Leave unsupported mappings absent. "
        "Source refs: "
        + json.dumps(
            [
                ref.model_dump(mode="json")
                for ref in (input.source_refs if source_refs is None else source_refs)
            ],
            separators=(",", ":"),
        )
        + "\nNullable schema: "
        + json.dumps(schema, separators=(",", ":"))
        + "\nObservations:\n"
        + observed
    )
