"""Small helpers for request telemetry and extraction-only prompts."""
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ...model.receipt import Receipt
from ...model.receipt_extraction import MultiImageReceipt


def extraction_prompt(image_count: int) -> str:
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
        if image_count > 1 else "Keep line_items in printed top-to-bottom order."
    )
    return (
        f"Extract the literal contents of ONE receipt from {image_count} ordered photos. {overlap} "
        "Preserve genuinely repeated purchases. Copy product names and abbreviations as printed; "
        "do not expand, translate, categorize or tag them. Read merchant_name from printed text, "
        "never guess from a logo. Positive prices are purchases; refunds and returned deposits "
        "have negative prices. Do not invent missing values. If unreadable, return only "
        '{"error":{"code":"unreadable","message":"Which part needs a clearer photo"}}. '
        "Return only JSON matching this schema: " + json.dumps(schema)
    )


def chat_telemetry(response: Any, messages: Sequence[Mapping[str, Any]], elapsed_ms: float) -> dict[str, Any]:
    prompt = [{"role": m.get("role"), "content": m.get("content")} for m in messages]
    result: dict[str, Any] = {
        "elapsed_ms": elapsed_ms,
        "prompt_sha256": hashlib.sha256(json.dumps(prompt, sort_keys=True, default=str).encode()).hexdigest(),
    }
    for field in ("load_duration", "prompt_eval_duration", "eval_duration"):
        value = getattr(response, field, None)
        if isinstance(value, (int, float)):
            result[field.removesuffix("_duration") + "_ms"] = value / 1_000_000
    for field in ("prompt_eval_count", "eval_count"):
        value = getattr(response, field, None)
        if isinstance(value, int):
            result[field] = value
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
