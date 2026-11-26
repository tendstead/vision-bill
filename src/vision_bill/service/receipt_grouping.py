"""Combine separately charged purchases for the ledger and draft editor."""

from collections.abc import Sequence
from decimal import Decimal, localcontext
from typing import Any, TypeVar

from ..model.extraction import (
    DraftLineItem,
    ExtractionDraft,
    InferenceEnvelope,
    PersistedInferenceEnvelope,
)
from ..model.ocr import SourceRef
from ..model.receipt import LineItem, Receipt

Document = TypeVar("Document", ExtractionDraft, Receipt)


def _purchase_units(row: DraftLineItem | LineItem) -> tuple[Decimal, Decimal] | None:
    total = row.total_price
    if not (row.description or "").strip() or total is None or not total.is_finite() or total <= 0:
        return None
    quantity = row.quantity or Decimal(1)
    unit = row.unit_price
    if unit is None:
        if quantity != 1:
            return None
        unit = total
    # Keep arithmetic exact in the local context even for unusually large model output.
    if any(
        not value.is_finite()
        or len(value.as_tuple().digits) > 20
        or abs(int(value.as_tuple().exponent)) > 12
        for value in (quantity, unit, total)
    ):
        return None
    return (quantity, unit) if quantity * unit == total else None


def group_purchases(
    document: Document,
    field_sources: dict[str, list[SourceRef]],
    *,
    tax_keys: Sequence[object] | None = None,
    row_mapping: dict[int, int] | None = None,
) -> tuple[Document, dict[str, list[SourceRef]]]:
    """Aggregate matching purchases and project their sources onto the new row indices."""
    rows = document.line_items or []
    if len(rows) < 2 or tax_keys is not None and len(tax_keys) != len(rows):
        return document, field_sources
    positions: dict[tuple[object, ...], int] = {}
    output: list[DraftLineItem | LineItem] = []
    quantities: dict[int, Decimal] = {}
    remap: dict[int, int] = {}
    combined: set[int] = set()
    with localcontext(prec=50):
        for index, row in enumerate(rows):
            units = _purchase_units(row)
            if units is None:
                remap[index] = len(output)
                output.append(row)
                continue
            quantity, unit = units
            key = (
                " ".join((row.description or "").casefold().split()),
                unit,
                tax_keys[index] if tax_keys is not None else None,
                row.spending_category,
                tuple(sorted(row.tags)) if row.tags is not None else None,
            )
            position = positions.setdefault(key, len(output))
            remap[index] = position
            if position == len(output):
                quantities[position] = quantity
                output.append(row)
                continue
            quantities[position] += quantity
            output[position] = output[position].model_copy(
                update={
                    "quantity": quantities[position],
                    "unit_price": unit,
                    "total_price": quantities[position] * unit,
                }
            )
            combined.add(position)
    if not combined:
        return document, field_sources

    if row_mapping is not None:
        row_mapping.update(remap)
    projected: dict[str, list[SourceRef]] = {}
    all_row_refs: dict[int, list[SourceRef]] = {}
    for path, refs in field_sources.items():
        prefix, _, tail = path.partition(".")
        index_text, separator, suffix = tail.partition(".")
        if prefix != "line_items" or not index_text.isdecimal() or int(index_text) not in remap:
            projected[path] = list(refs)
            continue
        position = remap[int(index_text)]
        key_path = f"line_items.{position}{separator}{suffix}"
        current = projected.setdefault(key_path, [])
        current.extend(ref for ref in refs if ref not in current)
        if position in combined:
            all_refs = all_row_refs.setdefault(position, [])
            all_refs.extend(ref for ref in refs if ref not in all_refs)
    for position, refs in all_row_refs.items():
        path = f"line_items.{position}"
        for suffix in ("", ".quantity", ".unit_price", ".total_price"):
            projected[path + suffix] = refs.copy()
    return document.model_copy(update={"line_items": output}), projected


def group_extracted_purchases(
    document: Document,
    envelope: InferenceEnvelope | PersistedInferenceEnvelope,
) -> tuple[Document, dict[str, list[SourceRef]]]:
    # An editable draft still holds the original occurrences. Published receipts
    # already exclude verified overlaps and carry their projected source indices.
    draft = isinstance(document, ExtractionDraft)
    fields = (
        (envelope.raw_field_sources or envelope.field_sources) if draft else envelope.field_sources
    )
    if any(
        draft or decision.get("verified") is not True for decision in envelope.overlap_decisions
    ) or any(finding.field_path.startswith("line_items.") for finding in envelope.findings):
        return document, fields
    observed_rows = (envelope.draft.line_items or []) if envelope.draft is not None else []
    removed = {decision.get("removed_index") for decision in envelope.overlap_decisions}
    bands = [
        (row.tax_code, row.tax_rate)
        for index, row in enumerate(observed_rows)
        if index not in removed
    ]
    return group_purchases(document, fields, tax_keys=bands or None)


def match_grouped_reference(
    reference: Receipt,
    extracted: Receipt,
    metadata: dict[str, Any],
) -> Receipt:
    """Use the compact view only when it matches the frozen purchase structure."""
    if len(extracted.line_items) <= len(reference.line_items):
        return extracted
    overlaps = metadata.get("overlap_decisions") or []
    if any(overlap.get("verified") is not True for overlap in overlaps):
        return extracted
    removed = {overlap.get("removed_index") for overlap in overlaps}
    raw = (metadata.get("draft") or {}).get("line_items") or []
    bands = [
        (row.get("tax_code"), row.get("tax_rate"))
        for index, row in enumerate(raw)
        if index not in removed
    ]
    row_mapping: dict[int, int] = {}
    grouped, _ = group_purchases(
        extracted,
        {},
        tax_keys=bands or None,
        row_mapping=row_mapping,
    )

    def structure(document: Receipt) -> list[tuple[str, Decimal | None, Decimal | None, Decimal]]:
        return [
            (
                " ".join(row.description.casefold().split()),
                row.quantity,
                row.unit_price,
                row.total_price,
            )
            for row in document.line_items
        ]

    if grouped is extracted or structure(grouped) != structure(reference):
        return extracted
    enrichment = metadata.get("classification_enrichment")
    if isinstance(enrichment, dict):
        missing = []
        for path in enrichment.get("missing_fields", []):
            index, separator, field = path.removeprefix("line_items.").partition(".")
            if path.startswith("line_items.") and index.isdecimal() and int(index) in row_mapping:
                path = f"line_items.{row_mapping[int(index)]}{separator}{field}"
            missing.append(path)
        enrichment["missing_fields"] = list(dict.fromkeys(missing))
    return grouped
