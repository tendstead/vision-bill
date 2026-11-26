"""Pure, versioned review of observed values; never repair printed arithmetic."""

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import ValidationError

from ..model.extraction import (
    DraftLineItem,
    DraftTaxLine,
    ExtractionDraft,
    InferenceEnvelope,
    PersistedInferenceEnvelope,
)
from ..model.item_category import CATEGORY_CODE_SET, SpendingCategory
from ..model.ocr import SourceRef
from ..model.product_tags import canonical_product_tag
from ..model.receipt import LineItem, Receipt, TaxLine
from ..model.review import (
    CHECKS_VERSION,
    AmountPolicy,
    FindingAcknowledgement,
    OriginalAvailability,
    ReviewEvaluation,
    ReviewFinding,
)


def editable_draft(envelope: InferenceEnvelope | PersistedInferenceEnvelope) -> ExtractionDraft:
    """Copy accepted enrichment to the editor while retaining the original row order and evidence."""
    assert envelope.draft is not None
    document = envelope.draft.model_copy(deep=True)
    receipt = envelope.receipt
    enrichment = envelope.classification_enrichment
    if receipt is None or enrichment.state not in ("completed", "unavailable"):
        return document
    unresolved = set(enrichment.missing_fields)
    if document.category is None and "category" not in unresolved:
        document.category = receipt.category

    raw_rows = document.line_items or []
    removed = {
        step.get("removed_index") for step in envelope.overlap_decisions
        if step.get("verified") is True and type(step.get("removed_index")) is int
    }
    kept = [row for index, row in enumerate(raw_rows) if index not in removed]
    if len(kept) != len(receipt.line_items):
        return document
    for index, (raw, labeled) in enumerate(zip(kept, receipt.line_items, strict=True)):
        if raw.description != labeled.description:
            continue
        if raw.tags is None and f"line_items.{index}.tags" not in unresolved:
            raw.tags = list(labeled.tags)
        if (raw.spending_category in (None, "unknown")
                and f"line_items.{index}.spending_category" not in unresolved):
            raw.spending_category = labeled.spending_category
    return document


# Match a complete product name to avoid treating an ingredient, e.g. carrot cake, as produce.
_PRODUCT_NAMES = {
    "möhre": "carrots", "möhren": "carrots", "karotte": "carrots", "carrot": "carrots",
    "remoulade": "sauces", "p&w remoulade": "sauces",
    "vitalgebäck": "pastries", "vitalgebäck mischung": "pastries",
    "grillfackel": "meat", "grillfackeln": "meat", "nackensteak": "meat",
    "rinderhack": "beef", "schweinenackensteak": "pork",
    "flaschenpfand": "deposit", "einwegpfand": "deposit", "mehrwegpfand": "deposit",
    "pfandrückgabe": "deposit", "leergut": "deposit", "bottle deposit": "deposit",
    "energy editions": "soft_drinks", "energy drink": "soft_drinks", "energydrink": "soft_drinks",
    "aufkleber": "stationery", "aufkleber set": "stationery", "sticker set": "stationery",
}
_PRODUCT_GROUPS = {
    tag: category for category, tags in {
        "vegetables": "carrots tomatoes onions potatoes salad mushrooms",
        "fruit": "apples bananas berries citrus", "bread_bakery": "bread rolls pastries",
        "meat": "beef pork poultry sausages", "fish": "seafood", "milk": "plant_milk",
        "pasta_rice_grains": "rice pasta", "cereals": "oats cereal",
        "butter_spreads": "butter spreads", "sauces_spices": "sauces spices",
        "coffee_tea": "coffee tea", "soft_drinks": "juice", "alcohol": "beer wine",
        "snacks_sweets": "biscuits chips chocolate ice_cream sweets",
        "cleaning": "cleaning_products laundry_detergent dish_soap dishwasher_tablets",
        "personal_care": "hygiene cosmetics shampoo soap toothpaste deodorant menstrual_products",
        "health": "medicines first_aid", "household": "bin_bags kitchenware paper_towels toilet_paper",
        "services": "service subscription", "clothing": "shoes", "hardware": "tools",
        "pet": "pet_food", "other": "books magazines stationery toys flowers",
    }.items() for tag in tags.split()
}


def _product_tag(product: str) -> str | None:
    # Strip abbreviation periods, e.g. "Hackfl. gem.", while retaining word boundaries.
    product = re.sub(r"(?<=\w)\.(?=\s|$)", "", product)
    if re.fullmatch(r"(?:hackfl(?:eisch)?|minced meat)(?: (?:gem|gemischt))?", product):
        return "meat"
    if re.fullmatch(r"(?:hfs )?s[- ]nackensteaks?", product):
        return "pork"
    if re.fullmatch(r"(?:hähn(?:chen)?[- ]?brust(?:f(?:ilet)?)?|chicken breast)", product):
        return "poultry"
    if re.fullmatch(
        r"(?:h&m )?(?:(?:damen|herren|kinder)[ -])?"
        r"(?:shorts|pyjamas?|schlafanzug|t[- ]?shirts?|jeans|pullover|leggings)"
        r"(?: (?:blau|blue|schwarz|black|weiss|white|xs|s|m|l|xl))*", product,
    ):
        return "clothing"
    return _PRODUCT_NAMES.get(product) or canonical_product_tag(product)


def suggest_product_labels(document: ExtractionDraft) -> None:
    """Seed empty editable labels from identifiable products; retain explicit empty tag lists."""
    for row in document.line_items or []:
        name = " ".join((row.description or "").casefold().split())
        name = re.sub(r"(?:\s+(?:xxl|\d+(?:[.,]\d+)?\s*(?:kg|g|ml|cl|l)))+$", "", name)
        organic = name.startswith(("bio ", "bio-", "organic "))
        product = re.sub(r"^(?:bio[ -]|organic )", "", name)
        tag = _product_tag(product)
        if tag is None:
            continue
        category = _PRODUCT_GROUPS.get(tag, tag)
        if row.tags is None:
            row.tags = [tag]
            if tag != category and category in {"fruit", "vegetables", "meat"}:
                row.tags.append(category)
            if category == "soft_drinks":
                row.tags.append("beverage")
            if organic and tag != "organic":
                row.tags.append("organic")
        if row.spending_category in (None, "unknown") and category in CATEGORY_CODE_SET:
            row.spending_category = cast(SpendingCategory, category)


def observation_values(document: Receipt | ExtractionDraft) -> dict[str, Any]:
    """Project optional enrichment defaults without changing any printed observation."""
    values = document.model_dump()
    if values.get("category") is None:
        values.pop("category", None)
    for row in values.get("line_items") or []:
        for key in ("tags", "spending_category"):
            if row.get(key) is None:
                row.pop(key, None)
    return values


def receipt_from_draft(document: ExtractionDraft) -> Receipt:
    """Preserve null observations; only optional enrichment uses Receipt defaults."""
    try:
        receipt = Receipt.model_validate(observation_values(document))
    except ValidationError as exc:
        raise ValueError("draft_incomplete") from exc
    if not receipt.merchant_name.strip() or any(
        not item.description.strip() for item in receipt.line_items
    ):
        raise ValueError("draft_incomplete")
    return receipt


def _canonical(value: Any) -> Any:
    if isinstance(value, Decimal):
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, (date, UUID)):
        return str(value)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _review_factor(value: Any) -> Any:
    # Earlier finding IDs used JSON numbers for quantity and tax rate. Keep those
    # IDs when the old representation preserves the exact observation.
    if isinstance(value, Decimal):
        legacy = float(value)
        if Decimal(str(legacy)) == value:
            return legacy
    return value


def material_document(receipt: Receipt | ExtractionDraft) -> dict[str, Any]:
    """Ordered printed observations, excluding classification/tag/identity metadata."""
    fields = ExtractionDraft.model_fields.keys() - {"confidence", "category", "merchant_display_name"}
    data = {key: value for key, value in receipt.model_dump().items() if key in fields}
    if data.get("line_items") is not None:
        data["line_items"] = [
            {key: item.get(key) for key in ("description", "quantity", "unit_price", "total_price")}
            for item in data["line_items"]
        ]
    if data.get("taxes") is not None:
        data["taxes"] = [
            {key: item.get(key) for key in ("name", "rate", "amount")} for item in data["taxes"]
        ]
    for item in data.get("line_items") or []:
        item["quantity"] = _review_factor(item["quantity"])
    for tax in data.get("taxes") or []:
        tax["rate"] = _review_factor(tax["rate"])
    return _canonical(data)  # type: ignore[no-any-return]


def _still_calculated(receipt: Receipt | ExtractionDraft, finding: ReviewFinding) -> bool:
    amount = finding.comparison.get("calculated")
    if amount is None:
        return False
    value: Any = receipt
    for segment in finding.field_path.split("."):
        if isinstance(value, list):
            if not segment.isdecimal() or int(segment) >= len(value):
                return False
            value = value[int(segment)]
        else:
            value = getattr(value, segment, None)
    try:
        return isinstance(value, Decimal) and value == Decimal(amount)
    except InvalidOperation:
        return False


def evaluate_receipt(
    receipt: Receipt | ExtractionDraft,
    amount_policy: AmountPolicy,
    field_sources: dict[str, list[SourceRef]],
    overlap_decisions: list[dict[str, Any]],
    originals: list[OriginalAvailability],
    *,
    retained_findings: Sequence[ReviewFinding] = (),
) -> ReviewEvaluation:
    material = {
        "document": material_document(receipt),
        "amount_policy": amount_policy.model_dump(),
        "field_sources": {
            k: [r.model_dump(mode="json") for r in v] for k, v in field_sources.items()
        },
        "overlap_decisions": overlap_decisions,
        "originals": [
            {
                "part_id": str(o.part_id) if o.part_id is not None else None,
                "photo_index": o.photo_index,
            }
            for o in originals
        ],
        "checks_version": CHECKS_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    findings: list[ReviewFinding] = []

    def finding(
        code: str,
        path: str,
        severity: Literal["integrity", "review"] = "review",
        comparison: dict[str, str | None] | None = None,
    ) -> None:
        identity = hashlib.sha256(f"{digest}:{code}:{path}".encode()).hexdigest()
        findings.append(
            ReviewFinding(
                id=identity,
                code=code,
                severity=severity,
                field_path=path,
                source_refs=field_sources.get(path, []),
                comparison=comparison or {},
            )
        )

    try:
        complete = Receipt.model_validate(observation_values(receipt))
    except ValidationError as exc:
        complete = None
        for error in exc.errors():
            finding("missing_mandatory_field", ".".join(map(str, error["loc"])), "integrity")
    if (not receipt.merchant_name or not receipt.merchant_name.strip()) and not any(
        f.field_path == "merchant_name" for f in findings
    ):
        finding("missing_mandatory_field", "merchant_name", "integrity")
    observed_items: Sequence[LineItem | DraftLineItem] = receipt.line_items or []
    for index, item in enumerate(observed_items):
        if not item.description or not item.description.strip():
            path = f"line_items.{index}.description"
            if not any(f.field_path == path for f in findings):
                finding("missing_mandatory_field", path, "integrity")
    observed_taxes: Sequence[TaxLine | DraftTaxLine] = receipt.taxes or []
    for index, tax in enumerate(observed_taxes):
        if tax.rate is not None and not 0 <= tax.rate <= 1:
            finding("invalid_tax_rate", f"taxes.{index}.rate", "integrity")
    if not originals:
        finding("missing_original", "originals", "integrity")
    for original in originals:
        if not original.available or not original.sha256_matches:
            finding(
                "original_integrity_failed",
                f"originals.{original.part_id if original.part_id is not None else original.photo_index}",
                "integrity",
            )
    original_ids = {o.part_id for o in originals}
    for path, refs in field_sources.items():
        if any(ref.part_id not in original_ids for ref in refs):
            finding("source_integrity_failed", path, "integrity")
        if not refs or any(ref.coordinate_space == "unknown" for ref in refs):
            finding("unknown_source_mapping", path)
    for index, decision in enumerate(overlap_decisions):
        if decision.get("verified") is not True:
            finding("ambiguous_overlap", f"overlap_decisions.{index}")
    if amount_policy.tax_basis == "unknown":
        finding("unknown_tax_basis", "amount_policy.tax_basis")
    if amount_policy.subtotal_origin == "unknown":
        finding("unknown_subtotal_origin", "amount_policy.subtotal_origin")
    if amount_policy.tax_information == "missing":
        finding("missing_tax_information", "amount_policy.tax_information")
    if receipt.discount_total and amount_policy.discount_treatment == "unknown":
        finding("unknown_discount_treatment", "amount_policy.discount_treatment")
    if receipt.tip and amount_policy.tip_treatment in {"unknown", "none"}:
        finding("unknown_tip_treatment", "amount_policy.tip_treatment")
    if complete is not None:
        # Use enough precision for exact sums/products of the admitted monetary inputs.
        discount = complete.discount_total or Decimal(0)
        tip = complete.tip or Decimal(0)
        numbers = [
            complete.subtotal,
            complete.total,
            discount,
            complete.tax_total,
            tip,
        ]
        for item in complete.line_items:
            numbers.append(item.total_price)
            if item.quantity is not None and item.unit_price is not None:
                numbers.extend((item.quantity, item.unit_price))
        numbers += [tax.amount for tax in complete.taxes]
        with localcontext() as context:
            context.prec = max(
                50,
                sum(len(n.as_tuple().digits) + abs(int(n.as_tuple().exponent)) for n in numbers)
                + 10,
            )

            def compare(
                code: str, path: str, observed: Decimal, expected: Decimal, tolerance: str
            ) -> None:
                if abs(observed - expected) > Decimal(tolerance):
                    finding(
                        code,
                        path,
                        comparison={
                            "observed": str(observed),
                            "expected": str(expected),
                            "tolerance": tolerance,
                        },
                    )

            for index, item in enumerate(complete.line_items):
                if item.quantity is not None and item.unit_price is not None:
                    compare(
                        "line_discrepancy",
                        f"line_items.{index}.total_price",
                        item.total_price,
                        item.quantity * item.unit_price,
                        "0.01",
                    )
            supported = (
                amount_policy.tax_basis != "unknown"
                and amount_policy.subtotal_origin != "unknown"
                and not any(
                    f.code in {"unknown_discount_treatment", "unknown_tip_treatment"}
                    for f in findings
                )
            )
            subtotal_is_net = amount_policy.tax_basis == "net" or amount_policy.subtotal_origin in (
                "derived_net",
                "printed_net",
            )
            if subtotal_is_net and amount_policy.tax_information == "missing":
                supported = False
            gross_lines_with_net_subtotal = amount_policy.tax_basis == "gross" and subtotal_is_net
            if gross_lines_with_net_subtotal and (
                (discount and amount_policy.discount_treatment == "subtract")
                or (tip and amount_policy.tip_treatment == "add")
            ):
                finding("uncertain_net_subtotal_basis", "subtotal")
                supported = False
            if supported:
                expected_subtotal = sum(
                    (item.total_price for item in complete.line_items), Decimal(0)
                )
                if gross_lines_with_net_subtotal:
                    expected_subtotal -= complete.tax_total
                compare(
                    "subtotal_discrepancy", "subtotal", complete.subtotal, expected_subtotal, "0.05"
                )
            if amount_policy.tax_information != "missing":
                compare(
                    "tax_discrepancy",
                    "tax_total",
                    complete.tax_total,
                    sum((tax.amount for tax in complete.taxes), Decimal(0)),
                    "0.02",
                )
                if amount_policy.tax_information == "explicit_zero" and complete.tax_total != 0:
                    finding("tax_information_conflict", "tax_total")
            if supported:
                expected = complete.subtotal
                if subtotal_is_net:
                    expected += complete.tax_total
                if amount_policy.discount_treatment == "subtract":
                    expected -= discount
                if amount_policy.tip_treatment == "add":
                    expected += tip
                compare("amount_discrepancy", "total", complete.total, expected, "0.05")
    for retained in retained_findings:
        if retained.code == "calculated_amount" and not _still_calculated(receipt, retained):
            continue
        if not any(
            f.code == retained.code and f.field_path == retained.field_path for f in findings
        ):
            identity = hashlib.sha256(
                f"{digest}:{retained.code}:{retained.field_path}".encode()
            ).hexdigest()
            findings.append(retained.model_copy(update={"id": identity}))
    integrity = any(f.severity == "integrity" for f in findings)
    return ReviewEvaluation(
        findings=findings,
        can_finalize=not integrity,
        can_auto_verify=not findings
        and not integrity
        and all(o.part_id is not None for o in originals),
        checks_version=CHECKS_VERSION,
    )


def persistent_findings(findings: Sequence[ReviewFinding]) -> list[ReviewFinding]:
    """Retain provider observations while recalculating our own prior check results."""
    recalculated_codes = {
        "missing_mandatory_field",
        "invalid_tax_rate",
        "missing_original",
        "original_integrity_failed",
        "source_integrity_failed",
        "unknown_source_mapping",
        "ambiguous_overlap",
        "unknown_tax_basis",
        "unknown_subtotal_origin",
        "missing_tax_information",
        "unknown_discount_treatment",
        "unknown_tip_treatment",
        "line_discrepancy",
        "subtotal_discrepancy",
        "tax_discrepancy",
        "tax_information_conflict",
        "amount_discrepancy",
        "uncertain_net_subtotal_basis",
    }
    return [finding for finding in findings if finding.code not in recalculated_codes]


def validate_acknowledgements(
    evaluation: ReviewEvaluation, acknowledgements: Sequence[FindingAcknowledgement]
) -> None:
    """An integrity failure can never be converted into a human exception."""
    if not evaluation.can_finalize or any(f.severity == "integrity" for f in evaluation.findings):
        raise ValueError("receipt_integrity_failed")
    ids = [ack.finding_id for ack in acknowledgements]
    expected = {f.id for f in evaluation.findings if f.severity == "review"}
    if (
        len(ids) != len(set(ids))
        or set(ids) != expected
        or any(not a.comment.strip() for a in acknowledgements)
    ):
        raise ValueError("review_acknowledgements_invalid")
