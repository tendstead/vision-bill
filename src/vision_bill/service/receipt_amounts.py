"""Recover missing receipt amounts from complete, tax-identified positions."""

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext

from ..model.extraction import DraftTaxLine, ExtractionDraft
from ..model.ocr import SourceRef
from ..model.review import AmountPolicy, ReviewFinding


def _currency_unit(currency: str | None) -> Decimal | None:
    # Only round where we know the currency's minor unit.
    if currency in {
        "EUR",
        "USD",
        "GBP",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
        "PLN",
        "CZK",
        "DKK",
        "SEK",
        "NOK",
    }:
        return Decimal("0.01")
    if currency in {"JPY", "KRW", "CLP"}:
        return Decimal(1)
    if currency in {"BHD", "JOD", "KWD", "OMR", "TND"}:
        return Decimal("0.001")
    return None


def _aldi_rates(draft: ExtractionDraft) -> dict[str, Decimal]:
    name = (draft.merchant_name or "").casefold()
    address = (draft.merchant_address or "").casefold()
    if (
        not re.search(r"\baldi[\s-]+s(?:ü|ue)d\b", name)
        or draft.currency != "EUR"
        or draft.date is None
        or draft.date < date(2021, 1, 1)
        or not re.search(
            r"deutschland|germany|\bd-\d{5}\b|(?:straße|strasse|str\.)[^\n,]*[,\s]+\d{5}\b", address
        )
    ):
        return {}
    return {"A": Decimal("0.07"), "B": Decimal("0.19")}


@dataclass
class _TaxBand:
    code: str
    rate: Decimal
    calculated: Decimal | None
    tax_index: int | None
    inputs: list[str]


def _tax_bands(
    draft: ExtractionDraft,
    policy: AmountPolicy,
    indices: list[int],
    discount: Decimal | None,
    unit: Decimal | None,
) -> list[_TaxBand]:
    rows = draft.line_items or []
    taxes = draft.taxes or []
    positions: dict[str, list[int]] = defaultdict(list)
    evidence: dict[str, set[Decimal]] = defaultdict(set)
    for i in indices:
        row = rows[i]
        code = (row.tax_code or "").strip().upper()
        if not code:
            if row.tax_rate is None or abs(int(row.tax_rate.as_tuple().exponent)) > 12:
                return []
            code = f"{row.tax_rate.normalize() * 100:f}%"
        positions[code].append(i)
        if row.tax_rate is not None:
            evidence[code].add(row.tax_rate)
    for tax in taxes:
        code = (tax.name or "").strip().upper()
        if code in positions and tax.rate is not None:
            evidence[code].add(tax.rate)

    fallback = _aldi_rates(draft)
    if any(code in fallback and values != {fallback[code]} for code, values in evidence.items()):
        fallback = {}  # A transaction-specific legend can use another convention entirely.
    rates: dict[str, Decimal] = {}
    for code in positions:
        candidates = evidence[code] or ({fallback[code]} if code in fallback else set())
        if len(candidates) != 1:
            return []
        rate = next(iter(candidates))
        if not 0 <= rate <= 1 or policy.tax_information == "explicit_zero" and rate != 0:
            return []
        rates[code] = rate

    matched: dict[str, int] = {}
    for i, tax in enumerate(taxes):
        name = (tax.name or "").strip().upper()
        alternatives = [code for code, rate in rates.items() if rate == tax.rate]
        if name in positions:
            code = name
        elif name and len(alternatives) == 1:
            code = alternatives[0]
        else:
            return []
        if code in matched:
            return []
        matched[code] = i

    bands = []
    for code, group in positions.items():
        base = sum((rows[i].total_price or Decimal(0) for i in group), Decimal(0)) - (discount or 0)
        rate = rates[code]
        tax_charge = base * rate
        if policy.tax_basis == "gross":
            tax_charge /= 1 + rate
        # Keep recoverable rates even if a missing allocation prevents calculating VAT.
        rounded = None
        if unit is not None and not (discount and len(positions) != 1) and tax_charge >= 0:
            rounded = tax_charge.quantize(unit, rounding=ROUND_HALF_UP)
        inputs = [f"line_items.{i}.total_price" for i in group]
        inputs.extend(f"line_items.{i}.tax_code" for i in group if rows[i].tax_code)
        inputs.extend(f"line_items.{i}.tax_rate" for i in group if rows[i].tax_rate is not None)
        j = matched.get(code)
        if j is not None and taxes[j].rate is not None:
            inputs.append(f"taxes.{j}.rate")
        if not evidence[code]:
            inputs.extend(["merchant_name", "merchant_address", "date", "currency"])
        if discount:
            inputs.append("discount_total")
        bands.append(_TaxBand(code, rate, rounded, j, inputs))
    return bands


def _fill_tax_bands(
    draft: ExtractionDraft,
    policy: AmountPolicy,
    bands: list[_TaxBand],
    record: Callable[[str, Decimal, list[str]], None],
) -> None:
    if not bands:
        return
    taxes = list(draft.taxes or [])
    amounts: list[Decimal] = []
    for band in bands:
        observed = taxes[band.tax_index].amount if band.tax_index is not None else None
        value = band.calculated if observed is None else observed
        if value is not None:
            amounts.append(value)
    total = sum(amounts, Decimal(0)) if len(amounts) == len(bands) else None
    if total is not None and draft.tax_total is not None and total != draft.tax_total:
        return
    for band in bands:
        j = band.tax_index
        if j is None:
            j = len(taxes)
            taxes.append(DraftTaxLine(name=band.code, rate=None, amount=None))
        if taxes[j].rate is None:
            taxes[j].rate = band.rate
            record(f"taxes.{j}.rate", band.rate, band.inputs)
        if taxes[j].amount is None and band.calculated is not None:
            taxes[j].amount = band.calculated
            record(f"taxes.{j}.amount", band.calculated, band.inputs)
    draft.taxes = taxes
    if policy.tax_information == "missing":
        policy.tax_information = "present"
    if draft.tax_total is None and total is not None:
        draft.tax_total = total
        record("tax_total", total, [f"taxes.{i}.amount" for i in range(len(taxes))])


def derive_missing_amounts(
    draft: ExtractionDraft,
    policy: AmountPolicy,
    row_indices: list[int],
    sources: dict[str, list[SourceRef]],
    *,
    tax_summary_complete: bool = False,
) -> list[ReviewFinding]:
    """Caller has established complete positions, adjustments and reconciled overlap indices."""
    rows = draft.line_items or []
    if not row_indices or any(rows[i].total_price is None for i in row_indices):
        return []
    if policy.tax_basis == "unknown":
        return []
    if (
        draft.discount_total
        and policy.discount_treatment == "unknown"
        or policy.discount_treatment == "subtract"
        and draft.discount_total is None
        or draft.tip
        and policy.tip_treatment in {"unknown", "none"}
        or policy.tip_treatment == "add"
        and draft.tip is None
    ):
        return []

    findings: list[ReviewFinding] = []

    def record(path: str, value: Decimal, inputs: list[str]) -> None:
        refs: list[SourceRef] = []
        for key in inputs:
            for ref in sources.get(key) or sources.get(key.rsplit(".", 1)[0], []):
                if ref not in refs:
                    refs.append(ref)
        sources[path] = refs or list(sources.get("input", []))
        findings.append(
            ReviewFinding(
                id=f"calculated_amount:{path}",
                code="calculated_amount",
                severity="info",
                field_path=path,
                source_refs=sources[path],
                comparison={"calculated": str(value), "inputs": ", ".join(inputs)},
            )
        )

    prices = [rows[i].total_price for i in row_indices]
    amounts = [x for x in prices if x is not None]
    line_paths = [f"line_items.{i}.total_price" for i in row_indices]
    discount = draft.discount_total if policy.discount_treatment == "subtract" else None
    tip = draft.tip if policy.tip_treatment == "add" else None
    adjustment_paths = (["discount_total"] if discount else []) + (["tip"] if tip else [])
    unit = _currency_unit(draft.currency)
    with localcontext() as ctx:
        numbers = [*amounts, discount or Decimal(0), tip or Decimal(0)]
        if any(abs(n.adjusted()) > 100 or abs(int(n.as_tuple().exponent)) > 100 for n in numbers):
            return []
        ctx.prec = 220 + len(str(len(amounts)))
        line_sum = sum(amounts, Decimal(0))
        charged = line_sum - (discount or 0) + (tip or 0)
        if policy.tax_basis == "gross":
            # A mismatching observed total can denote a missing or unreadable adjustment.
            if draft.total is not None and draft.total != charged:
                return []
            if draft.total is None:
                draft.total = charged
                record("total", charged, line_paths + adjustment_paths)

        bands = _tax_bands(draft, policy, row_indices, discount, None if tip else unit)
        _fill_tax_bands(draft, policy, bands, record)
        taxes = draft.taxes or []
        if (
            tax_summary_complete
            and taxes
            and all(tax.amount is not None for tax in taxes)
            and draft.tax_total is None
            and policy.tax_information == "present"
        ):
            draft.tax_total = sum((tax.amount or Decimal(0) for tax in taxes), Decimal(0))
            record("tax_total", draft.tax_total, [f"taxes.{i}.amount" for i in range(len(taxes))])
        if (
            policy.tax_information == "explicit_zero"
            and not taxes
            and draft.tax_total is None
            and not any(rows[i].tax_rate for i in row_indices)
        ):
            draft.tax_total = Decimal(0)
            if draft.taxes is None:
                draft.taxes = []
            record("tax_total", draft.tax_total, ["amount_policy.tax_information"])
        if draft.subtotal is None and (
            policy.tax_basis == "net"
            or draft.tax_total is not None
            and not discount
            and not tip
            and policy.tax_information != "missing"
        ):
            net = line_sum if policy.tax_basis == "net" else line_sum - (draft.tax_total or 0)
            draft.subtotal = net
            policy.subtotal_origin = "derived_net"
            record(
                "subtotal", net, line_paths + (["tax_total"] if policy.tax_basis == "gross" else [])
            )
        if draft.subtotal is None and policy.tax_basis == "gross":
            draft.subtotal = line_sum
            policy.subtotal_origin = "derived_gross"
            record("subtotal", line_sum, line_paths)
        if (
            draft.total is None
            and policy.tax_basis == "net"
            and draft.tax_total is not None
            and policy.tax_information != "missing"
        ):
            draft.total = charged + draft.tax_total
            record("total", draft.total, [*line_paths, "tax_total", *adjustment_paths])
    return findings
