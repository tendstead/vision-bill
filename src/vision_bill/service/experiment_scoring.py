"""Versioned extraction diagnostics; historical scoring remains untouched."""
import unicodedata
from datetime import time
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from ..model.receipt import Receipt

DETAIL_SCORING_VERSION = "3"
HEADER = ("merchant_name", "merchant_address", "receipt_number", "date", "time", "currency", "payment_method")
TOTALS = ("subtotal", "discount_total", "tax_total", "tip", "total")
ITEM = ("description", "quantity", "unit_price", "total_price")
TAX = ("name", "rate", "amount")


def normalized(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    return value


def _fraction(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 1.0


def detailed_score(expected: Receipt, actual: Receipt, *, classification: bool = True) -> dict[str, Any]:
    left, right = expected.model_dump(mode="json"), actual.model_dump(mode="json")
    differences: list[dict[str, Any]] = []
    strict: dict[str, list[bool]] = {k: [] for k in ("header", "totals", "line_items", "taxes")}
    normal: dict[str, list[bool]] = {k: [] for k in strict}
    labels: list[bool] = []

    def field(group: str, path: str, e: Any, a: Any, numeric: bool = False) -> None:
        exact = e == a
        if numeric and e is not None and a is not None:
            # Decimal serialization differences (12.9 vs 12.90) are equivalent;
            # a one-cent error is not forgiven. No lossy rounding is applied.
            equal = Decimal(str(e)) == Decimal(str(a))
        else:
            equal = normalized(e) == normalized(a)
        if path == "time" and isinstance(e, str) and isinstance(a, str):
            try:
                equal = time.fromisoformat(e) == time.fromisoformat(a)
            except ValueError:
                pass  # Unrecognized strings still require literal normalized equality.
        strict[group].append(exact)
        normal[group].append(equal)
        if not exact:
            differences.append({"path": path, "expected": e, "actual": a,
                                "equivalent": equal})

    for name in HEADER:
        field("header", name, left[name], right[name])
    for name in TOTALS:
        field("totals", name, left[name], right[name], numeric=True)
    missing = extra = matched = 0
    for group, fields, key in (("line_items", ITEM, "description"), ("taxes", TAX, "name")):
        es, ac = left[group], right[group]
        matcher = SequenceMatcher(a=[normalized(x[key]) for x in es],
                                  b=[normalized(x[key]) for x in ac], autojunk=False)
        for _, i, j, k, l in matcher.get_opcodes():
            paired = min(j - i, l - k)
            for offset in range(paired):
                e, a = es[i + offset], ac[k + offset]
                if group == "line_items" and normalized(e[key]) == normalized(a[key]):
                    matched += 1
                for name in fields:
                    field(group, f"{group}[{i + offset}].{name}", e[name], a[name],
                          numeric=name in {"quantity", "unit_price", "total_price", "rate", "amount"})
                if group == "line_items" and classification:
                    if e["spending_category"] != "unknown":
                        labels.append(normalized(e[key]) == normalized(a[key])
                                      and e["spending_category"] == a["spending_category"])
                    labels.append(set(e["tags"]) == set(a["tags"]))
            for index in range(i + paired, j):
                strict[group].extend([False] * len(fields))
                normal[group].extend([False] * len(fields))
                differences.append({"path": f"{group}[{index}]", "expected": es[index],
                                    "actual": None, "equivalent": False})
                if group == "line_items":
                    missing += 1
                    if classification and es[index]["spending_category"] != "unknown":
                        labels.append(False)
                    if classification and es[index]["tags"]:
                        labels.append(False)
            for index in range(k + paired, l):
                strict[group].extend([False] * len(fields))
                normal[group].extend([False] * len(fields))
                differences.append({"path": f"{group}[extra:{index}]", "expected": None,
                                    "actual": ac[index], "equivalent": False})
                if group == "line_items":
                    extra += 1
    if classification:
        labels.append(left["category"] == right["category"])
    weights = {"header": .20, "totals": .35, "line_items": .30, "taxes": .15}
    components = {k: _fraction(v) for k, v in normal.items()}
    scores = {**components,
              "strict": sum(_fraction(strict[k]) * w for k, w in weights.items()),
              "normalized": sum(components[k] * w for k, w in weights.items()),
              "classification": _fraction(labels) if classification else None,
              "ordered_items": _fraction([
                  index < len(right["line_items"]) and all(
                      item[field] == right["line_items"][index][field] for field in ITEM
                  )
                  for index, item in enumerate(left["line_items"])
              ]) if len(left["line_items"]) == len(right["line_items"]) else 0.0}
    return {"version": DETAIL_SCORING_VERSION, "scores": scores,
            "differences": differences,
            "rows": {"missing": missing, "extra": extra, "matched": matched},
            "checks": {
                "tax_lines_sum_matches_tax_total": sum((t.amount for t in actual.taxes), Decimal(0)) == actual.tax_total,
                "schema_net_plus_tax_matches_total": actual.subtotal - actual.discount_total + actual.tax_total + (actual.tip or 0) == actual.total,
                "line_totals_sum": str(sum((item.total_price for item in actual.line_items), Decimal(0))),
                "note": "Arithmetic checks are diagnostics; tax-inclusive receipt conventions may differ from the schema."
            }}
