"""Versioned extraction diagnostics; historical scoring remains untouched."""
import unicodedata
from datetime import time
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from ..model.receipt import Receipt

DETAIL_SCORING_VERSION = "4"
LEGACY_DETAIL_SCORING_VERSION = "3"
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


def detailed_score_v3(expected: Receipt, actual: Receipt, *, classification: bool = True) -> dict[str, Any]:
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
            # These v3 fields were JSON numbers. Their scale never affected strict scores.
            if path.endswith((".quantity", ".rate")):
                exact = equal
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
    return {"version": LEGACY_DETAIL_SCORING_VERSION, "scores": scores,
            "differences": differences,
            "rows": {"missing": missing, "extra": extra, "matched": matched},
            "checks": {
                "tax_lines_sum_matches_tax_total": sum((t.amount for t in actual.taxes), Decimal(0)) == actual.tax_total,
                "schema_net_plus_tax_matches_total": actual.subtotal - (actual.discount_total or 0) + actual.tax_total + (actual.tip or 0) == actual.total,
                "line_totals_sum": str(sum((item.total_price for item in actual.line_items), Decimal(0))),
                "note": "Arithmetic checks are diagnostics; tax-inclusive receipt conventions may differ from the schema."
            }}


OBSERVABLE_FIELDS = (*HEADER, *TOTALS, 'line_items', 'taxes')


def _description(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).split()).casefold()


def _item_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _description(row['description']),
        *(None if value is None else Decimal(str(value))
          for value in (row.get('quantity'), row.get('unit_price'), row['total_price'])),
    )


def _lcs(left: list[tuple[Any, ...]], right: list[tuple[Any, ...]]) -> list[list[int]]:
    """Maximum ordered exact matches, then lexicographically earliest index pairs."""
    n, m = len(left), len(right)
    lengths = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            lengths[i][j] = (1 + lengths[i + 1][j + 1] if left[i] == right[j]
                             else max(lengths[i + 1][j], lengths[i][j + 1]))
    matches: list[list[int]] = []
    start_i = start_j = 0
    remaining = lengths[0][0]
    while remaining:
        found = False
        for i in range(start_i, n):
            for j in range(start_j, m):
                if left[i] == right[j] and 1 + lengths[i + 1][j + 1] == remaining:
                    matches.append([i, j])
                    start_i, start_j = i + 1, j + 1
                    remaining -= 1
                    found = True
                    break
            if found:
                break
    return matches


def detailed_score(
    expected: Receipt | dict[str, Any], actual: Receipt | None, *, classification: bool = True,
    scored_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Add exact receipt/occurrence metrics without redefining historical v3 scores.

    The frozen mask names observable top-level fields; row collections include every
    ordered printed field and occurrence. Classification and confidence never qualify.
    A failed extraction remains a zero-score observation in the dataset denominator.
    """
    mask = list(OBSERVABLE_FIELDS) if scored_fields is None else list(scored_fields)
    if set(mask) - set(OBSERVABLE_FIELDS):
        raise ValueError('Unknown observable scoring field')
    left = expected.model_dump(mode='json') if isinstance(expected, Receipt) else expected
    legacy_expected = expected if isinstance(expected, Receipt) else Receipt.model_validate(expected)
    failed = actual is None
    right = actual.model_dump(mode='json') if actual is not None else {}
    litems = [_item_key(row) for row in left['line_items']]
    ritems = [_item_key(row) for row in right.get('line_items', [])]
    matches = _lcs(litems, ritems)
    tp, fp, fn = len(matches), len(ritems) - len(matches), len(litems) - len(matches)
    equality: dict[str, bool] = {}
    for key in HEADER:
        e, a = left[key], right.get(key)
        equal = normalized(e) == normalized(a)
        if key == 'time' and isinstance(e, str) and isinstance(a, str):
            try:
                equal = time.fromisoformat(e) == time.fromisoformat(a)
            except ValueError:
                pass
        equality[key] = equal
    for key in TOTALS:
        e, a = left.get(key, getattr(legacy_expected, key)), right.get(key)
        equality[key] = e == a if e is None or a is None else Decimal(str(e)) == Decimal(str(a))
    equality['line_items'] = litems == ritems
    def tax_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (normalized(row['name']), *(None if row[k] is None else Decimal(str(row[k]))
                                          for k in ('rate', 'amount')))
    equality['taxes'] = [tax_key(row) for row in left['taxes']] == [
        tax_key(row) for row in right.get('taxes', [])]
    result: dict[str, Any] = detailed_score_v3(legacy_expected, actual, classification=classification) if actual is not None else {
        'scores': {key: 0.0 for key in ('header', 'totals', 'line_items', 'taxes', 'strict',
                                      'normalized', 'ordered_items')},
        'differences': [], 'checks': {},
        'rows': {'missing': len(litems), 'extra': 0, 'matched': 0},
    }
    result.update(
        version=DETAIL_SCORING_VERSION, legacy_detail_version=LEGACY_DETAIL_SCORING_VERSION,
        scored_fields=mask, extraction_failed=failed,
        full_receipt_correct=not failed and all(equality[key] for key in mask),
        item_tp=tp, item_fp=fp, item_fn=fn, item_matches=matches,
        precision=tp / (tp + fp) if tp + fp else (0.0 if failed or fn else 1.0),
        recall=tp / (tp + fn) if tp + fn else (0.0 if failed else 1.0),
        exact_item_count=not failed and len(litems) == len(ritems),
        exact_item_order=not failed and litems == ritems,
        exact_amounts=not failed and all(equality[key] for key in TOTALS)
        and [row[1:] for row in litems] == [row[1:] for row in ritems]
        and equality['taxes'],
    )
    return result
