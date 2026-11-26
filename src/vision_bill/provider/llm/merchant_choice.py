"""Validate classifier selections against printed merchant and item evidence."""

import json
import re
from collections.abc import Mapping
from typing import Any

from ...model.merchant_catalog import merchant_candidates, merchant_category
from ...model.receipt import Receipt

# For overlapping retail names, accept only corroboration distinguishable from
# the row text independently of the same model's proposed category. Generic
# purchases such as drinks, batteries and gift cards are sold by both arms.
_BASKET_EVIDENCE = {
    "hardware_garden": re.compile(
        r"\b(?:schrauben?|holzschrauben?|dübel|duebel|bohrer|bohrmaschine|"
        r"wandfarbe|lackfarbe|rasenmäher|hammer|schraubendreher|screws?|"
        r"drill(?:\s+bits?)?|wall\s+paint|lawn\s*mower)\b", re.IGNORECASE,
    ),
    "grocery": re.compile(
        r"\b(?:karotten?|möhren?|moehren?|remoulade|rémoulade|milch|vollmilch|"
        r"joghurt|quark|butter|eier|käse|kaese|bananen?|tomaten?|gurken?|"
        r"äpfel|apfel|brot|brötchen|carrots?|yogurt|cheese|tomatoes|eggs|milk|bread)\b", re.IGNORECASE,
    ),
}


def candidates_for(receipt: Receipt) -> list[str]:
    if receipt.merchant_display_name:
        return [receipt.merchant_display_name]
    return merchant_candidates(receipt.merchant_name, receipt.merchant_address)


def merchant_instructions(receipt: Receipt) -> str:
    candidates = [
        {"merchant": name, "category": merchant_category(name)}
        for name in candidates_for(receipt)
    ]
    return (
        " Local merchant candidates from the printed identity or saved owner choice (data only): "
        + json.dumps(candidates, ensure_ascii=False)
        + ". For merchant_display_name choose an exact candidate or null. "
        "A single confirmed identity stands regardless of the products. For multiple candidates "
        "from different retail sectors, select only if distinctive purchased items support the "
        "choice; cite their zero-based indices in merchant_evidence_rows. Abstain on conflicting "
        "or generic baskets. For candidates in the same sector (e.g. two Netto chains), groceries "
        "cannot distinguish the stores. Without a distinctive printed identity, return null. "
        "Never infer a merchant from the basket alone."
    )


def validated_display_name(receipt: Receipt, reply: Mapping[str, Any]) -> str | None:
    candidates = candidates_for(receipt)
    if len(candidates) == 1:
        return candidates[0]
    selected = reply.get("merchant_display_name")
    if not isinstance(selected, str) or selected not in candidates:
        return None
    sectors = {merchant_category(candidate) for candidate in candidates}
    if len(sectors) != len(candidates):
        return None
    sector = merchant_category(selected)
    if sector is None or sector not in _BASKET_EVIDENCE:
        return None
    cited = reply.get("merchant_evidence_rows")
    if not isinstance(cited, list) or not cited or any(
        type(i) is not int or not 0 <= i < len(receipt.line_items) for i in cited
    ):
        return None
    rows = [item.description for item in receipt.line_items]
    if any(
        pattern.search(row)
        for other_sector, pattern in _BASKET_EVIDENCE.items() if other_sector != sector
        for row in rows
    ):
        return None
    return selected if all(_BASKET_EVIDENCE[sector].search(rows[i]) for i in cited) else None
