"""Conservative, tunable verification of model-proposed receipt overlaps."""

from decimal import Decimal
from difflib import SequenceMatcher

from ..model.receipt import LineItem


def verify_duplicate(
    first: LineItem,
    second: LineItem,
    *,
    name_similarity: float = 1.0,
    quantity_tolerance: Decimal = Decimal(0),
    total_tolerance: Decimal = Decimal(0),
) -> bool:
    """Compare name, item count and line total; caller must establish overlap.

    Defaults require exact equality after name whitespace/case normalization.
    Both quantities must be observed to verify an overlap.
    Similarity is in [0, 1]; numeric tolerances are absolute differences.
    Never use this alone to deduplicate a whole receipt: repeated purchases are valid.
    """
    if not 0 <= name_similarity <= 1 or quantity_tolerance < 0 or total_tolerance < 0:
        raise ValueError("Invalid duplicate verification tolerances")
    if first.quantity is None or second.quantity is None:
        return False
    left = " ".join(first.description.casefold().split())
    right = " ".join(second.description.casefold().split())
    return bool(left and right) and (
        SequenceMatcher(None, left, right, autojunk=False).ratio() >= name_similarity
        and abs(Decimal(str(first.quantity)) - Decimal(str(second.quantity))) <= quantity_tolerance
        and abs(first.total_price - second.total_price) <= total_tolerance
    )
