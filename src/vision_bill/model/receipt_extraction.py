"""Multi-photo extraction metadata, kept separate from the persisted receipt schema."""

from pydantic import BaseModel, Field

from ..helper.duplicates import verify_duplicate
from .receipt import LineItem, Receipt


class SourceLineItem(LineItem):
    source_image: int = Field(ge=0, strict=True, description="Zero-based input photo index")


class DuplicateCandidate(BaseModel):
    first_index: int = Field(ge=0, strict=True, description="Earlier line_items array index")
    second_index: int = Field(ge=0, strict=True, description="Repeated line_items array index")


class MultiImageReceipt(Receipt):
    line_items: list[SourceLineItem]  # type: ignore[assignment]
    duplicate_candidates: list[DuplicateCandidate] = Field(
        description="Pairs of array indices showing the SAME printed row in adjacent photos"
    )

    def reconciled(self, image_count: int) -> Receipt:
        sources = [item.source_image for item in self.line_items]
        if sources != sorted(sources) or any(source >= image_count for source in sources):
            raise ValueError(
                "Line items must be grouped in input photo order with valid source_image"
            )
        groups = validate_overlap(sources, self.duplicate_candidates)

        removed: set[int] = set()
        for pairs in groups.values():
            if all(verify_duplicate(self.line_items[a], self.line_items[b]) for a, b in pairs):
                removed.update(b for _, b in pairs)

        payload = self.model_dump(exclude={"duplicate_candidates", "line_items"})
        payload["line_items"] = [
            item.model_dump(exclude={"source_image"})
            for index, item in enumerate(self.line_items)
            if index not in removed
        ]
        return Receipt.model_validate(payload)


def validate_overlap(
    sources: list[int], candidates: list[DuplicateCandidate]
) -> dict[int, list[tuple[int, int]]]:
    """Validate source-boundary structure independently of missing monetary observations."""
    groups: dict[int, list[tuple[int, int]]] = {}
    for pair in candidates:
        a, b = pair.first_index, pair.second_index
        if not 0 <= a < b < len(sources):
            raise ValueError("Duplicate candidate indices are out of order or out of range")
        if sources[b] != sources[a] + 1:
            raise ValueError("Duplicate candidates must come from adjacent photos")
        groups.setdefault(sources[a], []).append((a, b))

    for source, pairs in groups.items():
        pairs.sort()
        earlier = [i for i, value in enumerate(sources) if value == source]
        later = [i for i, value in enumerate(sources) if value == source + 1]
        count = len(pairs)
        # Only an ordered suffix/prefix can be an overlap. Never search globally.
        expected = list(zip(earlier[-count:], later[:count], strict=False))
        if pairs != expected:
            raise ValueError("Duplicate candidates must describe an ordered photo-boundary overlap")
    return groups
