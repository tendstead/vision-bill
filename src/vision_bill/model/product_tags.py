"""Bilingual labels shared by the web editor and the model's classification contract."""

import json
from collections.abc import Sequence
from functools import cache
from importlib.resources import files


@cache
def _labels() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = json.loads(
        files(__package__).joinpath("product_tags.json").read_text(encoding="utf-8")
    )
    return data


@cache
def _label_codes() -> dict[str, str]:
    return {
        label.casefold(): code
        for code, translations in _labels().items()
        for label in (code, *translations.values())
    }


def canonical_product_tag(value: str) -> str | None:
    """Resolve a built-in code or its printed DE/EN label."""
    return _label_codes().get(value.strip().casefold())


def product_tag_catalog(available: Sequence[str]) -> dict[str, dict[str, str]]:
    """Look up the bilingual labels for built-in codes in the supplied vocabulary."""
    labels = _labels()
    return {code: labels[code.casefold()] for code in available if code.casefold() in labels}
