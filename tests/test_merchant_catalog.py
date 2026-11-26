import json
import re
from pathlib import Path
from typing import get_args

import pytest

from vision_bill.model.merchant_catalog import (
    is_generic_merchant,
    merchant_candidates,
    merchant_category,
    merchant_choices,
    new_merchant_name,
    suggest_merchant,
)
from vision_bill.model.receipt import Category


@pytest.mark.parametrize(
    ("printed", "display"),
    [
        ("HIT Handelsgruppe GmbH & Co. KG", "HIT Handelsgruppe"),
        ("HIT 003 St. Augustin (HIT Handelsgruppe GmbH & Co. KG)", "HIT Handelsgruppe"),
        ("HIT 027 Siegburg", "HIT Handelsgruppe"),
        ("HIT 003", "HIT Handelsgruppe"),
        ("HIT-Verbrauchermarkt Bonn", "HIT Handelsgruppe"),
        ("EDEKA Peter Schneider e.K.", "EDEKA"),
        ("Hennes & Mauritz B.V. & Co. KG", "H&M"),
        ("H & M Filiale 223", "H&M"),
        ("SNIPES SE", "SNIPES"),
        ("JACK AND JONES Bonn", "JACK & JONES"),
        ("Netto Marken-Discount AG & Co. KG", "Netto Marken-Discount"),
        ("Netto ApS & Co. KG", "Netto mit dem Scottie"),
        ("Kiosk Bonn", "Kiosk"),
        ("Siegburg Kiosk", "Kiosk"),
        ("Kiosk St. Augustin", "Kiosk"),
        ("Kiosk Sankt Augustin", "Kiosk"),
        ("k kiosk Filiale 12", "k kiosk"),
    ],
)
def test_printed_chain_and_generic_names(printed: str, display: str) -> None:
    assert suggest_merchant(printed) == display


@pytest.mark.parametrize(
    "printed",
    [
        "Netto",
        "Netto Bonn",
        "Schmidt am REWE",
        "HITZ Markt",
        "Hit Musikstudio",
        "Hit 027 Musikstudio",
        "Hit 027 Siegburg Musikstudio",
        "ALDI Spezialtransporte",
        "Netto 123 Bonn",
        "SNIPEShop",
        "Müller Bäckerei",
        "Kiosk am Rathaus",
        "Lenas Kiosk Bonn",
        "Kiosk Sonnenschein",
        "Bonn Getränkehandel",
        "Expert Übersetzungen",
        "",
    ],
)
def test_uncertain_matches_keep_the_independent_name(printed: str) -> None:
    assert suggest_merchant(printed, "Marktstraße 13, 53111 Bonn") is None


def test_location_requires_evidence() -> None:
    assert suggest_merchant("Kiosk Leipzig", "Hauptstraße 3, 04109 Leipzig") == "Kiosk"
    assert suggest_merchant("Kiosk Leipzig") is None


def test_create_name_preserves_location_and_spelling() -> None:
    assert new_merchant_name("  Café Lütt Bonn GmbH & Co. KG ") == "Café Lütt Bonn"
    assert new_merchant_name("Schneider e. K.") == "Schneider"
    assert new_merchant_name("Hotel KG am Rhein") == "Hotel KG am Rhein"


def test_picker_searches_aliases_and_categories_cover_the_catalog() -> None:
    assert merchant_choices("Hennes") == ["H&M"]
    assert merchant_choices("NETTO") == ["Netto Marken-Discount", "Netto mit dem Scottie"]
    choices = merchant_choices()
    assert len(choices) == len(set(choices))
    assert all(suggest_merchant(name) == name for name in choices)
    assert {merchant_category(name) for name in choices} == set(get_args(Category))
    assert merchant_category("Café Lütt") is None


def test_only_bare_descriptions_are_generic() -> None:
    assert is_generic_merchant("KIOSK")
    assert is_generic_merchant("Bäckerei GmbH")
    assert not is_generic_merchant("Lenas Kiosk")


@pytest.mark.parametrize(
    ("printed", "address"),
    [
        ("HIT St. Augustin", "Rathausallee 1, 53757 Sankt Augustin"),
        ("HIT Siegburg", "53721 Siegburg"),
        ("HIT 103 Leipzig", "Bahnhofstr. 4, 04109 Leipzig"),
    ],
)
def test_hit_town_matches_printed_address(printed: str, address: str) -> None:
    assert suggest_merchant(printed, address) == "HIT Handelsgruppe"
    assert suggest_merchant("Hit Musikstudio", address) is None


def test_aldi_banner_alone_cannot_identify_regional_company() -> None:
    assert suggest_merchant("ALDI", "53111 Bonn") == "ALDI"
    assert suggest_merchant("ALDI SÜD Filiale 123") == "ALDI SÜD"
    assert suggest_merchant("ALDI Nord Filiale 123") == "ALDI Nord"


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("dm", "drugstore"),
        ("ROSSMANN", "drugstore"),
        ("Müller", "drugstore"),
        ("DocMorris", "pharmacy"),
        ("IKEA", "home_furnishings"),
        ("OBI", "hardware_garden"),
        ("Thalia", "books_stationery"),
        ("Fressnapf", "pet_supplies"),
        ("Decathlon", "sporting_goods"),
        ("DHL", "postal_services"),
    ],
)
def test_merchant_receipt_category(name: str, category: str) -> None:
    assert merchant_category(name) == category


def test_all_receipt_categories_are_available_in_both_editor_languages() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib"
    codes = get_args(Category)
    types = (frontend / "types.ts").read_text()
    picker = types.split("export const CATEGORIES = [", 1)[1].split("]", 1)[0]
    assert tuple(re.findall(r"'([a-z_]+)'", picker)) == codes
    for language in ("de", "en"):
        labels = json.loads((frontend / "i18n" / f"{language}.json").read_text())["categories"]
        assert set(labels) == set(codes)
        assert all(label.strip() for label in labels.values())


@pytest.mark.parametrize(
    ("name", "candidates"),
    [
        ("HIT 027 Siegburg", ["HIT Handelsgruppe"]),
        ("Globus 42 Bonn", ["Globus Baumarkt", "Globus Markthallen"]),
        ("Globus Baumarkt", ["Globus Baumarkt"]),
        ("Netto Bonn", ["Netto Marken-Discount", "Netto mit dem Scottie"]),
        ("Netto ApS", ["Netto mit dem Scottie"]),
        ("ALDI", ["ALDI", "ALDI Nord", "ALDI SÜD"]),
        ("ALDI SÜD", ["ALDI SÜD"]),
        ("Globus Reisen", []),
        ("Netto Musikstudio", []),
        ("ALDI Spedition", []),
        ("Markant Musikstudio", []),
        ("Unbekannte Firma", []),
    ],
)
def test_candidates_require_a_printed_identity(name: str, candidates: list[str]) -> None:
    assert merchant_candidates(name) == candidates
    assert set(candidates) <= set(merchant_choices())
