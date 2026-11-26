"""Local merchant suggestions for German receipts. Printed evidence stays with the receipt."""

import re
import unicodedata

# The first spelling is the selectable display name. Additional spellings are
# recognizable trading names, legal entities, or common printed variants.
_GROUPS = {
    "grocery": (
        ("EDEKA", "E Center", "EDEKA Center"),
        ("REWE", "REWE Markt"),
        ("ALDI SÜD", "Aldi Sued"),
        ("ALDI Nord",),
        ("ALDI",),
        ("Lidl",),
        ("Kaufland",),
        ("PENNY",),
        ("Norma",),
        ("Netto Marken-Discount", "Netto Markendiscount", "Netto Marken Discount"),
        ("Netto mit dem Scottie", "Netto ApS", "Netto Stavenhagen"),
        ("HIT Handelsgruppe", "HIT", "HIT Markt", "HIT Verbrauchermarkt"),
        ("Globus Markthallen", "Globus Markthalle", "Globus SB Warenhaus"),
        ("Denns BioMarkt", "denn’s Biomarkt"),
        ("Alnatura",),
        ("Bio Company",),
        ("Marktkauf",),
        ("nahkauf",),
        ("famila",),
        ("Combi Verbrauchermarkt",),
        ("Getränke Hoffmann",),
        ("Trinkgut",),
        ("WASGAU", "WASGAU Frischemarkt"),
        ("K+K Klaas & Kock", "Klaas & Kock"),
        ("ebl-naturkost",),
        ("diska",),
        ("Tante Enso",),
        ("MARKANT",),
        ("Feinkost Käfer",),
        ("Vinzenzmurr",),
        ("HOL’AB!", "Holab Getränkemarkt"),
        ("Kiosk",),
        ("k kiosk",),
    ),
    "drugstore": (
        ("dm", "dm drogerie markt", "dm Drogeriemarkt"),
        ("ROSSMANN", "Dirk Rossmann"),
        ("Müller", "Müller Drogerie", "Müller Handels", "Müller Drogeriemarkt"),
        ("BUDNI", "Budnikowsky", "Iwan Budnikowsky"),
    ),
    "pharmacy": (
        ("DocMorris",),
        ("easyApotheke",),
    ),
    "clothing": (
        ("H&M", "H & M", "Hennes & Mauritz", "H und M"),
        ("JACK & JONES", "Jack and Jones", "Jack Jones"),
        ("SNIPES",),
        ("C&A", "C & A"),
        ("Peek & Cloppenburg",),
        ("Deichmann",),
        ("ZARA",),
        ("New Yorker",),
        ("Ernsting’s family", "Ernstings family"),
        ("Takko Fashion", "Takko"),
        ("KiK", "KiK Textilien"),
        ("Primark",),
        ("TK Maxx",),
        ("Hunkemöller",),
        ("s.Oliver",),
        ("bonprix",),
        ("HUMANIC",),
    ),
    "sporting_goods": (
        ("Decathlon",),
        ("INTERSPORT",),
    ),
    "electronics": (
        ("MediaMarkt", "Media Markt", "Media-Markt"),
        ("Saturn",),
        ("EURONICS",),
        ("Cyberport",),
        ("Conrad Electronic",),
        ("expert", "expert SE"),
        ("Apple Store", "Apple Retail Germany"),
        ("ARLT Computer",),
        ("notebooksbilliger.de",),
    ),
    "restaurant": (
        ("McDonald’s", "McDonalds", "McDonald s"),
        ("Burger King",),
        ("KFC",),
        ("Subway",),
        ("NORDSEE",),
        ("Vapiano",),
        ("L’Osteria", "Losteria"),
        ("Starbucks",),
        ("Coffee Fellows",),
        ("BackWerk",),
        ("Ditsch",),
        ("Kamps",),
        ("Domino’s Pizza", "Dominos Pizza"),
        ("Pizza Hut",),
    ),
    "fuel": (
        ("Aral",),
        ("Shell", "Shell Deutschland"),
        ("Esso",),
        ("JET",),
        ("TotalEnergies", "Total Tankstelle"),
        ("AVIA",),
        ("HEM",),
        ("Westfalen Tankstelle",),
        ("bft Tankstelle",),
        ("ORLEN",),
        ("star Tankstelle",),
        ("Q1", "Q1 Energie"),
        ("OIL! Tankstelle", "OIL! Tankstellen"),
        ("BayWa Tankstelle", "BayWa Tankstellen"),
    ),
    "entertainment": (
        ("CineStar",),
        ("CinemaxX",),
        ("UCI Kinowelt", "UCI Kino", "UCI Luxe"),
        ("Kinopolis",),
        ("Movie Park Germany",),
        ("Phantasialand",),
        ("Europa-Park",),
        ("Eventim", "CTS Eventim"),
    ),
    "hardware_garden": (
        ("OBI",),
        ("BAUHAUS",),
        ("HORNBACH",),
        ("toom Baumarkt", "toom"),
        ("Globus Baumarkt",),
        ("hagebaumarkt",),
        ("HELLWEG Baumarkt",),
        ("Raab Karcher",),
        ("BAUEN+LEBEN", "BAUEN und LEBEN"),
    ),
    "home_furnishings": (
        ("IKEA",),
        ("JYSK",),
        ("POCO",),
        ("XXXLutz",),
        ("Möbel Höffner",),
    ),
    "other": (
        ("Action", "Action Deutschland"),
        ("TEDi",),
        ("Woolworth",),
    ),
    "books_stationery": (
        ("Thalia",),
    ),
    "pet_supplies": (
        ("Fressnapf",),
    ),
    "postal_services": (
        ("Deutsche Post",),
        ("DHL",),
    ),
}


def _key(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold().replace("&", " und ")
    for umlaut, expanded in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        folded = folded.replace(umlaut, expanded)
    return " ".join(re.findall(r"[^\W_]+", folded))


_CATEGORIES = {names[0]: category for category, rows in _GROUPS.items() for names in rows}
_ALIASES = {_key(name): row[0] for rows in _GROUPS.values() for row in rows for name in row}

_MAX_ALIAS_WORDS = max(len(key.split()) for key in _ALIASES)

# Common words and surnames need stronger evidence than a leading token.
_EXACT_ONLY = {
    "action",
    "aldi",
    "avia",
    "bauhaus",
    "dm",
    "esprit",
    "expert",
    "hem",
    "hit",
    "jet",
    "kamps",
    "kik",
    "kiosk",
    "markant",
    "mueller",
    "new yorker",
    "nordsee",
    "norma",
    "penny",
    "saturn",
    "shell",
    "subway",
    "toom",
    "zara",
}
_LEGAL_SUFFIX = re.compile(
    r"(?:\s*,\s*|\s+)(?:(?:gmbh|ag|se|ug\s*\(haftungsbeschränkt\))\s*&\s*co\.?\s*"
    r"(?:kgaa|kg|ohg)|ug\s*\(haftungsbeschränkt\)|gmbh|ag|kg|ohg|gbr|se|e\.\s*k\.?)\s*$",
    re.IGNORECASE,
)
_REGIONAL_TOWNS = {"bonn", "siegburg", "sankt augustin"}


def _matches_town(place: str, address: str | None) -> bool:
    if not place:
        return False
    place = re.sub(r"^st ", "sankt ", place)
    if place in _REGIONAL_TOWNS:
        return True
    # Elsewhere, only accept a complete town from the printed postal address.
    towns = re.findall(r"(?:^|[,;\n]|\s)\d{5}\s+([^,;\n]+)", address or "")
    return any(place == re.sub(r"^st ", "sankt ", _key(town)) for town in towns)


def new_merchant_name(printed: str) -> str:
    """Keep independent businesses' names and locations as printed."""
    name = printed.strip()
    return _LEGAL_SUFFIX.sub("", name).rstrip() or name


def is_generic_merchant(name: str) -> bool:
    """Bare business descriptions need address evidence for a learned alias."""
    return _key(new_merchant_name(name)) in {
        "",
        "kiosk",
        "baeckerei",
        "apotheke",
        "supermarkt",
        "drogerie",
        "tankstelle",
        "restaurant",
        "cafe",
        "imbiss",
        "metzgerei",
        "buchhandlung",
        "biomarkt",
        "getraenkemarkt",
        "bakery",
        "pharmacy",
        "grocery",
        "shop",
        "store",
    }


def suggest_merchant(merchant_name: str, merchant_address: str | None = None) -> str | None:
    """Return the catalog display name when the printed header supports a match."""
    name = _key(new_merchant_name(merchant_name))
    if name in _ALIASES:
        return _ALIASES[name]
    words = name.split()
    if words[:1] == ["hit"]:
        if re.fullmatch(r"hit [0-9]{3}", name):
            return "HIT Handelsgruppe"
        legal = re.search(r"\(([^()]+)\)\s*$", merchant_name)
        if legal and _key(new_merchant_name(legal[1])) == "hit handelsgruppe":
            return "HIT Handelsgruppe"
        # HIT branch headers can contain a three-digit store number before the town.
        town = re.sub(r"^hit (?:[0-9]{3} )?", "", name)
        if _matches_town(town, merchant_address):
            return "HIT Handelsgruppe"
    if words[:2] == ["k", "kiosk"]:
        return "k kiosk"
    if "kiosk" in words:
        place = ""
        if words[0] == "kiosk":
            place = " ".join(words[1:])
        elif words[-1] == "kiosk":
            place = " ".join(words[:-1])
        if _matches_town(place, merchant_address):
            return "Kiosk"
        return None
    for end in range(min(len(words) - 1, _MAX_ALIAS_WORDS), 0, -1):
        prefix = " ".join(words[:end])
        if prefix in _ALIASES and prefix not in _EXACT_ONLY:
            return _ALIASES[prefix]
        if prefix in _ALIASES and words[end] in {"filiale", "markt", "tankstelle"}:
            return _ALIASES[prefix]
    return None


_AMBIGUOUS = {
    "globus": ("Globus Baumarkt", "Globus Markthallen"),
    "netto": ("Netto Marken-Discount", "Netto mit dem Scottie"),
    "aldi": ("ALDI", "ALDI Nord", "ALDI SÜD"),
}


def merchant_candidates(printed: str, address: str | None = None) -> list[str]:
    """Return possible identities supported by the printed brand and branch qualifiers."""
    match = suggest_merchant(printed, address)
    if match and match != "ALDI":
        return [match]
    brand, _, suffix = _key(new_merchant_name(printed)).partition(" ")
    if brand not in _AMBIGUOUS:
        return []
    # A short number/branch designator or a corroborated town can accompany the brand.
    town = re.sub(r"^(?:filiale |markt )?[0-9]{1,5}(?: |$)", "", suffix)
    if not town or _matches_town(town, address):
        return list(_AMBIGUOUS[brand])
    return []


def merchant_choices(query: str = "") -> list[str]:
    """Search canonical names and printed aliases, returning each display name once."""
    query_key = _key(query)
    return sorted({display for alias, display in _ALIASES.items() if query_key in alias}, key=_key)


def merchant_category(display_name: str) -> str | None:
    canonical = _ALIASES.get(_key(display_name))
    return _CATEGORIES.get(canonical) if canonical else None
