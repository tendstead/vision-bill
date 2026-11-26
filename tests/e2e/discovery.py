"""Opt-in e2e fixture discovery.

Fixtures live in tests/e2e/data (gitignored). A pair is tested when its
basename is listed in tests/e2e/data/fixtures.toml AND the image plus a
same-basename .json that validates as a Receipt exist. Broken or missing
entries are skipped with a warning, never fatal.
"""

import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vision_bill.model.receipt import Receipt

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class FixtureCase:
    name: str
    image_path: Path
    expected: dict
    expected_receipt: Receipt


def _warn(name: str, reason: str) -> None:
    print(f"e2e discovery: skipping fixture {name!r}: {reason}", file=sys.stderr)


def _read_manifest(data_dir: Path) -> list[str]:
    manifest = data_dir / "fixtures.toml"
    if not manifest.is_file():
        return []
    with manifest.open("rb") as fh:
        data = tomllib.load(fh)
    fixtures = data.get("fixtures", [])
    return [str(name) for name in fixtures]


def _find_image(data_dir: Path, name: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = data_dir / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_fixture_cases(data_dir: Path = DEFAULT_DATA_DIR) -> list[FixtureCase]:
    cases: list[FixtureCase] = []
    for name in _read_manifest(data_dir):
        image_path = _find_image(data_dir, name)
        if image_path is None:
            _warn(name, f"no image found for name {name!r}")
            continue
        json_path = data_dir / f"{name}.json"
        if not json_path.is_file():
            _warn(name, "missing ground-truth JSON")
            continue
        try:
            expected = json.loads(json_path.read_text(encoding="utf-8"))
            expected_receipt = Receipt.model_validate(expected)
            with Image.open(image_path) as image:
                image.load()
        except Exception as exc:  # noqa: BLE001 - discovery must never crash the suite
            _warn(name, f"invalid fixture: {type(exc).__name__}")
            continue
        cases.append(
            FixtureCase(
                name=name,
                image_path=image_path,
                expected=expected,
                expected_receipt=expected_receipt,
            )
        )
    return cases


def first_case(data_dir: Path = DEFAULT_DATA_DIR) -> FixtureCase | None:
    cases = load_fixture_cases(data_dir)
    return cases[0] if cases else None
