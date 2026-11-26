"""Materialize private test replies and image aliases before any inference request."""

import base64
import hashlib
import json
import shutil
import sys
import tomllib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from vision_bill.helper.document_image import document_rgb
from vision_bill.model.receipt import Receipt


def prepare(root: Path, directory: Path) -> None:
    destination = directory / "fixtures"
    shutil.copytree(root / "e2e/llm_stub/fixtures", destination)
    replies = directory / "replies"
    shutil.copytree(root / "e2e/llm_stub/replies", replies)
    # The security flow's tiny PNG is an explicit fixture, never an unknown-input fallback.
    (destination / "generic.png").write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="
        )
    )
    generic = {
        "confidence": 99,
        "merchant_name": "E2E Generic Store",
        "date": "2026-01-01",
        "currency": "EUR",
        "category": "other",
        "line_items": [
            {
                "description": "Generic Item",
                "quantity": 1,
                "unit_price": "5.00",
                "total_price": "5.00",
            }
        ],
        "subtotal": "5.00",
        "tax_total": "0",
        "total": "5.00",
        "payment_method": "cash",
    }
    (destination / "generic.json").write_text(json.dumps(generic))
    data = root / "tests/e2e/data"
    manifest = data / "fixtures.toml"
    names = tomllib.loads(manifest.read_text()).get("fixtures", []) if manifest.is_file() else []
    for name in names:
        truth = data / f"{name}.json"
        if not truth.is_file():
            continue
        for suffix in (".png", ".jpg", ".jpeg"):
            image = data / f"{name}{suffix}"
            if image.is_file():
                try:
                    Receipt.model_validate_json(truth.read_text())
                    with Image.open(image) as decoded:
                        decoded.load()
                except (ValueError, OSError, Image.DecompressionBombError) as exc:
                    if name in ("seed_receipt", "deposit_refund"):
                        raise ValueError(
                            f"mandatory fixture {name}: {type(exc).__name__}"
                        ) from None
                    print(
                        f"e2e prepare: skipping fixture {image.name!r}: {type(exc).__name__}",
                        file=sys.stderr,
                    )
                    break
                if (destination / image.name).exists():
                    raise ValueError("private fixture name collision")
                shutil.copyfile(image, destination / image.name)
                shutil.copyfile(truth, destination / truth.name)
                break
    aliases: dict[str, dict[str, str]] = {}
    registered: dict[str, str] = {}
    for image in sorted(destination.iterdir()):
        if image.suffix not in (".png", ".jpg", ".jpeg"):
            continue
        buffer = BytesIO()
        with Image.open(image) as source:
            document_rgb(ImageOps.exif_transpose(source)).save(buffer, format="PNG")
        identity = {
            "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "normalized_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        }
        for digest in identity.values():
            if digest in registered and registered[digest] != image.stem:
                raise ValueError("ambiguous fixture image alias")
            registered[digest] = image.stem
        aliases[image.stem] = identity
        if (replies / f"{image.stem}.complete.json").exists():
            continue
        observed = json.loads(image.with_suffix(".json").read_text())
        # Explicit complete test scenario; nullable partial files are separate artifacts.
        complete = {
            "currency": "EUR",
            "discount_total": "0",
            "taxes": [],
            "tip": None,
            **observed,
        }
        complete.pop("category", None)
        complete["line_items"] = [
            {key: value for key, value in line.items() if key not in {"tags", "spending_category"}}
            for line in observed["line_items"]
        ]
        complete["amount_policy"] = {
            "tax_basis": "net",
            "discount_treatment": "included_in_lines",
            "tip_treatment": "add" if complete["tip"] else "none",
            "subtotal_origin": "printed",
            "tax_information": "present" if complete["taxes"] else "explicit_zero",
        }
        (replies / f"{image.stem}.complete.json").write_text(json.dumps(complete))
        partial = {
            **complete,
            "discount_total": None,
            "tax_total": None,
            "line_items": [
                {**line, "quantity": None, "unit_price": None} if index == 0 else line
                for index, line in enumerate(complete["line_items"])
            ],
        }
        (replies / f"{image.stem}.partial.json").write_text(json.dumps(partial))
    (directory / "aliases.json").write_text(json.dumps(aliases))
