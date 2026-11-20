#!/usr/bin/env python3
"""Seal operator-approved PNG fixtures and model metadata for source OCR setup."""

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from ocr_contract.limits import NATIVE_SOURCE_MAX_BYTES, check_working_image_dimensions
from ocr_service.provision import (
    KINDS,
    MODEL_NAMES,
    Archive,
    Case,
    FileRecord,
    Fixtures,
    License,
    Provenance,
    Record,
    content_identity,
    extract_archive,
    file_digest,
    publish_directory,
    safe_relative,
)

ARCHIVE_NAME = "fixtures.tar"
LICENSE_NAME = "LICENSE.txt"
RECORD_LIMIT = 32 * 1024**2


class FixtureInputs(Record):
    license: License
    provenance: Provenance
    cases: dict[str, str]


def read_approved_file(path: Path, max_bytes: int) -> bytes:
    if path.resolve() != path:
        raise ValueError(f"Approved inputs require canonical paths without symlinks: {path}")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode):
            raise ValueError(f"Approved input must be a regular file: {path}")
        content = stream.read(max_bytes + 1)
    if not 0 < len(content) <= max_bytes:
        raise ValueError(f"Approved input is empty or exceeds its size bound: {path}")
    return content


def unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in items:
        if key in record:
            raise ValueError(f"Duplicate JSON key in approved input: {key}")
        record[key] = value
    return record


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(read_approved_file(path, RECORD_LIMIT), object_pairs_hook=unique_keys)
    if not isinstance(data, dict):
        raise TypeError(f"Approved input must contain a JSON object: {path}")
    return data


def require_public_https(source: str) -> None:
    parsed = urlsplit(source)
    if (
        not source.startswith("https://")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(char.isspace() or char == "\\" for char in source)
    ):
        raise ValueError("Approved asset sources require public HTTPS URLs without credentials")


def model_records(path: Path) -> dict[str, Archive]:
    models = read_json(path)
    if set(models) != set(MODEL_NAMES):
        raise ValueError("Exactly the four approved model identities are required")
    records = {name: Archive.model_validate(model) for name, model in models.items()}
    for model in records.values():
        require_public_https(model.source)
        require_public_https(model.provenance.url)
    return records


def read_png(path: Path) -> bytes:
    raw = read_approved_file(path, NATIVE_SOURCE_MAX_BYTES)
    try:
        with Image.open(io.BytesIO(raw), formats=("PNG",)) as image:
            check_working_image_dimensions(*image.size)
            image.verify()
        with Image.open(io.BytesIO(raw), formats=("PNG",)) as image:
            image.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(
            f"Approved fixture must be a valid PNG within native limits: {path}"
        ) from exc
    return raw


def add_file(archive: tarfile.TarFile, name: str, data: bytes) -> FileRecord:
    member = tarfile.TarInfo(name)
    member.mode = 0o644
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))
    return FileRecord(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def fixture_archive(specification: Path, archive_path: Path) -> Fixtures:
    spec = FixtureInputs.model_validate(read_json(specification))
    if set(spec.cases) != KINDS:
        raise ValueError("Exactly the five fixture roles are required: " + ", ".join(sorted(KINDS)))
    if not spec.license.spdx.strip() or not spec.provenance.revision.strip():
        raise ValueError("An explicit SPDX license and provenance revision are required")
    require_public_https(spec.provenance.url)
    root = specification.parent
    license_text = read_approved_file(root / safe_relative(spec.license.text_file), RECORD_LIMIT)
    if not license_text.decode("utf-8").strip():
        raise ValueError("Approved fixture license must contain nonempty UTF-8 text")
    license = spec.license.model_copy(update={"text_file": LICENSE_NAME})
    cases = []
    with tarfile.open(archive_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        files = {LICENSE_NAME: add_file(archive, LICENSE_NAME, license_text)}
        for kind in sorted(spec.cases):
            path = f"{kind}.png"
            raw = read_png(root / safe_relative(spec.cases[kind]))
            files[path] = add_file(archive, path, raw)
            cases.append(
                Case.model_validate(
                    {
                        "kind": kind,
                        "path": path,
                        "sha256": files[path].sha256,
                        "required_labels": [kind] if kind in {"table", "formula"} else [],
                        "license": license,
                        "provenance": spec.provenance,
                    }
                )
            )
    return Fixtures(
        source=ARCHIVE_NAME,
        sha256=file_digest(archive_path),
        size_bytes=archive_path.stat().st_size,
        unpacked_bytes=sum(row.size_bytes for row in files.values()),
        files=files,
        tree_sha256=content_identity({name: row.sha256 for name, row in files.items()}),
        license=license,
        provenance=spec.provenance,
        cases=cases,
    )


def runtime_pins(accelerator: str) -> tuple[str, dict[str, str]]:
    runtime_lock = json.loads((SERVICE / "runtime-lock.json").read_bytes())
    commit = runtime_lock["sources"]["paddle"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Runtime lock must pin the Paddle source commit")
    # Native build and installer recipes pin these three distributions.
    native_package = {"cuda": "paddlepaddle-gpu", "rocm": "paddlepaddle-dcu"}[accelerator]
    packages = {native_package: "3.3.1", "paddleocr": "3.7.0", "paddlex": "3.7.2"}
    requirements = (SERVICE / "requirements.runtime.lock").read_text()
    # Remaining direct roots come from the common hashed runtime lock. Source
    # setup closes over their installed METADATA once it has a candidate image.
    for name in ("numpy", "packaging", "pillow", "pillow-heif", "pydantic"):
        versions = re.findall(rf"^{name}==([\w.+!-]+)\s+\\$", requirements, re.MULTILINE)
        if len(versions) != 1:
            raise ValueError(f"Runtime lock must contain one direct pin for {name}")
        packages[name] = versions[0]
    return commit, packages


def prepare(accelerator: str, models_path: Path, fixtures_path: Path, output: Path) -> None:
    output = output.absolute()
    if output.parent.resolve(strict=True) != output.parent:
        raise ValueError("Output parent must be a canonical directory without symlinks")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}")
    models = model_records(models_path.absolute())
    commit, packages = runtime_pins(accelerator)
    with tempfile.TemporaryDirectory(prefix=".ocr-inputs-", dir=output.parent) as temporary:
        workspace = Path(temporary)
        staging = workspace / "approved"
        staging.mkdir(mode=0o700)
        archive = staging / ARCHIVE_NAME
        fixtures = fixture_archive(fixtures_path.absolute(), archive)
        extract_archive(archive, workspace / "verified", fixtures)
        manifest = {
            "schema_version": 1,
            "accelerator": accelerator,
            "gpu_index": 0,
            "expected_paddle_commit": commit,
            "expected_packages": packages,
            "models": {name: asset.model_dump(mode="json") for name, asset in models.items()},
            "fixtures": fixtures.model_dump(mode="json"),
        }
        bundle = staging / "bundle.json"
        bundle.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if bundle.stat().st_size > RECORD_LIMIT:
            raise ValueError("Prepared bundle exceeds its size bound")
        archive.chmod(0o600)
        bundle.chmod(0o600)
        publish_directory(staging, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accelerator", choices=("cuda", "rocm"))
    parser.add_argument("--models", required=True, type=Path, help="approved models.json mapping")
    parser.add_argument("--fixtures", required=True, type=Path, help="approved fixtures JSON input")
    parser.add_argument(
        "--output", required=True, type=Path, help="new directory for staged inputs"
    )
    args = parser.parse_args()
    try:
        prepare(args.accelerator, args.models, args.fixtures, args.output)
    except (OSError, ValueError, TypeError, KeyError, tarfile.TarError) as exc:
        print(f"OCR input preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared unbound OCR inputs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
