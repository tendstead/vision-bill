"""Prepare sealed records and optional archives for the four official Paddle models."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from ocr_service import paddle_snapshots
from ocr_service.provision import (
    Archive,
    FileRecord,
    content_identity,
    file_digest,
    publish_directory,
)


def prepare_model(model: paddle_snapshots.Model, tree: Path, archive: Path, source: str) -> Archive:
    paddle_snapshots.download_model(model, tree)
    paddle_snapshots.write_model_archive(tree, archive)
    files = {
        file.name: FileRecord(sha256=file_digest(file), size_bytes=file.stat().st_size)
        for file in sorted(tree.iterdir())
    }
    return Archive.model_validate(
        {
            "source": source,
            "sha256": file_digest(archive),
            "size_bytes": archive.stat().st_size,
            "unpacked_bytes": sum(row.size_bytes for row in files.values()),
            "files": files,
            "tree_sha256": content_identity({name: row.sha256 for name, row in files.items()}),
            "license": {
                "spdx": "Apache-2.0",
                "text_file": "LICENSE",
                "redistribution_allowed": True,
            },
            "provenance": {"url": model.url, "revision": model.revision},
        }
    )


def prepare(
    output: Path, archive_base_url: str | None = None, *, official_snapshots: bool = False
) -> None:
    if official_snapshots == (archive_base_url is not None):
        raise ValueError("Choose official snapshots or a hosted archive base URL")
    if archive_base_url is not None:
        base = urlsplit(archive_base_url)
        if (
            base.scheme != "https"
            or not base.hostname
            or base.username is not None
            or base.query
            or base.fragment
            or "?" in archive_base_url
            or "#" in archive_base_url
            or "\\" in archive_base_url
            or any(character.isspace() for character in archive_base_url)
        ):
            raise ValueError(
                "Provide an HTTPS archive base URL with no credentials, query or fragment"
            )
        archive_base_url = archive_base_url.rstrip("/")
    if os.path.lexists(output):
        raise FileExistsError(f"Output directory already exists: {output}")

    with TemporaryDirectory(prefix=".paddle-models-", dir=output.parent) as scratch:
        staged = Path(scratch) / "output"
        staged.mkdir()
        records = {}
        for name, model in paddle_snapshots.MODELS.items():
            with TemporaryDirectory(dir=scratch) as tree:
                filename = f"{name}.tar"
                if archive_base_url is None:
                    archive = Path(scratch) / filename
                    record = prepare_model(model, Path(tree), archive, model.source)
                    archive.unlink()
                else:
                    record = prepare_model(
                        model, Path(tree), staged / filename, f"{archive_base_url}/{filename}"
                    )
                records[name] = record.model_dump(mode="json")
        (staged / "models.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        publish_directory(staged, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new local output directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--archive-base-url",
        help="approved HTTPS destination for the generated archives (no upload is performed)",
    )
    source.add_argument(
        "--official-snapshots",
        action="store_true",
        help="record official file sources and canonical archive identities in models.json",
    )
    args = parser.parse_args()
    prepare(args.output, args.archive_base_url, official_snapshots=args.official_snapshots)
    print(f"Prepared {args.output / 'models.json'}.")


if __name__ == "__main__":
    main()
