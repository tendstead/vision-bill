#!/usr/bin/env python3
"""Create thumbnails for receipt images that don't have one yet.

Configuration comes from a ``.env`` file in the same directory as this script
(``scripts/.env``). If that file is absent, the application's normal config
sources (CWD ``.env`` / YAML) are used instead. The values that matter here are
the database connection and the image directories, e.g.:

    PG__HOST=localhost
    PG__PORT=5432
    PG__USER=vision_bill
    PG__PASSWORD=...
    PG__DB=vision_bill
    IMAGES__SAVE_DIR=/path/to/uploads
    IMAGES__TMP_DIR=/path/to/uploads_tmp
    # optional (defaults 512 / 80)
    IMAGES__THUMBNAIL_MAX_EDGE=512
    IMAGES__THUMBNAIL_QUALITY=80

Examples:
    uv run python scripts/generate_thumbnails.py --dry-run
    uv run python scripts/generate_thumbnails.py
    uv run python scripts/generate_thumbnails.py --limit 20

Idempotent: only touches rows where thumbnail_path IS NULL.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make `vision_bill` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

# Load the script-local .env (same directory as this file) BEFORE the app config
# is built, so its values take precedence over the CWD .env / YAML.
_SCRIPT_ENV = Path(__file__).resolve().parent / ".env"
if _SCRIPT_ENV.is_file():
    load_dotenv(_SCRIPT_ENV, override=True)

# The app config requires LLM/auth/PG fields at import time even though this
# script only needs PG + images. Fill any missing required keys with safe
# dummies (never used here) so `Settings()` always builds.
os.environ.setdefault("AUTH__SECRET_KEY", "backfill-not-used")
os.environ.setdefault("LLM__HOST", "localhost")
os.environ.setdefault("LLM__API_KEY", "none")
os.environ.setdefault("LLM__MODEL_NAME", "none")
os.environ.setdefault("LLM__TEMPERATURE", "0")
os.environ.setdefault("PG__USER", "postgres")
os.environ.setdefault("PG__PASSWORD", "postgres")

from vision_bill.config import Settings  # noqa: E402
from vision_bill.service.image_service import ImageService  # noqa: E402

_SELECT = (
    "SELECT id, image_path FROM images WHERE thumbnail_path IS NULL AND image_path IS NOT NULL"
)
_UPDATE = "UPDATE images SET thumbnail_path = $2 WHERE id = $1"


def _resolve(stored_path: str, extra_dirs: list[str]) -> Path | None:
    """Resolve a stored image_path (possibly container-absolute) to a local file.

    Stored paths may be container-absolute (e.g. /app/uploads/x.png); also probe
    the bare filename under the configured image directories.
    """
    stored = (stored_path or "").strip()
    if not stored:
        return None
    candidates = [stored, *[str(Path(d) / Path(stored).name) for d in extra_dirs if d]]
    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return p
    return None


async def _run(args: argparse.Namespace) -> int:
    import asyncpg

    settings = Settings()  # type: ignore[call-arg]
    image_service = None if args.dry_run else ImageService(settings.images)
    extra_dirs = [settings.images.save_dir, settings.images.tmp_dir]

    conn = await asyncpg.connect(dsn=settings.pg.pg_dsn)
    try:
        query = _SELECT
        if args.limit:
            query += f" LIMIT {int(args.limit)}"
        rows = await conn.fetch(query)

        processed = skipped = failed = 0
        for row in rows:
            path = _resolve(row["image_path"], extra_dirs)
            if path is None:
                skipped += 1
                continue
            if args.dry_run:
                processed += 1
                continue
            assert image_service is not None
            thumb = image_service.generate_thumbnail(path)
            if thumb is None:
                failed += 1
                continue
            if not args.dry_run:
                await conn.execute(_UPDATE, row["id"], str(thumb))
            processed += 1

        verb = "would process" if args.dry_run else "processed"
        detail = "decoding not attempted (dry-run)." if args.dry_run else f"{failed} failed to decode."
        print(f"{verb} {processed} image(s); {skipped} had no resolvable file; {detail}")
        return 0
    finally:
        await conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="report, do not write")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of rows processed")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
