#!/usr/bin/env python3
"""Send local receipt images to the PaddleOCR service and print its raw output.

This script exercises ONLY the OCR service (paddleocr-service). It does not
touch the LLM provider at all. For every image in the data directory it POSTs
the file to the service's /ocr endpoint (via the same PaddleOcrProvider the
application uses) and prints the raw JSON the service returns.

Usage:
    uv run python scripts/test_ocr_service.py [options]
    make script-test-ocr ARGS="--ocr-url http://localhost:8082"

Environment variables (or command-line options):
    OCR__SERVICE_URL   - PaddleOCR service URL (default: http://localhost:8082)
    OCR__TIMEOUT_SECONDS - Request timeout in seconds (default: 300)

Options:
    --ocr-url URL  - PaddleOCR service URL (overrides OCR__SERVICE_URL)
    --timeout N    - Request timeout in seconds (overrides OCR__TIMEOUT_SECONDS)
    --data-dir DIR - Directory with test images (default: tests/data)
    --output FILE  - Write all raw results to a JSON file
    --quiet        - Only print per-image status and a summary, not the raw JSON
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path so we can import vision_bill modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_bill.config import OcrSettings
from vision_bill.provider.factory import get_ocr_provider

# Image suffixes the PaddleOCR service accepts (jpeg/png/webp/bmp). We only
# scan for the formats present in tests/data by default.
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ocr-url",
        type=str,
        default=None,
        help="PaddleOCR service URL (overrides OCR__SERVICE_URL)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Request timeout in seconds (overrides OCR__TIMEOUT_SECONDS)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with test images (default: tests/data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write all raw results to a JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print per-image status and a summary, not the raw JSON",
    )
    return parser.parse_args()


def get_ocr_settings(args: argparse.Namespace) -> OcrSettings:
    """Build OcrSettings from environment or command-line args."""
    return OcrSettings(
        service_url=args.ocr_url or os.getenv("OCR__SERVICE_URL", "http://localhost:8082"),
        timeout_seconds=args.timeout or int(os.getenv("OCR__TIMEOUT_SECONDS", "300")),
    )


def find_images(data_dir: Path) -> list[Path]:
    """Find all supported image files in data_dir, sorted by name."""
    images: set[Path] = set()
    for suffix in IMAGE_SUFFIXES:
        images.update(data_dir.glob(f"*{suffix}"))
    return sorted(images, key=lambda p: p.name)


async def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    settings = get_ocr_settings(args)
    provider = get_ocr_provider(settings)

    data_dir = (
        Path(args.data_dir) if args.data_dir else Path(__file__).parent.parent / "tests" / "data"
    )
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return [], 2

    images = find_images(data_dir)
    if not images:
        print(
            f"No images found in {data_dir} (looked for {', '.join(IMAGE_SUFFIXES)})",
            file=sys.stderr,
        )
        return [], 2

    print(f"OCR service : {settings.service_url}")
    print(f"Timeout     : {settings.timeout_seconds}s")
    print(f"Data dir    : {data_dir}")
    print(f"Images      : {len(images)}")
    print()

    # Check the service is reachable before sending any images.
    print("Checking OCR service connection...")
    if not await provider.check_connection():
        print(f"Cannot connect to OCR service at {settings.service_url}", file=sys.stderr)
        return [], 1
    print("Connection OK")
    print()

    results: list[dict[str, Any]] = []
    ok = 0
    failed = 0

    for i, image in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {image.name}")
        start = time.time()
        try:
            raw = await provider.extract_text(image)
            elapsed_ms = (time.time() - start) * 1000
            ok += 1
            if args.quiet:
                print(f"  OK in {elapsed_ms:.0f}ms")
            else:
                print(json.dumps(raw, indent=2, ensure_ascii=False))
            results.append(
                {
                    "image": image.name,
                    "ok": True,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "raw": raw,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report any per-image failure and continue
            elapsed_ms = (time.time() - start) * 1000
            failed += 1
            print(f"  ERROR after {elapsed_ms:.0f}ms: {exc}", file=sys.stderr)
            results.append(
                {
                    "image": image.name,
                    "ok": False,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "error": str(exc),
                }
            )
        print()

    print("=" * 60)
    print(f"OCR service test: {ok} ok, {failed} failed, {len(images)} total")
    print("=" * 60)

    return results, 0 if failed == 0 else 1


def main() -> int:
    args = parse_args()
    results, exit_code = asyncio.run(run(args))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
