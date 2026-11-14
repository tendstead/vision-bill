#!/usr/bin/env python3
"""Standalone benchmark runner that calls vision-bill code directly.

This script runs a benchmark without requiring the Vision Bill API to be running.
It connects directly to the database, uses the paddleocr service, and calls the
LLM provider directly.

Usage:
    uv run python scripts/standalone_benchmark.py [options]

Environment variables (or command-line options):
    PG__HOST, PG__PORT, PG__USER, PG__PASSWORD, PG__DB - Database connection
    OCR__SERVICE_URL - PaddleOCR service URL (default: http://localhost:8082)
    LLM__PROVIDER - LLM provider: ollama or openai (default: ollama)
    LLM__HOST - LLM host (default: http://localhost:11434)
    LLM__MODEL_NAME - Model to use (required)
    LLM__API_KEY - API key for OpenAI provider (optional)
    LLM__TEMPERATURE - Temperature (default: 0.0)

Options:
    --limit N - Maximum number of receipts to benchmark (default: all verified)
    --category CAT - Only benchmark receipts in this category
    --model MODEL - Model to use (overrides LLM__MODEL_NAME)
    --db-host HOST - Database host (overrides PG__HOST)
    --db-port PORT - Database port (overrides PG__PORT)
    --db-user USER - Database user (overrides PG__USER)
    --db-password PASS - Database password (overrides PG__PASSWORD)
    --db-name NAME - Database name (overrides PG__DB)
    --ocr-url URL - OCR service URL (overrides OCR__SERVICE_URL)
    --llm-host HOST - LLM host (overrides LLM__HOST)
    --output FILE - Write results to JSON file
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

# Add src to path so we can import vision_bill modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_bill.config import LLMSettings, OcrSettings, PGSettings
from vision_bill.provider.factory import get_llm_provider, get_ocr_provider
from vision_bill.model.receipt import LineItem, Receipt, TaxLine
from vision_bill.service.benchmark_scoring import score_receipts

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single receipt benchmark."""

    receipt_id: str
    merchant: str
    date: str | None
    success: bool
    score: float | None = None
    scores: dict[str, float] | None = None
    error: str | None = None
    ocr_time_ms: float | None = None
    llm_time_ms: float | None = None
    total_time_ms: float | None = None
    attempts: int | None = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark run."""

    model: str
    provider: str
    total_receipts: int
    successful: int = 0
    failed: int = 0
    avg_score: float | None = None
    avg_ocr_time_ms: float | None = None
    avg_llm_time_ms: float | None = None
    avg_total_time_ms: float | None = None
    results: list[BenchmarkResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of receipts to benchmark"
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Only benchmark receipts in this category"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model to use (overrides LLM__MODEL_NAME)"
    )
    parser.add_argument(
        "--db-host", type=str, default=None, help="Database host (overrides PG__HOST)"
    )
    parser.add_argument(
        "--db-port", type=int, default=None, help="Database port (overrides PG__PORT)"
    )
    parser.add_argument(
        "--db-user", type=str, default=None, help="Database user (overrides PG__USER)"
    )
    parser.add_argument(
        "--db-password", type=str, default=None, help="Database password (overrides PG__PASSWORD)"
    )
    parser.add_argument(
        "--db-name", type=str, default=None, help="Database name (overrides PG__DB)"
    )
    parser.add_argument(
        "--ocr-url", type=str, default=None, help="OCR service URL (overrides OCR__SERVICE_URL)"
    )
    parser.add_argument("--llm-host", type=str, default=None, help="LLM host (overrides LLM__HOST)")
    parser.add_argument("--output", type=str, default=None, help="Write results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def get_pg_settings(args: argparse.Namespace) -> PGSettings:
    """Build PGSettings from environment or command-line args."""
    return PGSettings(
        user=args.db_user or os.getenv("PG__USER", "vision_bill"),
        password=args.db_password or os.getenv("PG__PASSWORD", "vision_bill"),
        db=args.db_name or os.getenv("PG__DB", "vision_bill"),
        host=args.db_host or os.getenv("PG__HOST", "localhost"),
        port=args.db_port or int(os.getenv("PG__PORT", "5432")),
    )


def get_ocr_settings(args: argparse.Namespace) -> OcrSettings:
    """Build OcrSettings from environment or command-line args."""
    return OcrSettings(
        service_url=args.ocr_url or os.getenv("OCR__SERVICE_URL", "http://localhost:8082"),
        timeout_seconds=int(os.getenv("OCR__TIMEOUT_SECONDS", "300")),
    )


def get_llm_settings(args: argparse.Namespace) -> LLMSettings:
    """Build LLMSettings from environment or command-line args."""
    from vision_bill.config import LLMProviderEnum

    provider = os.getenv("LLM__PROVIDER", "ollama")
    try:
        provider_enum = LLMProviderEnum(provider)
    except ValueError:
        provider_enum = LLMProviderEnum.OLLAMA

    return LLMSettings(
        provider=provider_enum,
        host=args.llm_host or os.getenv("LLM__HOST", "http://localhost:11434"),
        api_key=os.getenv("LLM__API_KEY", ""),
        model_name=args.model or os.getenv("LLM__MODEL_NAME", ""),
        temperature=float(os.getenv("LLM__TEMPERATURE", "0.0")),
    )


async def fetch_verified_receipts(
    pool: asyncpg.Pool,
    limit: int | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch verified receipts with their image paths."""
    query = """
        SELECT r.id, r.merchant_name, r.date, r.confidence, r.category,
               r.subtotal, r.tax_total, r.total, r.currency, r.payment_method,
               r.merchant_address, r.receipt_number, r.time,
               r.discount_total, r.tip,
               i.path as image_path,
               i.original_filename
        FROM receipts r
        JOIN images i ON r.id = i.receipt_id
        WHERE r.verified = true
    """
    params = []
    if category:
        query += " AND r.category = $1"
        params.append(category)
    query += " ORDER BY r.date DESC"
    if limit:
        query += " LIMIT $1"
        params.append(limit)

    rows = await pool.fetch(query, *params)
    return [dict(row) for row in rows]


async def fetch_receipt_details(
    pool: asyncpg.Pool,
    receipt_id: str,
) -> Receipt | None:
    """Fetch full receipt details including line items and taxes."""
    receipt_row = await pool.fetchrow(
        """
        SELECT r.id, r.merchant_name, r.date, r.confidence, r.category,
               r.subtotal, r.tax_total, r.total, r.currency, r.payment_method,
               r.merchant_address, r.receipt_number, r.time,
               r.discount_total, r.tip
        FROM receipts r
        WHERE r.id = $1
    """,
        receipt_id,
    )

    if not receipt_row:
        return None

    line_items = await pool.fetch(
        """
        SELECT description, quantity, unit_price, total_price, tags, spending_category
        FROM line_items
        WHERE receipt_id = $1
        ORDER BY id
    """,
        receipt_id,
    )

    taxes = await pool.fetch(
        """
        SELECT name, amount
        FROM tax_lines
        WHERE receipt_id = $1
        ORDER BY id
    """,
        receipt_id,
    )

    # Convert to Receipt model
    items = []
    for li in line_items:
        tags = li["tags"] if isinstance(li["tags"], list) else []
        items.append(
            LineItem(
                description=li["description"],
                quantity=li["quantity"],
                unit_price=li["unit_price"],
                total_price=li["total_price"],
                tags=tags,
                spending_category=li["spending_category"] or "unknown",
            )
        )

    tax_lines = []
    for t in taxes:
        tax_lines.append(TaxLine(name=t["name"], amount=t["amount"]))

    return Receipt(
        confidence=receipt_row["confidence"],
        merchant_name=receipt_row["merchant_name"],
        merchant_address=receipt_row["merchant_address"],
        receipt_number=receipt_row["receipt_number"],
        date=receipt_row["date"],
        time=receipt_row["time"],
        currency=receipt_row["currency"],
        category=receipt_row["category"],
        line_items=items,
        taxes=tax_lines,
        subtotal=receipt_row["subtotal"],
        discount_total=receipt_row["discount_total"],
        tax_total=receipt_row["tax_total"],
        tip=receipt_row["tip"],
        total=receipt_row["total"],
        payment_method=receipt_row["payment_method"],
    )


async def run_benchmark(args: argparse.Namespace) -> BenchmarkSummary:
    """Run the benchmark."""
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build settings
    pg_settings = get_pg_settings(args)
    ocr_settings = get_ocr_settings(args)
    llm_settings = get_llm_settings(args)

    if not llm_settings.model_name:
        logger.error("Model name required. Set LLM__MODEL_NAME or use --model")
        sys.exit(1)

    logger.info("Starting standalone benchmark")
    logger.info("Model: %s", llm_settings.model_name)
    logger.info("Provider: %s", llm_settings.provider)
    logger.info("OCR service: %s", ocr_settings.service_url)
    logger.info(
        "Database: %s@%s:%s/%s",
        pg_settings.user,
        pg_settings.host,
        pg_settings.port,
        pg_settings.db,
    )

    # Create providers
    llm_provider = get_llm_provider(llm_settings)
    ocr_provider = get_ocr_provider(ocr_settings)

    # Check connections
    logger.info("Checking LLM connection...")
    if not await llm_provider.check_connection():
        logger.error("Cannot connect to LLM provider")
        sys.exit(1)
    logger.info("LLM connection OK")

    logger.info("Checking OCR connection...")
    if not await ocr_provider.check_connection():
        logger.error("Cannot connect to OCR service")
        sys.exit(1)
    logger.info("OCR connection OK")

    # Connect to database
    logger.info("Connecting to database...")
    pool = await asyncpg.create_pool(dsn=pg_settings.pg_dsn)
    logger.info("Database connection OK")

    try:
        # Fetch verified receipts
        logger.info("Fetching verified receipts...")
        receipts = await fetch_verified_receipts(pool, args.limit, args.category)
        logger.info("Found %d verified receipts", len(receipts))

        if not receipts:
            logger.error("No verified receipts found")
            sys.exit(1)

        # Run benchmark
        summary = BenchmarkSummary(
            model=llm_settings.model_name,
            provider=str(llm_settings.provider),
            total_receipts=len(receipts),
            started_at=datetime.now().isoformat(),
        )

        for i, receipt in enumerate(receipts, 1):
            receipt_id = str(receipt["id"])
            image_path = receipt["image_path"]
            logger.info(
                "[%d/%d] Processing receipt %s (%s)",
                i,
                len(receipts),
                receipt_id,
                receipt.get("merchant_name", "unknown"),
            )

            # Check if image exists
            if not Path(image_path).exists():
                logger.warning("Image not found: %s", image_path)
                summary.results.append(
                    BenchmarkResult(
                        receipt_id=receipt_id,
                        merchant=receipt.get("merchant_name", "unknown"),
                        date=str(receipt.get("date", "")),
                        success=False,
                        error=f"Image not found: {image_path}",
                    )
                )
                summary.failed += 1
                continue

            # Get expected receipt details
            expected = await fetch_receipt_details(pool, receipt_id)
            if not expected:
                logger.warning("Could not fetch receipt details: %s", receipt_id)
                summary.results.append(
                    BenchmarkResult(
                        receipt_id=receipt_id,
                        merchant=receipt.get("merchant_name", "unknown"),
                        date=str(receipt.get("date", "")),
                        success=False,
                        error="Could not fetch receipt details",
                    )
                )
                summary.failed += 1
                continue

            # Run OCR + LLM pipeline
            try:
                start_time = time.time()

                # Stage 1: OCR
                ocr_start = time.time()
                ocr_result = await ocr_provider.extract_text(Path(image_path))
                ocr_time = (time.time() - ocr_start) * 1000

                # Stage 2: LLM
                llm_start = time.time()
                actual = await llm_provider.analyse_receipt_from_ocr_text(
                    llm_settings.model_name, ocr_result
                )
                llm_time = (time.time() - llm_start) * 1000

                total_time = (time.time() - start_time) * 1000

                # Score
                scores = score_receipts(expected, actual)

                logger.info(
                    "  Score: %.2f (header=%.2f, totals=%.2f, items=%.2f, taxes=%.2f) | "
                    "OCR: %.0fms, LLM: %.0fms, Total: %.0fms",
                    scores["overall"],
                    scores["header"],
                    scores["totals"],
                    scores["line_items"],
                    scores["taxes"],
                    ocr_time,
                    llm_time,
                    total_time,
                )

                summary.results.append(
                    BenchmarkResult(
                        receipt_id=receipt_id,
                        merchant=actual.merchant_name or receipt.get("merchant_name", "unknown"),
                        date=str(actual.date or receipt.get("date", "")),
                        success=True,
                        score=scores["overall"],
                        scores=scores,
                        ocr_time_ms=ocr_time,
                        llm_time_ms=llm_time,
                        total_time_ms=total_time,
                    )
                )
                summary.successful += 1

            except Exception as e:
                logger.error("  Error processing receipt: %s", e, exc_info=args.verbose)
                summary.results.append(
                    BenchmarkResult(
                        receipt_id=receipt_id,
                        merchant=receipt.get("merchant_name", "unknown"),
                        date=str(receipt.get("date", "")),
                        success=False,
                        error=str(e),
                    )
                )
                summary.failed += 1

        summary.finished_at = datetime.now().isoformat()

        # Calculate averages
        successful_results = [r for r in summary.results if r.success]
        if successful_results:
            scores = [r.score for r in successful_results if r.score is not None]
            if scores:
                summary.avg_score = sum(scores) / len(scores)
            ocr_times = [r.ocr_time_ms for r in successful_results if r.ocr_time_ms is not None]
            if ocr_times:
                summary.avg_ocr_time_ms = sum(ocr_times) / len(ocr_times)
            llm_times = [r.llm_time_ms for r in successful_results if r.llm_time_ms is not None]
            if llm_times:
                summary.avg_llm_time_ms = sum(llm_times) / len(llm_times)
            total_times = [
                r.total_time_ms for r in successful_results if r.total_time_ms is not None
            ]
            if total_times:
                summary.avg_total_time_ms = sum(total_times) / len(total_times)

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Model: {summary.model}")
        print(f"Provider: {summary.provider}")
        print(f"Total receipts: {summary.total_receipts}")
        print(f"Successful: {summary.successful}")
        print(f"Failed: {summary.failed}")
        if summary.avg_score is not None:
            print(f"Average score: {summary.avg_score:.4f}")
        if summary.avg_ocr_time_ms is not None:
            print(f"Average OCR time: {summary.avg_ocr_time_ms:.0f}ms")
        if summary.avg_llm_time_ms is not None:
            print(f"Average LLM time: {summary.avg_llm_time_ms:.0f}ms")
        if summary.avg_total_time_ms is not None:
            print(f"Average total time: {summary.avg_total_time_ms:.0f}ms")
        print(f"Started: {summary.started_at}")
        print(f"Finished: {summary.finished_at}")
        print("=" * 60)

        # Write output file if requested
        if args.output:
            output_data = {
                "model": summary.model,
                "provider": summary.provider,
                "total_receipts": summary.total_receipts,
                "successful": summary.successful,
                "failed": summary.failed,
                "avg_score": summary.avg_score,
                "avg_ocr_time_ms": summary.avg_ocr_time_ms,
                "avg_llm_time_ms": summary.avg_llm_time_ms,
                "avg_total_time_ms": summary.avg_total_time_ms,
                "started_at": summary.started_at,
                "finished_at": summary.finished_at,
                "results": [
                    {
                        "receipt_id": r.receipt_id,
                        "merchant": r.merchant,
                        "date": r.date,
                        "success": r.success,
                        "score": r.score,
                        "scores": r.scores,
                        "error": r.error,
                        "ocr_time_ms": r.ocr_time_ms,
                        "llm_time_ms": r.llm_time_ms,
                        "total_time_ms": r.total_time_ms,
                    }
                    for r in summary.results
                ],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {args.output}")

    finally:
        await pool.close()

    return summary


def main():
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
