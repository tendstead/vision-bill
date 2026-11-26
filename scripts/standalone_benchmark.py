"""Benchmark current qualified Gold for one application owner, without the API.

Usage:
    uv run python scripts/standalone_benchmark.py --user-id OWNER_UUID --model LOCAL_MODEL [options]

--user-id is required. Selection uses the owner's current explicitly qualified
human Gold in a repeatable-read read-only snapshot; legacy/bypass verified flags
alone are insufficient. Use SELECT-only reporting credentials and make every
ordered original available at its recorded path. No database mutation or new
qualification is performed. All selected cases, including failures, remain in
v4 metrics; historical success-only averages keep their separate meaning.

The runner calls the configured local native OCR service and text formatter.
PG__HOST, PG__PORT, PG__USER, PG__PASSWORD, PG__DB configure the database;
OCR__SERVICE_URL selects the OCR endpoint; LLM__PROVIDER, LLM__HOST,
LLM__MODEL_NAME, LLM__API_KEY and LLM__TEMPERATURE configure the formatter.
Keep endpoint/credentials local and private. No vision fallback consent is supplied.

--limit and --category filter qualified references, --model overrides the model;
--db-host/port/user/password/name, --ocr-url and --llm-host override connections.
Prefer private environment credentials over command-line passwords. --output writes
private JSON; --verbose enables diagnostics. See scripts/README.md for semantics.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from datetime import time as Time
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID

import asyncpg

# Add src to path so we can import vision_bill modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_bill.config import ImageSettings, LLMSettings, OcrSettings, PGSettings
from vision_bill.model.benchmark import GoldSnapshot
from vision_bill.model.experiment import ExperimentOptions, ExtractionPipeline
from vision_bill.model.receipt import LineItem, Receipt, TaxLine
from vision_bill.model.workflow import AccessScope
from vision_bill.provider.factory import get_llm_provider, get_ocr_provider
from vision_bill.provider.llm.base import LLMProvider
from vision_bill.provider.ocr.base import OcrProvider
from vision_bill.service.benchmark_scoring import score_receipts, validation_schema_sha256
from vision_bill.service.experiment_execution import ExperimentResult

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
    detail: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark run."""

    model: str
    provider: str
    total_receipts: int
    successful: int = 0
    failed: int = 0
    avg_score: float | None = None
    all_case_average_score: float = 0
    full_receipt_correct_rate: float = 0
    item_omissions: int = 0
    item_extras: int = 0
    manifest: dict[str, Any] = field(default_factory=dict)
    manifest_sha256: str = ""
    avg_ocr_time_ms: float | None = None
    avg_llm_time_ms: float | None = None
    avg_total_time_ms: float | None = None
    results: list[BenchmarkResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Application owner whose qualified Gold references are selected",
    )
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
    from vision_bill.config import settings
    return settings.ocr.model_copy(update={
        "enabled": True,
        "service_url": args.ocr_url or os.getenv("OCR__SERVICE_URL", "http://localhost:8082"),
        "timeout_seconds": int(os.getenv("OCR__TIMEOUT_SECONDS", "300")),
    })


def get_llm_settings(args: argparse.Namespace) -> LLMSettings:
    """Copy loaded LLM settings and validate explicit CLI overrides."""
    from vision_bill.config import settings

    overrides = {
        field: value
        for field, value in {"host": args.llm_host, "model_name": args.model}.items()
        if value is not None
    }
    return LLMSettings.model_validate(settings.llm.model_dump() | overrides)


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
               i.image_path,
               i.original_filename, i.additional_images
        FROM receipts r
        JOIN images i ON r.id = i.receipt_id
        WHERE r.verified = true
    """
    params: list[str | int] = []
    if category:
        query += " AND r.category = $1"
        params.append(category)
    query += " ORDER BY r.date DESC"
    if limit:
        query += f" LIMIT ${len(params) + 1}"
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
        ORDER BY position, id
    """,
        receipt_id,
    )

    taxes = await pool.fetch(
        """
        SELECT name, rate, amount
        FROM taxes
        WHERE receipt_id = $1
        ORDER BY position, id
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
        tax_lines.append(TaxLine(name=t["name"], rate=t["rate"], amount=t["amount"]))

    return Receipt(
        confidence=receipt_row["confidence"],
        merchant_name=receipt_row["merchant_name"],
        merchant_address=receipt_row["merchant_address"],
        receipt_number=receipt_row["receipt_number"],
        date=receipt_row["date"],
        time=receipt_row["time"].isoformat()
        if isinstance(receipt_row["time"], Time)
        else receipt_row["time"],
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


async def freeze_references(
    pool: asyncpg.Pool,
    pg_settings: PGSettings,
    scope: AccessScope,
    *,
    limit: int | None,
    category: str | None,
) -> list[GoldSnapshot]:
    """Capture qualified current truth in one read-only MVCC snapshot before inference."""
    from vision_bill.provider.db.benchmark_db import BenchmarkDB
    from vision_bill.provider.db.receipt_db import ReceiptDB
    from vision_bill.service.image_service import ImageService

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if scope.can_see_all:
        raise ValueError("standalone benchmark requires an explicit owner scope")
    db = BenchmarkDB(pool, ReceiptDB(pg_settings, original_checker=ImageService.check_originals))
    selected = []
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        if not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM users WHERE id=$1)", scope.user_id):
            raise ValueError("application user not found")
        candidates = await conn.fetch(
            """SELECT r.id FROM receipts r WHERE r.user_id=$1
            AND r.verified AND r.status='verified' AND r.verification_source='human'
            AND ($2::text IS NULL OR r.category=$2)
            AND EXISTS(SELECT 1 FROM benchmark_references b WHERE b.receipt_id=r.id AND b.owner_id=$1)
            ORDER BY r.date DESC,r.id DESC""",
            scope.user_id,
            category,
        )
        for candidate in candidates:
            try:
                preview = await db._reference_preview(conn, candidate["id"], scope, lock=False)
            except ValueError:
                continue  # Domain ineligibility only; database errors abort the entire selection.
            row = await conn.fetchrow(
                """SELECT * FROM benchmark_references WHERE receipt_id=$1
                AND review_revision=$2 AND document_sha256=$3 AND owner_id=$4""",
                candidate["id"],
                preview.review_revision,
                preview.document_sha256,
                scope.user_id,
            )
            if row is not None:
                selected.append(db._reference(row))
                if limit is not None and len(selected) == limit:
                    break
    return selected


async def execute_photos(
    provider: LLMProvider,
    ocr: OcrProvider,
    model: str,
    paths: list[Path],
    pg_settings: PGSettings,
    *,
    ocr_settings: OcrSettings | None = None,
) -> ExperimentResult:
    from vision_bill.model.experiment import ExperimentOptions, ExtractionPipeline
    from vision_bill.service.receipt_service import ReceiptService

    with TemporaryDirectory(prefix="vision-bill-benchmark-") as directory:
        service = ReceiptService(
            ImageSettings(save_dir=directory, tmp_dir=directory),
            pg_settings,
            provider,
            ocr_provider=ocr,
            ocr_settings=ocr_settings,
        )
        return await service.analyse_submission_with_metadata(
            model,
            paths,
            ExperimentOptions(pipeline=ExtractionPipeline.OCR, include_classification=False),
            [],
        )


async def execution_identity(
    provider: LLMProvider, ocr: OcrProvider, pg: PGSettings, settings: OcrSettings, model_id: str,
    *, qualify: bool = False,
) -> dict[str, Any]:
    from vision_bill.provider.llm.experiment import endpoint_fingerprint
    from vision_bill.service.experiment_execution import pipeline_identity
    from vision_bill.service.receipt_service import ReceiptService

    with TemporaryDirectory(prefix="vision-bill-benchmark-") as directory:
        service = ReceiptService(
            ImageSettings(save_dir=directory, tmp_dir=directory),
            pg,
            provider,
            ocr_provider=ocr,
            ocr_settings=settings,
        )
        options = ExperimentOptions(pipeline=ExtractionPipeline.OCR, include_classification=False)
        model = next(
            (m for m in await provider.get_available_models(capability="text") if m.id == model_id),
            None,
        )
        if model is None:
            raise ValueError("formatter_model_unavailable")
        return {
            "pipeline": await pipeline_identity(
                service, options, qualify=qualify, managed_model_ids=[model_id],
            ),
            "provider": type(provider).__name__,
            "endpoint_fingerprint": endpoint_fingerprint(provider),
            "requested_model": {"id": model.id, "digest": model.digest},
            "validation_schema_sha256": validation_schema_sha256(),
        }


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
    if ocr_provider is None or not await ocr_provider.check_connection():
        logger.error("Cannot connect to OCR service")
        sys.exit(1)
    logger.info("OCR connection OK")

    # Connect to database
    logger.info("Connecting to database...")
    pool = await asyncpg.create_pool(dsn=pg_settings.pg_dsn)
    logger.info("Database connection OK")

    try:
        from vision_bill.service.benchmark_scoring import PROMPT_VERSION
        from vision_bill.service.experiment_execution import image_digest
        from vision_bill.service.experiment_scoring import DETAIL_SCORING_VERSION, detailed_score

        scope = AccessScope(user_id=UUID(str(args.user_id)), can_see_all=False)
        references = await freeze_references(
            pool, pg_settings, scope, limit=args.limit, category=args.category
        )
        if not references:
            raise ValueError("No current qualified Gold references found")
        identity = await execution_identity(
            llm_provider, ocr_provider, pg_settings, ocr_settings, llm_settings.model_name,
            qualify=True,
        )
        manifest = {
            "version": 1,
            "scope": scope.model_dump(mode="json"),
            "identity": identity,
            "prompt_version": PROMPT_VERSION,
            "scoring_version": DETAIL_SCORING_VERSION,
            "references": [reference.model_dump(mode="json") for reference in references],
        }
        summary = BenchmarkSummary(
            model=llm_settings.model_name,
            provider=str(llm_settings.provider),
            total_receipts=len(references),
            started_at=datetime.now(UTC).isoformat(),
            manifest=manifest,
            manifest_sha256=sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        for reference in references:
            record = BenchmarkResult(
                receipt_id=str(reference.receipt_identity),
                merchant=str(reference.expected.get("merchant_name", "")),
                date=str(reference.expected.get("date", "")),
                success=False,
            )
            summary.results.append(record)
            try:
                # Every selected case, including invalid references/files, remains in the denominator.
                expected = Receipt.model_validate(reference.expected)
                paths = [Path(source["path"]) for source in reference.sources]
                hashes = [source["sha256"] for source in reference.sources]
                if (
                    not paths
                    or [await asyncio.to_thread(image_digest, path) for path in paths] != hashes
                ):
                    raise ValueError("source_identity_changed")
                if (
                    await execution_identity(
                        llm_provider,
                        ocr_provider,
                        pg_settings,
                        ocr_settings,
                        llm_settings.model_name,
                    )
                    != identity
                ):
                    raise ValueError("execution_identity_changed")
                result = await execute_photos(
                    llm_provider,
                    ocr_provider,
                    llm_settings.model_name,
                    paths,
                    pg_settings,
                    ocr_settings=ocr_settings,
                )
                record.metadata = result.metadata
                record.attempts = result.attempts
                if result.metadata.get("ordered_source_sha256") != hashes:
                    raise ValueError("executed_source_identity_changed")
                if any(
                    result.metadata.get(key) != identity["requested_model"].get(target)
                    for key, target in (("model_id", "id"), ("model_digest", "digest"))
                ):
                    raise ValueError("executed_model_identity_changed")
                runtime = identity["pipeline"].get("ocr_runtime") or {}
                if result.metadata.get("used_pipeline") != "ocr" or any(
                    result.metadata.get(key) != runtime.get(key)
                    for key in ("profile_id", "runtime_fingerprint")
                ):
                    raise ValueError("executed_ocr_identity_changed")
                record.scores = score_receipts(expected, result.receipt)
                record.score = record.scores["overall"]
                record.detail = detailed_score(
                    reference.expected,
                    result.receipt,
                    classification=False,
                    scored_fields=reference.scored_fields,
                )
                record.ocr_time_ms = (
                    result.metadata.get("telemetry", {}).get("stage_ms", {}).get("ocr", 0)
                )
                record.llm_time_ms = result.metadata.get("llm_ms", 0)
                record.total_time_ms = result.elapsed_ms
                record.success = True
                summary.successful += 1
            except Exception as exc:  # noqa: BLE001 - each selected case gets a report
                logger.error("Receipt processing failed: %s", type(exc).__name__)
                record.error = type(exc).__name__
                record.metadata = getattr(exc, "metadata", record.metadata)
                record.attempts = len(record.metadata.get("calls", []))
                try:
                    record.detail = detailed_score(
                        reference.expected,
                        None,
                        classification=False,
                        scored_fields=reference.scored_fields,
                    )
                except (ValueError, TypeError, KeyError):
                    record.detail = None  # Corrupt reference still contributes zero, never a fabricated Receipt.
                summary.failed += 1
        summary.all_case_average_score = (
            sum(r.detail["scores"]["normalized"] if r.detail else 0 for r in summary.results)
            / summary.total_receipts
        )
        summary.full_receipt_correct_rate = (
            sum(bool(r.detail and r.detail["full_receipt_correct"]) for r in summary.results)
            / summary.total_receipts
        )
        summary.item_omissions = sum(
            r.detail["item_fn"] if r.detail else 0 for r in summary.results
        )
        summary.item_extras = sum(r.detail["item_fp"] if r.detail else 0 for r in summary.results)

        summary.finished_at = datetime.now(UTC).isoformat()

        # Calculate averages
        successful_results = [r for r in summary.results if r.success]
        if successful_results:
            successful_scores = [r.score for r in successful_results if r.score is not None]
            if successful_scores:
                summary.avg_score = sum(successful_scores) / len(successful_scores)
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
        print(f"All-case normalized score: {summary.all_case_average_score:.4f}")
        print(f"Full-receipt correct rate: {summary.full_receipt_correct_rate:.4f}")
        print(f"Item omissions: {summary.item_omissions}; extras: {summary.item_extras}")
        print(f"Manifest SHA-256: {summary.manifest_sha256}")
        if summary.avg_score is not None:
            print(f"Historical average score (successful cases): {summary.avg_score:.4f}")
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
            output_data = asdict(summary)
            await asyncio.to_thread(Path(args.output).write_text, json.dumps(output_data, indent=2))
            print(f"Results written to {args.output}")

    finally:
        await pool.close()

    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
