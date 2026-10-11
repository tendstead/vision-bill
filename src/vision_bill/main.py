import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .api.auth import router as auth_router
from .api.benchmarks import router as benchmark_router
from .api.collections import router as collections_router
from .api.images import router as image_router
from .api.receipts import router as receipt_router
from .api.search import router as search_router
from .api.statistics import router as statistics_router
from .api.system.llm import router as llm_router
from .api.system.main import router as system_router
from .api.tags import router as tags_router
from .config import Settings, mark_startup_settings, settings
from .helper.logging_config import setup_logging
from .provider.db.collection_db import CollectionDB
from .provider.db.user_db import UserDB
from .provider.factory import get_llm_provider, get_ocr_provider
from .security.password import hash_password
from .service.analysis_scheduler import AnalysisScheduler
from .service.benchmark_service import BenchmarkService
from .service.collection_service import CollectionService
from .service.image_service import ImageService
from .service.receipt_service import ReceiptService

setup_logging()

logger = logging.getLogger(__name__)


async def bootstrap_admin(user_db: UserDB, settings: Settings) -> None:
    """Create the first admin from env on an empty DB, then backfill orphans.

    Idempotent: returns immediately once any user exists, so it is safe to run
    on every startup. The admin is created from ``auth.bootstrap_username`` /
    ``auth.bootstrap_password``; legacy receipts/images with no owner are
    assigned to that admin.
    """
    if await user_db.count_users() > 0:
        return
    username = settings.auth.bootstrap_username
    password = settings.auth.bootstrap_password
    if not username or not password:
        return
    try:
        admin = await user_db.create_user(
            username, hash_password(password, settings.auth), is_admin=True
        )
    except asyncpg.UniqueViolationError:
        logger.info("Bootstrap user '%s' already exists - skipping", username)
        return
    await user_db.set_owner_of_orphan_rows(admin.id)
    logger.info("Bootstrapped first admin user '%s' (id=%s)", username, admin.id)


# ── Startup / shutdown handlers ──────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    mark_startup_settings(settings)
    provider = get_llm_provider(settings.llm)

    # Create OCR provider if configured
    ocr_provider = None
    try:
        ocr_provider = get_ocr_provider(settings.ocr)
        logger.info("OCR provider initialized: %s", settings.ocr.provider)
    except ValueError as exc:
        logger.warning("Failed to initialize OCR provider: %s", exc)

    receipt_service = ReceiptService(
        settings.images, settings.pg, provider, ocr_provider=ocr_provider
    )
    try:
        await receipt_service.init_db()
    except Exception:
        logger.exception("Failed to initialise database - continuing without it")

    if receipt_service.db_ready:
        try:
            await bootstrap_admin(receipt_service.user_db, settings)
        except Exception:
            logger.exception("Failed to run authentication bootstrap")

    image_service = ImageService(settings.images)
    scheduler = AnalysisScheduler(
        settings=settings,
        provider=provider,
        receipt_service=receipt_service,
        image_db=receipt_service.image_db,
        image_service=image_service,
    )
    benchmark_service = (
        BenchmarkService(provider, receipt_service) if receipt_service.db_ready else None
    )
    collection_service = CollectionService(receipt_service) if receipt_service.db_ready else None
    if collection_service is not None:
        receipt_service.set_collection_db(CollectionDB(receipt_service.pool))

    app.state.receipt_service = receipt_service
    app.state.image_service = image_service
    app.state.llm_provider = provider
    app.state.user_db = receipt_service.user_db
    app.state.analysis_scheduler = scheduler
    app.state.benchmark_service = benchmark_service
    app.state.collection_service = collection_service

    await scheduler.start()
    if benchmark_service is not None:
        await benchmark_service.start()

    yield

    await scheduler.stop()
    if benchmark_service is not None:
        await benchmark_service.stop()

    try:
        await app.state.receipt_service.destroy_db()
    except Exception:
        logger.exception("Error while closing database pool")


app = FastAPI(title="Receipt Tracker API", lifespan=lifespan)

# JSON API routes
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(receipt_router, prefix="/api/v1/receipts", tags=["Receipts"])
app.include_router(image_router, prefix="/api/v1/images", tags=["Images"])
app.include_router(search_router, prefix="/api/v1/search", tags=["Search"])
app.include_router(statistics_router, prefix="/api/v1/statistics", tags=["Statistics"])
app.include_router(benchmark_router, prefix="/api/v1/benchmarks", tags=["Benchmarks"])
app.include_router(tags_router, prefix="/api/v1/tags", tags=["Tags"])
app.include_router(collections_router, prefix="/api/v1/collections", tags=["Collections"])
# System routes
app.include_router(system_router, prefix="/api/v1/system", tags=["System"])
# LLM routes
app.include_router(llm_router, prefix="/api/v1/llm", tags=["LLM"])
# ── Static frontend / SPA fallback ───────────────────────────────────
# Serves the built SPA from ``src/vision_bill/static`` with an index.html
# fallback for client-side routes. Registered after the API routers so
# /api routes keep their own behaviour; a no-op while no build exists.

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _register_spa_routes() -> None:
    if not (_STATIC_DIR / "index.html").is_file():
        logger.info("No static build in %s - SPA fallback not registered", _STATIC_DIR)
        return

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = (_STATIC_DIR / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(_STATIC_DIR):
            return FileResponse(candidate)
        return FileResponse(_STATIC_DIR / "index.html")


_register_spa_routes()
