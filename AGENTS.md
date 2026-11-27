# Vision-Bill Agent Guide

This document provides technical context for agents working on the vision-bill project.

## Project Overview
**vision-bill** is a privacy-first receipt/invoice tracking API with a bundled web frontend. It uses local multi-modal LLMs (via Ollama by default) to extract structured data from receipt images, so sensitive PII does not have to be sent to external cloud services. The system is multi-user (session-cookie authentication, per-user data ownership, admin privileges), runs a background analysis worker for queued uploads, and includes a benchmark subsystem for comparing extraction quality across models.

## Technology Stack
- **Language**: Python 3.12+
- **Web Framework**: FastAPI, Uvicorn
- **Frontend**: SvelteKit SPA (TypeScript, Tailwind CSS, Vitest unit tests, Playwright E2E) in `web/`
- **LLM Engines**:
    - Ollama (default, local inference, `ollama` Python SDK)
    - OpenAI-compatible providers (optional, `openai` Python SDK; host and API key are user-configured — see Privacy constraint)
- **Database**: PostgreSQL (`asyncpg` pool, raw SQL — no ORM in use)
- **Migrations**: Alembic (hand-written raw SQL)
- **Auth**: `argon2-cffi` (Argon2id password hashing, optional pepper), HMAC-signed stateless session cookie
- **Core Libraries**:
    - `pydantic` / `pydantic-settings`: Data validation and settings management (env, dotenv, and YAML sources).
    - `httpx`: Asynchronous HTTP client.
    - `python-magic`: File type identification (requires system `libmagic`).
    - `python-multipart`: Multipart form uploads.
    - `PyYAML`: Persistent application config file.
- **Package Managers**: `uv` (lockfile: `uv.lock`) for Python; npm (lockfile: `web/package-lock.json`) for the frontend.

## Repository Layout
- `src/vision_bill/` — Application package (API, services, providers, models, security, config).
- `src/vision_bill/static/` — Built SPA (gitignored except `.gitkeep`); produced by `make fe-sync` or baked in at Docker image build time.
- `web/` — SvelteKit SPA source (routes: login, receipts, queue, search, statistics, benchmarks, settings, upload; i18n in `en`/`de`).
- `alembic/` — Migration tooling; versions `0001`–`0003` in `alembic/versions/`.
- `tests/` — Pytest suite (`asyncio_mode = "auto"`). `tests/data/` (gitignored) holds local ground-truth fixtures: receipt images + expected JSON pairs.
- `scripts/` — Operational scripts; see `scripts/README.md` for setup (`.env.scripts`) and usage.
- `docs/` — Design and deployment documentation (gitignored; local-only).
- `image-sticher/` — Standalone client-side image-stitching HTML tool, served via nginx on port 8081 (`image-sticher/nginx_site.conf`, `docker-compose.image-stichter.yml`).
- `server_data/` — Runtime data mounted by Docker Compose (`logs/`, `uploads/`, `config/`); contents are gitignored.
- `Makefile` — Canonical dev workflow targets (see Development Workflow).
- `Dockerfile` / `docker-compose.yml` — Multi-stage containerized app + PostgreSQL.
- `.env.example` / `.env.scripts.example` — Environment templates (real `.env` / `.env.scripts` are gitignored).
- `dist/` — Python build artifacts (gitignored).

## Architecture & Key Modules
All application code lives under `src/vision_bill/`. Entry point: `main.py`.

### 1. API Layer (`api/`)
FastAPI routers, all mounted under `/api/v1/...` in `main.py`. **Every route requires an authenticated user** (`get_current_user`) except `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, and `GET /system/ui-config`. Routes backed by the database return `503` when the pool is unavailable. All entity IDs and path parameters are UUIDs; IDs are never sequential integers.

- `api/auth.py` — mounted at `/api/v1/auth` (public):
    - `POST /register` — creates a non-admin account and sets the session cookie; `201` on success, `403` when registration is disabled, `409` on duplicate username.
    - `POST /login`, `POST /logout`, `GET /me` — session management and current-user inspection.
- `api/images.py` — mounted at `/api/v1/images`:
    - `POST /?model_id=...&bypass_review=...` — multipart upload. Returns `201` + `Location` when analysed synchronously (body includes `receipt_id`), `202` + `Location` when queued (no reachable vision model, or the background worker claimed the image first), `415` for unsupported image types, `503` without DB. When `bypass_review` is effective (explicit query param, else `images.bypass_review_default`), the receipt is persisted as `verified` and the image is moved to permanent storage immediately.
    - `GET /` — lists the caller's images (newest first) with `?status=pending,failed`, `?limit`, `?offset`.
    - `POST /analyze` — manually trigger one pending-image analysis cycle.
    - `GET /{id}`, `GET /{id}/file` (streams the stored image bytes), `DELETE /{id}`.
- `api/receipts.py` — mounted at `/api/v1/receipts`:
    - `GET /`, `GET /{id}`, `PUT /{id}`, `DELETE /{id}` — list, inspect, update, delete persisted receipts.
    - `POST /{id}/verify` — marks a receipt verified and moves its image from tmp to permanent storage.
- `api/search.py` — mounted at `/api/v1/search`:
    - `GET /?query=...` — case-insensitive product search over verified line-item descriptions with price history.
- `api/statistics.py` — mounted at `/api/v1/statistics`:
    - `GET /?weeks=1..52` — verified-only spending aggregates (per currency, merchant, category, payment method, weekday, and weekly).
- `api/benchmarks.py` — mounted at `/api/v1/benchmarks` (**admin-only**, `require_admin`):
    - `POST /` — create a durable benchmark run; `202` + `Location`.
    - `GET /`, `GET /{run_id}` — list runs / run status with per-model summaries.
    - `POST /{run_id}/reevaluate` — transiently evaluate one receipt (body: `receipt_id`, `model_id`); nothing is persisted.
- `api/tags.py` — mounted at `/api/v1/tags`:
    - `GET /` — the line-item tag vocabulary (source of truth for the UI tag input).
    - `POST /` — create a tag idempotently (normalized lowercase name); `201` when created, `200` when it already existed.
- `api/system/main.py` — mounted at `/api/v1/system`:
    - `GET /ui-config` — public, pre-auth runtime flags for the UI (`bypass_review_default`, `registration_open`).
    - `GET /settings`, `PUT /settings` — admin settings page backend. Exposes LLM provider/host/model/temperature and `allow_registration` with per-key `sources` (`environment` / `config` / `default`) and a `restart_required` flag. `409` when changing a key that is controlled by an environment variable; temperature/model changes apply immediately (provider `update_runtime_settings`), provider/host changes need a restart. Never exposes credentials or DB settings.
- `api/system/llm.py` — mounted at `/api/v1/llm`: `GET /models` lists vision-capable models (id + parameter size) from the active provider.
- `api/helper/helper.py` — FastAPI dependencies (`get_receipt_service`, `get_user_db`, `get_image_service`, `get_analysis_scheduler`, `get_benchmark_service`) backed by `app.state`.

### 2. Security Layer (`security/`)
- `password.py` — Argon2id hashing; an optional `auth.pepper` salts passwords first (falls back to `secret_key`).
- `session.py` — HMAC-signed stateless session tokens (no server-side session store).
- `models.py` — `User` row model (`id`, `username`, `is_admin`, derived `can_see_all`).
- `dependencies.py` — `get_current_user` (resolves the `vb_session` cookie, loads the user, derives `can_see_all = is_admin and auth.admin_can_see_all`; `401` on any failure, `503` when the user store is down) and `require_admin` (`403` for non-admins).
- All receipts/images/queries are scoped to the caller; admins only see other users' data when `auth.admin_can_see_all` is true.

### 3. Service Layer (`service/`)
Core business logic:
- `receipt_service.py`: Orchestrates upload → LLM extraction → persistence. Delegates all DB access to `ReceiptDB`/`ImageDB`/`UserDB`. Also supports `extract_receipt_all_models` (concurrent extraction across every available model via `asyncio.gather`), product search, statistics, and the tag vocabulary.
- `image_service.py`: Image validation/inspection via a singleton `MagicService` (python-magic); manages temporary (`store_tmp_image`) and permanent (`store_perm_image`) image storage.
- `analysis_scheduler.py`: `AnalysisScheduler` — background worker started from the app lifespan. Drains the `pending` image queue with LLM extraction every `worker.check_interval_seconds` (default 300 s) and can be triggered on demand via `POST /images/analyze`. Both paths are guarded by an `asyncio.Lock` so cycles never overlap. There is deliberately no terminal "unreachable" state: if the provider is down, images stay `pending` and are retried next cycle.
- `benchmark_service.py` + `benchmark_scoring.py`: Durable benchmark runs. A run queues (model × receipt) tasks in PostgreSQL; a background worker leases, executes, and finishes tasks with retries. Ground truth is the user's **verified receipts** — benchmark data must be verified. Scoring (`score_receipts`) compares extracted vs. expected receipts per field; runs record a dataset fingerprint, prompt/scoring versions, and model digests (a task waits for retry when the model digest changed). Supports council policies (`all`/`material`/`custom`) for multi-model disagreement handling.

### 4. Provider Layer (`provider/`)
Abstraction over external dependencies:
- `provider/llm/base.py` — `LLMProvider` ABC: `get_available_models` (returns `ModelInfo` with id + parameter size), `analyse_receipt_from_model`, `analyse_receipt_with_metadata` (adds attempts + latency), `send_message`, `check_connection`, `update_runtime_settings(temperature)`. Shared `build_prompt(tags)` (embeds `Receipt.model_json_schema()` and the tag vocabulary) and `parse_llm_response`.
- `provider/llm/ollama.py` — `OllamaProvider`:
    - **Model Discovery**: Lists local Ollama models and filters to those with the `vision` capability.
    - **Self-Correction**: Retry loop (`RETRY_LIMIT = 3`) that appends the failed output + validation error back to the conversation so the model can emit corrected JSON.
- `provider/llm/openai.py` — `OpenAIProvider` (optional, user-configured host + API key):
    - **Model Discovery**: Lists models and filters to vision-capable ones; includes parameter-size metadata and model digests.
    - **Chat**: `send_message` with image content and a capped `reasoning_effort`.
    - **Self-Correction**: Same retry/repair loop as the Ollama provider.
- `provider/factory.py` — `get_llm_provider(settings.llm)`. `OLLAMA` and `OPENAI` are implemented; `ANTHROPIC` is declared in the enum but raises `ValueError`.
- `provider/db/receipt_db.py` — `ReceiptDB`: asyncpg pool, raw SQL DML for receipts, line items, taxes, and tags (`persist_receipt`, `get_receipt_with_details`, `update_receipt`, `verify_receipt`, `list_tags`, `create_tag`, ...). All row queries are user-scoped. DDL lives only in Alembic migrations.
- `provider/db/image_db.py` — persists image metadata and workflow state (`pending` → `processing` → `analyzed`/`failed`); image bytes remain in file storage managed by `ImageService`.
- `provider/db/user_db.py` — `UserDB`: users table, session user lookup, ownership backfill for legacy orphan rows.
- `provider/db/benchmark_db.py` — `BenchmarkDB`: benchmark runs, the durable task queue (`lease_task` / `retry_task` / `finish_task`), and per-model summary stats.

### 5. Model Layer (`model/`)
- `model/receipt.py` — Pydantic `Receipt` schema (incl. `confidence` 0–100, `category` on the receipt, line items, taxes, totals) used for LLM output enforcement. `LineItem` carries `tags: list[str]` (whitespace-normalized, case-insensitively deduped, max 100 chars per tag); per-line-item `category` was removed in migration 0003.
- `model/image.py` — Pydantic models for image info.
- `model/benchmark.py` — `BenchmarkCreate`, `BenchmarkRun`, `BenchmarkStatus`, `BenchmarkSummary`, `CouncilPolicy`.
- `model/search.py`, `model/statistics.py` — Search and statistics response models.
- `model/db/receipt.py`, `model/db/image.py` — DB row models (`ReceiptRow`, `ReceiptWithDetails`, `LineItemRow`, `TaxLineRow`, `ImageRow`, ...); receipt and image rows carry `user_id`.

### 6. Configuration (`config.py`)
Pydantic settings with three sources and **environment > dotenv > YAML** precedence:
- Process environment variables (highest), then `.env` / `.env.local` (nested delimiter `__`, e.g. `LLM__MODEL_NAME`, `PG__HOST`), then the persistent YAML file.
- **Persistent YAML**: `VISION_BILL_CONFIG_PATH` (default `/app/config/config.yaml`; `./server_data/config` in Docker). Created on first start from resolved env/default values (`initialize_config_file`); rewritten atomically with `0600` permissions by the admin settings page (`write_config_file`).
- Sections:
    - `api` — `port` (8080), `log_level`.
    - `images` — `save_dir`, `tmp_dir`, `bypass_review_default` (skip the manual review step on upload).
    - `llm` — `provider` (default `ollama`; also `openai`), `host`, `api_key` (OpenAI-compatible providers), `model_name`, `temperature`.
    - `worker` — `check_interval_seconds` (background analysis worker, default 300).
    - `pg` — user/password/db/host/port; exposes `database_url` (SQLAlchemy-style `PostgresDsn`) and `pg_dsn` (plain DSN for asyncpg).
    - `auth` — `secret_key` (required; signs the session cookie and is the fallback pepper), `session_cookie_name` (`vb_session`), `session_max_age_seconds` (14 days), `session_secure`, `bootstrap_username`/`bootstrap_password` (first-boot admin), `allow_registration` (default true), `admin_can_see_all` (default false), `pepper`.
- Admin-editable keys are `llm.provider`, `llm.host`, `llm.model_name`, `llm.temperature`, `auth.allow_registration` (`EDITABLE_ENV_KEYS`); their current source is reported by `editable_setting_sources()`, and `runtime_restart_required()` flags provider/host changes that need a restart.

### 7. App Lifecycle (`main.py`)
FastAPI `lifespan`:
1. `mark_startup_settings(settings)` records the live provider/host identity.
2. Creates the LLM provider (factory) and `ReceiptService`; `init_db()` failure is logged and the app continues without a DB.
3. `bootstrap_admin` (idempotent, only on an empty users table): creates the admin from `AUTH__BOOTSTRAP_USERNAME`/`PASSWORD` and assigns legacy orphan receipts/images to it.
4. Creates `ImageService`, `AnalysisScheduler` (started), and `BenchmarkService` (started when the DB is ready).
5. Exposes everything on `app.state`.
On shutdown: scheduler/benchmark worker stopped, then the DB pool is closed.

A SvelteKit SPA is served from `src/vision_bill/static` with an `index.html` fallback for client-side routes, registered after the API routers so `/api` routes keep their own behaviour (a 404 is raised for anything under `api/`); the registration is a no-op while no build exists.

## Database & Migrations
- PostgreSQL; schema owned by Alembic. Migrations are hand-written raw SQL in `alembic/versions/` (no autogenerate):
    - `0001_initial_schema.py` — complete initial schema; every entity primary and foreign key uses UUID.
    - `0002_analysis_integrity_and_indexes.py` — hardens analysis retries (image status check `pending/processing/analyzed/failed`) and adds indexes for verified-receipt reporting.
    - `0003_drop_line_item_category.py` — drops the unused `line_items.category` column (receipts keep their own `category`).
- Core tables include `users`, `receipts`, `line_items` (with a `tags` text array), `tax_lines`, `images`, `tags` (line-item tag vocabulary), and the benchmark run/task tables.
- Receipts carry a workflow: `status` (`unverified` → `verified`) plus `confidence` and `verified` columns; the image lives in tmp storage until `verify` (or bypass-review upload) moves it to permanent storage.
- Before starting the app: `uv run alembic upgrade head` (safe on existing DBs — migration 0001 uses `IF NOT EXISTS`).
- New migration: `uv run alembic revision -m "message"`, then edit `alembic/versions/<rev>_<message>.py` with explicit SQL.
- The image entrypoint (`docker-entrypoint.sh`) runs `alembic upgrade head` automatically before uvicorn, so both `docker compose up` and a bare `docker run` migrate on start.
- Deployment identity model (see `scripts/README.md` + `make db-configure-roles`): an owner/migration login for Alembic, a `vision_bill_runtime` group for the app, and a `vision_bill_readonly` group for reporting. The app cannot run as the read-only role.

## Development Workflow
A `Makefile` defines the canonical workflow (`make help` lists targets):
- `make setup` — `uv sync --extra dev` (installs ruff, mypy, pytest).
- `make migrate` — `uv run alembic upgrade head`.
- `make run` — run the API locally with auto-reload on port 8080.
- `make test` — `uv run --extra dev pytest tests/ -m "not e2e"` (unit suite; e2e excluded).
- **E2E tests** (separate from the fast unit suite): a self-contained Docker
  Compose stack (`docker-compose.e2e.yml`) boots the real app image +
  Postgres (host port 5433) + a deterministic OpenAI-compatible LLM stub
  (`e2e/llm_stub/server.py`, host port 9123). The app is on host port 53625
  (the Playwright default).
  - `make test-e2e` — starts the stack, waits for readiness
    (`e2e/wait_ready.py`), runs `pytest tests/e2e/`, and tears the stack
    down on exit (success, failure, or Ctrl-C).
  - `make fe-test-e2e` — same lifecycle around the Playwright run.
    (`VB_E2E_BASE_URL` overrides the app URL; running `playwright test`
    directly still works via the config's `webServer`.)
  - `make e2e-up` / `make e2e-down` — manual start / stop+wipe for
    interactive debugging.
  - Fixtures: opt-in pairs in `tests/e2e/data/` (gitignored) registered in
    `tests/e2e/data/fixtures.toml` — each valid `<name>.<img>+<name>.json`
    (validating as a `Receipt`) becomes one parametrized pipeline test case.
    See `tests/e2e/data/README.md`.
  - The stub loads its fixture map at container start — restart it after
    adding fixtures: `docker compose -f docker-compose.e2e.yml restart llm-stub`.
  - The settings e2e test rewrites `e2e/config/config.yaml` (it changes
    `llm.temperature`); restore with `git checkout -- e2e/config/config.yaml`
    after a run if a clean tree matters.
  - `make test` excludes e2e (`-m "not e2e"`). e2e is deliberately **not**
    in pre-commit (cold runs cost ~20–25 s per suite and need ports
    53625/5433/9123 free, which the dev stack may hold).
  - **CI** (`.github/workflows/ci.yml`): runs on every push/PR — `checks`
    (lint + unit + fe-verify), `e2e-backend`, `e2e-frontend`. The repo is
    public, so GitHub Actions minutes are unlimited and free; CI is the
    safety net that makes "forgot to run e2e locally" a non-issue.
- `make lint` — `uv run ruff check src` + `uv run mypy src` (mypy runs in `strict` mode; `alembic/` is excluded from both).
- `make docker-up` / `docker-down` / `docker-logs` — dev container workflow (builds from the working tree, tags `vision-bill:latest`).
- Operational scripts (values from `.env.scripts`; see `scripts/README.md`):
    - `make script-context-budget ARGS="..."` — measure Ollama prompt token usage.
    - `make script-create-benchmark ARGS="..."` — queue a benchmark run against the running API (admin account).
    - `make db-configure-roles` — create the PostgreSQL runtime/read-only privilege groups.
- Frontend (`web/`, Node via nvm; run `make fe-install` once after dependency changes):
    - `make fe-dev` — Vite dev server on :5173 (proxies `/api` → :8080).
    - `make fe-build` / `fe-sync` — build the SPA into `web/out` and copy it into `src/vision_bill/static`.
    - `make fe-check` (svelte-check + eslint + prettier), `make fe-test` (Vitest), `make fe-test-e2e` (Playwright), `make fe-verify` (check + test + build).

Additional notes:
- **Pre-commit**: `.pre-commit-config.yaml` runs three local hooks on every commit: `make lint`, `make test`, and `make fe-verify` (`uv` and Node/nvm must be on PATH).
- **Functional/benchmark scripts**: `scripts/run_functional_tests.py`, `scripts/llm_benchmark_runner.py`, `scripts/single_image.py`, and `scripts/test_external_api.py` are historical placeholders with hard-coded or mock behavior; they deliberately have no Make targets.
- **Test data**: `tests/data/` (gitignored) holds real receipt images alongside ground-truth JSON used to evaluate model extraction quality locally.

## Frontend
- SvelteKit SPA in `web/` (static adapter; built output is a plain SPA with an index fallback). Routes cover login, receipt review/editing, the pending queue, product search, statistics, benchmark runs, settings, and upload.
- i18n in `web/src/lib/i18n/` (`en.json`, `de.json`) with a parity test; API client and auth session handling in `web/src/lib/`.
- The built SPA is served by FastAPI from `src/vision_bill/static` (see App Lifecycle). In Docker the image build compiles the SPA itself (multi-stage), so a fresh clone always ships a current UI; locally use `make fe-sync` after building.

## Docker & Deployment
- `Dockerfile`: multi-stage — stage 1 (`node:24-slim`) builds the SvelteKit SPA; stage 2 (`python:3.12-slim` + uv) installs Python deps frozen from `uv.lock`, copies the app + Alembic tooling, bakes the built SPA into `src/vision_bill/static`, and installs `libmagic1` for python-magic.
- `docker-compose.yml`: app (`vision_bill`) + `postgres:18` (healthcheck-gated; **no host port published** — reachable only via the internal `postgres_net` bridge at hostname `db`). Postgres runs with explicit tuning (shared_buffers, max_connections, ...) and a 512M memory limit.
- App container: `env_file: .env`; mounts `./server_data/logs` → `/app/logs`, `./server_data/uploads` → `/app/uploads`, `./server_data/config` → `/app/config`; uses `host.docker.internal:host-gateway` so it can reach Ollama on the host; host port `${API__PORT:-8080}:8080`.
- The image entrypoint (`docker-entrypoint.sh`, a JSON-form `ENTRYPOINT` + `CMD`) runs `uv run alembic upgrade head` then `exec`s uvicorn as PID 1 for graceful shutdown.
- `docker-compose.image-stichter.yml` + `image-sticher/`: optional nginx-served image-stitching frontend on port 8081.

## Operational Notes & Constraints
- **Local Inference by Default**: The default provider is Ollama (local). Ensure vision models (e.g. Llama-3-Vision, Moondream, Gemma) are pre-downloaded on the host.
- **Optional OpenAI-Compatible Provider**: `LLM__PROVIDER=openai` sends image data to the user-configured endpoint (`LLM__HOST` + `LLM__API_KEY`). To preserve the privacy model, keep this pointed at a local endpoint unless the user explicitly opts into an external one.
- **Privacy**: PII is designed to stay on the local machine. Do not propose new external API calls for data processing unless explicitly requested and approved.
- **Authentication**: `AUTH__SECRET_KEY` is required (signs the session cookie and is the password pepper fallback). First admin comes from `AUTH__BOOTSTRAP_USERNAME`/`PASSWORD` on an empty DB, or via self-service registration (`AUTH__ALLOW_REGISTRATION`).
- **File Detection**: Uses `python-magic`; the system library `libmagic` must be available (`libmagic1` on Debian/Ubuntu).

## Contextual Hints
- Entry point: `src/vision_bill/main.py`
- Configuration: `src/vision_bill/config.py` (env > dotenv > YAML; nested env vars, e.g. `LLM__MODEL_NAME`, `PG__HOST`)
- DB access pattern: services never touch asyncpg directly — go through `provider/db/`
- LLM access pattern: services never call Ollama/OpenAI directly — go through `provider/llm/` + `provider/factory.py`
- Auth pattern: protect routes with `Depends(get_current_user)` (or `require_admin`); always scope queries with `user_id` + `can_see_all`
- Background work: the analysis queue is drained by `AnalysisScheduler`; benchmark tasks by `BenchmarkService` — do not add ad-hoc background loops

## Edit files
- The edit tool matches oldString byte-for-byte. Read the file again immediately before each edit, and again after every successful edit line content shifts.
- Copy oldString verbatim from the file you just read. Do not retype, reindent or normalise it. Preserve tabs, trailing spaces and blank lines.
- Keep oldString to the smallest span that is still unique — usually one or two lines, not a whole block.
- After two failed edits on the same file, stop and report. Do not switch to rewriting the file.
