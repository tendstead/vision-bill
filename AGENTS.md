# Vision-Bill Agent Guide

Technical context for agents working on the vision-bill project.

## Project Overview
**vision-bill** is a privacy-first receipt/invoice tracking service: one Go binary serves the JSON API and the bundled SvelteKit web app. It uses local multi-modal LLMs (Ollama by default) to extract structured data from receipt photos, so sensitive PII does not have to be sent to external cloud services. It is multi-user (session-cookie authentication, per-user data ownership, admin privileges), runs a background analysis worker for queued uploads, and has a benchmark subsystem for comparing extraction quality across models.

## Technology Stack
- **Language**: Go (`go.mod` pins the toolchain), `net/http` with the 1.22+ method/pattern mux
- **Database**: PostgreSQL via `pgx/v5` (explicit SQL, no ORM); schema owned by embedded SQL migrations
- **Frontend**: SvelteKit SPA (TypeScript, Tailwind, Vitest, Playwright) in `web/`, embedded into the binary (`cmd/server/static`)
- **LLM**: Ollama (default) and OpenAI-compatible endpoints, plain HTTP clients in `internal/llm`
- **Money**: exact decimals (`shopspring/decimal`); amounts are never `float64`. Columns: money `NUMERIC(14,2)`, quantity `NUMERIC(10,4)`, tax rate `NUMERIC(6,4)`; values are rounded half away from zero when written.
- **Images**: the binary is static (`CGO_ENABLED=0`). The container adds `vips` (libvips-tools) for WebP thumbnails and HEIC/HEIF decoding; without it thumbnails are off and HEIC/HEIF uploads are rejected (reported as `heic_supported` in `/api/v1/system/ui-config`).
- **Tests**: Go tests with a real PostgreSQL (`internal/*`), a black-box Go API suite against the compose stack (`e2e/suite`, build tag `e2e`), Playwright in `web/e2e`. No Python anywhere.

## Repository Layout
- `cmd/server/` — the server (`main.go`), embeds `static/` (gitignored build output).
- `cmd/vbctl/` — operational tools: `benchmark`, `thumbnails`, `context-budget` (also in the image as `/app/vbctl`).
- `internal/config` — settings: environment > `.env`/`.env.local` > persistent YAML; `Holder` for live updates.
- `internal/store` — pgx pool, embedded migrations (`migrations/`), and all SQL (users, receipts, collections, images, statistics, benchmarks).
- `internal/domain` — receipt schema + validation (`ParseReceipt`), rows/DTOs, categories, exact money helpers.
- `internal/security` — peppered Argon2id hashing and signed session tokens.
- `internal/llm` — providers, prompt (`BuildPrompt`), extraction/repair loop, multi-photo overlap reconciliation.
- `internal/files` — upload validation, tmp/permanent storage, thumbnails, HEIC conversion for models.
- `internal/analysis` — extraction service and the durable-queue `Scheduler`.
- `internal/benchmark` — scoring (`DetailedScore`, legacy `ScoreReceipts`), row alignment, run creation and the task worker.
- `internal/httpapi` — routes, validation, auth middleware, OpenAPI contract (`openapi.json`), docs pages.
- `internal/app` — database connection lifecycle and first-admin bootstrap; `internal/testdb` — throwaway test databases.
- `web/` — SvelteKit source (routes, i18n `en`/`de`, API client). `docker-compose.e2e.yml` + `e2e/` — e2e stack: app, Postgres and a deterministic OpenAI-compatible LLM stub (`e2e/stub`, Go; committed fixtures in `e2e/stub/fixtures`, opt-in local pairs in `e2e/data` via `fixtures.txt`), the API suite in `e2e/suite`, HEIC/rotation assets in `e2e/assets`. `scripts/` — release driver, Go test runner, DB role SQL.

## API (all under `/api/v1`)
Every route requires an authenticated user except `POST /auth/register`, `POST /auth/login`, `POST /auth/logout` and `GET /system/ui-config`. IDs are UUIDs. Errors are `{"detail": ...}`; request validation errors use the `[{type, loc, msg, ...}]` list shape with status 422. `GET /openapi.json`, `/docs` and `/redoc` serve the contract (`internal/httpapi/openapi.json`; a test checks that every contract operation is routed). `GET /healthz` is liveness, `GET /readyz` checks DB connectivity and migration state.

- `auth`: register (201/403 disabled/409 duplicate), login, logout, me.
- `images`: `POST /` multipart field `receipt` (1–10 photos, 20 MiB each; query `model_id`, `bypass_review`) → `201` analysed synchronously (single photo, model reachable) or `202` + `Location` when queued; `GET /`, `POST /analyze`, `GET /{id}`, `/{id}/file?part=`, `/{id}/thumb`, `DELETE /{id}`.
- `receipts`: list/detail/`PUT`/`DELETE`, `POST /{id}/verify` (moves photos to permanent storage; repeatable after interruption). `PUT` commits the receipt edit before validating `collection_ids` (unknown collection → 404 but the edit persists).
- `collections`, `tags` (+ `/categories`, `/category-suggestion`), `search`, `statistics` (+ `/items`), `llm/models`, `system/settings` (admin), `benchmarks` (admin: create/list/status/results/capabilities/reevaluate).

## Behaviour to preserve
- Scoping: all receipts/images/collections are filtered by owner; admins only see others' data when `auth.admin_can_see_all` is set (`can_see_all`).
- Degraded start: the server listens even when PostgreSQL is down; DB-backed routes answer `503`, `ui-config` keeps working, the DB connection is retried in the background. A migration/adoption problem (unknown revision, schema drift) is fatal.
- LLM outage never blocks login, browsing, review or durable upload acceptance: uploads are stored and queued; the scheduler retries every `worker.check_interval_seconds`. Expired processing leases (15 min) are reclaimed.
- Sessions: `vb_session` cookie, HMAC-signed `"{user_id}.{exp}.{sig}"`, HttpOnly, SameSite=Lax; passwords are Argon2id over `HMAC-SHA256(pepper, password)` hex (pepper falls back to `auth.secret_key`).
- The container listens on :8080 regardless of `api.port` (override with `VB_LISTEN_ADDR`).

## Database & Migrations
`internal/store/migrations/*.sql` are applied in order inside one transaction under an advisory lock, recorded in `vision_bill_go_migrations`. `001_baseline.sql` is the schema at revision 0010. An existing database carrying the legacy `alembic_version` marker is adopted only at exactly `0010` with a matching column inventory (`schema_columns.txt`); other revisions, newer revisions and unmarked databases with tables are refused. Fresh installs also get the marker table so existing tooling keeps working. To change the schema add the next numbered SQL file (and update `schema_columns.txt` if you want drift detection of new columns).

## Development Workflow (`make help`)
- `make lint` (gofmt + go vet), `make test` (Go tests with the race detector; starts a throwaway PostgreSQL container unless `VB_TEST_DATABASE_URL` points at one), `make build`, `make run`.
- `make fe-install | fe-dev | fe-build | fe-sync | fe-check | fe-test | fe-verify`; `fe-sync` copies the SPA into `cmd/server/static` for embedding.
- `make test-e2e` (Go API suite) / `make fe-test-e2e` (Playwright) boot `docker-compose.e2e.yml` (app on 53625, Postgres 5433, LLM stub 9123, `up --wait` on healthchecks), run the suite and tear down. The stub mode is global state, so the suite runs sequentially (`-p 1`). `make e2e-up` / `e2e-down` for manual debugging. The e2e settings test rewrites `e2e/config/config.yaml`; restore with `git checkout -- e2e/config/config.yaml`.
- Tools: `make script-context-budget`, `script-create-benchmark`, `script-thumbnails`, `db-configure-roles` (see `scripts/README.md`).
- Pre-commit runs `make lint`, `make test`, `make fe-verify`.

## Docker & Deployment
`Dockerfile` is multi-stage: Node builds the SPA, Go builds `/vision-bill` and `/vbctl` (static), the runtime stage is `debian:bookworm-slim` + `libvips-tools` (the build fails when vips lacks HEIF/WebP support). It runs as the image default user so existing `server_data/` ownership keeps working. `docker-compose.yml`: app + `postgres:18` (healthcheck-gated, no published DB port), mounts `./server_data/{logs,uploads,uploads_tmp,config}`; the app reaches Ollama on the host through `host.docker.internal`. Migrations run in-process at start.

## Constraints
- **Privacy**: receipt data goes only to the configured inference endpoint (`LLM__HOST`). Do not add external calls for data processing without explicit approval.
- **Config**: `AUTH__SECRET_KEY` is required. Admin-editable keys: `llm.provider|host|model_name|temperature`, `auth.allow_registration`; keys controlled by environment variables answer `409` on change; provider/host changes need a restart.
- Background work: the analysis queue is drained by `analysis.Scheduler`, benchmark tasks by `benchmark.Service`; do not add ad-hoc loops.
- Never log or commit real receipts, credentials or backups (`server_data/` is ignored).

## Edit files
- The edit tool matches text byte-for-byte; read a file immediately before editing it and re-read after each edit.
- Keep matches short and unique; after two failed edits on a file, stop and report instead of rewriting it.
