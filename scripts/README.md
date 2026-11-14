# Operational scripts

For unattended, isolated comparisons across OCR engines, see
[`ocr_harness/README.md`](ocr_harness/README.md). Run `make ocr-compare
ARGS="all --data-dir /path/to/receipts --limit 2"` to build, prepare and test
all enabled configurations. No application deployment is needed.

The supported scripts run through Make so local endpoints and database values
can live in one ignored `.env.scripts` file.

## Setup

```bash
cp .env.scripts.example .env.scripts
```

Edit the local file as needed. Make exports its values before invoking a
script, so they take precedence over values loaded from the application's
`.env` files. To use another file, pass `SCRIPTS_ENV=/path/to/file`.

## Context-budget measurement

This asks the local Ollama API for exact prompt token counts. It can optionally
read tags from the running application and sample receipt image paths from
PostgreSQL.

```bash
make script-context-budget ARGS="--model glm-ocr:latest"
make script-context-budget ARGS="--model glm-ocr:latest --image tests/data/lidl.jpg"
make script-context-budget ARGS="--model glm-ocr:latest --db-sample 24"
```

The command sends image content to the configured `OLLAMA_HOST`. Keep that URL
pointing at the local Ollama instance to preserve Vision Bill's privacy model.

## Create a benchmark run

The API must be running, the account must be an admin, and selected receipt IDs
must be UUIDs for verified receipts. Put `VB_USERNAME` and `VB_PASSWORD` in the
ignored `.env.scripts`, or pass them as arguments.

```bash
make script-create-benchmark ARGS="--model glm-ocr:latest --limit 10"
make script-create-benchmark ARGS="--receipt-id 125b2a52-60ac-4e1f-b9bf-d223f15c8cd1"
```

## Standalone benchmark (no API required)

Run a benchmark directly against the database and services without starting
the Vision Bill API. Useful for testing against remote databases or NAS data.

```bash
# Basic usage with environment variables
make script-standalone-benchmark ARGS="--model glm-ocr:latest --limit 20"

# Specify database connection
make script-standalone-benchmark ARGS="--model glm-ocr:latest --db-host 192.168.1.100 --limit 20"

# Output results to JSON file
make script-standalone-benchmark ARGS="--model glm-ocr:latest --limit 20 --output benchmark-results.json"
```

Required environment variables (or command-line options):
- `LLM__MODEL_NAME` or `--model` - Model to use
- `PG__HOST`, `PG__USER`, `PG__PASSWORD`, `PG__DB` - Database connection (or `--db-*` options)

Optional:
- `OCR__SERVICE_URL` or `--ocr-url` - PaddleOCR service URL (default: `http://localhost:8082`)
- `LLM__HOST` or `--llm-host` - LLM host (default: `http://localhost:11434`)
- `OCR__TIMEOUT_SECONDS` - OCR timeout in seconds (default: 300)

## Test the OCR service (no API, no LLM)

Send the local receipt images in `tests/data/` straight to the PaddleOCR
service and print the raw JSON it returns. This exercises only the OCR stage
of the two-stage pipeline — it never calls the LLM. Useful for checking that
the service is up and returning sane layout/text blocks before wiring it into
extraction.

```bash
make script-test-ocr
make script-test-ocr ARGS="--ocr-url http://localhost:8082"
make script-test-ocr ARGS="--quiet --output /tmp/ocr-results.json"
```

Options:
- `--ocr-url URL` - PaddleOCR service URL (default: `OCR__SERVICE_URL` or `http://localhost:8082`)
- `--timeout N` - request timeout in seconds (default: `OCR__TIMEOUT_SECONDS` or 300)
- `--data-dir DIR` - directory with test images (default: `tests/data`)
- `--output FILE` - write all raw results to a JSON file
- `--quiet` - print per-image status and a summary only, not the raw JSON

The service must be reachable (e.g. `docker compose up paddleocr-service`).
Keep the URL pointed at the local service to preserve Vision Bill's privacy
model.

After changing `paddleocr-service/server.py`, rebuild that service with
`docker compose up -d --build paddleocr-service`. A checkout alone does not
update code baked into a running container. The compact response contains
`text` and recognized `blocks` in reading order; image pixels are excluded.
The diagnostic command treats missing/empty transcription as failure even
if the server returns HTTP 200. Quiet output reports recognized characters.

For a small CPU smoke test, put one receipt image in a separate directory:

```bash
make script-test-ocr ARGS="--data-dir /path/to/one-receipt --timeout 900 --quiet"
```

The Compose OCR service has no published host port. Run a client on the same
Docker network or use an explicitly configured local endpoint; `localhost:8082`
works only when that port is actually exposed. CPU inference can take several
minutes for one receipt.

The updated app's benchmark form supports forcing the `OCR` pipeline for a
single-image verified receipt. Disable OCR fallback for diagnostics so a
direct-vision result cannot hide an OCR failure. Start with temperature 0,
classification disabled, and thinking disabled if the selected model supports
it. The benchmark deadline covers both OCR and LLM generation (at most 600
seconds); increasing the OCR client's timeout does not extend that deadline.
Rebuilding the app from this branch is necessary to expose these merged
benchmark controls and applies the new benchmark migration on startup.

## Generate thumbnails for existing images

Creates a WebP thumbnail (`images.thumbnail_path`) for any image that does not
have one yet. Configure the database and image directories in `scripts/.env`
(same directory as the script):

```bash
PG__HOST=localhost
PG__PORT=5432
PG__USER=vision_bill
PG__PASSWORD=...
PG__DB=vision_bill
IMAGES__SAVE_DIR=/path/to/uploads
IMAGES__TMP_DIR=/path/to/uploads_tmp
```

Run (directly, not via Make):

```bash
uv run python scripts/generate_thumbnails.py --dry-run   # preview
uv run python scripts/generate_thumbnails.py             # do it
uv run python scripts/generate_thumbnails.py --limit 20  # cap per run
```

It is idempotent: only rows with `thumbnail_path IS NULL` are touched, so
re-running is safe. New uploads create their thumbnails automatically; this
script is only for backfilling images that predate the feature.

## PostgreSQL roles

Use three kinds of database identity in a deployed installation:

- An owner/migration login runs Alembic and owns schema objects. Do not use it
  for the application process.
- A runtime login belongs to the `vision_bill_runtime` group. It can read and
  modify application rows, but cannot create or alter schema objects.
- A reporting login belongs to `vision_bill_readonly`. Use this for ad-hoc
  analysis and scripts that only query data.

The application itself cannot use the read-only role: uploads, receipt review,
user authentication, and benchmark processing all require writes. To create or
refresh the two privilege groups, set `MIGRATION_DATABASE_URL` to the owner
connection and run:

```bash
make db-configure-roles
```

Create login roles with strong passwords separately, then grant exactly one
group to each login:

```sql
CREATE ROLE vision_bill_app LOGIN PASSWORD 'replace-me';
GRANT vision_bill_runtime TO vision_bill_app;

CREATE ROLE vision_bill_reports LOGIN PASSWORD 'replace-me';
GRANT vision_bill_readonly TO vision_bill_reports;
ALTER ROLE vision_bill_reports SET default_transaction_read_only = on;
ALTER ROLE vision_bill_reports SET statement_timeout = '5s';
```

Do not make either login a table owner or superuser: owners bypass ordinary
table grants. `configure_db_roles.sql` must run as the same owner Alembic uses,
because PostgreSQL default privileges are scoped to the object-creating role.

## Prototype files

`run_functional_tests.py`, `llm_benchmark_runner.py`, `single_image.py`, and
`test_external_api.py` are historical placeholders with hard-coded or mock
behavior. They deliberately have no Make targets. Promote or remove each one
only after replacing its placeholder behavior; in particular, do not connect
`test_external_api.py` to receipt data because that would violate the local-only
inference constraint.
