# Operational tools

The tools are subcommands of `vbctl` (`cmd/vbctl`), which is also shipped inside
the container image as `/app/vbctl`. Locally they run through Make so endpoints
and database values can live in one ignored `.env.scripts` file.

## Setup

```bash
cp .env.scripts.example .env.scripts
```

Edit the local file as needed. Make exports its values before invoking a tool,
so they take precedence over values loaded from the application's `.env` files.
To use another file, pass `SCRIPTS_ENV=/path/to/file`.

## Context-budget measurement

This asks the local Ollama API for exact prompt token counts. It can optionally
read tags from the running application and sample receipt image paths from
PostgreSQL.

```bash
make script-context-budget ARGS="-model glm-ocr:latest"
make script-context-budget ARGS="-model glm-ocr:latest -image tests/data/lidl.jpg"
make script-context-budget ARGS="-model glm-ocr:latest -db-sample 24"
```

The command sends image content to the configured `OLLAMA_HOST`. Keep that URL
pointing at the local Ollama instance to preserve Vision Bill's privacy model.
The live tag list needs an unauthenticated tag endpoint, which the API does not
offer: pass `-tags-file tags.json` (a JSON array of tag names) to include it.

## Create a benchmark run

The API must be running, the account must be an admin, and selected receipt IDs
must be UUIDs for verified receipts. Put `VB_USERNAME` and `VB_PASSWORD` in the
ignored `.env.scripts`, or pass them as arguments.

```bash
make script-create-benchmark ARGS="-model glm-ocr:latest -limit 10"
make script-create-benchmark ARGS="-receipt-id 125b2a52-60ac-4e1f-b9bf-d223f15c8cd1"
```

The command prints the run id.

## Generate thumbnails for existing images

Creates a WebP thumbnail (`images.thumbnail_path`) for any image that does not
have one yet. It reads the application's normal configuration (environment,
`.env`, YAML): the database connection and the image directories matter here.
It needs the `vips` helper (present in the container image).

```bash
make script-thumbnails ARGS="-dry-run"   # preview
make script-thumbnails                   # do it
make script-thumbnails ARGS="-limit 20"  # cap per run
# inside the container:
docker compose exec vision_bill /app/vbctl thumbnails -dry-run
```

It is idempotent: only rows with `thumbnail_path IS NULL` are touched, so
re-running is safe. New uploads create their thumbnails automatically; this
tool is only for backfilling images that predate the feature.

## PostgreSQL roles

Use three kinds of database identity in a deployed installation:

- An owner/migration login runs the schema migrations at application start and
  owns schema objects. If the application runs as a restricted login instead,
  run one start with the owner login first so migrations can be applied.
- A runtime login belongs to the `vision_bill_runtime` group. It can read and
  modify application rows, but cannot create or alter schema objects.
- A reporting login belongs to `vision_bill_readonly`. Use this for ad-hoc
  analysis and tools that only query data.

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
table grants. `configure_db_roles.sql` must run as the same owner that applies
the migrations, because PostgreSQL default privileges are scoped to the
object-creating role.
