# Operational scripts

Supported Make targets read the trusted shell-compatible ignored `.env.scripts`:

```bash
cp .env.scripts.example .env.scripts
```

Edit it for deliberately selected local services. `SCRIPTS_ENV=/private/tools.env`
selects another file. Make exports those values over application dotenv settings.
The base Compose DB has no host port: host-side scripts need an explicitly
reachable connection/network; template ports do not publish one. Keep credentials
in private files, not ARGS or shell history. Reports and traces may contain receipt
text and must remain private. See [upgrade/roles/recovery](../UPGRADE_AND_VERIFY.md).

Run these diagnostics explicitly or test your own receipts through the UI.
Build, installation and ordinary startup leave benchmark creation to the operator.
Persisted pending work can resume under policy and readiness guards. Choose your
own LLM endpoint and model. Native OCR qualification runs through explicit
maintenance and is separate from LLM operation.
For native shared-GPU work, all supported clients must use the application's same
configured provider, lifecycle mode, endpoint spelling, OCR settings and underlying
`OCR__COORDINATION_PATH` file. See [OpenAI/llama.cpp setup](../OPENAI_COMPATIBLE.md).
A host-side path must refer to the same file mounted at `/app/config` in Compose.
Managed standalone/forced OCR also requires production OCR selected (canonical
nonblank `OCR__ENDPOINT` or legacy `OCR__ENABLED=true`) and
`OCR__MANAGE_MODEL_LIFECYCLE=true`; it is rejected before load when ordinary app
LLM work would not participate. Explicit unmanaged CPU/replay modes are separate.
See [model ownership and recovery](../paddleocr-service/README.md#internal-model-ownership).

## Context-budget diagnostic

```bash
make script-context-budget ARGS="--model installed-local-model"
make script-context-budget ARGS="--model installed-local-model --image /private/sample.png"
make script-context-budget ARGS="--model installed-local-model --db-sample 24"
```

This diagnostic is Ollama-only; selecting an OpenAI-compatible application backend
does not change its protocol. With managed OCR, its endpoint spelling must also
match the application's shared ownership identity.
`--host`/`OLLAMA_HOST` selects local Ollama; `--app-url` reads tag vocabulary,
`--tags-file` supplies a JSON tag list, and `--db-sample` reads existing image paths.
The command sends selected images to that endpoint. It requests a prompt count
with `num_predict=1`, bounding requested context by reported capacity and diagnostic
`--num-ctx-cap` (default 16384). Missing/zero/truncated counts are rejected; show metadata
can supply context if tags metadata does not. It counts image/reserved output once;
projected tag growth is an estimate. `--output-budget` defaults 2000 and `--tag-probe`
defaults 250. These diagnostic values do not set the app's formatter 32768, OCR 4096 or
model context window and do not automatically alter resolution/precision/limits.
With managed OCR enabled, generation probes use the shared handoff and release
their exact selected model before another OCR phase.

## Create an API benchmark

The API must be running and the account admin. References must have ordinary human
verification and explicit qualification of the **current** document/sources through
the benchmark reference preview/confirm UI/API. Legacy or bypass `verified` is
insufficient. Put `VB_USERNAME`/`VB_PASSWORD` in the private env.

```bash
make script-create-benchmark ARGS="--model installed-local-model --limit 10"
make script-create-benchmark ARGS="--model installed-local-model --receipt-id 125b2a52-60ac-4e1f-b9bf-d223f15c8cd1"
```

Supply at least one explicit `--model` and use each exact model ID once. Repeat
`--receipt-id` to select individual references or omit it to use current eligible
Gold in the caller's scope. Use `--category`, `--max-source-confidence` and `--limit` to filter
the dataset. Set the per-request timeout with `--request-timeout-seconds`
(default 300, maximum 600). The CLI still accepts council policy, thresholds and
flags as legacy run metadata. They have no effect on scoring, model execution or
receipt flags. See `--help` for supported CLI options and `BenchmarkCreate` for
the full API contract.

This CLI queues a single run using the configured upload pipeline.
The web app also supports independent photo cases, explicit single-pipeline
runs and paired direct/OCR runs.
Use the admin benchmark form or the documented case and run endpoints for
those workflows. [Benchmark walkthrough and API](../UPGRADE_AND_VERIFY.md#benchmarks-and-reproducibility)

## Standalone benchmark (no API)

This command runs OCR + LLM against qualified receipts only. Dedicated photo
cases and paired comparisons require the browser or API. Supply a real
application owner UUID and current explicitly qualified human Gold.
Use a genuine SELECT-only reporting login and a connection able to read the
application tables. Original files must be available at their recorded paths.

```bash
make script-standalone-benchmark ARGS="--user-id 125b2a52-60ac-4e1f-b9bf-d223f15c8cd1 --model installed-local-model --limit 20 --output /private/benchmark-results.json"
```

`--user-id` is mandatory; omission never means all owners. `--category`, `--limit`,
`--db-host`, `--db-port`, `--db-user`, `--db-name`, `--ocr-url`, `--llm-host` are
actual selection/connection flags. Prefer private `PG__PASSWORD` over
`--db-password`. The formatter uses the loaded application settings, including
`LLM__PROVIDER`, `LLM__HOST`, `LLM__MODEL_NAME`, `LLM__API_KEY`,
`LLM__TEMPERATURE` and `LLM__MODEL_LIFECYCLE`. Only an explicit `--llm-host` or
`--model` overrides its corresponding field. Use `LLM__PROVIDER=openai` for an
OpenAI-compatible endpoint. `OLLAMA_HOST` is for the context-budget diagnostic.
`PG__*` provides database defaults. For both app-aware manual CLIs,
`script-standalone-benchmark` and `script-test-ocr`, URL precedence is
`--ocr-url` > `OCR__SERVICE_URL` > `http://localhost:8082`.
`OCR__ENDPOINT` selects production OCR but does not supply these manual CLI URLs.
For managed ownership, keep production OCR selection enabled and set the scripts'
`OCR__SERVICE_URL` or pass `--ocr-url` to the same intended service, together with
the matching coordination and LLM settings above. The script explicitly enables
OCR and always selects the OCR pipeline, preserving configured coordination/
lifecycle settings.
It reads an owner-scoped, repeatable-read read-only snapshot, without row locks,
DB mutation, API startup or human-reference qualification writes. Explicit run
creation loads the managed OCR worker, records its runtime identity and parks it.
The first benchmark prediction uses a selected benchmark input.
Every selected case, including changed or missing sources and extraction failures, stays in the v4 denominator.
An ambiguous post-observation classification failure retains available evidence
and counts as a failed case. Upload-only completion recovery applies to queued
submissions.

Output includes manifest hash and model/source/profile/prompt/schema/output/context
identity, observed-field mask, full-receipt correctness, ordered item precision/
recall, exact count/order and amounts. `avg_score` remains separately labeled
**Historical average score (successful cases)**. Full-receipt correctness and
the all-case average use all selected cases. OCR formatting, repairs and
classification receive text and source evidence. Unknown context remains null.
Expected reference values are used only for scoring.

## Native OCR-only diagnostic

Prefer the independent HTTP client with an actually released, qualified standalone
OCR bundle; see [service modes/client](../paddleocr-service/README.md#independent-service-and-client).
For a VisionBill-managed pair, set the OCR-only bundle's `OCR_SERVICE_MODE=managed`
and match shared coordination/LLM identities. Combined bundles already fix managed
mode. Do not use unmanaged lifecycle opt-out to start a native shared pair.

The advanced app-aware diagnostic below requires a deliberately deployed/physically
qualified service and a reachable private endpoint; base source Compose publishes
no OCR host port. For native build/manual qualification prerequisites see the
[native guide](../paddleocr-service/README.md). No real public native release or
hardware qualification was performed here.

```bash
make script-test-ocr ARGS="--ocr-url http://127.0.0.1:8082 --data-dir /private/synthetic-images --timeout 600 --quiet --output /private/ocr-results.json"
```

This URL works only when your deployment explicitly exposes it. Without `--ocr-url`,
this diagnostic also uses `OCR__SERVICE_URL`, then `http://localhost:8082`, regardless
of canonical `OCR__ENDPOINT`; keep both selections on the same intended managed
service. An app-aware managed client needs managed service mode; use the independent
HTTP client for standalone mode instead. A checkout does not update a running image.

`--timeout`/`OCR__TIMEOUT_SECONDS` defaults 300. Effective requests are bounded at
600 seconds; increasing it cannot extend the app whole-attempt deadline. This
explicit diagnostic activates its OCR provider and performs no LLM generation.
Managed mode also unloads/verifies Vision Bill's configured LLM before native OCR,
using the same coordination path and dedicated endpoint as the app. It treats empty
transcription as failure even with HTTP200. Current native v2 JSON includes text,
working-coordinate blocks, profile/runtime identity and actual working PNG payload;
raw output can contain pixels and receipt text. `--quiet` suppresses raw console
output, but `--output` is still private. Use the [source upgrade procedure](../paddleocr-service/README.md#dedicated-ocr-from-source) after changing native source. Maintainers qualify the exact image and device before publication.
No production CPU fallback or `--timeout 900` smoke recipe is supported.

## Thumbnails and optional stitcher

`scripts/generate_thumbnails.py` loads `scripts/.env` and application defaults.
Configure its genuine DB credentials and `IMAGES__SAVE_DIR`/`IMAGES__TMP_DIR` for the
chosen installation. It only selects rows with null thumbnail_path:

```bash
uv run python scripts/generate_thumbnails.py --dry-run
uv run python scripts/generate_thumbnails.py --limit 20
```

Dry-run reads the DB/filesystem, counts resolvable candidates and missing paths;
it **does not decode images**, validate image integrity, create directories, generate
thumbnails or update rows. A candidate count is not a successful-generation count.
Actual mode makes white-composited EXIF-normalized WebP thumbnails; new uploads
already do so nonfatally. It does not regenerate existing thumbnails automatically.

```bash
docker compose -f docker-compose.image-stichter.yml up -d
```

This optional standalone tool binds tracked `image-sticher/image-stitcher.html`
and serves it on 8081. It keeps selected photo order despite asynchronous reads,
invalidates export on new/reordered batches and blocks incomplete/failed exports.
It does not upload to the app automatically or promote blanket stitching/splitting
into the production extraction path.

## Database roles, backup and upgrade

See [the credential-specific procedure](../UPGRADE_AND_VERIFY.md#credentials-and-grants)
for owner migrations, runtime/readonly grants and missing-group bootstrap. Missing
groups require admin/CREATEROLE first; then `make db-configure-roles` must use the
actual Alembic owner for grants/default privileges on current/future tables and
sequences. `MIGRATION_DATABASE_URL` is only this Make tool's connection; entrypoint
Alembic uses `PG__*`. Runtime credentials cannot apply pending owner DDL. Stop/drain
old workers, back up/restore-test DB and files, migrate to 0025 with owner credentials,
refresh grants and start only compatible workers. No automatic destructive backout.
`bash scripts/pg-backup.sh --help` lists backup/restore options.

## Comparisons and historical prototypes

The [OCR harness](ocr_harness/README.md) is isolated from production native OCR.
The independent historical CPU Paddle target is built by
`python3 -m scripts.ocr_harness build --only pp-medium`; its lock does not certify
physical inference here. The [split experiment](receipt_split_experiment.md)
remains separate. [OCR historical results](ocr_harness/BENCHMARK_RESULTS.md) and
[split historical results](receipt_split_results.md) retain their original numerical
meaning and selected datasets; do not substitute them for current GPU qualification.

`run_functional_tests.py`, `llm_benchmark_runner.py`, `single_image.py`,
`test_external_api.py` are historical placeholders with hard-coded/mock behavior
and no supported Make targets. `test_gpc_tagging.py` is also not a supported
benchmark target. Do not contact prototype external endpoints with receipt data.
