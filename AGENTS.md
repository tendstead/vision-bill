# Vision-Bill Agent Guide

## Scope and architecture

vision-bill is a local-first, multi-user receipt API and bundled SvelteKit SPA.
Operators supply an existing Ollama or OpenAI-compatible endpoint and explicitly
select a model. The app and OCR distributions include no LLM runtime or weights.
Use only the operator's configured endpoint. Keep receipt contents out of
general logs. For either provider, admin connection tests use the saved formatter
connection and read only the model list. Preserve manual model entry and require
separate evidence of image capability. Preserve original bytes, human corrections, exact money and owners.

Public operator contracts are in [UPGRADE_AND_VERIFY.md](UPGRADE_AND_VERIFY.md),
[MULTI_IMAGE_RECEIPTS.md](MULTI_IMAGE_RECEIPTS.md),
[paddleocr-service/README.md](paddleocr-service/README.md) and
[scripts/README.md](scripts/README.md) and
[OPENAI_COMPATIBLE.md](OPENAI_COMPATIBLE.md). These describe current behavior; historical
experiment reports do not certify production hardware or accuracy.

## Layout and stack

- Python 3.12+, FastAPI/Uvicorn, Pydantic settings/models, asyncpg/raw SQL,
  PostgreSQL, Alembic hand-written migrations, Argon2id and HMAC session cookies.
- `src/vision_bill/main.py`: app lifespan, providers, services, schedulers, SPA.
- `src/vision_bill/api/`: authenticated routers and public auth/UI config routes.
- `src/vision_bill/model/`: Receipt, nullable extraction, OCR/source, workflow,
  review and experiment/benchmark contracts.
- `src/vision_bill/service/`: orchestration, checks, extraction and benchmarks.
- `src/vision_bill/provider/db/`: raw SQL and transactional ownership/CAS/leases;
  services delegate DB access here. `provider/llm/`: model discovery, extraction,
  repairs/classification; services do not call provider SDKs directly.
- `alembic/versions/`: migrations 0001–0025. No application DDL/autogenerate.
- `paddleocr-service/ocr_service/`: isolated native pipeline, qualification and
  disposable worker lifecycle. `build/` and locks pin runtime/build candidates.
- `frontend/`: SvelteKit static SPA, TypeScript, Tailwind, Vitest, Playwright,
  English/German dictionaries. Built `out/` and app `static/` are generated.
- `e2e/`: unique owned stack runner, strict LLM/OCR stubs and fixture preparation.
- `scripts/`: operational CLIs, CPU OCR comparison and separate split experiment.
- `image-sticher/image-stitcher.html`: optional standalone stitcher, served by
  `docker-compose.image-stichter.yml` on 8081 (historical directory/file spelling).
- `server_data/`, real fixtures, reports/traces, models and local dotenv files are
  private/gitignored. `docs/` is ignored local scratch: put public guides in tracked
  root/existing public paths. Never link public prose to ignored scratch artifacts.

## API and ownership

All `/api/v1` data routes require authentication. Auth register/login/logout,
`GET /system/ui-config` and the boolean `GET /system/ready` health check are public.
Benchmarks/settings require admin. Admin access
across owners additionally requires `auth.admin_can_see_all`; preserve scope on
all reads/mutations, sources, drafts, retries, collections and benchmark references.
Entity IDs are UUIDs. DB-unavailable paths return 503. Browser data requests bind
the originating cookie actor with `X-Vision-Bill-Expected-User` or the private
media query parameter `vb_expected_user`. Malformed or conflicting expectations
return 400. Actor mismatch returns 409 before the protected operation.

- `POST /images`: 1–10 repeated multipart `receipt` fields, at most 20 MiB/photo;
  committed 202 + durable ID + Location for **all** uploads. Options `model_id`,
  `bypass_review`, deprecated compatibility `allow_ocr_fallback`, and owner-scoped
  `Idempotency-Key` preserve explicit presence/intent, ordered hashes and first defaults.
- `GET /images`, `GET /images/{id}`, `/file?part=N`, `/thumb`,
  `/views/{view_id}`, `/extractions/{extraction_id}` expose scoped state/evidence.
  No successful synchronous upload shortcut remains.
- `POST /images/analyze` triggers a shared analysis cycle and returns only results
  visible to the caller. `/images/{id}/retry` requires expected_generation/request_key.
  Draft GET reads current state; PATCH requires expected_revision;
  `/draft/finalize` also takes request_key.
- Receipts list/get/update/delete/verify retain exact amounts and file paths.
  Human review uses the current preview, displayed operation/review revisions
  and the exact finding acknowledgements. Integrity failures block completion.
  Verification and bypass preserve the original and working files.
- Deletion commits rows plus durable owned-file cleanup metadata, then unlinks
  validated originals/thumbnail/working derivatives. Same authorized DELETE resumes
  interrupted cleanup. Active/referenced resources conflict; no arbitrary sweeper.
- Search/statistics use verified receipts. Collections preserve owner-scoped active
  membership, even for see-all admin changes. Tags are normalized vocabulary.
- Benchmark create requires explicit unique model IDs and returns 202/Location.
  Status, results and reevaluation use current owner permissions. GET/POST
  `/benchmarks/references/{receipt_id}` previews and explicitly qualifies current
  human Gold. Verification provenance and current document identity govern Gold.
- Dedicated test cases use `/benchmarks/cases` and ordered `photos` multipart
  fields. Case creation only stores originals and an unconfirmed reference.
  Keep case storage, revision-fenced confirmation and benchmark tasks separate
  from the receipt ledger, spending statistics and ordinary submission queue.

## Workflow and contracts

`analysis_scheduler.py` coordinates readiness and queue draining;
`submission_service.py` executes fenced whole attempts; `workflow_db.py` owns durable
claims/generations/retry/backoff/drafts/extractions/views/deletion. No ad-hoc worker
loops. Default whole-attempt deadline 600s, max three attempts, 30/60s backoff,
cap 300s, lease 900s; readiness/model-start waits can remain pending at zero attempts.
Stale completions are fenced. A later-photo failure retains only the validated
prefix. Legacy linked receipts keep their original provenance. An all-photo
observation checkpoint precedes optional enrichment. Its completion-only claim
rechecks frozen policy/endpoint, generation, original/working artifacts, current
receipt and review state, and settled resource ownership. It spends no inference
attempt and appends a separate idempotent history event. Keep ambiguous dispatches
held through recovery. See [upload recovery](MULTI_IMAGE_RECEIPTS.md#complete-observations-and-optional-labels).

`extraction_pipeline.py` selects direct vision only without OCR, otherwise OCR
followed by strict text/evidence-only formatting, repair and classification. No
ordinary failure path sends images to the LLM. Deprecated fallback request fields
remain replay/history compatibility only; explicit admin direct benchmarks compare
pipelines. Freeze execution identities and all ordered hashes. A validated exact
OCR prefix can be reused, but formatting/completed output requires every photo.
Policy mismatch holds work before claim; retry `policy: "current"` explicitly adopts
the current snapshot with history/CAS/request replay. Never overwrite completed
receipts, human edits or live claims. See the upload guide.

Printed observations live in nullable `ExtractionDraft`. Category and tags are
optional downstream enrichment. Missing currency stays missing. Missing
amounts may be computed from complete observed rows and sufficient tax and
adjustment evidence. Record calculated values in review provenance.
Complete observations can survive safely settled label failure with an explicit
`classification_enrichment` status. Missing optional labels stay outside
financial and integrity checks.
Keep the printed merchant and editable display name separate. Only an explicit
save/finalization/verification by the receipt owner teaches their private
merchant directory. The shared bilingual product tag dictionary lives in
`src/vision_bill/model/product_tags.json` and is bundled into both the SPA and
Python package. Benchmarks use frozen references.
Only proposed adjacent suffix/prefix overlaps that pass shared duplicate verification
may be removed. Preserve repeated purchase observations. The ledger and draft
editor may aggregate their exact quantities and amounts with projected sources.
`receipt_review.py` checks version 3 performs exact Decimal diagnostics
and policy-dependent financial checks without changing printed arithmetic. Review
exceptions require one exact current finding ID/nonblank comment each; integrity
findings cannot be waived. Local time supports valid seconds and microseconds.

`ImageService` preserves original bytes/hash, creates thumbnails nonfatally and
saves exact validated working PNG artifacts. Shared white RGB composition is
`white-document-rgb-v2`; EXIF correction and native orientation are distinct. Geometric unwarping is off.
Working geometry stays in working coordinates and inverse mapping remains null.
Shared source limits are 20 MiB for the app and 25 MiB for the native service
with 40,000,000 pixels, a derived RGB8 PNG bound and bounded metadata/response
encoding. See the [native contract](paddleocr-service/README.md#fixed-extraction-policy).

## Native OCR policy

Base app OCR is off. By default the source launcher waits until the app and
database are ready, then builds the selected CUDA/ROCm image only after the
admin requests Dedicated OCR. The host owns Docker access. A managed source
upgrade can reuse a compatible admitted image after offline GPU binding.
Prepared installations start an idle OCR controller. Activation requests
verified model downloads and a versioned prepared installation. Ordinary
installation, restart and worker creation must make zero model predictions.
The explicit release qualifier owns all fixture inference. Default OCR Dockerfile
requires a selected accelerator.
Linux x86_64 containers target NVIDIA Turing/RTX 20+ and AMD RDNA2/RX 6000+
on native Linux and WSL 2. The CUDA probe's vendor floor is SM 7.0. Consumer
qualification starts at Turing SM 7.5. Source installations recheck the observed
GPU UUID when reusing an admitted native image. AMD Dedicated OCR is experimental
and driver support is board- and host-specific.
The source launcher and `make ocr-build-rocm` select the local GFX architecture
and KFD or WSL transport. The explicit target list selects architectures.
The `dual` build transport includes both GPU interfaces. Release plans carry
their own target list.
Record the selected build profile separately from physical qualification.
Never claim universal AMD support or GPU qualification from CPU tests.

Fixed PaddleOCR-VL-1.6/v1.6, PP-DocLayoutV3, native Paddle, batch 1 everywhere,
queues/chart/seal off, orientation on, unwarping off, min 112896/max 2000000 **per VLM region**
including tables, OCR 4096. Official upstream grid alignment is not a whole-photo
2 MP resize. Active models are orientation, layout and VLM. Preserve earlier
UVDoc assets and saved working views with the installation that produced them. Only the VLM text decoder uses BF16 on selected CUDA SM80+ devices.
Turing, ROCm, static models and visual/projector layers use FP32. Keep the
backend, pixel and token policy fixed. EXIF applies independently. Formatter output defaults
32768; discovered context capacity remains separate and may be unknown.

`/health` is liveness. `/ready` is worker readiness and `/info` carries the
capability, profile and runtime identity. Dedicated OCR serves a verified
installation. A changed image, device or model requires validation before use.
Readiness polling detects an already-explicitly-loaded idle worker death and
schedules bounded coordinated replacement;
failed worker initialization stays unready, with bounded recovery.
Actual CUDA/ROCm builds, official CUDA lock resolution, physical inference/quality
and current CPU comparator installation/inference are unexecuted here.

## Benchmarks and scripts

`benchmark_service.py` / `benchmark_db.py` freeze current explicitly human-qualified
Gold and manifest identity, queue one job per eligible model/input/pipeline and
fence leases/begin/retry/finish. Data/reference changes are checked transactionally.
Single direct and single OCR requests schedule one pipeline. An explicit
`comparison: true` creates two linked runs against the same captured inputs.
Include text-only models only in the OCR arm. History and result reads never
dispatch inference. Use scoring version 5 for every input in a run containing
dedicated test cases, including any qualified receipts in that run. Score only the
confirmed observations and report optional classification separately. New
receipt-only runs use version 4. V4 exact full-receipt,
occurrence-aware items, amounts/taxes and all-case denominators supplement the
separately named historical success-only averages/v3 scores. Do not redefine old
results or leak expected values into inference. Standalone benchmarks require
`--user-id` and read a consistent SELECT-only snapshot. See script help/current
parser before documenting flags. Historical CPU transcripts have their own explicit
loader contract; they do not become native OCR v2 by rerunning a CPU adapter.

## Configuration, lifecycle and migration credentials

`config.py` precedence: process environment > `.env`/`.env.local` > persistent YAML.
Nested names use `__`. Canonical `OCR__ENDPOINT` selects OCR when nonblank and
direct vision when explicitly blank, overriding legacy enabled/service URL fields.
Absent canonical endpoint preserves legacy disabled-URL parsing; a URL alone does
not enable legacy OCR. Do not add an active blank endpoint to the source template:
it would defeat existing GPU overlays. `VISION_BILL_CONFIG_PATH` defaults
`/app/config/config.yaml`;
admin writes are atomic/0600. OpenAI inference can opt into `LLM__MODEL_LIFECYCLE=llamacpp`
for a dedicated router with exact canonical IDs, autoload on and eager loads off.
The default `none` cannot release models for shared managed OCR; Ollama ignores this
OpenAI-only option. In llama.cpp mode, inference/control refuse redirects; release requires
exact router status `unloaded`. Explicit adoption binds selected models and
matching resources to the original shared coordinator. Keep the dedicated
external endpoint quiescent or use a separate device for explicit native
maintenance. Shared locks coordinate participating services. Preserve journal
ownership and recovery before endpoint, topology or lifecycle changes. Separate
GPUs keep the native OCR owner and confirmed park without requiring LLM model
management. See
[backend setup and adoption](OPENAI_COMPATIBLE.md).
Settings reads expose authentication state without credentials. Admin secret
writes are origin-bound. Provider, host and effective authentication changes
need restart. Model, temperature and registration apply live.
`AUTH__SECRET_KEY` is required, optional pepper otherwise falls back to it. Empty
users table permits bootstrap admin and legacy ownership backfill. Registration
default true, admin-wide view default false. Preserve auth/session boundaries.

Lifespan creates provider/services, initializes DB, bootstraps admin, starts analysis
and benchmark workers when ready; shutdown stops workers before closing DB.
SPA fallback is registered after API routes. DB initialization failures are sanitized.

Current migration head `0025`. Stop/drain old app/benchmark/standalone workers
**before** schema, queue or policy cutover; never run old/new workers concurrently. Alembic uses settings `PG__*`, including on the image entrypoint;
`MIGRATION_DATABASE_URL` is only the grant tool input. Bootstrap missing group roles
as admin/CREATEROLE, execute grants/default ACL as the actual migration owner,
migrate with owner PG credentials, refresh grants, then run app with runtime login.
An already-current schema is prerequisite for runtime-credential entrypoint startup.
Never imply automatic migration/runtime credential switching or destructive downgrade.
See the public upgrade runbook for backup, legacy mapping and compatible backout.

## Development and verification

`make help` is the command index. `make setup` synchronizes the Python environment;
use a private `UV_PROJECT_ENVIRONMENT` in shared worktrees and do not sync a shared
`.venv` symlink. `make migrate` applies Alembic; `make run` serves dev API on 8080.
`make fe-install` uses npm ci; `make fe-dev` proxies 5173→8080; `make fe-sync` builds
and copies static assets. Do not change locks/dependencies as a prose edit.

Prepared bundles are the end-user path; source Compose is advanced development.
See [distribution maintenance](distribution/README.md) for five external-LLM
variants, private app bootstrap, paired and standalone file-only preparation, passive
standalone checks of prepared or qualified state, optional explicit physical qualification,
retained PID1 crash fences and exact-source artifact gates. Native source contexts
are independent of the app. The [native guide](paddleocr-service/README.md#local-qualification-inputs)
documents guarded source qualification.
No real native images/bundles were published or physically qualified here.

Canonical checks: `make lint` (Ruff/mypy on app/native/build/e2e/harness/scripts),
`make test` (non-E2E pytest), `make fe-verify` (Svelte/lint/format/Vitest/E2E types/build),
`make release-test`, `make distribution-test`, `make distribution-pid1-test`,
`make compose-check`, `make test-e2e`, `make fe-test-e2e`.
E2E uses `e2e/run.py` to allocate unique projects, ports and private fixture copies.
Its app configuration is seeded in a project-owned volume. Cleanup is scoped to
runner-owned test resources. `make e2e-up/down` requires explicit
`E2E_STATE`; no shared fixed stack may be adopted. Playwright can use
`VB_E2E_BASE_URL` for an intentionally supplied stack. Stubs validate known original
and normalized hashes; synthetic tests are not model-quality/hardware evidence.
[Fixture rules](tests/e2e/data/README.md) and [accepted counts/limits](UPGRADE_AND_VERIFY.md#verification-evidence)
state the evidence boundaries. CI verifies software; manual release is separate.

Pre-commit hooks run lint/unit/frontend checks. Documentation-only verification
should follow the assigned scope; never claim unrun broad gates. Prototype scripts
named in the scripts guide have no supported Make targets. Keep real fixtures and generated artifacts
private and out of Git/build contexts. Do not contact placeholder external APIs.

## Edit files
- The edit tool matches oldString byte-for-byte. Read the file again immediately before each edit, and again after every successful edit line content shifts.
- Copy oldString verbatim from the file you just read. Do not retype, reindent or normalise it. Preserve tabs, trailing spaces and blank lines.
- Keep oldString to the smallest span that is still unique — usually one or two lines, not a whole block.
- After two failed edits on the same file, stop and report. Do not switch to rewriting the file.
