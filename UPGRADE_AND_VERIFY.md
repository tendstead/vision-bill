# Unreleased package: upgrade and verification

This guide covers durable receipt processing, strict optional OCR, managed OCR/LLM
handoff and prepared deployments with an external LLM. The verification table records
the software baseline and component carry boundaries. See
[backend setup, ownership and changes](OPENAI_COMPATIBLE.md). The original PR14 optional
OCR integration informed this package. PR15's splitting experiments and
historical results remain separate evidence. The native profile uses document
orientation classification with geometric unwarping disabled and a 2,000,000-pixel
maximum per recognition region. Previous installations could unwarp photos.
Retain their stored originals, working images, OCR outputs and model assets.
Each saved working view and its coordinates describe the run that produced it.

## Resulting processing order

1. Authenticate, validate one to ten ordered originals, retain their bytes/hashes,
   resolve upload intent and commit the submission. Return 202 plus `Location`.
2. Wait for the frozen pipeline's usable model/OCR capabilities before spending an
   attempt. Resolve/freeze actual execution identities; claim with generation,
   token and lease fencing.
3. Normalize document pixels and extract all photos, using direct vision when OCR
   is off or the explicitly enabled native sidecar when OCR is selected. Persist
   exact working views and validated exact-prefix OCR checkpoints; formatting and
   complete receipt output require all ordered photos.
   Managed OCR first confirms release of owned LLMs and later confirms native OCR
   worker exit before any LLM formatting request.
4. Format nullable printed observations with the selected LLM. Reconcile only
   verified adjacent overlaps. Save complete observations before optional
   classification, preserving original descriptions and printed amounts. When
   labels cannot be obtained after safe settlement, keep the observations and show
   the classification state. An uncertain request holds work for reconciliation.
5. Save an immutable extraction and editable draft/receipt, evaluate source,
   overlap and financial checks, then require human correction/acknowledgement as
   necessary. Clean bypass can verify, but cannot qualify benchmark Gold.
6. Explicitly qualify current human-reviewed Gold before freezing a benchmark
   dataset and running live extraction or validated replay against it.

The accepted source reviews corrected failures that synchronous/verified-only
instructions concealed: upload acknowledgement now follows commit; readiness waits
do not exhaust attempts; later-photo failure cannot save a prefix as a completed
receipt; OCR never grants ordinary image fallback; nullable observations survive
classification; financial checks expose uncertainty without changing printed money;
review and retry use revision/generation fences; deletion can resume owned cleanup;
Gold freezes current human evidence rather than trusting a legacy verified bit.
The additional audits repaired idle native recovery, selected AMD target profiles,
transparent thumbnails, account/view lifetimes, localized findings, role/connection
identity seams, isolated build inputs and truthful script/comparator behavior.
These are correctness changes, not measured OCR accuracy or speed improvements.

## Backup and upgrade order

For a base source installation running only Vision Bill and PostgreSQL, the
first local OCR registration defaults to waiting for Settings activation.
The explicit `--defer-build` flag selects the same behavior. Use the original
checkout and Compose project. Before updating a still-waiting first registration, stop
its host launcher, update that checkout, and rerun the same `--defer-build`
command. A revision change refreshes the app image before waiting for activation.
Once the managed OCR installation exists, use `--restart`
to start the recorded images again. [First source registration](paddleocr-service/README.md#start-vision-bill-first-and-build-ocr-from-settings)

For a source installation created with `ocr_source_up.py`, use the original
checkout and the documented `--upgrade` procedure after backup and drain.
An app-only update can reuse a recorded native image when its image ID, recipe,
source and runtime locks, accelerator inputs and physical GPU pass verification.
The launcher repeats the offline binding check. Older records without build
evidence require one native build and verification. A shared runtime-lock change
can trigger this once for either accelerator. Keep the model volume and, on CUDA,
the verified wheel and lock snapshots.

A stopped, settled original deployment with its records, release and volumes intact
can rebuild deleted app or OCR images through `--upgrade`. The read-only handoff
check uses a separate pinned CPU image. `--restart` needs the recorded images.
If the native build succeeded before the app build failed, retry with the
saved native image tag if its OCR inputs match the updated checkout. [Source upgrade and retry steps](paddleocr-service/README.md#upgrade-an-existing-source-installation)

For a CUDA source rebuild, keep the selected channel, resolved lock,
attestation and private wheel cache, including its `build-…` snapshots. A
snapshot can verify a saved native image even if the checkout has since built
another channel. Source installations recheck the recorded CUDA GPU UUID when
present. For ROCm, use the original GPU.
On hosts with several AMD GPUs, set its Linux render node or its WSL HIP
device index and GFX target. To adopt the corrected SDK target lists and
BuildKit checkpoints, rebuild the ROCm image once with `--upgrade`. Keep the
previous image tag and build evidence with the backup. [Linux and Windows/WSL upgrade commands](paddleocr-service/README.md#upgrade-an-existing-source-installation) · [GPU selection and build inputs](paddleocr-service/README.md#build-inputs)

For a prepared upgrade, keep the old Compose project name and volume mounts.
Use a schema-v2 bundle for the new installation. Changing the extraction directory
does not require new volumes. Copy the private runtime file or use a private
empty `runtime.env` if the saved configuration already specifies the intended
endpoint. The app reuses credentials, settings, receipts and uploads on these
volumes. The retired LLM model volume is separate and stays operator-managed.

To verify the formatter after upgrading, open **Settings → Receipt formatter**
as an admin and use **Test saved connection**. Check the returned model IDs
against the intended service. Discovery sends no receipt and runs no inference.
The exact model ID can also be entered manually. Both providers support a
credentialless connection or a Bearer key saved through Settings.

If a saved configuration points to a removed bundled Ollama hostname, the app
retains that selection. Pending work waits until you deliberately configure a
reachable endpoint and installed model or restore the old reachable service.
Reconcile owned work on the old endpoint before changing identities, then adopt
the current endpoint and policy for eligible unfinished jobs as needed.
See [prepared cutover](distribution/README.md#stop-cut-over-and-recover) and
[managed endpoint setup](OPENAI_COMPATIBLE.md#adopt-the-managed-pair).

Keep the existing app image/source identity, configuration, auth secret/pepper,
PostgreSQL roles and ownership, original uploads (including legacy temporary
paths), working artifacts, thumbnails and the `benchmark_cases/` directory
under the configured image save directory.
A database-only backup is insufficient.
Use protected storage and test restoration to an isolated database/filesystem
before changing the installation. Freeze writes or take a consistent coordinated
snapshot so DB references and files agree. Preserve existing exact amounts; this
upgrade does not recalculate historical values or automatically re-OCR receipts.

**Stop/drain all old analysis and benchmark workers before migrating or starting
new queue logic.** Do not run old and new workers concurrently. Stop the app while
keeping PostgreSQL available for migration/backup. The entrypoint automatically
runs Alembic, so an ordinary `compose up` is already a migration operation.

The helper accepts a trusted shell-compatible private environment file. It resolves
the configured PostgreSQL image, uses custom-format `pg_dump`, refuses overwrite
and restores in one transaction. A private env must describe an actually reachable
host/network; the base Compose database has no published host port. For example,
with `PG__HOST=db`, `PG__PORT=5432` in that file and the base network available:

```bash
bash scripts/pg-backup.sh --backup /private/backup.dump \
  --env-file /private/backup.env --network postgres_net
# Restore to a prepared isolated target described by a different private env:
bash scripts/pg-backup.sh --restore /private/backup.dump \
  --env-file /private/restore.env --network isolated_restore_net
```

`--replace` explicitly drops objects present in the backup before restore; use it
only for an intentionally replaceable target. Backup/restore need genuine database
privileges and separately provisioned login/group roles. No operator backup,
restore, migration or service command was run while writing this guide.

### Credentials and grants

Use a genuine schema owner/migration login, a separate app login in
`vision_bill_runtime`, and a reporting login in `vision_bill_readonly`. The app
requires writes and cannot use the reporting login. Neither runtime nor reporting
login should be an owner or superuser. The initial Compose `POSTGRES_USER` is a
bootstrap database identity, not automatic provisioning of this role separation.

Creating missing NOLOGIN groups in `scripts/configure_db_roles.sql` requires an
administrator/CREATEROLE identity. An ordinary NOCREATEROLE migration owner cannot
bootstrap missing groups. Have the database administrator create them first:

```sql
CREATE ROLE vision_bill_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
CREATE ROLE vision_bill_readonly NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
```

Then run the unchanged grant procedure **as the actual Alembic object owner**, with
`MIGRATION_DATABASE_URL` and `PG__DB` set in a protected `.env.scripts`:

```bash
make db-configure-roles
```

The SQL grants CONNECT/schema USAGE, runtime SELECT/INSERT/UPDATE/DELETE and
readonly SELECT on all current tables, runtime USAGE/SELECT on sequences, and
matching default privileges for future tables/sequences created by that owner.
Run it before migration to establish defaults and again afterwards for new objects.
An administrator provisions separate login passwords and grants the appropriate
group. Readonly reporting logins may also set `default_transaction_read_only=on`
and a bounded `statement_timeout`.

**Alembic uses `PG__USER`, `PG__PASSWORD`, `PG__DB`, `PG__HOST`, `PG__PORT` from
application settings. `MIGRATION_DATABASE_URL` belongs only to the Make grant tool;
it does not switch entrypoint/Alembic credentials.** Set actual owner `PG__*`
credentials in a protected migration environment before running:

```bash
uv run alembic current
uv run alembic upgrade head
uv run alembic current
```

Run from a prepared checkout/environment on a network that reaches the intended
DB. Clear conflicting process/dotenv settings and use a private
`VISION_BILL_CONFIG_PATH` as appropriate: process env > dotenv > YAML precedence
still applies. No DSN/password needs to appear in public commands or shell history.
The resulting head is **0025**. Before starting uvicorn with runtime credentials,
complete owner migration and grants. The current image still executes Alembic at
startup using the runtime `PG__*` identity; it can only check an already-current
schema, not apply pending owner DDL. Future upgrades must repeat the owner step.
Do not temporarily leave the app serving as the owner to simplify migration.

### Fresh and legacy mapping

Fresh installations apply 0001 through 0025. An installation at 0021 applies
0022 through 0025. An installation at 0010 first applies 0011–0021 for workflow,
review, benchmark references, claim fencing, cleanup and optional receipt values.
Migration 0011 preserves linked receipts, verified flags, amounts and historical
paths and marks their submissions analyzed; they are not re-OCRed. Receipt-free
legacy failed/processing images become pending with attempt_count 0/max_attempts
3, retaining prior status/error/processing history in `legacy_events`. Historical
benchmark attempt counts that cannot be known are not invented by migration 0017.
0014 retains exact Decimal/NUMERIC values; 0018 supplies durable owned deletion
cleanup. Migration 0019 keeps retry policy choice for replay-safe explicit adoption.
Migration 0020 adds durable observation checkpoints for completion-only recovery.
Migration 0021 allows absent discounts, quantities and unit prices while preserving
stored values. A downgrade to 0020 stops if those columns already contain nulls.
Migration 0022 creates owner-scoped benchmark cases and distinguishes their
tasks from ordinary receipt tasks. Existing tasks keep their receipt identity
and deletion protection. Migration 0023 records the comparison group and
pipeline on each paired run. Existing single runs retain their original history.
Migration 0024 adds everyday product tags. Existing vocabulary rows and receipt
labels retain their identities. A downgrade to 0023 leaves vocabulary entries in place.
Migration 0025 adds an optional display name and an account-scoped merchant
directory with aliases. It leaves historical printed names and human corrections
intact. Existing receipts can acquire a display name through normal review.
Downgrading to 0024 refuses to discard stored merchant display names, aliases
or directory entries. Use the compatible app or restore the coordinated backup
to a separate installation when a rollback would lose saved choices.

A downgrade from 0023 to 0022 refuses if a comparison exists. A downgrade
from 0022 to 0021 refuses if any case exists, including an unconfirmed draft.
If either guard fires, keep the data and use a compatible app version, or
restore the coordinated pre-upgrade backup into an isolated target.

After head/grants, start only the compatible new application. Verify login and
owner/admin scope, polling of a new synthetic upload, a preserved old receipt and
its sources/amounts, draft review, and an explicitly qualified synthetic benchmark
reference. Linked old receipts require no extraction. Keep private receipts and
raw traces out of general logs/reports.

### Disable, pause and recover

Keep workers stopped during a migration failure; fix owner credentials/grants or
restore the coordinated backup to an isolated target before cutover. For a safe
pause, stop the app/workers and retain database/files. To disable OCR for **new**
uploads, set the canonical `OCR__ENDPOINT=` explicitly blank, ensure legacy
`OCR__REQUIRED=false`, and restart deliberately. A nonblank canonical endpoint
selects OCR and takes precedence over legacy `enabled`/`service_url`. Without it,
legacy `OCR__ENABLED=false` can coexist with a retained URL without enabling OCR.
Changing the LLM provider or endpoint, including for direct vision or separate
GPUs, holds previously queued inference before claim. Pending inference without
a saved endpoint fingerprint (upload policies through version 3) also waits for
explicit adoption. Changed OCR routing and historical fallback grants can
likewise hold work.
Use an authorized retry with current `expected_generation`, new `request_key` and
`policy: "current"` to adopt current policy explicitly. Default `preserve` retains
intent. History keeps the original request and old effective snapshot; human-edited
drafts, live claims and completed receipts cannot be overwritten. This is not a
queue rewrite or historical reprocessing. Never start an old worker on the new
queue/schema and never automate a
destructive Alembic downgrade. Data-bearing downgrade guards intentionally refuse
loss. Backout needs a compatible image or explicit restore of the coordinated
pre-upgrade database/files, with new workers stopped and post-backup writes handled.

Deletion commits database removal plus an owner-scoped durable manifest before
unlinking files. If cleanup is interrupted, retry the same authorized image/receipt
DELETE; missing already-removed files are acceptable, changed identities/symlinks
are not. Cleanup metadata remains until all owned cleanup succeeds. This is not a
general background sweeper. Active or still-referenced resources return conflicts;
benchmark references can prevent receipt deletion. Do not manually remove arbitrary
paths or assume every failed HTTP response means the DB deletion rolled back.

### Recover from a complete observation checkpoint

When the saved complete observation matches the source and execution state, a
completion-only retry can publish the reviewed receipt after ownership is
reconciled. This uses the existing generation and inference-attempt count.
It makes no new OCR, formatter or classification request. A failed optional
classification leaves usable observations and records the label outcome. A
request whose delivery or settlement is uncertain keeps the checkpoint on hold.

New benchmark runs use prompt version 7. Historical runs keep their results
and recorded prompt version. Reconfirm a qualified receipt reference when the
extraction schema changes. Changing a merchant display name preserves the
receipt dataset fingerprint. The app benchmark can match separately extracted
purchases to a grouped receipt reference when their quantities, unit prices
and amounts agree. Dedicated benchmark cases retain their annotated rows.

New complete observation checkpoints use formatter contract version 6 and
extraction schema 3, including optional position tax evidence. Completed receipts
and saved drafts retain their values. An older complete checkpoint waits with
`execution_policy_changed`. To extract again, explicitly retry with current
policy after reconciling any outstanding model request. The retry archives the
old observation and starts a new generation. Pending uploads without a complete
observation use the installed formatter once their saved routing policy permits
inference. See [multi-photo receipts](MULTI_IMAGE_RECEIPTS.md#extraction-and-source-evidence)
for calculation and correction behavior.

For an unmanaged classification hold, inspect and reconcile the exact submitted
image using the configured endpoint identity. In a prepared bundle use the
`app` service. Source Compose uses `vision_bill`.

```bash
image_id=your-image-uuid
docker compose exec app python -m vision_bill.distribution.handoff \
  reconcile-classification --image-id "$image_id"
# After independently confirming the previous request has settled:
docker compose exec app python -m vision_bill.distribution.handoff \
  reconcile-classification --image-id "$image_id" \
  --expected-generation "$generation" --checkpoint-id "$checkpoint_id" \
  --request-settled
```

Use the exact generation and checkpoint ID from the status result. If a managed
owner has an outstanding request, use the original endpoint and coordination
storage for recovery. See [native ownership](paddleocr-service/README.md#internal-model-ownership).

## Review and financial contract

Checks version **3** separates `integrity` blockers from review exceptions.
Required missing observations, unavailable/changed originals and invalid source
references prevent finalization/verification. Unknown source mapping and unresolved
overlaps require review. Observations remain nullable; human correction fills
actual missing mandatory data before finalization. Drafts can save partial edits.
A clothing receipt might print “T-shirt” on one line and “SKU 123 · M · blue ·
12.90” on the next. Together these describe one item charged 12.90. If the
same shirt is independently charged a second time, keep both items.

For both new and migrated receipts, **Verify** and **Save & verify** fetch
a preview for the current revision. Enter a comment for each current review
finding in the dialog. A clean **Save & verify** completes immediately.
API clients fetch the same preview through `GET /api/v1/receipts/<id>/verify`
and submit its revisions and acknowledgements to the matching POST.

Receipt and draft responses encode monetary amounts, line quantities and tax
rates as decimal strings. Write requests accept strings and JSON numbers. Clients
that use binary floating point should keep the strings when editing or resubmitting
printed values. Discount, tip, quantity and unit price can remain absent on the finalized receipt.
Transaction and line totals remain required. Correct a present tax rate to its
decimal fraction before finalizing, such as `0.19` for a printed 19%. A prior
observation of `19` stays visible until corrected.

Migration 0021 permits these nulls without updating existing receipt amounts.
Its downgrade refuses to erase their meaning after new nullable values are saved.

`AmountPolicy` states tax basis (`gross/net/unknown`), subtotal origin
(`printed/printed_net/derived_net/unknown`), tax information (`present/explicit_zero/missing`),
discount treatment (`included_in_lines/subtract/unknown`) and tip treatment
(`included/add/outside_receipt/none/unknown`). Unsupported/unknown conventions
produce findings, not a forced net-plus-tax equation. With supported policy the
line-price basis describes the items. Choose `printed` for a printed subtotal
on the same basis as those items, or `printed_net` for an explicitly printed
net subtotal for the entire transaction. A net amount for one VAT band can
cover only part of the receipt. Choose `derived_net` when the full net subtotal
was calculated from sufficient evidence. A net subtotal under gross lines is
compared with the line sum less the printed tax. Separate discounts and tips can have an uncertain tax basis and trigger a finding.
Only `subtract` deducts a discount and only `add` adds a tip. Explicit zero tax conflicts with a nonzero tax.

Decimal comparisons flag differences **greater than** 0.01 for quantity×unit price
versus line total when both factors exist, 0.05 for subtotal, 0.02 for tax sum
and 0.05 for total. These are
review tolerances in recorded monetary units, not score tolerances or permission
to round stored values. V4 exact amount metrics forgive no monetary discrepancy.
Local receipt time accepts valid `HH:MM`, `HH:MM:SS` and up to six fractional second
digits; preserve seconds/microseconds. Invalid clock values are rejected.

Use `expected_operation_revision` and `expected_review_revision` from current
receipt review state. Every current review finding needs exactly one
`{finding_id, comment}` acknowledgement with a nonblank comment; stale, extra or
duplicate IDs are invalid. The operation locks/rechecks current state; stale CAS
returns conflict. Integrity findings cannot be acknowledged away. Changes can
invalidate findings/revisions and require fresh human review. Bypass can auto-verify
only with no current findings and identified originals; it is not human verification
or Gold qualification. Verify updates DB/history and preserves file paths.

The browser sends `X-Vision-Bill-Expected-User` on protected requests and
`vb_expected_user` on private media links. A different authenticated account
returns `409 session_actor_changed` before the affected operation. Intent
checks complement normal session and owner authorization.

## Benchmarks and reproducibility

### Prepare the inputs

In **Benchmarks**, search existing receipts by merchant or receipt number,
select the intended entries, then open the reference preview. Check the
expected values and the originals before confirming them. The picker respects
the current administrator's receipt scope. Admin reference preview/qualification lives at
`/api/v1/benchmarks/references/<receipt_id>` (GET then POST with
`expected_review_revision` and `expected_document_sha256`). Gold requires current
ordinary human verification, exact acknowledgements and complete available ordered
originals with matching hashes. Legacy/bypass verified flags are insufficient.
The immutable reference records observed-field mask, expected data, review/schema/
scoring versions, owner/actor identity and source evidence. Edits require current
qualification; run creation checks the current reference again transactionally.

For a separate test photo, open **Upload photos and manage benchmark cases**
in the benchmark form. Choose **New case**, enter a name and select one to ten
photos in reading order, up to 20 MiB each. Uploading
creates a private case immediately. It schedules zero OCR or LLM calls. The
ordinary upload queue, receipt ledger and spending reports receive no entries.
Enter the printed values you want to test. Leave unobserved amounts blank and
enter a zero only when the reference should assert zero. Save a draft to
continue later. Check the values against the displayed originals and confirm
the reference before choosing **Use in benchmark**. Editing the reference
requires confirming the new revision. Earlier runs keep their captured reference.

The admin API accepts multipart `name` and repeated `photos` fields at
`POST /api/v1/benchmarks/cases` (201 with Location). Read case detail at that
location, and read each ordered original at `/photos/{zero-based index}`.
`PATCH` the case with its current `expected_revision`, the `expected` draft
and an explicit `confirm: true` for confirmation. A stale revision returns 409.

### Choose and compare pipelines

The benchmark form lists models from the active formatter endpoint. After changing
provider, host or authentication in Settings, restart the app when prompted.
Click **Refresh models** to query the active endpoint again after models or
capabilities change. Select one or more eligible model IDs explicitly. If an
ID disappears on refresh, remove it from the selection before starting the run.

The optional run temperature accepts values from 0 to 2, including an explicit
zero. Leave it blank to use the active formatter temperature at queue time.
Benchmark selections take effect without a restart and apply only to that run.

Choose a mode, then select the models to run. **Image directly to LLM** sends
the originals to the selected vision models. **OCR + LLM** uses the dedicated
OCR service and sends the resulting text and evidence to selected text-capable
models. OpenAI-compatible models that explicitly advertise audio-only input
or embedding-only output are excluded from the OCR list. Older endpoints can
still offer models without capability metadata for text formatting.
Each choice creates a single run. **Compare both** explicitly requests
two linked runs over the same ordered inputs and frozen expectations. It needs
OCR and at least one model that supports both paths. A selected text-only
model participates on the OCR side, with the direct side marked inapplicable.
Each model and input on the OCR side runs its own OCR step in the live worker.
A worker claims one model/input task at a time. The order in which you click
model names does not set the execution order.

For an API run, send `POST /api/v1/benchmarks` with explicit `model_ids`,
`receipt_ids` and/or `case_ids`, and `experiment.pipeline` set to
`direct_vision` or `ocr`. To pair them, set `comparison: true` with
`direct_vision`. The 202 response and Location identify the direct run.
Read both entries with `GET /api/v1/benchmarks?comparison_id=<group UUID>`.
A single-run API request that omits a pipeline follows the configured upload policy.

Results show the average over all terminal cases for each eligible model and
pipeline. A failed extraction counts as zero. Open cases make the average
provisional. The difference appears when the model finishes every input on both
sides, both runs have the same dataset fingerprint and scoring version, and
neither run has failed. It is the OCR average minus the direct average, in percentage points. A move from
65% to 85% is +20 percentage points. Open individual results to inspect the
expected and actual values, missing or extra items, errors and timing. Loading
or refreshing results reads recorded work. Explicit reevaluation executes the
selected model again. Runs containing independent cases use scoring version
5 for every input, including qualified receipts in the same run. For example,
a reference containing only a total gives that total the entire extraction score.
When a reference includes item rows, missing or extra rows count even if some
cells were left blank. Optional classification has a separate score. New
receipt-only runs use version 4.

Admin runs freeze source/model digests,
profile/runtime, prompt/schema/source/output/context metadata and policy. Direct
vision needs vision capability; OCR needs text formatting;
normal upload follows frozen OCR configuration. Replay must match every ordered
photo/hash/profile/runtime and retained artifact; no expected/reference values enter
inference. Ordinary OCR is strict text/evidence-only, including failures, repairs
and classification. Direct vision comparison is an explicit admin pipeline choice;
legacy fallback fields/results are compatibility/history, not fresh upload permission.
New Gold selections under extraction schema 2 require fresh reference confirmation.
Previously qualified references and recorded benchmark results remain available
in their original runs. Create a new run to execute or reevaluate a case from
prompt version 4. A queued case bound to that version fails before model inference.
Changed model digests wait rather than silently drifting.
Observed masks prevent absent reference observations from being treated as truth.

V4 reports full-receipt correctness, occurrence-aware ordered item precision/recall,
exact item count/order, exact amounts/taxes and failed-case denominators. The full-receipt correct rate and all-case average count completed failures in
the denominator. The historical success-only average and v3 weighted score
remain separate measures. Frozen Gold is not a population accuracy claim.
The [standalone CLI](scripts/README.md) requires an explicit owner `--user-id` and
uses a consistent read-only snapshot; it cannot benchmark all owners by omission.
Historical CPU transcripts use a separate explicit adapter contract, never fake
native qualification/geometry. See the [harness guide](scripts/ocr_harness/README.md).

## Verification evidence

The following historical verification covers runtime source `2ab3efe6a7ee765c0c2b61f89c4426e822eb397b`.
These are local software results. Public CI for the published commit is separate.
Repeated executions of the same case are not added to the counts below.

| Gate | Observed result and exact boundary |
| --- | --- |
| Python/static | 1,750 passing tests, 17 host PID1 skips and 416 deselections at `41995d4`. Ruff and mypy passed. Backend source is unchanged through `2ab3efe`. |
| Frontend | 258 passing tests in 33 files at `2ab3efe`, with static checks and production build passing. |
| Browser E2E | Full `41995d4` run passed 102 of 103 cases. The upload replay fix at `2ab3efe` passed the previously failing case and all 28 scoped cases on the rebuilt image. |
| Backend E2E | 51 API cases, 45 OCR policy cases, two build-context cases and eight discovery cases passed across the verified runs at `41995d4`. The OCR suite reran the 15 cases affected by a local port collision. |
| PostgreSQL | 317 of 318 cases passed at the same backend baseline, `bbbc06b`. The last case passed with a corrected test observer at `c60f993`, alongside controlled reproduction and repeat checks. |
| Release/distribution | 14 release tests, 27 distribution tests, three Compose configurations and 69 CPU container PID1 checks passed at `41995d4`, with those components unchanged through `2ab3efe`. |
| Prepared app | The exact `2ab3efe` image passed fresh and restored schema 0020 installs with a credentialless or authenticated OpenAI-compatible test endpoint. The renamed bundle reused the project and volumes, retained credentials, settings and a synthetic receipt, and accepted ordinary registration. |

Local application tests used synthetic receipts and CPU service doubles. See
[current branch CI](https://github.com/Peckerhacker10203/vision-bill/actions?query=branch%3Acodex%2Fvision-bill-update-package)
for the published commit's conclusions.

The November 21 WSL 2 CUDA checkpoint-loading observation reached worker
readiness, then the receipt request returned HTTP 504. Software gates cover
contracts, packaging and simulated lifecycle behavior. Neither that run nor CI
establishes completed receipt quality, throughput, peak VRAM or qualification
across the intended NVIDIA Turing+ and AMD RDNA2+ device families. Release
maintainers record the selected physical device for each published native
variant. Additional boards or hosts need separate evidence to claim qualification. [Native evidence and prerequisites](paddleocr-service/README.md)

For future changes, canonical gates are `make lint`, `make test`, `make fe-verify`,
`make release-test`, `make distribution-test`, `make distribution-pid1-test`,
`make compose-check`, `make test-e2e`, `make fe-test-e2e`.
Use isolated environments and unique owned E2E stacks. Safe documentation checks
can run help/parsers, local links and synthetic effective Compose without launching
services; they cannot replace real operator/hardware acceptance. Publishing remains
a separate [manual release](RELEASES.md) action; no tag/release/deployment is implied.

### Admin formatter settings and environment defaults

New installations default to 32768 formatter output tokens per generation. Source
Compose on Linux and WSL reads the existing checkout’s `.env` on container
creation. If it pins `LLM__MAX_OUTPUT_TOKENS=8192`, change that line to
`LLM__MAX_OUTPUT_TOKENS=32768` to adopt the new budget, or leave your chosen
value in place. Recreate the app with the same Compose project and overlays
during the upgrade. Existing installations can also retain a value in the mounted
YAML under `llm.max_output_tokens`. Settings shows the effective formatter output
limit after startup.

Existing `.env` and container values continue to supply settings until an admin
changes a field and clicks **Save** in Settings. Changed provider, HTTP(S) host,
model, and temperature fields are saved in the private mounted YAML.
An allowlisted `admin_setting_overrides` list records only adopted environment
fields; those YAML values then take precedence across backend restarts. Registration
remains locked when intentionally controlled by environment. An unchanged field is never adopted merely because the full form was submitted.
Secrets, API keys, database configuration and managed distribution identity remain
under their existing controls and are never copied from the environment by this save.

Model and temperature changes apply to subsequent work immediately. Provider or
host changes retain the existing restart contract: Settings keeps a warning visible
until the operator restarts the backend. The browser does not restart containers or
receive Docker credentials. To return an adopted field to environment control, an
operator can remove its path from `admin_setting_overrides` in the private YAML and
restart the backend; preserve the private mount and its permissions.

Settings submits only edited field paths, so another open admin tab cannot overwrite
unrelated settings with stale form values. Older API clients may still send a full
PUT; these legacy requests retain environment locks. Adopting an environment
default requires an explicit `changed_paths` mask, as sent by Settings. Each formatter save refreshes Dedicated OCR status so a binding or restart
requirement remains visible.

Origin bindings protect credentials when the endpoint changes. An admin can
edit the path on the same origin. To move a key saved in Settings to a new
origin, select the new host and explicitly supply its key, then restart the app.

For inherited Ollama authentication, set `LLM__HOST` and `OLLAMA_API_KEY`
together in the deployment and recreate the app service. An explicitly rotated
deployment origin takes precedence over a stale host previously adopted in
the browser. An explicit **None** or **Bearer API key** selection in Settings
continues to control that connection.

For legacy OpenAI-compatible deployments, a host edit with an empty or `none`
key records the original `llm_credential_origin` in private YAML. If an added
deployment key would go to a different browser-selected origin, startup
requires operator review. Verify the destination before updating the private
binding. A host and key explicitly rotated together in the deployment take
precedence over a stale browser host. After recreating the app, verify the host
with **Test saved connection**.
