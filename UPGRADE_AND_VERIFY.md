# Unreleased package: upgrade and verification

This guide covers durable receipt processing, strict optional OCR, managed OCR/LLM
handoff and prepared deployments with an external LLM. The verification table records
the software baseline and component carry boundaries. See
[backend setup, ownership and changes](OPENAI_COMPATIBLE.md). The original PR14 optional
OCR integration informed this package. PR15's splitting experiments and
historical results remain separate evidence. The native profile uses document
orientation, unwarping and a 2,000,000-pixel per-region maximum.

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

For a source installation created with `ocr_source_up.py`, use the original
checkout and the documented `--upgrade` procedure after backup and drain.
`--restart` reuses the previously recorded images. [Source upgrade steps](paddleocr-service/README.md#upgrade-an-existing-source-installation)

For a prepared upgrade, keep the old Compose project name and volume mounts.
Use a schema-v2 bundle for the new installation. Changing the extraction directory
does not require new volumes. Copy the private runtime file or use a private
empty `runtime.env` if the saved configuration already specifies the intended
endpoint. The app reuses credentials, settings, receipts and uploads on these
volumes. The retired LLM model volume is separate and stays operator-managed.

If a saved configuration points to a removed bundled Ollama hostname, the app
retains that selection. Pending work waits until you deliberately configure a
reachable endpoint and installed model or restore the old reachable service.
Reconcile owned work on the old endpoint before changing identities, then adopt
the current endpoint and policy for eligible unfinished jobs as needed.
See [prepared cutover](distribution/README.md#stop-cut-over-and-recover) and
[managed endpoint setup](OPENAI_COMPATIBLE.md#adopt-the-managed-pair).

Keep the existing app image/source identity, configuration, auth secret/pepper,
PostgreSQL roles and ownership, original uploads (including legacy temporary
paths), working artifacts and thumbnails. A database-only backup is insufficient.
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
The resulting head is **0021**. Before starting uvicorn with runtime credentials,
complete owner migration and grants. The current image still executes Alembic at
startup using the runtime `PG__*` identity; it can only check an already-current
schema, not apply pending owner DDL. Future upgrades must repeat the owner step.
Do not temporarily leave the app serving as the owner to simplify migration.

### Fresh and legacy mapping

Fresh installations apply 0001 through 0021. Existing supported 0010 installations
apply migrations 0011–0021 for workflow, review, benchmark references,
claim fencing, deletion, observation checkpoints and optional receipt values.
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
decimal fraction before finalizing, such as `0.07` for 7%.

Migration 0021 permits these nulls without updating existing receipt amounts.
Its downgrade refuses to erase their meaning after new nullable values are saved.

`AmountPolicy` states tax basis (`gross/net/unknown`), subtotal origin
(`printed/derived_net/unknown`), tax information (`present/explicit_zero/missing`),
discount treatment (`included_in_lines/subtract/unknown`) and tip treatment
(`included/add/outside_receipt/none/unknown`). Unsupported/unknown conventions
produce findings, not a forced net-plus-tax equation. With supported policy the
line sum checks subtotal (subtract tax for a gross derived-net subtotal); total
adds tax for net/derived-net, subtracts discount only for `subtract`, and adds tip
only for `add`. Gross derived-net with separate discount/tip can be uncertain and
is flagged rather than guessed. Explicit zero tax conflicts with a nonzero tax.

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

Admin reference preview/qualification lives at
`/api/v1/benchmarks/references/<receipt_id>` (GET then POST with
`expected_review_revision` and `expected_document_sha256`). Gold requires current
ordinary human verification, exact acknowledgements and complete available ordered
originals with matching hashes. Legacy/bypass verified flags are insufficient.
The immutable reference records observed-field mask, expected data, review/schema/
scoring versions, owner/actor identity and source evidence. Edits require current
qualification; run creation checks the current reference again transactionally.

Admin runs explicitly select models/receipts and freeze source/model digests,
profile/runtime, prompt/schema/source/output/context metadata and policy. Direct
vision needs vision capability; OCR needs text formatting;
normal upload follows frozen OCR configuration. Replay must match every ordered
photo/hash/profile/runtime and retained artifact; no expected/reference values enter
inference. Ordinary OCR is strict text/evidence-only, including failures, repairs
and classification. Direct vision comparison is an explicit admin pipeline choice;
legacy fallback fields/results are compatibility/history, not fresh upload permission.
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

The reviewed runtime source is `2ab3efe6a7ee765c0c2b61f89c4426e822eb397b`.
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

Actual official CUDA wheel-lock resolution, CUDA/ROCm native compilation, physical
GPU inference/qualification, current historical CPU image installation/inference,
quality/latency/VRAM measurements remain **UNEXECUTED** here. CPU stubs and source/
CMake checks certify software contracts only. No VRAM/performance blocker was
established, and universal AMD support is not claimed. Intended Linux consumer
scope is RX 6000+/RDNA2 and later selected candidates, each requiring its own real
qualification. See [native prerequisites and commands](paddleocr-service/README.md).

For future changes, canonical gates are `make lint`, `make test`, `make fe-verify`,
`make release-test`, `make distribution-test`, `make distribution-pid1-test`,
`make compose-check`, `make test-e2e`, `make fe-test-e2e`.
Use isolated environments and unique owned E2E stacks. Safe documentation checks
can run help/parsers, local links and synthetic effective Compose without launching
services; they cannot replace real operator/hardware acceptance. Publishing remains
a separate [manual release](RELEASES.md) action; no tag/release/deployment is implied.

### Admin formatter settings and environment defaults

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

A real private API key is bound to its trusted endpoint origin: browser host edits
may change paths within that origin, but cannot send the key to another origin.
With an empty or `none` key, a host edit is allowed and records the original
`llm_credential_origin` in private YAML. Adding a real key later refuses startup
when the effective endpoint differs from that origin. An explicit operator
rotation of the deployment host takes precedence over stale browser host adoption
so a newly rotated key follows its deployment endpoint. The operator must verify
the new endpoint and approve the credential binding in private configuration
before restarting; removing the origin marker after that review permits the
verified endpoint. Never put keys in a host URL.
