# 2. Stale and failing queued images

## Problem

- An image that can never be processed (file missing, undecodable, always
  invalid model output, always too slow because it is too large) stays in the
  queue.
- Timeouts are retried with a doubling delay since 0.2.0 (`timed_out`,
  `images.retry_after`), but `failed` is retried every worker cycle forever and
  nothing ever gives up. Nothing may run indefinitely.
- The queue shows only the last error. A retry (automatic or via "Analyze now")
  leaves no visible trace: no attempt count, no next retry time, no history.
- The header queue badge refreshes only on navigation.

## Already in place

- Timeouts are non-final with doubling delay (10 min ... 6 h).
- `images.analysis_attempts` (counts timeouts only) and `images.retry_after`.
- Oversized JPEG/PNG photos are shrunk for the model (3072 px).
- `unreadable` is the only final status.
- The queue page polls every 15 s and after "Analyze now".

## PR #16 overlap (David Hupperich) - align before building

PR #16 reworked the analysis queue (Python, Alembic `0011`, `0019`, `0020`).
It is the closest match to this plan, so names and semantics should line up.
Credit him (see the README).

| PR #16 feature | Where | Relation to this plan |
| --- | --- | --- |
| Per-image `attempt_count`, `max_attempts` (1-3, default 3), `retry_at`, `result_code`, `wait_reason`, `generation`, lease (`claim_token`, `lease_expires_at`) | migration `0011_durable_analysis_jobs`, `SubmissionStatus` in `src/vision_bill/model/workflow.py` | Same data we plan (`attempts`, `max_attempts`, retry time). **Use his field names in the API: `attempt_count`, `max_attempts`, `retry_at`, `result_code`, `wait_reason`.** Our 0.2.1 columns `analysis_attempts`/`retry_after` stay as the storage names and are mapped in the API (or renamed by a later additive migration). |
| Queue UI: "attempt N of M", "next try at ...", the wait reason, a result code with a "previous attempt" label | `frontend/src/routes/queue/+page.svelte`, `frontend/src/lib/diagnostics.ts`, `queue.test.ts` | This is exactly the state line we planned. Port the labels and diagnostic code lists instead of inventing new ones. |
| **Retry request**: `RetryRequest{expected_generation, request_key, policy: preserve|current}`, stored in `image_retry_requests` so a repeated request is idempotent | migration `0011`, `0019_execution_policy_retry_identity`, `src/vision_bill/api/images.py` | Our `POST /images/{id}/retry`. Adopt `expected_generation` (optimistic concurrency, the UI cannot retry an image that changed meanwhile) and the idempotent `request_key`. `policy` (retry with the frozen or the current settings) belongs to the OCR/execution-policy part; decide in plan 4. |
| Statuses: adds `needs_review` (draft awaiting human review) | `images_status_check` in `0011` | Our "gave up" is a timestamp, not a status, so the constraint stays untouched. Adding `needs_review` (plan 4) means relaxing the constraint: safe for old code (it never writes it) but an explicit decision. |
| Upload `Idempotency-Key` per user | `ux_images_owner_upload_key` in `0011`, `src/vision_bill/api/images.py` | Needed for plan 3 as well. |

What PR #16 does **not** have, so it remains ours: a per-attempt history table
(`image_attempts`), the explicit "gave up" state with a reason, permanent-error
classification, and the oversized-image shrink. Ask him why `max_attempts` is
capped at 3 before fixing our default of 5.

Plan adjustments that follow from this:

- API field names follow PR #16 (see the table).
- `POST /images/{id}/retry` takes `expected_generation` and a `request_key`.
- Decide the default `max_attempts` (3 in PR #16, 5 here) together.

## Design

### Data (migration `004_image_attempts.sql`, additive)

- `images.gave_up_at timestamptz NULL` and `images.gave_up_reason text NULL`.
  The status stays one of the existing values (`failed`, `timed_out`) because
  the `images_status_check` constraint is fixed and the previous release must
  keep working. "Gave up" is `gave_up_at IS NOT NULL`; such images are no longer
  claimable.
- `images.last_attempt_at timestamptz NULL`.
- New table `image_attempts(id, image_id -> images ON DELETE CASCADE,
  started_at, finished_at, outcome, error, duration_ms, model_id)`: one row per
  attempt. This is the answer to "I only see the last failure".
- `analysis_attempts` now counts every failed attempt, not only timeouts.

### Rules

- One shared retry policy for `failed` and `timed_out`: doubling delay from a
  base (default 5 min, cap 6 h), `retry_after` set after every failed attempt.
- Maximum attempts (`worker.max_attempts`, default 5; environment and YAML like
  the other worker settings). After the last failed attempt the image gets
  `gave_up_at` and a reason.
- Permanent errors give up immediately, no retries: image file missing,
  unsupported or undecodable content. The reason says so.
- Timeouts give up after the maximum with a reason that says what to do:
  "too large or too slow: use a smaller photo or raise
  `llm.analysis_timeout_seconds`".
- A provider outage is not an attempt: the image is never claimed while the
  model is unreachable (current behaviour, keep).
- No automatic deletion. Gave-up images wait for the user (privacy: nothing
  disappears silently). An optional admin bulk action "delete all gave-up
  images" is cheap to add.

### API

- `GET /images` rows and `GET /images/{id}` get `attempts`, `max_attempts`,
  `last_attempt_at`, `retry_after`, `gave_up_at`, `gave_up_reason`.
- `GET /images/{id}/attempts`: the attempt history (owner or admin).
- `POST /images/{id}/retry`: clears `gave_up_at`/`retry_after`, resets the
  attempt counter, triggers the worker; owner or admin; `409` if the image is
  being processed or analysed.
- `POST /images/analyze` (existing): wakes the worker; images still waiting for
  their delay are not forced (a user must not be able to hammer the model);
  `retry` is the explicit per-image override.

### UI

- Queue rows show a state line instead of only the error: "Retrying, attempt 2
  of 5, next try 14:32", "Processing", "Gave up after 5 attempts: too large",
  with a button "Retry now" and "Delete".
- Expandable attempt history per image (time, outcome, error, duration).
- "Analyze now" shows progress (the queue page already polls) and the header
  badge refreshes on an interval and after mutations, not only on navigation.
- Texts in `en` and `de`.

## Test plan

- Store: attempt counting, backoff, give-up transition, claimable excludes
  gave-up images, retry resets.
- Analysis: permanent vs transient errors, the attempt rows written.
- HTTP: new fields, `retry` permissions and conflicts, scoping.
- API e2e: a broken model and an always-slow stub end in "gave up".
- Playwright: state line, retry now, history, badge refresh.

## Order of work

1. Migration, store, scheduler policy, tests.
2. API fields and `retry`/`attempts` endpoints, openapi.
3. UI.

## Open questions

- Default maximum of 5 attempts and base delay 5 min: acceptable?
- Should a gave-up image notify (snackbar/badge colour) the owner?
