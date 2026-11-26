# Uploads, multi-photo receipts and review

Use one to ten ordered photos (at most 20 MiB each) of one receipt. Include header
and total and a few overlapping rows between adjacent photos; reorder before
upload. Repeated purchases elsewhere remain distinct.

## Durable acceptance and polling

`POST /api/v1/images` accepts repeated multipart `receipt` fields. **Single and
multi-photo uploads both return 202**, after committing originals and submission
identity. `Location: /api/v1/images/<id>` identifies the authenticated status
resource. The response ID is the submission/image UUID, not a completed receipt.
A disconnected client can resume by ID; `/upload?job=<id>` survives reload.
Queue shows pending/processing/errors; completed receipts appear in Receipts.

Query parameters include `model_id` and `bypass_review`. Deprecated
`allow_ocr_fallback` is accepted for compatibility and request identity only; it
does not grant ordinary image fallback. An `Idempotency-Key` is owner-scoped:
replaying the same
ordered original hashes and explicitly supplied options returns the same ID;
changed content or intent returns 409. Omitting a flag differs from explicitly
sending false. The first resolved defaults are frozen, so changing server settings
does not reinterpret a replay. Generated paths and thumbnails do not determine
request identity. In the browser, reselect the same ordered photo bytes to
recover an ambiguous submission. Saved review preferences appear after those
photos match. Changing the review choice records a new intent and preserves
the other matched options. After a file-read error, retry the local photo check
before submitting.

Read `status`, `terminal`, `generation`, `attempt_count`, `max_attempts`, `retry_at`,
`wait_reason`, `result_code`, `draft_available`, `observations_available`,
`receipt_id` and `source_views`. Readiness checks the exact requested or configured
LLM model. Provider availability, an incompatible OCR mode, missing supported
model capability and a changed handoff binding can hold work before inference.
A pre-work wait leaves the attempt budget available. Terminal
states are `analyzed`, `needs_review`, `failed`, `timed_out`, `unreadable`.
EN/DE UI explanations distinguish waiting from failed attempts; raw codes remain
secondary diagnostics. Acceptance is not a claim that extraction succeeded.

## Attempts and explicit recovery

Each whole attempt has a 600-second maximum including OCR, discovery, formatter
and repair work; defaults allow three automatic attempts with 30/60-second
backoff, configurable cap 300 seconds and a 900-second claim lease. Lower configured
deadlines apply. Generation/claim tokens fence stale completions and expired leases.
Unusable readiness is checked before claiming. Retryable failures retry within the
bound; unreadable input is terminal. A later photo failure never saves a successful
prefix as a completed receipt or all-photo checkpoint. A validated exact OCR prefix
may be retained/reused across retries; formatting requires every ordered photo.

For `analysis_incomplete`, the authenticated `GET /api/v1/images/<id>` gives
the `latest_extraction_id` of the most recently recorded attempt. Read
`GET /api/v1/images/<id>/extractions/<latest_extraction_id>` and check
`attempt_number`, `envelope.findings[].code` and `envelope.telemetry.finish_reason`.
A provider-reported `length` finish indicates that generation reached a limit.
An absent finish reason leaves the cause open. Extraction records can contain
private receipt content. The application log event `analysis_no_draft` summarizes
the recorded call count, timing and token usage using bounded diagnostic categories.

`POST /api/v1/images/<id>/retry` takes `expected_generation` and a nonempty
`request_key`. Replaying that key returns its original outcome; a new retry must
use the current generation and a new key. Conflicts return 409. Retrying preserves
sources and frozen intent by default (`policy: "preserve"`). New uploads bind the
provider and its credential-free endpoint fingerprint for direct vision and OCR
formatting, including separate GPUs. Changes to that destination, OCR routing,
or historical image-fallback permissions can hold pending work with
`wait_reason: "execution_policy_changed"` before claim. Pending inference from
older releases, including rows without a snapshot, also requires explicit
adoption because its original LLM destination is unknown. Inference-free
completion can still use an existing checkpoint that proves the original
destination and matches current policy. To adopt current server policy
explicitly, send `policy: "current"` with the current generation and a new
request key. The server preserves the original request/fingerprint and old effective
policy in history, increments generation and clears incompatible execution/checkpoint
state. Retry-key replay binds the policy choice too. Live claims, linked receipts
and human-edited/finalized drafts cannot be overwritten; completed receipts are
never reprocessed. Stop and drain workers before a policy cutover. Existing
submissions retain their frozen settings until explicit adoption.

## Extraction and source evidence

Vision Bill uses the operator's external LLM endpoint and selected model.
Direct vision sends all ordered photos when OCR is disabled and requires a
model that accepts images. When OCR is selected by server configuration, it
processes each photo with the fixed native profile. Formatting, JSON repair and
classification use text and structured evidence only. OCR failure cannot switch
an upload to image inference. Requested and actual model IDs, digests and OCR
profile/runtime remain recorded. Deprecated fallback fields and old results
retain their historical meaning. Admins can explicitly select direct vision
for a benchmark comparison.

Ollama and OpenAI-compatible endpoints receive the same field instructions and
nullable draft schema. In OCR mode the formatter receives the ordered transcript,
layout blocks and permitted source references for each photo. A product name,
its size or colour and a price on the following line can describe one purchase.
The extraction retains every separately priced occurrence. The draft editor
and saved receipt can combine matching purchases at the same unit price and tax
band by adding the quantities and line amounts. Combined rows retain the
source references of their occurrences. Unresolved photo overlaps and
unreadable prices remain available for review.

Vision Bill validates the returned JSON before preparing the review form.
For example, a printed 19% belongs in the tax-rate field as `0.19`. If the
model returns `19`, the formatter can ask once for just that field using the
same evidence. Live formatting corrections share up to 60 seconds of the
original attempt and reserve time to save validated observations. A targeted
field correction uses up to 2,048 output tokens. An unusable correction can
still leave a validated draft for manual review. The application log records
the model phase, runtime, token counts where returned, stop reason and
validation category. Editable drafts suggest categories and tags for recognized
products alongside any labels already supplied.

Missing totals can be calculated when the captured rows and adjustments identify
the complete charge. VAT needs a rate and an unambiguous base for every relevant
tax band. An unallocated discount across several tax bands can leave those
amounts unresolved while the gross subtotal remains calculable. Calculated
values carry an informational review notice with source evidence. Readable
printed totals keep precedence.
[Review and financial contract](UPGRADE_AND_VERIFY.md#review-and-financial-contract)

Only model-proposed ordered suffix/prefix overlaps between adjacent photos may be
removed. `helper/duplicates.py::verify_duplicate` checks description, quantity and
line total (exact defaults, normalized case/whitespace in descriptions). Every pair
must pass, and both quantities must be observed. If either quantity is missing,
the pair stays in the receipt. No global deduplication or blanket image splitting is applied.

Original bytes, hashes and photo/part identities are retained. Working PNGs have
their own IDs, dimensions, hashes and transform metadata. EXIF normalization is
independent of learned orientation classification. Transparent pixels are composited
on white (`white-document-rgb-v2`). New extractions leave geometric unwarping off.
OCR boxes use the stored working image’s coordinates. Older extractions may
contain an unwarped working image. Review them with their original transform
metadata and geometry. The inverse mapping field remains `null`.
Authenticated `/file?part=N` serves originals; `/views/<view_id>` serves validated
working PNGs with private/no-store caching. Review displays both. The app and
native service share bounded PNG, JSON and text handling for photos up to
40,000,000 pixels. See the [native resource limits](paddleocr-service/README.md#fixed-extraction-policy).

## Complete observations and optional labels

An all-photo result is validated and checkpointed before optional classification.
Incomplete drafts and successful prefixes keep their separate recovery paths.
The extraction record exposes `classification_enrichment` with a state, a reason
and missing field paths where applicable. Optional receipt categories, item
categories and tags can be unavailable or skipped. Older records can have
`not_recorded` status. Missing labels remain separate from financial and integrity
findings. Safely settled enrichment failures preserve the valid receipt.

An unresolved request or teardown can hold publication with
`wait_reason: "model_request_unsettled"`. The checkpoint retains the complete
observations and provenance. When ownership has been safely reconciled, the
scheduler can finish the same generation from that checkpoint even after the
last allowed inference attempt. It validates the original and working bytes,
frozen execution settings and current review state. Completion issues no new OCR,
formatter or classification request, spends no inference attempt and adds a
separate history event. An explicit retry into a new generation is a different
operation.

For an unmanaged classification hold, inspect from an application environment
using the same configured inference endpoint as the held submission:

```sh
python -m vision_bill.distribution.handoff reconcile-classification \
  --image-id "$SUBMISSION_ID"
```

After independently confirming that the previous request at that endpoint has
settled, provide the exact generation and checkpoint ID from inspection together
with the attestation:

```sh
python -m vision_bill.distribution.handoff reconcile-classification \
  --image-id "$SUBMISSION_ID" --expected-generation "$GENERATION" \
  --checkpoint-id "$CHECKPOINT_ID" --request-settled
```

A stale identity, busy claim or mismatched checkpoint blocks the operation.
Managed holds use the dedicated endpoint and app journal recovery procedure.
See [managed recovery](paddleocr-service/README.md#internal-model-ownership).

## Drafts, checks and deletion

Printed extraction observations can be null, including currency and amounts;
classification/category/tags are downstream enrichment. A partial draft is saved
for correction through `GET/PATCH /api/v1/images/<id>/draft` with
`expected_revision`; finalization takes `expected_revision` and `request_key`.
Missing required values block finalization. A receipt can be finalized with a
blank discount, tip, quantity or unit price when the printed information is
absent. Each line still needs a description and a total. Where both quantity
and unit price are known, their product is checked against the line total.

In the review form, **Cash** in English and **Bar** in German both save
`cash`. Confidence accepts whole percentages from 0 through 100. **Save
draft** stores the edits and refreshes the findings for that revision. A
field link in the findings opens the relevant input. Use **Finalize draft**
after the saved revision meets the required checks. If saving fails, the
form keeps your edits available for correction.

Line-item tag suggestions use the selected interface language. Choosing
**Kaffee** in German or **Coffee** in English records `coffee`. Switching
languages relabels built-in tags on existing receipts. Custom names outside
the built-in catalog retain their spelling.

The text classifier receives the valid receipt categories, item spending
categories and current tag vocabulary. For built-in tags it sees both German
and English terms. It can assign several relevant tags to one item. Both
interface languages use the same codes, so changing the language keeps
earlier classifications searchable. Printed descriptions and trader names
remain in their source language.

The editor keeps the merchant's printed name and address alongside an editable
display name. German chains have a built-in directory covering common trading
names. An identified municipal kiosk can be grouped under **Kiosk**, while
a named kiosk or another independent business can keep its own name. Select
an existing merchant or type a new one. Finishing the draft or explicitly
saving or verifying your own receipt learns the choice. Automatic analysis,
benchmark runs and saving an unfinished draft do not teach your directory.
New uploads can reuse your previous merchant choices. A private directory is
scoped to its account. Merchant suggestions use the authenticated
`POST /api/v1/merchants/suggestions` with the lookup text in its JSON body.

Receipt review uses checks version 3. See the
[amount policy and human acknowledgement contract](UPGRADE_AND_VERIFY.md#review-and-financial-contract).
**Verify** and **Save & verify** fetch the current server preview. Enter comments
for the displayed review findings. The request binds these acknowledgements to
the exact finding IDs and the operation and review revisions. Integrity failures block even with an acknowledgement.
Previously bypassed, legacy or older-version verification can receive explicit
human review of the current receipt. Use the current `GET /api/v1/receipts/<id>/verify` preview and
submit both expected revisions and acknowledgements to the matching POST.
Automatic bypass requires complete observations, clean current checks and
identified originals. Verification preserves original and working files. Benchmark
Gold qualification
requires separate human confirmation of the current document fingerprint and
review revision.

Resources and mutations are owner-scoped; admins cross owners only with
`AUTH__ADMIN_CAN_SEE_ALL=true`. Account switches invalidate old UI requests,
queries, notices and view state. The browser binds JSON requests with
`X-Vision-Bill-Expected-User` and private media URLs with `vb_expected_user`.
An actor mismatch returns 409 before the protected operation. The signed cookie and
ordinary authorization rules determine access. Deliberate API clients may omit
the actor expectation and act as the cookie user. Deletion first commits authorized
database removal and a durable owned-file manifest, then removes validated
originals, thumbnails and working derivatives and syncs their directories. An
authorized user can discard an unfinalized submission that needs review from the
queue. An active analysis or a submission linked to a receipt blocks queue deletion.
Failed cleanup stays pending. Repeat the same authorized DELETE to finish it. See the
[upgrade and recovery runbook](UPGRADE_AND_VERIFY.md).
