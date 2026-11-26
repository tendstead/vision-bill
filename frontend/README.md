# Vision Bill frontend

SvelteKit/TypeScript SPA with static adapter, Tailwind, TanStack Query and EN/DE
localization. FastAPI serves the built SPA with client-route fallback; `/api`
keeps API routing. Docker builds the frontend from source.

From the repository root:

```bash
make fe-install       # npm ci, including the required patch-package postinstall
make fe-dev           # :5173, /api proxy to :8080
make fe-verify        # Svelte/ESLint/Prettier, Vitest, E2E types, production build
make fe-sync          # build and copy frontend/out into src/vision_bill/static
make fe-test-e2e      # complete Playwright run in a unique owned app/stub/DB stack
```

`npm run preview` inside `frontend` previews the static build. It still needs an
API endpoint; preview does not start PostgreSQL or inference. `make fe` exposes
other package scripts, for example `make fe CMD=typecheck:e2e`.

Routes cover login, uploads, queue, receipt/draft review, collections, search,
statistics, benchmarks and settings. Uploads use durable 202 acceptance and status
polling. `/upload?job=<id>` resumes after reload. After an ambiguous response, select
the same photos in the same order to recover the original request key and options.
The saved review choice appears after the photo bytes match the request for the
current account. A read failure offers a local retry before submission. Changing
the review choice creates a new request and preserves the other saved options.
Review preserves exact decimal strings, ordered row identities, original/current descriptions and local receipt times
including valid seconds/microseconds. Source views distinguish original and working
coordinates; unknown inverse transforms remain null. EN/DE findings explain checks2,
readiness waits, policy holds and explicit current-policy retries. Both editors
show the printed seller and an editable merchant display name. They use localized
receipt categories, item categories and multi-select tags. The shared tag
dictionary is a build input from the adjacent Python package in this checkout.
The upload UI
follows the server's OCR configuration. Historical fallback fields are retained for
request replay. See [the upload and review workflow](../MULTI_IMAGE_RECEIPTS.md).

Auth requests are serialized and guarded by session generation. Account changes
clear queries and remount authenticated views. Protected browser requests send
`X-Vision-Bill-Expected-User` and private media URLs carry `vb_expected_user`.
The server rejects a mismatch with the authenticated cookie. Delayed replies stay
bound to their originating account and view. Backend ownership remains authoritative.

Verification uses the revision of the displayed receipt. Receipts awaiting a human
re-review require a fresh preview. Gold selection requires explicit human
qualification of the current document and sources. Existing benchmark runs retain
their frozen references.

In the admin **Benchmarks** form, select existing qualified receipts or open
**Benchmark cases** to upload ordered photos outside the receipt ledger and
analysis queue. Confirm expected values against the originals. Leave unknown
fields blank and enter zero or confirm empty lists only when observed.

Choose **Image directly to LLM**, **OCR + LLM**, or **Compare both** and
select exact model IDs. The OCR options require configured OCR. A comparison
runs both pipelines on the same inputs for vision-capable models and runs
text-only models on the OCR arm. Runs containing dedicated cases use v5 to
score confirmed fields. Receipt-only v4 results also show full-receipt
correctness. The all-case score includes failures at zero, and the historical
success-only average is displayed separately.

Admin **Settings** includes a GPU setup and recovery guide for Dedicated OCR.
Download and extract the offline operator guides ZIP to follow local links
between manuals. For a checkout, open `paddleocr-service/README.md` for local
GPU selection and build settings. For a downloaded release, open
`distribution/README.md` for its image and launcher instructions.
The individual Markdown guides are also available. Local
frontend builds and the dev server use Python 3 to assemble the ZIP. Edit the [source guide](../paddleocr-service/README.md)
and [bundle guide](../distribution/README.md) and rebuild the frontend to
refresh the downloads.

Keep private receipts, traces, Playwright outputs and local dotenv files out of Git
and builds. `.dockerignore` excludes generated test reports and credentials;
[the shell patch](patches/README.md) remains a required build input.
