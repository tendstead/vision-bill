# 4. Cherry-picks from PR #16 (OCR and more)

PR #16, "Use external LLM endpoints with optional PaddleOCR-VL 1.6"
(David Hupperich, 188 commits, branch `codex/vision-bill-update-package`), is
**not** an OCR-only change. It is a large package on top of the Python backend.
Nothing can be merged or rebased; every wanted part is re-implemented in Go.
Credit rules: see the [README](README.md). The inventory below was made from
`git diff --stat`, the Alembic migrations and a few key files, not from reading
the whole diff.

Base of the PR: `144a58e7` (the last Python release state). Size: about 115k
added lines, 567 files.

## Where the lines are

| Area | Added lines (files) | What it is |
| --- | ---: | --- |
| `tests/` | 42.8k (149) | Python unit and e2e tests; ports of tests are only useful where the feature is ported |
| `frontend/` | 18.3k (154) | Svelte UI: benchmarks (cases, comparison, references), queue diagnostics, review, settings, OCR source viewer |
| `paddleocr-service/` | 16.9k (79) | the optional PaddleOCR service (about 3.3k are lock files) |
| `src/vision_bill/` | 15.6k (80) | Python backend: workflow DB (2k), LLM provider rework (1k), submissions, GPU hand-off, review, dedicated OCR, merchants |
| `scripts/` | 15.3k (37) | OCR harness (5.5k, with 3.3k lock file), source-up tool, distribution tooling and tests, standalone benchmark |
| `alembic/` | 0.6k (17) | migrations `0011`-`0025` |
| `distribution/`, workflows, compose files, docs | about 5k | prepared deployment bundles (CUDA/ROCm compose files, release workflows), `UPGRADE_AND_VERIFY.md`, `OPENAI_COMPATIBLE.md`, `HAENDLERVERZEICHNIS.md` |

## Features, by relation to plans 1-3

Already covered by our plans (listed there as **PR #16 overlap**; port with
credit): paired benchmark comparison, benchmark references and cases, benchmark
method/results, durable queue with attempts and retry requests, upload
idempotency, the public submission status, review drafts.

Other features, to decide (adopt / adapt / skip):

| Feature | Where | Notes |
| --- | --- | --- |
| **External LLM endpoints / connection probing**: bounded model discovery for a saved connection, statuses `connected`, `authentication_failed`, `unreachable`, `invalid_response`, `origin_mismatch`; llama.cpp provider | `src/vision_bill/provider/llm/connection.py`, `llamacpp.py`, `OPENAI_COMPATIBLE.md` | Overlaps with plan 1 step D (named connections). Strong candidate: the probe statuses and limits (5 s, 2 MiB, 4096 models) are worth porting. |
| **Optional OCR pipeline** (PaddleOCR-VL, OCR then LLM, vision fallback off) | `paddleocr-service/`, `src/vision_bill/provider/ocr/`, `service/extraction_pipeline.py`, `dedicated_ocr.py` | The original goal of the PR. Benchmarks currently report `ocr_available: false`; our manifests reserve `pipeline: "ocr"`. Keep optional, off by default. Port after plans 1-2 so it can be benchmarked. |
| **GPU/model hand-off** between OCR and the LLM on one GPU | `service/gpu_handoff.py`, `model/handoff.py`, `distribution/handoff.py` | Only needed with managed OCR on a shared GPU. Defer. |
| **Review drafts** (extraction ends in an editable draft, review history, overlap decisions, finalize with request key) | `api/drafts.py`, `service/receipt_review.py`, migrations `0012`, `0013`, `0015`, `0018` | Large product change (status `needs_review`). Decide together with plan 3 (the `review` state for consumers). |
| **Merchant directory** (German retailers: normalised display name + category suggestions) | `HAENDLERVERZEICHNIS.md`, `model/merchant_catalog.py`, `provider/db/merchants.py`, migration `0025` | Self-contained and useful without OCR. Good small, early pick. |
| **Everyday product tags** | `model/product_tags.json`, migration `0024` | Small; check against the current tag vocabulary. |
| **Receipt amounts**: optional unprinted `discount_total`, exact decimals | migrations `0014`, `0021` | Go already stores exact decimals; check the nullable `discount_total`. |
| **Distribution bundles** (prepared installs for CUDA/ROCm, release workflows) | `distribution/`, `scripts/distribution*.mjs`, `.github/workflows/distribution-*.yml`, compose files | Separate topic from the Go server; probably out of scope for now. |
| **Settings preservation, execution policy** | `tests/test_settings_preservation.py`, `service/execution_policy.py` | See plan 1 (options in the manifest). |

## Database compatibility

**`main` is the only source of truth for the schema.** PR #16's Alembic
revisions `0011`-`0025` are ignored: David works with a test server only, there
is no production database on that branch, so no adoption path for it is built
(the Go server keeps adopting revision `0010` only, `UPGRADING.md`). A test
database that ran his branch is recreated. Ported features get new additive Go
migrations (`003`, `004`, ... in the order they land), **not** a replay of the
Alembic files; migration numbers in his files are only a reference for what a
feature needs.

## Suggested process for the session

1. Ask David which parts he considers finished and which are experimental (the
   OCR part is reported as not working yet).
2. Walk the tables above and mark each row adopt / adapt / skip.
3. One branch and PR per adopted row, smallest first (merchant directory,
   connection probing), each with the credit trailer.
4. Release as minor versions while in 0.x.

## Notes

- A running prototype stack exists on the development machine
  (`vision-bill-pr16-rocm`: database, ROCm formatter, llama.cpp, paddleocr); its
  app container is stopped.
- The local branch `pr16` (fetched from `origin/pull/16/head`) can be used for
  `git show pr16:<path>` while porting.
