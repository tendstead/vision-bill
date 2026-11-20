# Split receipt extraction experiment

This is an experimental runner, separate from the application and from the OCR
integration PR. It compares one local Gemma model on the same private receipt
photos. It does not persist receipts, initialize a database, or change uploads.

```sh
uv run python scripts/receipt_split_experiment.py \
  --dataset server_data/ocr-harness/paired/dataset.json \
  --ocr-report server_data/ocr-harness/truenas-photo-results/report.json \
  --output server_data/split-experiment/photo-comparison
```

The dataset manifest contains `cases` with unique `id`, ordered `images`, and an
`expected` Receipt JSON path. Paths resolve relative to the manifest. Real data
and generated artifacts must stay under the gitignored private directories.
The OCR report is the local harness's existing JSON report; the runner validates
source image hashes before reusing successful PP-medium transcripts. No OCR
engine is installed or run by this script.

Modes:

- `vision_all`: all photos in one request, existing extraction prompt and overlap
  reconciliation schema from the application.
- `vision_parts`: one request per photo with a nullable partial-receipt schema;
  merge the extracted pieces in code.
- `ocr_all`: all cached photo transcripts in one request, merged by the model.
- `ocr_parts`: one partial extraction per cached photo transcript, merge in code.
- `vision_tiles`: split each photo into horizontal crops of at most 1,000 pixels
  with 150 pixels of overlap, extract independently and merge in code. Opt in
  using `--modes vision_tiles`; this mode is not a default.
- `ocr_rows`: split cached photo transcripts into at most 16 recognized text
  lines with three lines of overlap, using the same partial prompt and merger.
  Opt in using `--modes ocr_rows`; adjust with `--ocr-rows`. A text line can be
  a header or quantity continuation, not necessarily a purchased item.

The endpoint must be same-machine Ollama (loopback). Default model is
`gemma4-12b:latest`; its actual digest is recorded. Requests are sequential,
temperature zero, thinking disabled, and generation capped at 8,192 tokens per
attempt. A request has 180 seconds for up to three validation attempts. Unsplit
jobs have 180 seconds total; split jobs have 600 seconds total. These limits are
explicit experimental choices, **not equivalent compute budgets**. Report both
completion and total wall time, and check whether split successes also finish
within 180 seconds. Cached model requests and replayed OCR are not fresh timing
measurements. Input loading and report generation are outside the job deadline.

The partial prompt changes alongside the input grouping. This evaluates complete
strategies; it does not isolate grouping alone. Fields absent from a partial
photo should be null. Required missing fields fail final Receipt validation.
Receipt-level totals are selected from the last non-null piece, never summed.
The first non-null header is retained; disagreements are recorded.

Merging only compares the suffix and prefix of adjacent accumulated parts.
At least two consecutive rows must match descriptions, quantities, unit prices,
and line totals exactly after description whitespace/case normalization to
propose removal. Single-row matches are retained and flagged. This can still
remove legitimate identical purchases or miss overlaps after OCR errors, so
**every merged candidate requires human review**. It is not ready for automatic
persistence. No reference value is used in extraction or merging.

Private output includes:

- `index.html`: source images, parsed partial results, merge decisions, final
  candidates, scoring diagnostics, and expandable raw model requests/replies.
- `calls/*.json`: cached raw replies, validation attempts, model timing/token
  telemetry and input/prompt/model/source fingerprints; no image base64.
- `results.json` and `summary.json`: per-case and aggregate comparisons.
- `crops/*.png`: actual tile inputs when testing `vision_tiles`.

The weighted score reuses detailed experiment scoring, excludes classification
and payment method (the reference payment method can reflect user knowledge),
and counts failures as zero. Exact totals, item counts, and all ordered item
quantities/unit prices/line totals are reported separately. These ten selected
receipts are a regression sample, not a population accuracy estimate.

Before production integration, replace heuristic overlap decisions with stronger
source-position evidence, explicitly handle ambiguous/conflicting fields and
missing parts, test cancellation/retry behavior, and add regression coverage.

[receipt_split_results.md](receipt_split_results.md) is the retained historical
selected-case report. Its numerical values and score meanings have not been
rewritten to represent current v4 Gold evaluation. This PR15 experiment remains
separate from stabilized PR14 optional OCR integration; neither source import is
claimed as a remotely merged PR. Production uses validated adjacent overlaps and
durable all-photo extraction, not blanket tiles/rows/heuristic merging. The colleague
attachment supplied settings only; no colleague project logic is used. See the
[current workflow and evidence](../UPGRADE_AND_VERIFY.md).
