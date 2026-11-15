# Local split-extraction experiment — 2026-11-15

**Conclusion: do not enable automatic splitting from this prototype.** Smaller
pieces bounded generation on the problematic receipt, but the assembled results
still contained extra items and missing fields. Splitting by photo alone did not
fix its timeout. The application and upload behavior are unchanged.

## Setup

Branch `experiment/split-receipt-extraction`, based on updated main `144a58e`.
One configured local Ollama model (`gemma4-12b:latest`, digest beginning
`596b21d708ed`), temperature zero, thinking disabled, 8,192 generated tokens per
attempt. Sequential requests, 180 seconds per request including up to three
validation attempts. Combined jobs have 180 seconds total; split jobs have 600
seconds total. These are different maximum job budgets; all targeted chunk
requests completed without needing the larger job allowance beyond 180 seconds.

The private regression sample contains ten verified receipts, fourteen prepared
photos and 106 expected items (1–32 per receipt). Images were already upright and
limited to a 2,000-pixel longest side. Both strategies use those same inputs.
This is a deliberately selected small sample, not a population accuracy estimate.

The standalone runner reuses main's receipt models, full-receipt extraction
prompt, overlap reconciliation and detailed scoring. It does not exercise the
HTTP upload endpoint or persist receipts. Partial extraction uses a different,
nullable schema and prompt, so this compares complete strategies rather than
isolating image grouping alone. No references are supplied to extraction or
merging. PP-OCRv6 Medium transcripts are replayed from the earlier private OCR
comparison, checking each source photo's SHA-256.

These are fresh main-based comparisons with the standalone runner's shared
request/repair loop. They are not a rescore of the earlier OCR PR's service-path
benchmark, which used different prompt/repair behavior. Compare strategies within
this table rather than interpreting the earlier 80.0% versus 78.3% as a regression.

## Ten-receipt comparison

| Strategy | Valid receipts | Exact totals | Same item count | All ordered item numbers exact | Weighted score |
|---|---:|---:|---:|---:|---:|
| Vision: all photos together | 9/10 | 9/10 | 7/10 | 5/10 | 64.9% |
| Vision: independent photos + merge | 8/10 | 8/10 | 5/10 | 4/10 | 56.4% |
| OCR: combined transcripts | 10/10 | 10/10 | 8/10 | 4/10 | **78.3%** |
| OCR: independent transcripts + merge | 5/10 | 5/10 | 4/10 | 2/10 | 37.5% |

“Valid” means the final candidate passed the application receipt schema, not that
it was correct. Failures count as zero. The weighted score uses existing detailed
field scoring, excluding classification and payment method: the verified payment
method can reflect the user's knowledge rather than printed evidence. It is not
the percentage of entirely correct receipts. “All ordered item numbers exact”
requires matching item counts and every quantity, unit price and line total in
order. Numbers above are for schema-valid final candidates only.

Several incomplete candidates retained usable items but could not satisfy the
final schema, commonly because `subtotal` was null. The literal partial prompt
intentionally prohibits inventing missing printed totals. This exposes a missing
finalization policy; it does not imply that every receipt field or item was lost.
Missing merchants, conflicting headers and actual item errors also occurred.

## Targeted finer chunks

The 32-item, three-photo receipt caused the repeatable timeout. The other targeted
receipt has 25 items in a single photo. Image crops are at most 1,000 pixels high
with 150 pixels of overlap. OCR chunks contain at most sixteen recognized text
lines with three lines of overlap. Text lines can include headers and quantity
continuations rather than individual products.

| Receipt | Strategy | Pieces completed | Model/job seconds | Candidate item count | Final schema valid |
|---|---|---:|---:|---:|---|
| 32 items / 3 photos | Image crops | 9/9 | 176.2 | 41 | No: missing subtotal |
| 32 items / 3 photos | OCR text chunks | 9/9 | 173.5 | 38 | No: missing subtotal |
| 25 items / 1 photo | Image crops | 3/3 | 58.4 | 28 | Yes |
| 25 items / 1 photo | OCR text chunks | 5/5 | 160.9 | 27 | No: missing subtotal |

All four strategies completed their pieces within 180 seconds total, but none
recovered the correct item count or all ordered item numbers. OCR time is excluded:
recognized text was cached. Timings describe this serial local run with warm
server caches; they are not end-to-end production latency measurements.

## What the traces showed

- The difficult receipt's unfinished responses hit the 8,192-token output cap.
  Single-photo and single-transcript attempts produced more than 100 item entries;
  the subsequent repair ran into the overall deadline. Image size alone therefore
  does not explain the observed failure: excessive/repeated generation matters.
- Those receipts contain legitimate repeated purchases. A global duplicate remover
  or a rule deleting repeated products would corrupt the receipt.
- Cropping bounded the requests, but overlap could not be resolved reliably from
  item text and prices. On the 25-item receipt, overlapping product rows differed
  between extractions in descriptions, quantities and/or prices. The conservative
  matcher retained them, leaving 28 items. Loosening matching would risk deleting
  genuine purchases or choosing the wrong price.
- The nullable partial schema prevents fabricated metadata but needs a separate
  finalization policy for required fields such as unprinted net subtotals. This
  prototype deliberately leaves those cases invalid and requiring review.

The next design should preserve physical source-row identity/position through
extraction, resolve disagreements explicitly, and handle header/totals separately
from bounded item groups. Keep the ordinary path for receipts that already work;
this experiment does not justify a blanket split policy.

## Reproduction and verification

See [runner instructions](receipt_split_experiment.md). Private fixtures, images,
transcripts, raw replies, per-attempt telemetry, source snapshots, merge decisions,
and the self-contained stage report remain under the ignored
`server_data/split-experiment/` directory and are not distributed.

The matrix contains forty unique jobs; the targeted tests add four. The original
long command was interrupted after 39 completed jobs; only the missing final job
was resumed. One affected multi-photo control was rerun after a regression test
caught overlap validation outside the repair loop. The replacement correctly
allows overlap validation errors to receive repair attempts; the earlier control
is excluded from the table. Source snapshots/fingerprints preserve both runner
versions. Other strategies' prompts and extraction/merge behavior are unchanged
by that correction.

Validation: eight focused experiment tests; backend lint/type checks and unit
suite; frontend checks, 74 unit tests and build. A bounded independent runner
review found no Important/Critical findings within the experimental scope.
