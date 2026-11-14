# Local receipt extraction benchmark

Results from 2026-11-14, using the paired runner and extraction changes in
`03035a48bd39c127e993c7f2408b07932109028e`.

## Dataset and method

- Ten verified receipts: 14 photos, 106 reference line items, nine merchant
  groups, five purchase categories, and one to 32 items per receipt.
- A purposive coverage sample including faint print, dense item lists,
  landscape captures, and overlapping multi-photo receipts. It is not a
  random sample or a statistically representative accuracy estimate.
- Identical prepared photos for both paths: EXIF orientation applied, three
  sideways photos manually rotated, and each photo capped at a 2000-pixel
  maximum edge while preserving aspect ratio. Original bytes were retained.
- Vision-only received all ordered photos through the existing receipt-service
  multi-photo path. OCR→LLM reused PP-OCR Medium transcripts from each photo,
  concatenated in capture order with overlaps retained for the model to resolve.
  This experimental orchestration does not add production multi-photo OCR support.
- Real receipt-service/provider extraction and existing detailed field scoring;
  no HTTP upload API, application deployment, database initialization, or receipt
  persistence. All inference ran locally, with model backends used sequentially.
- Temperature zero, thinking disabled, classification disabled, and no vision
  fallback from the OCR path. Each pipeline had a 180-second overall deadline
  and an 8192-token limit per generation. Qwen and NuExtract profiles used 32K
  context. Model identities and effective serving configurations were recorded.

Private references, photos, OCR text, raw model attempts, and HTML inspection
reports are excluded from Git. Only aggregate results are published here.

## Results

All eight combinations were evaluated on the same ten receipts: 80 pipeline
jobs, with 75 validated service returns and five failures.

| Model / input | Valid JSON | Exact total | Same item count | All item numbers exact | Weighted score | Median model/service time |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B Q4 / vision | 10/10 | 10/10 | 7/10 | 3/10 | 76.2% | 50.9 s |
| Qwen3.5-9B Q4 / PP-OCR Medium | 10/10 | 10/10 | 8/10 | 3/10 | 72.1% | 9.3 s |
| Qwen3.5-9B Q8 / vision | 9/10 | 9/10 | 8/10 | 3/10 | 69.6% | 57.8 s |
| Qwen3.5-9B Q8 / PP-OCR Medium | 10/10 | 10/10 | 7/10 | 3/10 | 72.5% | 14.4 s |
| Gemma 12B / vision | 9/10 | 9/10 | 7/10 | 5/10 | 65.9% | 15.1 s |
| Gemma 12B / PP-OCR Medium | 10/10 | 10/10 | 9/10 | 4/10 | **80.0%** | 17.9 s |
| NuExtract3 Q8 / vision | 8/10 | 8/10 | 7/10 | 2/10 | 61.6% | 8.3 s |
| NuExtract3 Q8 / PP-OCR Medium | 9/10 | 9/10 | 6/10 | 1/10 | 61.1% | 5.5 s |

**Weighted score:** existing normalized field score, with classification and
payment method excluded. The reference payment method can reflect user knowledge
not present in the image. Failed pipelines count as zero. This score is not the
percentage of completely correct receipts.

**All item numbers exact:** equal item counts, with every ordered line's quantity,
unit price, and total price matching the reference exactly. Product descriptions
are excluded from this particular measure. Same item count alone does not imply
correct items or prices.

**Timing:** median model/service elapsed time among successful jobs. OCR was
replayed, so these times exclude OCR recognition and are not end-to-end pipeline
latencies. Loading and shared-machine conditions can affect timings and deadlines.

## Failures and configuration findings

- Qwen Q8 and Gemma vision-only each timed out on the longest three-photo receipt.
- NuExtract vision failed overlap-metadata validation on two receipts; its OCR
  path timed out on the longest receipt.
- NuExtract used the author's structured-extraction template with a receipt field
  template, nonnegative-discount schema instructions, and a model-side adapter
  mapping standard repair history into native instructions/previous-output inputs.
- Calibration exposed Qwen copying the output schema instead of populating it.
  Shared extraction prompts now distinguish receipt values from schema definitions
  and explicitly identify OCR input as text. A controlled retry and regression
  tests verified the correction.
- The initial 16K Qwen profile could not hold a long multi-photo repair conversation.
  The reported matrix uses 32K. Earlier calibration attempts are kept separately
  and are not mixed into the results above.

## Interpretation

PP-OCR Medium→Gemma improves the weighted score by **14.05 percentage points**
against Gemma vision-only on this prepared sample, and completes all ten receipts.
It does not uniformly improve line-item money: all ordered quantities/prices match
on four OCR→Gemma receipts versus five vision-only receipts.

OCR also did not consistently improve Qwen's aggregate score, and NuExtract3 did
not outperform the best existing combination in this configuration. These results
support keeping OCR→Gemma as a useful candidate while retaining field-level review;
they do not establish a universal OCR winner or an automatic-verification threshold.

See [README.md](README.md#paired-vision-only-versus-ocr--llm) for manifest formats,
local target configuration, execution limits, caching, and the `make receipt-compare`
command. The private fixtures and raw artifacts are not distributed with the repo.
