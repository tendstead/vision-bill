# Receipt OCR comparison harness

## Step-by-step view

Open `server_data/ocr-harness/results/report.html` in a browser. It is a
self-contained local file showing the input image, OCR transcription and
grouped text sent to Gemma, raw model responses (including repair attempts),
and the validated result. Expand the request/prompt and expected-field
differences when diagnosing a mismatch. All receipt/model text is escaped;
the report has no scripts or external resources.

Regenerate the HTML from existing results without Docker or model calls:

```bash
python3 -m scripts.ocr_harness report
```

Older JSON caches have no raw-response trace; the report says so explicitly.
Capture a new trace while retaining the OCR cache with `--refresh-llm`:

```bash
uv run python -m scripts.ocr_harness run --only pp-small \
  --data-dir server_data/ocr-harness/upright --limit 1 \
  --llm-host http://127.0.0.1:11434 --refresh-llm
```

Fresh traced runs use `ReceiptService.analyse_submission_with_metadata` with
saved OCR text and return its validated receipt. They do not call the HTTP
API, connect to the database, or persist receipts. Legacy caches are labeled
as provider results rather than service returns. `trace.json` stores prompts
and raw response text, including failed attempts, but excludes model thinking.

One runner, thin adapters, no application deployment or database connection.
Python 3.12+ and Docker are required. Only the optional Gemma/scoring stage
requires vision-bill's Python environment (`uv run python` or `.venv/bin/python`).

```bash
# Build isolated images, download models without receipts, then compare two images.
python3 -m scripts.ocr_harness all --data-dir /path/to/receipts --limit 2

# Repeat comparisons using the prepared environments and cached results.
python3 -m scripts.ocr_harness run --data-dir /path/to/receipts --limit 20

# A subset or a changed experiment.
python3 -m scripts.ocr_harness run --only pp-small,easyocr-de,tesseract-fast \
  --data-dir /path/to/receipts --limit 10 --timeout 120
```

`engines.json` contains seven enabled configurations: PP-OCRv6 Small/Medium,
PP-OCRv5 Latin, EasyOCR German/English, Tesseract Fast/Best German/English,
and docTR FAST+CRNN. docTR records missing German checkpoint characters as
warnings; vocabulary limitations are not silently treated as full support.
Tesseract uses segmentation mode 4; add a copied config with `"psm":6` to
compare another mode. Paddle's pinned 3.3.1 oneDNN/PIR path failed in live tests;
`mkldnn` defaults to false. Set it explicitly in a copied config to investigate.

Build and preparation failures have per-engine log files. Recognition runs
sequentially with no network, four CPU cores, an 8 GB memory ceiling and a
120-second timeout per engine/image. Preparation uses four GB and a separate
900-second timeout. Override recognition resources per engine using
`"limits":{"cpus":"4","memory":"8g"}`. These are test ceilings, not measured
memory requirements. A timeout kills the process group and removes its Docker
container; errors never stop later engines. Exit status is nonzero if any job
failed, even though a complete partial-results report is written.

Default private, Git-ignored work directory: `server_data/ocr-harness/`.

- `models/`: local model weights shared with isolated workers.
- `preparation.json`, `prepare-*.log`, `build-*.log`: setup outcomes.
- `results/report.json`, `.csv`, `.md`: comparison results, including failures.
- `results/jobs/<hash>/ocr.json`: compact text, original boxes/confidence,
  coordinate-grouped `row_text`, versions and warnings. Confidence values are
  engine-specific, not comparable calibrated probabilities.
- `worker.log` in each job: engine diagnostics, potentially receipt text.

Cache keys include image contents, engine options, Docker image ID, cached
model/config contents, adapter/normalizer code and resource limits. Only
successful recognition is cached. Editing a model invalidates the cache;
adding unrelated models also invalidates it conservatively. Cache hits retain
the original elapsed time and are explicitly marked. Timing includes container,
Python and model startup, so it measures isolated jobs rather than warm-service
throughput. `--force` reruns successes. Reports describe the latest selected
matrix; use a different `--work-dir` to preserve another experiment.

Images are supplied as-is. Keep upright/cropped/resized controls in a separate
dataset directory and record the transformation. Use the same variant for
every engine; do not treat improvements from changed inputs as model-only
improvements. The live smoke dataset's original and upright controls were
preserved separately for this reason.

Keep these artifacts local: images, raw text, receipts and logs can contain PII.
Model downloads happen only in `prepare`, without any image directory mounted.
The worker normalizes output rather than retaining pixel arrays/model internals.

## Optional receipt JSON and accuracy

Run the same cached OCR through the local Ollama model, without repeating OCR:

```bash
uv run python -m scripts.ocr_harness run --data-dir /path/to/receipts --limit 10 \
  --llm-host http://127.0.0.1:11434 --llm-model gemma4-12b:latest
```

This stage uses vision-bill's actual OCR-to-Receipt provider and repair loop,
temperature 0, thinking/classification disabled and no direct-vision fallback.
It sends coordinate-grouped text, not images. It rejects public endpoints and
disables proxy environment variables and redirects. Each receipt has an outer
600-second model-stage deadline including repair calls. Successful extraction
is cached separately by text/prompt/schema/provider/model identity. Expected
values are rescored even when extraction is cached.

For reviewed ground truth, place a valid `Receipt` JSON next to its image:
`shop.png` + `shop.json`. It is your responsibility to review that sidecar.
Existing detailed receipt scoring compares totals, quantities, prices and
missing/extra rows without classification. Scores are weighted field scores,
not percentages of perfectly recognized receipts. Without a sidecar, accuracy
is left empty; successful JSON alone is not evidence of correct extraction.
Without `--llm-host`, no LLM is called and no receipt-level accuracy is claimed.

## Adding another OCR

Add one worker function and a configuration entry, or provide an external
command adapter in a custom `--engines /path/to/engines.json`:

```json
[{"id":"custom","command":["/absolute/python","/absolute/adapter.py","{image}","{result}"]}]
```

The command writes JSON to `{result}`: `{"text":"...","blocks":[{"text":"...",
"bbox":[[0,0],[100,20]],"confidence":0.95}],"versions":{},"warnings":[]}`.
Coordinates must be image pixels; confidence may be null. Commands are argv
arrays, never shell strings. Custom commands manage their own dependencies
and privacy isolation; only built-in Docker adapters enforce offline execution.
Treat engine manifests as trusted executable configuration.

The existing HTTP OCR adapter is disabled by default: its address must be
configured for a reachable local service, and its revision must identify the
server/model deployment. It accepts the same compact `text`/`blocks` contract;
remote model changes require manually updating `revision`. It does not change
or redeploy your current PaddleOCR-VL service.
