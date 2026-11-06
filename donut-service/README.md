# Donut receipt reference experiment

This adds an independent CPU HTTP service beside PaddleOCR. It does not change
Vision Bill's providers, database, receipt upload flow, benchmark UI, Paddle
wrapper, existing Makefile, or pre-commit hooks. These are exploratory model
outputs, not ground-truth OCR and not an accuracy claim.

## Model and language limitations

The default is `naver-clova-ix/donut-base-finetuned-cord-v2` with task token
`<s_cord-v2>`. It returns CORD fields such as `menu`, `sub_total`, and `total`.
It does not return Vision Bill's Receipt schema, arbitrary categories/tags,
confidence scores, or bounding boxes. Missing fields are not synthesized.
German receipt accuracy is unverified. Poor results may reflect the checkpoint's
training domain, rather than a limitation of every Donut model.

A base pretrained Donut model is not itself a receipt extractor. To try another
receipt fine-tune set `DONUT_MODEL` and its matching `DONUT_TASK_PROMPT`; optionally
set `DONUT_REVISION` to a Hugging Face commit hash for reproducible downloads.
An incompatible checkpoint/task pair will not be corrected automatically.
This patch includes no training or fine-tuning.

Reference: https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2
Usage: https://huggingface.co/docs/transformers/model_doc/donut

## Apply and start

Apply from the root of your local Vision Bill checkout:

```bash
git apply --check /path/to/donut-reference.patch
git apply /path/to/donut-reference.patch
make -f Makefile.donut donut-test
make -f Makefile.donut donut-up
make -f Makefile.donut donut-logs
```

The files are new additions, so local edits to `paddleocr-service/server.py` are
not touched. Default port: **127.0.0.1:8083**. Check readiness with:

```bash
curl --fail http://127.0.0.1:8083/health
```

First startup downloads the public model files to a persistent Docker volume.
No receipt images are uploaded to Hugging Face. Subsequent starts reuse those
files. After the first successful startup, `DONUT_OFFLINE=1 make -f Makefile.donut
donut-up` disables model downloads (the full checkpoint must already be cached).
Container startup is not complete until the model loads; failed downloads stop
startup rather than reporting a healthy model. The image build downloads Python
packages. The runtime intentionally uses CPU PyTorch; it does not assume support
for an RX6950XT. Default inference threads: 2, configurable through
`DONUT_CPU_THREADS`. CPU latency and memory use still need measuring on your NAS.
One Uvicorn worker owns one loaded model. Concurrent inference returns 429 rather
than creating an unbounded queue. No custom process supervisor is introduced.

## Inspect one or several images

Run the client in Docker; the host needs no Python packages. Only the selected
receipt directory is mounted read-only:

```bash
DONUT_RECEIPT_DIR=/absolute/path/to/receipts \
  make -f Makefile.donut donut-reference \
  ARGS="/receipts/receipt.jpg" > donut-result.json

DONUT_RECEIPT_DIR=/absolute/path/to/receipts \
  make -f Makefile.donut donut-reference \
  ARGS="/receipts/crop-1.jpg /receipts/crop-2.jpg" > donut-crops.json
```

Each image/crop is parsed independently. Overlapping crops are not deduplicated
or merged: examine their results before implementing a receipt-level merge.
The checkpoint's fixed image preprocessing can lose detail on very long images.

To run the same images against your existing Paddle wrapper as well:

```bash
DONUT_RECEIPT_DIR=/absolute/path/to/receipts \
  make -f Makefile.donut donut-reference \
  ARGS="/receipts/receipt.jpg --paddle-url http://host.docker.internal:8082" \
  > donut-paddle-comparison.json
```

The Paddle endpoint must be reachable from this container and expose multipart
`POST /ocr` with an `image` field. The existing Paddle service is not started or
published by this patch. If it only exists on an internal Docker network, use
its existing client or explicitly arrange connectivity. Do not expose it beyond
your local network just for this comparison.

The client preserves both responses and wall-clock times, runs services
sequentially, and exits nonzero on a failed request. It refuses responses larger
than 2 MiB so a pixel-array JSON dump cannot consume all available memory. Error
records still appear in the JSON output. No model-result cache hides repeated
inference time. `elapsed_ms` inside Donut excludes model loading; the client's
elapsed time includes the complete HTTP request. Compare warm runs separately
from first startup.

Donut's `/ocr` response contains `prediction`, bounded `raw_sequence`, model and
revision, oriented image dimensions, generated token count, and `truncated`.
`truncated: true` means EOS was not generated; do not treat its partial prediction
as complete. Default generation cap: 768 new tokens, configurable with
`DONUT_MAX_NEW_TOKENS` and capped by the checkpoint's decoder capacity. Raising
the limit can substantially increase CPU time. There are no automatic retries.
A client timeout does not cancel already-running model inference.

To call the service directly:

```bash
curl --fail -F image=@receipt.jpg http://127.0.0.1:8083/ocr
```

Uploads are limited to 25 MiB and 32 megapixels. Responses omit image pixels,
base64 image data, and internal tensors. Images stay in memory for the request;
the service does not persist them. Neither outputs nor receipt contents are
logged intentionally.

## Stop and verify

```bash
make -f Makefile.donut donut-down
```

This retains the model cache. The separate `donut-test` target runs boundary tests
and lint in Docker without downloading model weights. It does not replace the
project's existing `make lint`, `make test`, or `make fe-verify` checks. A real
checkpoint smoke test and your receipt benchmark are required to evaluate model
quality. The experiment is not registered in the benchmark UI.
