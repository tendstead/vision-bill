# vision-bill — Your Private Invoice Tracker

vision-bill extracts receipts with local models and provides a bundled web UI for
review, collections, search, spending statistics and explicit admin model comparisons.
Choose an external LLM endpoint and an explicit model. OCR is optional. Ollama and
[OpenAI-compatible backends](OPENAI_COMPATIBLE.md), including llama.cpp, are supported.
Keep endpoints local to retain local processing. Argon2 passwords, session cookies,
owner-scoped data and an optional admin-wide view support multiple users.

## Install a prepared bundle

The intended end-user installation is a **downloaded, verified prebuilt bundle**,
then `docker compose up -d` from its extracted directory. Choose app with an external
LLM, independent CUDA/ROCm OCR, or app with optional Dedicated OCR on CUDA/ROCm
and an external LLM. Every app variant uses your endpoint. The combined CUDA
bundle includes a Windows launcher for Docker Desktop. Combined CUDA and ROCm
bundles also include a Linux launcher. The OCR controller starts idle
and downloads the pinned Paddle models when an admin first turns **Dedicated
OCR** On or explicitly requests setup in the configuration. The four model
downloads total about 2.1 GB. The controller verifies their hashes and parks.
The first receipt loads PaddleOCR-VL 1.6 and runs the first prediction. Setup,
restart and the Settings connection check do not process sample images.
[Installation guide](distribution/README.md)

**Status: Unreleased.** Source candidates and release tooling are available.
Public native bundle downloads require a release run with real artifacts and physical
qualification of the selected GPU.

Once a verified bundle exists, app variants expose the UI at
`http://127.0.0.1:8080` by default. First setup creates a real bootstrap administrator
and private persisted credentials. Registration creates ordinary users; the first
visitor is never promoted. Repeated startup preserves credentials and data.
Read the [upgrade guide](UPGRADE_AND_VERIFY.md) before using existing data.

Use **Appearance** to choose Light (black text on white), Black (white text on
black), or Purple (the original palette). The choice is saved in the browser.
Receipt review shows the photo alongside the fields on desktop. On smaller
screens, switch between **Data** and **Receipt** to inspect the source while
keeping your edits.

## Processing and review

Every upload (one to ten ordered photos, at most 20 MB each) returns committed
`202 Accepted`, a durable ID and `Location` for authenticated status polling.
The worker waits for usable capabilities before claiming an attempt, extracts
printed observations, retains partial drafts, checks source/amount/overlap evidence,
and applies classification downstream. A complete saved observation can finish
after interruption without repeating inference once ownership is settled. Missing
printed observations stay null. A receipt without a discount, tip, quantity or
unit price can be finalized if the required figures reconcile. Unavailable
categories or tags remain visible. Human corrections and original bytes survive review.
Verify changes database workflow state without moving files. See the
[multi-photo/API workflow](MULTI_IMAGE_RECEIPTS.md).

For external OCR, set the canonical `OCR__ENDPOINT` to an HTTP(S) URL. Set it
explicitly blank for initial direct vision. With a prepared GPU profile, an admin
can still opt in through the Dedicated OCR toggle. Leave it unset for a new
Dedicated OCR installation. A nonblank external endpoint remains operator-owned
and cannot be switched with that toggle. Turning Dedicated OCR Off routes new
work to direct vision. The OCR service stays parked, and turning it On again
reuses the installed models. When OCR is configured, ordinary formatting,
repairs and classification use **text/evidence only**; OCR failure never sends
receipt images to the LLM. Deprecated fallback fields remain for request replay
and history, not as new permission. Compare OCR and direct vision only through an
explicit admin benchmark. Changed routing can hold pending work until an explicit
current-policy retry; completed receipts are never automatically reprocessed.

## Configuration and comparisons

Settings normally use process environment > dotenv > persistent YAML. An explicit
admin save can adopt changed safe formatter fields as described below. Prepared bundles
store `/app/config/config.yaml` in the private `app-config` volume. Developer
Compose binds `server_data/config` there and requires the configuration directory
and files to be owned by the app container's effective user. The admin Settings
page reports sources and lets an admin edit formatter provider, endpoint, model
and temperature, including safe fields initially supplied by environment defaults.
An explicit save adopts changed fields into persistent settings. Model, temperature
and registration changes apply immediately; provider and endpoint changes require
a restart. Credentials remain in operator configuration and are not exposed there.
Formatter output
defaults to 8192 tokens; OCR's 4096-token cap and model context capacity are separate.
Unknown context stays unknown. Managed same-GPU OCR/LLM work requires verified
Ollama or the explicit dedicated llama.cpp router lifecycle, stable identities and
shared coordination storage. Explicitly adopt the dedicated endpoint and selected
model identities before running managed work. Incompatible pairs wait before claim. This coordinates
owned models and leaves the physical GPU available to independent processes.

Benchmark Gold requires human verification plus explicit qualification of the
**current** document and sources. Bypass or legacy `verified` alone is insufficient.
Admin runs freeze Gold, ordered source hashes and execution identity; v4 diagnostics
include failed cases and exact receipt/item/amount checks. See
[scripts and role setup](scripts/README.md) and the
[comparison harness](scripts/ocr_harness/README.md). Historical CPU comparisons and
[split experiments](scripts/receipt_split_experiment.md) remain research evidence;
blanket splitting and global product deduplication are not production behavior.
Benchmarks are explicitly requested. App installation and startup use the
configured LLM endpoint. Dedicated OCR reuses its verified local models.
Maintainers can run physical hardware qualification explicitly. Pending work
can resume when its readiness and
policy checks pass.

## Advanced: develop from source

Python 3.12+, PostgreSQL, uv and Node/npm are used. `make help` lists commands;
`make setup`, `make migrate`, `make run`, `make fe-install` and `make fe-dev`
prepare a developer installation. For source Compose, copy `.env.example` to a
private `.env`, set a random auth secret, database credentials and explicit LLM
provider, reachable endpoint and installed model ID,
then use `docker compose up -d --build`. For a fresh source installation, set
`AUTH__BOOTSTRAP_USERNAME` and `AUTH__BOOTSTRAP_PASSWORD` before the first account
if you need admin access. Ordinary users can register from the login page.
Base Compose starts the app and PostgreSQL. To try a local Dedicated OCR
candidate, prepare the GPU profile once and then enable it from admin Settings.
The admin settings include the LLM authentication mode and the choice of a shared
or separate OCR GPU.
Use CUDA on Linux or WSL 2 with Docker Desktop and NVIDIA, or ROCm on native
Linux with a selected Radeon. [Source GPU setup](paddleocr-service/README.md#dedicated-ocr-from-source)

`make lint`, `make test`, `make fe-verify`, `make release-test`,
`make distribution-test`, `make distribution-pid1-test`, `make compose-check`,
`make test-e2e` and `make fe-test-e2e` are software gates. E2E creates/removes unique
owned stacks. See the [architecture guide](AGENTS.md),
[frontend guide](frontend/README.md),
[accepted evidence and limits](UPGRADE_AND_VERIFY.md#verification-evidence), and
[release procedures](RELEASES.md).
