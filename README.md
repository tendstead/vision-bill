# vision-bill — Your Private Invoice Tracker

vision-bill extracts receipts with local models and provides a bundled web UI for
review, collections, search, spending statistics and explicit admin model comparisons.
Choose an external LLM endpoint and an explicit model. OCR is optional. Ollama and
[OpenAI-compatible backends](OPENAI_COMPATIBLE.md), including llama.cpp, are supported.
Keep endpoints local to retain local processing. Argon2 passwords, session cookies,
owner-scoped data and an optional admin-wide view support multiple users.

## Install a prepared bundle

The intended end-user installation is a **downloaded, verified prebuilt bundle**,
started from its extracted directory with the launcher or manual Compose
commands for your host. Choose app with an external
LLM, independent CUDA/ROCm OCR, or app with optional Dedicated OCR on CUDA/ROCm
and an external LLM. Every app variant uses your endpoint. The combined CUDA
bundle includes a Windows launcher for Docker Desktop. Combined CUDA and ROCm
bundles also include a Linux launcher. The OCR controller starts idle
and downloads the pinned Paddle models when an admin first turns **Dedicated
OCR** On or explicitly requests setup in the configuration. The controller verifies the three active models
for orientation, layout and recognition, then parks. Geometric unwarping is off.
Each recognition region can use up to 2,000,000 pixels.
The first receipt loads PaddleOCR-VL 1.6 and runs the first prediction. Setup,
restart and the Settings connection check do not process sample images. In an
independent OCR bundle, run the explicit model preparation command once.
Subsequent startup checks the cache. Physical qualification is optional on the
operator’s device.
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

A successful receipt upload (one to ten ordered photos, at most 20 MiB each)
returns `202 Accepted`, a durable job ID and a `Location` for authenticated status polling.
The worker waits for usable capabilities before claiming an attempt, extracts
printed observations, retains partial drafts, checks source/amount/overlap evidence,
and applies classification downstream. A complete saved observation can finish
after interruption without repeating inference once ownership is settled. Missing
printed observations stay null. A receipt without a discount, tip, quantity or
unit price can be finalized if the required figures reconcile. Unavailable
categories or tags remain visible. When correcting a draft, select the payment
method in the current language and enter confidence as a whole percentage from
0 to 100. Saving refreshes the checks before finalization. Product tags such
as **Coffee** or **Kaffee** use the same stored value across both languages.
Each item can carry several tags. Add or remove them in the editor. The
classifier receives separate choices for the whole receipt and for each item.
The interface language controls their labels. Product descriptions and the
seller printed on the receipt keep their original language.

The merchant field groups branches under a common name. For example, an identified
HIT branch appears as **HIT Handelsgruppe** with a **Grocery** suggestion.
The review form also shows the seller as printed and the branch address. Choose
a suggested merchant or type the trading name of an independent shop. An
explicit receipt save or verification, including finalizing a corrected draft,
remembers that choice in your account for future receipts. Keeping an unfinished
draft saves your edits without adding a merchant to the directory.
For ambiguous trading names the classifier can choose among locally matched
brands using distinctive purchased items. Groceries alone cannot identify a
particular supermarket. For covered German brands, see the
[merchant directory and its official sources](HAENDLERVERZEICHNIS.md).
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
and registration changes apply immediately. Restart the app after changing the
provider, endpoint or effective authentication. Both providers support **None**
or **Bearer API key** in Settings. Save the connection, then use **Test saved
connection** to list model IDs without sending a receipt or running inference.
Choose a listed ID or enter the exact ID manually, then save your selection.
Settings shows only the credential status.

Formatter output
defaults to 32768 tokens; OCR's 4096-token cap and model context capacity are separate.
Unknown context stays unknown. Managed same-GPU OCR/LLM work requires verified
Ollama or the explicit dedicated llama.cpp router lifecycle, stable identities and
shared coordination storage. Explicitly adopt the dedicated endpoint and selected
model identities before running managed work. Incompatible pairs wait before claim. This coordinates
owned models and leaves the physical GPU available to independent processes.

In **Benchmarks**, admins can pick human-verified receipts or upload separate
test cases. A case stores its photos without running OCR or an LLM and stays
outside the receipt ledger and spending reports. Confirm its expected values,
then choose an image-model run, an OCR-plus-LLM run, or both. Refresh the live
model list and select one or more eligible IDs. Set a temperature for this run,
including zero, or leave it blank to use the formatter default. In a paired run,
results show each model’s scores side by side. Models that finish both pipelines
on the same inputs also show the difference in percentage points. [Benchmark walkthrough and API](UPGRADE_AND_VERIFY.md#benchmarks-and-reproducibility)

Receipt references require explicit qualification of the current document and
sources. Runs freeze the expected values, source hashes and execution identity. See
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
Base Compose starts the app and PostgreSQL. To let an administrator start the
first local GPU build later, run `python3 scripts/ocr_source_up.py cuda`
from the source checkout, or select `rocm` for AMD. This is the default source
launcher flow. Use `--prepare-now` for an immediate native build. Keep the launcher running on
the host. A vision LLM on a separate GPU can process receipts while the native
image builds. Shared-GPU activation follows the handoff requirements in Settings. Enabling **Dedicated OCR** in Settings starts
the build and then installs the pinned OCR models.
The admin settings include the LLM authentication mode and the choice of a shared
or separate OCR GPU.
The source profiles target NVIDIA Turing (RTX 20 series) and later with CUDA,
and include experimental AMD Dedicated OCR on native Linux and Windows with WSL 2
for selected RDNA2 and later GFX targets. NVIDIA uses Docker Desktop on Windows. AMD uses the WSL GPU bridge
and Docker Engine inside Ubuntu. For a local source installation, CUDA uses
the official wheel for the selected channel. ROCm compiles for the selected
Radeon's GFX architecture and host interface. Check the exact board and driver
against AMD’s OS-specific compatibility table. Verified distribution bundles
use the native image already built for the release. [Source GPU setup](paddleocr-service/README.md#dedicated-ocr-from-source) · [Receipt logs in PowerShell or Linux](paddleocr-service/README.md#read-the-logs-for-a-receipt)

`make lint`, `make test`, `make fe-verify`, `make release-test`,
`make distribution-test`, `make distribution-pid1-test`, `make compose-check`,
`make test-e2e` and `make fe-test-e2e` are software gates. E2E creates/removes unique
owned stacks. See the [architecture guide](AGENTS.md),
[frontend guide](frontend/README.md),
[accepted evidence and limits](UPGRADE_AND_VERIFY.md#verification-evidence), and
[release procedures](RELEASES.md).
