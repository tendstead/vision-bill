# PaddleOCR-VL 1.6 installation and operation

These are Linux x86_64/Python 3.12 **qualification candidates**. Neither CUDA nor
any Radeon target has been hardware-qualified here. Do not advertise a target
until its real `qualification.json` has `qualified: true`. No weights or real
receipt fixtures belong in Git or the Docker build context.

## Fixed extraction policy

PaddleOCR-VL-1.6 (`pipeline_version=v1.6`), PP-DocLayoutV3, native Paddle,
FP32; batch 1 throughout, internal PaddleX queues off, chart/seal off. Document
orientation classification detects page rotation and turns sideways or upside-down
pages upright. UVDoc unwarping flattens curved or perspective-distorted
pages. Both are enabled. EXIF orientation normalization applies smartphone
metadata independently before native preprocessing. Region min 112896 /
max 2000000 pixels (including table regions), OCR max_new_tokens 4096, no extra
sampling parameters. The official VLM grid alignment applies **per region**;
2MP is not a whole-working-image downscale. No platform-specific precision,
backend, resolution or token reduction is permitted. Formatter 8192/context
capacity are separate app/model facts, and unknown context stays null.

Original bytes/hash/photo identity survive first-frame/EXIF conversion and
white RGB transparency composition (`white-document-rgb-v2`). Actual preprocessor
pixels become a separately hashed working PNG with transform metadata. Native v2
responses contain this PNG and recognized blocks in **working coordinates**.
No inverse UVDoc transform is fabricated (`inverse_mapping=null`). Raw diagnostic
responses can contain receipt text/pixels and belong only in private artifacts.

The app accepts up to 20 MiB per source photo and the native service up to 25 MiB.
Positive working dimensions are bounded at 40,000,000 pixels. The shared contract
allows up to 182,533,476 bytes of metadata-free RGB8 working PNG and 8 MiB of
canonical UTF-8 JSON metadata, with a 200,000-character OCR text limit. The
251,766,576-byte complete response bound includes padded base64 and JSON framing.
The PNG bound accounts for row filters, worst-shape DEFLATE and chunk overhead.
Readers validate lengths, dimensions, PNG integrity and hashes before accepting
working artifacts. The 2 MP setting still applies to VLM regions.

The OCR container hosts the complete PaddleOCR/PaddleX pipeline with the native
PaddlePaddle inference engine. Layout, orientation, unwarping and VLM recognition
execute in its disposable native worker. Vision Bill uses its private HTTP API.
The operator's separate text or vision LLM endpoint remains the receipt formatter.

## Dedicated OCR from source

The source setup command builds a local candidate, selects the accelerator
profile, binds approved inputs to the exact Docker image and selected device, and
starts Compose with an idle OCR controller. Run it from the repository root with
Python 3.12+, a local Linux x86_64 Docker engine and Docker Compose 2.30+.
Automatic fixture preparation uses `uv`. CUDA builds also use `make` and access
the official Paddle wheel host. Prepare the
app's private `.env` once and reuse its Compose project name, app settings and
data on subsequent starts. The command supplies the OCR image and bundle metadata
automatically. [Source app configuration](../README.md#advanced-develop-from-source)

### NVIDIA CUDA on Windows or Linux

On Windows, install or update WSL 2 with Ubuntu (`wsl --install -d Ubuntu` on a
fresh system and `wsl --update`), install a current NVIDIA Windows driver with
WSL GPU support, and enable Docker Desktop's WSL 2 engine and Ubuntu integration.
Run the commands inside Ubuntu. Keep the source checkout and working files on
Ubuntu's Linux filesystem, such as `~/src/vision-bill`. Linux file ownership and
permissions and build I/O are more reliable there than under `/mnt/c`.
For a separate WSL test checkout, choose a new Compose project name. Stop the
old source stack from its original checkout with `docker compose down` before
starting the test. That preserves its database volume and bind-mounted files.
To continue an existing installation, use its original project and data paths.
[WSL setup](https://learn.microsoft.com/en-us/windows/wsl/install) ·
[Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/)

For a verified prebuilt Windows bundle, use its included Windows launcher.
The following steps are for building a candidate from this checkout.
[Windows bundle setup](../distribution/README.md#start-a-verified-windows-bundle)

On native Linux, configure the NVIDIA driver and NVIDIA Container Toolkit for
Docker GPU access. `OCR_CUDA_DEVICE_ID` selects one host GPU by index or UUID
and defaults to `0`. Run the source launcher from this checkout.

```bash
python3 scripts/ocr_source_up.py cuda
```

The launcher binds the official model inventory to the selected image. Five
sample images are available for explicit release qualification. The first admin
activation downloads the pinned model files and verifies their contents. The
Windows or Linux host supplies its own GPU driver. Installing these models
does not install host graphics drivers or firmware. Automatic CUDA source preparation
uses the official CUDA 12.9 candidate by default. Set `OCR_CUDA_CHANNEL=cu126`
to select CUDA 12.6. Use a host driver that supports the selected runtime and
verify it on the actual board. [CUDA build inputs](#build-inputs)

### AMD ROCm on native Linux

Use the host's `/dev/kfd` and the selected `/dev/dri/renderD*` node. The source
command derives the two container group IDs from those device nodes and reads
the selected node's GFX architecture from the Linux KFD topology. Set
`OCR_RENDER_NODE` when choosing a different GPU from the default render node.
Set `OCR_AMDGPU_TARGETS` explicitly for a multi-target build or when the host
cannot report the selected architecture. The candidate checks the actual HIP
target before starting the service. [ROCm build inputs](#build-inputs)

```bash
python3 scripts/ocr_source_up.py rocm
```

To test different approved inputs, pass their directory as a second argument
after the accelerator. It must contain `bundle.json` and its declared local
fixture tar. CUDA uses the same `cu129` default with custom inputs.
[Prepare custom inputs](../distribution/README.md#prepare-approved-native-inputs)

### Enable and restart

Keep the existing `.env` OCR settings for a base install using the template
defaults or an explicitly blank `OCR__ENDPOINT`. A configured external OCR route
must be reconciled before switching. With a shared GPU and llama.cpp, use router
mode with autoload enabled, startup loading disabled, and all models unloaded
before activation. Configure `LLM__MODEL_LIFECYCLE=llamacpp` and use the exact
installed model ID. Keep your LLM endpoint reachable from the app container.
[llama.cpp router setup](../OPENAI_COMPATIBLE.md#configure-a-dedicated-llamacpp-router)

Sign in as an administrator and open **Settings → Dedicated OCR**. For an LLM
on another GPU or server, clear **LLM and OCR share a GPU**, save the GPU
configuration and leave the LLM running. Turn **Dedicated OCR** On.
The controller checks the image, GPU and package metadata, provisions the four
pinned model directories (about 2.1 GB) and publishes the verified installation.
The service starts parked. Uploading a receipt loads the pipeline and the first
prediction uses that receipt. Enabling OCR, polling readiness, restarting and
loading the worker do not process a sample image. On a clean setup failure,
switch **Dedicated OCR** on again to retry, or choose **Resume direct vision**
to accept new direct-vision uploads without OCR.
A retry rechecks the owned model cache, reuses valid files and can replace a
damaged model tree from a verified archive. If the archive itself is damaged,
the controller can fetch its pinned replacement. If restoration finds damaged
files, Settings requests recovery and the OCR container log records the reason.
When the status reports an unsettled GPU handoff, follow the recovery guide.
Turning it Off routes new work to direct vision and keeps the verified cache for
reuse. On a shared GPU, Vision Bill confirms that the paired formatter model is
unloaded before loading OCR. In both GPU modes, it waits for the OCR worker
process to exit before requesting formatting with text and evidence. An
unconfirmed park blocks formatting. Other applications still have normal GPU
access. When available, `/ready` and `/info` include a snapshot of free and total
memory from the selected OCR device before worker initialization. A confirmed
allocation failure surfaces as `ocr_gpu_out_of_memory`.
[Activation and retry](../distribution/README.md#enable-dedicated-ocr)

Use `--restart` on subsequent starts to reuse the prepared app and OCR images,
Compose project and GPU selection. Use `rocm` for AMD. Before replacing an
existing prepared runtime, drain it and plan a cutover that preserves its
activation and handoff state. [Stop, cut over and recover](../distribution/README.md#stop-cut-over-and-recover)

```bash
python3 scripts/ocr_source_up.py cuda --restart
```

If the native image finished building before initial preparation was interrupted,
use `--reuse-native-image` with the local `source-build-…` tag printed by the launcher.
It checks the selected image against this checkout and continues preparing the app.

### Upgrade an existing source installation

Run from the original checkout. Back up the installation and let active jobs
finish. Stop the old deployment with its original project, profile and all
three Compose files. Keep its named volumes, private settings and bind-mounted
files. For a CUDA deployment with the checkout's default project name:

```bash
docker compose --profile ocr -f docker-compose.yml -f docker-compose.cuda.yml \
  -f server_data/ocr-source/cuda/compose.local.json down
# Update this checkout to the desired revision, keeping server_data and .env.
python3 scripts/ocr_source_up.py cuda --upgrade
```

If you originally selected a custom project name, use that same `-p NAME` on
the stop command. For ROCm use `rocm` in the overlay name, directory and launcher
argument. The upgrade verifies the original deployment claim, retains its
project, device and approved model sources, then builds and starts the new
images. If the build fails before the launcher publishes its new deployment
record, `--restart` starts the previous recorded images. If the record was
published and container startup fails, `--restart` retries the **new** images.

For rollback, drain and stop the new deployment, account for writes since the
backup, then restore the previous checkout and its original image tags. Restore
the backed-up `server_data/ocr-source/<accelerator>/` directory, including
`compose.local.json`, `admitted.sha256` and the original release directory.
Restore the matching configuration, OCR activation, coordination and model
state, database, original uploads and working files from the same recovery
point before starting the old generation. [Compatible backout](../UPGRADE_AND_VERIFY.md#backup-and-upgrade-order)

When upgrading a failed initial
installation, switch **Dedicated OCR** on again in Settings. An installation
that was already ready is checked and migrated under the existing ownership
locks. Preparation reuses verified model assets. If the new image deliberately
changes approved package pins or model sources, pass the matching approved
input directory after the accelerator, before `--upgrade`.

CUDA upgrades reuse the original wheel channel. For older source records the
launcher reads the lock from the installed image, so a CUDA 12.6 installation
stays on `cu126`. To switch deliberately, set `OCR_CUDA_CHANNEL=cu129` or
`cu126` in your shell or the checkout's `.env` before running `--upgrade`.

A CPU limit previously set with `docker update` belongs to the container that
was replaced. To restore a previous limit of 6.4 CPUs on a CUDA source
installation, run this in the original checkout after the upgrade. Use your
previous limit and select `rocm` for a Radeon installation. The recorded
project name selects the new OCR container.

```bash
accelerator=cuda
overlay="server_data/ocr-source/$accelerator/compose.local.json"
project=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["x-vision-bill-source"]["project"])' "$overlay")
ocr_id=$(docker compose -p "$project" --profile ocr -f docker-compose.yml \
  -f "docker-compose.$accelerator.yml" -f "$overlay" ps -q paddleocr-service)
if [ -n "$ocr_id" ]; then
  docker update --cpus 6.4 "$ocr_id" && docker inspect --format '{{.HostConfig.NanoCpus}}' "$ocr_id"
fi
```

The example reports `6400000000` nanocpus when the limit is applied.

## Consumer GPU release targets

Prepare a binary bundle for the accelerator and host platform before an end
user starts the installation. The app uses the same Dedicated OCR settings on
all GPU profiles. The native runtime checks the selected device and the prepared installation.
The release report records which physical device passed the full diagnostic.

| Accelerator | Intended device families | Architectures to qualify | Host |
| --- | --- | --- | --- |
| CUDA | NVIDIA RTX 2000, 3000, 4000 and 5000 | Turing SM 7.5, Ampere SM 8.6, Ada SM 8.9, Blackwell SM 12.0 | Linux, including Docker Desktop with WSL 2 on Windows |
| ROCm | AMD Radeon RX 6000, RX 7000 and RX 9000 | RDNA2 `gfx1030/1031/1032/1034`, RDNA3 `gfx1100/1101/1102`, RDNA4 `gfx1200/1201` | Linux with the matching ROCm driver and device access |

These are qualification targets. Release evidence records the architecture,
driver and host environment for actual runs of the configured OCR pipeline.
The automatic source launcher builds the selected ROCm device's architecture.
The standalone ROCm build defaults to the listed RDNA2 candidates.
[NVIDIA GPU architectures](https://developer.nvidia.com/cuda/gpus) ·
[TheRock targets at the pinned revision](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/cmake/therock_amdgpu_targets.cmake) ·
[Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/)

Evaluate one CUDA bundle against all four NVIDIA generations before splitting
the binaries by architecture. Paddle publishes a 3.3.1 CPython 3.12 Linux x86_64
wheel in its CUDA 12.9 channel. CUDA 12.8 introduced native SM 12.0 compilation.
Verify the wheel and linked libraries, then run device allocation and the
full OCR pipeline on representative hardware.
[Official Paddle CUDA 12.9 wheel index](https://www.paddlepaddle.org.cn/packages/stable/cu129/paddlepaddle-gpu/) ·
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#new-features)

Run the full native qualification per architecture and host environment you
intend to support. For CUDA, include representative RTX 20-, 30-, 40- and
50-series devices, including lower-VRAM configurations, and test native Linux
and Docker Desktop with WSL 2 independently. For ROCm, qualify the recorded GFX
target on native Linux. Record the image and model identities, installed driver,
available VRAM, host OS and Docker version with each report. The CUDA
`driver_version` value in the report is the driver API version. Record the
installed NVIDIA driver release separately. Then activate
Dedicated OCR in the app and verify a multi-photo receipt through OCR, confirmed
worker shutdown, and LLM formatting. [Native qualification](#what-qualifies)

## Build inputs

`runtime-lock.json` records the immutable Ubuntu base, Ubuntu package snapshot,
APT TLS bootstrap image, Paddle/PaddleOCR/PaddleX/TheRock source pins,
compatibility patch hash, Rust archive hash, default RDNA2 target list and
upstream optional exclusions. The selected build candidate profile is recorded
in build evidence/kernels.json.
The common Python execution and native build locks contain real resolved hashes.
`requirements.txt` is not the production build/install contract; use the selected
Dockerfile and hashed accelerator/runtime locks.
Source packages are built without dependency resolution, using the hashed build
bootstrap. The archived PaddleOCR/PaddleX sources use their pinned release
versions through `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*`.

The checked-in `requirements.cuda.lock` is unresolved. The source setup command
resolves the selected channel on a connected host when a matching lock is
unavailable. To resolve it manually, run `make ocr-lock-cuda` for `cu129` or
`OCR_CUDA_CHANNEL=cu126 make ocr-lock-cuda` for `cu126`. The resolver selects the
exact official Paddle GPU 3.3.1 cp312 x86_64 wheel for that channel, inspects its
METADATA, freezes the full closure, hashes the bytes, and emits the lock plus
`requirements.cuda.attestation.json`. Review both outputs and use the same
channel for `make ocr-build-cuda`. Keep the lock, its attestation and the host
cache at `${XDG_CACHE_HOME:-~/.cache}/vision-bill/cuda-wheels`. With a
matching lock and cached SHA-256, the build reuses those bytes locally.
A source upgrade from `936c26d` can need one download of the full Paddle
wheel. With a matching lock, the launcher first checks the previous image's
pip cache for that URL and verifies any bytes it finds. If the wheel is
missing or its hash differs, the launcher fetches it from the attested URL.
An offline upgrade needs the matching lock and verified wheel, along with the
other required build inputs. Starting the previously recorded images with
`--restart` skips this build. BuildKit
can reuse the app dependency layers and, on ROCm, the built TheRock SDK when
only Paddle or the app changes. PostgreSQL uses a separate prebuilt image.
The Docker build requires a valid, resolved
lock and attestation for the official GPU wheel. Validation parses actual
executable requirements, preserves every entry of the pinned common runtime lock,
and requires the complete attestation. Includes, install options, duplicate or
conditional native requirements and comment-only evidence are rejected. The Make
targets provide the pinned packaging parser in an isolated uv tool environment.

`make ocr-build-rocm` builds the full TheRock SDK and Paddle from the approved
source pins. Fresh Git clones request `--filter=blob:none` for TheRock, Paddle
and their recursive submodules. Git retains commit history and tags, downloads
file contents for the checked-out revisions, and can fetch historical blobs on
demand. The recipe initializes TheRock's recursive Git submodules at their
recorded pins before applying upstream patches. It checks the source inventory
before building `patchelf` and reports the path and pin of a missing checkout.
Provision substantial disk/RAM/CPU capacity first; this is a large
native compilation, not an ordinary sidecar dependency install. `OCR_BUILD_JOBS`
sets the job limit for each CMake build (including TheRock subprojects) and
concurrent Git submodule fetches. TheRock can run multiple subprojects at once,
so their combined compiler count can exceed this setting. Model weights
are provisioned separately for qualification. The versioned patch replaces both
legacy HIP target lists, uses the flat SDK and same-pin legacy FindHIP helpers,
requires MIOpen, and stops the forced download of FlashAttention. FP32 runs with
FlashAttention off.

Build evidence is retained under `/opt/ocr/build-evidence`: original gitlinks,
recursive actual checkout HEADs and dirty patches, patch/`.smrev` hashes, native
wheel hashes, CMake caches, compiler commands/log, OS packages, and required
library/payload hashes. Default build candidates are gfx1030/1031/1032/1034.
`OCR_AMDGPU_TARGETS` selects a semicolon-separated **build-only** candidate list:

```sh
OCR_AMDGPU_TARGETS='gfx1030;gfx1100' make ocr-build-rocm
```

Selection changes both TheRock settings and Paddle HIP compiler inputs and is
sealed into `kernels.json`. It does not prove support: even syntactically accepted
architectures must compile, provide inspected libraries/kernels and pass the
physical gate on that exact board. The actual observed HIP architecture must
match the embedded selected profile. Rebuild/requalify after profile changes;
there is no runtime allowlist or architecture spoofing. Intended Linux consumer
scope starts at RX6000/RDNA2 and includes later build candidates requiring their
own physical qualification; no board is actually qualified here.
Optional upstream exclusions (including hipBLASLt's gfx1100 fallback) stay intact;
their host headers/APIs are retained, with no claim of RDNA2 fused/Lt kernels.

Missing required libraries, incorrect Paddle HIP target flags and parsed missing
required target payloads fail the build. Packed/JIT formats which LLVM cannot
prove remain `offline_status: unverified`, with all inspected bytes hashed.
Only a real selected-GFX FP32 GPU matmul/convolution probe with checked results,
followed by complete model inference and fresh replay, can close that gate.
Unknown compile/API/kernel/probe failures are failures, never downgrades.

## Local qualification inputs

Use ignored `server_data/ocr-runtime/` for the private corpus, model directories,
and outputs; `server_data/ocr-cache/` is the actual `PADDLE_PDX_CACHE_HOME` mount.
Provision the four exact approved model directories separately into private
`server_data/ocr-runtime/models/` before qualification (Compose exposes them under
`/var/lib/ocr/models/`). This repository has no automatic model-download Make target;
copy/provision the intended model bytes using the official model distribution on
the target host, then retain their identities in qualification. The mounted cache
is not a substitute for complete explicit model directories:

- `PP-DocLayoutV3`
- `PP-LCNet_x1_0_doc_ori`
- `UVDoc`
- `PaddleOCR-VL-1.6-0.9B`

Create `candidate.runtime.json` matching `ocr_service.profile.RuntimeSettings`:
`accelerator`, `gpu_index` (inside the container), all four `model_dirs` using
absolute container paths, `cache_dir: /var/cache/paddlex`,
`qualification_manifest: /var/lib/ocr/static-manifest.json`,
`expected_packages: {}` for initial sealing,
`expected_paddle_commit: 7688495538f4d6c1893f084dd238a402e8f68ab6`, and
`image_identity` containing the actual image identity expected by the selected
qualification route. For prepared releases this is the immutable registry manifest
reference; the separate config blob hash must equal the actual local Docker `.Id`.
`docker image inspect --format '{{.Id}}' IMAGE` returns the config ID, not the
registry manifest digest. Never substitute it for a published image reference.
The qualifier fills the exact installed active dependency versions/content hashes;
a nonempty supplied map must match exactly. The sealer includes `paddlex[ocr]`
and all active marker/extras-aware transitive dependencies, with exactly one
native Paddle distribution. No dependency identities are hand-entered.

`fixtures.toml` must have exactly five `[[cases]]` entries, one each with `kind`
`upright`, `rotated`, `warped`, `table`, `formula`; each entry specifies `path`
(relative to the TOML file or an absolute container path) and the actual original
file `sha256`. Use five distinct real inputs. For `table` and `formula`, also set
`required_labels = ["table"]` or `["formula"]`. Their native results must contain
that structure. Keep the upright input at `fixtures/upright.png` for the default
startup path, or explicitly change `OCR_QUALIFICATION_IMAGE_PATH`.

Use the supervised source Make targets for the explicit offline hardware gate.
Drain analysis, benchmark and standalone clients. Stop the source app and sidecar
using the selected overlay. If they share the device with Vision Bill's dedicated
LLM endpoint, quiesce that endpoint and confirm that no requests remain in flight.
Release only Vision Bill's selected model IDs. Confirm absence with Ollama's
`/api/ps`, or confirm `status.value=unloaded` for each canonical llama.cpp router ID
in `GET /models`. See [backend control](../OPENAI_COMPATIBLE.md).

```sh
accelerator=cuda
docker compose --profile ocr -f docker-compose.yml \
  -f "docker-compose.$accelerator.yml" stop vision_bill paddleocr-service
OCR_MAINTENANCE_INTENT=external-llm-quiesced \
  OCR_APP_HANDOFF_LOCK=/app/config/gpu-handoff.lock \
  make "ocr-qualify-$accelerator"
```

Use `accelerator=rocm` for Radeon. Set `OCR_RENDER_NODE`, `OCR_RENDER_GID` and
`OCR_VIDEO_GID` to the actual host node and groups. For NVIDIA,
`OCR_CUDA_DEVICE_ID` selects the host device and its container index is normally 0.
The original app lock path must be absolute under the shared `/app/config` mount.
Supply its actual location if overridden. Use the invocation-local
`OCR_MAINTENANCE_INTENT=separate-ocr-device` when OCR uses a separate GPU. These
attestations describe operator-verified conditions. The locks and metadata cannot
prove external quiescence.

The target refuses to run while the source app or sidecar is running or restarting.
It uses the selected local image, requires the original app handoff and native
installation locks, and gives the qualification run a writable runtime mount.
The low-level `ocr_service.qualify` command requires a live installer lock inherited
from a supervisor. Run the guarded Make target to qualify source images. Failures
exit nonzero with sanitized stage/type diagnostics. Each native child has a
600-second bound and is killed and reaped on timeout.

## What qualifies

The acyclic sequence is installed dependency inspection → separate PIR/legacy
artifact inspector subprocesses → exact unsealed native construction and hooks →
seal → fresh instrumented `build_pipeline` → `adapter.qualify` → gated `predict`.
No fabricated coverage enters construction. Exported DenseTensor inventory is
CPU artifact evidence; actual compiled device/FP32 evidence comes from native
hooks. Static Layout/Orientation/UVDoc are audited separately and every floating
exported parameter must bind by its exact preserved name to an observed GPU
input. Shape matching, arbitrary zipping and many-to-one binding are forbidden.

Each corpus case executes twice during collection. Every individual forward must
have the same complete floating hook record set; a corpus union is never accepted.
Unproven CPU floats, optimized-away/renamed weight bindings, varying branch
coverage, missing callbacks or unstable fresh-process identifiers fail the gate.
Two independent fresh workers replay each case, first qualifying the sealed
upright smoke through the qualification adapter. Reports distinguish startup, actual
cold smoke, first case **after smoke**, and warm case timings; they do not claim
cold case inference after warming with smoke. Dynamic VL parameters and all three
static models must pass FP32/device checks on each replay.

Only a full pass publishes `qualification.runtime.json` and a successful report.
Publication errors keep qualification false and remove stale or partially written
serving settings; the CLI exits nonzero. A failure report is written when storage
permits. `qualification.json` records exact profile,
model/image/runtime identities, actual device/GFX, source hashes, timing and audit
counts/digests. It contains no OCR text or images. Only the actual target is added
to `qualified_targets`; repeat the release gate on each physical target.

## Independent service and client

The service, neutral `ocr_contract` and HTTP-only client build from the independent
`paddleocr-service/` context; they do not import the VisionBill application. Prefer
verified [prepared OCR bundles](../distribution/README.md) for installation once
actually released. No such native release was produced here.

OCR-only bundles default to `OCR_SERVICE_MODE=standalone`. One request owns the
whole serial native load/infer/park lifecycle and returns validated original/working
identity and geometry. Managed owner tokens/lifecycle routes are rejected in this
mode. For a VisionBill pair, explicitly set `OCR_SERVICE_MODE=managed` before start:
the app/client supplies generation-fenced lifecycle ownership. Combined bundles
always use managed mode. Never mix standalone HTTP ownership with a managed pair.

From a prepared Python environment in the source checkout, an explicit standalone
request is:

```bash
python paddleocr-service/client.py /private/synthetic.png \
  --url http://127.0.0.1:8082 --timeout-seconds 600
```

`--media-type` can override MIME inference. The client sends the exact original
once with request UUID, photo index and SHA256, validates the matching response,
and prints JSON that may contain receipt text/pixels; keep output private. It needs
HTTP/contract dependencies, no app or native framework import. The endpoint must
actually be exposed by your standalone deployment; source Compose publishes no
OCR host port. The app's optional endpoint selector is independent of service mode.

## First setup and serving handoff

Standalone prepared startup runs the cache-only `ocr-init` check before serving.
It checks the selected installation against the
[pinned release assets](release-bundle.schema.json), image config/source and device
identities, licenses, qualification and model/fixture inventories. An exact healthy
cache is reused. A missing, damaged or mismatched cache requires explicit
`ocr-maintenance` using the prepared bundle's `ocr-maintenance` profile. During
maintenance, `ocr_service.initialize maintain` provisions verified OCR assets and
runs full native qualification before publishing the runtime and smoke paths.
Maintenance runs independently of receipt attempts and comparative benchmarks.
Combined bundles use file-only preparation after an explicit Dedicated OCR request.
Their shared model handoff lock lives at `/coord/gpu-handoff.lock`. See the
[prepared bundle instructions](../distribution/README.md).

Each maintenance invocation requires `--maintenance-intent external-llm-quiesced`
or `--maintenance-intent separate-ocr-device`. For an app/OCR pair, also give the
original shared app lock, usually `--handoff-lock /app/config/gpu-handoff.lock`.
Drain participating clients and stop the app and OCR service first. Confirm
external quiescence or separate devices before setting the intent. OCR-only
maintenance uses its native installation lock. See the bundle instructions for
the exact Compose service and image identity arguments.

Managed serving starts parked with no native child or loaded models. `/health`
reports liveness. `/ready` stays false with `ocr_parked` until the native worker
loads. `/info` exposes worker state and runtime identity. Polling
never cold-loads OCR merely to answer readiness. An already-explicitly-loaded idle
worker's death can schedule one bounded coordinated replacement, fenced against
active inference/park. A failed worker start exposes its error and stays unready.
Parking disables recovery until a new explicit load; shutdown does not revive workers.

Model/package/image/source/target changes invalidate the installation identity.
The app distinguishes parked `can_load` from loaded-worker readiness,
waits before claim on unavailable/incompatible managed pairs, and does not silently
opt out. A cold unmanaged native pipeline also waits before claim.

The CUDA and ROCm source overlays start the UI activation controller. For a
manually qualified offline source runtime, add `docker-compose.ocr-qualification.yml`
after the selected accelerator overlay and start `paddleocr-service`. That
overlay mounts `server_data/ocr-runtime` and selects the explicit runtime.
Configure the app's external OCR endpoint separately when using that setup.
Prepared installations select their runtime through the release manifest and
image identities. Ordinary OCR uses text-only LLM
formatting and does not fall back to image inference. See
[worker drain and explicit adoption](../UPGRADE_AND_VERIFY.md#disable-pause-and-recover).

The initializer and OCR entrypoint retain PID1 with `init: false`. Keep that
supervision and its owned native processes. An exclusive installation lock
excludes maintenance from serving, and per-role locks exclude duplicate native
roles. Durable markers survive hard death, including retained legacy Ollama
markers from prior bundles. `initialize maintain --force` requalifies only under
the ordinary lock and crash fences. Ordinary startup preserves these markers.
After hard death, externally confirm all managed containers recorded by the
markers and their descendants are gone. Only then archive the applicable
`/coord/installation.active-*.json` markers and retain the installation and role
lockfiles. Preserve the user's volumes through recovery. The installation fence
and the app's model-handoff journal are separate.

## Internal model ownership

With `OCR__MANAGE_MODEL_LIFECYCLE=true` (default), Linux app processes, uploads,
benchmarks including direct vision, and supported standalone clients share one
advisory lock with topology-specific durable journals. `OCR__COORDINATION_PATH` defaults to
`gpu-handoff.lock` beside `VISION_BILL_CONFIG_PATH`; Compose uses its existing
`/app/config` bind mount. Every process must use the **same underlying file**, even when host/container
paths differ, and agree on the OCR resource identity. Shared-GPU processes also
agree on the LLM identity. Keep the directory private to Vision Bill and use
local storage supporting flock. Before work on a
shared GPU is admitted, explicitly adopt the configured pair and selected models with
the operator CLI. The binding requires a dedicated endpoint and disabled eager
loads. See the [adoption and endpoint instructions](../OPENAI_COMPATIBLE.md).

One owner loads OCR, processes every receipt photo, parks OCR and confirms native
child exit before formatting, repairs and classification. On a shared GPU, that
owner also releases and verifies its exact owned LLM model IDs before loading OCR
and before releasing ownership. On separate GPUs, the LLM endpoint manages its own
model residency. `/lifecycle/claim`, `/load` and `/park` (all under the
`/lifecycle` prefix) and `/ocr` use generation-fenced owner tokens; stale requests
cannot park/reload a successor. These internal routes must stay on the private
application network. No device-exclusive mode, foreign process termination, or
unloading of every model returned by provider discovery is used.

Use a **dedicated Vision Bill LLM endpoint** for managed sharing; the backend cannot
identify which client owns a shared resident model. Unrelated GPU programs and
unrelated model IDs remain allowed; their memory use can still cause capacity
failures. OpenAI-compatible inference has no standard unload contract. Its default
`LLM__MODEL_LIFECYCLE=none` is rejected for managed sharing; explicit `llamacpp`
enables the dedicated router protocol and waits for exact-ID `unloaded` confirmation.
A successful unload POST, sleeping state or missing model is insufficient. See
[OpenAI-compatible backend setup](../OPENAI_COMPATIBLE.md). Pure LLM OpenAI operation
remains supported. `OCR__MANAGE_MODEL_LIFECYCLE=false` is used by explicit CPU/replay
software stubs;
it makes no GPU handoff claim. Do not recommend it for a native shared pair; a cold
unmanaged pipeline waits before claim and cannot activate the parked worker.

Managed forced-OCR benchmarks, standalone OCR and frozen OCR-enabled uploads
require current production OCR selected (canonical nonblank endpoint or legacy
`OCR__ENABLED=true`) and
`OCR__MANAGE_MODEL_LIFECYCLE=true` on every app worker. Otherwise ordinary direct
LLM work would not share their ownership. Forced qualification/execution fails
before any load; previously frozen managed uploads wait before claim without
spending attempts or rewriting their saved intent. Benchmark capabilities mark
managed OCR unavailable in this configuration. Pure LLM deployments and explicitly
unmanaged CPU/replay modes remain supported.

Creating a managed OCR benchmark explicitly loads the worker, reads its profile
and runtime fingerprint, and parks it again. The frozen identity excludes
transient residency. Execution rechecks it before recording a result.
Recreate old benchmark manifests after changing this lifecycle contract.

Failed teardown or unconfirmed unload blocks the next phase. A crashed/cancelled
or failed model request whose completion is unknown leaves `unsettled=true` in
the journal, even if a momentary residency probe looks idle. The 600-second
attempt deadline includes cleanup. Bounded cleanup uses time reserved within that
deadline. If settlement cannot be confirmed, pending/unsettled state stays durable
for the next owner's confirmed claim/release barrier. A late response cannot
clear a successor's journal. An upload with saved complete observations can finish
later without further inference once ownership is reconciled and the original
source, generation and execution policy still match. See
[upload recovery](../MULTI_IMAGE_RECEIPTS.md#complete-observations-and-optional-labels).
For manual recovery, stop/drain all Vision Bill app/standalone workers and its
sidecar, then establish that the dedicated LLM endpoint has no outstanding
requests (restart only that dedicated endpoint if necessary). Unload/verify only
the journal's owned model IDs using `/api/ps` for Ollama or exact-ID
`GET /models` status `unloaded` for llama.cpp. Only after those checks, with all owners stopped,
archive/reset only the adjacent `.json` handoff journal and restart parked services;
retain the lockfile inode so all participants continue locking the same file.
Never reset ownership while workers are active, delete live lockfiles or restart a
shared foreign runtime. Installation crash markers require the stricter external
all-managed-containers-removed procedure above; clearing the app journal does not
clear that fence.

The historical CPU PP-OCR comparator has an independent hashed build target in
`scripts/ocr_harness/Dockerfile`, selected with
`python3 -m scripts.ocr_harness build --only pp-medium`; it is not a production
sidecar or native v2 qualification source. See the [harness](../scripts/ocr_harness/README.md).

CPU-only unit tests and `docker buildx build --check` validate contracts/recipes,
not native build success, device kernel support or GPU qualification.

Official CUDA lock resolution, native CUDA/ROCm builds, physical inference and
qualification, hardware quality/latency/VRAM and current CPU comparator installation/
inference remain UNEXECUTED here. No VRAM/performance blocker was established.
Historical numerical reports remain historical; see the
[accepted software evidence](../UPGRADE_AND_VERIFY.md#verification-evidence).
