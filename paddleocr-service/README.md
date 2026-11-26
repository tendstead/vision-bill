# PaddleOCR-VL 1.6 installation and operation

These are Linux x86_64/Python 3.12 **qualification candidates**. Neither CUDA nor
any Radeon target has been hardware-qualified here. Do not advertise a target
until its real `qualification.json` has `qualified: true`. No weights or real
receipt fixtures belong in Git or the Docker build context.

## Fixed extraction policy

Each photo goes through the native PaddleOCR-VL-1.6 pipeline once. Vision
Bill runs batch size 1 with internal queues off. The pinned PaddleX
pipeline enables layout analysis and leaves document preprocessing off by
default. Vision Bill enables the orientation classifier for photographed
receipts and leaves geometric document unwarping off. [Official pipeline configuration](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml) · [Installed receipt profile](profile.json)

| Stage | What it does for a receipt |
| --- | --- |
| Orientation | After applying the photo's EXIF metadata, PP-LCNet_x1_0_doc_ori reads the page content and predicts a quarter-turn (0°, 90°, 180° or 270°). [Model guide](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/docs/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md) |
| Layout | PP-DocLayoutV3 locates 25 types of document regions and predicts their reading order. PaddleX crops the regions and can merge adjacent blocks before recognition. [Model guide](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md) |
| Recognition | PaddleOCR-VL-1.6-0.9B transcribes eligible crops with prompts for text, tables or formulas. The receipt profile leaves chart and seal recognition off. [Pipeline guide](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/docs/pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.en.md) |

The pinned pipeline normally gives each recognized region 112,896–1,003,520
input pixels. Vision Bill raises the maximum to 2,000,000, including for table
crops. PaddleX resizes each crop to its patch grid. A page with several large
regions can therefore exceed 2 MP of total VLM input. Generation can produce
up to 4,096 new tokens **per crop** and can stop earlier. The receipt formatter
has a separate output budget and model context. [PaddleX region sizing and generation](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/paddlex/inference/pipelines/paddleocr_vl/pipeline.py#L498-L533)
· [Per-task pixel defaults](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/paddlex/inference/pipelines/paddleocr_vl/pipeline.py#L663-L684)

The CUDA worker uses BF16 for the VLM text decoder on a selected GPU with
compute capability 8.0 or higher. Turing (SM 7.5) and ROCm use FP32.
The visual encoder, projector, orientation and layout models retain
FP32. Precision selection leaves the region and token limits unchanged. The model guides report
individual inference timings on Tesla T4 (orientation) and A100
(layout), excluding pre- and postprocessing. They supply no end-to-end timing
for this receipt profile on a desktop GPU. [Pinned native predictor](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/paddlex/inference/models/doc_vlm/predictor.py#L55-L78)

Original bytes/hash/photo identity survive first-frame/EXIF conversion and
white RGB transparency composition (`white-document-rgb-v2`). Actual preprocessor
pixels become a separately hashed working PNG with transform metadata. Native v2
responses contain this PNG and recognized blocks in **working coordinates**.
The transform records the observed rotation angle, working coordinates and
`unwarping_model=null`. Its `inverse_mapping` field remains `null`. Raw diagnostic
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
PaddlePaddle inference engine. Layout, orientation and VLM recognition
execute in its disposable native worker. Vision Bill uses its private HTTP API.
The operator's separate text or vision LLM endpoint, including a remote
llama.cpp server, formats the completed OCR text. PaddleX also offers an
optional inference server (such as vLLM) for its own VLM recognition stage.
Vision Bill runs that stage locally in the OCR worker. The LLM endpoint
setting controls the formatter. [PaddleX VLM inference services](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/docs/pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.en.md#3-using-vlm-inference-services)

## Dedicated OCR from source

By default the source launcher starts the app and waits for an administrator
to request the GPU build from Settings. Use `--prepare-now` to build immediately.
Both modes bind the approved inputs to the exact image and selected device.
Run the launcher from the repository root with
Python 3.12+, a local Linux x86_64 Docker engine and Docker Compose 2.30+.
Automatic fixture preparation uses `uv`. CUDA builds also use `make`. A cold
CUDA build needs access to the official Paddle wheel host and dependency
sources. Keep the validated inputs for later builds. Prepare the
app's private `.env` once and reuse its Compose project name, app settings and
data on subsequent starts. The command supplies the OCR image and bundle metadata
automatically. [Source app configuration](../README.md#advanced-develop-from-source)

### Start Vision Bill first and build OCR from Settings

Choose the accelerator on the host. From the original checkout, run one of
these commands for a first local OCR registration. The optional `--defer-build`
flag explicitly selects this default.

| Accelerator | Start the app and wait for activation |
| --- | --- |
| NVIDIA CUDA | `python3 scripts/ocr_source_up.py cuda` |
| AMD ROCm (experimental) | `python3 scripts/ocr_source_up.py rocm` |

The launcher starts Vision Bill and PostgreSQL, or verifies and reuses the
existing base installation in the same Compose project. Keep its host process
running to handle the first activation. To run it in the background, for example
with ROCm, use the following command from your Linux or WSL shell.

```bash
(umask 077; nohup python3 -u scripts/ocr_source_up.py rocm --defer-build > ~/vision-bill-ocr-build.log 2>&1 < /dev/null &)
```

Sign in as an administrator and turn on **Settings → Dedicated OCR**. The host
process uses the GPU and driver selection described below, builds and verifies
the native image, and starts only the OCR controller. Settings shows when the
request is waiting or has failed. The host log shows the build steps. The app
and database stay available. On separate GPUs, the configured vision LLM can process
receipts while OCR builds. On a shared GPU, activation still requires the LLM
handoff shown in Settings. The controller then verifies the pinned models and
downloads missing assets. Activation sends no test images. Docker access stays with the host process; the app
container receives no Docker socket.

AMD Dedicated OCR is experimental on native Linux and Windows with WSL 2.
A cold TheRock and Paddle source build can take hours. Upstream estimates
about 200 GB free for TheRock alone. Allow additional temporary disk space
for Paddle, Docker layers and the final image on the filesystem used by
Docker. On retry the same BuildKit builder can reuse completed stages while
its cache is retained. [Build inputs and disk requirements](#build-inputs)

To update the checkout before the first activation, stop the waiting launcher
(Ctrl-C in its terminal, or stop that specific background process). Update the
original checkout, then rerun the same `--defer-build` command with the original
project and GPU selection. If the revision changed, the launcher refreshes the
app image and waits again. The OCR image build still requires activation in Settings.

To build the native image immediately, use the vendor-specific commands below.
Once the managed installation exists, use its restart and upgrade procedures.
[Enable and restart](#enable-and-restart) · [Upgrade an existing source installation](#upgrade-an-existing-source-installation)

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
Docker GPU access. `OCR_CUDA_DEVICE_ID` selects the host GPU by index or UUID
and defaults to `0`. When the host driver exposes the selected GPU, the source
launcher can detect an unsupported architecture or driver before downloading
build inputs. On a multi-GPU host where the numeric selection is ambiguous,
set the full `GPU-…` UUID. When the host probe is unavailable, the launcher
checks the selected GPU inside the built image. Source setup records the
full UUID observed there and the service rechecks it at runtime. A changed
device ordering cannot silently move OCR to a different GPU. Run the launcher
from this checkout.

```bash
python3 scripts/ocr_source_up.py cuda --prepare-now
```

The launcher binds the official model inventory to the selected image. Five
sample images are available for explicit release qualification. The first admin
activation downloads the pinned model files and verifies their contents. The
Windows or Linux host supplies its own GPU driver. Installing these models
does not install host graphics drivers or firmware. Automatic CUDA source preparation
uses the official CUDA 12.9 candidate by default. Set `OCR_CUDA_CHANNEL=cu126`
to select CUDA 12.6. Use a host driver that supports the selected runtime and
check driver compatibility before starting. PaddleOCR lists compute capability
7.0+ and CUDA 11.8+ for PaddlePaddle generally. Vision Bill's preflight accepts
SM 7.0 (Volta) and newer with a CUDA 12-capable driver for its CUDA 12 wheels.
The project's consumer qualification targets start at GeForce RTX 20 (Turing,
SM 7.5). PaddleOCR's separate Blackwell instructions use
a driver supporting CUDA 12.9 or later. For a fresh RTX 50-series install,
use Vision Bill's default `cu129`. [PaddleOCR device requirements](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html#environment-requirements) · [Blackwell setup](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html) · [CUDA build inputs](#build-inputs)

On WSL 2 with a selected CUDA SM 12.0 GPU (for example, an RTX 5080), each
cold OCR worker assembles the PaddleOCR-VL checkpoint in system RAM, then
copies the weights to the selected GPU. Allow temporary RAM headroom in
WSL while it loads. Other GPU candidates assemble the weights on the GPU.
Receipt inference runs on the selected GPU on both paths.

The OCR container reports runtime and model verification time, effective component precision, and total cold-load time in `ocr_worker_load`. The checkpoint loader emits `ocr_vlm_checkpoint_load`. On the WSL host-memory
path, `stage=assemble` is checkpoint assembly, `stage=transfer` is the copy to
the GPU and `stage=ready` means the VLM weights are ready. The receipt's status
in the queue shows when processing finishes.

### AMD ROCm on Linux or Windows with WSL 2

Dedicated OCR on AMD is experimental on both native Linux and WSL 2. The source
recipe includes targets for RDNA2 and later. Compatibility depends on the exact
board, OS, ROCm version and host driver. Check the applicable AMD compatibility
table before choosing either host. A listed GFX build target alone does not
establish vendor support or successful receipt inference.
[Radeon Linux compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html) · [Radeon WSL compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)

**Native Linux.** Install a compatible AMD kernel driver and Docker Engine.

Use the host's `/dev/kfd` and an AMD `/dev/dri/renderD*` node. The source
launcher selects the sole AMD KFD render node automatically. With several
AMD nodes, set `OCR_RENDER_NODE` to the one you want. The launcher reads its
GFX architecture from the host's KFD topology and derives the two container
group IDs from the device nodes. Set `OCR_AMDGPU_TARGETS` to the board's known
GFX target if topology cannot identify it. The built image checks the actual
HIP target before startup. [ROCm build inputs](#build-inputs)

```bash
python3 scripts/ocr_source_up.py rocm --prepare-now
```

**Windows with WSL 2.** Install AMD's Windows driver for WSL and use a supported
Ubuntu distro with Docker Engine inside it. The Windows host supplies the GPU
driver. Check that the distro exposes `/dev/dxg` and
`/usr/lib/wsl/lib/libdxcore.so`. Run the same source command inside
Ubuntu. The launcher selects the WSL build and adds `docker-compose.rocm-wsl.yml`
to the ROCm configuration. For one AMD GPU at HIP index `0`, it can read the
GFX target from an existing `rocminfo`. If discovery is unavailable, or there
are multiple GPUs, set `OCR_AMDGPU_TARGETS` to the selected board's actual GFX
architecture and `OCR_ROCM_DEVICE_ID` to its HIP index. The Windows display
index can differ. For example, on a machine where HIP device 0 is `gfx1100`:

```bash
OCR_AMDGPU_TARGETS=gfx1100 OCR_ROCM_DEVICE_ID=0 python3 scripts/ocr_source_up.py rocm --prepare-now
```

The launcher needs an unambiguous selection before building. The host driver
and firmware are installed separately.
[AMD WSL setup](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html) · [Pinned ROCm DXG design](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/docs/development/wsl_rocdxg.md)

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
The controller checks the image, GPU and package metadata, provisions the three
pinned model directories and publishes the verified installation.
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

During a source build, `ocr_source_phase` reports the CUDA or ROCm image
build, native image configuration verification and app image build. The final
line for each phase includes `elapsed_s` and a status of `complete` or
`interrupted`. Use `BUILDKIT_PROGRESS=plain` to see individual build steps.
For ROCm, check the launcher's `Local ROCm build` line for the selected
transport and GFX targets. Inside the build, `ocr_rocm_phase` reports
wall-clock seconds for preparing TheRock (`therock_prepare`), compiling LLVM
(`llvm_compile`) and finishing the SDK (`therock_finish`). Source preparation
(`therock_sources`) is part of `therock_prepare`. After LLVM, separate steps
configure and build RCCL (`rccl_configure`, `rccl_compile`), configure MIOpen
and build its prerequisites (`miopen_configure`), then compile MIOpen
(`miopen_compile`). The remaining SDK build reports `sdk_compile`. Paddle reports `paddle_sources` and
`paddle_compile`. A BuildKit cache hit shows `CACHED` and has no new phase
timing.

`ocr_rocm_checkpoint` reports the directory `path` and its `allocated_bytes`
before cleanup. Compare the build tree after `prepare`, `llvm` and `finish`.
Successive measurements can include the same files. The image retains the
measurements in `/opt/ocr/build-evidence/therock-checkpoints.txt`.
The launcher's `ocr_source_image` line gives the image ID and Docker's
logical `image_size_bytes`. For disk usage on this builder, inspect the
BuildKit cache separately. Downloaded models occupy their own volume.

If the native image finished building before initial preparation was interrupted,
use `--reuse-native-image` with the local `source-build-…` tag printed by the launcher.
It checks the selected image against this checkout and continues preparing the app.
For CUDA it reads the saved image's lock hash offline and requires a match in
the validated checkout lock and attestation or a retained private build
snapshot for that channel. Reusing the native image skips wheel resolution
and download. Any remaining app build steps still need their own inputs.
For immediate recovery of an interrupted first installation, combine
`--prepare-now` with `--reuse-native-image SOURCE_BUILD_TAG`. For an
existing installation, follow the [upgrade procedure](#upgrade-an-existing-source-installation).

### Read the logs for a receipt

Use the Docker engine that owns the installation. For Docker Desktop use
PowerShell. For AMD on WSL use the terminal in the Ubuntu distro running Docker
Engine. Identify both running containers by their Compose project and service.
Source service names are `vision_bill` and `paddleocr-service`. Combined bundles
use `app` and `ocr`. Substitute the two container IDs below, keeping the same
project for both.

**PowerShell**

```powershell
docker ps --filter label=com.docker.compose.service --format 'table {{.ID}}  {{.Names}}  {{.Image}}'
$AppContainer = Read-Host 'App container ID'
$OcrContainer = Read-Host 'OCR container ID'
docker logs --since 30m --timestamps $AppContainer 2>&1
docker logs --since 30m --timestamps $OcrContainer 2>&1
```

**Linux or WSL shell**

```bash
docker ps --filter label=com.docker.compose.service --format 'table {{.ID}}  {{.Names}}  {{.Image}}'
read -rp 'App container ID: ' app_container
read -rp 'OCR container ID: ' ocr_container
docker logs --since 30m --timestamps "$app_container" 2>&1
docker logs --since 30m --timestamps "$ocr_container" 2>&1
```

Use `docker logs --since 1m --follow --timestamps` with one container ID in
each terminal to follow a queued receipt. Match timestamps with its queue entry.
An HTTP 200 from worker load or readiness records startup. Check the receipt
request, worker shutdown, formatter and final queue status to locate a later
failure. An HTTP 504 records a deadline. It does not establish an allocation
failure or receipt completion.

During activation, `ocr_model_prepare` records verification and whether each
model was already cached or newly installed. During official asset transfers,
`ocr_model_download` logs progress as data arrives at roughly 15-second
intervals. `received_bytes` and `mean_mbit_s` show bytes received and average
megabits per second for that file. Small files may finish between updates.

For a queued job, use the scheduler's `submission_id` in the app log to follow
the submission across pages and formatting. Each `ocr_page` dispatch records
its `request_id`, photo index and time budget. Use the same request ID to match the app’s
dispatch and response to the native worker dispatch, child start and completion
or failure. The worker generation identifies the process that ran the stages. After the worker parks,
`ocr_formatter` records the start and completion of text formatting.

In the OCR log, `ocr_worker_request` identifies admission, native execution
and result reading. Compare the phase, elapsed time and remaining budget.
After a failure, `cleanup_safe` reports whether cleanup was confirmed.
During handoff, `ocr_worker_park` records the request, the start of worker
teardown after acquiring the locks, and the final parked or failed state.
`ocr_receipt_stage` follows normalization, orientation, layout,
VLM crops, conversion and publication. At INFO look for the first VLM call,
progress every 16 completed regions and the final `regions_completed` count.
DEBUG adds each crop's start and finish. `worker_peak_rss_kib` measures peak
host memory over the worker's lifetime. When present, `gpu_allocated_bytes`
measures allocations tracked by Paddle. Neither value measures total device
VRAM use.

The application logger uses `API__LOG_LEVEL`. The native controller and worker
use `OCR_LOG_LEVEL`. Both default to `INFO`, which omits successful health polls. `DEBUG` shows those
polls. For more detail set the relevant
level to `DEBUG`, recreate the affected service using the installation's
launcher, and return the level to `INFO` after diagnosis. For source Compose,
set the levels in the project's private `.env`. In prepared bundles, put the
app setting in `runtime.env`. For a manual start put the OCR setting in the
Compose `.env` or shell. When using a bundled launcher, set it in the launcher’s
environment. These settings control Vision Bill's own loggers.
Keep diagnostic logs private and remove credentials, receipt content and
identifying details before sharing excerpts.

### Upgrade an existing source installation

Run from the original checkout. Back up the installation and let active jobs
finish. If an OCR load timed out on a separate GPU, keep that topology and
switch **Dedicated OCR** off on the old stack. Wait for Settings to confirm
it is off. An installed service can park a failed worker while leaving the
remote LLM loaded. After upgrading, turn OCR back on and resume the existing
receipt in the queue. If it reports changed settings, review them and choose
the current-policy retry for that receipt.

An ordinary managed `--upgrade` can reuse the exact local OCR image when its
recorded build recipe, source and runtime locks, and build inputs still match.
For CUDA that includes the selected channel and validated wheel lock. For ROCm
it includes the selected GFX target and host transport. The launcher checks the
image identity and repeats the binding check against the physical GPU offline
before starting the upgraded deployment. An app-only change can therefore skip
the native image build. Keep the CUDA attestation and its private build snapshots.

An older deployment without this build record goes through the normal native
build and verification once. A change to the shared native runtime lock can
also require one such pass on either accelerator, including CUDA when only the
ROCm recipe prompted the lock update. Matching BuildKit layers and verified
wheel bytes can still be reused. The model volume is separate. Activation
verifies the required trees and can reuse valid local archives to restore files.

Stop the old deployment with its original project, profile and complete
Compose file set. Keep its named volumes, private settings and bind-mounted
files. For a CUDA deployment with the checkout's default project name:

```bash
docker compose --profile ocr -f docker-compose.yml -f docker-compose.cuda.yml \
  -f server_data/ocr-source/cuda/compose.local.json down
# Update this checkout to the desired revision, keeping server_data and .env.
python3 scripts/ocr_source_up.py cuda --upgrade
```

If you originally selected a custom project name, add the same `-p NAME` to
the stop command. For an existing **ROCm** source installation, choose the
stop command for the host that owns the original Docker project.

**Native Linux**

```bash
docker compose --profile ocr -f docker-compose.yml -f docker-compose.rocm.yml \
  -f server_data/ocr-source/rocm/compose.local.json down
```

**Windows with WSL 2** — open the Ubuntu distro running Docker Engine, change
to the original checkout and run this in its shell.

```bash
docker compose --profile ocr -f docker-compose.yml -f docker-compose.rocm.yml \
  -f docker-compose.rocm-wsl.yml -f server_data/ocr-source/rocm/compose.local.json down
```

On either ROCm host, update that checkout to the desired revision, keeping
`server_data/` and `.env`. Run the upgrade in the same shell. This Bash example
retains the build log and propagates a failed build's exit status.

```bash
set -o pipefail
BUILDKIT_PROGRESS=plain python3 scripts/ocr_source_up.py rocm --upgrade 2>&1 | tee -a ~/vision-bill-rocm-upgrade.log
```

Moving from the old single-step TheRock build to this recipe requires one
rebuild to create the new intermediate layers. A normal `--restart` starts
the recorded images. If the initial `--prepare-now` build failed, update the
checkout and rerun `python3 scripts/ocr_source_up.py rocm --prepare-now`
with the same GPU selection. [ROCm build inputs and cache](#build-inputs)

**If the native build finished but the app build failed on `server_data/config`**,
keep that directory's private permissions. The app image builds from repository
inputs and excludes `server_data/`. Update the original checkout to the revision
with the build-context fix. Leave the old project stopped and retain its volumes,
private `.env`, deployment record and release directory.

Find the completed native tag in the failed build's output. To check which
CUDA tags still exist on the same Docker engine, run
`docker image ls --filter 'reference=vision-bill-ocr-cuda:source-build-*'`.
Replace the placeholder below with the 32 hexadecimal characters from that
specific build. For ROCm, use `rocm` in both the accelerator and image name.

```bash
python3 scripts/ocr_source_up.py cuda --upgrade \
  --reuse-native-image 'vision-bill-ocr-cuda:source-build-<saved-build-uuid>'
```

The launcher verifies the native candidate against the checkout and selected
GPU, then builds the app. The usual project ownership and handoff checks still
apply. If the tag is missing or its sources no longer match, use the ordinary
`--upgrade` command above to rebuild from the current checkout.

**If the old app or OCR image was deleted**, use the same `--upgrade` command.
Keep the stopped original project, its volumes and ownership claim, the
private `.env`, and the source deployment record and release directory in
`server_data/ocr-source/<accelerator>/`. The launcher verifies the original
coordination volume and checks both handoff journals with a read-only CPU
container. It uses the repository-pinned Python image and pulls it if missing.
A clean result lets the launcher rebuild the app and OCR images with the
recorded device and model sources.

* **Docker or inspector error.** Read Docker's reported error. Check access to
  the pinned image and the original coordination volume, then retry the upgrade.
* **Lock held.** Identify and drain the original owner before retrying.
* **Outstanding handoff.** Preserve both journals. Recover the original stack
  and settle the reported work with its original endpoint. For an OCR-only
  pending handoff, switch Dedicated OCR off and wait for Settings to show
  Disabled. Stop the original project and retry the upgrade. If its recorded
  images are missing, use the narrowly scoped retired-worker procedure below
  if applicable. To restart the old generation, restore its exact images first.
* **Invalid journal or ambiguous volume identity.** Keep the original volume,
  deployment metadata and backup together. Investigate the reported ownership
  or integrity issue before proceeding. [Handoff recovery](#internal-model-ownership)

**Recover a retired separate-GPU OCR worker.** A distinct opt-in upgrade path
is available when the original local OCR worker is fully stopped, its deployment
is retired, and only that worker’s separate-GPU OCR handoff remains. Keep the
original project, volumes, source deployment record and release. Back up the
installation before recovery. Drain the original Compose project and take it
down with its volumes retained. Verify that Compose removed its default network
and `postgres_net`. Containers attached to the original OCR volumes must be gone,
including stopped containers.
In the original checkout, use the recorded project and accelerator:

```bash
python3 scripts/ocr_source_up.py cuda --upgrade --recover-stopped-ocr
```

For ROCm replace `cuda` with `rocm`. The launcher checks the original volume
ownership and exclusive locks, saved separate-GPU topology, exact local managed
OCR identity and a single valid pending OCR-only handoff. It preserves the
original journal bytes in a private `gpu-handoff.ocr-only.recovered-<uuid>.json`
next to the journal in the coordination volume. A hard stop can also leave
`installation.active-ocr.json`, which the native supervisor writes before starting
its worker. For a single valid marker from the removed OCR worker, recovery
acquires the original OCR role lock, verifies container removal and archives the
marker before settling the journal. The saved files stay on the coordination
volume. If Settings still shows a failed activation after the upgrade, enable
Dedicated OCR again. Check the existing receipt in the queue before retrying it.
A shared-GPU handoff, unresolved external work, a foreign owner, an initializer
marker, or damaged or unrecognized records require separate investigation.
Use ordinary `--upgrade` for a clean deployment.

If a build fails before the new deployment record is published, fix the build
error and retry `--upgrade`. An available set of original recorded images can
also be started with `--restart`. If a new record was published before
container startup failed, `--restart` retries its recorded **new** images.

For rollback, drain and stop the new deployment, account for writes since the
backup, then restore the previous checkout and its original image tags. Restore
the backed-up `server_data/ocr-source/<accelerator>/` directory, including
`compose.local.json`, `admitted.sha256` and the original release directory.
Restore the matching configuration, OCR activation, coordination and model
state, database, original uploads and working files from the same recovery
point before starting the old generation. [Compatible backout](../UPGRADE_AND_VERIFY.md#backup-and-upgrade-order)

Older profiles used UVDoc to unwarp pages.
A source upgrade accepts the existing four-model approved directory or a
current three-model directory. It binds the three active model records to the
new candidate, retaining their approved sources, package pins and fixture
archive. The original release manifest, digest and UVDoc assets remain
available for recovery. Keep the original photos, stored working PNGs and OCR
observations. New runs retain the 2,000,000-pixel cap per recognition region.
For a before/after comparison, submit the same original photo and
compare each extraction with its own stored working view. Historical latency
or accuracy measurements describe the profile that produced them.

When upgrading a failed initial
installation, switch **Dedicated OCR** on again in Settings. An installation
that was already ready is checked and migrated under the existing ownership
locks. Preparation reuses verified model assets. If the new image deliberately
changes approved package pins or model sources, pass the matching approved
input directory after the accelerator, before `--upgrade`.

CUDA upgrades preserve the channel saved in the original deployment record or
its image configuration. For an older record that lacks usable channel
metadata, the launcher can read the lock from the original image if it still
exists. If neither source identifies the channel, set `OCR_CUDA_CHANNEL=cu126`
or `OCR_CUDA_CHANNEL=cu129` to the intended channel in the shell or checkout’s
`.env`, then retry `--upgrade`. The same setting selects a deliberate channel
change.

A CPU limit previously set with `docker update` belongs to the container that
was replaced. To restore a previous limit of 6.4 CPUs on a CUDA source
installation, run this in the original checkout after the upgrade. Use your
previous limit and select `rocm` for a Radeon installation. The recorded
project name selects the new OCR container.

```bash
accelerator=cuda
overlay="server_data/ocr-source/$accelerator/compose.local.json"
project=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["x-vision-bill-source"]["project"])' "$overlay")
ocr_id=$(docker ps -q --filter "label=com.docker.compose.project=$project" \
  --filter label=com.docker.compose.service=paddleocr-service)
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
| CUDA | GeForce RTX 20, 30, 40 and 50 series | Turing SM 7.5, Ampere SM 8.6, Ada SM 8.9, Blackwell SM 12.0 | Linux, including Docker Desktop with WSL 2 on Windows |
| ROCm | AMD Radeon RX 6000, RX 7000 and RX 9000 | RDNA2 `gfx1030/1031/1032/1034`, RDNA3 `gfx1100/1101/1102`, RDNA4 `gfx1200/1201` | Native Linux or Windows with WSL 2, with the corresponding driver and device access |

These are qualification targets. Release evidence records the architecture,
driver and host environment for actual runs of the configured OCR pipeline.
Both the local source launcher and `make ocr-build-rocm` build for the
selected architecture and host GPU interface. The nine Radeon targets in the
table are recognized for automatic selection. A detected architecture outside
this list needs an explicit `OCR_AMDGPU_TARGETS` for an experimental build.
A developer can explicitly request multiple target architectures for a compatibility build.
[NVIDIA GPU architectures](https://developer.nvidia.com/cuda/gpus) ·
[TheRock targets at the pinned revision](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/cmake/therock_amdgpu_targets.cmake) ·
[Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/)

Evaluate one CUDA bundle against all four NVIDIA generations before splitting
the binaries by architecture. Paddle publishes a 3.3.1 CPython 3.12 Linux x86_64
wheel in its CUDA 12.9 channel. Both CUDA channels keep the existing 3.3.1
pin. The listed 3.4.0 CUDA 12.9 wheel remains a candidate pending verification
of its metadata, content hash and dependency lock. CUDA 12.8 introduced native SM 12.0 compilation.
Verify the wheel and linked libraries, then run device allocation and the
full OCR pipeline on representative hardware.
[Official Paddle CUDA 12.9 wheel index](https://www.paddlepaddle.org.cn/packages/stable/cu129/paddlepaddle-gpu/) ·
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#new-features)

The current publication gate runs full native qualification on one selected
physical device for each OCR variant being published. Its report identifies the
specific device, image, model and runtime. Other architectures and host paths
remain implementation targets until independently measured. The CUDA
`driver_version` field records the driver API version. For comparisons between
installations, also record the installed driver release, VRAM, OS and Docker
version. End-to-end receipt evidence includes OCR completion, worker shutdown
and LLM formatting. [Native qualification](#what-qualifies)

## Build inputs

`runtime-lock.json` records the immutable Ubuntu base, Ubuntu package snapshot,
APT TLS bootstrap image, Paddle/PaddleOCR/PaddleX/TheRock source pins,
compatibility patch hash, Rust archive hash, Radeon compatibility targets and
upstream optional exclusions. The selected build candidate profile is recorded
in build evidence/kernels.json.
The common Python execution and native build locks contain real resolved hashes.
`requirements.txt` is not the production build/install contract; use the selected
Dockerfile and hashed accelerator/runtime locks.
Source packages are built without dependency resolution, using the hashed build
bootstrap. The archived PaddleOCR/PaddleX sources use their pinned release
versions through `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*`.

The checked-in `requirements.cuda.lock` is unresolved. The source launcher uses
a valid matching lock or restores a validated cached resolution for the same
channel and resolver inputs. A cold resolution needs a connected host.
To refresh from the official index, run `make ocr-lock-cuda` for `cu129` or
`OCR_CUDA_CHANNEL=cu126 make ocr-lock-cuda` for `cu126`. The resolver selects the
exact official Paddle GPU 3.3.1 cp312 x86_64 wheel for that channel, inspects its
METADATA, freezes the full closure, hashes the bytes, and emits the lock plus
`requirements.cuda.attestation.json`. Review both outputs and use the same
channel for `make ocr-build-cuda`. Keep the lock, its attestation and the host
cache at `${XDG_CACHE_HOME:-~/.cache}/vision-bill/cuda-wheels`. With a
matching lock and cached SHA-256, the build reuses those bytes locally.
Each CUDA build consumes an immutable `build-…` snapshot in this private
cache containing the lock, its attestation and a hard link to the verified
wheel. Concurrent builds using different channels can therefore share a
checkout. Release builds resolve the official channel afresh and use the same
snapshot mechanism for the wheel, installed lock and release attestation.
The resolver reports verified wheel cache hits or the official download URL on stderr.
During a transfer it reports bytes, elapsed time and average throughput. It makes at most
three requests in total per transfer. It resumes a partial download
when the server supplies a known length, a strong ETag and a matching byte range.
Without a usable validator it restarts at byte zero.
A source upgrade from `936c26d` can need one download of the full Paddle
wheel. With a matching lock, the launcher first checks the previous image's
pip cache for that URL and verifies any bytes it finds. If the wheel is
missing or its hash differs, the launcher fetches it from the attested URL.
An offline upgrade needs the matching lock and verified wheel, along with the
other required build inputs. Starting the previously recorded images with
`--restart` skips this build. BuildKit
can reuse the app dependency layers and, on ROCm, the built TheRock SDK when
only Paddle or the app changes. CUDA and ROCm use separate Python build and
runtime environments. The ROCm runtime retains its SDK libraries and GPU kernel
assets. PostgreSQL uses a separate prebuilt image.
The Docker build requires a valid, resolved
lock and attestation for the official GPU wheel. Validation parses actual
executable requirements, preserves every entry of the pinned common runtime lock,
and requires the complete attestation. Includes, install options, duplicate or
conditional native requirements and comment-only evidence are rejected. The Make
targets provide the pinned packaging parser in an isolated uv tool environment.

`make ocr-build-rocm` builds the TheRock SDK and Paddle from the approved
source pins for the selected targets. A native Linux build uses KFD. A WSL
build also compiles the DXG bridge against pinned Windows headers. The KFD
build skips those WSL-only inputs. The builder fetches the pinned TheRock root commit at depth one
with all its files in the initial transfer. Checking out that commit works
offline. Paddle and the recursive submodules still use `--filter=blob:none`
and need access to their public remotes to retrieve file contents.
Before applying upstream patches, the recipe initializes the recursive
submodules at their recorded pins, including MIOpen's nested dependencies.
It skips the documentation-only root submodule `rocm-docs`. Its exact Git
pin remains in the root gitlink inventory and in `omitted_root_gitlinks`.
The builder checks the source inventory
before building `patchelf` and reports the path and pin of a missing checkout.
BuildKit saves the prepared source/build tree, LLVM, RCCL, MIOpen configuration
with its prerequisites, and compiled MIOpen in separate layers. Retry the same
command on the same builder with the same GFX targets and transport. Completed
layers show `CACHED` while their inputs and cache entries match. A failure
during MIOpen compilation restarts that step. A subsequent failure building
the remaining SDK can reuse compiled MIOpen. The MIOpen SQLite patch is
introduced after its configuration layer, so a change to that patch retains the
earlier dependency builds. These layers keep intermediate build files in the
local builder cache and use additional disk until BuildKit evicts them. Older
recipes that put the whole post-LLVM SDK
build in one step can reuse their completed LLVM layer. Their failed step has
no committed per-target checkpoint. For example, an old failed MIOpen step
showing `443/463` must rerun its uncommitted work. With the new recipe, a
successful MIOpen configuration and prerequisite layer can survive a later
MIOpen compile failure. The DVC cache can also reuse verified assets.

Upstream estimates approximately 200 GB free and several hours for a TheRock
source build. The intermediate layers retain large trees for later retries.
Check free space on the builder's filesystem and use `docker buildx du` to
inspect its cache usage.
[TheRock source build requirements at the selected commit](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/README.md#building-from-source)

`OCR_BUILD_JOBS=auto` is the local ROCm default. Inside the builder it selects
at least one job, limited by available CPUs, the cgroup CPU quota and one job
per 4 GiB of available memory (including the cgroup limit). Set a positive
integer to override it, for example `OCR_BUILD_JOBS=2 make ocr-build-rocm`.
The setting applies to each CMake build and concurrent submodule fetches.
TheRock can run multiple subprojects at once,
so their combined compiler count can exceed this setting. LLVM linking and
background builds each use one job, and Flang compilation uses at most two.
The builder supplies `/usr/bin/python3` for AMD-SMI and `tclsh` for the
profiler's SQLite source generator. Projects that use TheRock's pinned SQLite
receive its headers through CMake. The local SDK omits Mirage and hipDNN
samples. Model weights are provisioned separately for qualification. The
versioned Paddle patch replaces both legacy HIP target lists, uses the flat
SDK and same-pin legacy FindHIP helpers,
requires MIOpen, and stops the forced download of FlashAttention. FP32 runs with
FlashAttention off.

Build evidence is retained under `/opt/ocr/build-evidence`: original gitlinks,
recursive actual checkout HEADs and dirty patches, patch/`.smrev` hashes, native
wheel hashes, CMake caches, compiler commands/log, OS packages, and required
library/payload hashes. New ROCm images store `paddle-build.log.gz` and,
when generated, `paddle-compile-commands.json.gz` after kernel inspection.
Older images can contain the uncompressed files.

For a developer build spanning Linux KFD and Windows
WSL DXG, explicitly request both interfaces and a semicolon-separated **build-only**
target list. This example builds without a GPU:

```sh
OCR_ROCM_TRANSPORT=dual OCR_AMDGPU_TARGETS='gfx1030;gfx1100' make ocr-build-rocm
```

For a local KFD or WSL build using raw Compose, set `OCR_AMDGPU_TARGETS`
explicitly. The Make target and source launcher perform host discovery.
For a direct Dockerfile build, pass the target list as the build argument
`AMDGPU_TARGETS` and the chosen interface as `OCR_ROCM_TRANSPORT`.

The same selected GFX list goes to Paddle's HIP compiler and TheRock's
`THEROCK_AMDGPU_TARGETS`, `THEROCK_DIST_AMDGPU_TARGETS` and
`THEROCK_TEST_AMDGPU_TARGETS`. The pinned RCCL library uses the test target
list even with tests disabled. Earlier builds left that list empty, so TheRock
expanded it to every registered architecture. Before compiling LLVM, the
recipe checks the generated communication, math and ML library toolchains.
RCCL, rocBLAS and MIOpen must have the full selected list. Optional libraries
may use a subset. The optional `rccl-translate` step extracts device code
from `librccl.so` and runs only if RCCL's filtered list includes `gfx1250`.
After LLVM, the recipe configures RCCL and inspects Ninja's kernel compile
and link commands before running them. A missing or extra RCCL device
target stops the build. Successful checks leave
`therock-toolchain-targets.txt` and `therock-rccl-targets.txt` in
`/opt/ocr/build-evidence/`. [Pinned RCCL declaration](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/comm-libs/CMakeLists.txt#L30-L35) · [Upstream test-list default](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/cmake/therock_amdgpu_targets.cmake#L345-L354)

For a build containing only the listed RDNA2 targets, hipBLASLt retains its
host API for rocBLAS and builds its embedded HIP helpers for the selected GPUs.
It skips its tuned device library and clients, avoiding the upstream `gfx1100`
fallback. hipSPARSELt likewise retains its API and selected-GPU helpers for
all nine listed Radeon targets, which upstream excludes from its tuned sparse
library. For mixed profiles, each library builds tuned kernels only for the
selected architectures it supports.

`kernels.json` records the selected profile and the inspected Paddle, rocBLAS
and MIOpen payloads. The build checks cover generated target flags and the
payloads the inspector can read. Compiler backends, source files and package
metadata can still name other architectures. The actual HIP device must
match the image profile at startup. Rebuild and requalify when changing that
profile.

Missing required libraries, incorrect Paddle HIP target flags and parsed missing
required target payloads fail the build. Packed/JIT formats which LLVM cannot
prove remain `offline_status: unverified`, with all inspected bytes hashed.
Only a real selected-GFX FP32 GPU matmul/convolution probe with checked results,
followed by complete model inference and fresh replay, can close that gate.
Unknown compile/API/kernel/probe failures are failures, never downgrades.

## Local qualification inputs

Use ignored `server_data/ocr-runtime/` for the private corpus, model directories,
and outputs; `server_data/ocr-cache/` is the actual `PADDLE_PDX_CACHE_HOME` mount.
Provision the three exact approved model directories separately into private
`server_data/ocr-runtime/models/` before qualification (Compose exposes them under
`/var/lib/ocr/models/`). This repository has no automatic model-download Make target;
copy/provision the intended model bytes using the official model distribution on
the target host, then retain their identities in qualification. The mounted cache
is not a substitute for complete explicit model directories:

- `PP-DocLayoutV3`
- `PP-LCNet_x1_0_doc_ori`
- `PaddleOCR-VL-1.6-0.9B`

Create `candidate.runtime.json` matching `ocr_service.profile.RuntimeSettings`:
`accelerator`, `gpu_index` (inside the container), all three `model_dirs` using
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
file `sha256`. Use five distinct inputs. For `table` and `formula`, also set
`required_labels = ["table"]` or `["formula"]`. Their native results must contain
that structure. During physical qualification, the qualifier selects the
`upright` case from this manifest for its cold smoke test. Normal worker
startup performs no fixture inference.

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

Use `accelerator=rocm` for Radeon. On native Linux, set `OCR_RENDER_NODE`,
`OCR_RENDER_GID` and `OCR_VIDEO_GID` to the actual host node and groups.
On WSL, add `-f docker-compose.rocm-wsl.yml` for the stop command and select
the GPU with `OCR_ROCM_DEVICE_ID`. The Make target selects the WSL overlay. For NVIDIA,
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
hooks. The static layout and orientation models are audited separately and every floating
exported parameter must bind by its exact preserved name to an observed GPU
input. Shape matching, arbitrary zipping and many-to-one binding are forbidden.

Each corpus case executes twice during collection. Every individual forward must
have the same complete floating hook record set; a corpus union is never accepted.
Unproven CPU floats, optimized-away/renamed weight bindings, varying branch
coverage, missing callbacks or unstable fresh-process identifiers fail the gate.
Two independent fresh workers replay each case, first qualifying the sealed
upright smoke through the qualification adapter. Reports distinguish startup, actual
cold smoke, first case **after smoke**, and warm case timings; they do not claim
cold case inference after warming with smoke. Dynamic VLM parameters must pass their component precision and device checks.
Both static models must pass FP32/device checks on each replay.

Only a full pass publishes `qualification.runtime.json` and a successful report.
Publication errors keep qualification false and remove stale or partially written
serving settings; the CLI exits nonzero. A failure report is written when storage
permits. `qualification.json` records exact profile,
model/image/runtime identities, actual device/GFX, source hashes, timing and audit
counts/digests. It contains no OCR text or images. Only the actual target is added
to `qualified_targets`. Additional claims need their own physical evidence.

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

Standalone bundles provide explicit file-only setup. The profiled
`ocr-prepare` service runs `ocr_service.initialize prepare`, checks the image
and selected GPU, provisions approved local or HTTPS model assets, and writes
verified serving state without processing images. On startup `ocr-init check`
accepts either a prepared installation or a physically qualified one. It
validates the selected state and remains passive. Repeating `prepare` can
repair damaged model files in an owned schema-2 cache with intact metadata.
Corrupt selection metadata, damaged qualified evidence and unknown versions
block file-only preparation. Preserve the original state before restoring a
verified backup. For a known damaged legacy certificate, explicit forced
physical qualification can create a new selection after asset integrity is
restored. [Standalone recovery](../distribution/README.md#native-cache-check-and-maintenance)
The first standalone receipt owns the load/infer/park cycle.

For optional hardware evidence, `ocr-maintenance` runs
`ocr_service.initialize maintain` and exercises the qualification corpus.
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
active inference/park. If the worker reaches its own startup limit, `/ready`
returns HTTP 503 with `ocr_startup_timeout`. A confirmed GPU allocation failure
reports `ocr_gpu_out_of_memory`. The app can mark an attempt `timed_out`
while native startup is still running. The worker can reach its startup
limit later. After a failed managed load, the service sets
`can_claim_after_failed_load` only once it confirms the child has exited.
A new owner can then claim the next bounded queue attempt. Polling alone
does not start a worker. If exit cannot be confirmed, the GPU remains held.
Parking disables recovery until a new explicit load; shutdown does not revive workers.

Model/package/image/source/target changes invalidate the installation identity.
The app distinguishes parked `can_load` from loaded-worker readiness,
waits before claim on unavailable/incompatible managed pairs, and does not silently
opt out. A cold unmanaged native pipeline also waits before claim.

The CUDA and ROCm source overlays start the UI activation controller. For a
manually qualified offline source runtime, add `docker-compose.ocr-qualification.yml`
after the selected accelerator overlay. For ROCm on WSL, append
`docker-compose.rocm-wsl.yml` last to include the DXG bridge. Start
`paddleocr-service`. The qualification overlay mounts `server_data/ocr-runtime` and selects the explicit runtime.
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
and the app's model-handoff journal are separate. A stopped source deployment
can use the [retired-worker procedure](#upgrade-an-existing-source-installation)
when only the local OCR worker and its matching journal need recovery.

## Internal model ownership

With `OCR__MANAGE_MODEL_LIFECYCLE=true` (default), managed OCR uploads,
OCR benchmarks and supported managed standalone clients coordinate through one
advisory lock with topology-specific durable journals. Direct-vision benchmarks
use the coordinator when the LLM and OCR share a GPU. On separate devices,
direct-vision benchmarks run independently of the OCR worker.
`OCR__COORDINATION_PATH` defaults to
`gpu-handoff.lock` beside `VISION_BILL_CONFIG_PATH`; Compose uses its existing
`/app/config` bind mount. Every participating process must use the **same underlying file**, even when host/container
paths differ, and agree on the OCR resource identity. Shared-GPU processes also
agree on the LLM identity. Keep the directory private to Vision Bill and use
local storage supporting flock. For a shared GPU, activating Dedicated OCR in
Settings adopts the configured pair and selected model after the administrator
confirms the handoff conditions. For a separately managed OCR endpoint sharing
the LLM's GPU, adopt the pair and selected models with the operator CLI.
Shared-GPU adoption requires a dedicated LLM endpoint with eager loading disabled.
[Adoption and endpoint instructions](../OPENAI_COMPATIBLE.md)

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

In the November 21, 2026 WSL 2 CUDA run, the checkpoint log showed 210 seconds
from host assembly to transfer and 3.45 seconds from transfer to weights ready.
Managed load and the following readiness check both returned HTTP 200. The
subsequent receipt request returned HTTP 504. This run establishes cold-load
progress on one installation. Receipt quality, completed latency, peak VRAM
and release qualification still need measurements on the actual target.
[Accepted software evidence](../UPGRADE_AND_VERIFY.md#verification-evidence)
