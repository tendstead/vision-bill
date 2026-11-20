# Prepared installations and distribution maintenance

Install a prepared bundle by downloading a verified release, configuring the
selected variant and running Docker Compose. Public native bundles require a
release run with physical GPU qualification. To build a local candidate before
publication, use the source setup. [Dedicated OCR from source](../paddleocr-service/README.md#dedicated-ocr-from-source)

## Choose an installation

The Linux AMD64 bundles pin container images by registry manifest digest. They
contain `compose.yaml`, `release.json`, `SHA256SUMS`, `check-compose.sh`, a runtime
environment template and the selected assets and evidence under `release/`.

| Variant | Included services | Operator supplies |
| --- | --- | --- |
| `app` | Config initializer, PostgreSQL, app | External LLM endpoint and explicit model ID. Optional OCR endpoint. |
| `ocr-cuda` | Cache gate, maintenance and independent OCR | Qualified NVIDIA device and approved OCR assets. |
| `ocr-rocm` | Cache gate, maintenance and independent OCR | Qualified selected AMD device and approved OCR assets. |
| `app-ocr-cuda` | App, PostgreSQL and an idle Dedicated OCR controller | Qualified NVIDIA device and a compatible LLM endpoint/model. |
| `app-ocr-rocm` | App, PostgreSQL and an idle Dedicated OCR controller | Qualified selected AMD device and a compatible LLM endpoint/model. |

Provision the chosen model on your own Ollama or OpenAI-compatible endpoint.
Installation starts the listed services. Combined bundles fetch the four model
snapshots or archives from pinned HTTPS sources after setup is requested.
Benchmarking is a separate operator action.
Native read-only routes and lifecycle control belong on trusted private networks.

## Start a verified Windows bundle

Extract a verified `app-ocr-cuda` bundle. Open Docker Desktop with its Linux
WSL 2 engine and a supported NVIDIA Windows driver. The bundled launcher uses
Docker Desktop directly. After verifying the release and checksum manifest,
open PowerShell in the extracted directory and run the verified script.

```powershell
Unblock-File -LiteralPath .\start-windows.ps1
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File .\start-windows.ps1
```

First startup asks for your LLM connection mode, endpoint and model ID. Choose
`llamacpp` for the supported dedicated router on a shared GPU. For an LLM on a
separate GPU or server, use ordinary inference and save the separate GPU setting
in the web UI. The launcher keeps the local
configuration and Compose project for subsequent starts. It opens the web UI
with the OCR controller idle. Sign in and enable **Dedicated OCR** when you are
ready to download and verify the model files. Use the launcher again to start
the same installation. Keep `runtime.env`, `windows-project.state`, and the
Docker project volumes together when upgrading. The command sets script
policy for this PowerShell process. [Enable Dedicated OCR](#enable-dedicated-ocr)

## Start a verified Linux bundle

Extract the verified combined bundle for CUDA or ROCm on a Linux GPU host.
Use the NVIDIA driver and Container Toolkit for CUDA. For ROCm, the selected
Linux device and render node must be available to Docker. Start from the
extracted bundle directory.

```bash
./start-linux.sh
```

The starter verifies the bundle, collects your private LLM endpoint and model
on first use, and starts the app with an idle OCR controller. Use it again to
reuse the same project. Keep `runtime.env`, `.vision-bill-install`, and the
Docker project volumes together when upgrading.
[Enable Dedicated OCR](#enable-dedicated-ocr)

## Configure and start

Use Docker Compose 2.30 or later for literal environment file support. Verify the
release/source/image identities and supplied checksums before starting. Keep
`COMPOSE_PROJECT_NAME` in the shell or Compose's `.env`; keep the app settings in
a separate private `runtime.env`.

```bash
sha256sum -c SHA256SUMS
export COMPOSE_PROJECT_NAME=vision-bill
install -m 0600 runtime.env.example runtime.env
```

The copy command is for fresh setup. Keep the real runtime file on an existing
deployment. For a fresh app, populate `LLM__PROVIDER` (`ollama` or `openai`),
`LLM__HOST` and `LLM__MODEL_NAME` explicitly. An OpenAI-compatible example follows.
Use literal values without shell quoting, `export` or interpolation in `runtime.env`.

```dotenv
LLM__PROVIDER=openai
LLM__HOST=http://host.docker.internal:8083/v1
LLM__MODEL_NAME=your-installed-model-id
LLM__API_KEY=none
```

Use the endpoint's actual model ID and reachable URL. Supply the real key for an
authenticated endpoint. Fresh credentialless OpenAI installs can omit the key or
use `none`. Ollama ignores it. On restored installations, a private empty runtime
file takes its values from the saved configuration. Admin Settings can adopt
changed safe formatter fields from environment defaults through an explicit save.
Provider and endpoint changes require an app restart; model and temperature
changes apply live. Credentials stay in operator configuration.

For external OCR, a nonblank `OCR__ENDPOINT` selects its endpoint and an explicit
blank value starts with OCR Off. In a prepared combined profile the browser
toggle can subsequently opt in, including installations with an older explicitly
blank endpoint. A nonblank external endpoint remains operator-owned and cannot
be changed by the Dedicated OCR toggle. For a new Dedicated OCR installation,
leave `OCR__ENDPOINT` unset and do not enable the legacy OCR flags. A saved
external route must be deliberately
reconciled before switching. Ordinary OCR formatting, repair and classification
use text and evidence. Direct vision needs an image-capable selected model.

App port defaults to `127.0.0.1:8080`, with `APP_BIND_ADDRESS` and `APP_PORT`
available for Compose interpolation. Independent OCR defaults to
`127.0.0.1:8082`, with `OCR_BIND_ADDRESS` and `OCR_PORT`. Set
`OCR_CUDA_DEVICE_ID` for NVIDIA or `OCR_RENDER_NODE` for Radeon as needed.
Generated ROCm bundles run the native service as container root. The source
launcher derives device group IDs from the host. Linux AMD
target selection starts with RX 6000/RDNA2 candidates. Qualify the exact device
before relying on it. [ROCm source setup](../paddleocr-service/README.md#amd-rocm-on-native-linux)

```bash
sh ./check-compose.sh
docker compose up -d
```

For a standalone `ocr-cuda` or `ocr-rocm` installation, complete the native
maintenance step below before `up`.

### Enable Dedicated OCR

In a combined install, sign in as the administrator and open **Settings →
Dedicated OCR**. Leave **LLM and OCR share a GPU** selected if both use the same
physical GPU. Verify that your LLM endpoint is dedicated to Vision Bill, has
eager loading disabled, and has unloaded its models. With llama.cpp use a
dedicated private router. If the LLM runs on another GPU or an external
server, clear this setting and save the GPU configuration. Its models can stay
loaded. [llama.cpp router setup](../OPENAI_COMPATIBLE.md#configure-a-dedicated-llamacpp-router)

Confirm the shared-GPU conditions if applicable and turn **Dedicated OCR** On.
The controller verifies the image, device, packages and four pinned models. It
downloads missing model files and starts the service parked. No sample images
run during installation or worker startup. The first real receipt loads the
models and runs OCR. Images stay with local OCR, and the formatter receives
text and layout evidence. On separate devices Vision Bill leaves the LLM loaded.
During setup, inference waits. After a clean setup failure, switch **Dedicated
OCR** on again to retry, or select **Resume direct vision**. An unsettled
handoff requires recovery before switching routes. Turning OCR off routes new
work to direct vision and keeps the service parked with its model files installed. Turning it On again reuses
the verified generation instead of repeating installation. Frozen pending work
may still require an explicit current-policy retry.

For a configuration-driven installation, set both values in `runtime.env` once
the same conditions are met.

```dotenv
OCR_ACTIVATION_ON_STARTUP=true
OCR_ACTIVATION_CONFIRM_HANDOFF=true
```

`OCR_ACTIVATION_MAINTENANCE_INTENT` describes GPU ownership during setup. To select
separate GPUs for normal receipt processing, save **LLM and OCR share a GPU**
as off in Settings. Startup consent records one durable request. Fixing a failed
generation requires an explicit retry.

### Native cache check and maintenance

In a standalone OCR bundle, `ocr-init` checks the complete qualified cache.
On a cold or changed installation, it reports that explicit maintenance is
required. It cannot start qualification or load models. The profiled
`ocr-maintenance` service verifies the release and runs full qualification
before publishing complete selected state.

Stop participating app, OCR, benchmark and standalone owners. Reconcile
outstanding work and ensure the external LLM is quiesced or OCR has its own GPU.
Use one truthful invocation-local maintenance intent. The commands below use
the CUDA example and read its config digest from `release.json`. For ROCm use
`ocr-rocm` instead. `jq` is used only to read that metadata field.

```bash
image_id=$(jq -er '.images["ocr-cuda"].config_digest' release.json)
docker compose --profile ocr-maintenance run --rm --no-deps ocr-maintenance \
  maintain --bundle /release/bundle.json --state-dir /var/lib/ocr \
  --image-config /release/image-config.json --image-id "$image_id" \
  --maintenance-intent external-llm-quiesced
```

With physically separate hardware use `--maintenance-intent separate-ocr-device`.
Use `--force` for a deliberate rerun of physical qualification. The initializer
retains its crash and ownership fences.

For a combined bundle, sign in and use the Dedicated OCR controls described
above. If an installed service is unavailable, check `docker compose logs ocr`
and restart its original project with its volumes intact. If a prior handoff
is unresolved, follow the [recovery procedure](#stop-cut-over-and-recover)
before retrying. A combined installation obtains its serving metadata through
the activation controller. The standalone `maintain` command produces a
different installation record for the physical gate.

Independent OCR bundles default to `OCR_SERVICE_MODE=standalone`. Direct clients
use that service without an app or an LLM. Before pairing it with Vision Bill,
set `OCR_SERVICE_MODE=managed` and arrange the original shared coordinator
storage and dedicated endpoint. Adopt the endpoint and selected model IDs
deliberately before managed work. Dedicated OCR records adoption during setup.
See [native API and client](../paddleocr-service/README.md#independent-service-and-client)
and [endpoint adoption](../OPENAI_COMPATIBLE.md#adopt-the-managed-pair).

### Credentials and persistence

`config-init` creates private configuration and initial admin credentials in
`app-config`, and the PostgreSQL password in `pg-secret`. The directories use
0700 and private configuration and secret files use 0600. PostgreSQL mounts its
data and a read-only password volume. Restored compatible state reuses the
original credentials instead of generating replacements. After startup, read the
initial admin credentials only in your private terminal.

```bash
docker compose exec app cat /app/config/initial-admin.yaml
```

Sign in with that administrator. Self-registration, when enabled, creates ordinary
users. Keep the initial-admin file, configuration and auth secret/pepper private
and backed up. The app migrates the database to schema 0021. See
[upgrade and grants](../UPGRADE_AND_VERIFY.md).

### Stop, cut over and recover

Take a consistent backup of the database, uploads, working views, credentials,
config and native state. Stop the old owners before switching deployment or
running migration. Keep the original Compose project name and volume mounts even
if the new bundle is extracted to a renamed directory. The app and database
volume identities are separate from the retired bundled Ollama model volume.
Leave that old model volume in place. Move to a deliberately configured external
endpoint if the saved host named the removed Ollama service. Startup retains the
saved endpoint and queued jobs can wait until it is reachable or updated.
Handle unfinished routing changes through
[explicit current-policy retry](../MULTI_IMAGE_RECEIPTS.md#attempts-and-explicit-recovery).

An earlier combined bundle may have generated `OCR__ENDPOINT=http://ocr:8082`
in `runtime.env` and saved that same route under `ocr` in `config.yaml`. Before
upgrading that bundle, reconcile its original handoff journal and binding using
the old deployment. Stop all participants, remove the generated endpoint from
`runtime.env`, and remove the corresponding saved `ocr.endpoint` or set the
saved `ocr.enabled` to false for a legacy `ocr.service_url` entry. Keep the old
configuration backup and state volumes. Start the new combined bundle and
enable Dedicated OCR to create the new binding. A genuinely external OCR
endpoint uses the app-only variant and its own original coordinator.

To switch an already activated project to external OCR, reconcile the current
binding and stop its app, OCR controller and standalone clients. Start the
app-only bundle with the same project and app data volumes, and configure the
new external endpoint and its matching coordinator. Preserve the old native
state and its locks for recovery.

Native initializer and OCR roles use container PID1 supervision. Keep their
installation and role lockfiles intact. A retained active marker blocks work
regardless of age. For hard-crash recovery, confirm that every participant using
the original coordination volume has stopped and that its processes are gone,
including retired managed Ollama containers if applicable. Only then archive
the relevant `installation.active-*.json` marker. Reconcile any separately
retained app journal with the old endpoint before changing identity. Preserve
persistent volumes during Compose teardown.
See [native ownership and recovery](../paddleocr-service/README.md#internal-model-ownership).

When Dedicated OCR reports an unsettled direct LLM request, confirm settlement
at the original LLM endpoint. Stop the app, OCR service and standalone clients
that share the project before archiving the retained markers. On separate GPUs,
the LLM can remain loaded. For a return to a shared GPU, prepare the dedicated
LLM endpoint for the shared-mode idle and ownership checks before activation.

```bash
docker compose run --rm --no-deps --entrypoint sh app -ec '
  dir=/activation/request/direct-calls
  if [ -d "$dir" ]; then
    archive=$(mktemp -d /activation/request/reconciled.XXXXXX)
    mv "$dir" "$archive/calls"
  fi'
docker compose up -d
```

For source installs, use `vision_bill` in place of `app` and keep the same
Compose files and profile as the original deployment. Include the generated
`server_data/ocr-source/<accelerator>/compose.local.json` after the accelerator
overlay when using the prepared source deployment.

## Maintainers: real artifact inputs and workflows

The distribution generator is [scripts/distribution.mjs](../scripts/distribution.mjs),
with [artifacts.schema.json](artifacts.schema.json). CPU-only assembly is:

```bash
node scripts/distribution.mjs --manifest /private/artifacts.json \
  --variant app --output /private/new-bundle
```

`--variant` accepts the five choices above. Output must not already exist. This
validates exact asset bytes/evidence and assembles a bundle; it does not build,
pull, qualify or publish. Manifest schema version 2 records the version, exact source repository and
commit, platform, image references, separate config digests, asset hashes and
sizes, and app-check evidence. Native variants add their release
bundle, raw image config, full successful qualification, seal and CUDA attestation
where applicable. The [native bundle schema](../paddleocr-service/release-bundle.schema.json)
defines exact four-model archive trees, five real fixture cases, licenses, provenance,
source/profile identities and actual target proof. Use actual artifact bytes and
qualification evidence. Schema-v1 manifests and the retired `combined-*` variant
names are rejected. Version 2 ships no LLM server or LLM weights.

### Prepare approved native inputs

The source launcher uses the project's pinned Paddle snapshot metadata and five
synthetic fixture images by default. A release maintainer can use those inputs
or stage approved alternatives. They exercise the native workflow on the selected
device. Representative accuracy measurements use a separate evaluation dataset.
[Project fixtures](../paddleocr-service/first-run-fixtures/README.md)

To refresh the complete metadata from the four immutable official repositories,
use the following command on a networked maintainer machine. The revision-specific
sources allow the controller to reconstruct and verify the original archives.
Use the generated `models.json` with the shipped fixture manifest in the next
preparation step.

```bash
PYTHONPATH=paddleocr-service uv run --no-project --with pydantic==2.13.4 --with packaging==26.2 \
  python paddleocr-service/build/prepare_paddle_models.py \
  --official-snapshots --output /private/official-paddle-inputs
```

Run the preparation commands from the repository root with Python 3.12+ and
`uv`. Use a private directory outside Git and Docker build contexts. Create
the parent directories first and choose fresh output directories. Allow
several gigabytes for the downloaded files and generated model archives.

Prepare deterministic tar archives from the pinned original Paddle files for
`PP-DocLayoutV3`, `PP-LCNet_x1_0_doc_ori`, `UVDoc` and `PaddleOCR-VL-1.6-0.9B`.
Provide the approved HTTPS base URL where you will serve those exact tar
files. Replace the example URL with one reachable by the OCR container at
activation. The command writes four `.tar` files and `models.json` with their
verified identities, licenses and upstream provenance. Host the archives at
the recorded URLs before attempting activation or release qualification.

```bash
PYTHONPATH=paddleocr-service uv run --no-project --with pydantic==2.13.4 --with packaging==26.2 \
  python paddleocr-service/build/prepare_paddle_models.py \
  --archive-base-url https://artifacts.example/ocr/models --output /private/ocr-models
```

Use `paddleocr-service/first-run-fixtures/fixtures.json` for the project samples.
For custom inputs, create `/private/approved-fixtures.json` with five distinct, approved PNGs
and a nonempty UTF-8 license file. Each asset path is relative to this JSON
file. Use the actual license and source revision, and verify redistribution
rights for all five images before staging. The table and formula images must
contain those structures. An example manifest follows.

```json
{
  "license": {"spdx": "CC-BY-4.0", "text_file": "LICENSE.txt", "redistribution_allowed": true},
  "provenance": {"url": "https://artifacts.example/ocr/fixtures", "revision": "approved-fixtures-v1"},
  "cases": {
    "upright": "upright.png",
    "rotated": "rotated.png",
    "warped": "warped.png",
    "table": "table.png",
    "formula": "formula.png"
  }
}
```

```bash
uv run --no-project --with pydantic==2.13.4 --with packaging==26.2 --with Pillow==12.3.0 \
  python paddleocr-service/build/prepare_ocr_inputs.py cuda \
  --models /private/ocr-models/models.json --fixtures /private/approved-fixtures.json \
  --output /private/approved-cuda-inputs
```

Use `rocm` and a fresh output directory for AMD. The output is an unbound
`bundle.json` and its local `fixtures.tar`. For a source install, pass that
directory directly to `scripts/ocr_source_up.py`. For the release workflow,
stage both files together under `cuda/` or `rocm/` beside the reviewed
`release-inputs.json` at the exact source/version/variant path below. The
producer includes the local fixture tar by its declared hash and size. The
combined variants read model archives from the approved HTTPS URLs. The
source command and release builder each bind the manifest to their built image.

### Run the distribution workflows

1. [Prepare approved distribution inputs](../.github/workflows/distribution-inputs.yml)
   runs `scripts/distribution-inputs.mjs --stage` on the runner labeled
   `vision-bill-release-inputs`, in environment `distribution-inputs`. A maintainer
   stages a reviewed schema-v2 `release-inputs.json` and exactly the selected native
   assets, approved OCR model inputs and licenses under the fixed trusted root
   `/var/lib/vision-bill/distribution-inputs/<source SHA>/<version>/<variant>/`.
   It rejects symlinks/untrusted or writable ownership paths and missing approved files;
   it never takes an arbitrary host directory, downloads assets or publishes. Its
   `distribution-inputs` artifact carries a bound source/version/variant/actor/run
   receipt and the exact asset inventory.
2. [Verified prebuilt distribution](../.github/workflows/distribution-release.yml)
   consumes `inputs_run_id`. Guarding verifies successful same-fork producer workflow
   identity, current producer admin/maintain permission, source SHA, version, variant,
   actor/run receipt and asset hashes. `--verify DIR` checks the downloaded inventory.
   The consumer rejects extra/missing staged files. The selected exact source runs
   CI before build/qualification and publication.
3. The release driver `scripts/distribution-release.mjs` executes `--phase build
   --inputs DIR --output DIR`, then `--phase qualify --output DIR`, then only after
   success `--phase publish --output DIR`. These are workflow-bound phases requiring
   dispatch identity/environment, not standalone user installation commands.
   Build creates exact-source OCI candidates without publication; qualify runs the
   native initializer/full physical gate on the actual selected GPU. App-only choice
   validates the CPU bundle and has no native qualification claim.
4. Publish revalidates transported OCI/config/evidence identities, pushes immutable
   images, verifies **anonymous pullability**, then publishes release assets last.
   A failure before the last step must not create a successful release.

Both workflows require explicit dispatch on the configured fork:
`DISTRIBUTION_FORK_REPOSITORY` must equal the repository, `source_sha` must equal the
actual dispatch `github.sha`, and version/variant must match the staged plan. Only
admin/maintain actors qualify. These workflows must first be adopted on the fork's
**default branch** for GitHub to expose manual dispatch, with environments, package
permissions and operator runners configured. This source change does not adopt them,
merge/update any main branch, dispatch a run or publish a release.

Native runners need actual CUDA or selected Radeon hardware, Docker/buildx, skopeo,
uv and adequate build/archive capacity. Labels are `vision-bill-cuda` and
`vision-bill-rocm`. Native release qualification requires explicit
`DISTRIBUTION_MAINTENANCE_INTENT` for the runner's actual LLM quiescence or separate
OCR device. ROCm selected targets must agree with compile inputs, kernel
inventory and the actual qualified AMD device. For CUDA, `release-inputs.json`
defaults to `cu129` for new candidates. Set `"cuda_channel": "cu126"` to select
CUDA 12.6 explicitly. On multi-GPU runners set `cuda_device_id` to the physical
GPU index as a string or to the NVIDIA GPU UUID. It defaults to `"0"`. Only
that GPU is exposed to the release qualification process, at container index
zero. The producer records the selected channel in the candidate
plan and manifest, and resolves and attests the official GPU wheel for that
channel. Existing candidate plans and manifests without a channel describe
CUDA 12.6.

OCI registry **manifest** digest identifies transport/pull bytes; local Docker image
`.Id` identifies the config blob. Neither substitutes for the other. The producer
records both, verifies raw config hashes, source/revision labels, model digests,
package hashes and actual device results. Curated licensed real models, redistributable
fixtures, immutable base images and physical runners remain external prerequisites;
the producer code is shipped, those proofs are not. Software stubs/PID1 tests never
certify native model accuracy, speed, VRAM usage or universal AMD support.

The older app-only image release is documented in [RELEASES.md](../RELEASES.md);
its success does not imply any native bundle is qualified or available.
