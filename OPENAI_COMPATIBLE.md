# OpenAI-compatible LLM backends

Vision Bill can send text and image requests to an OpenAI-compatible API, including
llama.cpp. Keep the endpoint local for local receipt processing. Every app
installation uses an operator-selected endpoint and model. OpenAI compatibility
covers inference. Model loading and unloading need a separate backend contract.

## Connect an inference endpoint

For a prepared bundle, set the following in private `runtime.env` using literal
values and mode 0600. Use private `.env` for source Compose. Replace the example
endpoint and model with your own. The provider key can be omitted on fresh
credentialless prepared installs or set to `none`. Supply the real key when the
server requires authentication.

```dotenv
LLM__PROVIDER=openai
LLM__HOST=http://host.docker.internal:8083/v1
LLM__MODEL_NAME=your-installed-model-id
LLM__API_KEY=none
```

The URL must be reachable from the app container. Direct vision requires a model
that supports image input. For OCR on a separate GPU, clear **LLM and OCR share a
GPU** and save under **Settings → Dedicated OCR** before enabling OCR. An
OpenAI-compatible endpoint can use model lifecycle `none` and keep its models
loaded. Use the router setup below for a shared GPU.

### Authenticate and discover models in Settings

Under **Settings → Receipt formatter**, choose the provider and enter its base
URL. For native Ollama, use the server root (for example,
`http://host.docker.internal:11434`). For an OpenAI-compatible service, use
its API base (for example, `http://host.docker.internal:8083/v1`). A reverse
proxy path can be part of either base URL. Remove query parameters and URL
fragments before saving the host. Enter credentials in the authentication field.

Choose **None** for the standard local Ollama API or a credentialless local
OpenAI-compatible server. Choose **Bearer API key** for a proxy that requires
a Bearer token, direct access to Ollama Cloud, or OpenAI's hosted API. Enter
the key or access token for the chosen service. A local Ollama server can also
sign in to Ollama Cloud itself. In that setup the local server handles cloud
authentication. [Ollama authentication](https://docs.ollama.com/api/authentication) ·
[OpenAI authentication](https://developers.openai.com/api/reference/overview#authentication)

Save the settings, then click **Test saved connection**. Vision Bill requests
only the model list, using `GET /api/tags` for Ollama or `GET /models` under
the configured OpenAI-compatible API base. The test sends no receipt or prompt
and runs no inference. Select a returned model ID, or enter an exact ID manually
if discovery is unavailable or the list omits your model. Save the model choice.
A listed ID confirms that the endpoint advertises the model. If the model server
includes a connection credential in a published ID or model detail, Vision Bill
rejects that catalog response. Check the model server's catalog configuration
if it reports an invalid model list. For direct vision,
check the model's documented image-input support separately.
[Ollama list models](https://docs.ollama.com/api/tags) ·
[OpenAI list models](https://developers.openai.com/api/reference/resources/models/methods/list)

A Bearer key saved in Settings is bound to the selected provider and endpoint
origin (scheme, hostname and port). Settings only shows whether a credential is
configured. Existing deployments can keep their original credential source.
For inherited OpenAI authentication, the connection test uses the deployment's
organization and project headers, including custom header overrides, just as
the active model client does. Selecting **None** or **Bearer API key** uses the
explicit settings for that connection and omits those inherited headers.
`LLM__API_KEY` applies to OpenAI-compatible endpoints and a deployment-supplied
`OLLAMA_API_KEY` applies to Ollama. Choosing **None** explicitly disables
authentication for that connection. Restart the app service when Settings
reports a pending configuration. [Installation guide](distribution/README.md)

## Configure a dedicated llama.cpp router

Use a llama.cpp build exposing the router management API described in its
[server guide](https://github.com/ggml-org/llama.cpp/blob/1ef8c5d68495d69b065d28e780264ee01595b6b4/tools/server/README.md#using-multiple-models).
A stock single-model server or an arbitrary OpenAI-compatible server does not
provide this contract merely by accepting chat completions.

Deploy a dedicated Vision Bill router. Start it in router mode without selecting
one model on the server command line; configure local GGUF files through its model
directory or presets. Keep autoload enabled and set every preset's
`load-on-startup=false`. Model discovery and readiness checks must not eagerly
load weights. `--models-max=1` alone does not coordinate the separate OCR service.

Replace the active LLM values in your private configuration. The following
example shows a separately managed OCR endpoint.

```dotenv
LLM__PROVIDER=openai
LLM__HOST=http://host.docker.internal:8083/v1
LLM__API_KEY=none
LLM__MODEL_NAME=receipt-formatter
LLM__MODEL_LIFECYCLE=llamacpp
OCR__ENDPOINT=http://your-private-managed-ocr:8082
OCR__MANAGE_MODEL_LIFECYCLE=true
```

The LLM URL must reach your deliberately deployed router. Set the OCR endpoint to
your actual managed service; an independent OCR-only bundle must use
`OCR_SERVICE_MODE=managed` when paired with VisionBill. `app-ocr-cuda` and `app-ocr-rocm` set it to managed. See [prepared installation](distribution/README.md). The example uses the
Docker host gateway; a host process can use `http://127.0.0.1:8083/v1`. A host router must
listen on an interface reachable from the app container and remain on a trusted
private network. `none` is a placeholder for a router without authentication;
otherwise use its configured API key in the ignored private configuration.
The same Bearer key authenticates inference and lifecycle requests.

For a combined **Dedicated OCR** install, leave `OCR__ENDPOINT` unset and keep
the LLM settings above. Enable Dedicated OCR in admin Settings after the router
is idle. The action checks residency and records the selected handoff binding
on a shared GPU before native setup. For a local source installation this
activation also starts the first native image build after the app and database
are ready. Use the manual adoption below for separately managed OCR.
[Source activation](paddleocr-service/README.md#start-vision-bill-first-and-build-ocr-from-settings)

Replace `receipt-formatter` with its exact canonical `id` from the router's
`GET /models` response. Managed release does not resolve aliases. OCR formatting
can use a text-only model and remains text/evidence-only on all ordinary failure,
repair and classification paths. Direct vision without OCR, or an explicit admin
direct-vision comparison, needs an image-capable model. For older servers with
absent capability
metadata, `LLM__VISION_MODEL_IDS=["receipt-vision"]` can declare known image support;
it never overrides explicit contrary metadata. Output/context limits still apply;
this setting does not give a text model vision support.

The normal formatter output cap is 32768 tokens. Native PaddleOCR-VL 1.6 uses
up to 4096 output tokens and 2,000,000 input pixels per region. Document
orientation is enabled and geometric unwarping is disabled. The lifecycle API is independent of the GPU backend selected when
deploying llama.cpp.

## Adopt the managed pair

Adoption is required for the shared-GPU configuration described above. An
LLM configured on a separate server uses ordinary inference without adoption.

Drain participating work before adoption. Start the dedicated router with all
models unloaded, all startup loading disabled and access limited to coordinated
Vision Bill model-loading requests. Start the configured managed OCR and app
using the original shared coordinator path. From a prepared app bundle, use:

```bash
model_id=your-installed-model-id
docker compose exec app python -m vision_bill.distribution.handoff adopt \
  --model-id "$model_id" --dedicated-coordinated-requests \
  --external-eager-loading-disabled
docker compose exec app python -m vision_bill.distribution.handoff status \
  --model-id "$model_id"
```

Use the `vision_bill` service name for source Compose. Add one `--model-id` for
each deliberately authorized ID when setting up the binding. Keep the exact
configured lock path, or pass that same path with `--handoff-lock`. Adoption
checks for clean owned state and supported idle residency evidence. A busy lock,
unknown residency, unaccounted models, damaged state or a changed binding holds
work before inference. The attestations describe the endpoint that the operator
actually set up.

## Sequential GPU handoff

On a shared GPU, one internal owner coordinates the complete job:

1. Park any abandoned OCR worker and release the job's previously recorded LLM IDs.
2. Load native OCR on demand and process every receipt photo.
3. Park OCR and confirm that its native worker has exited.
4. Send OpenAI-compatible requests; router autoload loads the requested LLM.
   Formatting, repairs and classification share the same owner.
5. Request that LLM's unload and wait for its canonical router status to become
   `unloaded` before another owner can load OCR.

On separate GPUs, the app still loads and parks the native OCR worker, confirms
its exit, then sends text and evidence to the LLM. LLM requests use ordinary
OpenAI-compatible inference without model-management calls.

The shared-GPU control protocol is `GET /models` and
`POST /models/unload {"model":"canonical-id"}`. A successful POST only schedules
termination. Vision Bill polls until the exact model reports
`status.value="unloaded"`; sleeping, missing IDs, unknown/duplicate states or an
empty model list do not confirm release. Only the journal's exact owned model IDs
are released. Other discovered models are never swept.

Release has at most 30 seconds within the existing absolute attempt deadline.
An unconfirmed/failed/cancelled request keeps durable ownership unresolved and
blocks subsequent OCR. See [recovery](paddleocr-service/README.md#internal-model-ownership).

Every app worker and supported standalone client must use matching topology and
OCR resource identity plus the same underlying Linux `OCR__COORDINATION_PATH`
file and adjacent journals. On a shared GPU, they must also agree on the LLM
provider, lifecycle mode and endpoint. Route model-loading requests for a shared
GPU through those owners. Other GPU applications can continue running.

Use the final inference URL without redirects. Lifecycle and managed inference
requests refuse redirects and ignore environment proxy variables. A host ending
in `/v1` uses sibling management routes; for example
`http://host/llama/v1` maps to `http://host/llama/models/unload`. Userinfo,
query/fragment components and ambiguous encoded/dot paths are rejected.

## Other providers, changes and verification

`LLM__MODEL_LIFECYCLE=none` is the default for OpenAI-compatible providers without
an implemented unload API. It supports direct vision and separately hosted LLMs.
On a shared GPU, native OCR waits before claim when unloading support or stable
identities are missing. No attempt is spent in that state. Selecting
`llamacpp` alone does not enable automatic unloading while production managed
OCR is off. An explicitly unmanaged
CPU/replay test policy makes no shared-GPU handoff claim. Do not use lifecycle
opt-out for a native shared pair; a cold unmanaged native pipeline waits before
claim and cannot activate the parked worker without ownership.
Ollama uses its own lifecycle and ignores a retained OpenAI-only lifecycle option.

Drain and stop all owners before changing provider, endpoint or lifecycle mode.
Reconcile outstanding requests and owned models on the **old endpoint** first;
the existing journal binds its endpoint identity. Only after confirmed release,
with all owners stopped, archive the reconciled handoff journal and binding
according to the recovery procedure. Keep the lockfile inode intact. Restart
with the intended configuration and explicitly adopt the new pair.

Prepare the model on your endpoint before connecting Vision Bill. Benchmarks are
explicit requests, and persisted pending work can resume under readiness and
policy guards. Native qualification runs through explicit maintenance. Software
verification uses CPU doubles. Real endpoint and accelerator qualification
require separate evidence.
The context-budget diagnostic remains Ollama-only; the app and standalone
benchmark use the configured provider.
