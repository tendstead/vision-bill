#!/usr/bin/env node
// CPU-only release assembly. No pulls, builds, inference or publication.
import { createHash } from "node:crypto";
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  lstatSync,
  realpathSync,
  renameSync,
  rmSync,
  mkdtempSync,
  existsSync,
  openSync,
  readSync,
  closeSync,
  copyFileSync,
} from "node:fs";
import { dirname, resolve, join, relative, sep } from "node:path";
import { pathToFileURL } from "node:url";

const schema = JSON.parse(
  readFileSync(
    new URL("../distribution/artifacts.schema.json", import.meta.url),
  ),
);
const variants = [
  "app",
  "ocr-cuda",
  "ocr-rocm",
  "app-ocr-cuda",
  "app-ocr-rocm",
];
export const cudaChannels = ["cu126", "cu129"];
export const defaultCudaChannel = "cu129";
export const legacyCudaChannel = "cu126";
export const cudaLockDigestLabel = "io.vision-bill.cuda.lock-sha256";
const cudaWheelUrl =
  /^https:\/\/(?:www\.paddlepaddle\.org\.cn|paddle-whl\.(?:bj|cdn)\.bcebos\.com)\/(packages\/)?stable\/(cu126|cu129)\/(paddlepaddle-gpu\/)?paddlepaddle_gpu-3\.3\.1-cp312-cp312-(?:linux_x86_64|manylinux[\w.]*x86_64)\.whl$/;
const nativeAssetPaths = ({ cuda_channel, ...paths }) => Object.values(paths);
const hash = (value) => createHash("sha256").update(value).digest("hex");
const canonical = (value) =>
  Array.isArray(value)
    ? value.map(canonical)
    : value && typeof value === "object"
      ? Object.fromEntries(
          Object.keys(value)
            .sort()
            .map((key) => [key, canonical(value[key])]),
        )
      : value;
const json = (value) => JSON.stringify(canonical(value), null, 2) + "\n";
const fail = (message) => {
  throw Error(message);
};
const check = (condition, message) => {
  if (!condition) fail(message);
};
function schemaCheck(value, rule, path = "manifest", root = schema) {
  if (rule.$ref)
    return schemaCheck(
      value,
      rule.$ref
        .split("/")
        .slice(1)
        .reduce((node, key) => node[key], root),
      path,
      root,
    );
  if ("const" in rule) check(value === rule.const, `${path}: invalid constant`);
  if (rule.enum) check(rule.enum.includes(value), `${path}: invalid choice`);
  if (rule.type === "object") {
    check(
      value !== null && typeof value === "object" && !Array.isArray(value),
      `${path}: expected object`,
    );
    check(
      Object.keys(value).length >= (rule.minProperties ?? 0),
      `${path}: incomplete object`,
    );
    check(
      Object.keys(value).length <= (rule.maxProperties ?? Infinity),
      `${path}: oversized object`,
    );
    for (const key of rule.required ?? [])
      check(Object.hasOwn(value, key), `${path}.${key}: required`);
    for (const [key, child] of Object.entries(value)) {
      const childRule = rule.properties?.[key] ?? rule.additionalProperties;
      check(childRule !== false, `${path}.${key}: unknown field`);
      if (typeof childRule === "object")
        schemaCheck(child, childRule, `${path}.${key}`, root);
    }
  } else if (rule.type === "string") {
    check(typeof value === "string", `${path}: expected string`);
    check(value.length >= (rule.minLength ?? 0), `${path}: empty string`);
    if (rule.pattern)
      check(new RegExp(rule.pattern).test(value), `${path}: invalid identity`);
  } else if (rule.type === "integer")
    check(
      Number.isSafeInteger(value) &&
        value >= (rule.minimum ?? 0) &&
        value > (rule.exclusiveMinimum ?? -Infinity) &&
        value <= (rule.maximum ?? Infinity),
      `${path}: invalid size`,
    );
  else if (rule.type === "boolean")
    check(typeof value === "boolean", `${path}: expected boolean`);
  else if (rule.type === "array") {
    check(
      Array.isArray(value) &&
        value.length >= (rule.minItems ?? 0) &&
        value.length <= (rule.maxItems ?? Infinity),
      `${path}: invalid array`,
    );
    for (const [index, child] of value.entries())
      schemaCheck(child, rule.items, `${path}[${index}]`, root);
  }
}
function safePath(path) {
  check(
    typeof path === "string" &&
      /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(path) &&
      path.split("/").every((part) => part && part !== "." && part !== ".."),
    "Unsafe asset path",
  );
  return path;
}
function rejectPlaceholders(value) {
  if (typeof value === "string") {
    check(
      !/(?:\b(?:example|placeholder|replace_me|todo|tbd|unexecuted|unresolved)\b|[<>]|\$\{)/i.test(
        value,
      ),
      "Placeholder release identity",
    );
    check(
      !/(?:^|@sha256:|^sha256:)([a-f0-9])\1{39,63}$/.test(value),
      "Placeholder digest",
    );
    if (value.startsWith("https://")) isHttpsAssetSource(value);
  } else if (value && typeof value === "object")
    for (const child of Object.values(value)) rejectPlaceholders(child);
}
export function validateArtifacts(manifest) {
  schemaCheck(manifest, schema);
  rejectPlaceholders(manifest);
  for (const path of Object.keys(manifest.assets)) safePath(path);
  const requireAsset = (path) => {
    safePath(path);
    check(Object.hasOwn(manifest.assets, path), `Missing asset: ${path}`);
  };
  requireAsset(manifest.verification.app_checks);
  for (const [accelerator, native] of Object.entries(manifest.native ?? {})) {
    check(
      manifest.images[`ocr-${accelerator}`],
      "Missing native image identity",
    );
    for (const path of nativeAssetPaths(native)) requireAsset(path);
    if (accelerator === "cuda")
      check(native.cuda_lock, "Missing resolved CUDA lock evidence");
    else
      check(
        !Object.hasOwn(native, "cuda_channel"),
        "CUDA channel requires CUDA image",
      );
  }
  return manifest;
}
export function isHttpsAssetSource(source) {
  if (typeof source !== "string" || !source.startsWith("https://"))
    return false;
  check(
    /^https:\/\/[^/\s\\][^\s\\]*$/.test(source),
    "Absolute HTTPS asset URL required",
  );
  const url = new URL(source);
  check(
    !url.username &&
      !url.password &&
      ![...url.searchParams.keys()].some((key) =>
        /secret|token|signature|password|api.?key/i.test(key),
      ),
    "Credential-bearing asset URL",
  );
  return true;
}
const volume = (source, target, readOnly = false) => ({
  type: "volume",
  source,
  target,
  ...(readOnly ? { read_only: true } : {}),
});
const bind = (source, target) => ({
  type: "bind",
  source: `./release/${source}`,
  target,
  read_only: true,
  bind: { create_host_path: false },
});
const done = (name) => ({
  [name]: { condition: "service_completed_successfully" },
});
const activationHandoffLock = "/coord/gpu-handoff.lock";
const activationMounts = (app) => [
  volume("ocr-activation-request", "/activation/request", !app),
  volume("ocr-activation-status", "/activation/status", app),
  volume("ocr-activation-lock", "/activation/lock"),
];
const device = (accelerator) =>
  accelerator === "cuda"
    ? {
        deploy: {
          resources: {
            reservations: {
              devices: [
                {
                  driver: "nvidia",
                  device_ids: ["${OCR_CUDA_DEVICE_ID:-0}"],
                  capabilities: ["gpu"],
                },
              ],
            },
          },
        },
      }
    : {
        devices: [
          "/dev/kfd:/dev/kfd",
          "${OCR_RENDER_NODE:-/dev/dri/renderD128}:${OCR_RENDER_NODE:-/dev/dri/renderD128}",
        ],
      };

const composeCheck = `#!/bin/sh
set -eu
version=$(docker compose version --short) || exit 1
version=\${version#v}
major=\${version%%.*}
rest=\${version#*.}
minor=\${rest%%.*}
case "$major:$minor" in *[!0-9:]*|:*) echo 'Docker Compose >=2.30 is required for literal runtime.env delivery.' >&2; exit 1;; esac
if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 30 ]; }; then
  echo 'Docker Compose >=2.30 is required for literal runtime.env delivery; upgrade the Compose plugin.' >&2
  exit 1
fi
runtime="$(dirname "$0")/runtime.env"
if [ -e "$runtime" ] && { [ -L "$runtime" ] || [ "$(stat -c %a "$runtime")" != "600" ]; }; then
  echo 'runtime.env must be a regular private file with mode 0600; run chmod 0600 runtime.env.' >&2
  exit 1
fi
exec docker compose -f "$(dirname "$0")/compose.yaml" config --quiet
`;
function runtimeExample(activation) {
  return `# Requires Docker Compose >=2.30 (raw env_file support).
# Copy to runtime.env and chmod 0600 runtime.env before running check-compose.sh.
# Values are literal: do not add quoting, export, or shell interpolation.
# Set COMPOSE_PROJECT_NAME separately in the shell or .env; reuse the old project
# name and volumes for upgrades, even when moving or renaming this bundle folder.
# Fresh config requires all three explicit values below. Restored config needs no seeds.
# LLM__PROVIDER=ollama
# LLM__HOST=http://host.docker.internal:11434
# LLM__MODEL_NAME=your-explicit-model
# Optional API key and temperature.
# LLM__API_KEY=literal-key
# LLM__TEMPERATURE=0
# Optional lifecycle control for a dedicated OpenAI-compatible llama.cpp router.
# LLM__MODEL_LIFECYCLE=llamacpp
# For externally managed OCR, explicitly set OCR__ENDPOINT to the service URL.
# Omit to preserve saved configuration, or set blank to explicitly disable OCR.
${activation ? `# Use Dedicated OCR in admin Settings to set up the selected GPU profile.
# Or explicitly request setup once on startup.
# OCR_ACTIVATION_ON_STARTUP=false
# Set true with an idle, dedicated LLM endpoint and eager loading disabled.
# OCR_ACTIVATION_CONFIRM_HANDOFF=false
# Attest that the external LLM is quiesced, or select separate-ocr-device.
# OCR_ACTIVATION_MAINTENANCE_INTENT=external-llm-quiesced
` : ""}`;
}

export function renderBundle(manifest, variant) {
  validateArtifacts(manifest);
  check(variants.includes(variant), "Unknown distribution variant");
  const appOcr = variant.startsWith("app-ocr-");
  const app = variant === "app" || appOcr;
  const accelerator = variant === "app" ? null : variant.split("-").at(-1);
  const names = [
    ...(app ? ["app", "postgres"] : []),
    ...(accelerator ? [`ocr-${accelerator}`] : []),
  ];
  for (const name of names)
    check(manifest.images[name], `Missing ${name} image`);
  if (accelerator)
    check(manifest.native?.[accelerator], "Missing native qualification");
  const services = {},
    volumes = {};
  const assets = new Set([manifest.verification.app_checks]);
  function service(name, imageName, fields) {
    services[name] = {
      image: manifest.images[imageName].reference,
      platform: manifest.platform,
      ...fields,
    };
  }
  if (app) {
    for (const name of [
      "app-config",
      "pg-secret",
      "db-data",
      "uploads",
      "uploads-tmp",
      "logs",
    ])
      volumes[name] = {};
    service("config-init", "app", {
      restart: "no",
      env_file: [{ path: "./runtime.env", format: "raw" }],
      extra_hosts: ["host.docker.internal:host-gateway"],
      environment: {
        VISION_BILL_CONFIG_PATH: "/app/config/config.yaml",
      },
      command: [
        "init",
        "--config-dir",
        "/app/config",
        "--postgres-password-file",
        "/run/pg-secret/password",
      ],
      volumes: [
        volume("app-config", "/app/config"),
        volume("pg-secret", "/run/pg-secret"),
      ],
    });
    service("db", "postgres", {
      restart: "unless-stopped",
      depends_on: done("config-init"),
      environment: {
        POSTGRES_USER: "vision_bill",
        POSTGRES_DB: "vision_bill",
        POSTGRES_PASSWORD_FILE: "/run/pg-secret/password",
        PGDATA: "/var/lib/postgresql/data/pgdata",
      },
      volumes: [
        volume("db-data", "/var/lib/postgresql"),
        volume("pg-secret", "/run/pg-secret", true),
      ],
      healthcheck: {
        test: ["CMD-SHELL", "pg_isready -U vision_bill -d vision_bill"],
        interval: "5s",
        timeout: "5s",
        retries: 30,
      },
    });
    service("app", "app", {
      restart: "unless-stopped",
      depends_on: {
        ...done("config-init"),
        db: { condition: "service_healthy" },
      },
      environment: {
        VISION_BILL_CONFIG_PATH: "/app/config/config.yaml",
        ...(appOcr
          ? {
              VISION_BILL_OCR_ACTIVATION: "true",
              OCR_ACTIVATION_ENDPOINT: "http://ocr:8082",
              OCR_ACTIVATION_HANDOFF_LOCK_PATH: activationHandoffLock,
            }
          : {}),
      },
      env_file: [{ path: "./runtime.env", format: "raw" }],
      extra_hosts: ["host.docker.internal:host-gateway"],
      ports: ["${APP_BIND_ADDRESS:-127.0.0.1}:${APP_PORT:-8080}:8080"],
      ...(appOcr
        ? {
            healthcheck: {
              test: [
                "CMD",
                "python",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v1/system/ui-config', timeout=3)",
              ],
              interval: "5s",
              timeout: "5s",
              retries: 36,
            },
          }
        : {}),
      volumes: [
        volume("app-config", "/app/config"),
        volume("uploads", "/app/uploads"),
        volume("uploads-tmp", "/app/uploads_tmp"),
        volume("logs", "/app/logs"),
        ...(appOcr
          ? [volume("coordination", "/coord"), ...activationMounts(true)]
          : []),
      ],
    });
  }
  if (accelerator) {
    volumes["ocr-state"] = {};
    volumes.coordination = {};
    const native = manifest.native[accelerator];
    for (const path of nativeAssetPaths(native)) assets.add(path);
    const mounts = [
      volume("ocr-state", "/var/lib/ocr"),
      volume("coordination", "/coord"),
      bind(`${accelerator}`, "/release"),
    ];
    // Each accelerator's bundle owns one directory, including any local model archives.
    for (const path of Object.keys(manifest.assets))
      if (path.startsWith(`${accelerator}/`)) assets.add(path);
    check(
      native.bundle === `${accelerator}/bundle.json` &&
        native.image_config === `${accelerator}/image-config.json`,
      "Native release paths must match initializer mounts",
    );
    const lock = {
      OCR_INSTALLATION_LOCK_PATH: "/coord/installation.lock",
      OCR_STATE_DIR: "/var/lib/ocr",
    };
    if (appOcr) {
      const controlMounts = activationMounts(false);
      for (const mount of controlMounts) volumes[mount.source] = {};
      service("ocr", `ocr-${accelerator}`, {
        init: false,
        restart: "unless-stopped",
        command: ["python", "-m", "ocr_service.activation"],
        environment: {
          ...lock,
          OCR_IMAGE_ID: manifest.images[`ocr-${accelerator}`].config_digest,
          OCR_HANDOFF_LOCK_PATH: activationHandoffLock,
        },
        volumes: [...mounts, ...controlMounts],
        ...device(accelerator),
      });
    } else {
      service("ocr-init", `ocr-${accelerator}`, {
        init: false,
        restart: "no",
        entrypoint: ["python", "-m", "ocr_service.initialize"],
        command: [
          "check",
          "--bundle",
          "/release/bundle.json",
          "--state-dir",
          "/var/lib/ocr",
          "--image-config",
          "/release/image-config.json",
          "--image-id",
          manifest.images[`ocr-${accelerator}`].config_digest,
        ],
        environment: lock,
        volumes: mounts,
        ...device(accelerator),
      });
      service("ocr-maintenance", `ocr-${accelerator}`, {
        ...services["ocr-init"],
        profiles: ["ocr-maintenance"],
        command: ["maintain", ...services["ocr-init"].command.slice(1)],
      });
      service("ocr", `ocr-${accelerator}`, {
        init: false,
        restart: "unless-stopped",
        command: ["python", "-m", "ocr_service.entrypoint"],
        depends_on: done("ocr-init"),
        environment: {
          ...lock,
          OCR_SERVICE_MODE: "${OCR_SERVICE_MODE:-standalone}",
        },
        volumes: mounts,
        ports: ["${OCR_BIND_ADDRESS:-127.0.0.1}:${OCR_PORT:-8082}:8082"],
        ...device(accelerator),
      });
    }
  }
  const installationLabel = {
    "io.vision-bill.installation": "${VISION_BILL_INSTALLATION_ID:-manual}",
  };
  if (appOcr) {
    for (const service of Object.values(services))
      service.labels = { ...service.labels, ...installationLabel };
    for (const volume of Object.values(volumes))
      volume.labels = { ...volume.labels, ...installationLabel };
  }
  const metadata = {
    schema_version: 2,
    version: manifest.version,
    source: manifest.source,
    platform: manifest.platform,
    variant,
    images: Object.fromEntries(
      names.map((name) => [name, manifest.images[name]]),
    ),
    verification: manifest.verification,
    ...(accelerator
      ? { native: { [accelerator]: manifest.native[accelerator] } }
      : {}),
    assets: Object.fromEntries(
      [...assets].sort().map((path) => [path, manifest.assets[path]]),
    ),
  };
  return {
    files: {
      "compose.yaml": json({
        name: "${COMPOSE_PROJECT_NAME:?Set a stable project name and reuse it for upgrades}",
        services,
        volumes,
        ...(appOcr ? { networks: { default: { labels: installationLabel } } } : {}),
      }),
      "runtime.env.example": runtimeExample(appOcr),
      "check-compose.sh": composeCheck,
      "release.json": json(metadata),
      ...(appOcr
        ? {
            "start-linux.sh": readFileSync(
              new URL("../distribution/start-linux.sh", import.meta.url),
              "utf8",
            ),
          }
        : {}),
      ...(variant === "app-ocr-cuda"
        ? {
            "start-windows.ps1": readFileSync(
              new URL("../distribution/start-windows.ps1", import.meta.url),
              "utf8",
            ),
          }
        : {}),
    },
    assets: [...assets].sort(),
  };
}
export function fileIdentity(path) {
  const descriptor = openSync(path, "r"),
    digest = createHash("sha256"),
    buffer = Buffer.allocUnsafe(1024 * 1024);
  let size = 0;
  try {
    let count;
    while ((count = readSync(descriptor, buffer, 0, buffer.length, null)) > 0) {
      digest.update(buffer.subarray(0, count));
      size += count;
    }
  } finally {
    closeSync(descriptor);
  }
  return { sha256: digest.digest("hex"), size_bytes: size };
}
function checkedRead(base, path, record) {
  safePath(path);
  const root = realpathSync(base),
    full = resolve(root, path);
  check(
    !relative(root, realpathSync(full)).startsWith(`..${sep}`),
    "Asset outside manifest directory",
  );
  let cursor = full;
  while (cursor !== root) {
    check(!lstatSync(cursor).isSymbolicLink(), "Symlink release asset");
    cursor = dirname(cursor);
  }
  check(lstatSync(full).isFile(), "Nonregular release asset");
  const actual = fileIdentity(full);
  check(
    actual.size_bytes === record.size_bytes && actual.sha256 === record.sha256,
    `Asset hash identity mismatch: ${path}`,
  );
  return full;
}
export function validateCudaLock(lock, channel) {
  const wheel =
    typeof lock?.wheel_url === "string" && cudaWheelUrl.exec(lock.wheel_url);
  const attestedChannel = Object.hasOwn(lock ?? {}, "cuda_channel")
    ? lock.cuda_channel
    : legacyCudaChannel;
  check(
    cudaChannels.includes(channel) &&
      attestedChannel === channel &&
      lock?.index ===
        `https://www.paddlepaddle.org.cn/packages/stable/${channel}/paddlepaddle-gpu/` &&
      wheel?.[0] === lock.wheel_url &&
      wheel?.[2] === channel &&
      (!wheel[1] || wheel[3]),
    "Official CUDA channel/index/wheel evidence mismatch",
  );
  check(
    lock.metadata_name === "paddlepaddle-gpu" &&
      lock.metadata_version === "3.3.1" &&
      lock.platform === "linux-x86_64-cp312" &&
      Array.isArray(lock.requires_dist) &&
      lock.requires_dist.length > 0 &&
      lock.requires_dist.every(
        (requirement) => typeof requirement === "string",
      ) &&
      ["index_sha256", "wheel_sha256", "metadata_sha256", "lock_sha256"].every(
        (key) =>
          typeof lock[key] === "string" && /^[a-f0-9]{64}$/.test(lock[key]),
      ),
    "Official CUDA wheel/metadata resolution evidence missing",
  );
}
function verifyExecutableCudaLock(text, lock) {
  // uv emits one hashed requirement per logical line. Comments and wrapped
  // hashes are allowed, but requirement files and installer options are not.
  const statements = text
    .replace(/(^|[ \t]+)#[^\r\n]*/gm, "")
    .replace(/\\[ \t]*\r?\n/g, " ")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const paddle = [];
  const expectedHash = "--hash=sha256:" + lock.wheel_sha256;
  for (const statement of statements) {
    const entry =
      /^([a-zA-Z0-9][a-zA-Z0-9._-]*)\s*(?:@\s*(\S+)|==\s*[^\s;]+)((?:[ \t]+--hash=sha256:[a-f0-9]{64})+)$/.exec(
        statement,
      );
    check(entry, "Invalid pinned requirement in executable CUDA lock");
    const [, rawName, url, hashes] = entry;
    const name = rawName.toLowerCase().replace(/[-_.]+/g, "-");
    if (name.startsWith("paddlepaddle"))
      paddle.push({ name, url, hashes: hashes.trim().split(/\s+/) });
  }
  check(
    paddle.length === 1 &&
      paddle[0].name === "paddlepaddle-gpu" &&
      paddle[0].url === lock.wheel_url &&
      paddle[0].hashes.every((value) => value === expectedHash),
    "Executable CUDA lock must contain the one attested Paddle wheel pin",
  );
}
function verifyEvidence(manifest, variant, contents) {
  const read = (path) => {
    check(
      manifest.assets[path].size_bytes <= 32 * 1024 * 1024,
      "Metadata exceeds size bound",
    );
    return JSON.parse(readFileSync(contents[path]));
  };
  const app = read(manifest.verification.app_checks);
  check(
    app.passed === true &&
      app.source_commit === manifest.source.commit &&
      ["backend", "frontend", "e2e-backend", "e2e-frontend"].every((name) =>
        app.checks?.includes(name),
      ),
    "App source checks are incomplete",
  );
  if (variant === "app") return;
  const accelerator = variant.split("-").at(-1),
    native = manifest.native[accelerator],
    image = manifest.images[`ocr-${accelerator}`];
  const bundle = read(native.bundle),
    config = read(native.image_config),
    report = read(native.qualification);
  const nativeSchema = JSON.parse(
    readFileSync(
      new URL(
        "../paddleocr-service/release-bundle.schema.json",
        import.meta.url,
      ),
    ),
  );
  schemaCheck(bundle, nativeSchema, "native bundle", nativeSchema);
  check(
    JSON.stringify(Object.keys(bundle.models).sort()) ===
      JSON.stringify(
        [
          "PP-DocLayoutV3",
          "PP-LCNet_x1_0_doc_ori",
          "PaddleOCR-VL-1.6-0.9B",
          "UVDoc",
        ].sort(),
      ),
    "Exactly four approved model identities required",
  );
  check(
    bundle.expected_packages.paddleocr === "3.7.0" &&
      bundle.expected_packages.paddlex === "3.7.2",
    "Pinned OCR packages required",
  );
  const paddle = Object.keys(bundle.expected_packages).filter((name) =>
    name.toLowerCase().startsWith("paddlepaddle"),
  );
  check(
    paddle.length === 1 &&
      ["paddlepaddle", "paddlepaddle-gpu", "paddlepaddle-dcu"].includes(
        paddle[0],
      ) &&
      bundle.expected_packages[paddle[0]].split("+")[0] === "3.3.1",
    "Pinned native Paddle required",
  );
  check(
    paddle[0] !== "paddlepaddle-dcu" || accelerator === "rocm",
    "The paddlepaddle-dcu distribution requires the ROCm accelerator",
  );
  check(
    bundle.source_files["profile.json"] === bundle.profile_sha256 &&
      bundle.source_files["runtime-lock.json"],
    "Source profile/lock identities missing",
  );
  const allowedAssets = new Set([
    manifest.verification.app_checks,
    ...nativeAssetPaths(native),
  ]);
  const activation = variant.startsWith("app-ocr-");
  for (const [name, archive] of Object.entries({
    ...bundle.models,
    fixtures: bundle.fixtures,
  })) {
    for (const path of Object.keys(archive.files)) {
      safePath(path);
      check(
        !path
          .split("/")
          .slice(0, -1)
          .some(
            (_, i, parts) => archive.files[parts.slice(0, i + 1).join("/")],
          ),
        "Archive path collision",
      );
    }
    check(
      archive.files[safePath(archive.license.text_file)]?.size_bytes > 0,
      "Sealed license text missing",
    );
    check(
      Object.values(archive.files).reduce(
        (sum, row) => sum + row.size_bytes,
        0,
      ) === archive.unpacked_bytes,
      "Complete unpacked inventory missing",
    );
    const tree = Object.fromEntries(
      Object.entries(archive.files)
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([path, row]) => [path, row.sha256]),
    );
    check(
      hash(JSON.stringify(tree)) === archive.tree_sha256,
      "Complete archive tree hash mismatch",
    );
    if (!isHttpsAssetSource(archive.source)) {
      check(
        !activation || name === "fixtures",
        "Dedicated OCR model assets require HTTPS",
      );
      const path = `${accelerator}/${safePath(archive.source)}`;
      check(
        contents[path] &&
          manifest.assets[path].sha256 === archive.sha256 &&
          manifest.assets[path].size_bytes === archive.size_bytes,
        `Missing archive identity: ${name}`,
      );
      allowedAssets.add(path);
    }
  }
  check(
    new Set(bundle.fixtures.cases.map((row) => row.kind)).size === 5 &&
      new Set(bundle.fixtures.cases.map((row) => row.sha256)).size === 5,
    "Five distinct fixture kinds/identities required",
  );
  for (const row of bundle.fixtures.cases) {
    check(
      bundle.fixtures.files[safePath(row.path)]?.sha256 === row.sha256 &&
        bundle.fixtures.files[safePath(row.license.text_file)]?.size_bytes > 0,
      "Fixture inventory/license mismatch",
    );
    if (["table", "formula"].includes(row.kind))
      check(
        row.required_labels?.includes(row.kind),
        "Native structure qualification label missing",
      );
  }
  check(
    config.os === "linux" && config.architecture === "amd64",
    "Wrong OCI platform",
  );
  check(
    ["", "0", "root", "0:0", "root:root"].includes(config.config?.User ?? ""),
    "Prepared native image must run as root for selected device owner access",
  );
  check(
    `sha256:${manifest.assets[native.image_config].sha256}` ===
      image.config_digest &&
      bundle.image_config_sha256 === image.config_digest.slice(7),
    "Raw image config digest identity mismatch",
  );
  check(
    bundle.accelerator === accelerator &&
      bundle.image_identity === image.reference,
    "Native bundle image identity mismatch",
  );
  check(
    report.qualified === true &&
      report.stage === "complete" &&
      report.accelerator === accelerator &&
      report.image_identity === image.reference,
    "Native hardware qualification incomplete",
  );
  const seal = read(native.seal);
  const modelContent = Object.fromEntries(
    Object.entries(bundle.models).map(([name, model]) => [
      name,
      model.tree_sha256,
    ]),
  );
  check(
    json(seal.model_content) === json(modelContent) &&
      json(report.model_content) === json(modelContent),
    "Qualified report/seal model inventory differs from distributed models",
  );
  check(
    json(seal.qualified_target) ===
      json(
        Object.fromEntries(
          Object.entries(report.device ?? {}).filter(
            ([key]) => key !== "compute_probe",
          ),
        ),
      ),
    "Qualified seal device identity differs from report",
  );
  check(
    seal.smoke_sha256 ===
      bundle.fixtures.cases.find((row) => row.kind === "upright").sha256,
    "Qualified seal smoke fixture differs from distributed fixture",
  );
  check(
    report.manifest_sha256 === manifest.assets[native.seal].sha256 &&
      ["profile_id", "runtime_fingerprint"].every((key) =>
        /^[a-f0-9]{64}$/.test(report[key]),
      ) &&
      seal.image_identity === image.reference &&
      seal.paddle_identity === `paddle:${bundle.expected_paddle_commit}`,
    "Native qualification evidence seal mismatch",
  );
  const target = report.device?.gfx_target || report.device?.name;
  check(
    target && report.qualified_targets?.includes(target),
    "Actual qualified device target missing",
  );
  if (accelerator === "rocm")
    check(
      /^gfx[0-9a-f]+$/.test(report.device?.gfx_target),
      "Actual AMD GFX target evidence missing",
    );
  check(
    report.cases?.length === 10 &&
      ["upright", "rotated", "warped", "table", "formula"].every((kind) =>
        [0, 1].every(
          (worker) =>
            report.cases.filter(
              (row) =>
                row.kind === kind &&
                row.fresh_worker === worker &&
                row.source_sha256 ===
                  bundle.fixtures.cases.find((fixture) => fixture.kind === kind)
                    .sha256 &&
                row.passed === true,
            ).length === 1,
        ),
      ),
    "Full native fresh-worker qualification missing",
  );
  if (accelerator === "cuda") {
    const lockPath = "cuda/requirements.cuda.lock";
    allowedAssets.add(lockPath);
    const lock = read(native.cuda_lock);
    validateCudaLock(lock, native.cuda_channel ?? legacyCudaChannel);
    check(
      contents[lockPath] &&
        manifest.assets[lockPath].size_bytes <= 32 * 1024 * 1024 &&
        manifest.assets[lockPath].sha256 === lock.lock_sha256,
      "Resolved CUDA lock hash mismatch",
    );
    verifyExecutableCudaLock(readFileSync(contents[lockPath], "utf8"), lock);
    const measuredLock = config.config?.Labels?.[cudaLockDigestLabel];
    check(
      measuredLock === lock.lock_sha256 ||
        (measuredLock === undefined &&
          native.cuda_channel === undefined &&
          !Object.hasOwn(lock, "cuda_channel")),
      "Candidate image CUDA lock differs from attested executable lock",
    );
  }
  if (activation)
    check(
      Object.keys(contents).every((path) => allowedAssets.has(path)),
      "Unexpected Dedicated OCR release asset",
    );
  // Native initializer independently enforces the complete release-bundle schema,
  // source locks, model inventories/licenses and fresh hardware qualification.
  rejectPlaceholders(bundle);
  rejectPlaceholders(report);
}
export function writeBundle(manifest, variant, output, base) {
  const rendered = renderBundle(manifest, variant),
    contents = {};
  for (const path of rendered.assets)
    contents[path] = checkedRead(base, path, manifest.assets[path]);
  verifyEvidence(manifest, variant, contents);
  check(
    !existsSync(output),
    "Output already exists; immutable bundles cannot be overwritten",
  );
  mkdirSync(dirname(resolve(output)), { recursive: true });
  const temporary = mkdtempSync(
    join(dirname(resolve(output)), ".distribution-"),
  );
  try {
    const identities = {};
    for (const [path, bytes] of Object.entries(rendered.files)) {
      writeFileSync(join(temporary, path), bytes, {
        mode: path.endsWith(".sh") ? 0o755 : 0o644,
        flag: "wx",
      });
      identities[path] = hash(bytes);
    }
    for (const [path, source] of Object.entries(contents)) {
      const destination = join(temporary, "release", path);
      mkdirSync(dirname(destination), { recursive: true });
      copyFileSync(source, destination);
      identities[`release/${path}`] = manifest.assets[path].sha256;
    }
    writeFileSync(
      join(temporary, "SHA256SUMS"),
      Object.keys(identities)
        .sort()
        .map((path) => `${identities[path]}  ${path}\n`)
        .join(""),
      { mode: 0o644, flag: "wx" },
    );
    renameSync(temporary, output);
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
}
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  try {
    const args = process.argv.slice(2),
      options = {};
    check(
      args.length === 6,
      "Usage: --manifest FILE --variant VARIANT --output DIR",
    );
    for (let i = 0; i < args.length; i += 2) {
      check(
        ["--manifest", "--variant", "--output"].includes(args[i]) &&
          !options[args[i]],
        "Invalid CLI option",
      );
      options[args[i]] = args[i + 1];
    }
    const path = resolve(options["--manifest"]);
    writeBundle(
      JSON.parse(readFileSync(path)),
      options["--variant"],
      resolve(options["--output"]),
      dirname(path),
    );
  } catch (error) {
    process.stderr.write(`Distribution failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}
