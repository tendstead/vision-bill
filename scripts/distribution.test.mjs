import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  readdirSync,
  chmodSync,
  renameSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { spawnSync } from "node:child_process";
import "./windows-launcher-project.test.mjs";
import {
  validateArtifacts,
  renderBundle,
  writeBundle,
} from "./distribution.mjs";

const hash = (value) => createHash("sha256").update(value).digest("hex");
const digest = (value) => `sha256:${hash(value)}`;
const image = (name) => ({
  reference: `ghcr.io/maintainer/vision-bill-${name}@${digest(name)}`,
  config_digest: digest(`${name}-config`),
});
function fixture({ cudaChannel, attestation = {}, lock, measuredLock } = {}) {
  const cuda = {
    index: `https://www.paddlepaddle.org.cn/packages/stable/${cudaChannel ?? "cu126"}/paddlepaddle-gpu/`,
    index_sha256: hash("index"),
    wheel_url:
      "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl",
    wheel_sha256: hash("wheel"),
    metadata_sha256: hash("metadata"),
    metadata_name: "paddlepaddle-gpu",
    metadata_version: "3.3.1",
    requires_dist: ["numpy"],
    platform: "linux-x86_64-cp312",
    ...(cudaChannel ? { cuda_channel: cudaChannel } : {}),
    ...attestation,
  };
  const pin = `paddlepaddle-gpu @ ${cuda.wheel_url} --hash=sha256:${cuda.wheel_sha256}`;
  const executable = lock ? lock(pin) : `${pin}\n`;
  cuda.lock_sha256 = hash(executable);

  const assets = {};
  const bytes = {};
  function asset(path, value) {
    const content = typeof value === "string" ? value : JSON.stringify(value);
    bytes[path] = content;
    assets[path] = {
      sha256: hash(content),
      size_bytes: Buffer.byteLength(content),
    };
    return path;
  }
  const manifest = {
    schema_version: 2,
    version: "1.2.3",
    platform: "linux/amd64",
    source: {
      repository: "maintainer/vision-bill",
      commit: hash("source").slice(0, 40),
    },
    images: {
      app: image("app"),
      postgres: image("postgres"),
      "ocr-cuda": image("ocr-cuda"),
      "ocr-rocm": image("ocr-rocm"),
    },
    verification: { app_checks: "checks.json" },
    native: {},
    assets,
  };
  asset("checks.json", {
    passed: true,
    source_commit: manifest.source.commit,
    checks: ["backend", "frontend", "e2e-backend", "e2e-frontend"],
  });
  for (const accelerator of ["cuda", "rocm"]) {
    const config = JSON.stringify({
      architecture: "amd64",
      os: "linux",
      config: {
        Cmd: ["python", "-m", "ocr_service.entrypoint"],
        ...(accelerator === "cuda" && cudaChannel
          ? {
              Labels: {
                "io.vision-bill.cuda.lock-sha256":
                  measuredLock === undefined ? hash(executable) : measuredLock,
              },
            }
          : {}),
      },
    });
    manifest.images[`ocr-${accelerator}`].config_digest = digest(config);
    const nativeLicense = {
      spdx: "Apache-2.0",
      text_file: "LICENSE",
      redistribution_allowed: true,
    };
    const provenance = {
      url: "https://huggingface.co/PaddlePaddle",
      revision: hash("revision"),
    };
    const archive = (name) => {
      const files = {
        LICENSE: { sha256: hash("license-text"), size_bytes: 12 },
        [name]: { sha256: hash(name), size_bytes: 15 },
      };
      return {
        source: `https://models.vision-bill.org/${name}.tar`,
        sha256: hash(`${name}-tar`),
        size_bytes: 128,
        unpacked_bytes: 27,
        files,
        tree_sha256: hash(
          JSON.stringify(
            Object.fromEntries(
              Object.entries(files)
                .sort()
                .map(([path, record]) => [path, record.sha256]),
            ),
          ),
        ),
        license: nativeLicense,
        provenance,
      };
    };
    const models = Object.fromEntries(
      [
        "PP-DocLayoutV3",
        "PP-LCNet_x1_0_doc_ori",
        "UVDoc",
        "PaddleOCR-VL-1.6-0.9B",
      ].map((name) => [name, archive(name)]),
    );
    const fixtures = archive("upright");
    fixtures.cases = ["upright", "rotated", "warped", "table", "formula"].map(
      (kind) => ({
        kind,
        path: kind,
        sha256: hash(kind),
        required_labels: ["table", "formula"].includes(kind) ? [kind] : [],
        license: nativeLicense,
        provenance,
      }),
    );
    for (const kind of ["rotated", "warped", "table", "formula"])
      fixtures.files[kind] = { sha256: hash(kind), size_bytes: 15 };
    fixtures.unpacked_bytes = 87;
    fixtures.tree_sha256 = hash(
      JSON.stringify(
        Object.fromEntries(
          Object.entries(fixtures.files)
            .sort()
            .map(([path, record]) => [path, record.sha256]),
        ),
      ),
    );
    const bundle = {
      schema_version: 1,
      accelerator,
      gpu_index: 0,
      image_identity: manifest.images[`ocr-${accelerator}`].reference,
      image_config_sha256: hash(config),
      models,
      fixtures,
      profile_sha256: hash("profile"),
      source_files: {
        "profile.json": hash("profile"),
        "runtime-lock.json": hash("source-lock"),
      },
      expected_paddle_commit: hash("paddle-source").slice(0, 40),
      expected_packages: {
        paddleocr: "3.7.0",
        paddlex: "3.7.2",
        "paddlepaddle-gpu": "3.3.1",
      },
    };
    const seal = {
      image_identity: bundle.image_identity,
      paddle_identity: `paddle:${bundle.expected_paddle_commit}`,
      model_content: Object.fromEntries(
        Object.entries(models).map(([name, row]) => [name, row.tree_sha256]),
      ),
      qualified_target:
        accelerator === "rocm"
          ? { gfx_target: "gfx1030", name: "Radeon RX 6800" }
          : { name: "NVIDIA RTX 3090" },
      smoke_sha256: fixtures.cases.find((row) => row.kind === "upright").sha256,
    };
    const report = {
      qualified: true,
      stage: "complete",
      accelerator,
      image_identity: bundle.image_identity,
      profile_id: hash("profile-id"),
      runtime_fingerprint: hash("runtime-id"),
      model_content: structuredClone(seal.model_content),
      manifest_sha256: hash(JSON.stringify(seal)),
      device:
        accelerator === "rocm"
          ? { gfx_target: "gfx1030", name: "Radeon RX 6800" }
          : { name: "NVIDIA RTX 3090" },
      qualified_targets: [
        accelerator === "rocm" ? "gfx1030" : "NVIDIA RTX 3090",
      ],
      cases: ["upright", "rotated", "warped", "table", "formula"].flatMap(
        (kind) =>
          [0, 1].map((fresh_worker) => ({
            kind,
            fresh_worker,
            passed: true,
            source_sha256: fixtures.cases.find((row) => row.kind === kind)
              .sha256,
          })),
      ),
    };
    manifest.native[accelerator] = {
      bundle: asset(`${accelerator}/bundle.json`, bundle),
      image_config: asset(`${accelerator}/image-config.json`, config),
      qualification: asset(`${accelerator}/qualification.json`, report),
      seal: asset(`${accelerator}/qualification-manifest.json`, seal),
      ...(accelerator === "cuda"
        ? {
            cuda_lock: asset("cuda/lock-attestation.json", cuda),
            ...(cudaChannel ? { cuda_channel: cudaChannel } : {}),
          }
        : {}),
    };
    if (accelerator === "cuda")
      asset("cuda/requirements.cuda.lock", executable);
  }
  return { manifest, bytes };
}
const variants = [
  "app",
  "ocr-cuda",
  "ocr-rocm",
  "app-ocr-cuda",
  "app-ocr-rocm",
];
function temp(t) {
  const path = mkdtempSync(join(tmpdir(), "vb-distribution-"));
  t.after(() => rmSync(path, { recursive: true, force: true }));
  return path;
}
function seed(path, bytes) {
  for (const [name, value] of Object.entries(bytes)) {
    mkdirSync(resolve(path, name, ".."), { recursive: true });
    writeFileSync(join(path, name), value);
  }
}

// A wrong digest or omitted release evidence must make production validation fail.
test("artifact validation rejects absent, mutable, placeholder and contradictory identities", () => {
  const { manifest } = fixture();
  assert.equal(validateArtifacts(manifest), manifest);
  for (const change of [
    (m) => delete m.source.commit,
    (m) => (m.source.commit = "0".repeat(40)),
    (m) => (m.images.app.reference = "ghcr.io/maintainer/app:latest"),
    (m) => (m.images.app.config_digest = "sha256:" + "0".repeat(64)),
    (m) => (m.images.app.reference = `ghcr.io/example/app@${digest("a")}`),
    (m) =>
      (m.images.app.reference =
        "ghcr.io/maintainer/app@sha256:" + "0".repeat(64)),
    (m) => delete m.native.cuda.qualification,
    (m) => delete m.native.cuda.cuda_lock,
    (m) => (m.llm = {}),
    (m) => (m.images["ollama-cuda"] = image("retired")),
    (m) => (m.assets["../outside"] = { sha256: hash("x"), size_bytes: 1 }),
    (m) => (m.images.app.password = "secret"),
    (m) => (m.unexpected = true),
  ]) {
    const bad = structuredClone(manifest);
    change(bad);
    assert.throws(() => validateArtifacts(bad));
  }
});

test("bundles are deterministic, isolated, immutable and contain only prebuilt images", () => {
  const { manifest } = fixture();
  for (const variant of variants) {
    const rendered = renderBundle(manifest, variant);
    assert.deepEqual(
      rendered,
      renderBundle(structuredClone(manifest), variant),
    );
    const reorder = (value) =>
      Array.isArray(value)
        ? value.map(reorder)
        : value && typeof value === "object"
          ? Object.fromEntries(
              Object.entries(value)
                .reverse()
                .map(([key, child]) => [key, reorder(child)]),
            )
          : value;
    assert.deepEqual(rendered, renderBundle(reorder(manifest), variant));
    const compose = JSON.parse(rendered.files["compose.yaml"]);
    const content = rendered.files["compose.yaml"];
    assert.doesNotMatch(
      content,
      /"build"|container_name|"external"|\/workspace|docker.sock|POSTGRES_PASSWORD"|AUTH__SECRET_KEY/,
    );
    for (const service of Object.values(compose.services))
      assert.match(service.image, /@sha256:[a-f0-9]{64}$/);
    for (const service of Object.values(compose.services))
      for (const mount of service.volumes ?? [])
        if (mount.type === "bind") assert.match(mount.source, /^\.\/release\//);
    if (variant === "app") {
      assert.deepEqual(Object.keys(compose.services).sort(), [
        "app",
        "config-init",
        "db",
      ]);
      assert.equal(compose.services.app.environment.OCR__ENDPOINT, undefined);
      assert.doesNotMatch(content, /ollama:11434|installation.lock/);
    } else if (variant.startsWith("ocr-")) {
      assert.equal(
        compose.services.ocr.environment.OCR_SERVICE_MODE,
        "${OCR_SERVICE_MODE:-standalone}",
      );
      assert.ok(compose.services.ocr.ports);
      assert.equal(compose.services.app, undefined);
      assert.equal(compose.services.ollama, undefined);
      assert.equal(compose.services["ocr-init"].command[0], "check");
      const maintenance = compose.services["ocr-maintenance"];
      assert.deepEqual(maintenance.profiles, ["ocr-maintenance"]);
      assert.equal(maintenance.depends_on, undefined);
      assert.equal(maintenance.command[0], "maintain");
      assert.ok(!maintenance.command.includes("--maintenance-intent"));
      assert.ok(!maintenance.command.includes("--handoff-lock"));
      assert.equal(
        maintenance.environment.OCR_MAINTENANCE_REQUIRE_HANDOFF,
        undefined,
      );
      assert.equal(compose.services["ocr-init"].restart, "no");
      assert.equal(compose.services["ocr-init"].init, false);
      assert.equal(compose.services.ocr.init, false);
      if (variant.endsWith("rocm"))
        assert.equal(compose.services.ocr.group_add, undefined);
      assert.deepEqual(compose.services.ocr.depends_on["ocr-init"], {
        condition: "service_completed_successfully",
      });
      assert.match(
        compose.services["ocr-init"].command.join(" "),
        /--image-id sha256:[a-f0-9]{64}/,
      );
      assert.equal(
        compose.services["ocr-init"].environment.OCR_INSTALLATION_LOCK_PATH,
        "/coord/installation.lock",
      );
      assert.equal(
        compose.services.ocr.environment.OCR_INSTALLATION_LOCK_PATH,
        "/coord/installation.lock",
      );
    }
    assert.ok(
      rendered.assets.every(
        (name) =>
          !name.startsWith(variant.endsWith("cuda") ? "rocm/" : "cuda/"),
      ),
    );
  }
  assert.throws(() => renderBundle(manifest, "latest"));
});

test("combined bundle starts a private idle controller without prequalifying or selecting OCR", () => {
  const { manifest } = fixture();
  for (const accelerator of ["cuda", "rocm"]) {
    const rendered = renderBundle(manifest, `app-ocr-${accelerator}`);
    const { services, volumes } = JSON.parse(rendered.files["compose.yaml"]);
    assert.deepEqual(Object.keys(services).sort(), [
      "app",
      "config-init",
      "db",
      "ocr",
    ]);
    assert.deepEqual(services.ocr.command, [
      "python",
      "-m",
      "ocr_service.activation",
    ]);
    assert.equal(
      services.ocr.environment.OCR_IMAGE_ID,
      manifest.images[`ocr-${accelerator}`].config_digest,
    );
    assert.equal(services.ocr.env_file, undefined);
    assert.equal(services.ocr.ports, undefined);
    assert.equal(services.app.environment.VISION_BILL_OCR_ACTIVATION, "true");
    assert.equal(
      services.app.environment.OCR_ACTIVATION_ENDPOINT,
      "http://ocr:8082",
    );
    assert.equal(
      services.app.environment.OCR_ACTIVATION_HANDOFF_LOCK_PATH,
      services.ocr.environment.OCR_HANDOFF_LOCK_PATH,
    );
    assert.equal(services.app.environment.OCR__ENDPOINT, undefined);
    assert.equal(services.app.environment.OCR__COORDINATION_PATH, undefined);
    assert.equal(
      services["config-init"].environment.VISION_BILL_MANAGED_OCR,
      undefined,
    );
    assert.doesNotMatch(rendered.files["runtime.env.example"], /^OCR__ENDPOINT=/m);
    assert.match(rendered.files["runtime.env.example"], /^# OCR_ACTIVATION_CONFIRM_HANDOFF=false$/m);
    const app = Object.fromEntries(
      services.app.volumes.map((mount) => [mount.target, mount]),
    );
    const ocr = Object.fromEntries(
      services.ocr.volumes.map((mount) => [mount.target, mount]),
    );
    for (const [target, appReadOnly, ocrReadOnly] of [
      ["/activation/request", false, true],
      ["/activation/status", true, false],
      ["/activation/lock", false, false],
      ["/coord", false, false],
    ]) {
      assert.equal(app[target].source, ocr[target].source);
      assert.equal(app[target].type, "volume");
      assert.equal(ocr[target].type, "volume");
      assert.deepEqual(volumes[app[target].source], {
        labels: { "io.vision-bill.installation": "${VISION_BILL_INSTALLATION_ID:-manual}" },
      });
      assert.equal(!!app[target].read_only, appReadOnly);
      assert.equal(!!ocr[target].read_only, ocrReadOnly);
    }
    assert.deepEqual(Object.keys(ocr).sort(), [
      "/activation/lock",
      "/activation/request",
      "/activation/status",
      "/coord",
      "/release",
      "/var/lib/ocr",
    ]);
    assert.equal(ocr["/release"].read_only, true);
    assert.equal(ocr["/var/lib/ocr"].read_only, undefined);
  }
});

test("materialization verifies raw OCI config and qualification before atomic complete bundle publication", (t) => {
  const directory = temp(t);
  const { manifest, bytes } = fixture();
  seed(directory, bytes);
  const output = join(directory, "out");
  writeBundle(manifest, "app-ocr-rocm", output, directory);
  const checksums = readFileSync(join(output, "SHA256SUMS"), "utf8");
  for (const line of checksums.trim().split("\n")) {
    const [expected, path] = line.split("  ");
    assert.equal(hash(readFileSync(join(output, path))), expected);
  }
  const metadata = JSON.parse(readFileSync(join(output, "release.json")));
  assert.equal(metadata.source.commit, manifest.source.commit);
  assert.equal(metadata.variant, "app-ocr-rocm");
  assert.throws(() => writeBundle(manifest, "app-ocr-rocm", output, directory));
  writeFileSync(join(directory, "rocm/image-config.json"), "{}");
  const failed = join(directory, "failed");
  assert.throws(
    () => writeBundle(manifest, "ocr-rocm", failed, directory),
    /identity|hash|digest/i,
  );
  assert.ok(!readdirSync(directory).includes("failed"));
});

test("only the combined CUDA distribution ships a checksummed Windows launcher", (t) => {
  const directory = temp(t);
  const { manifest, bytes } = fixture();
  const name = "start-windows.ps1";
  for (const variant of variants) {
    assert.equal(
      Object.hasOwn(renderBundle(manifest, variant).files, name),
      variant === "app-ocr-cuda",
    );
  }
  seed(directory, bytes);
  const output = join(directory, "out");
  writeBundle(manifest, "app-ocr-cuda", output, directory);
  const launcher = readFileSync(join(output, name));
  assert.deepEqual(
    launcher,
    readFileSync(new URL(`../distribution/${name}`, import.meta.url)),
  );
  assert.ok(
    readFileSync(join(output, "SHA256SUMS"), "utf8")
      .split("\n")
      .includes(`${hash(launcher)}  ${name}`),
  );
  assert.match(
    launcher.toString(),
    /^Assert-Bundle\r?\nif \(\$VerifyOnly\).*\r?\n[^]*Get-Command docker\.exe/m,
  );
  assert.doesNotMatch(launcher.toString(), /\[string\]\$ProjectName\s*=/);
  assert.match(
    launcher.toString(),
    /\$project = Get-InstallProject[^]*Assert-ProjectClaim \$project\r?\n\s*Assert-ProjectResources \$project\.Name \$project\.Owner[^]*up --detach[^]*Assert-ProjectClaim \$project\r?\n\s*Assert-ProjectResources \$project\.Name \$project\.Owner[^]*initial-admin\.yaml/,
  );
});

test("combined CUDA and ROCm bundles ship the Linux starter and label owned resources", (t) => {
  const directory = temp(t);
  const { manifest, bytes } = fixture();
  const name = "start-linux.sh";
  const label = "io.vision-bill.installation";
  for (const variant of variants) {
    const { files } = renderBundle(manifest, variant);
    const combined = variant.startsWith("app-ocr-");
    assert.equal(Object.hasOwn(files, name), combined);
    const compose = JSON.parse(files["compose.yaml"]);
    for (const resource of [
      ...Object.values(compose.services),
      ...Object.values(compose.volumes),
      ...Object.values(compose.networks ?? {}),
    ]) {
      assert.equal(
        resource.labels?.[label],
        combined ? "${VISION_BILL_INSTALLATION_ID:-manual}" : undefined,
      );
    }
  }
  seed(directory, bytes);
  const output = join(directory, "out");
  writeBundle(manifest, "app-ocr-rocm", output, directory);
  const launcher = readFileSync(join(output, name));
  assert.deepEqual(
    launcher,
    readFileSync(new URL(`../distribution/${name}`, import.meta.url)),
  );
  assert.ok(
    readFileSync(join(output, "SHA256SUMS"), "utf8")
      .split("\n")
      .includes(`${hash(launcher)}  ${name}`),
  );
});

test("the Windows launcher verifies the complete file inventory without Docker", (t) => {
  const powershellCommands = process.platform === "win32" ? ["powershell.exe", "pwsh"] : ["pwsh"];
  const powershell = powershellCommands.find((command) => {
    return spawnSync(command, ["-NoProfile", "-Command", "exit 0"]).status === 0;
  });
  if (!powershell) return t.skip("PowerShell is unavailable");

  const directory = temp(t);
  const { manifest, bytes } = fixture();
  seed(directory, bytes);
  const output = join(directory, "out");
  writeBundle(manifest, "app-ocr-cuda", output, directory);
  const checksum = hash(readFileSync(join(output, "SHA256SUMS")));
  const verify = (expected = checksum) =>
    spawnSync(
      powershell,
      [
        "-NoProfile",
        "-NonInteractive",
        ...(process.platform === "win32" ? ["-ExecutionPolicy", "RemoteSigned"] : []),
        "-File",
        join(output, "start-windows.ps1"),
        "-VerifyOnly",
        "-ExpectedManifestSha256",
        expected,
      ],
      { encoding: "utf8" },
    );

  const success = verify();
  assert.equal(success.status, 0, success.stderr);
  assert.match(success.stdout, /Bundle files verified/);
  assert.notEqual(verify("0".repeat(64)).status, 0);

  writeFileSync(join(output, "runtime.env"), "LLM__API_KEY=$literal#value\n");
  writeFileSync(join(output, "windows-project.state"), `owned test ${"a".repeat(32)}\n`);
  assert.equal(verify().status, 0);
  writeFileSync(join(output, "release", "unexpected.txt"), "unexpected");
  assert.match(verify().stderr, /Unexpected file in the bundle/);
  rmSync(join(output, "release", "unexpected.txt"));

  writeFileSync(join(output, "compose.yaml"), "modified");
  assert.match(verify().stderr, /Checksum mismatch: compose\.yaml/);
});

test("CUDA bundle requires the selected channel and exact official wheel evidence", (t) => {
  const official =
    "https://paddle-whl.cdn.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl";
  const accepted = [
    {},
    { wheel_url: official.replace("/paddlepaddle-gpu/", "/") },
    {
      wheel_url: official.replace(
        "paddle-whl.cdn.bcebos.com/stable/",
        "www.paddlepaddle.org.cn/packages/stable/",
      ),
    },
  ];
  const rejected = [
    { cuda_channel: undefined },
    { cuda_channel: "cu126" },
    {
      index:
        "https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/",
    },
    { wheel_url: official.replace("/cu129/", "/cu126/") },
    { wheel_url: official.replace("/stable/", "/foreign/") },
    { wheel_url: official.replace(".com/", ".com.evil.invalid/") },
    { wheel_url: official.replace(".com/", ".com:443/") },
    { wheel_url: official.replace("https://", "https://user@") },
    { wheel_url: official + "?download=1" },
    { wheel_url: official + "#wheel" },
    { wheel_url: official.replace("3.3.1", "3.3.2") },
    { wheel_url: official.replaceAll("cp312", "cp313") },
    { wheel_url: official.replace("x86_64", "aarch64") },
  ];
  for (const change of [...accepted, ...rejected]) {
    const directory = temp(t);
    const { manifest, bytes } = fixture({
      cudaChannel: "cu129",
      attestation: { wheel_url: official, ...change },
    });
    seed(directory, bytes);
    const output = join(directory, "out");
    if (accepted.includes(change)) {
      writeBundle(manifest, "ocr-cuda", output, directory);
      const metadata = JSON.parse(readFileSync(join(output, "release.json")));
      assert.equal(metadata.native.cuda.cuda_channel, "cu129");
      delete manifest.native.cuda.cuda_channel;
      assert.throws(
        () =>
          writeBundle(
            manifest,
            "ocr-cuda",
            join(directory, "wrong"),
            directory,
          ),
        /official CUDA/i,
      );
    } else {
      assert.throws(
        () => writeBundle(manifest, "ocr-cuda", output, directory),
        /official CUDA/i,
      );
    }
  }
});

test("CUDA bundle verifies executable Paddle pin and the lock measured by the built image", (t) => {
  const wheelUrl =
    "https://paddle-whl.cdn.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl";
  const approvedDirectory = temp(t);
  const approved = fixture({
    cudaChannel: "cu129",
    attestation: { wheel_url: wheelUrl },
    lock: (pin) =>
      "# Generated by uv\n" + pin.replace(" --hash", " \\\n    --hash") + "\n",
  });
  seed(approvedDirectory, approved.bytes);
  writeBundle(
    approved.manifest,
    "ocr-cuda",
    join(approvedDirectory, "out"),
    approvedDirectory,
  );
  for (const lock of [
    () => "resolved-lock",
    (pin) => pin.replace("/cu129/", "/cu126/"),
    (pin) => pin.replace(hash("wheel"), hash("wrong-wheel")),
    (pin) => `${pin}\npaddlepaddle==3.3.1 --hash=sha256:${hash("other-wheel")}\n`,
    (pin) => `${pin} ; python_version == "3.12"\n`,
    (pin) => `${pin}\n-r alternate.lock\n`,
  ]) {
    const directory = temp(t);
    const { manifest, bytes } = fixture({
      cudaChannel: "cu129",
      attestation: { wheel_url: wheelUrl },
      lock,
    });
    seed(directory, bytes);
    assert.throws(
      () => writeBundle(manifest, "ocr-cuda", join(directory, "out"), directory),
      /executable CUDA lock|Paddle.*pin/i,
    );
  }
  for (const measuredLock of [null, hash("wrong-lock")]) {
    const directory = temp(t);
    const { manifest, bytes } = fixture({
      cudaChannel: "cu129",
      attestation: { wheel_url: wheelUrl },
      measuredLock,
    });
    seed(directory, bytes);
    assert.throws(
      () => writeBundle(manifest, "ocr-cuda", join(directory, "out"), directory),
      /candidate.*CUDA lock/i,
    );
  }
});

test("CUDA Make targets pass an expected channel by default and on explicit selection", (t) => {
  const directory = temp(t);
  writeFileSync(
    join(directory, "uv"),
    '#!/bin/sh\nprintf "%s\\n" "$@" >&2\nprintf "%s\\n" "$TEST_CONTEXT"\n',
    {
      mode: 0o755,
    },
  );
  writeFileSync(
    join(directory, "docker"),
    '#!/bin/sh\ntest "$OCR_CUDA_WHEEL_CONTEXT" = "$TEST_CONTEXT"\n',
    {
      mode: 0o755,
    },
  );
  for (const target of ["ocr-build-cuda", "ocr-lock-cuda"]) {
    for (const channel of [undefined, "", "cu126"]) {
      const env = {
        ...process.env,
        PATH: directory + ":" + process.env.PATH,
        TEST_CONTEXT: directory,
      };
      if (channel === undefined) delete env.OCR_CUDA_CHANNEL;
      else env.OCR_CUDA_CHANNEL = channel;
      const result = spawnSync("make", ["--no-print-directory", target], {
        encoding: "utf8",
        env,
      });
      assert.equal(result.status, 0, result.stderr);
      assert.ok(
        result.stderr.split("\n").includes("--channel=" + (channel || "cu129")),
      );
    }
  }
});

test("same-hash evidence with wrong config or incomplete hardware qualification fails closed", (t) => {
  for (const mutate of [
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.manifest_sha256 = hash("wrong-seal");
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
    (m, b) => {
      m.images["ocr-rocm"].config_digest = digest("different-config");
    },
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.qualified = false;
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.device.gfx_target = "gfx1100";
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
    (m, b) => {
      const path = m.native.rocm.qualification;
      const r = JSON.parse(b[path]);
      r.cases.pop();
      b[path] = JSON.stringify(r);
      m.assets[path] = { sha256: hash(b[path]), size_bytes: b[path].length };
    },
  ]) {
    const path = temp(t);
    const { manifest, bytes } = fixture();
    mutate(manifest, bytes);
    seed(path, bytes);
    assert.throws(() =>
      writeBundle(manifest, "ocr-rocm", join(path, "out"), path),
    );
  }
});

test("external native bundle requires complete model, fixture and licensed archive inventories", (t) => {
  for (const mutate of [
    (b) => delete b.models.UVDoc,
    (b) => delete b.models.UVDoc.license,
    (b) => (b.models.UVDoc.tree_sha256 = hash("wrong-tree")),
    (b) =>
      (b.models.UVDoc.files["../unsafe"] = {
        sha256: hash("unsafe"),
        size_bytes: 1,
      }),
    (b) => b.fixtures.cases.pop(),
    (b) => (b.fixtures.cases[4].required_labels = []),
    (b) => (b.expected_packages.paddleocr = "3.6.0"),
  ]) {
    const path = temp(t);
    const { manifest, bytes } = fixture();
    const name = manifest.native.rocm.bundle;
    const bundle = JSON.parse(bytes[name]);
    mutate(bundle);
    bytes[name] = JSON.stringify(bundle);
    manifest.assets[name] = {
      sha256: hash(bytes[name]),
      size_bytes: bytes[name].length,
    };
    seed(path, bytes);
    assert.throws(() =>
      writeBundle(manifest, "ocr-rocm", join(path, "out"), path),
    );
  }
});

test("native release materialization recognizes the ROCm DCU wheel and rejects conflicting identities", (t) => {
  for (const [accelerator, identities, rejection] of [
    ["rocm", ["paddlepaddle-dcu"]],
    ["rocm", ["paddlepaddle"]],
    ["rocm", ["paddlepaddle-gpu"]],
    ["rocm", ["paddlepaddle-dcu", "paddlepaddle-gpu"], /native Paddle/i],
    ["rocm", ["paddlepaddle-dcu", "PaddlePaddle_DCU"], /native Paddle/i],
    ["cuda", ["paddlepaddle-dcu"], /accelerator/i],
  ]) {
    const directory = temp(t);
    const { manifest, bytes } = fixture();
    const name = manifest.native[accelerator].bundle;
    const bundle = JSON.parse(bytes[name]);
    delete bundle.expected_packages["paddlepaddle-gpu"];
    for (const identity of identities)
      bundle.expected_packages[identity] = "3.3.1";
    bytes[name] = JSON.stringify(bundle);
    manifest.assets[name] = {
      sha256: hash(bytes[name]),
      size_bytes: Buffer.byteLength(bytes[name]),
    };
    seed(directory, bytes);
    const output = join(directory, "out");
    const materialize = () =>
      writeBundle(manifest, `ocr-${accelerator}`, output, directory);
    if (rejection) assert.throws(materialize, rejection);
    else {
      materialize();
      const packaged = JSON.parse(readFileSync(join(output, "release", name)));
      assert.equal(packaged.expected_packages[identities[0]], "3.3.1");
    }
  }
});

test("combined materialization requires pinned HTTPS model sources and permits only the declared local fixture archive", (t) => {
  const directory = temp(t);
  const { manifest, bytes } = fixture();
  const bundlePath = manifest.native.rocm.bundle;
  const bundle = JSON.parse(bytes[bundlePath]);
  const setAsset = (path, content) => {
    bytes[path] = content;
    manifest.assets[path] = {
      sha256: hash(content),
      size_bytes: Buffer.byteLength(content),
    };
  };
  const fixtureBytes = "sealed synthetic fixture archive";
  bundle.fixtures.source = "fixtures.tar";
  bundle.fixtures.sha256 = hash(fixtureBytes);
  bundle.fixtures.size_bytes = Buffer.byteLength(fixtureBytes);
  setAsset("rocm/fixtures.tar", fixtureBytes);
  setAsset(bundlePath, JSON.stringify(bundle));
  seed(directory, bytes);
  const accepted = join(directory, "accepted");
  writeBundle(manifest, "app-ocr-rocm", accepted, directory);
  assert.ok(
    readdirSync(join(accepted, "release", "rocm")).includes("fixtures.tar"),
  );
  assert.doesNotMatch(
    readFileSync(join(accepted, "SHA256SUMS"), "utf8"),
    /UVDoc\.tar/,
  );

  const modelBytes = "sealed synthetic model archive";
  setAsset("rocm/UVDoc.tar", modelBytes);
  seed(directory, bytes);
  const extra = join(directory, "extra");
  assert.throws(
    () => writeBundle(manifest, "app-ocr-rocm", extra, directory),
    /unexpected.+asset/i,
  );

  bundle.models.UVDoc.source = "UVDoc.tar";
  bundle.models.UVDoc.sha256 = hash(modelBytes);
  bundle.models.UVDoc.size_bytes = Buffer.byteLength(modelBytes);
  setAsset(bundlePath, JSON.stringify(bundle));
  seed(directory, bytes);
  const local = join(directory, "local");
  assert.throws(
    () => writeBundle(manifest, "app-ocr-rocm", local, directory),
    /HTTPS.*model|model.*HTTPS/i,
  );
  writeBundle(manifest, "ocr-rocm", join(directory, "standalone"), directory);
});

test("CLI generates every variant and Docker Compose validates effective config without services", (t) => {
  const directory = temp(t);
  const { manifest, bytes } = fixture();
  seed(directory, bytes);
  writeFileSync(join(directory, "artifacts.json"), JSON.stringify(manifest));
  const env = {
    ...process.env,
    COMPOSE_PROJECT_NAME: "vb-distribution-test",
    OCR_RENDER_NODE: "/dev/dri/renderD129",
  };
  delete env.OCR_RENDER_GID;
  delete env.OCR_VIDEO_GID;
  delete env.OCR_SERVICE_MODE;
  for (const variant of variants) {
    const output = join(directory, variant);
    const generated = spawnSync(
      process.execPath,
      [
        "scripts/distribution.mjs",
        "--manifest",
        join(directory, "artifacts.json"),
        "--variant",
        variant,
        "--output",
        output,
      ],
      { encoding: "utf8", env },
    );
    assert.equal(generated.status, 0, generated.stderr);
    writeFileSync(join(output, "runtime.env"), "", { mode: 0o600 });
    const composed = spawnSync(
      "docker",
      [
        "compose",
        "--profile",
        "ocr-maintenance",
        "-f",
        join(output, "compose.yaml"),
        "config",
        "--format",
        "json",
      ],
      { encoding: "utf8", env },
    );
    assert.equal(composed.status, 0, composed.stderr);
    const effective = JSON.parse(composed.stdout);
    if (variant.endsWith("-rocm")) {
      for (const name of ["ocr", "ocr-init", "ocr-maintenance"]) {
        const service = effective.services[name];
        if (!service) continue;
        assert.deepEqual(
          service.devices.map(({ source, target }) => [source, target]),
          [
            ["/dev/kfd", "/dev/kfd"],
            ["/dev/dri/renderD129", "/dev/dri/renderD129"],
          ],
          name,
        );
      }
    }
    if (variant.startsWith("app-ocr")) {
      assert.equal(effective.services.app.environment.OCR__ENDPOINT, undefined);
      assert.match(
        effective.volumes["ocr-activation-request"].name,
        /^vb-distribution-test_/,
      );
    }
    if (variant.startsWith("ocr-")) {
      assert.equal(
        effective.services.ocr.environment.OCR_SERVICE_MODE,
        "standalone",
      );
      const managed = spawnSync(
        "docker",
        [
          "compose",
          "-f",
          join(output, "compose.yaml"),
          "config",
          "--format",
          "json",
        ],
        { encoding: "utf8", env: { ...env, OCR_SERVICE_MODE: "managed" } },
      );
      assert.equal(managed.status, 0, managed.stderr);
      assert.equal(
        JSON.parse(managed.stdout).services.ocr.environment.OCR_SERVICE_MODE,
        "managed",
      );
    }
    for (const name of ["ocr-init", "ocr", "ollama"])
      if (effective.services[name])
        assert.equal(effective.services[name].init, false);
    assert.ok(
      !Object.values(effective.services).some((service) => service.build),
    );
  }
});

test("manual distribution workflow gates fork/exact-SHA, native qualification, publication and assets in order", () => {
  const workflow = readFileSync(
    ".github/workflows/distribution-release.yml",
    "utf8",
  );
  assert.match(workflow, /workflow_dispatch:/);
  assert.doesNotMatch(workflow, /^  (push|pull_request|schedule):/m);
  assert.match(workflow, /DISTRIBUTION_FORK_REPOSITORY/);
  assert.match(workflow, /github\.sha/);
  assert.match(workflow, /needs: \[verify, prepare\]/);
  assert.ok(
    workflow.indexOf("Qualify exact native image") <
      workflow.indexOf("Publish verified images"),
  );
  assert.ok(
    workflow.indexOf("Verify registry pullability") <
      workflow.indexOf("Publish release assets last"),
  );
  assert.match(
    readFileSync(".github/workflows/ci.yml", "utf8"),
    /make distribution-test/,
  );
});

test("maintainer request rejects foreign forks, events, mutable source and missing real inputs", async () => {
  const release = await import("./distribution-release.mjs");
  const env = {
    GITHUB_EVENT_NAME: "workflow_dispatch",
    GITHUB_REPOSITORY: "maintainer/vision-bill",
    DISTRIBUTION_FORK_REPOSITORY: "maintainer/vision-bill",
    GITHUB_SHA: hash("source").slice(0, 40),
    DISTRIBUTION_SOURCE_SHA: hash("source").slice(0, 40),
    DISTRIBUTION_VERSION: "1.2.3",
    DISTRIBUTION_VARIANT: "app-ocr-rocm",
    GITHUB_RUN_ID: "123",
  };
  const request = release.releaseRequest(env);
  assert.equal(request.variant, "app-ocr-rocm");
  for (const change of [
    { GITHUB_EVENT_NAME: "push" },
    { DISTRIBUTION_FORK_REPOSITORY: "" },
    { GITHUB_REPOSITORY: "upstream/vision-bill" },
    { DISTRIBUTION_SOURCE_SHA: hash("other").slice(0, 40) },
    { DISTRIBUTION_VERSION: "latest" },
    { DISTRIBUTION_VARIANT: "all" },
  ])
    assert.throws(() => release.releaseRequest({ ...env, ...change }));
  const plan = {
    schema_version: 2,
    source: { repository: request.repository, commit: request.sha },
    version: request.version,
    platform: "linux/amd64",
    postgres: image("postgres").reference,
    rocm: {
      targets: ["gfx1030"],
      render_node: "/dev/dri/renderD128",
      render_gid: 109,
      video_gid: 44,
    },
  };
  assert.equal(release.validateReleaseInputs(plan, request), plan);
  assert.throws(() =>
    release.validateReleaseInputs({ ...plan, cuda_channel: "cu129" }, request),
  );
  const cudaRequest = { ...request, variant: "app-ocr-cuda" };
  const { rocm, ...cudaPlan } = plan;
  assert.equal(release.validateReleaseInputs(cudaPlan, cudaRequest), cudaPlan);
  const uuid = "GPU-a1234567-aaaa-bbbb-cccc-12345678abcd";
  assert.deepEqual(release.cudaQualificationGpuArgs(), ["--gpus", "device=0"]);
  for (const id of ["1", uuid]) {
    release.validateReleaseInputs(
      { ...cudaPlan, cuda_device_id: id },
      cudaRequest,
    );
    assert.deepEqual(release.cudaQualificationGpuArgs(id), [
      "--gpus",
      `device=${id}`,
    ]);
  }
  for (const id of [1, null, "all", "1,2", "GPU-wrong", "-1"]) {
    assert.throws(() =>
      release.validateReleaseInputs(
        { ...cudaPlan, cuda_device_id: id },
        cudaRequest,
      ),
    );
  }
  assert.throws(() =>
    release.validateReleaseInputs({ ...plan, cuda_device_id: "0" }, request),
  );
  for (const cuda_channel of ["cu126", "cu129"]) {
    const selected = { ...cudaPlan, cuda_channel };
    assert.equal(
      release.validateReleaseInputs(selected, cudaRequest),
      selected,
    );
  }
  for (const cuda_channel of ["cu130", null, "latest", "cu129 "]) {
    assert.throws(() =>
      release.validateReleaseInputs({ ...cudaPlan, cuda_channel }, cudaRequest),
    );
  }
  for (const mutate of [
    (p) => (p.ollama_base = image("retired").reference),
    (p) => (p.llm = {}),
    (p) => (p.schema_version = 1),
    (p) => (p.rocm.targets = ["any"]),
    (p) => (p.rocm.render_node = "/dev"),
    (p) => (p.source.commit = hash("old").slice(0, 40)),
  ]) {
    const bad = structuredClone(plan);
    mutate(bad);
    assert.throws(() => release.validateReleaseInputs(bad, request));
  }
  const rootPlan = structuredClone(plan);
  delete rootPlan.rocm.render_gid;
  delete rootPlan.rocm.video_gid;
  assert.equal(release.validateReleaseInputs(rootPlan, request), rootPlan);
});

test("qualification and publication reject a CUDA candidate that differs from the producer plan", (t) => {
  const directory = temp(t);
  const { manifest } = fixture();
  const plan = {
    schema_version: 2,
    source: manifest.source,
    version: manifest.version,
    platform: manifest.platform,
    cuda_channel: "cu129",
  };
  seed(directory, {
    "manifest.json": JSON.stringify(manifest),
    "plan.json": JSON.stringify(plan),
    "images.json": "{}",
  });
  for (const phase of ["qualify", "publish"]) {
    const result = spawnSync(
      process.execPath,
      [
        "scripts/distribution-release.mjs",
        "--phase",
        phase,
        "--output",
        directory,
      ],
      {
        encoding: "utf8",
        env: {
          ...process.env,
          GITHUB_EVENT_NAME: "workflow_dispatch",
          GITHUB_REPOSITORY: manifest.source.repository,
          DISTRIBUTION_FORK_REPOSITORY: manifest.source.repository,
          GITHUB_SHA: manifest.source.commit,
          DISTRIBUTION_SOURCE_SHA: manifest.source.commit,
          DISTRIBUTION_VERSION: manifest.version,
          DISTRIBUTION_VARIANT: "ocr-cuda",
          GITHUB_RUN_ID: "123",
          DISTRIBUTION_CI_RESULT: "success",
        },
      },
    );
    assert.equal(result.status, 1, result.stderr);
    assert.match(
      result.stderr,
      /Candidate CUDA channel differs from approved producer plan/,
    );
  }
});

test("release binds approved package pins and source identities to the exact candidate image", async () => {
  const { candidateBundleEvidence } =
    await import("./distribution-release.mjs");
  const bundle = {
    accelerator: "cuda",
    expected_packages: { "paddlepaddle-gpu": "3.3.1", paddleocr: "3.7.0" },
  };
  const expected_packages = { ...bundle.expected_packages, numpy: "2.2.6" };
  const source_files = {
    "profile.json": hash("profile"),
    "runtime-lock.json": hash("runtime"),
  };
  const actual = candidateBundleEvidence(
    "immutable-candidate:local",
    bundle,
    (command, args) => {
      assert.equal(command, "docker");
      assert.deepEqual(args.slice(0, 8), [
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python",
        "immutable-candidate:local",
        "-c",
      ]);
      const [probe, pins, accelerator] = args.slice(8);
      assert.match(probe, /bind_approved_dependencies/);
      assert.deepEqual(JSON.parse(pins), bundle.expected_packages);
      assert.equal(accelerator, bundle.accelerator);
      return Buffer.from(JSON.stringify({ expected_packages, source_files }));
    },
  );
  assert.deepEqual(actual, { expected_packages, source_files });
});

test("native bundles reject non-root image config", (t) => {
  const directory = temp(t),
    { manifest, bytes } = fixture(),
    native = manifest.native.rocm;
  const config = JSON.parse(bytes[native.image_config]);
  config.config.User = "1000:1000";
  bytes[native.image_config] = JSON.stringify(config);
  manifest.images["ocr-rocm"].config_digest = digest(
    bytes[native.image_config],
  );
  const bundle = JSON.parse(bytes[native.bundle]);
  bundle.image_config_sha256 = hash(bytes[native.image_config]);
  bytes[native.bundle] = JSON.stringify(bundle);
  for (const name of [native.bundle, native.image_config])
    manifest.assets[name] = {
      sha256: hash(bytes[name]),
      size_bytes: bytes[name].length,
    };
  seed(directory, bytes);
  assert.throws(
    () => writeBundle(manifest, "ocr-rocm", join(directory, "out"), directory),
    /root/,
  );
});

test("publication cannot create assets after qualification, upload or pullability failure", async () => {
  const { publishVerified } = await import("./distribution-release.mjs");
  const { manifest } = fixture();
  for (const failure of ["verify", "image", "pull", null]) {
    const calls = [];
    const op = (method) => async () => {
      calls.push(method);
      if (method === failure) throw Error("gate failed");
    };
    const promise = publishVerified(manifest, "app-ocr-cuda", {
      verify: op("verify"),
      image: op("image"),
      pull: op("pull"),
      assets: op("assets"),
    });
    if (failure) {
      await assert.rejects(promise, /gate failed/);
      assert.ok(!calls.includes("assets"));
    } else {
      await promise;
      assert.equal(calls.at(-1), "assets");
      assert.ok(calls.indexOf("image") > calls.indexOf("verify"));
      assert.ok(calls.indexOf("pull") > calls.lastIndexOf("image"));
    }
  }
});

test("failed qualification container creation never removes a preexisting named container", async () => {
  const { withOwnedContainer } = await import("./distribution-release.mjs");
  const removed = [],
    id = hash("owned-container");
  assert.throws(
    () =>
      withOwnedContainer(
        {
          create() {
            throw Error("name collision");
          },
          remove(value) {
            removed.push(value);
          },
        },
        () => {},
      ),
    /name collision/,
  );
  assert.deepEqual(removed, []);
  assert.throws(
    () =>
      withOwnedContainer(
        {
          create: () => id,
          remove(value) {
            removed.push(value);
          },
        },
        () => {
          throw Error("gate failed");
        },
      ),
    /gate failed/,
  );
  assert.deepEqual(removed, [id]);
});

test("partial publication can reuse only the exact already-qualified image digest", async () => {
  const { publishImageIfNeeded } = await import("./distribution-release.mjs");
  const expected = image("app");
  const copied = [];
  await publishImageIfNeeded(expected, {
    lookup: async () => null,
    copy: async () => copied.push("copy"),
  });
  await publishImageIfNeeded(expected, {
    lookup: async () => expected.reference.split("@")[1],
    copy: async () => copied.push("wrong"),
  });
  assert.deepEqual(copied, ["copy"]);
  await assert.rejects(
    publishImageIfNeeded(expected, {
      lookup: async () => digest("foreign"),
      copy: async () => copied.push("wrong"),
    }),
  );
  await assert.rejects(
    publishImageIfNeeded(expected, {
      lookup: async () => {
        throw Error("denied");
      },
      copy: async () => copied.push("wrong"),
    }),
    /denied/,
  );
  assert.deepEqual(copied, ["copy"]);
});

test("CPU PID1 gate selects current Python minor and read-only environment without a shell or init process", async () => {
  const { pid1Command } = await import("./distribution-pid1-test.mjs");
  const args = pid1Command(
    {
      version: "3.13",
      site_packages: "/private/env/lib/python3.13/site-packages",
    },
    "/private/repo",
  );
  assert.deepEqual(args.slice(0, 6), [
    "run",
    "--rm",
    "--init=false",
    "--network",
    "none",
    "--mount",
  ]);
  assert.ok(args.includes("python:3.13-slim"));
  assert.ok(
    args.includes(
      "type=bind,source=/private/repo,target=/private/repo,readonly",
    ),
  );
  assert.ok(
    args.includes(
      "type=bind,source=/private/env/lib/python3.13/site-packages,target=/testdeps,readonly",
    ),
  );
  assert.deepEqual(
    args.slice(
      args.indexOf("python:3.13-slim") + 1,
      args.indexOf("python:3.13-slim") + 4,
    ),
    ["python", "-m", "pytest"],
  );
  assert.ok(
    args.includes("tests/test_ocr_initialize.py") &&
      args.includes("tests/test_ocr_provision.py"),
  );
  assert.throws(() =>
    pid1Command(
      { version: "latest", site_packages: "/private/env" },
      "/private/repo",
    ),
  );
  assert.match(
    readFileSync(".github/workflows/ci.yml", "utf8"),
    /make distribution-pid1-test/,
  );
});

test("qualification evidence cannot be reused for internally consistent replacement models, fixtures or devices", (t) => {
  for (const replace of [
    "model",
    "fixture",
    "fixture-table",
    "device",
    "report-model",
    "seal-model",
  ]) {
    const directory = temp(t),
      { manifest, bytes } = fixture(),
      native = manifest.native.rocm;
    const bundle = JSON.parse(bytes[native.bundle]),
      report = JSON.parse(bytes[native.qualification]),
      seal = JSON.parse(bytes[native.seal]);
    if (replace === "model") {
      const model = bundle.models.UVDoc;
      model.sha256 = hash("different-archive");
      model.files.UVDoc.sha256 = hash("different-weights");
      model.tree_sha256 = hash(
        JSON.stringify(
          Object.fromEntries(
            Object.entries(model.files)
              .sort()
              .map(([name, row]) => [name, row.sha256]),
          ),
        ),
      );
    }
    if (replace === "fixture" || replace === "fixture-table") {
      const row = bundle.fixtures.cases.find(
        (row) => row.kind === (replace === "fixture" ? "upright" : "table"),
      );
      row.sha256 = hash("different-upright");
      bundle.fixtures.files[row.path].sha256 = row.sha256;
      bundle.fixtures.tree_sha256 = hash(
        JSON.stringify(
          Object.fromEntries(
            Object.entries(bundle.fixtures.files)
              .sort()
              .map(([name, file]) => [name, file.sha256]),
          ),
        ),
      );
    }
    if (replace === "device") {
      report.device.name = "Different Radeon";
      report.device.gfx_target = "gfx1031";
      report.qualified_targets = ["gfx1031"];
    }
    if (replace === "report-model")
      report.model_content.UVDoc = hash("wrong-report-model");
    if (replace === "seal-model") {
      delete seal.model_content.UVDoc;
      report.manifest_sha256 = hash(JSON.stringify(seal));
    }
    for (const [path, value] of [
      [native.bundle, bundle],
      [native.qualification, report],
      [native.seal, seal],
    ]) {
      bytes[path] = JSON.stringify(value);
      manifest.assets[path] = {
        sha256: hash(bytes[path]),
        size_bytes: bytes[path].length,
      };
    }
    seed(directory, bytes);
    const output = join(directory, "out");
    assert.throws(
      () => writeBundle(manifest, "ocr-rocm", output, directory),
      /model|fixture|device|seal|qualification/i,
    );
    assert.ok(!readdirSync(directory).includes("out"));
  }
});

test("approved producer run is bound to exact workflow, maintainer actor, source and manual completion", async () => {
  const { validateInputRun, PRODUCER_WORKFLOW } =
    await import("./distribution-inputs.mjs");
  const request = {
    repository: "maintainer/vision-bill",
    sha: hash("source").slice(0, 40),
    version: "1.2.3",
    variant: "app",
  };
  const workflow = { id: 7, path: PRODUCER_WORKFLOW },
    permission = { role_name: "maintain", user: { login: "maintainer" } };
  const run = {
    id: 123,
    workflow_id: 7,
    path: PRODUCER_WORKFLOW,
    head_sha: request.sha,
    head_repository: { full_name: request.repository },
    event: "workflow_dispatch",
    conclusion: "success",
    actor: { login: "maintainer" },
    triggering_actor: { login: "maintainer" },
  };
  assert.equal(
    validateInputRun(run, request, workflow, permission),
    "maintainer",
  );
  for (const mutate of [
    (r) => (r.workflow_id = 8),
    (r) => (r.path = ".github/workflows/other.yml"),
    (r) => (r.actor.login = "foreign"),
    (r) => (r.triggering_actor.login = "foreign"),
    (r) => (r.head_sha = hash("old").slice(0, 40)),
    (r) => (r.event = "push"),
    (r) => (r.conclusion = "failure"),
  ]) {
    const bad = structuredClone(run);
    mutate(bad);
    assert.throws(() => validateInputRun(bad, request, workflow, permission));
  }
  assert.throws(() =>
    validateInputRun(run, request, workflow, {
      ...permission,
      role_name: "write",
    }),
  );
});

test("producer stages only approved exact-version inputs and consumer verifies actor-bound complete inventory", async (t) => {
  const { stageInputs, verifyStagedInputs } =
    await import("./distribution-inputs.mjs");
  const directory = temp(t),
    sourceRoot = join(directory, "approved"),
    request = {
      repository: "maintainer/vision-bill",
      sha: hash("source").slice(0, 40),
      version: "1.2.3",
      variant: "app-ocr-rocm",
      run: "123",
    };
  const selected = join(
      sourceRoot,
      request.sha,
      request.version,
      request.variant,
    ),
    { manifest, bytes } = fixture();
  const plan = {
    schema_version: 2,
    source: manifest.source,
    version: request.version,
    platform: manifest.platform,
    postgres: manifest.images.postgres.reference,
    rocm: { targets: ["gfx1030"], render_node: "/dev/dri/renderD128" },
  };
  mkdirSync(selected, { recursive: true, mode: 0o700 });
  seed(selected, {
    "release-inputs.json": JSON.stringify(plan),
    "rocm/bundle.json": bytes["rocm/bundle.json"],
    "private-extra.txt": "must not upload",
  });
  const output = join(directory, "artifact");
  stageInputs(
    request,
    { actor: "maintainer", role: "maintain" },
    sourceRoot,
    output,
  );
  assert.ok(!readdirSync(output).includes("private-extra.txt"));
  verifyStagedInputs(output, request, { actor: "maintainer", run: "123" });
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "maintainer", run: "124" }),
  );
  writeFileSync(join(output, "unexpected.txt"), "must not consume");
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "maintainer", run: "123" }),
  );
  rmSync(join(output, "unexpected.txt"));
  writeFileSync(
    join(selected, "release-inputs.json"),
    JSON.stringify({ ...plan, version: "1.2.4" }),
  );
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "wrong-version"),
    ),
  );
  writeFileSync(join(selected, "release-inputs.json"), JSON.stringify(plan));
  const localBundle = JSON.parse(bytes["rocm/bundle.json"]);
  localBundle.models.UVDoc.source = "../../private-extra.txt";
  writeFileSync(
    join(selected, "rocm/bundle.json"),
    JSON.stringify(localBundle),
  );
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "traversal"),
    ),
  );
  localBundle.models.UVDoc.source = "model.tar";
  const localBytes = "synthetic local archive transport";
  localBundle.models.UVDoc.sha256 = hash(localBytes);
  localBundle.models.UVDoc.size_bytes = Buffer.byteLength(localBytes);
  writeFileSync(join(selected, "rocm/model.tar"), localBytes);
  writeFileSync(
    join(selected, "rocm/bundle.json"),
    JSON.stringify(localBundle),
  );
  assert.throws(
    () =>
      stageInputs(
        request,
        { actor: "maintainer", role: "maintain" },
        sourceRoot,
        join(directory, "combined-local-artifact"),
      ),
    /HTTPS.*model|model.*HTTPS/i,
  );
  const standaloneRequest = { ...request, variant: "ocr-rocm" };
  const standalone = join(
    sourceRoot,
    standaloneRequest.sha,
    standaloneRequest.version,
    standaloneRequest.variant,
  );
  const { postgres, ...standalonePlan } = plan;
  seed(standalone, {
    "release-inputs.json": JSON.stringify(standalonePlan),
    "rocm/bundle.json": JSON.stringify(localBundle),
    "rocm/model.tar": localBytes,
  });
  const localOutput = join(directory, "standalone-local-artifact");
  stageInputs(
    standaloneRequest,
    { actor: "maintainer", role: "maintain" },
    sourceRoot,
    localOutput,
  );
  verifyStagedInputs(localOutput, standaloneRequest, {
    actor: "maintainer",
    run: "123",
  });
  assert.equal(
    readFileSync(join(localOutput, "rocm/model.tar"), "utf8"),
    localBytes,
  );
  writeFileSync(join(standalone, "rocm/model.tar"), "changed archive");
  assert.throws(() =>
    stageInputs(
      standaloneRequest,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "wrong-archive"),
    ),
  );
  writeFileSync(join(selected, "rocm/bundle.json"), bytes["rocm/bundle.json"]);
  assert.throws(() =>
    verifyStagedInputs(
      output,
      { ...request, version: "1.2.4" },
      { actor: "maintainer", run: "123" },
    ),
  );
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "foreign", run: "123" }),
  );
  writeFileSync(join(output, "release-inputs.json"), "changed");
  assert.throws(() =>
    verifyStagedInputs(output, request, { actor: "maintainer", run: "123" }),
  );
  assert.throws(() =>
    stageInputs(
      { ...request, variant: "../escape" },
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "bad"),
    ),
  );
  const { symlinkSync, linkSync } = await import("node:fs");
  rmSync(join(selected, "release-inputs.json"));
  writeFileSync(join(directory, "outside"), "private");
  linkSync(join(directory, "outside"), join(selected, "release-inputs.json"));
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "hardlink"),
    ),
  );
  rmSync(join(selected, "release-inputs.json"));
  symlinkSync(
    join(directory, "outside"),
    join(selected, "release-inputs.json"),
  );
  writeFileSync(join(directory, "outside"), "private");
  assert.throws(() =>
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      sourceRoot,
      join(directory, "symlink"),
    ),
  );
  const producer = readFileSync(
      ".github/workflows/distribution-inputs.yml",
      "utf8",
    ),
    consumer = readFileSync(
      ".github/workflows/distribution-release.yml",
      "utf8",
    );
  assert.match(producer, /workflow_dispatch:/);
  assert.match(producer, /actions\/upload-artifact@/);
  assert.match(producer, /name: distribution-inputs/);
  assert.match(consumer, /--check-run/);
  assert.match(consumer, /--verify/);
  assert.match(consumer, /DISTRIBUTION_INPUTS_ACTOR/);
});

const externalFixture = fixture;
const externalVariants = [
  "app",
  "ocr-cuda",
  "ocr-rocm",
  "app-ocr-cuda",
  "app-ocr-rocm",
];
function resolveExternal(t, variant, root, project = "vb-upgrade-stable") {
  const { manifest, bytes } = externalFixture();
  seed(root, bytes);
  writeBundle(manifest, variant, root + "/bundle", root);
  const directory = root + "/bundle";
  const syntheticLiteralKey =
    'synthetic$HOME${NOT_EXPANDED}#hash "quotes" \\literal';
  const environment =
    `LLM__PROVIDER=openai\nLLM__HOST=http://host.docker.internal:1234/v1\nLLM__MODEL_NAME=operator-model\nLLM__API_KEY=${syntheticLiteralKey}\n` +
    (variant.startsWith("app-ocr")
      ? "OCR__ENDPOINT=http://external-ocr:8082\n"
      : "");
  writeFileSync(join(directory, "runtime.env"), environment, { mode: 0o600 });
  const env = { ...process.env, COMPOSE_PROJECT_NAME: project };
  const command = ["compose", "-f", join(directory, "compose.yaml"), "config"];
  const quiet = spawnSync("docker", [...command, "--quiet"], {
    encoding: "utf8",
    env,
  });
  assert.equal(quiet.status, 0, "quiet Compose resolution failed");
  assert.equal(quiet.stdout, "");
  const result = spawnSync("docker", [...command, "--format", "json"], {
    encoding: "utf8",
    env,
  });
  assert.equal(result.status, 0, "Compose JSON resolution failed");
  return {
    effective: JSON.parse(result.stdout),
    syntheticLiteralKey,
    directory,
  };
}
test("external_only_variants_preserve_graph", () => {
  const { manifest } = externalFixture();
  for (const variant of externalVariants) {
    const rendered = renderBundle(manifest, variant);
    const compose = JSON.parse(rendered.files["compose.yaml"]);
    assert.equal(
      compose.name,
      "${COMPOSE_PROJECT_NAME:?Set a stable project name and reuse it for upgrades}",
    );
    assert.equal(compose.services.ollama, undefined);
    assert.equal(compose.services["model-init"], undefined);
    assert.equal(compose.volumes["ollama-data"], undefined);
    assert.doesNotMatch(rendered.files["release.json"], /"llm"|"ollama"/);
    assert.equal(JSON.parse(rendered.files["release.json"]).schema_version, 2);
    assert.equal(
      !!compose.services.app,
      variant === "app" || variant.startsWith("app-ocr"),
    );
    assert.equal(!!compose.services.ocr, variant !== "app");
    assert.match(rendered.files["runtime.env.example"], /LLM__MODEL_NAME/);
  }
});
test("canonical_runtime_env_reaches_init_and_app", (t) => {
  for (const variant of externalVariants) {
    const { effective, syntheticLiteralKey } = resolveExternal(
      t,
      variant,
      temp(t),
    );
    if (effective.services.app) {
      for (const name of ["app", "config-init"]) {
        // Compose serializes literal dollars as $$ for its round-trip config output.
        const actual = effective.services[
          name
        ].environment.LLM__API_KEY.replaceAll("$$", "$");
        assert.ok(actual === syntheticLiteralKey, "literal API key changed");
        assert.equal(
          effective.services[name].environment.LLM__MODEL_NAME,
          "operator-model",
        );
        assert.ok(
          effective.services[name].extra_hosts.includes(
            "host.docker.internal=host-gateway",
          ),
        );
      }
      assert.ok(
        !effective.services["config-init"].command.includes("--provider"),
      );
      if (variant.startsWith("app-ocr"))
        assert.equal(
          effective.services.app.environment.OCR__ENDPOINT,
          "http://external-ocr:8082",
        );
    }
  }
});
test("uncommenting_optional_runtime_assignments_reaches_init_and_app", (t) => {
  const root = temp(t);
  const { manifest, bytes } = externalFixture();
  seed(root, bytes);
  const directory = join(root, "bundle");
  writeBundle(manifest, "app", directory, root);
  const environment = readFileSync(
    join(directory, "runtime.env.example"),
    "utf8",
  ).replace(/^# .*LLM__(?:API_KEY|TEMPERATURE|MODEL_LIFECYCLE)=.*$/gm, (line) =>
    line.slice(2),
  );
  writeFileSync(join(directory, "runtime.env"), environment, { mode: 0o600 });
  const result = spawnSync(
    "docker",
    [
      "compose",
      "-f",
      join(directory, "compose.yaml"),
      "config",
      "--format",
      "json",
    ],
    {
      encoding: "utf8",
      env: { ...process.env, COMPOSE_PROJECT_NAME: "vb-optional-env-example" },
    },
  );
  assert.equal(
    result.status,
    0,
    "Compose could not resolve the uncommented example assignments",
  );
  const { services } = JSON.parse(result.stdout);
  for (const name of ["app", "config-init"]) {
    assert.ok(services[name].environment.LLM__API_KEY === "literal-key");
    assert.equal(services[name].environment.LLM__TEMPERATURE, "0");
    assert.equal(services[name].environment.LLM__MODEL_LIFECYCLE, "llamacpp");
  }
});
test("missing_optional_ocr_does_not_override_yaml", (t) => {
  const { effective } = resolveExternal(t, "app", temp(t));
  assert.equal(effective.services.app.environment.OCR__ENDPOINT, undefined);
});
test("v1_combined_inputs_rejected", async () => {
  const { manifest } = externalFixture();
  manifest.schema_version = 1;
  assert.throws(() => validateArtifacts(manifest));
  assert.throws(() =>
    renderBundle(externalFixture().manifest, "combined-rocm"),
  );
  const { validateReleaseInputs } = await import("./distribution-release.mjs");
  assert.throws(() =>
    validateReleaseInputs({ schema_version: 1 }, { variant: "app" }),
  );
});
test("bundle_rename_keeps_explicit_project_volumes", (t) => {
  const root = temp(t);
  const original = resolveExternal(t, "app-ocr-rocm", root);
  const first = original.effective;
  const renamed = join(root, "different-bundle-name");
  renameSync(original.directory, renamed);
  const result = spawnSync(
    "docker",
    [
      "compose",
      "-f",
      join(renamed, "compose.yaml"),
      "config",
      "--format",
      "json",
    ],
    {
      encoding: "utf8",
      env: { ...process.env, COMPOSE_PROJECT_NAME: "vb-upgrade-stable" },
    },
  );
  assert.equal(result.status, 0, "renamed bundle failed Compose resolution");
  const second = JSON.parse(result.stdout);
  assert.deepEqual(
    Object.values(second.volumes).map((v) => v.name),
    Object.values(first.volumes).map((v) => v.name),
  );
  assert.equal(first.name, "vb-upgrade-stable");
  assert.equal(
    first.volumes["app-config"].name,
    "vb-upgrade-stable_app-config",
  );
});

test("raw Compose minimum version and runtime privacy fail before config resolution", (t) => {
  const directory = temp(t);
  const bin = join(directory, "bin");
  mkdirSync(bin);
  const script = renderBundle(fixture().manifest, "app").files[
    "check-compose.sh"
  ];
  writeFileSync(join(directory, "check-compose.sh"), script);
  writeFileSync(join(directory, "runtime.env"), "", { mode: 0o600 });
  writeFileSync(
    join(bin, "docker"),
    '#!/bin/sh\nif [ "$2" = "version" ]; then echo "$TEST_COMPOSE_VERSION"; else touch "$TEST_CONFIG_CALLED"; fi\n',
    { mode: 0o700 },
  );
  const env = {
    ...process.env,
    PATH: `${bin}:${process.env.PATH}`,
    TEST_CONFIG_CALLED: join(directory, "called"),
  };
  const old = spawnSync("sh", [join(directory, "check-compose.sh")], {
    encoding: "utf8",
    env: { ...env, TEST_COMPOSE_VERSION: "2.29.7" },
  });
  assert.notEqual(old.status, 0);
  assert.match(old.stderr, /Compose >=2.30/);
  assert.ok(!readdirSync(directory).includes("called"));
  const supported = spawnSync("sh", [join(directory, "check-compose.sh")], {
    encoding: "utf8",
    env: { ...env, TEST_COMPOSE_VERSION: "v2.30.0" },
  });
  assert.equal(supported.status, 0, supported.stderr);
  assert.ok(readdirSync(directory).includes("called"));
  rmSync(join(directory, "called"));
  chmodSync(join(directory, "runtime.env"), 0o644);
  const unsafe = spawnSync("sh", [join(directory, "check-compose.sh")], {
    encoding: "utf8",
    env: { ...env, TEST_COMPOSE_VERSION: "2.40.3" },
  });
  assert.notEqual(unsafe.status, 0);
  assert.match(unsafe.stderr, /mode 0600/);
  assert.ok(!readdirSync(directory).includes("called"));
});

test("v2 producers stage and verify every external-only variant while native bundles retain v1", async (t) => {
  const { stageInputs, verifyStagedInputs } =
    await import("./distribution-inputs.mjs");
  for (const variant of externalVariants) {
    const root = temp(t);
    const { manifest, bytes } = fixture();
    const request = {
      repository: manifest.source.repository,
      sha: manifest.source.commit,
      version: manifest.version,
      variant,
      run: "123",
    };
    const accelerator = variant === "app" ? null : variant.split("-").at(-1);
    const app = variant === "app" || variant.startsWith("app-ocr-");
    const plan = {
      schema_version: 2,
      source: manifest.source,
      version: manifest.version,
      platform: manifest.platform,
      ...(app ? { postgres: manifest.images.postgres.reference } : {}),
      ...(accelerator === "cuda" ? { cuda_channel: "cu129" } : {}),
      ...(accelerator === "rocm"
        ? { rocm: { targets: ["gfx1030"], render_node: "/dev/dri/renderD128" } }
        : {}),
    };
    const selected = join(root, request.sha, request.version, variant);
    mkdirSync(selected, { recursive: true, mode: 0o700 });
    seed(selected, {
      "release-inputs.json": JSON.stringify(plan),
      ...(accelerator
        ? {
            [`${accelerator}/bundle.json`]: bytes[`${accelerator}/bundle.json`],
          }
        : {}),
      "llm/LICENSE": "retired delivery must never be staged",
    });
    const output = join(root, "artifact");
    stageInputs(
      request,
      { actor: "maintainer", role: "maintain" },
      root,
      output,
    );
    verifyStagedInputs(output, request, { actor: "maintainer", run: "123" });
    assert.ok(!readdirSync(output).includes("llm"));
    const receiptPath = join(output, "input-receipt.json");
    const receipt = JSON.parse(readFileSync(receiptPath));
    assert.equal(receipt.schema_version, 2);
    if (accelerator)
      assert.equal(
        JSON.parse(readFileSync(join(output, accelerator, "bundle.json")))
          .schema_version,
        1,
      );
    receipt.schema_version = 1;
    writeFileSync(receiptPath, JSON.stringify(receipt));
    assert.throws(() =>
      verifyStagedInputs(output, request, { actor: "maintainer", run: "123" }),
    );
    plan.schema_version = 1;
    writeFileSync(join(selected, "release-inputs.json"), JSON.stringify(plan));
    assert.throws(() =>
      stageInputs(
        request,
        { actor: "maintainer", role: "maintain" },
        root,
        join(root, "old-artifact"),
      ),
    );
  }
});

test("app source context excludes retired runtime while native source retains its supervisor", () => {
  const appIgnore = readFileSync(".dockerignore", "utf8");
  assert.doesNotMatch(
    appIgnore,
    /!distribution\/ollama|!paddleocr-service\/ocr_service/,
  );
  const nativeDockerfile = readFileSync(
    "paddleocr-service/Dockerfile.cuda",
    "utf8",
  );
  assert.match(nativeDockerfile, /ocr_service/);
});

test("qualification scratch uses only verified daemon-owned volumes and preserves primary errors", async () => {
  const { withOwnedVolumes } = await import("./distribution-release.mjs");
  const calls = [],
    labels = { "vision-bill.release.nonce": "test-nonce" };
  const driver = {
    create: (kind, expected) => {
      assert.deepEqual(expected, labels);
      const name = hash(kind);
      calls.push(["create", name]);
      return name;
    },
    inspect: (name) => ({ Name: name, Labels: labels }),
    remove: (name) => {
      calls.push(["remove", name]);
    },
  };
  assert.equal(
    withOwnedVolumes(driver, labels, ({ state, coord }) => {
      assert.equal(state, hash("state"));
      assert.equal(coord, hash("coord"));
      calls.push(["container-removed"]);
      return "root-wrote-private-state";
    }),
    "root-wrote-private-state",
  );
  assert.deepEqual(calls.slice(-3), [
    ["container-removed"],
    ["remove", hash("coord")],
    ["remove", hash("state")],
  ]);
  const primary = Error("qualification failed");
  assert.throws(
    () =>
      withOwnedVolumes(
        {
          ...driver,
          remove(name) {
            throw Error(`busy ${name}`);
          },
        },
        labels,
        () => {
          throw primary;
        },
      ),
    (error) =>
      error === primary &&
      error.cleanupFailures.length === 2 &&
      error.cleanupFailures[0].resource === hash("coord"),
  );
  const removed = [];
  assert.throws(
    () =>
      withOwnedVolumes(
        {
          ...driver,
          create: (kind) => {
            if (kind === "coord") throw Error("second create failed");
            return hash(kind);
          },
          remove: (name) => removed.push(name),
        },
        labels,
        () => {},
      ),
    /second create failed/,
  );
  assert.deepEqual(removed, [hash("state")]);
  assert.throws(
    () =>
      withOwnedVolumes(
        {
          ...driver,
          inspect: (name) => ({ Name: name, Labels: {} }),
          remove: (name) => {
            throw Error("must not remove foreign volume");
          },
        },
        labels,
        () => {},
      ),
    /cleanup/,
  );
});

test("native release dispatch requires explicit operator maintenance attestation", () => {
  const workflow = readFileSync(
    new URL("../.github/workflows/distribution-release.yml", import.meta.url),
    "utf8",
  );
  assert.match(
    workflow,
    /maintenance_intent:[\s\S]*options: \[not-applicable, external-llm-quiesced, separate-ocr-device\]/,
  );
  assert.match(
    workflow,
    /DISTRIBUTION_MAINTENANCE_INTENT: \$\{\{ inputs.maintenance_intent \}\}/,
  );
});
