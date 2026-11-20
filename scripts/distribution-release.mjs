#!/usr/bin/env node
// Maintainer-only driver. Importing it and running its tests never builds/publishes.
import { createHash, randomUUID } from "node:crypto";
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  readdirSync,
  lstatSync,
  mkdtempSync,
  rmSync,
  copyFileSync,
} from "node:fs";
import { join, resolve, dirname } from "node:path";
import { tmpdir } from "node:os";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import {
  validateArtifacts,
  cudaChannels,
  cudaLockDigestLabel,
  defaultCudaChannel,
  legacyCudaChannel,
  validateCudaLock,
  renderBundle,
  writeBundle,
  fileIdentity,
} from "./distribution.mjs";
import { githubApi, existingImage } from "./release.mjs";
const check = (ok, reason) => {
  if (!ok) throw Error(reason);
};
const hash = (value) => createHash("sha256").update(value).digest("hex");
const json = (value) => JSON.stringify(value, null, 2) + "\n";
const variants = [
  "app",
  "ocr-cuda",
  "ocr-rocm",
  "app-ocr-cuda",
  "app-ocr-rocm",
];
const immutable = (value) =>
  typeof value === "string" &&
  /^[a-z0-9][a-z0-9.:-]*\/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}$/.test(
    value,
  ) &&
  !/@sha256:([a-f0-9])\1{63}$/.test(value);
const run = (command, args, options = {}) =>
  execFileSync(command, args, { maxBuffer: 64 * 1024 * 1024, ...options });
const read = (path) => JSON.parse(readFileSync(path));
function save(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, json(value));
}
export function releaseRequest(env) {
  check(
    env.GITHUB_EVENT_NAME === "workflow_dispatch",
    "Only explicit workflow dispatch may release",
  );
  check(
    env.DISTRIBUTION_FORK_REPOSITORY &&
      env.GITHUB_REPOSITORY === env.DISTRIBUTION_FORK_REPOSITORY,
    "Repository is not the configured release fork",
  );
  check(
    /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(
      env.GITHUB_REPOSITORY,
    ),
    "Invalid repository",
  );
  check(
    /^[a-f0-9]{40}$/.test(env.GITHUB_SHA) &&
      env.DISTRIBUTION_SOURCE_SHA === env.GITHUB_SHA,
    "Dispatch must name its exact source SHA",
  );
  check(
    /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/.test(
      env.DISTRIBUTION_VERSION,
    ),
    "Exact version required",
  );
  check(
    variants.includes(env.DISTRIBUTION_VARIANT) &&
      /^[0-9]+$/.test(env.GITHUB_RUN_ID),
    "Invalid release request",
  );
  return {
    repository: env.GITHUB_REPOSITORY,
    sha: env.GITHUB_SHA,
    version: env.DISTRIBUTION_VERSION,
    variant: env.DISTRIBUTION_VARIANT,
    run: env.GITHUB_RUN_ID,
  };
}
export function cudaQualificationGpuArgs(deviceId = "0") {
  check(
    typeof deviceId === "string" &&
      /^(?:[0-9]+|GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i.test(
        deviceId,
      ),
    "Select one CUDA device by numeric index or GPU UUID",
  );
  return ["--gpus", `device=${deviceId}`];
}
export function validateReleaseInputs(plan, request) {
  check(variants.includes(request.variant), "Unknown distribution variant");
  const app =
    request.variant === "app" || request.variant.startsWith("app-ocr-");
  const allowed = [
    "schema_version",
    "source",
    "version",
    "platform",
    ...(app ? ["postgres"] : []),
    ...(request.variant.endsWith("cuda")
      ? ["cuda_channel", "cuda_device_id"]
      : []),
    ...(request.variant.endsWith("rocm") ? ["rocm"] : []),
  ];
  check(
    plan && Object.keys(plan).every((key) => allowed.includes(key)),
    "Unknown release input",
  );
  check(
    plan.schema_version === 2 &&
      plan.version === request.version &&
      plan.platform === "linux/amd64" &&
      plan.source?.repository === request.repository &&
      plan.source?.commit === request.sha &&
      Object.keys(plan.source).sort().join(",") === "commit,repository",
    "Release inputs must match exact dispatch source/version/platform",
  );
  if (app)
    check(
      immutable(plan.postgres),
      "Immutable single-platform PostgreSQL manifest required",
    );
  if (Object.hasOwn(plan, "cuda_channel"))
    check(cudaChannels.includes(plan.cuda_channel), "Unsupported CUDA channel");
  if (Object.hasOwn(plan, "cuda_device_id"))
    cudaQualificationGpuArgs(plan.cuda_device_id);
  if (request.variant.endsWith("rocm")) {
    check(
      plan.rocm &&
        Object.keys(plan.rocm).every((key) =>
          ["targets", "render_node", "render_gid", "video_gid"].includes(key),
        ),
      "Explicit AMD build/device selection required",
    );
    check(
      Array.isArray(plan.rocm.targets) &&
        plan.rocm.targets.length > 0 &&
        plan.rocm.targets.every((target) => /^gfx[0-9a-f]+$/.test(target)) &&
        new Set(plan.rocm.targets).size === plan.rocm.targets.length,
      "Actual AMD target list required",
    );
    check(
      /^\/dev\/dri\/renderD[0-9]+$/.test(plan.rocm.render_node) &&
        [plan.rocm.render_gid, plan.rocm.video_gid]
          .filter((id) => id !== undefined)
          .every((id) => Number.isSafeInteger(id) && id >= 0),
      "Selected AMD render node/groups required",
    );
  }
  return plan;
}
export async function publishVerified(manifest, variant, operations) {
  const metadata = JSON.parse(
    renderBundle(manifest, variant).files["release.json"],
  );
  await operations.verify();
  for (const [name, image] of Object.entries(metadata.images))
    if (name !== "postgres") await operations.image(name, image);
  for (const [name, image] of Object.entries(metadata.images))
    await operations.pull(name, image);
  await operations.assets();
}
export async function publishImageIfNeeded(image, operations) {
  const existing = await operations.lookup();
  if (existing !== null) {
    check(
      image.reference.endsWith(`@${existing}`),
      "Published image differs from qualified immutable image",
    );
    return;
  }
  await operations.copy();
}
export function withOwnedContainer(driver, operation) {
  const id = driver.create();
  check(
    /^[a-f0-9]{64}$/.test(id),
    "Docker did not return an owned container ID",
  );
  let primary;
  try {
    return operation(id);
  } catch (error) {
    primary = error;
    throw error;
  } finally {
    try {
      driver.remove(id);
    } catch (error) {
      if (!primary) {
        const cleanup = Error(`Owned container cleanup failed: ${id}`);
        cleanup.cleanupFailures = [{ resource: id, error }];
        throw cleanup;
      }
      primary.cleanupFailures = [
        ...(primary.cleanupFailures ?? []),
        { resource: id, error },
      ];
    }
  }
}
export function dockerVolumeDriver(execute = run) {
  return {
    create(_role, labels) {
      return execute(
        "docker",
        [
          "volume",
          "create",
          ...Object.entries(labels).flatMap(([key, value]) => [
            "--label",
            `${key}=${value}`,
          ]),
        ],
        { encoding: "utf8" },
      ).trim();
    },
    inspect(name) {
      return JSON.parse(
        execute("docker", ["volume", "inspect", name], { encoding: "utf8" }),
      )[0];
    },
    remove(name) {
      execute("docker", ["volume", "rm", name], { stdio: "inherit" });
    },
  };
}
export function withOwnedVolumes(driver, labels, operation) {
  check(
    typeof labels?.["vision-bill.release.nonce"] === "string" &&
      labels["vision-bill.release.nonce"].length > 0,
    "Owned scratch requires an exact nonce label",
  );
  const owned = [],
    volumes = {};
  let primary;
  try {
    for (const role of ["state", "coord"]) {
      const name = driver.create(role, labels);
      check(
        /^[a-f0-9]{64}$/.test(name),
        "Docker did not return a daemon-generated volume name",
      );
      owned.push(name);
      volumes[role] = name;
    }
    return operation(volumes);
  } catch (error) {
    primary = error;
    throw error;
  } finally {
    const failures = [];
    for (const name of owned.reverse()) {
      try {
        const record = driver.inspect(name);
        check(
          record?.Name === name &&
            Object.entries(labels).every(
              ([key, value]) => record.Labels?.[key] === value,
            ),
          "Owned volume label differs; preserved resource",
        );
        driver.remove(name);
      } catch (error) {
        failures.push({ resource: name, error });
      }
    }
    if (failures.length) {
      if (primary)
        primary.cleanupFailures = [
          ...(primary.cleanupFailures ?? []),
          ...failures,
        ];
      else {
        const error = Error(
          `Owned volume cleanup failed: ${failures.map(({ resource }) => resource).join(", ")}`,
        );
        error.cleanupFailures = failures;
        throw error;
      }
    }
  }
}
function assetsIn(directory, prefix = "") {
  const result = {};
  for (const entry of readdirSync(directory)) {
    check(
      /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(entry),
      "Unsafe release input filename",
    );
    const path = join(directory, entry),
      relative = prefix + entry,
      stat = lstatSync(path);
    check(!stat.isSymbolicLink(), "Symlink release inputs forbidden");
    if (stat.isDirectory())
      Object.assign(result, assetsIn(path, relative + "/"));
    else {
      check(stat.isFile(), "Nonregular release input");
      result[relative] = fileIdentity(path);
    }
  }
  return result;
}
function copyAssets(source, target) {
  const assets = assetsIn(source);
  for (const path of Object.keys(assets)) {
    mkdirSync(dirname(join(target, path)), { recursive: true });
    copyFileSync(join(source, path), join(target, path));
  }
}
function inspectArchive(path) {
  const bytes = run("skopeo", ["inspect", "--raw", `oci-archive:${path}`]),
    manifest = JSON.parse(bytes);
  check(
    manifest.config?.digest && !manifest.manifests,
    "A single linux/amd64 OCI image manifest is required",
  );
  return {
    digest: `sha256:${hash(bytes)}`,
    config_digest: manifest.config.digest,
  };
}
export function candidateBundleEvidence(tag, bundle, execute = run) {
  return JSON.parse(
    execute("docker", [
      "run",
      "--rm",
      "--network",
      "none",
      "--entrypoint",
      "python",
      tag,
      "-c",
      [
        "import json, sys",
        "from ocr_service.bootstrap import bind_approved_dependencies",
        "from ocr_service.provision import source_identities",
        "pins = bind_approved_dependencies(json.loads(sys.argv[1]), accelerator=sys.argv[2])",
        "print(json.dumps(dict(expected_packages=pins, source_files=source_identities())))",
      ].join("\n"),
      JSON.stringify(bundle.expected_packages),
      bundle.accelerator,
    ]),
  );
}
function build(request, inputs, output) {
  check(
    run("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim() ===
      request.sha,
    "Checkout differs from exact dispatch SHA",
  );
  run("git", ["diff", "--exit-code"], { stdio: "inherit" });
  const plan = validateReleaseInputs(
    read(join(inputs, "release-inputs.json")),
    request,
  );
  check(!lstatExists(output), "Candidate output already exists");
  mkdirSync(output, { recursive: true });
  const manifest = {
      schema_version: 2,
      version: request.version,
      source: plan.source,
      platform: plan.platform,
      images: {},
      assets: {},
      verification: { app_checks: "checks.json" },
    },
    records = {};
  const appOcr = request.variant.startsWith("app-ocr"),
    app = request.variant === "app" || appOcr;
  const accelerator =
    request.variant === "app" ? null : request.variant.split("-").at(-1);
  if (accelerator === "cuda") plan.cuda_channel ??= defaultCudaChannel;
  const ownedTags = [];
  function candidate(name, dockerfile, context, args = []) {
    const archive = join(output, `${name}.oci.tar`);
    run(
      "docker",
      [
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
        "--provenance=false",
        "--sbom=false",
        "--label",
        `org.opencontainers.image.source=https://github.com/${request.repository}`,
        "--label",
        `org.opencontainers.image.revision=${request.sha}`,
        "--label",
        `org.opencontainers.image.version=v${request.version}-${request.variant}`,
        "--annotation",
        `index,manifest:org.opencontainers.image.source=https://github.com/${request.repository}`,
        "--annotation",
        `index,manifest:org.opencontainers.image.revision=${request.sha}`,
        "--annotation",
        `index,manifest:org.opencontainers.image.version=v${request.version}-${request.variant}`,
        "--output",
        `type=oci,dest=${archive}`,
        "--file",
        dockerfile,
        ...args,
        context,
      ],
      { stdio: "inherit" },
    );
    const identity = inspectArchive(archive),
      repository = `ghcr.io/${request.repository.toLowerCase()}-${name}`;
    manifest.images[name] = {
      reference: `${repository}@${identity.digest}`,
      config_digest: identity.config_digest,
    };
    const tag = `vision-bill-release-${request.run}-${randomUUID()}-${name}:candidate`;
    run("skopeo", ["copy", `oci-archive:${archive}`, `docker-daemon:${tag}`], {
      stdio: "inherit",
    });
    ownedTags.push(tag);
    check(
      run("docker", ["image", "inspect", "--format", "{{.Id}}", tag], {
        encoding: "utf8",
      }).trim() === identity.config_digest,
      "Loaded image differs from OCI config",
    );
    records[name] = { archive: `${name}.oci.tar`, tag };
    return tag;
  }
  try {
    if (app) {
      const tag = candidate("app", "Dockerfile", ".");
      run(
        "docker",
        [
          "run",
          "--rm",
          "--network",
          "none",
          "--entrypoint",
          "python",
          tag,
          "-c",
          "from pathlib import Path; import vision_bill.distribution.bootstrap; assert Path('/app/src/vision_bill/static/index.html').is_file()",
        ],
        { stdio: "inherit" },
      );
      const bytes = run("skopeo", [
          "inspect",
          "--raw",
          `docker://${plan.postgres}`,
        ]),
        postgres = JSON.parse(bytes);
      check(
        postgres.config?.digest &&
          plan.postgres.endsWith(`@sha256:${hash(bytes)}`),
        "PostgreSQL must identify the exact amd64 image manifest",
      );
      manifest.images.postgres = {
        reference: plan.postgres,
        config_digest: postgres.config.digest,
      };
    }
    if (accelerator) {
      const nativeDirectory = join(output, accelerator);
      copyAssets(join(inputs, accelerator), nativeDirectory);
      const args = [];
      let cudaLockDigest;
      if (accelerator === "cuda") {
        const channel = plan.cuda_channel;
        const resolver = [
          "run",
          "--no-project",
          "--with",
          "packaging==26.2",
          "python",
          "paddleocr-service/build/resolve_cuda.py",
          "--channel",
          channel,
        ];
        run("uv", resolver, { stdio: "inherit" });
        const wheelContext = run(
          "uv",
          [
            ...resolver,
            "--check",
            "paddleocr-service/requirements.cuda.lock",
            "--wheel-context",
          ],
          { encoding: "utf8" },
        ).trim();
        const attestation = readFileSync(
          "paddleocr-service/requirements.cuda.attestation.json",
        );
        const lock = readFileSync("paddleocr-service/requirements.cuda.lock");
        const metadata = JSON.parse(attestation);
        validateCudaLock(metadata, channel);
        cudaLockDigest = hash(lock);
        check(
          cudaLockDigest === metadata.lock_sha256,
          "Resolved CUDA lock differs from attestation",
        );
        writeFileSync(
          join(nativeDirectory, "requirements.cuda.lock"),
          lock,
        );
        writeFileSync(
          join(nativeDirectory, "lock-attestation.json"),
          attestation,
        );
        args.push(
          "--build-arg",
          `OCR_CUDA_CHANNEL=${channel}`,
          "--build-context",
          `cuda-wheel=${wheelContext}`,
          "--label",
          cudaLockDigestLabel + "=" + cudaLockDigest,
        );
      } else
        args.push(
          "--build-arg",
          `AMDGPU_TARGETS=${plan.rocm.targets.join(";")}`,
        );
      const name = `ocr-${accelerator}`,
        tag = candidate(
          name,
          `paddleocr-service/Dockerfile.${accelerator}`,
          "paddleocr-service",
          args,
        );
      if (accelerator === "cuda") {
        const installedLock = "/opt/locks/requirements.cuda.lock";
        const measured = run(
          "docker",
          [
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "sha256sum",
            tag,
            installedLock,
          ],
          { encoding: "utf8" },
        ).trim();
        check(
          measured === cudaLockDigest + "  " + installedLock,
          "Candidate image CUDA lock differs from attested executable lock",
        );
      }
      // Export the exact Docker image-save config, then compare to actual inspect ID.
      const saved = join(output, "native-image-save.tar");
      run("docker", ["image", "save", "--output", saved, tag], {
        stdio: "inherit",
      });
      const savedManifest = JSON.parse(
        run("tar", ["-xOf", saved, "manifest.json"]),
      );
      check(
        savedManifest.length === 1 &&
          /^[a-zA-Z0-9._/-]+$/.test(savedManifest[0].Config) &&
          !savedManifest[0].Config.split("/").includes(".."),
        "Invalid image-save config record",
      );
      const config = run("tar", ["-xOf", saved, savedManifest[0].Config]);
      rmSync(saved);
      check(
        `sha256:${hash(config)}` === manifest.images[name].config_digest,
        "Raw Docker config differs from actual image ID",
      );
      writeFileSync(join(nativeDirectory, "image-config.json"), config);
      const bundle = read(join(nativeDirectory, "bundle.json"));
      check(
        bundle.accelerator === accelerator && bundle.gpu_index === 0,
        "Prepared single-device bundle must use gpu_index zero",
      );
      bundle.image_identity = manifest.images[name].reference;
      bundle.image_config_sha256 = hash(config);
      const { expected_packages, source_files } = candidateBundleEvidence(
        tag,
        bundle,
      );
      bundle.expected_packages = expected_packages;
      bundle.source_files = source_files;
      bundle.profile_sha256 = bundle.source_files["profile.json"];
      save(join(nativeDirectory, "bundle.json"), bundle);
      manifest.native = {
        [accelerator]: {
          bundle: `${accelerator}/bundle.json`,
          image_config: `${accelerator}/image-config.json`,
          qualification: `${accelerator}/qualification.json`,
          seal: `${accelerator}/qualification-manifest.json`,
          ...(accelerator === "cuda"
            ? {
                cuda_lock: `${accelerator}/lock-attestation.json`,
                cuda_channel: plan.cuda_channel,
              }
            : {}),
        },
      };
    }
    save(join(output, "manifest.json"), manifest);
    save(join(output, "images.json"), records);
    save(join(output, "plan.json"), plan);
  } finally {
    // Clean only tags created by this unique run; archives remain for publication.
    for (const tag of ownedTags)
      run("docker", ["image", "rm", tag], { stdio: "inherit" });
  }
}
function lstatExists(path) {
  try {
    lstatSync(path);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}
function verifyCandidateCudaChannel(plan, manifest, accelerator) {
  if (accelerator === "cuda")
    check(
      (manifest.native?.cuda?.cuda_channel ?? legacyCudaChannel) ===
        (plan.cuda_channel ?? legacyCudaChannel),
      "Candidate CUDA channel differs from approved producer plan",
    );
}
function qualify(request, output, env) {
  check(
    env.DISTRIBUTION_CI_RESULT === "success",
    "Exact-source CI gate must pass first",
  );
  const manifest = read(join(output, "manifest.json")),
    records = read(join(output, "images.json")),
    plan = validateReleaseInputs(read(join(output, "plan.json")), request);
  const accelerator =
    request.variant === "app" ? null : request.variant.split("-").at(-1);
  verifyCandidateCudaChannel(plan, manifest, accelerator);
  if (accelerator) {
    check(
      ["external-llm-quiesced", "separate-ocr-device"].includes(
        env.DISTRIBUTION_MAINTENANCE_INTENT,
      ),
      "Native qualification requires explicit operator maintenance intent",
    );
    const name = `ocr-${accelerator}`,
      record = records[name],
      archive = join(output, record.archive);
    check(
      inspectArchive(archive).config_digest ===
        manifest.images[name].config_digest,
      "Candidate config changed",
    );
    run(
      "skopeo",
      ["copy", `oci-archive:${archive}`, `docker-daemon:${record.tag}`],
      { stdio: "inherit" },
    );
    const gpu =
      accelerator === "cuda"
        ? cudaQualificationGpuArgs(plan.cuda_device_id)
        : [
            "--device",
            "/dev/kfd:/dev/kfd",
            "--device",
            plan.rocm.render_node,
            ...(plan.rocm.render_gid === undefined
              ? []
              : ["--group-add", String(plan.rocm.render_gid)]),
            ...(plan.rocm.video_gid === undefined
              ? []
              : ["--group-add", String(plan.rocm.video_gid)]),
          ];
    let primary;
    try {
      withOwnedVolumes(
        dockerVolumeDriver(),
        { "vision-bill.release.nonce": randomUUID() },
        (volumes) =>
          withOwnedContainer(
            {
              create: () =>
                run(
                  "docker",
                  [
                    "create",
                    "--init=false",
                    ...gpu,
                    "--mount",
                    `type=bind,src=${resolve(output, accelerator)},dst=/release,readonly`,
                    "--mount",
                    `type=volume,src=${volumes.state},dst=/var/lib/ocr`,
                    "--mount",
                    `type=volume,src=${volumes.coord},dst=/coord`,
                    "--entrypoint",
                    "python",
                    record.tag,
                    "-m",
                    "ocr_service.initialize",
                    "maintain",
                    "--maintenance-intent",
                    env.DISTRIBUTION_MAINTENANCE_INTENT,
                    "--bundle",
                    "/release/bundle.json",
                    "--state-dir",
                    "/var/lib/ocr",
                    "--image-config",
                    "/release/image-config.json",
                    "--image-id",
                    manifest.images[name].config_digest,
                  ],
                  { encoding: "utf8" },
                ).trim(),
              remove: (id) =>
                run("docker", ["rm", "-f", id], { stdio: "inherit" }),
            },
            (container) => {
              run("docker", ["start", "--attach", container], {
                stdio: "inherit",
              });
              // /selected is a symlink; dereference the selected immutable report explicitly.
              run(
                "docker",
                [
                  "cp",
                  "-L",
                  `${container}:/var/lib/ocr/selected/qualification.json`,
                  join(output, accelerator, "qualification.json"),
                ],
                { stdio: "inherit" },
              );
              run(
                "docker",
                [
                  "cp",
                  "-L",
                  `${container}:/var/lib/ocr/selected/manifest.json`,
                  join(output, accelerator, "qualification-manifest.json"),
                ],
                { stdio: "inherit" },
              );
              const report = read(
                join(output, accelerator, "qualification.json"),
              );
              if (accelerator === "rocm")
                check(
                  plan.rocm.targets.includes(report.device?.gfx_target),
                  "Actual AMD board was not among selected build targets",
                );
            },
          ),
      );
    } catch (error) {
      primary = error;
      throw error;
    } finally {
      try {
        run("docker", ["image", "rm", record.tag], { stdio: "inherit" });
      } catch (error) {
        if (!primary) throw error;
        primary.cleanupFailures = [
          ...(primary.cleanupFailures ?? []),
          { resource: record.tag, error },
        ];
      }
    }
  }
  save(join(output, "checks.json"), {
    passed: true,
    source_commit: request.sha,
    checks: ["backend", "frontend", "e2e-backend", "e2e-frontend"],
    workflow_run: `https://github.com/${request.repository}/actions/runs/${request.run}`,
  });
  // Hash only customer evidence/data; never buffer or scan multi-GB OCI transport.
  manifest.assets = {
    "checks.json": fileIdentity(join(output, "checks.json")),
    ...(accelerator
      ? assetsIn(join(output, accelerator), `${accelerator}/`)
      : {}),
  };
  save(join(output, "manifest.json"), manifest);
  const validated = join(output, "verified-bundle");
  writeBundle(manifest, request.variant, validated, output);
  const envForConfig = {
    ...env,
    COMPOSE_PROJECT_NAME: "vision-bill-release-validation",
    OCR_RENDER_GID: "109",
    OCR_VIDEO_GID: "44",
  };
  writeFileSync(join(validated, "runtime.env"), "", { mode: 0o600 });
  run("sh", [join(validated, "check-compose.sh")], {
    env: envForConfig,
    stdio: "inherit",
  });
}
async function publish(request, output, env) {
  check(
    env.DISTRIBUTION_CI_RESULT === "success",
    "Exact-source CI gate must pass first",
  );
  const manifest = read(join(output, "manifest.json")),
    records = read(join(output, "images.json")),
    plan = validateReleaseInputs(read(join(output, "plan.json")), request);
  verifyCandidateCudaChannel(plan, manifest, request.variant.split("-").at(-1));
  check(
    manifest.source.repository === request.repository &&
      manifest.source.commit === request.sha &&
      manifest.version === request.version,
    "Prepared artifacts differ from request",
  );
  const releaseTag = `distribution-v${request.version}-${request.variant}`;
  const archive = join(
    output,
    `vision-bill-${request.version}-${request.variant}.tar.gz`,
  );
  const assembled = join(output, "publication-bundle");
  const anonymousConfig = mkdtempSync(join(tmpdir(), "vb-distribution-pull-"));
  const anonymousAuth = join(anonymousConfig, "config.json");
  save(anonymousAuth, { auths: {} });
  check(
    env.GH_TOKEN && env.GITHUB_ACTOR,
    "Maintainer registry/API credentials required",
  );
  const api = githubApi(request.repository, env.GH_TOKEN);
  try {
    await publishVerified(manifest, request.variant, {
      async verify() {
        validateArtifacts(manifest);
        writeBundle(manifest, request.variant, assembled, output);
        // Never overwrite an existing release or image tag.
        check(
          !(await api(`/releases/tags/${releaseTag}`)),
          "Release already exists",
        );
        for (const [name, record] of Object.entries(records)) {
          check(
            record.archive === `${name}.oci.tar` && /^[a-z0-9-]+$/.test(name),
            "Unsafe candidate archive",
          );
          const actual = inspectArchive(join(output, record.archive)),
            expected = manifest.images[name];
          check(
            expected?.reference.startsWith(
              `ghcr.io/${request.repository.toLowerCase()}-${name}@`,
            ) &&
              expected?.reference.endsWith(`@${actual.digest}`) &&
              expected.config_digest === actual.config_digest,
            "Transported OCI identity changed",
          );
        }
      },
      async image(name, image) {
        const tag = `${image.reference.split("@")[0]}:v${request.version}-${request.variant}`;
        // Reuse the tested 404-only absence classifier. Authentication, rate-limit,
        // network failures and any existing manifest abort without overwriting.
        await publishImageIfNeeded(image, {
          lookup: () =>
            existingImage(
              {
                repository: request.repository,
                image: image.reference.split("@")[0],
              },
              {
                sha: request.sha,
                version: `v${request.version}-${request.variant}`,
              },
              env.GH_TOKEN,
              env.GITHUB_ACTOR,
            ),
          copy: async () => {
            run(
              "skopeo",
              [
                "copy",
                "--preserve-digests",
                `oci-archive:${join(output, records[name].archive)}`,
                `docker://${tag}`,
              ],
              { stdio: "inherit" },
            );
          },
        });
      },
      async pull(name, image) {
        const raw = run("skopeo", [
          "inspect",
          "--authfile",
          anonymousAuth,
          "--raw",
          `docker://${image.reference}`,
        ]);
        check(
          image.reference.endsWith(`@sha256:${hash(raw)}`),
          "Registry manifest digest changed",
        );
        run(
          "docker",
          [
            "--config",
            anonymousConfig,
            "pull",
            "--platform",
            "linux/amd64",
            image.reference,
          ],
          {
            stdio: "inherit",
          },
        );
        check(
          run(
            "docker",
            ["image", "inspect", "--format", "{{.Id}}", image.reference],
            { encoding: "utf8" },
          ).trim() === image.config_digest,
          `Pulled ${name} config differs from qualified image`,
        );
      },
      async assets() {
        run(
          "tar",
          [
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--use-compress-program=gzip -n",
            "-cf",
            archive,
            "-C",
            assembled,
            ".",
          ],
          { stdio: "inherit" },
        );
        writeFileSync(
          archive + ".sha256",
          `${fileIdentity(archive).sha256}  ${archive.split("/").at(-1)}\n`,
        );
        const notes = join(output, "release-notes.txt");
        writeFileSync(
          notes,
          `Prebuilt ${request.variant} installation for ${request.repository}@${request.sha}.\nQualification applies only to the device recorded in release metadata. Initial installation independently verifies and qualifies its actual device.\n`,
        );
        run(
          "gh",
          [
            "release",
            "create",
            releaseTag,
            archive,
            archive + ".sha256",
            "--repo",
            request.repository,
            "--target",
            request.sha,
            "--title",
            `Vision-Bill ${request.version} (${request.variant})`,
            "--notes-file",
            notes,
          ],
          { stdio: "inherit" },
        );
      },
    });
  } finally {
    rmSync(anonymousConfig, { recursive: true, force: true });
  }
}
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  try {
    const options = {},
      args = process.argv.slice(2);
    check(args.length % 2 === 0, "Expected named options");
    for (let i = 0; i < args.length; i += 2) {
      check(
        ["--phase", "--inputs", "--output"].includes(args[i]) &&
          !options[args[i]],
        "Invalid driver option",
      );
      options[args[i]] = args[i + 1];
    }
    const request = releaseRequest(process.env),
      output = resolve(options["--output"]);
    if (options["--phase"] === "build")
      build(request, resolve(options["--inputs"]), output);
    else if (options["--phase"] === "qualify")
      qualify(request, output, process.env);
    else if (options["--phase"] === "publish")
      await publish(request, output, process.env);
    else throw Error("Unknown phase");
  } catch (error) {
    process.stderr.write(`Distribution release stopped: ${error.message}\n`);
    for (const { resource, error: cleanup } of error.cleanupFailures ?? [])
      process.stderr.write(
        `Owned resource cleanup failed: ${resource}: ${cleanup.message}\n`,
      );
    process.exitCode = 1;
  }
}
