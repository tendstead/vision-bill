#!/usr/bin/env node
// Maintainer input staging only; never downloads, builds, qualifies or publishes.
import {
  copyFileSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  writeFileSync,
  existsSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  releaseRequest,
  validateReleaseInputs,
} from "./distribution-release.mjs";
import {
  activeModelArchives,
  fileIdentity,
  isHttpsAssetSource,
} from "./distribution.mjs";
export const PRODUCER_WORKFLOW = ".github/workflows/distribution-inputs.yml";
const STAGING_ROOT = "/var/lib/vision-bill/distribution-inputs";
const SYSTEM_OWNER = lstatSync("/").uid;
const check = (ok, message) => {
  if (!ok) throw Error(message);
};
const read = (path) => JSON.parse(readFileSync(path));
const equal = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const actorName = (value) =>
  typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9-]{0,38}$/.test(value);
const safe = (value) => {
  check(
    typeof value === "string" &&
      /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(value) &&
      value.split("/").every((part) => part && !part.startsWith(".")),
    "Unsafe approved input path",
  );
  return value;
};
function selection(request) {
  check(
    /^[a-f0-9]{40}$/.test(request.sha) &&
      /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/.test(
        request.version,
      ) &&
      ["app", "ocr-cuda", "ocr-rocm", "app-ocr-cuda", "app-ocr-rocm"].includes(
        request.variant,
      ),
    "Invalid exact input selection",
  );
  return join(request.sha, request.version, request.variant);
}
function trusted(path, file = false) {
  const absolute = resolve(path);
  check(realpathSync(absolute) === absolute, "Symlink approved input path");
  let cursor = absolute;
  while (true) {
    const stat = lstatSync(cursor);
    check(
      !stat.isSymbolicLink() &&
        [SYSTEM_OWNER, process.getuid()].includes(stat.uid),
      "Untrusted input ownership",
    );
    check(
      !(stat.mode & 0o022) ||
        (stat.isDirectory() &&
          stat.uid === SYSTEM_OWNER &&
          !!(stat.mode & 0o1000)),
      "Writable untrusted input path",
    );
    if (cursor === absolute && file)
      check(
        stat.isFile() && stat.nlink === 1,
        "Input must be a single-link regular file",
      );
    else check(stat.isDirectory(), "Input ancestor must be a directory");
    if (cursor === "/") break;
    cursor = dirname(cursor);
  }
  return absolute;
}
function approvedFiles(directory, request) {
  selection(request);
  validateReleaseInputs(
    read(trusted(join(directory, "release-inputs.json"), true)),
    request,
  );
  const appOcr = request.variant.startsWith("app-ocr"),
    accelerator =
      request.variant === "app" ? null : request.variant.split("-").at(-1);
  const files = new Map([["release-inputs.json", null]]);
  if (accelerator) {
    const name = `${accelerator}/bundle.json`,
      bundle = read(trusted(join(directory, name), true));
    check(
      bundle.schema_version === 1 && bundle.accelerator === accelerator,
      "Incomplete selected native input bundle",
    );
    const activeModels = activeModelArchives(bundle.models);
    files.set(name, null);
    for (const [name, archive] of Object.entries({
      ...bundle.models,
      fixtures: bundle.fixtures,
    })) {
      check(
        archive &&
          /^[a-f0-9]{64}$/.test(archive.sha256) &&
          Number.isSafeInteger(archive.size_bytes) &&
          archive.size_bytes > 0 &&
          archive.license?.redistribution_allowed === true,
        "Approved archive identity/license missing",
      );
      if (!isHttpsAssetSource(archive.source)) {
        check(
          !appOcr || name === "fixtures",
          "Dedicated OCR model assets require HTTPS",
        );
        const path = safe(archive.source);
        if (name !== "fixtures" && !Object.hasOwn(activeModels, name)) continue;
        files.set(`${accelerator}/${path}`, {
          sha256: archive.sha256,
          size_bytes: archive.size_bytes,
        });
      }
    }
  }
  return files;
}
export function validateInputRun(run, request, workflow, permission) {
  check(
    Number.isSafeInteger(workflow.id) &&
      workflow.path === PRODUCER_WORKFLOW &&
      run.workflow_id === workflow.id &&
      run.path === PRODUCER_WORKFLOW,
    "Inputs were not produced by the approved workflow",
  );
  check(
    run.head_sha === request.sha &&
      run.head_repository?.full_name === request.repository &&
      run.event === "workflow_dispatch" &&
      run.conclusion === "success",
    "Inputs require successful manual exact-source producer run",
  );
  const actor = run.actor?.login;
  check(
    actorName(actor) &&
      run.triggering_actor?.login === actor &&
      permission.user?.login === actor &&
      ["admin", "maintain"].includes(
        permission.role_name ?? permission.permission,
      ),
    "Producer actor is not the bound maintainer",
  );
  return actor;
}
export function stageInputs(request, maintainer, root, output) {
  check(
    actorName(maintainer.actor) &&
      ["admin", "maintain"].includes(maintainer.role) &&
      /^[0-9]+$/.test(request.run),
    "Maintainer producer identity required",
  );
  const directory = trusted(join(trusted(root), selection(request))),
    files = approvedFiles(directory, request);
  const records = {};
  for (const [name, expected] of files) {
    const identity = fileIdentity(trusted(join(directory, safe(name)), true));
    check(
      !expected || equal(identity, expected),
      "Approved local archive identity mismatch",
    );
    records[name] = identity;
  }
  check(!existsSync(output), "Input artifact output already exists");
  mkdirSync(output, { recursive: true, mode: 0o700 });
  for (const name of files.keys()) {
    const target = join(output, name);
    mkdirSync(dirname(target), { recursive: true, mode: 0o700 });
    copyFileSync(join(directory, name), target);
    check(
      equal(fileIdentity(target), records[name]),
      "Input changed during staging",
    );
  }
  writeFileSync(
    join(output, "input-receipt.json"),
    JSON.stringify(
      {
        schema_version: 2,
        source: { repository: request.repository, commit: request.sha },
        version: request.version,
        variant: request.variant,
        producer: {
          workflow: PRODUCER_WORKFLOW,
          actor: maintainer.actor,
          run: request.run,
        },
        assets: records,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx", mode: 0o600 },
  );
}
function inventory(directory, prefix = "") {
  const names = [];
  for (const name of readdirSync(directory)) {
    const relative = safe(prefix + name),
      path = join(directory, name),
      stat = lstatSync(path);
    trusted(path, stat.isFile());
    if (stat.isDirectory()) names.push(...inventory(path, relative + "/"));
    else names.push(relative);
  }
  return names.sort();
}
export function verifyStagedInputs(directory, request, producer) {
  const receipt = read(trusted(join(directory, "input-receipt.json"), true));
  check(
    receipt.schema_version === 2 &&
      receipt.source?.repository === request.repository &&
      receipt.source?.commit === request.sha &&
      receipt.version === request.version &&
      receipt.variant === request.variant &&
      receipt.producer?.workflow === PRODUCER_WORKFLOW &&
      receipt.producer.actor === producer.actor &&
      receipt.producer.run === producer.run,
    "Input receipt source/version/producer identity mismatch",
  );
  const files = approvedFiles(directory, request);
  check(
    equal(Object.keys(receipt.assets ?? {}).sort(), [...files.keys()].sort()) &&
      equal(
        inventory(directory),
        [...files.keys(), "input-receipt.json"].sort(),
      ),
    "Incomplete or unexpected producer asset inventory",
  );
  for (const [name, expected] of files) {
    const actual = fileIdentity(trusted(join(directory, name), true));
    check(
      equal(actual, receipt.assets[name]) &&
        (!expected || equal(actual, expected)),
      "Producer asset hash identity mismatch",
    );
  }
}
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  try {
    const args = process.argv.slice(2),
      request = releaseRequest(process.env);
    if (args[0] === "--stage" && args.length === 1) {
      check(process.env.RUNNER_TEMP, "Runner temp missing");
      stageInputs(
        request,
        {
          actor: process.env.GITHUB_ACTOR,
          role: process.env.DISTRIBUTION_MAINTAINER_ROLE,
        },
        STAGING_ROOT,
        join(process.env.RUNNER_TEMP, "distribution-inputs"),
      );
    } else if (args[0] === "--check-run" && args.length === 4)
      process.stdout.write(
        validateInputRun(read(args[1]), request, read(args[2]), read(args[3])) +
          "\n",
      );
    else if (args[0] === "--verify" && args.length === 2)
      verifyStagedInputs(resolve(args[1]), request, {
        actor: process.env.DISTRIBUTION_INPUTS_ACTOR,
        run: process.env.DISTRIBUTION_INPUTS_RUN_ID,
      });
    else throw Error("Invalid producer command");
  } catch (error) {
    process.stderr.write(`Distribution inputs rejected: ${error.message}\n`);
    process.exitCode = 1;
  }
}
