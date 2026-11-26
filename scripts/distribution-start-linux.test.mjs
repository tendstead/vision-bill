import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const token = "1234567890abcdef".repeat(3);
const project = `vision-bill-${token.slice(0, 16)}`;
const foreign = "f".repeat(48);
const claim = `${project}-vision-bill-launcher-owner`;
const launcher = new URL("../distribution/start-linux.sh", import.meta.url);
const hash = (data) => createHash("sha256").update(data).digest("hex");
const mockDocker = `#!/usr/bin/env node
const fs = require("node:fs");
const file = process.env.MOCK_STATE;
const state = JSON.parse(fs.readFileSync(file, "utf8"));
const args = process.argv.slice(2);
const [kind, command] = args;
const change = () => fs.writeFileSync(file, JSON.stringify(state));
const show = (text) => process.stdout.write(text + "\\n");
if (kind === "context") {
  show(command === "show" ? "default" : "unix:///var/run/docker.sock");
} else if (kind === "info") {
  show("linux|x86_64");
} else if (kind === "volume" && command === "ls") {
  for (const [name, item] of Object.entries(state.volumes)) {
    show([name, item.driver, item.project, item.role, item.owner].join("|"));
  }
} else if (kind === "volume" && command === "create") {
  const name = args.at(-1);
  const owner = args[args.indexOf("--label") + 1].split("=")[1];
  state.volumes[name] ??= { driver: "local", claim: owner };
  change();
  show(name);
} else if (kind === "volume" && command === "inspect") {
  const name = args.at(-1);
  const item = state.volumes[name];
  if (!item) process.exit(1);
  show([name, item.driver, item.claim].join("|"));
} else if (kind === "ps") {
  for (const [name, item] of Object.entries(state.containers)) {
    show([name, item.project, item.role, item.owner].join("|"));
  }
} else if (kind === "network" && command === "ls") {
  for (const [name, item] of Object.entries(state.networks)) {
    show([name, item.project, item.role, item.owner].join("|"));
  }
} else if (kind === "compose") {
  const owner = process.env.VISION_BILL_INSTALLATION_ID;
  const project = process.env.COMPOSE_PROJECT_NAME;
  if (args.includes("--quiet") && args.includes("config")) {
    if (!owner || !project) process.exit(1);
    state.checks++;
    change();
  } else if (args.includes("--volumes")) {
    show("app-config");
  } else if (args.includes("--services")) {
    show("app");
    show("ocr");
  } else if (args.includes("up")) {
    if (!args.includes("--wait") || !args.includes("--no-build")) process.exit(1);
    state.starts++;
    state.volumes[project + "_app-config"] = { driver: "local", project, role: "app-config", owner };
    state.networks[project + "_default"] = { project, role: "default", owner };
    for (const role of ["app", "ocr"]) {
      state.containers[project + "-" + role + "-1"] = { project, role, owner };
    }
    change();
  } else if (args.includes("exec")) {
    state.reads++;
    change();
    show("password: first-run-admin");
  } else {
    process.exit(1);
  }
} else {
  process.exit(1);
}
`;

function fixture(t, variant = "app-ocr-cuda") {
  const root = mkdtempSync(join(tmpdir(), "vision-bill-linux-test-"));
  t.after(() => rmSync(root, { force: true, recursive: true }));
  const bundle = join(root, "bundle with spaces");
  const bin = join(root, "bin");
  const state = join(root, "daemon.json");
  mkdirSync(join(bundle, "release"), { recursive: true });
  mkdirSync(bin);
  writeFileSync(join(bin, "docker"), mockDocker, { mode: 0o755 });
  copyFileSync(launcher, join(bundle, "start-linux.sh"));
  const files = {
    "check-compose.sh":
      "#!/bin/sh\nexec docker compose -f compose.yaml config --quiet\n",
    "compose.yaml": "{}\n",
    "release.json":
      JSON.stringify(
        {
          schema_version: 2,
          platform: "linux/amd64",
          variant,
          assets: {},
        },
        null,
        2,
      ) + "\n",
    "runtime.env.example": "# literal runtime settings\n",
    "release/checks.json": "{}\n",
  };
  for (const [path, contents] of Object.entries(files)) {
    writeFileSync(join(bundle, path), contents);
  }
  const sums = [...Object.keys(files), "start-linux.sh"]
    .map((path) => `${hash(readFileSync(join(bundle, path)))}  ${path}\n`)
    .join("");
  writeFileSync(join(bundle, "SHA256SUMS"), sums);
  const save = (overrides = {}) =>
    writeFileSync(
      state,
      JSON.stringify({
        volumes: {},
        containers: {},
        networks: {},
        checks: 0,
        starts: 0,
        reads: 0,
        ...overrides,
      }),
    );
  save();
  const run = (args = [], input = "", env = {}) =>
    spawnSync("bash", [join(bundle, "start-linux.sh"), ...args], {
      cwd: root,
      env: {
        ...process.env,
        PATH: `${bin}:${process.env.PATH}`,
        MOCK_STATE: state,
        ...env,
      },
      encoding: "utf8",
      input,
    });
  return {
    bundle,
    sums,
    run,
    save,
    daemon: () => JSON.parse(readFileSync(state, "utf8")),
  };
}

test("a prepared ROCm bundle starts with literal private settings and reuses its owned project", (t) => {
  const { bundle, daemon, run } = fixture(t, "app-ocr-rocm");
  const apiKey = `$token # \\ "quoted"`;
  const input = `llamacpp\n\nmodel:exact\n${apiKey}\n`;
  const first = run(["--port", "8123"], input);
  assert.equal(first.status, 0, first.stderr);
  assert.match(first.stdout, /http:\/\/127\.0\.0\.1:8123\//);
  assert.match(first.stdout, /Settings.+Dedicated OCR/);
  assert.match(first.stdout, /first-run-admin/);
  assert.ok(!`${first.stdout}${first.stderr}`.includes(apiKey));
  assert.equal(statSync(join(bundle, "runtime.env")).mode & 0o777, 0o600);
  assert.equal(
    statSync(join(bundle, ".vision-bill-install")).mode & 0o777,
    0o600,
  );
  assert.equal(
    readFileSync(join(bundle, "runtime.env"), "utf8"),
    [
      "LLM__PROVIDER=openai",
      "LLM__HOST=http://host.docker.internal:8083/v1",
      "LLM__MODEL_NAME=model:exact",
      "LLM__MODEL_LIFECYCLE=llamacpp",
      `LLM__API_KEY=${apiKey}`,
      "",
    ].join("\n"),
  );
  const firstState = daemon();
  const second = run([], "", { COMPOSE_PROJECT_NAME: "unrelated-project" });
  assert.equal(second.status, 0, second.stderr);
  assert.doesNotMatch(second.stdout, /first-run-admin/);
  assert.deepEqual(daemon().volumes, firstState.volumes);
  assert.equal(daemon().checks, 2);
  assert.equal(daemon().starts, 2);
  const explicit = run(["--show-initial-admin"]);
  assert.equal(explicit.status, 0, explicit.stderr);
  assert.match(explicit.stdout, /first-run-admin/);
  assert.equal(daemon().reads, 2);
});

test("rejects foreign exact daemon claims and Compose resources before starting or showing credentials", (t) => {
  const { bundle, daemon, run, save } = fixture(t);
  writeFileSync(join(bundle, ".vision-bill-install"), `${token}\n`, {
    mode: 0o600,
  });
  writeFileSync(join(bundle, "runtime.env"), "", { mode: 0o600 });
  const ownedClaim = { [claim]: { driver: "local", claim: token } };
  for (const overrides of [
    { volumes: { [claim]: { driver: "local", claim: foreign } } },
    {
      volumes: {
        ...ownedClaim,
        [`${project}_app-config`]: { driver: "local" },
      },
    },
    {
      volumes: {
        ...ownedClaim,
        "unexpected-volume": {
          driver: "local",
          project,
          role: "app-config",
          owner: token,
        },
      },
    },
    {
      volumes: ownedClaim,
      containers: {
        [`${project}-app-1`]: { project, role: "app", owner: foreign },
      },
    },
    {
      volumes: ownedClaim,
      containers: {
        "unexpected-container": { project, role: "app", owner: token },
      },
    },
    {
      volumes: ownedClaim,
      networks: {
        [`${project}_default`]: { project, role: "default", owner: foreign },
      },
    },
  ]) {
    save(overrides);
    const result = run(["--show-initial-admin"]);
    assert.notEqual(result.status, 0, result.stdout);
    assert.match(result.stderr, /claimed|ownership/i);
    assert.equal(daemon().starts, 0);
    assert.equal(daemon().reads, 0);
  }
});

test("verify-only checks the exact file inventory before contacting Docker", (t) => {
  const { bundle, daemon, run, sums } = fixture(t);
  const verify = (expected = hash(sums)) =>
    run(["--verify-only", "--expected-manifest-sha256", expected]);
  assert.match(verify().stdout, /Bundle files verified/);
  assert.notEqual(verify("0".repeat(64)).status, 0);
  const added = join(bundle, "release", "unlisted");
  writeFileSync(added, "unexpected");
  assert.match(verify().stderr, /unexpected/i);
  rmSync(added);
  const metadata = join(bundle, "release", "checks.json");
  rmSync(metadata);
  symlinkSync("../compose.yaml", metadata);
  assert.match(verify().stderr, /regular|symbolic/i);
  rmSync(metadata);
  writeFileSync(metadata, "edited");
  assert.match(verify().stderr, /checksum/i);
  assert.equal(daemon().checks, 0);
  assert.equal(daemon().starts, 0);
});

test("rejects a remote Docker endpoint before any Compose operation", (t) => {
  const { daemon, run } = fixture(t);
  const result = run([], "", {
    DOCKER_CONTEXT: "",
    DOCKER_HOST: "ssh://user@other-host",
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /local.*Docker/i);
  assert.equal(daemon().checks, 0);
  assert.equal(daemon().starts, 0);
});
