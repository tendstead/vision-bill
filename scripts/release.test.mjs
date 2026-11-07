import { test } from "node:test";
import assert from "node:assert/strict";
import * as release from "./release.mjs";

const sha = "a".repeat(40);
const request = { run: "123", sha, bump: "patch" };
function repository(initial = []) {
  const records = structuredClone(initial);
  return {
    records,
    async list() {
      return structuredClone(records);
    },
    async create(record) {
      await new Promise((resolve) => setTimeout(resolve, 1));
      if (records.some((r) => r.version === record.version)) return false;
      records.push(structuredClone(record));
      return true;
    },
  };
}
test("manual releases reject non-default branches, non-manual events and invalid inputs", () => {
  const env = {
    GITHUB_EVENT_NAME: "workflow_dispatch",
    GITHUB_REF: "refs/heads/main",
    GITHUB_SHA: sha,
    GITHUB_RUN_ID: "123",
    GITHUB_REPOSITORY: "Example/App",
  };
  const event = {
    repository: { default_branch: "main" },
    inputs: { bump: "minor" },
  };
  assert.deepEqual(release.releaseRequest(env, event), {
    run: "123",
    sha,
    bump: "minor",
    repository: "Example/App",
    image: "ghcr.io/example/app",
  });
  for (const change of [
    { GITHUB_REF: "refs/heads/feature" },
    { GITHUB_REF: "refs/tags/v1.0.0" },
    { GITHUB_EVENT_NAME: "pull_request" },
    { GITHUB_SHA: "x\noutput=bad" },
    { GITHUB_RUN_ID: "../bad" },
    { GITHUB_REPOSITORY: "owner/repo;echo bad" },
  ]) {
    assert.throws(() => release.releaseRequest({ ...env, ...change }, event));
  }
  for (const bump of ["patch; echo bad", "", undefined])
    assert.throws(() =>
      release.releaseRequest(env, { ...event, inputs: { bump } }),
    );
});
test("strict semantic increments reset subordinate numbers and reject unsafe values", () => {
  for (const [version, bump, want] of [
    ["0.0.0", "patch", "0.0.1"],
    ["1.2.9", "minor", "1.3.0"],
    ["1.2.9", "major", "2.0.0"],
  ])
    assert.equal(release.nextVersion(version, bump), want);
  for (const version of ["01.2.3", "1.2", "1.2.3-rc.1", "9007199254740991.0.0"])
    assert.throws(() => release.nextVersion(version, "major"));
  assert.throws(() => release.nextVersion("1.0.0", "invalid"));
});
test("atomic allocation handles concurrent runs and preserves exact retry identity", async () => {
  const repo = repository([{ version: "2.9.9", sha: "old" }]);
  const results = await Promise.all([
    release.allocateVersion(repo, request),
    release.allocateVersion(repo, { ...request, run: "124", bump: "minor" }),
  ]);
  assert.equal(new Set(results.map((r) => r.version)).size, 2);
  assert.equal(results[0].version, "2.9.10");
  assert.deepEqual(await release.allocateVersion(repo, request), results[0]);
  await assert.rejects(
    release.allocateVersion(repo, { ...request, sha: "b".repeat(40) }),
    /changed/,
  );
  await assert.rejects(
    release.allocateVersion(repo, { ...request, bump: "major" }),
    /changed/,
  );
});
test("a publication failure retains allocation and cannot create a GitHub release", async () => {
  const repo = repository();
  const record = await release.allocateVersion(repo, request);
  const calls = [];
  await assert.rejects(
    release.publishRelease(record, {
      async image() {
        throw Error("registry down");
      },
      async release() {
        calls.push("release");
      },
    }),
  );
  assert.deepEqual(calls, []);
  const retry = await release.allocateVersion(repo, request);
  await release.publishRelease(retry, {
    async image() {
      calls.push("image");
      return "sha256:" + "d".repeat(64);
    },
    async release(r, digest) {
      calls.push([r.version, digest]);
    },
  });
  assert.deepEqual(calls, ["image", ["0.0.1", "sha256:" + "d".repeat(64)]]);
  assert.equal(repo.records.length, 1);
});
test("existing registry images must match source, revision and version before reuse", () => {
  const record = { ...request, version: "1.2.3" };
  const source = "https://github.com/example/app";
  const manifest = {
    annotations: {
      "org.opencontainers.image.source": source,
      "org.opencontainers.image.revision": sha,
      "org.opencontainers.image.version": "1.2.3",
    },
  };
  assert.doesNotThrow(() => release.verifyImage(manifest, record, source));
  for (const key of Object.keys(manifest.annotations))
    assert.throws(
      () =>
        release.verifyImage(
          { annotations: { ...manifest.annotations, [key]: "wrong" } },
          record,
          source,
        ),
      /match/,
    );
  assert.throws(() => release.verifyImage({}, record, source), /match/);
});

test("GitHub API errors never leak response bodies or credentials", async () => {
  const api = release.githubApi(
    "example/app",
    "secret-token",
    async (url, options) => {
      assert.equal(url, "https://api.github.com/repos/example/app/git/refs");
      assert.equal(options.headers.Authorization, "Bearer secret-token");
      return new Response("private information secret-token", { status: 403 });
    },
  );
  await assert.rejects(
    api("/git/refs", "POST", {}),
    (error) =>
      error.status === 403 &&
      !error.message.includes("secret-token") &&
      !error.message.includes("private information"),
  );
  const missing = release.githubApi(
    "example/app",
    "token",
    async () => new Response(null, { status: 404 }),
  );
  assert.equal(await missing("/releases/tags/v1.0.0"), null);
  await assert.rejects(
    missing("/git/refs", "POST", {}),
    (error) => error.status === 404,
  );
});
test("tag adapter includes legacy versions, ignores prereleases and retries atomic conflicts", async () => {
  const paths = [];
  const api = async (path, method = "GET", body) => {
    paths.push([path, method, body]);
    if (path.startsWith("/git/matching-refs/"))
      return [
        { ref: "refs/tags/v1.2.3", object: { type: "commit", sha } },
        { ref: "refs/tags/v9.0.0-rc.1", object: { type: "commit", sha } },
      ];
    if (path === "/git/tags") return { sha: "tag-object" };
    if (path === "/git/refs")
      throw Object.assign(Error("conflict"), { status: 422 });
    throw Error(`Unexpected path ${path}`);
  };
  const repo = release.tagRepository(api);
  assert.deepEqual(await repo.list(), [{ version: "1.2.3" }]);
  assert.equal(await repo.create({ ...request, version: "1.2.4" }), false);
  assert.equal(paths[2][2].ref, "refs/tags/v1.2.4");
  assert.equal(paths[1][2].object, sha);
});
test("tag adapter reads reservations and rejects tampered commit metadata", async () => {
  const record = { ...request, version: "1.2.3" };
  let target = sha;
  const api = async (path) =>
    path.startsWith("/git/matching-refs/")
      ? [
          {
            ref: "refs/tags/v1.2.3",
            object: { type: "tag", sha: "tag-object" },
          },
        ]
      : {
          message: "Tendstead release allocation\n" + JSON.stringify(record),
          object: { type: "commit", sha: target },
        };
  const repo = release.tagRepository(api);
  assert.deepEqual(await repo.list(), [record]);
  target = "b".repeat(40);
  await assert.rejects(repo.list(), /Invalid release allocation/);
});
test("registry 404 permits a new image but denied and unavailable registries fail closed", async () => {
  const req = { repository: "example/app", image: "ghcr.io/example/app" };
  const record = { ...request, version: "1.2.3" };
  for (const status of [404, 401, 403, 429, 500]) {
    const fetcher = async (url, options) => {
      if (url.startsWith("https://ghcr.io/token?")) {
        assert.equal(
          options.headers.Authorization,
          "Basic " + Buffer.from("actor:secret").toString("base64"),
        );
        return Response.json({ token: "registry-token" });
      }
      assert.equal(url, "https://ghcr.io/v2/example/app/manifests/1.2.3");
      assert.equal(options.headers.Authorization, "Bearer registry-token");
      return new Response(null, { status });
    };
    if (status === 404)
      assert.equal(
        await release.existingImage(req, record, "secret", "actor", fetcher),
        null,
      );
    else
      await assert.rejects(
        release.existingImage(req, record, "secret", "actor", fetcher),
        /lookup failed/,
      );
  }
  await assert.rejects(
    release.existingImage(
      req,
      record,
      "secret",
      "actor",
      async () => new Response(null, { status: 403 }),
    ),
    /authorization failed/,
  );
});
test("existing matching image is reused by digest and mismatched images abort", async () => {
  const req = { repository: "example/app", image: "ghcr.io/example/app" };
  const record = { ...request, version: "1.2.3" };
  const manifest = {
    annotations: {
      "org.opencontainers.image.source": "https://github.com/example/app",
      "org.opencontainers.image.revision": sha,
      "org.opencontainers.image.version": "1.2.3",
    },
  };
  const digest = "sha256:" + "d".repeat(64);
  const fetcher = async (url) =>
    url.startsWith("https://ghcr.io/token?")
      ? Response.json({ token: "registry-token" })
      : Response.json(manifest, {
          headers: { "docker-content-digest": digest },
        });
  assert.equal(
    await release.existingImage(req, record, "secret", "actor", fetcher),
    digest,
  );
  manifest.annotations["org.opencontainers.image.revision"] = "wrong";
  await assert.rejects(
    release.existingImage(req, record, "secret", "actor", fetcher),
    /match/,
  );
});
test("invalid image digest prevents GitHub release creation", async () => {
  let created = false;
  await assert.rejects(
    release.publishRelease(
      { ...request, version: "1.0.0" },
      {
        async image() {
          return "bad";
        },
        async release() {
          created = true;
        },
      },
    ),
    /invalid digest/,
  );
  assert.equal(created, false);
});

test("full release binds tag, image and release to dispatch; recovery skips rebuilding and protects credentials", async () => {
  const { mkdtempSync, writeFileSync, readFileSync, rmSync } =
    await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = mkdtempSync(join(tmpdir(), "release-test-"));
  try {
    const eventPath = join(dir, "event.json");
    const summary = join(dir, "summary");
    writeFileSync(
      eventPath,
      JSON.stringify({
        repository: { default_branch: "main" },
        inputs: { bump: "minor" },
      }),
    );
    const env = {
      GITHUB_EVENT_NAME: "workflow_dispatch",
      GITHUB_REF: "refs/heads/main",
      GITHUB_SHA: sha,
      GITHUB_RUN_ID: "123",
      GITHUB_REPOSITORY: "example/app",
      GITHUB_EVENT_PATH: eventPath,
      GITHUB_STEP_SUMMARY: summary,
      GITHUB_TOKEN: "private-token",
      GITHUB_ACTOR: "actor",
    };
    const digest = "sha256:" + "d".repeat(64);
    let tag;
    let published;
    let builds = 0;
    let releaseBody;
    let failRelease = true;
    const command = (program, args, options) => {
      if (program === "git") {
        assert.deepEqual(args, ["rev-parse", "HEAD"]);
        return sha + "\n";
      }
      assert.equal(program, "docker");
      builds++;
      assert.equal(options.env.GITHUB_TOKEN, undefined);
      assert.ok(!args.join(" ").includes("private-token"));
      assert.deepEqual(args.slice(0, 4), [
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
      ]);
      assert.equal(
        args[args.indexOf("--tag") + 1],
        "ghcr.io/example/app:0.1.0",
      );
      assert.ok(
        args.includes(
          "index,manifest:org.opencontainers.image.revision=" + sha,
        ),
      );
      assert.deepEqual(args.slice(-2), ["--push", "."]);
      writeFileSync(
        args[args.indexOf("--metadata-file") + 1],
        JSON.stringify({ "containerimage.digest": digest }),
      );
      published = digest;
    };
    const api = async (path, method = "GET", body) => {
      if (path.startsWith("/git/matching-refs/"))
        return tag
          ? [
              {
                ref: "refs/tags/v0.1.0",
                object: { type: "tag", sha: "object" },
              },
            ]
          : [];
      if (path === "/git/tags" && method === "POST") {
        assert.equal(body.object, sha);
        tag = { message: body.message, object: { type: "commit", sha } };
        return { sha: "object" };
      }
      if (path === "/git/tags/object") return tag;
      if (path === "/git/refs") {
        assert.equal(body.ref, "refs/tags/v0.1.0");
        return {};
      }
      if (path === "/releases/tags/v0.1.0")
        return releaseBody
          ? { body: releaseBody.body, draft: false, prerelease: false }
          : null;
      if (path === "/releases/generate-notes") {
        assert.equal(body.target_commitish, sha);
        return { body: "Changes" };
      }
      if (path === "/releases") {
        assert.equal(published, digest);
        if (failRelease) throw Error("GitHub temporarily unavailable");
        releaseBody = body;
        return {};
      }
      throw Error("Unexpected API " + path);
    };
    const dependencies = {
      env,
      command,
      api,
      lookup: async () => published || null,
    };
    await assert.rejects(
      release.runWorkflow(dependencies),
      /temporarily unavailable/,
    );
    assert.equal(builds, 1);
    assert.equal(releaseBody, undefined);
    failRelease = false;
    await release.runWorkflow(dependencies);
    assert.equal(builds, 1);
    assert.equal(releaseBody.tag_name, "v0.1.0");
    assert.equal(releaseBody.target_commitish, sha);
    assert.match(releaseBody.body, /ghcr.io\/example\/app@sha256:dddd/);
    assert.ok(!JSON.stringify(releaseBody).includes("private-token"));
    await release.runWorkflow(dependencies);
    assert.equal(builds, 1);
    assert.match(readFileSync(summary, "utf8"), /v0.1.0/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("tag discovery handles more than 100 refs without unsupported pagination", async () => {
  let calls = 0;
  const refs = Array.from({ length: 101 }, (_, i) => ({
    ref: `refs/tags/v1.0.${i}`,
    object: { type: "commit", sha },
  }));
  const repo = release.tagRepository(async () => {
    if (++calls > 1) throw Error("Matching references is not paginated");
    return refs;
  });
  const record = await release.allocateVersion(
    {
      ...repo,
      async create() {
        return true;
      },
    },
    request,
  );
  assert.equal(record.version, "1.0.101");
  assert.equal(calls, 1);
});
test("a successful registry response without a digest cannot be reused", async () => {
  const req = { repository: "example/app", image: "ghcr.io/example/app" };
  const record = { ...request, version: "1.2.3" };
  const fetcher = async (url) =>
    url.startsWith("https://ghcr.io/token?")
      ? Response.json({ token: "token" })
      : Response.json({
          annotations: {
            "org.opencontainers.image.source": "https://github.com/example/app",
            "org.opencontainers.image.revision": sha,
            "org.opencontainers.image.version": "1.2.3",
          },
        });
  await assert.rejects(
    release.existingImage(req, record, "secret", "actor", fetcher),
    /invalid digest/,
  );
});
