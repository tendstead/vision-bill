import { readFileSync, mkdtempSync, rmSync, appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const semver = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const marker = "Tendstead release allocation\n";
function parts(version) {
  if (!semver.test(version)) throw Error("Invalid semantic version");
  const values = version.split(".").map(Number);
  if (values.some((n) => !Number.isSafeInteger(n)))
    throw Error("Version exceeds safe integer range");
  return values;
}
export function nextVersion(version, bump) {
  let [major, minor, patch] = parts(version);
  if (bump === "patch") patch++;
  else if (bump === "minor") {
    minor++;
    patch = 0;
  } else if (bump === "major") {
    major++;
    minor = 0;
    patch = 0;
  } else throw Error("Invalid release bump");
  const result = `${major}.${minor}.${patch}`;
  parts(result);
  return result;
}
function compare(a, b) {
  const x = parts(a),
    y = parts(b);
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return 0;
}
export function releaseRequest(env, event) {
  if (
    env.GITHUB_EVENT_NAME !== "workflow_dispatch" ||
    !event.repository?.default_branch ||
    env.GITHUB_REF !== `refs/heads/${event.repository.default_branch}`
  )
    throw Error("Release requires a manual dispatch on the default branch");
  if (
    !/^[a-f0-9]{40}$/.test(env.GITHUB_SHA || "") ||
    !/^\d+$/.test(env.GITHUB_RUN_ID || "")
  )
    throw Error("Invalid release identity");
  if (
    !/^[a-zA-Z0-9][a-zA-Z0-9-]*\/[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(
      env.GITHUB_REPOSITORY || "",
    )
  )
    throw Error("Invalid repository");
  const bump = event.inputs?.bump;
  if (!["patch", "minor", "major"].includes(bump))
    throw Error("Invalid release bump");
  return {
    run: env.GITHUB_RUN_ID,
    sha: env.GITHUB_SHA,
    bump,
    repository: env.GITHUB_REPOSITORY,
    image: `ghcr.io/${env.GITHUB_REPOSITORY.toLowerCase()}`,
  };
}
export async function allocateVersion(repository, request) {
  for (let attempt = 0; attempt < 20; attempt++) {
    const records = await repository.list();
    const existing = records.find((r) => r.run === request.run);
    if (existing) {
      if (existing.sha !== request.sha || existing.bump !== request.bump)
        throw Error("Allocated release identity changed");
      return existing;
    }
    const baseline = records.reduce(
      (v, r) => (compare(r.version, v) > 0 ? r.version : v),
      "0.0.0",
    );
    const record = {
      run: request.run,
      sha: request.sha,
      bump: request.bump,
      version: nextVersion(baseline, request.bump),
    };
    if (await repository.create(record)) return record;
  }
  throw Error("Concurrent allocation did not settle; rerun this workflow");
}
export async function publishRelease(record, publisher) {
  const digest = await publisher.image(record);
  if (!/^sha256:[a-f0-9]{64}$/.test(digest || ""))
    throw Error("Registry returned an invalid digest");
  await publisher.release(record, digest);
  return digest;
}
export function verifyImage(manifest, record, source) {
  const expected = {
    "org.opencontainers.image.source": source,
    "org.opencontainers.image.revision": record.sha,
    "org.opencontainers.image.version": record.version,
  };
  for (const [key, value] of Object.entries(expected))
    if (manifest.annotations?.[key] !== value)
      throw Error(
        "Existing image does not match this release; refusing to overwrite it",
      );
}
export function githubApi(repository, token, fetcher = fetch) {
  return async (path, method = "GET", body) => {
    const response = await fetcher(
      `https://api.github.com/repos/${repository}${path}`,
      {
        method,
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "Content-Type": "application/json",
        },
        body: body === undefined ? undefined : JSON.stringify(body),
      },
    );
    if (response.status === 404 && method === "GET") return null;
    if (!response.ok) {
      const error = Error(`GitHub API ${method} failed (${response.status})`);
      error.status = response.status;
      throw error;
    }
    return response.status === 204 ? null : response.json();
  };
}
export function tagRepository(api) {
  return {
    async list() {
      const records = [];
      // This GitHub endpoint returns all matching refs; it is not paginated.
      const refs = (await api("/git/matching-refs/tags/v")) || [];
      for (const ref of refs) {
        const version = ref.ref.replace("refs/tags/v", "");
        if (!semver.test(version)) continue;
        parts(version);
        let record = { version };
        if (ref.object.type === "tag") {
          const tag = await api(`/git/tags/${ref.object.sha}`);
          if (tag?.message.startsWith(marker)) {
            record = JSON.parse(tag.message.slice(marker.length));
            if (
              record.version !== version ||
              record.sha !== tag.object.sha ||
              tag.object.type !== "commit" ||
              !/^\d+$/.test(record.run || "") ||
              !["patch", "minor", "major"].includes(record.bump)
            )
              throw Error("Invalid release allocation tag");
          }
        }
        records.push(record);
      }
      return records;
    },
    async create(record) {
      const tag = await api("/git/tags", "POST", {
        tag: `v${record.version}`,
        message: marker + JSON.stringify(record),
        object: record.sha,
        type: "commit",
      });
      try {
        await api("/git/refs", "POST", {
          ref: `refs/tags/v${record.version}`,
          sha: tag.sha,
        });
        return true;
      } catch (error) {
        if (error.status === 422) return false;
        throw error;
      }
    },
  };
}
// Only 404 means absent. Authentication, rate-limit and network errors fail closed.
export async function existingImage(
  request,
  record,
  token,
  actor,
  fetcher = fetch,
) {
  const repository = request.image.slice("ghcr.io/".length);
  const auth = await fetcher(
    `https://ghcr.io/token?service=ghcr.io&scope=${encodeURIComponent(`repository:${repository}:pull,push`)}`,
    {
      headers: {
        Authorization: `Basic ${Buffer.from(`${actor}:${token}`).toString("base64")}`,
      },
    },
  );
  if (!auth.ok) throw Error(`GHCR authorization failed (${auth.status})`);
  const credentials = await auth.json();
  if (!credentials.token) throw Error("GHCR authorization returned no token");
  const response = await fetcher(
    `https://ghcr.io/v2/${repository}/manifests/${record.version}`,
    {
      headers: {
        Authorization: `Bearer ${credentials.token}`,
        Accept:
          "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json",
      },
    },
  );
  if (response.status === 404) return null;
  if (!response.ok)
    throw Error(`GHCR image lookup failed (${response.status})`);
  verifyImage(
    await response.json(),
    record,
    `https://github.com/${request.repository}`,
  );
  const digest = response.headers.get("docker-content-digest");
  if (!/^sha256:[a-f0-9]{64}$/.test(digest || ""))
    throw Error("GHCR returned an invalid digest");
  return digest;
}
export async function runWorkflow({
  env = process.env,
  command = execFileSync,
  api,
  lookup = existingImage,
} = {}) {
  const request = releaseRequest(
    env,
    JSON.parse(readFileSync(env.GITHUB_EVENT_PATH, "utf8")),
  );
  if (
    command("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim() !==
    request.sha
  )
    throw Error("Checkout does not match dispatch commit");
  if (!env.GITHUB_TOKEN || !env.GITHUB_ACTOR)
    throw Error("GitHub workflow credentials are required");
  api ??= githubApi(request.repository, env.GITHUB_TOKEN);
  const record = await allocateVersion(tagRepository(api), request);
  const source = `https://github.com/${request.repository}`;
  const digest = await publishRelease(record, {
    async image(r) {
      const existing = await lookup(
        request,
        r,
        env.GITHUB_TOKEN,
        env.GITHUB_ACTOR,
      );
      if (existing) return existing;
      const directory = mkdtempSync(join(tmpdir(), "release-"));
      try {
        const metadata = join(directory, "metadata.json");
        const labels = {
          "org.opencontainers.image.source": source,
          "org.opencontainers.image.revision": r.sha,
          "org.opencontainers.image.version": r.version,
        };
        const args = [
          "buildx",
          "build",
          "--platform",
          "linux/amd64",
          "--provenance=mode=min",
          "--metadata-file",
          metadata,
          "--tag",
          `${request.image}:${r.version}`,
        ];
        for (const [key, value] of Object.entries(labels))
          args.push(
            "--label",
            `${key}=${value}`,
            "--annotation",
            `index,manifest:${key}=${value}`,
          );
        // Build subprocess receives no GitHub API token. Registry credentials are
        // managed by docker/login-action outside the build context.
        const { GITHUB_TOKEN: _token, ...buildEnv } = env;
        command("docker", [...args, "--push", "."], {
          stdio: "inherit",
          env: buildEnv,
        });
        return JSON.parse(readFileSync(metadata, "utf8"))[
          "containerimage.digest"
        ];
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    },
    async release(r, imageDigest) {
      const existing = await api(`/releases/tags/v${r.version}`);
      const reference = `${request.image}@${imageDigest}`;
      if (existing) {
        if (
          existing.draft ||
          existing.prerelease ||
          !existing.body?.includes(reference)
        )
          throw Error("Existing GitHub release does not match published image");
        return;
      }
      const notes = await api("/releases/generate-notes", "POST", {
        tag_name: `v${r.version}`,
        target_commitish: r.sha,
      });
      await api("/releases", "POST", {
        tag_name: `v${r.version}`,
        target_commitish: r.sha,
        name: `v${r.version}`,
        body: `${notes.body}\n\nImage: \`${request.image}:${r.version}\`\nDigest: \`${reference}\`\nCommit: \`${r.sha}\``,
        draft: false,
        prerelease: false,
        make_latest: "false",
      });
    },
  });
  if (env.GITHUB_STEP_SUMMARY)
    appendFileSync(
      env.GITHUB_STEP_SUMMARY,
      `Released **v${record.version}**\n\nImage: \`${request.image}:${record.version}\`\n\nDigest: \`${digest}\`\n\nCommit: \`${record.sha}\`\n`,
    );
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  runWorkflow().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
