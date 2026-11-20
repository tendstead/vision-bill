#!/usr/bin/env node
// CPU-only coordinator/provision tests run as actual container PID1.
import { execFileSync } from "node:child_process";
import { isAbsolute, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";
import {
  withOwnedContainer,
  withOwnedVolumes,
  dockerVolumeDriver,
} from "./distribution-release.mjs";

export function ownedVolumeFixture(image, execute = execFileSync) {
  process.stdout.write(
    `CPU owned-volume fixture host uid: ${process.getuid()}\n`,
  );
  const labels = { "vision-bill.release.nonce": randomUUID() };
  const volumes = [];
  withOwnedVolumes(dockerVolumeDriver(execute), labels, ({ state, coord }) => {
    volumes.push(state, coord);
    withOwnedContainer(
      {
        create: () =>
          execute(
            "docker",
            [
              "create",
              "--network",
              "none",
              "--mount",
              `type=volume,src=${state},dst=/state`,
              "--mount",
              `type=volume,src=${coord},dst=/coord`,
              image,
              "python",
              "-c",
              "import os,pathlib; assert os.getuid()==0; p=pathlib.Path('/state/private'); p.mkdir(mode=0o700); (p/'owned').write_text('root-only'); pathlib.Path('/coord/lock').touch(mode=0o600); print('CPU owned-volume fixture: root wrote private state')",
            ],
            { encoding: "utf8" },
          ).trim(),
        remove: (id) =>
          execute("docker", ["rm", "-f", id], { stdio: "inherit" }),
      },
      (id) =>
        execute("docker", ["start", "--attach", id], { stdio: "inherit" }),
    );
  });
  for (const name of volumes) {
    let exists = false;
    try {
      execute("docker", ["volume", "inspect", name], { stdio: "pipe" });
      exists = true;
    } catch (error) {
      if (!error.stderr?.toString().includes("no such volume")) throw error;
    }
    if (exists) throw Error(`Owned fixture volume was not removed: ${name}`);
  }
}
export function pid1Command(environment, root) {
  if (
    !/^3\.[0-9]+$/.test(environment.version) ||
    ![environment.site_packages, root].every(
      (path) =>
        typeof path === "string" && isAbsolute(path) && !/[,:\r\n]/.test(path),
    )
  )
    throw Error("Invalid current Python test environment");
  return [
    "run",
    "--rm",
    "--init=false",
    "--network",
    "none",
    "--mount",
    `type=bind,source=${root},target=${root},readonly`,
    "--mount",
    `type=bind,source=${environment.site_packages},target=/testdeps,readonly`,
    "--env",
    `PYTHONPATH=/testdeps:${root}/src:${root}/tests:${root}/paddleocr-service`,
    "--env",
    "PYTHONDONTWRITEBYTECODE=1",
    "--workdir",
    root,
    `python:${environment.version}-slim`,
    "python",
    "-m",
    "pytest",
    "tests/test_ocr_initialize.py",
    "tests/test_ocr_provision.py",
    "-q",
    "-p",
    "no:cacheprovider",
  ];
}
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  try {
    const root = fileURLToPath(new URL("../", import.meta.url)).replace(
      /\/$/,
      "",
    );
    const environment = JSON.parse(
      execFileSync(
        "uv",
        [
          "run",
          "--extra",
          "dev",
          "python",
          "-c",
          'import json,sys,sysconfig; print(json.dumps({"version":f"{sys.version_info.major}.{sys.version_info.minor}","site_packages":sysconfig.get_paths()["purelib"]}))',
        ],
        { cwd: root, encoding: "utf8" },
      ),
    );
    execFileSync("docker", pid1Command(environment, root), {
      stdio: "inherit",
    });
    ownedVolumeFixture(`python:${environment.version}-slim`);
  } catch {
    process.stderr.write("CPU PID1 distribution test gate failed\n");
    process.exitCode = 1;
  }
}
