#!/usr/bin/env python3
"""Build a local GPU OCR candidate and start source Compose with an idle controller."""

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import platform
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
NATIVE = "paddleocr-service"
RECORD_LIMIT = 32 * 1024**2
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
NATIVE_PLATFORM = "linux/amd64"
DEFERRED_STATUS = "/native-build/status.json"
# The official Python base also pinned in the native Dockerfiles.
JOURNAL_INSPECTOR = "python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")
MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}


@dataclass(frozen=True)
class DeferredApp:
    project: str
    image_id: str
    status_dir: Path
    revision: str

    def overlay(self) -> dict[str, Any]:
        service = app_image_service(self.image_id)
        service["environment"].update(
            VISION_BILL_OCR_NATIVE_BUILD_STATUS=DEFERRED_STATUS, OCR__ENDPOINT="",
        )
        service["healthcheck"]["test"][-1] = (
            "import urllib.request; "
            "urllib.request.urlopen('http://127.0.0.1:8080/api/v1/system/ready', timeout=5)"
        )
        service["volumes"] = [{
            "type": "bind", "source": str(self.status_dir), "target": "/native-build",
            "read_only": True, "bind": {"create_host_path": False},
        }]
        return {
            "x-vision-bill-deferred": {
                "project": self.project, "image_id": self.image_id,
                "checkout": str(ROOT), "revision": self.revision,
            },
            "services": {"vision_bill": service},
        }

# Use the verifiers shipped in the exact built image. Neither provisioning nor
# qualification can run during this offline metadata-only invocation.
BIND_BUNDLE = """
import ctypes, json, os, subprocess, sys
from pathlib import Path
from ocr_service.bootstrap import bind_approved_dependencies
from ocr_service.device_inventory import inspect_device_identity
from ocr_service.initialize import verify_image
from ocr_service.paddle_snapshots import is_snapshot_reference
from ocr_service.profile import content_identity
from ocr_service.provision import Archive, MODEL_NAMES, ReleaseBundle, file_digest, source_identities, verify_sources

image_id, config_id, release, expected_sources, *verification = sys.argv[1:]
root = Path(release)
manifest = json.loads((root / 'bundle.json').read_bytes())
sources = source_identities()
if expected_sources and sources != json.loads(expected_sources):
    raise ValueError('Reused OCR image sources, profile or runtime lock differ from this checkout')
manifest.update(image_identity=image_id, image_config_sha256=config_id[7:],
                profile_sha256=sources['profile.json'], source_files=sources)
# Migrate only the staged, approved four-model input. Existing deployment
# records keep their exact contents and image-specific verification on restart.
models = manifest['models']
if 'UVDoc' not in MODEL_NAMES and set(models) == set(MODEL_NAMES) | {'UVDoc'}:
    retired = Archive.model_validate(models.pop('UVDoc'))
    # The previous published snapshot manifest sealed this entire UVDoc record.
    if is_snapshot_reference(retired.source) and content_identity(retired.model_dump(mode='json')) != (
        '0ac3019fae2929a39bb9cac71c04ea0e78e3d4ea06ec6358f3cfc96dc3a6c614'
    ):
        raise ValueError('Legacy official UVDoc snapshot differs from the approved record')
# Validate in the bind before emitting the final manifest, without touching assets.
bundle = ReleaseBundle.model_validate(manifest)
verify_sources(bundle)
verify_image(bundle, root / 'image-config.json', config_id)
if bundle.accelerator == 'cuda':
    channel, digest, lock, resolver = verification
    subprocess.run([sys.executable, resolver, '--check', lock, '--channel', channel], check=True)
    if file_digest(Path(lock)) != digest:
        raise ValueError('Built CUDA lock differs from the selected checkout lock')
manifest['expected_packages'] = bind_approved_dependencies(
    bundle.expected_packages, accelerator=bundle.accelerator,
)
bundle = ReleaseBundle.model_validate(manifest)
device = inspect_device_identity(bundle.accelerator, bundle.gpu_index)
if bundle.accelerator == 'cuda':
    count = ctypes.c_int()
    if ctypes.CDLL('libcuda.so.1').cuDeviceGetCount(ctypes.byref(count)) or count.value != 1:
        raise ValueError('Docker must expose exactly one selected CUDA device')
    from uuid import UUID
    actual_uuid = 'gpu-' + str(UUID(hex=device['uuid']))
    expected_uuid = os.environ.get('VISION_BILL_CUDA_SELECTION', '')
    if expected_uuid and not actual_uuid.startswith(expected_uuid):
        raise ValueError('Docker CUDA device differs from the selected host GPU')
    if tuple(device['compute_capability']) < (7, 0) or device['driver_version'] < 12000:
        raise ValueError('The selected OCR GPU requires SM 7.0 or newer and a CUDA 12-capable driver')
if bundle.accelerator == 'rocm':
    from ocr_service.targets import verify_rocm_image_target
    kernels = json.loads(Path('/opt/ocr/build-evidence/kernels.json').read_bytes())
    verify_rocm_image_target(kernels, device)
bound = bundle.model_dump(mode='json')
if bundle.accelerator == 'cuda':
    bound['_bound_cuda_uuid'] = device['uuid']
else:
    bound['_bound_rocm_uuid'] = device['uuid']
print(json.dumps(bound, sort_keys=True, indent=2))
"""


def unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in items:
        if key in record:
            raise ValueError("Duplicate key in approved bundle")
        record[key] = value
    return record


def trusted_file(path: Path) -> os.stat_result:
    details = path.lstat()
    system_owner = Path("/").stat().st_uid
    trusted_owners = {system_owner, os.getuid()}
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_uid not in trusted_owners
        or details.st_mode & 0o022
        or path.resolve() != path
    ):
        raise ValueError(f"Approved input must be a trusted regular file: {path}")
    for parent in path.parents:
        owner = parent.lstat()
        shared_sticky = owner.st_uid == system_owner and owner.st_mode & stat.S_ISVTX
        if not stat.S_ISDIR(owner.st_mode) or (
            not shared_sticky and (owner.st_uid not in trusted_owners or owner.st_mode & 0o022)
        ):
            raise ValueError(f"Approved input parent must be a trusted directory: {parent}")
    return details


def https_asset(source: str) -> bool:
    url = urlsplit(source)
    if not url.scheme:
        return False
    if (
        not source.startswith("https://")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or any(char.isspace() or char == "\\" for char in source)
        or any(
            re.search(r"secret|token|signature|password|api.?key", key, re.IGNORECASE)
            for key, _ in parse_qsl(url.query)
        )
    ):
        raise ValueError("Approved remote assets require an HTTPS URL without credentials")
    return True


def path_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkout_sources() -> dict[str, str]:
    root = ROOT / NATIVE
    paths = [
        path
        for directory in ("ocr_service", "ocr_contract")
        for path in (root / directory).rglob("*.py")
    ]
    paths.extend(root / name for name in ("server.py", "profile.json", "runtime-lock.json"))
    return {path.relative_to(root).as_posix(): path_digest(path) for path in paths}


def stage_approved_inputs(
    source: Path, destination: Path | None, accelerator: str
) -> dict[str, Any]:
    manifest_path = source / "bundle.json"
    if trusted_file(manifest_path).st_size > RECORD_LIMIT:
        raise ValueError("Approved bundle exceeds its size limit")
    manifest: dict[str, Any] = json.loads(manifest_path.read_bytes(), object_pairs_hook=unique_keys)
    if manifest["accelerator"] != accelerator or manifest["gpu_index"] != 0:
        raise ValueError(
            "Approved bundle must select this accelerator and container device index 0"
        )
    models = manifest["models"]
    if not isinstance(models, dict) or not all(
        https_asset(asset["source"]) for asset in models.values()
    ):
        raise ValueError("Browser activation requires approved HTTPS model archives")
    fixtures = manifest["fixtures"]
    fixture_source = fixtures["source"]
    if not https_asset(fixture_source):
        parts = PurePosixPath(fixture_source).parts
        if (
            not parts
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", fixture_source)
            or any(not part or part.startswith(".") for part in fixture_source.split("/"))
            or parts[0] in {"bundle.json", "image-config.json", "compose.local.json"}
        ):
            raise ValueError("Unsafe approved fixture archive path")
        archive = source / fixture_source
        size = fixtures["size_bytes"]
        if type(size) is not int or not 0 < size <= 20 * 1024**3:
            raise ValueError("Invalid approved fixture archive size")
        if trusted_file(archive).st_size != size:
            raise ValueError("Approved fixture archive size mismatch")
        staged = archive
        if destination is not None:
            staged = destination / fixture_source
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, staged)
        if path_digest(staged) != fixtures["sha256"] or staged.stat().st_size != size:
            raise ValueError("Approved fixture archive identity mismatch")
    if destination is not None:
        (destination / "bundle.json").write_text(json.dumps(manifest))
    return manifest


def image_tag(accelerator: str, image_id: str) -> str:
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("Invalid native Docker engine image ID")
    return f"vision-bill-ocr-{accelerator}:source-{image_id[7:]}"


def app_image_service(image_id: str) -> dict[str, Any]:
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("Invalid application Docker engine image ID")
    return {
        "image": f"vision-bill:source-{image_id[7:]}",
        "pull_policy": "never",
        "healthcheck": {
            "test": [
                "CMD", "python", "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v1/system/ui-config', timeout=5)",
            ],
            "interval": "3s",
            "timeout": "6s",
            "retries": 10,
            "start_period": "20s",
        },
        "environment": {"OCR_ACTIVATION_ON_STARTUP": "false"},
    }


def managed_overlay(
    accelerator: str,
    release: Path,
    image_id: str,
    app_id: str,
    project: str,
    device_environment: dict[str, str],
    bundle_digest: str,
    *,
    config_id: str,
) -> dict[str, Any]:
    if not IMAGE_ID.fullmatch(config_id):
        raise ValueError("Invalid native Docker image configuration ID")
    return {
        "x-vision-bill-source": {
            "project": project,
            "device_environment": device_environment,
            "bundle_sha256": bundle_digest,
            "image_ids": {"vision_bill": app_id, NATIVE: image_id},
            "image_config_sha256": config_id[7:],
        },
        "services": {
            "vision_bill": app_image_service(app_id),
            NATIVE: {
                "image": image_tag(accelerator, image_id),
                "platform": NATIVE_PLATFORM,
                "pull_policy": "never",
                "environment": {"OCR_IMAGE_ID": config_id},
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(release),
                        "target": "/release",
                        "read_only": True,
                        "bind": {"create_host_path": False},
                    }
                ],
            },
        },
    }


@contextmanager
def build_phase(name: str) -> Iterator[None]:
    started = time.monotonic()
    print(f"ocr_source_phase phase={name} status=started", flush=True)
    try:
        yield
    except BaseException:
        print(f"ocr_source_phase phase={name} status=interrupted elapsed_s={time.monotonic() - started:.1f}", flush=True)
        raise
    print(f"ocr_source_phase phase={name} status=complete elapsed_s={time.monotonic() - started:.1f}", flush=True)


def command(*args: str, capture: bool = False) -> str:
    show_progress = "--build-context" in args
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture and not show_progress else None,
        text=True,
    )
    return result.stdout or ""


def stage_first_run_inputs(release: Path, accelerator: str) -> None:
    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=release.parent) as temporary:
        prepared = Path(temporary) / "approved"
        command(
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "--with",
            "pydantic==2.13.4",
            "--with",
            "packaging==26.2",
            "--with",
            "pillow==12.3.0",
            "python",
            f"{NATIVE}/build/prepare_ocr_inputs.py",
            accelerator,
            "--models",
            f"{NATIVE}/build/paddle-model-snapshots.json",
            "--fixtures",
            f"{NATIVE}/first-run-fixtures/fixtures.json",
            "--output",
            str(prepared),
            capture=True,
        )
        stage_approved_inputs(prepared, release, accelerator)


def build_history_blob(digest: str) -> bytes:
    # An empty build ref selects the builder's latest record. The explicit
    # digest addresses a blob on that node independently of the selected record.
    args = ("docker", "buildx", "history", "inspect", "attachment", "", digest)
    with subprocess.Popen(
        args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=False
    ) as process:
        assert process.stdout is not None
        record: bytes = process.stdout.read(RECORD_LIMIT + 1)
        if len(record) > RECORD_LIMIT:
            process.kill()
            raise ValueError("Docker image metadata exceeds its size limit")
        if status := process.wait():
            raise subprocess.CalledProcessError(status, args)
    return record


def resolve_image_config(image_id: str, read_blob: Callable[[str], bytes]) -> tuple[str, bytes]:
    def read_verified(digest: str) -> tuple[dict[str, Any], bytes]:
        if not isinstance(digest, str) or not IMAGE_ID.fullmatch(digest):
            raise ValueError("Invalid Docker image metadata digest")
        raw = read_blob(digest)
        if not 0 < len(raw) <= RECORD_LIMIT or hashlib.sha256(raw).hexdigest() != digest[7:]:
            raise ValueError("Docker image metadata digest does not match the exact bytes")
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise TypeError("Invalid Docker image metadata record")
        return record, raw

    digest = image_id
    record, raw = read_verified(digest)
    if "manifests" in record:
        if record.get("schemaVersion") != 2 or not isinstance(record["manifests"], list):
            raise ValueError("Invalid Docker image index")
        native = [
            entry["digest"]
            for entry in record["manifests"]
            if isinstance(entry, dict)
            and entry.get("mediaType") in MANIFEST_TYPES
            and (entry.get("platform") or {}).get("os") == "linux"
            and (entry.get("platform") or {}).get("architecture") == "amd64"
            and (entry.get("annotations") or {}).get("vnd.docker.reference.type")
            != "attestation-manifest"
        ]
        if len(native) != 1:
            raise ValueError("Docker image index must include one runnable linux/amd64 manifest")
        digest = native[0]
        record, raw = read_verified(digest)
        if record.get("schemaVersion") != 2 or "layers" not in record:
            raise ValueError("Invalid Docker platform image manifest")
    if record.get("schemaVersion") == 2 and "layers" in record:
        digest = record["config"]["digest"]
        record, raw = read_verified(digest)
    if (
        record.get("os") != "linux"
        or record.get("architecture") != "amd64"
        or not isinstance(record.get("rootfs"), dict)
        or record["rootfs"].get("type") != "layers"
        or not isinstance(record["rootfs"].get("diff_ids"), list)
    ):
        raise ValueError("Docker image configuration must describe a linux/amd64 root filesystem")
    return digest, raw


def export_config(release: Path, image_id: str) -> str:
    print("Verifying the native Docker image configuration…", flush=True)
    try:
        config_id, config = resolve_image_config(image_id, build_history_blob)
    except subprocess.CalledProcessError:
        print(
            "Buildx metadata is unavailable. Exporting the Docker image for verification. "
            "Large images may take several minutes and require temporary disk space.",
            flush=True,
        )
        with tempfile.TemporaryDirectory(dir=release.parent) as temporary:
            archive = Path(temporary) / "image.tar"
            command("docker", "image", "save", "--output", str(archive), image_id)
            with tarfile.open(archive, "r:") as saved:

                def archive_blob(digest: str) -> bytes:
                    try:
                        return read_tar_record(saved, f"blobs/sha256/{digest[7:]}")
                    except KeyError:
                        if digest != image_id:
                            raise
                    # Older classic stores write the config at the path named in manifest.json.
                    entries = json.loads(read_tar_record(saved, "manifest.json"))
                    if len(entries) != 1:
                        raise ValueError("Docker export did not select one image")
                    return read_tar_record(saved, entries[0]["Config"])

                config_id, config = resolve_image_config(image_id, archive_blob)
    (release / "image-config.json").write_bytes(config)
    return config_id


def read_tar_record(archive: tarfile.TarFile, name: str) -> bytes:
    entry = archive.getmember(name)
    if not entry.isreg() or not 0 < entry.size <= RECORD_LIMIT:
        raise ValueError("Invalid Docker image export record")
    stream = archive.extractfile(entry)
    if stream is None:
        raise ValueError("Missing Docker image export record")
    with stream:
        return stream.read()


def need_host(accelerator: str, *, build: bool, automatic_inputs: bool) -> None:
    if sys.version_info < (3, 12):
        raise ValueError("Python 3.12 or later is required on the host")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Run this command on Linux x86_64 or in Windows WSL 2 x86_64")
    tools = ["docker"]
    if build and accelerator == "cuda":
        tools.append("make")
    if automatic_inputs or (build and accelerator == "cuda"):
        tools.append("uv")
    for tool in tools:
        if not shutil.which(tool):
            raise ValueError(f"Required executable is missing: {tool}")
    if not (ROOT / ".env").is_file():
        raise ValueError("Set up the stable configuration file .env before running this command")
    # Freeze the effective daemon choice before querying or modifying it. The CLI
    # gives an explicit context priority over DOCKER_HOST.
    endpoint = os.environ.get("DOCKER_HOST")
    context = os.environ.get("DOCKER_CONTEXT")
    if context or not endpoint:
        context = context or command("docker", "context", "show", capture=True).strip()
        endpoint = command(
            "docker", "context", "inspect", context, "--format", "{{.Endpoints.docker.Host}}",
            capture=True,
        ).strip()
        os.environ["DOCKER_CONTEXT"] = context
    if not endpoint or not endpoint.startswith("unix:///"):
        raise ValueError("Use a local Docker engine with host bind mounts and GPU access")
    engine = command(
        "docker", "info", "--format", "{{.OSType}}/{{.Architecture}}", capture=True
    ).strip()
    if engine not in {"linux/x86_64", "linux/amd64"}:
        raise ValueError("Docker must use a Linux x86_64 engine")
    command("docker", "compose", "version", capture=True)
    if build:
        command("docker", "buildx", "version", capture=True)


def bootstrap(
    accelerator: str,
    inputs: Path | None,
    *,
    restart: bool = False,
    reuse_native_image: str | None = None,
    upgrade: bool = False,
    recover_stopped_ocr: bool = False,
    defer_build: bool = False,
) -> None:
    # An upgrade may keep its exact admitted native image and need only Compose.
    prepare_native = not (restart or defer_build or upgrade)
    need_host(accelerator, build=prepare_native, automatic_inputs=prepare_native and inputs is None)
    deployment = ROOT / "server_data" / "ocr-source"
    state = deployment / accelerator
    for directory in (deployment, state):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        details = directory.lstat()
        if (
            directory.resolve() != directory
            or details.st_uid != os.getuid()
            or details.st_mode & 0o077
        ):
            raise ValueError("Source deployment state must be a private operator-owned directory")
    compose = source_compose(accelerator)
    if (
        accelerator == "rocm"
        and not restart
        and not upgrade
        and not defer_build
        and "docker-compose.rocm-wsl.yml" not in compose
    ):
        select_rocm_render_node(compose)
    lock = deployment / "deploy.lock"
    with lock.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if defer_build:
            start_deferred(compose, state, accelerator, inputs, reuse_native_image=reuse_native_image)
            return
        deploy(
            compose,
            state,
            accelerator,
            inputs,
            restart=restart,
            reuse_native_image=reuse_native_image,
            upgrade=upgrade,
            recover_stopped_ocr=recover_stopped_ocr,
        )


def gpu_args(accelerator: str, native: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    if accelerator == "cuda":
        reservations = native["deploy"]["resources"]["reservations"]["devices"]
        if len(reservations) != 1:
            raise ValueError("Select exactly one CUDA device reservation")
        ids = reservations[0]["device_ids"]
        if len(ids) != 1 or not re.fullmatch(r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+)", ids[0]):
            raise ValueError("Select one CUDA device by numeric index or GPU UUID")
        return ["--gpus", f"device={ids[0]}"], {"OCR_CUDA_DEVICE_ID": ids[0]}

    entries = native.get("devices", [])
    devices = {entry["target"]: entry["source"] for entry in entries}
    if devices.get("/dev/dxg") == "/dev/dxg":
        if len(entries) != 1 or not Path("/dev/dxg").is_char_device():
            raise ValueError("WSL 2 ROCm requires the /dev/dxg character device")
        index = native["environment"]["HIP_VISIBLE_DEVICES"]
        if not re.fullmatch(r"(?:0|[1-9][0-9]{0,3})", index) or int(index) > 1024:
            raise ValueError("Select one AMD WSL GPU by numeric index in OCR_ROCM_DEVICE_ID")
        library = "/usr/lib/wsl/lib/libdxcore.so"
        if not Path(library).is_file():
            raise ValueError(f"Missing Windows WSL driver library: {library}")
        return [
            "--device", "/dev/dxg:/dev/dxg",
            "--mount", f"type=bind,source={library},target=/usr/lib/libdxcore.so,readonly",
            "--env", f"HIP_VISIBLE_DEVICES={index}",
        ], {"OCR_ROCM_DEVICE_ID": index}

    kfd = "/dev/kfd"
    render = next((target for target in devices if target != kfd), "")
    if (
        len(entries) != 2
        or devices.get(kfd) != kfd
        or devices.get(render) != render
        or not re.fullmatch(r"/dev/dri/renderD[0-9]+", render)
    ):
        raise ValueError("ROCm requires /dev/kfd and one selected DRI render node")
    vendor = Path("/sys/class/drm") / Path(render).name / "device/vendor"
    if vendor.is_file() and vendor.read_text().strip().lower() != "0x1002":
        raise ValueError(f"Selected DRI render node {render} is not an AMD device")
    arguments: list[str] = []
    groups: list[str] = []
    for device in (kfd, render):
        details = Path(device).stat()
        if not stat.S_ISCHR(details.st_mode):
            raise ValueError(f"ROCm device is not a character device: {device}")
        group = str(details.st_gid)
        groups.append(group)
        arguments.extend(("--device", f"{device}:{device}", "--group-add", group))
    return arguments, {
        "OCR_RENDER_NODE": render,
        "OCR_VIDEO_GID": groups[0],
        "OCR_RENDER_GID": groups[1],
    }


def reject_other_profile(state: Path, accelerator: str, project: str) -> None:
    other = "cuda" if accelerator == "rocm" else "rocm"
    for filename, field in (
        ("compose.local.json", "x-vision-bill-source"),
        ("deferred.app.json", "x-vision-bill-deferred"),
    ):
        record = state.parent / other / filename
        if record.exists():
            trusted_file(record)
            identity = json.loads(record.read_bytes())[field]
            if identity["project"] == project:
                raise ValueError("This Compose project is already prepared for another accelerator")


def rocm_render_target(render: Path, topology: Path = KFD_TOPOLOGY) -> str:
    """Read the native Linux KFD architecture for the selected DRM minor."""
    try:
        minor = os.minor(render.stat().st_rdev)
        versions = []
        for properties in sorted(topology.glob("[0-9]*/properties")):
            values = dict(line.split(maxsplit=1) for line in properties.read_text().splitlines())
            if values.get("drm_render_minor") == str(minor) and values.get("vendor_id") == "4098":
                versions.append(int(values["gfx_target_version"]))
        if len(versions) != 1 or versions[0] <= 0:
            raise ValueError("Missing or ambiguous KFD GPU topology")
        major, remainder = divmod(versions[0], 10_000)
        minor_version, stepping = divmod(remainder, 100)
        if not 1 <= major <= 99 or minor_version > 15 or stepping > 15:
            raise ValueError("Invalid KFD GPU architecture")
        return f"gfx{major}{minor_version:x}{stepping:x}"
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Cannot determine the GFX target for {render}. Select the render node "
            "and set OCR_AMDGPU_TARGETS to its actual native architecture."
        ) from exc


def source_compose(accelerator: str) -> tuple[str, ...]:
    compose: tuple[str, ...] = (
        "docker",
        "compose",
        "--profile",
        "ocr",
        "-f",
        "docker-compose.yml",
        "-f",
        f"docker-compose.{accelerator}.yml",
    )
    if accelerator == "rocm" and "microsoft" in platform.release().lower():
        compose += ("-f", "docker-compose.rocm-wsl.yml")
    return compose


def select_rocm_render_node(compose: tuple[str, ...], topology: Path = KFD_TOPOLOGY) -> None:
    """Select the only compute-capable AMD render node when Compose has no explicit choice."""
    environment = command(*compose, "config", "--environment", capture=True)
    if any(
        line.startswith("OCR_RENDER_NODE=") and line.partition("=")[2]
        for line in environment.splitlines()
    ):
        return
    nodes = set()
    for properties in topology.glob("[0-9]*/properties"):
        values = dict(line.split(maxsplit=1) for line in properties.read_text().splitlines())
        minor = values.get("drm_render_minor", "")
        if (
            values.get("vendor_id") == "4098"
            and minor.isdecimal()
            and int(minor) >= 128
            and int(values.get("gfx_target_version", "0")) > 0
        ):
            nodes.add(f"/dev/dri/renderD{int(minor)}")
    if not nodes:
        return
    if len(nodes) != 1:
        raise ValueError(
            f"Multiple AMD render nodes: {', '.join(sorted(nodes))}. Set OCR_RENDER_NODE."
        )
    os.environ["OCR_RENDER_NODE"] = nodes.pop()


def wsl_rocm_target(index: str) -> str:
    # rocminfo enumerates HSA agents; its ordering is not a HIP device index.
    # Only a sole GPU has an unambiguous mapping. All other cases need a manual target.
    help_text = (
        "Set OCR_AMDGPU_TARGETS to the selected GPU's exact GFX architecture "
        "and OCR_ROCM_DEVICE_ID to its HIP index."
    )
    binary = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    if index != "0" or os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise ValueError(help_text)
    environment = dict(os.environ)
    for mask in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        environment.pop(mask, None)
    try:
        result = subprocess.run(
            [binary], check=True, capture_output=True, text=True, timeout=10, env=environment
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"WSL GPU identity unavailable. {help_text}") from exc
    agents = re.split(r"(?m)^\s*Agent [0-9]+\s*$", result.stdout)[1:]
    gpus = [agent for agent in agents if re.search(r"(?m)^\s*Device Type:\s*GPU\s*$", agent)]
    if len(gpus) == 1:
        name = re.search(r"(?m)^\s*Name:\s*(\S+)\s*$", gpus[0])
        vendor = re.search(r"(?m)^\s*Vendor Name:\s*AMD\s*$", gpus[0])
        if vendor and name and re.fullmatch(r"gfx[0-9a-f]{3,8}", name[1]):
            return name[1]
    raise ValueError(f"Cannot map WSL HIP device {index} to one AMD GPU. {help_text}")


def configure_rocm_build(candidate: dict[str, Any], devices: dict[str, str]) -> None:
    args = candidate["build"]["args"]
    native_transport = "wsl" if "OCR_ROCM_DEVICE_ID" in devices else "kfd"
    transport = args.get("OCR_ROCM_TRANSPORT") or native_transport
    if transport not in {"wsl", "kfd", "dual"}:
        raise ValueError("OCR_ROCM_TRANSPORT must be kfd, wsl or dual")
    if devices and transport not in {native_transport, "dual"}:
        raise ValueError(f"Selected device requires OCR_ROCM_TRANSPORT={native_transport} or dual")
    configured = args.get("AMDGPU_TARGETS")
    profile = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "paddleocr-service/ocr_service/targets.py")
    )
    if configured:
        targets = profile["parse_targets"](configured)
    else:
        if native_transport == "kfd":
            detected = rocm_render_target(Path(devices["OCR_RENDER_NODE"]))
        else:
            detected = wsl_rocm_target(devices["OCR_ROCM_DEVICE_ID"])
        if detected not in profile["DEFAULT_TARGETS"]:
            raise ValueError(
                f"{detected} is outside the local Radeon candidate list. "
                "Set OCR_AMDGPU_TARGETS explicitly for an experimental build."
            )
        targets = [detected]
    os.environ.update(OCR_AMDGPU_TARGETS=";".join(targets), OCR_ROCM_TRANSPORT=transport)
    print(f"Local ROCm build: {transport}, {', '.join(targets)}", flush=True)


def cuda_build_channel(
    compose: tuple[str, ...],
    candidate: dict[str, Any],
    previous: dict[str, Any] | None,
    saved_config: Path | None = None,
) -> str:
    channel = os.environ.get("OCR_CUDA_CHANNEL")
    if previous is not None and not channel:
        # The resolved build argument conflates an explicit .env selection with
        # the overlay's default. Only the former overrides an installed channel.
        environment = command(*compose, "config", "--environment", capture=True)
        channel = next(
            (
                line.removeprefix("OCR_CUDA_CHANNEL=")
                for line in environment.splitlines()
                if line.startswith("OCR_CUDA_CHANNEL=")
            ),
            None,
        ) or previous.get("cuda_channel")
        if not channel and saved_config is not None:
            # BuildKit's RUN history records the actual argument used for the
            # wheel check/install. An ARG default alone does not establish it.
            history = json.loads(saved_config.read_bytes()).get("history", [])
            installed_channels = set()
            for layer in history:
                instruction = layer.get("created_by", "")
                if (
                    'resolve_cuda.py --check /opt/locks/requirements.cuda.lock --channel "$OCR_CUDA_CHANNEL"'
                    in instruction
                    and "pip install --require-hashes" in instruction
                ):
                    match = re.match(
                        r"RUN \|1 OCR_CUDA_CHANNEL=(cu126|cu129) /bin/sh -c ", instruction
                    )
                    if match:
                        installed_channels.add(match[1])
            if len(installed_channels) == 1:
                channel = installed_channels.pop()
        if not channel:
            # Older source records omit the channel. Read the lock from the immutable
            # image actually admitted, independently of the current checkout lock.
            image_id = previous["image_ids"][NATIVE]
            if not IMAGE_ID.fullmatch(image_id):
                raise ValueError("Invalid original native image ID")
            try:
                channel = command(
                    "docker",
                    "run",
                    "--rm",
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "--entrypoint",
                    "python",
                    image_id,
                    "-c",
                    "from pathlib import Path\n"
                    "from runpy import run_path\n"
                    "validate = run_path('/opt/ocr/build/resolve_cuda.py')['validate_lock']\n"
                    "print(validate(Path('/opt/locks/requirements.cuda.lock')).get('cuda_channel', 'cu126'))",
                    capture=True,
                ).strip()
            except subprocess.CalledProcessError as exc:
                raise ValueError(
                    "Cannot read the previously installed CUDA channel from the saved metadata "
                    "or original image. Set OCR_CUDA_CHANNEL=cu126 or cu129 to the intended "
                    "wheel channel and retry --upgrade."
                ) from exc
    if previous is None:
        channel = (
            channel or candidate.get("build", {}).get("args", {}).get("OCR_CUDA_CHANNEL") or "cu129"
        )
    if channel not in ("cu126", "cu129"):
        raise ValueError("OCR_CUDA_CHANNEL must be cu126 or cu129")
    return channel


def existing_controllers(compose: tuple[str, ...]) -> list[str]:
    return command(*compose, "ps", "--all", "--quiet", NATIVE, capture=True).split()


def claim_origin(identity: dict[str, Any], release: Path) -> dict[str, Any]:
    """Keep the first generation's daemon ownership label across in-place upgrades."""
    if "claim_origin" in identity:
        origin: dict[str, Any] = identity["claim_origin"]
        if not re.fullmatch(r"release-[a-zA-Z0-9_-]+", origin["release"]):
            raise ValueError("Invalid original release directory")
        return origin
    return {
        "release": release.name,
        "bundle_sha256": identity["bundle_sha256"],
        "image_ids": identity["image_ids"],
    }


def project_claim(
    identity: dict[str, Any], accelerator: str, release: Path
) -> tuple[str, str, str]:
    project: str = identity["project"]
    name = "vision-bill-source-claim-" + hashlib.sha256(project.encode()).hexdigest()
    label = "io.vision-bill.source-owner"
    origin = claim_origin(identity, release)
    generation = {
        "checkout": str(ROOT),
        "accelerator": accelerator,
        "bundle_sha256": origin["bundle_sha256"],
        "image_ids": origin["image_ids"],
        "release": str(release.with_name(origin["release"])),
    }
    owner = hashlib.sha256(json.dumps(generation, sort_keys=True).encode()).hexdigest()
    return name, label, owner


def has_project_claim(identity: dict[str, Any], accelerator: str, release: Path) -> bool:
    return volume_has_claim(*project_claim(identity, accelerator, release))


def volume_has_claim(name: str, label: str, owner: str) -> bool:
    try:
        records = json.loads(command("docker", "volume", "inspect", name, capture=True))
    except subprocess.CalledProcessError:
        return False
    return len(records) == 1 and (records[0].get("Labels") or {}).get(label) == owner


# No GPU, model state or previous image is needed to inspect the original volume.
CHECK_UPGRADE_HANDOFF = """
import sys
from pathlib import Path
from ocr_service.initialization_lock import inspect_stopped_handoff
print(inspect_stopped_handoff(Path(sys.argv[1])))
"""


def verify_upgrade_handoff(
    config: dict[str, Any], previous: dict[str, Any], *, recover_stopped_ocr: bool = False
) -> None:
    """Check the stopped source project's original coordination volume before rebuilding."""
    volume = "ocr-coordination"
    lock = "/coord/gpu-handoff.lock"
    for name, key in (
        ("vision_bill", "OCR_ACTIVATION_HANDOFF_LOCK_PATH"),
        (NATIVE, "OCR_HANDOFF_LOCK_PATH"),
    ):
        service = config["services"][name]
        mounts = [m for m in service.get("volumes", []) if m.get("target") == "/coord"]
        if (
            service.get("environment", {}).get(key) != lock
            or len(mounts) != 1
            or mounts[0].get("type") != "volume"
            or mounts[0].get("source") != volume
            or mounts[0].get("volume", {}).get("subpath")
        ):
            raise ValueError("Cannot identify the source deployment's original GPU handoff mount")
    actual = config.get("volumes", {}).get(volume, {}).get("name")
    if not isinstance(actual, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", actual):
        raise ValueError("Missing original source coordination volume")

    inspector = importlib.import_module(
        ".ocr_stopped_recovery" if __package__ else "ocr_stopped_recovery", package=__package__
    )
    program = inspector.inspection_program(CHECK_UPGRADE_HANDOFF)

    def inspect_journals() -> str:
        if compose_volumes(previous["project"], volume) != [actual]:
            raise ValueError("Cannot uniquely identify the original project's coordination volume")
        try:
            verdict = command(
                "docker",
                "run",
                "--rm",
                "--pull=missing",
                "--platform",
                NATIVE_PLATFORM,
                "--network=none",
                "--read-only",
                "--user=0:0",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--mount",
                f"type=volume,source={actual},target=/coord,readonly,volume-nocopy",
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=1m",
                "--entrypoint",
                "python",
                JOURNAL_INSPECTOR,
                "-B",
                "-c",
                program,
                lock,
                capture=True,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"Could not run the pinned read-only handoff inspector on Docker volume {actual}. "
                f"Check access to {JOURNAL_INSPECTOR} and the coordination volume. "
                f"Docker reported: {(exc.stderr or '').strip()}"
            ) from exc
        if compose_volumes(previous["project"], volume) != [actual]:
            raise ValueError("The original coordination volume changed during the upgrade check")
        return verdict

    verdict = inspect_journals()
    if recover_stopped_ocr and verdict == "pending gpu-handoff.ocr-only.json":
        inspector.recover_stopped_ocr(config, previous, run=command, helper_image=JOURNAL_INSPECTOR)
        verdict = inspect_journals()
    if verdict == "busy":
        raise ValueError("The original GPU handoff lock is held; drain the owning process first.")
    if verdict == "pending gpu-handoff.ocr-only.json":
        raise ValueError(
            "Outstanding work in gpu-handoff.ocr-only.json. On the original stack switch "
            "Dedicated OCR off in Settings and wait for Disabled, or use --upgrade "
            "--recover-stopped-ocr to verify and settle the stopped local OCR owner."
        )
    if verdict == "pending gpu-handoff.json":
        raise ValueError(
            "Outstanding shared GPU handoff in gpu-handoff.json; recover the original "
            "coordinator and its recorded resources before upgrading. Preserve the journals."
        )
    if verdict != "clean":
        raise ValueError(
            f"The GPU handoff journal failed validation ({verdict}); "
            "preserve the coordination volume and recover its ownership state before upgrading."
        )


def acquire_project_claim(identity: dict[str, Any], accelerator: str, release: Path) -> None:
    acquire_volume_claim(*project_claim(identity, accelerator, release), project=identity["project"])


def acquire_volume_claim(name: str, label: str, owner: str, *, project: str) -> None:
    # Docker serializes creation by name and reuses an existing local volume
    # without replacing its creation labels, including during concurrent calls.
    command(
        "docker",
        "volume",
        "create",
        "--driver",
        "local",
        "--label",
        f"{label}={owner}",
        name,
        capture=True,
    )
    if not volume_has_claim(name, label, owner):
        raise ValueError(f"Compose project {project!r} is already claimed by another source deployment")


def bind_sources(
    record: dict[str, Any], targets: dict[str, str], *, writable: bool
) -> dict[str, str]:
    return {
        mount["Destination"]: mount["Source"]
        for mount in record["Mounts"]
        if mount.get("Type") == "bind"
        and mount.get("RW") is writable
        and mount["Destination"] in targets
    }


def bind_sources_match(expected: dict[str, str], observed: dict[str, str], image: str) -> bool:
    if expected == observed:
        return True
    if (
        "microsoft" not in platform.release().lower()
        or expected.keys() != observed.keys()
        or not IMAGE_ID.fullmatch(image)
        or any(
            observed[target] != source
            and not observed[target].startswith(
                "/run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/"
            )
            for target, source in expected.items()
        )
    ):
        return False
    mounts: list[str] = []
    for target, source in expected.items():
        if any(char in source for char in ',"\n\r'):
            return False
        mounts.extend(("--mount", f"type=bind,source={source},target={target}"))
    # Docker Desktop translates WSL source paths at the daemon boundary. An
    # unstarted container proves the exact translation without running the image.
    probe = command(
        "docker",
        "create",
        "--pull",
        "never",
        "--network",
        "none",
        *mounts,
        "--entrypoint",
        "/bin/true",
        image,
        capture=True,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", probe):
        raise ValueError("Cannot verify the Docker Desktop bind-probe container ID")
    try:
        records = json.loads(command("docker", "container", "inspect", probe, capture=True))
        return len(records) == 1 and bind_sources(records[0], expected, writable=True) == observed
    finally:
        command("docker", "container", "rm", probe, capture=True)


def compose_volumes(project: str, volume: str) -> list[str]:
    return command(
        "docker",
        "volume",
        "ls",
        "--quiet",
        "--filter",
        f"label=com.docker.compose.project={project}",
        "--filter",
        f"label=com.docker.compose.volume={volume}",
        capture=True,
    ).split()


def verify_existing_app(
    compose: tuple[str, ...], project: str, *, allow_orphan: bool = False
) -> None:
    containers = command(*compose, "ps", "--all", "--quiet", "vision_bill", capture=True).split()
    if not containers:
        db = command(*compose, "ps", "--all", "--quiet", "db", capture=True).split()
        if not allow_orphan and (db or compose_volumes(project, "db_data")):
            raise ValueError(
                "Cannot verify ownership of the existing database volume. Start the base app in its original checkout or use a fresh project"
            )
        if db:
            records = json.loads(command("docker", "container", "inspect", *db, capture=True))
            if len(records) != 1 or (records[0]["Config"].get("Labels") or {}).get(
                "com.docker.compose.project.working_dir"
            ) != str(ROOT):
                raise ValueError(
                    "Cannot verify existing database checkout; keep its original deployment"
                )
        return
    records = json.loads(command("docker", "container", "inspect", *containers, capture=True))
    config = json.loads(
        command(*compose, "config", "--no-env-resolution", "--format", "json", capture=True)
    )
    configured = config["services"]["vision_bill"].get("volumes", [])
    expected = {
        f"/app/{name}": str(ROOT / "server_data" / name)
        for name in ("logs", "uploads", "uploads_tmp", "config")
    }
    current = {
        mount["target"]: mount["source"]
        for mount in configured
        if mount.get("type") == "bind"
        and not mount.get("read_only", False)
        and mount["target"] in expected
    }
    if (
        len(records) != 1
        or current != expected
        or any(Path(source).resolve() != Path(source) for source in expected.values())
    ):
        raise ValueError(
            "Cannot verify existing app checkout and data bind origins; keep its original deployment"
        )
    app = records[0]
    labels = app["Config"].get("Labels") or {}
    actual = bind_sources(app, expected, writable=True)
    if labels.get("com.docker.compose.project.working_dir") != str(ROOT) or not bind_sources_match(
        expected, actual, app.get("Image", "")
    ):
        raise ValueError(
            "Cannot verify existing app checkout and data bind origins; keep its original deployment"
        )


def reject_existing_ocr(compose: tuple[str, ...], project: str) -> None:
    if existing_controllers(compose) or compose_volumes(project, "ocr-activation-request"):
        raise ValueError("Existing OCR activation state must stay with its original deployment")


def verify_prepared_release(overlay: dict[str, Any], state: Path, accelerator: str) -> None:
    release = Path(overlay["services"][NATIVE]["volumes"][0]["source"])
    details = release.lstat()
    if not stat.S_ISDIR(details.st_mode) or release.parent != state or details.st_mode & 0o077:
        raise ValueError("Prepared release must be a private directory inside its deployment")
    identity = overlay["x-vision-bill-source"]
    config_digest = identity.get("image_config_sha256", identity["image_ids"][NATIVE][7:])
    config_id = f"sha256:{config_digest}"
    if (
        not IMAGE_ID.fullmatch(config_id)
        or overlay["services"][NATIVE]["environment"]["OCR_IMAGE_ID"] != config_id
    ):
        raise ValueError("Prepared release has an inconsistent native image configuration ID")
    for filename, digest in (
        ("bundle.json", identity["bundle_sha256"]),
        ("image-config.json", config_digest),
    ):
        path = release / filename
        if trusted_file(path).st_size > RECORD_LIMIT or path_digest(path) != digest:
            raise ValueError(f"Prepared release file identity changed: {filename}")
    stage_approved_inputs(release, None, accelerator)


def was_admitted(managed: Path) -> bool:
    marker = managed.with_name("admitted.sha256")
    if not marker.exists():
        return False
    if trusted_file(marker).st_size not in (64, 129):
        return False
    return path_digest(managed) in marker.read_text().splitlines()


def record_admission(managed: Path, replacement: Path | None = None) -> None:
    marker = managed.with_name("admitted.sha256")
    pending = marker.with_suffix(".pending")
    admitted = path_digest(managed)
    if replacement is not None:
        # Both records are ours across the atomic change of compose.local.json.
        admitted += "\n" + path_digest(replacement)
    pending.write_text(admitted)
    pending.replace(marker)


READ_DEFERRED_REQUEST = """
from ocr_contract.activation import ActivationStore
store = ActivationStore()
request = store.read_request()
if request is not None and store.runtime_enabled(request):
    print(request.generation)
"""


def deferred_app_container(compose: tuple[str, ...], app: DeferredApp) -> str | None:
    containers = command(*compose, "ps", "--quiet", "vision_bill", capture=True).split()
    if not containers:
        return None
    if len(containers) != 1 or not re.fullmatch(r"[0-9a-f]{12,64}", containers[0]):
        raise ValueError("Cannot identify the deferred application container")
    container = containers[0]
    image, health = command(
        "docker", "container", "inspect", "--format",
        "{{.Image}} {{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}",
        container, capture=True,
    ).split()
    if image != app.image_id:
        raise ValueError("The deferred application image changed")
    return container if health == "healthy" else None


def deferred_generation(compose: tuple[str, ...], app: DeferredApp) -> str | None:
    container = deferred_app_container(compose, app)
    if container is None:
        return None
    generation = command(
        "docker", "exec", container, "python", "-c", READ_DEFERRED_REQUEST, capture=True,
    ).strip()
    if not generation:
        return None
    try:
        if str(UUID(generation)) == generation:
            return generation
    except ValueError:
        pass
    raise ValueError("Invalid deferred OCR activation generation")


def start_deferred(
    compose: tuple[str, ...], state: Path, accelerator: str, inputs: Path | None,
    *, reuse_native_image: str | None = None,
) -> None:
    deferred = importlib.import_module(
        ".ocr_deferred" if __package__ else "ocr_deferred", package=__package__,
    )
    managed = state / "compose.local.json"
    if managed.exists():
        deploy(compose, state, accelerator, None, restart=True)
        return
    config = json.loads(command(*compose, "config", "--no-env-resolution", "--format", "json", capture=True))
    project = config["name"]
    compose = (*compose, "--project-name", project)
    reject_other_profile(state, accelerator, project)
    revision = command("git", "rev-parse", "HEAD", capture=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("Cannot identify the source checkout")
    status_dir = state / "native-build-status"
    status_dir.mkdir(mode=0o700, exist_ok=True)
    details = status_dir.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid() or details.st_mode & 0o077:
        raise ValueError("Deferred build status must be in a private operator-owned directory")
    record = state / "deferred.app.json"
    saved_app: DeferredApp | None = None
    project_hash = hashlib.sha256(project.encode()).hexdigest()
    owner = hashlib.sha256(f"{ROOT}\n{accelerator}".encode()).hexdigest()
    claim = (f"vision-bill-deferred-claim-{project_hash}", "io.vision-bill.source-owner", owner)
    if record.exists():
        if trusted_file(record).st_size > 8192:
            raise ValueError("Deferred application record is too large")
        saved = json.loads(record.read_bytes())
        identity = saved["x-vision-bill-deferred"]
        saved_app = DeferredApp(project, identity["image_id"], status_dir, identity["revision"])
        if saved != saved_app.overlay() or existing_controllers(compose):
            raise ValueError("The deferred application project or controller changed")
        if inspect_image(app_image_service(saved_app.image_id)["image"]) != saved_app.image_id:
            raise ValueError("The deferred application image changed")
        if not volume_has_claim(*claim):
            raise ValueError("Cannot verify the previously deferred Compose project on this Docker engine")
    else:
        reject_existing_ocr(compose, project)
    verify_existing_app(compose, project, allow_orphan=saved_app is not None)
    if saved_app is None:
        acquire_volume_claim(*claim, project=project)
    if saved_app is not None and saved_app.revision == revision:
        app = saved_app
    else:
        with build_phase("app_image_build"):
            command(*compose, "build", "vision_bill")
        image_id = inspect_image(config["services"]["vision_bill"]["image"])
        app = DeferredApp(project, image_id, status_dir, revision)
        overlay = app.overlay()
        command("docker", "image", "tag", image_id, overlay["services"]["vision_bill"]["image"])
        pending = record.with_suffix(".pending")
        pending.write_text(json.dumps(overlay))
        pending.replace(record)

    status_path = status_dir / "status.json"
    # Only this launcher can publish the native record or start the controller.
    deferred.initialize_status(status_path, interrupted_safe=True)
    app_compose = (*compose, "-f", str(record))
    command(*app_compose, "up", "--detach", "--no-build", "--wait", "--wait-timeout", "300", "vision_bill", "db")
    if deferred_app_container(app_compose, app) is None:
        raise ValueError("Vision Bill did not reach readiness before OCR preparation")
    print("Vision Bill and PostgreSQL are ready. Waiting for Dedicated OCR in Settings.", flush=True)

    def build_native() -> None:
        try:
            if command("git", "rev-parse", "HEAD", capture=True).strip() != app.revision:
                raise ValueError("The checkout revision changed while waiting for OCR")
            need_host(accelerator, build=True, automatic_inputs=inputs is None)
            if accelerator == "rocm" and "docker-compose.rocm-wsl.yml" not in compose:
                select_rocm_render_node(compose)
            deploy(
                compose, state, accelerator, inputs,
                reuse_native_image=reuse_native_image, deferred_app=app,
            )
        except Exception as exc:
            try:
                handoff_possible = managed.exists() or bool(existing_controllers(compose))
            except (OSError, subprocess.CalledProcessError):
                handoff_possible = True
            if handoff_possible:
                raise deferred.DeferredHandoffError(
                    "The native controller may have started. Inspect the original project and use --restart."
                ) from exc
            raise

    deferred.wait_for_ocr(status_path, lambda: deferred_generation(app_compose, app), build_native)


def deploy(
    compose: tuple[str, ...],
    state: Path,
    accelerator: str,
    inputs: Path | None,
    *,
    restart: bool = False,
    reuse_native_image: str | None = None,
    upgrade: bool = False,
    recover_stopped_ocr: bool = False,
    deferred_app: DeferredApp | None = None,
) -> None:
    if recover_stopped_ocr and not upgrade:
        raise ValueError("--recover-stopped-ocr requires --upgrade")
    if reuse_native_image is not None and (
        restart
        or not re.fullmatch(
            rf"vision-bill-ocr-{accelerator}:source-build-[0-9a-f]{{32}}", reuse_native_image
        )
    ):
        raise ValueError("Reuse requires a local source-build tag for the selected accelerator")
    managed = state / "compose.local.json"
    if upgrade and restart:
        raise ValueError("Choose either --restart or --upgrade")
    if restart:
        trusted_file(managed)
        overlay = json.loads(managed.read_bytes())
        identity = overlay["x-vision-bill-source"]
        verify_prepared_release(overlay, state, accelerator)
        for service, expected_id in identity["image_ids"].items():
            actual_id = inspect_image(overlay["services"][service]["image"])
            if actual_id != expected_id:
                raise ValueError(f"Managed source image changed for {service}")
        reject_other_profile(state, accelerator, identity["project"])
        os.environ.update(identity["device_environment"])
        selected = (*compose, "--project-name", identity["project"], "-f", str(managed))
        release = Path(overlay["services"][NATIVE]["volumes"][0]["source"])
        project = identity["project"]
        admitted = was_admitted(managed) and has_project_claim(identity, accelerator, release)
        verify_existing_app(selected, project, allow_orphan=admitted)
        acquire_project_claim(identity, accelerator, release)
        verify_existing_app(selected, project, allow_orphan=admitted)
        containers = existing_controllers(selected)
        if containers:
            records = json.loads(
                command("docker", "container", "inspect", *containers, capture=True)
            )
            if len(records) != 1 or records[0]["Image"] != identity["image_ids"][NATIVE]:
                raise ValueError("The original Compose project contains a different OCR image")
            expected = {"/release": str(release)}
            actual = bind_sources(records[0], expected, writable=False)
            if not bind_sources_match(expected, actual, identity["image_ids"][NATIVE]):
                raise ValueError(
                    "The original Compose project contains a different OCR release bind"
                )
        record_admission(managed)
        start(*selected)
        return
    previous: dict[str, Any] | None = None
    previous_digest = ""
    if upgrade:
        trusted_file(managed)
        prior = json.loads(managed.read_bytes())
        verify_prepared_release(prior, state, accelerator)
        previous = prior["x-vision-bill-source"]
        old_release = Path(prior["services"][NATIVE]["volumes"][0]["source"])
        if not was_admitted(managed) or not has_project_claim(previous, accelerator, old_release):
            raise ValueError("Upgrade requires the original admitted Compose project")
        previous_digest = path_digest(managed)
        compose = (*compose, "--project-name", previous["project"])
        device_settings = dict(previous["device_environment"])
        # A previously implicit GPU override belongs to the last deployment. Explicit
        # selectors from this upgrade's shell or .env must describe the new candidate.
        selectors = {"OCR_ROCM_DEVICE_ID", "OCR_CUDA_DEVICE_ID", "OCR_RENDER_NODE"}
        environment = command(*compose, "config", "--environment", capture=True)
        for line in environment.splitlines():
            key, _, value = line.partition("=")
            if key in selectors and key in device_settings and value:
                device_settings[key] = value
        os.environ.update(device_settings)
        if command(*compose, "ps", "--all", "--quiet", "vision_bill", NATIVE, capture=True).strip():
            raise ValueError(
                "Drain jobs, then run docker compose down with the original overlays and keep all volumes"
            )
        if inputs is None:
            inputs = old_release
    elif managed.exists():
        raise ValueError(
            "Source OCR is already prepared. Use --restart for its installed images or --upgrade for a new build"
        )
    build_tag = f"source-build-{uuid4().hex}"
    os.environ.update({"VB_IMAGE_TAG": build_tag, "OCR_IMAGE_TAG": build_tag})
    config = json.loads(
        command(*compose, "config", "--no-env-resolution", "--format", "json", capture=True)
    )
    project = config["name"]
    reject_other_profile(state, accelerator, project)
    if previous is None:
        compose = (*compose, "--project-name", project)
        if deferred_app is None:
            reject_existing_ocr(compose, project)
        elif deferred_app.project != project or existing_controllers(compose):
            raise ValueError("The Compose project changed before starting deferred OCR")
    elif project != previous["project"]:
        raise ValueError("Cannot change Compose project during upgrade")
    verify_existing_app(compose, project, allow_orphan=upgrade)
    candidate = config["services"][NATIVE]
    devices, device_environment = gpu_args(accelerator, candidate)
    os.environ.update(device_environment)
    saved_target = None
    selection_error = None
    if accelerator == "rocm" and reuse_native_image is None:
        try:
            configure_rocm_build(candidate, device_environment)
        except ValueError as exc:
            # An admitted WSL installation may have been built with an explicit
            # GFX target on a host without rocminfo. Try its proven profile only
            # for reuse; a fresh build still requires an actual target selection.
            old_proof = (previous or {}).get("native_build", {})
            saved_target = old_proof.get("selected_args", {}).get("AMDGPU_TARGETS")
            args = candidate["build"]["args"]
            if (not saved_target or args.get("AMDGPU_TARGETS")
                    or "OCR_ROCM_DEVICE_ID" not in device_environment):
                raise
            resolved = {**candidate, "build": {**candidate["build"], "args": {**args, "AMDGPU_TARGETS": saved_target}}}
            configure_rocm_build(resolved, device_environment)
            selection_error = exc
    if accelerator == "cuda":
        channel = cuda_build_channel(
            compose, candidate, previous,
            old_release / "image-config.json" if previous is not None else None,
        )
        os.environ["OCR_CUDA_CHANNEL"] = channel
        cuda_probe = importlib.import_module(
            ".ocr_cuda_probe" if __package__ else "ocr_cuda_probe", package=__package__
        )
        expected_uuid = cuda_probe.preflight(device_environment["OCR_CUDA_DEVICE_ID"], channel)
        devices.extend(("--env", f"VISION_BILL_CUDA_SELECTION={expected_uuid}"))
        if previous is not None:
            # A managed selector may be an ordinal on WSL. Carry the original
            # hardware identity through the upgrade even if indexing changed.
            prior_uuid = prior["services"][NATIVE]["environment"].get("OCR_CUDA_EXPECTED_UUID")
            if prior_uuid and device_environment == previous["device_environment"]:
                devices.extend(("--env", f"OCR_CUDA_EXPECTED_UUID={prior_uuid}"))
    if previous is not None:
        verify_upgrade_handoff(config, previous, recover_stopped_ocr=recover_stopped_ocr)
    reuse = importlib.import_module(
        ".ocr_native_reuse" if __package__ else "ocr_native_reuse", package=__package__
    )
    selected_args = (
        {"OCR_CUDA_CHANNEL": channel} if accelerator == "cuda" else
        {"AMDGPU_TARGETS": os.environ.get("OCR_AMDGPU_TARGETS", ""),
         "OCR_ROCM_TRANSPORT": os.environ.get("OCR_ROCM_TRANSPORT", "")}
    )
    fingerprint = None
    auto_reuse = False
    if reuse_native_image is None:
        fingerprint = reuse.build_fingerprint(ROOT / NATIVE, accelerator, candidate.get("build", {}), selected_args)
        if previous is not None and previous.get("native_build"):
            admitted = reuse.admitted_candidate(
                previous, fingerprint, device_environment, old_release, checkout_sources(),
                prior["services"][NATIVE]["image"],
            )
            if admitted:
                try:
                    local_id = inspect_image(admitted)
                except subprocess.CalledProcessError:
                    local_id = ""
                if local_id == previous["image_ids"][NATIVE]:
                    reuse_native_image = admitted
                    auto_reuse = True
                    print("Reusing the admitted native OCR image for this application upgrade.", flush=True)
    if selection_error is not None and not auto_reuse:
        raise selection_error
    native_image = reuse_native_image or candidate["image"]
    image_id = ""
    expected_sources: dict[str, str] = {}
    if reuse_native_image is not None:
        image_id = inspect_image(native_image)
        expected_sources = checkout_sources()
        print(f"Checking the local {accelerator.upper()} OCR candidate {native_image}…", flush=True)
    release = Path(tempfile.mkdtemp(prefix="release-", dir=state))
    pending = release / "compose.local.json"
    try:
        if inputs is None:
            stage_first_run_inputs(release, accelerator)
        else:
            stage_approved_inputs(inputs, release, accelerator)
        cuda_verification: list[str] = []
        if accelerator == "cuda":
            lock = [
                "uv",
                "run",
                "--no-project",
                "--with",
                "packaging==26.2",
                "python",
                "paddleocr-service/build/resolve_cuda.py",
                "--check",
                "paddleocr-service/requirements.cuda.lock",
            ]
            if auto_reuse and previous is not None:
                # The admitted proof binds the originally validated lock to this
                # immutable image. BIND_BUNDLE checks its attestation and hash again.
                digest = previous["native_build"]["cuda_lock_sha256"]
            elif reuse_native_image is not None:
                lock.insert(2, "--offline")
                digest = command(
                    *lock, "--reuse-image", image_id, "--channel", channel, capture=True
                ).strip()
            else:
                context_args = [*lock, "--build-context", "--channel", channel]
                if previous is not None:
                    context_args.extend(("--prior-image", previous["image_ids"][NATIVE]))
                context = command(*context_args, capture=True).strip()
                os.environ["VISION_BILL_VERIFIED_CUDA_WHEEL_CONTEXT"] = context
                digest = path_digest(Path(context) / "requirements.cuda.lock")
            cuda_verification = [
                channel,
                digest,
                "/opt/locks/requirements.cuda.lock",
                "/opt/ocr/build/resolve_cuda.py",
            ]
        if reuse_native_image is None:
            with build_phase(f"{accelerator}_image_build"):
                if accelerator == "cuda":
                    command("make", "ocr-build-cuda")
                else:
                    command(*compose, "build", NATIVE)
            image_id = inspect_image(native_image, report_size=True)
        with build_phase("verify_image_config"):
            if auto_reuse and previous is not None:
                if image_id != previous["image_ids"][NATIVE]:
                    raise ValueError("Previously admitted native image tag changed")
                config_id = reuse.copy_proven_config(previous, old_release, release)
            else:
                config_id = export_config(release, image_id)
        print("Validating the native OCR packages and GPU selection…", flush=True)
        # The metadata probe has neither model cache/state mounts nor a network.
        bound = command(
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--platform",
            NATIVE_PLATFORM,
            "--network",
            "none",
            *devices,
            "--volume",
            f"{release}:/release:ro",
            "--entrypoint",
            "python",
            image_id,
            "-c",
            BIND_BUNDLE,
            image_id,
            config_id,
            "/release",
            json.dumps(expected_sources) if expected_sources else "",
            *cuda_verification,
            capture=True,
        )
        manifest = json.loads(bound)
        bound_device_uuid = manifest.pop(f"_bound_{accelerator}_uuid")
        if not isinstance(bound_device_uuid, str) or not re.fullmatch(r"[0-9a-f]{32}", bound_device_uuid):
            raise ValueError("Invalid bound native GPU UUID")
        if auto_reuse and previous is not None and bound_device_uuid != previous["native_build"]["device_uuid"]:
            raise ValueError("Selected GPU differs from the device admitted with this native image")
        if reuse_native_image is not None and (
            inspect_image(native_image) != image_id or manifest["source_files"] != expected_sources
        ):
            raise ValueError("Reused OCR image or its sources changed during verification")
        (release / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        command("docker", "image", "tag", image_id, image_tag(accelerator, image_id))
        if deferred_app is None:
            with build_phase("app_image_build"):
                command(*compose, "build", "vision_bill")
            app_id = inspect_image(config["services"]["vision_bill"]["image"])
        else:
            if deferred_app_container(compose, deferred_app) is None:
                raise ValueError("The deferred application is no longer ready")
            app_id = deferred_app.image_id
        overlay = managed_overlay(
            accelerator,
            release,
            image_id,
            app_id,
            project,
            device_environment,
            path_digest(release / "bundle.json"),
            config_id=config_id,
        )
        if fingerprint:
            if fingerprint != reuse.build_fingerprint(ROOT / NATIVE, accelerator, candidate.get("build", {}), selected_args):
                raise ValueError("Native build inputs changed during verification")
            overlay["x-vision-bill-source"]["native_build"] = {
                "inputs_sha256": fingerprint, "device_uuid": bound_device_uuid, "selected_args": selected_args,
            }
            if cuda_verification:
                overlay["x-vision-bill-source"]["native_build"]["cuda_lock_sha256"] = digest
        if cuda_verification:
            overlay["x-vision-bill-source"]["cuda_channel"] = channel
            overlay["services"][NATIVE]["environment"]["OCR_CUDA_EXPECTED_UUID"] = bound_device_uuid
        if previous is not None:
            overlay["x-vision-bill-source"]["claim_origin"] = claim_origin(previous, old_release)
        if deferred_app is not None:
            overlay["services"]["vision_bill"] = deferred_app.overlay()["services"]["vision_bill"]
        command("docker", "image", "tag", app_id, overlay["services"]["vision_bill"]["image"])
        pending.write_text(json.dumps(overlay))
        if previous is not None:
            if (
                not has_project_claim(previous, accelerator, old_release)
                or command(
                    *compose, "ps", "--all", "--quiet", "vision_bill", NATIVE, capture=True
                ).strip()
            ):
                raise ValueError("Original project changed or restarted during upgrade")
            # The detached build can outlast a stop/start/stop of the old stack.
            # Recheck both journals before admitting new images, without recovery.
            verify_upgrade_handoff(config, previous)
            record_admission(managed, pending)
        pending.replace(managed)
    except BaseException:
        if not managed.exists() or (upgrade and path_digest(managed) == previous_digest):
            shutil.rmtree(release)
        raise
    # A failed start can leave containers. Preserve the validated record so
    # restart can retry this exact image and the device selection.
    selected = (*compose, "-f", str(managed))
    acquire_project_claim(overlay["x-vision-bill-source"], accelerator, release)
    if (upgrade or deferred_app is not None) and existing_controllers(selected):
        raise ValueError("An OCR controller started during source preparation")
    if not upgrade and deferred_app is None:
        reject_existing_ocr(selected, project)
    verify_existing_app(selected, project, allow_orphan=upgrade)
    record_admission(managed)
    if deferred_app is None:
        start(*selected)
    else:
        command(*selected, "up", "--detach", "--no-build", "--no-deps", "--wait", "--wait-timeout", "300", NATIVE)


def inspect_image(image: str, *, report_size: bool = False) -> str:
    fmt = "{{.Id}} {{.Size}}" if report_size else "{{.Id}}"
    metadata = command("docker", "image", "inspect", "--format", fmt, image, capture=True).strip()
    image_id, _, size = metadata.partition(" ")
    if report_size and size.isdecimal():
        print(f"ocr_source_image id={image_id} image_size_bytes={size}", flush=True)
    return image_id


def start(*compose: str) -> None:
    command(*compose, "up", "--detach", "--no-build", "--wait", "--wait-timeout", "300")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accelerator", choices=("cuda", "rocm"))
    parser.add_argument(
        "approved_inputs",
        nargs="?",
        type=Path,
        metavar="APPROVED_DIRECTORY",
        help="use a custom approved directory in place of the repository snapshots and fixtures",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--restart", action="store_true", help="start the last prepared source deployment"
    )
    mode.add_argument(
        "--upgrade",
        action="store_true",
        help="build updated images for the original, drained Compose project",
    )
    mode.add_argument(
        "--defer-build",
        action="store_true",
        help="start Vision Bill and PostgreSQL, then wait for an admin to request the native build (default)",
    )
    mode.add_argument(
        "--prepare-now",
        action="store_true",
        help="explicitly build and verify the native OCR image before starting the stack",
    )
    parser.add_argument(
        "--reuse-native-image",
        metavar="SOURCE_BUILD_TAG",
        help="verify and use a local OCR candidate from an earlier source build",
    )
    parser.add_argument(
        "--recover-stopped-ocr",
        action="store_true",
        help="during --upgrade, reconcile a pending local OCR-only handoff after verified shutdown",
    )
    args = parser.parse_args()
    if args.restart and args.reuse_native_image is not None:
        parser.error("--restart and --reuse-native-image cannot be combined")
    if args.recover_stopped_ocr and not args.upgrade:
        parser.error("--recover-stopped-ocr requires --upgrade")
    if args.approved_inputs is not None and args.restart:
        parser.error("Choose either APPROVED_DIRECTORY or --restart")
    try:
        bootstrap(
            args.accelerator,
            Path(os.path.abspath(args.approved_inputs)) if args.approved_inputs else None,
            restart=args.restart,
            reuse_native_image=args.reuse_native_image,
            upgrade=args.upgrade,
            recover_stopped_ocr=args.recover_stopped_ocr,
            defer_build=not (args.prepare_now or args.upgrade or args.restart),
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"Source OCR preparation failed in {exc.cmd[0]} (exit {exc.returncode}).",
            file=sys.stderr,
        )
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        return 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, tarfile.TarError) as exc:
        print(f"Source OCR preparation failed: {exc}", file=sys.stderr)
        return 1
    print("Source stack started. Open Settings → Dedicated OCR to enable setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
