"""Prepare model files for serving, or explicitly run the full native release qualification.

The operator supplies the actual raw OCI image config and Docker inspect image ID;
no Docker socket or guessed self-image identity is used inside this container.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from ocr_contract.activation import MaintenanceIntent

from .device_inventory import DeviceIdentityUnavailable, inspect_device_identity
from .gpu_memory import GPU_OOM_CODE, GPU_OOM_EXIT
from .initialization_lock import lock_path, maintenance_handoff
from .logging_config import configure_native_logging
from .provision import (
    KINDS,
    ModelDownloadFailed,
    ProvisionedRuntime,
    ReleaseBundle,
    file_digest,
    load_release_bundle,
    model_paths,
    provision_bundle,
    provision_models,
    safe_relative,
    source_identities,
    verify_cases,
    verify_sources,
    verify_tree,
)
from .supervision import (
    InstallationBusy,
    UncleanInstallation,
    check_fences,
    installation_lock,
    managed_installation,
    run_owned_qualification,
)
from .supervision import ManagedInterrupted as InitializationInterrupted
from .targets import verify_rocm_image_target

logger = logging.getLogger("ocr_service.initialize")


class InitializationFailed(RuntimeError):
    """Native qualification did not commit a complete release gate result."""

    reason_code = "native_initialization_required"


class InitializationGpuOutOfMemory(InitializationFailed):
    reason_code = GPU_OOM_CODE


def owned_qualification_process(command: list[str]) -> int:
    return run_owned_qualification(command, quiet=True)


def verify_image(bundle: ReleaseBundle, config: Path, image_id: str) -> None:
    if config.stat().st_size > 32 * 1024**2:
        raise ValueError("Image configuration exceeds bounds")
    digest = file_digest(config)
    if image_id != "sha256:" + digest or digest != bundle.image_config_sha256:
        raise ValueError("Actual image config bytes/inspect ID differ from release identity")
    raw = json.loads(config.read_bytes())
    if raw.get("architecture") != "amd64" or raw.get("os") != "linux":
        raise ValueError("Only Linux x86_64 native images are supported")
    if raw.get("rootfs", {}).get("type") != "layers" or not isinstance(
        raw["rootfs"].get("diff_ids"), list
    ):
        raise ValueError("Actual OCI image configuration with rootfs identity required")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _asset_paths(bundle: ReleaseBundle, state: Path) -> ProvisionedRuntime:
    root = state / "bundles" / bundle.identity
    for name, asset in bundle.models.items():
        verify_tree(root / "models" / name, asset)
    verify_tree(root / "fixtures", bundle.fixtures)
    verify_cases(root / "cases.toml", bundle.fixtures)
    smoke = next(case for case in bundle.fixtures.cases if case.kind == "upright")
    return ProvisionedRuntime(
        root,
        {name: root / "models" / name for name in bundle.models},
        root / "cases.toml",
        root / "fixtures" / smoke.path,
        bundle.identity,
    )


def verify_qualification(root: Path, bundle: ReleaseBundle, assets: ProvisionedRuntime) -> None:
    report = json.loads((root / "qualification.json").read_bytes())
    runtime = json.loads((root / "runtime.json").read_bytes())
    manifest = root / "manifest.json"
    if (
        report.get("qualified") is not True
        or report.get("stage") != "complete"
        or report.get("image_identity") != bundle.image_identity
        or report.get("accelerator") != bundle.accelerator
    ):
        raise ValueError("Native release gate did not qualify the selected identity")
    if (
        not report.get("profile_id")
        or not report.get("runtime_fingerprint")
        or report.get("manifest_sha256") != file_digest(manifest)
    ):
        raise ValueError("Incomplete native qualification publication")
    cases = report.get("cases", [])
    if len(cases) != 10 or {(row.get("kind"), row.get("fresh_worker")) for row in cases} != {
        (kind, repetition) for kind in KINDS for repetition in range(2)
    }:
        raise ValueError("Full five-case/two-fresh-worker release gate is required")
    expected = {
        "accelerator": bundle.accelerator,
        "gpu_index": bundle.gpu_index,
        "model_dirs": {name: str(path) for name, path in assets.model_dirs.items()},
        "cache_dir": str(root / "cache"),
        "qualification_manifest": str(manifest),
        "expected_packages": bundle.expected_packages,
        "expected_paddle_commit": bundle.expected_paddle_commit,
        "image_identity": bundle.image_identity,
    }
    if (
        runtime != expected
        or json.loads((root / "qualification.runtime.json").read_bytes()) != expected
    ):
        raise ValueError("Published native runtime differs from selected release settings")
    selection = json.loads((root / "selection.json").read_bytes())
    if selection != {"bundle_id": bundle.identity, "smoke_path": str(assets.smoke_path)}:
        raise ValueError("Selected smoke fixture differs from qualified release")


def selected_installation(
    state: Path, bundle: ReleaseBundle, image_id: str, device: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    selected = state / "selected"
    if not selected.is_symlink():
        if selected.exists():
            raise ValueError("Selected installation must be an atomic pointer")
        return None
    root = selected.resolve(strict=True)
    if root.parent != state / "installations":
        raise ValueError("Selected installation escapes owned state")
    record = root / "complete.json"
    if record.is_symlink() or not record.is_file():
        raise ValueError("Selected completion must be a regular file")
    completion = json.loads(record.read_bytes())
    if completion.get("bundle_id") != bundle.identity or completion.get("image_id") != image_id:
        return None
    if completion.get("device_identity") != device:
        return None
    return root, completion


def cached_installation(
    state: Path, bundle: ReleaseBundle, image_id: str, device: dict[str, Any]
) -> Path | None:
    selected = selected_installation(state, bundle, image_id, device)
    if selected is None:
        return None
    root, completion = selected
    if "schema_version" in completion:
        if type(completion["schema_version"]) is int and completion["schema_version"] == 2:
            return None  # A file-only installation does not satisfy an explicit release gate.
        raise ValueError("Unknown installed model metadata")
    files = completion.get("files", {})
    required = {
        "bundle.json",
        "candidate.runtime.json",
        "runtime.json",
        "qualification.json",
        "qualification.runtime.json",
        "manifest.json",
        "selection.json",
        "maintenance.json",
    }
    if set(files) != required:
        raise ValueError("Selected completion inventory is incomplete")
    for name, digest in files.items():
        path = root / safe_relative(name)
        if path.is_symlink() or not path.is_file() or file_digest(path) != digest:
            raise ValueError("Selected installation content changed")
    if json.loads((root / "bundle.json").read_bytes()) != bundle.model_dump(mode="json"):
        raise ValueError("Selected bundle identity changed")
    assets = _asset_paths(bundle, state)
    verify_qualification(root, bundle, assets)
    verify_device(root, device)
    return root


def verify_device(root: Path, device: dict[str, Any]) -> None:
    report = json.loads((root / "qualification.json").read_bytes())
    if any(report.get("device", {}).get(key) != value for key, value in device.items()):
        raise ValueError("Qualified device differs from verified metadata")


def check_installation(
    bundle_path: Path, state_dir: Path, image_config: Path, image_id: str
) -> Path:
    """Validate an existing installation without provisioning or loading native models."""
    with installation_lock(lock_path(), exclusive=False):
        check_fences(lock_path(), "check")
        bundle = load_release_bundle(bundle_path)
        verify_image(bundle, image_config, image_id)
        state = state_dir.resolve()
        if not (state / "selected").exists():
            raise InitializationFailed(
                "native_initialization_required: run explicit prepare with --maintenance-intent after stopping OCR and attesting external LLM quiescence or a separate OCR device"
            )
        device = inspect_device_identity(bundle.accelerator, bundle.gpu_index)
        cached = cached_installation(state, bundle, image_id, device)
        if cached is not None:
            return cached
        prepared = _cached_prepared(state, bundle, image_id, device)
        if prepared is None:
            raise InitializationFailed(
                "native_initialization_required: selected bundle/image/device changed; run explicit prepare"
            )
        verify_package_pins(bundle)
        verify_prepared_target(bundle, device)
        return prepared


@contextmanager
def new_installation(state: Path) -> Iterator[Path]:
    """Publish one complete candidate atomically and preserve the last good selection on failure."""
    installations = state / "installations"
    if installations.is_symlink():
        raise ValueError("Installation directory escapes owned state")
    installations.mkdir(exist_ok=True, mode=0o700)
    root = installations / uuid4().hex
    selected = state / "selected"
    target = root.relative_to(state)
    root.mkdir(parents=True, mode=0o700)
    try:
        yield root
        temporary = state / (".selected-" + uuid4().hex)
        try:
            temporary.symlink_to(target, target_is_directory=True)
            temporary.replace(selected)
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        # A signal can raise after the atomic replacement succeeds. Read the
        # pointer under the exclusive lock before deleting any candidate.
        if not selected.is_symlink() or selected.readlink() != target:
            shutil.rmtree(root)
        raise


def initialize(
    bundle_path: Path,
    state_dir: Path,
    image_config: Path,
    image_id: str,
    *,
    maintenance_intent: Literal["external-llm-quiesced", "separate-ocr-device"] | MaintenanceIntent,
    handoff_lock: Path | None = None,
    force: bool = False,
    require_remote_models: bool = False,
    retry_collection_timeout: bool = False,
) -> Path:
    if maintenance_intent not in {"external-llm-quiesced", "separate-ocr-device"}:
        raise ValueError("Explicit maintenance intent is required for this invocation")
    bundle = load_release_bundle(bundle_path)
    verify_image(bundle, image_config, image_id)
    # Use the app's original lock inode; hold it until the native scope has drained.
    with (
        maintenance_handoff(handoff_lock),
        managed_installation("initializer", lock_path()),
    ):
        bundle = load_release_bundle(bundle_path)
        verify_image(bundle, image_config, image_id)
        if require_remote_models and any(
            source.scheme != "https" or not source.hostname
            for source in (urlsplit(model.source) for model in bundle.models.values())
        ):
            raise ValueError("Dedicated OCR activation requires pinned HTTPS model archives")
        device = inspect_device_identity(bundle.accelerator, bundle.gpu_index)
        state = state_dir.resolve()
        if (
            not force
            and (cached := cached_installation(state, bundle, image_id, device)) is not None
        ):
            return cached
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Tighten only this owned native state root, including fresh daemon volumes.
        state.chmod(0o700)
        assets = provision_bundle(bundle, state)
        with new_installation(state) as root:
            runtime = {
                "accelerator": bundle.accelerator,
                "gpu_index": bundle.gpu_index,
                "model_dirs": {name: str(path) for name, path in assets.model_dirs.items()},
                "cache_dir": str(root / "cache"),
                "qualification_manifest": str(root / "manifest.json"),
                "expected_packages": bundle.expected_packages,
                "expected_paddle_commit": bundle.expected_paddle_commit,
                "image_identity": bundle.image_identity,
            }
            write_json(root / "bundle.json", bundle.model_dump(mode="json"))
            write_json(root / "candidate.runtime.json", runtime)
            write_json(
                root / "maintenance.json",
                {
                    "invocation_id": uuid4().hex,
                    "intent": maintenance_intent,
                    "bundle_id": bundle.identity,
                    "image_id": image_id,
                    "device_identity": device,
                },
            )
            output = root / "qualification.json"
            command = [
                sys.executable,
                "-m",
                "ocr_service.qualify",
                "--accelerator",
                bundle.accelerator,
                "--runtime",
                str(root / "candidate.runtime.json"),
                "--cases",
                str(assets.cases_path),
                "--output",
                str(output),
            ]
            exit_code = owned_qualification_process(command)
            if exit_code and retry_collection_timeout:
                from .qualify import COLLECTION_WAIT_TIMEOUT_EXIT, is_collection_wait_timeout

                if exit_code == COLLECTION_WAIT_TIMEOUT_EXIT and is_collection_wait_timeout(
                    output, accelerator=bundle.accelerator, image_identity=bundle.image_identity
                ):
                    # The supervisor has drained the failed invocation. Reset
                    # only this unpublished candidate's cache and evidence.
                    cache = root / "cache"
                    if os.path.lexists(cache):
                        shutil.rmtree(cache)
                    output.unlink(missing_ok=True)
                    output.with_suffix(".runtime.json").unlink(missing_ok=True)
                    (root / "manifest.json").unlink(missing_ok=True)
                    logger.warning("native_qualification_retry_collection_timeout")
                    exit_code = owned_qualification_process(command)
            if exit_code == GPU_OOM_EXIT:
                raise InitializationGpuOutOfMemory(GPU_OOM_CODE)
            if exit_code:
                from .qualify import STATIC_HOOK_FAILURE_CODE, read_static_hook_failure

                diagnostic = read_static_hook_failure(output)
                if diagnostic is not None:
                    logger.warning(
                        "%s model=%s reason=%s input=%d output=%d",
                        STATIC_HOOK_FAILURE_CODE,
                        diagnostic.model_name,
                        diagnostic.reason_code,
                        diagnostic.input_count,
                        diagnostic.output_count,
                    )
                raise InitializationFailed(
                    "Full native qualification failed; selected settings were not published"
                )
            shutil.copyfile(root / "qualification.runtime.json", root / "runtime.json")
            write_json(
                root / "selection.json",
                {"bundle_id": bundle.identity, "smoke_path": str(assets.smoke_path)},
            )
            verify_sources(bundle)
            _asset_paths(bundle, state)
            verify_qualification(root, bundle, assets)
            if inspect_device_identity(bundle.accelerator, bundle.gpu_index) != device:
                raise ValueError("Native device changed during maintenance")
            verify_device(root, device)
            files = {path.name: file_digest(path) for path in root.iterdir() if path.is_file()}
            write_json(
                root / "complete.json",
                {
                    "bundle_id": bundle.identity,
                    "image_id": image_id,
                    "device_identity": device,
                    "files": files,
                },
            )
        return root


class PreparationFailed(ValueError):
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


class PreparationRequired(RuntimeError):
    """A missing, previous-format or changed installation requires a new atomic publication."""


def verify_package_pins(bundle: ReleaseBundle) -> None:
    from .profile import verify_installed_versions

    try:
        verify_installed_versions(bundle.expected_packages, accelerator=bundle.accelerator)
    except (metadata.PackageNotFoundError, ValueError) as exc:
        raise PreparationFailed("native_package_mismatch") from exc


def verify_prepared_target(bundle: ReleaseBundle, device: dict[str, Any]) -> None:
    if bundle.accelerator == "rocm":
        evidence = json.loads(Path("/opt/ocr/build-evidence/kernels.json").read_bytes())
        verify_rocm_image_target(evidence, device)


def _prepared_inputs(
    bundle_path: Path, config: Path, image_id: str, *, standalone: bool = False,
) -> tuple[ReleaseBundle, dict[str, Any]]:
    bundle = load_release_bundle(bundle_path)
    verify_image(bundle, config, image_id)
    for asset in bundle.models.values():
        source = urlsplit(asset.source)
        if standalone and not source.scheme:
            safe_relative(asset.source)
        elif source.scheme != "https" or not source.hostname:
            raise PreparationFailed("native_activation_inputs_invalid")
    verify_package_pins(bundle)
    try:
        device = inspect_device_identity(bundle.accelerator, bundle.gpu_index)
        verify_prepared_target(bundle, device)
    except (DeviceIdentityUnavailable, ValueError, OSError, KeyError) as exc:
        raise PreparationFailed("native_device_identity_unavailable") from exc
    return bundle, device


def _prepared_runtime(
    root: Path, bundle: ReleaseBundle, state: Path, device: dict[str, Any]
) -> dict[str, Any]:
    from .profile import PreparedRuntimeSettings

    return PreparedRuntimeSettings(
        schema_version=2,
        accelerator=bundle.accelerator,
        gpu_index=bundle.gpu_index,
        cache_dir=root / "cache",
        model_dirs=model_paths(bundle, state),
        model_content={name: asset.tree_sha256 for name, asset in bundle.models.items()},
        expected_packages=bundle.expected_packages,
        expected_paddle_commit=bundle.expected_paddle_commit,
        image_identity=bundle.image_identity,
        device_identity=device,
    ).model_dump(mode="json")


def _cached_prepared(
    state: Path, bundle: ReleaseBundle, image_id: str, device: dict[str, Any]
) -> Path | None:
    selection = selected_installation(state, bundle, image_id, device)
    if selection is None:
        return None
    root, complete = selection
    if "schema_version" not in complete:
        return None  # Legacy qualified installation; reuse its pinned archive cache.
    if type(complete["schema_version"]) is not int or complete["schema_version"] != 2:
        raise ValueError("Unknown installed model metadata")
    runtime_path = root / "runtime.json"
    if (
        runtime_path.is_symlink()
        or not runtime_path.is_file()
        or complete.get("runtime_sha256") != file_digest(runtime_path)
    ):
        raise ValueError("Selected runtime changed")
    if json.loads(runtime_path.read_bytes()) != _prepared_runtime(root, bundle, state, device):
        raise ValueError("Selected runtime differs from the pinned release")
    try:
        if (state / "models").is_symlink():
            raise ValueError("Model cache must use an owned directory")
        for name, path in model_paths(bundle, state).items():
            verify_tree(path, bundle.models[name])
    except (ValueError, OSError) as exc:
        raise PreparationFailed("native_model_integrity_failed") from exc
    return root


def check_prepared_installation(
    bundle_path: Path, state_dir: Path, image_config: Path, image_id: str
) -> Path:
    """Passive restart: verify files, image, package pins and device without starting Paddle."""
    with installation_lock(lock_path(), exclusive=False):
        check_fences(lock_path(), "check")
        bundle, device = _prepared_inputs(bundle_path, image_config, image_id)
        root = _cached_prepared(state_dir.resolve(), bundle, image_id, device)
        if root is None:
            raise PreparationRequired
        return root


def prepare_installation(
    bundle_path: Path,
    state_dir: Path,
    image_config: Path,
    image_id: str,
    *,
    maintenance_intent: MaintenanceIntent,
    handoff_lock: Path | None,
    repair: bool = False,
    standalone: bool = False,
) -> Path:
    """Install the verified model trees for the active profile."""
    MaintenanceIntent(maintenance_intent)
    with maintenance_handoff(handoff_lock), managed_installation("initializer", lock_path()):
        bundle, device = _prepared_inputs(bundle_path, image_config, image_id, standalone=standalone)
        state = state_dir.resolve()
        if standalone:
            qualified = cached_installation(state, bundle, image_id, device)
            if qualified is not None:
                return qualified
        try:
            cached = _cached_prepared(state, bundle, image_id, device)
        except PreparationFailed as exc:
            if not repair or exc.reason_code != "native_model_integrity_failed":
                raise
            cached = None
        if cached is not None:
            return cached
        try:
            state.mkdir(parents=True, exist_ok=True, mode=0o700)
            state.chmod(0o700)
            provision_models(bundle, state, repair=repair)
            if inspect_device_identity(bundle.accelerator, bundle.gpu_index) != device:
                raise PreparationFailed("native_device_identity_unavailable")
            with new_installation(state) as root:
                runtime_path = root / "runtime.json"
                write_json(runtime_path, _prepared_runtime(root, bundle, state, device))
                write_json(
                    root / "complete.json",
                    {
                        "schema_version": 2,
                        "bundle_id": bundle.identity,
                        "image_id": image_id,
                        "device_identity": device,
                        "runtime_sha256": file_digest(runtime_path),
                    },
                )
            return root
        except PreparationFailed:
            raise
        except DeviceIdentityUnavailable as exc:
            raise PreparationFailed("native_device_identity_unavailable") from exc
        except ModelDownloadFailed as exc:
            raise PreparationFailed("native_model_download_failed") from exc
        except (ValueError, tarfile.TarError) as exc:
            raise PreparationFailed("native_model_integrity_failed") from exc
        except OSError as exc:
            raise PreparationFailed("native_installation_write_failed") from exc


def main() -> None:
    configure_native_logging()
    try:
        _run_cli()
    except InitializationInterrupted as exc:
        raise SystemExit(128 + exc.signum) from None
    except InitializationGpuOutOfMemory:
        raise SystemExit(GPU_OOM_CODE) from None
    except InitializationFailed:
        raise SystemExit("native_initialization_required") from None
    except InstallationBusy:
        raise SystemExit("native_installation_busy") from None
    except UncleanInstallation:
        raise SystemExit("native_activation_recovery_required") from None
    except DeviceIdentityUnavailable:
        raise SystemExit("native_device_identity_unavailable") from None
    except PreparationFailed as exc:
        allowed = {
            "native_activation_inputs_invalid", "native_package_mismatch",
            "native_device_identity_unavailable", "native_model_download_failed",
            "native_model_integrity_failed", "native_installation_write_failed",
        }
        raise SystemExit(exc.reason_code if exc.reason_code in allowed else "native_initialization_failed") from None
    except OSError:
        raise SystemExit("native_installation_io_failed") from None
    except (ValueError, KeyError, TypeError, AttributeError, tarfile.TarError):
        raise SystemExit("native_activation_inputs_invalid") from None
    except Exception:  # noqa: BLE001 - native libraries may embed private inputs in their exceptions
        raise SystemExit("native_initialization_failed") from None


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("operation", nargs="?", choices=("check", "prepare", "maintain"), default="check")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--state-dir", type=Path, default=Path(os.environ.get("OCR_STATE_DIR", "/var/lib/ocr"))
    )
    parser.add_argument("--image-config", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument(
        "--maintenance-intent",
        choices=("external-llm-quiesced", "separate-ocr-device"),
        help="Invocation-local operator attestation: external LLM is quiesced, or OCR uses a separate GPU; metadata checks do not prove this",
    )
    parser.add_argument(
        "--handoff-lock",
        type=Path,
        help="Operator-supplied original app gpu-handoff.lock path on its original shared mount (required by paired maintenance); quiescence attestation remains required",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Requalify in maintain; never bypasses locks or retained crash fences",
    )
    parser.add_argument("--print-source-identities", action="store_true")
    try:
        args, unknown = parser.parse_known_args()
    except argparse.ArgumentError:
        parser.error("Invalid initializer arguments; see --help")
    if unknown:
        parser.error("Unrecognized initializer arguments; see --help")
    if args.print_source_identities:
        print(json.dumps(source_identities(), sort_keys=True, indent=2))
        return
    if not args.bundle or not args.image_config or not args.image_id:
        parser.error("--bundle, --image-config and actual --image-id are required")
    if args.operation != "check" and not args.maintenance_intent:
        parser.error(f"{args.operation} requires invocation-local --maintenance-intent")
    if args.force and args.operation != "maintain":
        parser.error("--force requires maintain")
    if (
        args.operation != "check"
        and os.environ.get("OCR_MAINTENANCE_REQUIRE_HANDOFF") == "true"
        and not args.handoff_lock
    ):
        parser.error("Paired maintenance requires the original app --handoff-lock path explicitly")
    if args.operation == "check" and (args.maintenance_intent or args.handoff_lock):
        parser.error("--maintenance-intent and --handoff-lock require prepare or maintain")
    if args.operation == "check":
        check_installation(args.bundle, args.state_dir, args.image_config, args.image_id)
    elif args.operation == "prepare":
        prepare_installation(
            args.bundle, args.state_dir, args.image_config, args.image_id,
            maintenance_intent=MaintenanceIntent(args.maintenance_intent),
            handoff_lock=args.handoff_lock, standalone=True, repair=True,
        )
    else:
        initialize(
            args.bundle,
            args.state_dir,
            args.image_config,
            args.image_id,
            maintenance_intent=args.maintenance_intent,
            handoff_lock=args.handoff_lock,
            force=args.force,
        )


if __name__ == "__main__":
    main()
