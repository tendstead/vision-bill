import hashlib
import io
import json
import os
import platform
import runpy
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_ocr_provision import release as ocr_release  # type: ignore[import-not-found] # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ocr_source_up.py"


def test_rocm_qualification_preserves_selected_render_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import ocr_source_up

    render = "/dev/dri/renderD129"
    original_stat = Path.stat

    def device_stat(path: Path, **kwargs: Any) -> Any:
        gids = {"/dev/kfd": 21, render: 22}
        if str(path) in gids:
            return SimpleNamespace(st_mode=stat.S_IFCHR, st_gid=gids[str(path)])
        return original_stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", device_stat)
    devices = [{"source": "/dev/kfd", "target": "/dev/kfd"}, {"source": render, "target": render}]
    arguments, environment = ocr_source_up.gpu_args("rocm", {"devices": devices})
    assert arguments == [
        "--device",
        "/dev/kfd:/dev/kfd",
        "--group-add",
        "21",
        "--device",
        "/dev/dri/renderD129:/dev/dri/renderD129",
        "--group-add",
        "22",
    ]
    assert environment == {
        "OCR_RENDER_NODE": render,
        "OCR_VIDEO_GID": "21",
        "OCR_RENDER_GID": "22",
    }


def test_wsl_rocm_binding_uses_only_dxg_and_one_hip_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import ocr_source_up

    monkeypatch.setattr(Path, "is_char_device", lambda path: str(path) == "/dev/dxg")
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "/usr/lib/wsl/lib/libdxcore.so")
    native = {
        "devices": [{"source": "/dev/dxg", "target": "/dev/dxg"}],
        "environment": {"HIP_VISIBLE_DEVICES": "2"},
    }
    arguments, environment = ocr_source_up.gpu_args("rocm", native)
    assert arguments == [
        "--device", "/dev/dxg:/dev/dxg", "--mount",
        "type=bind,source=/usr/lib/wsl/lib/libdxcore.so,target=/usr/lib/libdxcore.so,readonly",
        "--env", "HIP_VISIBLE_DEVICES=2",
    ]
    assert environment == {"OCR_ROCM_DEVICE_ID": "2"}
    for selection in ("1,2", "-1", "all", "1025"):
        native["environment"] = {"HIP_VISIBLE_DEVICES": selection}
        with pytest.raises(ValueError, match="one AMD WSL GPU"):
            ocr_source_up.gpu_args("rocm", native)


@pytest.mark.parametrize(
    "renders",
    [
        [{"source": "/dev/dri/renderD129", "target": "/dev/dri/renderD128"}],
        [
            {"source": "/dev/dri/renderD128", "target": "/dev/dri/renderD128"},
            {"source": "/dev/dri/renderD129", "target": "/dev/dri/renderD129"},
        ],
        [
            {"source": "/dev/dri/renderD129", "target": "/dev/dri/renderD129"},
            {"source": "/dev/dri/renderD129", "target": "/dev/dri/renderD129"},
        ],
    ],
)
def test_rocm_qualification_rejects_remapped_or_extra_render_devices(
    renders: list[dict[str, str]],
) -> None:
    from scripts import ocr_source_up

    devices = [{"source": "/dev/kfd", "target": "/dev/kfd"}, *renders]
    with pytest.raises(ValueError, match="one selected DRI render node"):
        ocr_source_up.gpu_args("rocm", {"devices": devices})


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
@pytest.mark.parametrize("mode", ["defer", "prepare", "upgrade"])
def test_source_command_prepares_default_inputs(
    accelerator: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    calls = []

    def bootstrap(
        chosen: str,
        inputs: Path | None,
        *,
        restart: bool,
        reuse_native_image: str | None,
        upgrade: bool,
        recover_stopped_ocr: bool,
        defer_build: bool = False,
    ) -> None:
        calls.append((chosen, inputs, restart, reuse_native_image, upgrade, recover_stopped_ocr, defer_build))

    candidate = f"vision-bill-ocr-{accelerator}:source-build-" + "a" * 32
    flags = {"defer": [], "prepare": ["--prepare-now"], "upgrade": ["--upgrade", "--reuse-native-image", candidate]}[mode]
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), accelerator, *flags])
    monkeypatch.setattr(ocr_source_up, "bootstrap", bootstrap)
    assert ocr_source_up.main() == 0
    assert calls == [(accelerator, None, False, candidate if mode == "upgrade" else None,
                      mode == "upgrade", False, mode == "defer")]


def test_source_command_rejects_restart_with_reuse_before_bootstrap(monkeypatch, capsys):
    from scripts import ocr_source_up

    candidate = "vision-bill-ocr-cuda:source-build-" + "a" * 32
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "cuda", "--restart", "--reuse-native-image", candidate])
    monkeypatch.setattr(ocr_source_up, "bootstrap", lambda *args, **kwargs: pytest.fail("bootstrap called"))
    with pytest.raises(SystemExit) as exc:
        ocr_source_up.main()
    assert exc.value.code == 2
    assert "--restart and --reuse-native-image cannot be combined" in capsys.readouterr().err


def test_source_command_rejects_supplied_inputs_with_restart() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "cuda", "approved", "--restart"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "APPROVED_DIRECTORY" in result.stderr
    assert "--restart" in result.stderr


def docker_metadata() -> tuple[str, str, str, dict[str, bytes]]:
    blobs: dict[str, bytes] = {}

    def add(record: dict[str, Any]) -> str:
        raw = json.dumps(record, indent=2).encode()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        blobs[digest] = raw
        return digest

    config = add(
        {"architecture": "amd64", "os": "linux", "rootfs": {"type": "layers", "diff_ids": []}}
    )
    manifest = add({"schemaVersion": 2, "config": {"digest": config}, "layers": []})
    platform = {"os": "linux", "architecture": "amd64"}
    descriptor = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "platform": platform,
    }
    index = add(
        {
            "schemaVersion": 2,
            "manifests": [
                {
                    **descriptor,
                    "digest": "sha256:" + "0" * 64,
                    "annotations": {
                        "vnd.docker.reference.type": "attestation-manifest",
                    },
                },
                {**descriptor, "digest": manifest},
            ],
        }
    )
    return config, manifest, index, blobs


@pytest.mark.parametrize("target", [0, 1, 2])
def test_export_verifies_exact_config_for_each_engine_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: int
) -> None:
    from scripts import ocr_source_up

    config, manifest, index, blobs = docker_metadata()
    monkeypatch.setattr(ocr_source_up, "build_history_blob", blobs.__getitem__, raising=False)
    assert ocr_source_up.export_config(tmp_path, (config, manifest, index)[target]) == config
    assert (tmp_path / "image-config.json").read_bytes() == blobs[config]


@pytest.mark.parametrize("target", [0, 1, 2])
def test_export_refuses_a_changed_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: int
) -> None:
    from scripts import ocr_source_up

    config, manifest, index, blobs = docker_metadata()
    digest = (config, manifest, index)[target]
    blobs[digest] += b" "
    monkeypatch.setattr(ocr_source_up, "build_history_blob", blobs.__getitem__, raising=False)
    with pytest.raises(ValueError, match="Docker image metadata digest"):
        ocr_source_up.export_config(tmp_path, index)
    assert not (tmp_path / "image-config.json").exists()


def test_export_falls_back_to_an_archive_that_preserves_containerd_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    config, _, index, blobs = docker_metadata()

    def unavailable(digest: str) -> bytes:
        raise subprocess.CalledProcessError(1, ["docker", "buildx"])

    def save(*args: str, capture: bool = False) -> str:
        assert args[:4] == ("docker", "image", "save", "--output")
        assert args[-1] == index
        with tarfile.open(args[4], "w") as archive:
            for digest, raw in blobs.items():
                entry = tarfile.TarInfo(f"blobs/sha256/{digest[7:]}")
                entry.size = len(raw)
                archive.addfile(entry, io.BytesIO(raw))
        return ""

    monkeypatch.setattr(ocr_source_up, "build_history_blob", unavailable, raising=False)
    monkeypatch.setattr(ocr_source_up, "command", save)
    release = tmp_path / "release"
    release.mkdir()
    assert ocr_source_up.export_config(release, index) == config
    assert (release / "image-config.json").read_bytes() == blobs[config]
    assert sorted(tmp_path.iterdir()) == [release]


def test_staging_keeps_approved_metadata_and_checks_fixture_bytes(tmp_path: Path) -> None:
    from scripts.ocr_source_up import stage_approved_inputs

    approved = tmp_path / "approved"
    approved.mkdir()
    fixture = approved / "assets" / "fixtures.tar"
    fixture.parent.mkdir()
    fixture.write_bytes(b"sealed fixture archive")
    manifest = {
        "accelerator": "rocm",
        "gpu_index": 0,
        "models": {"model": {"source": "https://example.org/model.tar"}},
        "fixtures": {
            "source": "assets/fixtures.tar",
            "size_bytes": 22,
            "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        },
    }
    (approved / "bundle.json").write_text(json.dumps(manifest))
    destination = tmp_path / "staged"
    destination.mkdir()
    assert stage_approved_inputs(approved, destination, "rocm") == manifest
    assert (destination / "assets/fixtures.tar").read_bytes() == b"sealed fixture archive"
    assert json.loads((approved / "bundle.json").read_text()) == manifest

    fixture.write_bytes(b"changed archive bytes!")
    with pytest.raises(ValueError, match="fixture archive"):
        stage_approved_inputs(approved, destination, "rocm")


def test_managed_overlay_pins_native_image_and_disables_startup_enable(tmp_path: Path) -> None:
    from scripts.ocr_source_up import managed_overlay

    image_id = "sha256:" + "d" * 64
    config_id = "sha256:" + "c" * 64
    app_id = "sha256:" + "e" * 64
    result = managed_overlay(
        "cuda",
        tmp_path,
        image_id,
        app_id,
        "test",
        {"OCR_CUDA_DEVICE_ID": "0"},
        "a" * 64,
        config_id=config_id,
    )
    app = result["services"]["vision_bill"]
    native = result["services"]["paddleocr-service"]
    assert app["image"] == "vision-bill:source-" + "e" * 64
    assert "/api/v1/system/ui-config" in app["healthcheck"]["test"][-1]
    assert app["environment"] == {"OCR_ACTIVATION_ON_STARTUP": "false"}
    assert native["image"] == "vision-bill-ocr-cuda:source-" + "d" * 64
    assert native["pull_policy"] == "never"
    assert native["environment"]["OCR_IMAGE_ID"] == config_id
    assert result["x-vision-bill-source"]["image_config_sha256"] == config_id[7:]
    assert native["volumes"] == [
        {
            "type": "bind",
            "source": str(tmp_path),
            "target": "/release",
            "read_only": True,
            "bind": {"create_host_path": False},
        }
    ]


@pytest.mark.parametrize(
    ("automatic", "requested_channel", "compose_channel", "selected_channel", "reuse"),
    [
        (False, None, None, "cu129", False),
        (True, None, "cu129", "cu129", True),
        (True, "cu126", "cu126", "cu126", False),
        (False, None, "cu126", "cu126", False),
    ],
)
def test_failed_candidate_cannot_create_restart_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    automatic: bool,
    requested_channel: str | None,
    compose_channel: str | None,
    selected_channel: str,
    reuse: bool,
) -> None:
    from scripts import ocr_source_up

    approved = tmp_path / "approved"
    approved.mkdir()
    (approved / "bundle.json").write_text(
        json.dumps(
            {
                "accelerator": "cuda",
                "gpu_index": 0,
                "models": {"model": {"source": "https://example.org/model.tar"}},
                "fixtures": {"source": "https://example.org/fixtures.tar"},
            }
        )
    )
    state = tmp_path / "state"
    state.mkdir()
    managed = state / "compose.local.json"
    config = {
        "name": "test",
        "services": {
            "paddleocr-service": {
                "image": "vision-bill-ocr-cuda:candidate",
                "build": {"args": {"OCR_CUDA_CHANNEL": compose_channel}},
                "deploy": {"resources": {"reservations": {"devices": [{"device_ids": ["2"]}]}}},
            }
        },
    }
    config_id, _, image_id, blobs = docker_metadata()
    image_config = blobs[config_id]
    candidate = "vision-bill-ocr-cuda:source-build-" + "a" * 32
    prepare_paths: list[Path] = []

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "config" in args:
            return json.dumps(config)
        if "ps" in args or args[1:3] == ("volume", "ls"):
            return ""
        if args[0] == "uv" and any(arg.endswith("/prepare_ocr_inputs.py") for arg in args):
            assert automatic
            assert {"pydantic==2.13.4", "packaging==26.2", "pillow==12.3.0"} <= set(args)
            assert args[args.index("--models") + 1].endswith("/build/paddle-model-snapshots.json")
            assert args[args.index("--fixtures") + 1].endswith("/first-run-fixtures/fixtures.json")
            output = Path(args[args.index("--output") + 1])
            assert output.parent.parent == state
            assert output.parent.stat().st_mode & 0o077 == 0
            prepare_paths.append(output.parent)
            prepared = subprocess.run(
                [sys.executable, *args[args.index("python") + 1 :]],
                cwd=SCRIPT.parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            assert prepared.returncode == 0, prepared.stderr
            return prepared.stdout
        if "--check" in args:
            assert args[-2:] == ("--channel", selected_channel)
            return "f" * 64 if "--reuse-image" in args else str(SCRIPT.parents[1] / "paddleocr-service")
        if args[:2] == ("make", "ocr-build-cuda"):
            assert not reuse
            return ""
        if args[1:3] == ("image", "inspect"):
            if reuse:
                assert args[-1] == candidate
            return image_id
        if args[1] == "run":
            assert args[2:11] == (
                "--rm",
                "--pull",
                "never",
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                "--gpus",
                "device=2",
            )
            binder = args.index("-c")
            assert args[binder - 1] == image_id
            assert args[binder + 2 : binder + 5] == (image_id, config_id, "/release")
            assert bool(args[binder + 5]) is reuse
            release = Path(args[args.index("--volume") + 1].removesuffix(":/release:ro"))
            assert (release / "image-config.json").read_bytes() == image_config
            if automatic:
                assert {path.name for path in release.iterdir()} == {
                    "bundle.json",
                    "fixtures.tar",
                    "image-config.json",
                }
                with tarfile.open(release / "fixtures.tar") as archive:
                    assert set(archive.getnames()) == {
                        "LICENSE.txt",
                        "upright.png",
                        "rotated.png",
                        "warped.png",
                        "table.png",
                        "formula.png",
                    }
                assert all(not path.exists() for path in prepare_paths)
            raise subprocess.CalledProcessError(1, args)
        pytest.fail(f"Unexpected subprocess command: {args}")

    for key in ("OCR_CUDA_DEVICE_ID", "OCR_CUDA_CHANNEL", "VB_IMAGE_TAG", "OCR_IMAGE_TAG"):
        monkeypatch.delenv(key, raising=False)
    if requested_channel is not None:
        monkeypatch.setenv("OCR_CUDA_CHANNEL", requested_channel)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    monkeypatch.setattr(ocr_source_up, "build_history_blob", blobs.__getitem__)
    with pytest.raises(subprocess.CalledProcessError):
        ocr_source_up.deploy(
            ("docker", "compose"),
            state,
            "cuda",
            None if automatic else approved,
            reuse_native_image=candidate if reuse else None,
        )
    assert len(prepare_paths) == int(automatic)
    assert not managed.exists()
    assert not list(state.iterdir())


@pytest.mark.parametrize(
    "altered", [
        None, "lock", "sources", "legacy", "bad_retired", "wrong_snapshot", "unexpected", "missing",
        "wrong_gpu", "two_gpus", "old_gpu", "old_driver",
    ]
)
def test_native_binding_uses_full_exact_package_closure(
    ocr_release: tuple[Path, dict[str, Any], Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    altered: str | None,
) -> None:
    from importlib import metadata
    from types import SimpleNamespace

    from ocr_service import bootstrap, device_inventory, provision

    from scripts.ocr_source_up import BIND_BUNDLE

    path, manifest, image_config = ocr_release
    current_models = dict(manifest["models"])
    for name, record in current_models.items():
        record["source"] = f"https://example.org/approved/{name}.tar"
    if altered in {"legacy", "bad_retired", "wrong_snapshot", "unexpected", "missing"}:
        retired = dict(current_models["PP-DocLayoutV3"], source="https://example.org/approved-uvdoc.tar")
        if altered == "bad_retired":
            retired["tree_sha256"] = "f" * 64
        if altered == "wrong_snapshot":
            retired["source"] = "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/resolve/" + "f" * 40
        manifest["models"]["UVDoc"] = retired
        if altered == "unexpected":
            manifest["models"]["unsigned-extra"] = retired
        if altered == "missing":
            del manifest["models"]["PP-DocLayoutV3"]
    approved = manifest["expected_packages"]
    approved["numpy"] = "2.4.0"
    path.write_text(json.dumps(manifest, indent=2))
    original = path.read_bytes()
    closure = {**approved, "pydantic": "2.12.0"}
    config_id = "sha256:" + hashlib.sha256(image_config.read_bytes()).hexdigest()
    image_id = "sha256:" + "d" * 64
    sources = provision.source_identities()
    if altered == "sources":
        sources["runtime-lock.json"] = "f" * 64
    lock = path.parent / "requirements.cuda.lock"
    lock.write_text("expected CUDA lock")
    resolver = path.parent / "resolve_cuda.py"
    resolver.write_text(
        "import sys; assert sys.argv[1:] == " + repr(["--check", str(lock), "--channel", "cu129"])
    )
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if altered == "lock":
        lock.write_text("wrong CUDA lock")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "-c",
            image_id,
            config_id,
            str(path.parent),
            json.dumps(sources),
            "cu129",
            digest,
            str(lock),
            str(resolver),
        ],
    )
    monkeypatch.setattr(
        metadata, "distributions", lambda: [SimpleNamespace(metadata={"Name": "paddlepaddle-gpu"})]
    )
    monkeypatch.setattr(bootstrap, "seal_dependencies", lambda name, **kwargs: (closure, {}))
    import ctypes
    from uuid import UUID

    expected_uuid = UUID("aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb")
    monkeypatch.setenv("VISION_BILL_CUDA_SELECTION", f"gpu-{expected_uuid}")
    def count_devices(count):
        count._obj.value = 2 if altered == "two_gpus" else 1
        return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda *_: SimpleNamespace(cuDeviceGetCount=count_devices))
    monkeypatch.setattr(device_inventory, "inspect_device_identity", lambda *args: {
        "uuid": "f" * 32 if altered == "wrong_gpu" else expected_uuid.hex,
        "compute_capability": [6, 1] if altered == "old_gpu" else [7, 0],
        "driver_version": 11080 if altered == "old_driver" else 12090,
    })
    binder = path.parent / "bind.py"
    binder.write_text(BIND_BUNDLE)
    if altered in {"wrong_gpu", "two_gpus", "old_gpu", "old_driver"}:
        with pytest.raises(ValueError, match="CUDA|SM 7.0"):
            runpy.run_path(str(binder))
        return
    if altered == "lock":
        with pytest.raises(ValueError, match="Built CUDA lock differs"):
            runpy.run_path(str(binder))
        return
    if altered == "sources":
        with pytest.raises(ValueError, match="Reused OCR image sources"):
            runpy.run_path(str(binder))
        return
    if altered in {"bad_retired", "wrong_snapshot", "unexpected", "missing"}:
        with pytest.raises(ValueError):
            runpy.run_path(str(binder))
        assert path.read_bytes() == original
        return
    runpy.run_path(str(binder))
    assert path.read_bytes() == original
    bound = json.loads(capsys.readouterr().out)
    assert bound["_bound_cuda_uuid"] == expected_uuid.hex
    assert bound["expected_packages"] == closure
    assert bound["image_identity"] == image_id
    assert bound["image_config_sha256"] == config_id[7:]
    assert bound["models"] == current_models
    assert bound["fixtures"] == manifest["fixtures"]


@pytest.fixture
def managed_release(tmp_path: Path) -> tuple[Path, str, str]:
    release = tmp_path / "release-validated"
    release.mkdir(mode=0o700)
    image_config = b'{"architecture":"amd64"}'
    (release / "image-config.json").write_bytes(image_config)
    manifest = json.dumps(
        {
            "accelerator": "cuda",
            "gpu_index": 0,
            "models": {
                name: {"source": f"https://example.org/{name}.tar"}
                for name in ("PP-DocLayoutV3", "PP-LCNet_x1_0_doc_ori", "UVDoc", "PaddleOCR-VL-1.6-0.9B")
            },
            "fixtures": {"source": "https://example.org/fixtures.tar"},
        }
    ).encode()
    (release / "bundle.json").write_bytes(manifest)
    return (
        release,
        "sha256:" + hashlib.sha256(image_config).hexdigest(),
        hashlib.sha256(manifest).hexdigest(),
    )


@pytest.mark.parametrize(
    ("bundle_altered", "legacy"), [(False, False), (False, True), (True, False)]
)
def test_restart_checks_pinned_images_and_waits_on_original_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_release: tuple[Path, str, str],
    bundle_altered: bool,
    legacy: bool,
) -> None:
    from scripts import ocr_source_up

    release, config_id, bundle_digest = managed_release
    native_id = config_id if legacy else "sha256:" + "d" * 64
    app_id = "sha256:" + "e" * 64
    overlay = ocr_source_up.managed_overlay(
        "cuda",
        release,
        native_id,
        app_id,
        "original",
        {"OCR_CUDA_DEVICE_ID": "2"},
        bundle_digest,
        config_id=config_id,
    )
    if legacy:
        del overlay["x-vision-bill-source"]["image_config_sha256"]
    (tmp_path / "compose.local.json").write_text(json.dumps(overlay))
    calls = []

    def docker_boundary(*args: str, capture: bool = False) -> str:
        calls.append(args)
        if args[1:3] == ("image", "inspect"):
            return app_id if args[-1].startswith("vision-bill:") else native_id
        if "ps" in args or "up" in args or args[1:3] == ("volume", "ls"):
            return ""
        pytest.fail(f"Unexpected subprocess command: {args}")

    monkeypatch.setenv("OCR_CUDA_DEVICE_ID", "0")
    monkeypatch.setattr(ocr_source_up, "acquire_project_claim", lambda *args: None)
    monkeypatch.setattr(ocr_source_up, "has_project_claim", lambda *args: False)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    if bundle_altered:
        (release / "bundle.json").write_text("altered")
        with pytest.raises(ValueError, match="Prepared release file identity changed"):
            ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)
        assert not calls
        return
    ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)
    assert calls[-1] == (
        "docker",
        "compose",
        "--project-name",
        "original",
        "-f",
        str(tmp_path / "compose.local.json"),
        "up",
        "--detach",
        "--no-build",
        "--wait",
        "--wait-timeout",
        "300",
    )
    assert os.environ["OCR_CUDA_DEVICE_ID"] == "2"


def test_fixture_inputs_reject_writable_ancestor(tmp_path: Path) -> None:
    from scripts.ocr_source_up import stage_approved_inputs

    approved = tmp_path / "approved"
    approved.mkdir()
    approved.chmod(0o770)
    (approved / "bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="trusted directory"):
        stage_approved_inputs(approved, tmp_path / "staged", "cuda")


def test_restart_rejects_a_different_container_image_in_its_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_release: tuple[Path, str, str],
) -> None:
    from scripts import ocr_source_up

    release, native_id, bundle_digest = managed_release
    app_id = "sha256:" + "e" * 64
    overlay = ocr_source_up.managed_overlay(
        "cuda",
        release,
        native_id,
        app_id,
        "original",
        {"OCR_CUDA_DEVICE_ID": "2"},
        bundle_digest,
        config_id=native_id,
    )
    (tmp_path / "compose.local.json").write_text(json.dumps(overlay))

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if args[1:3] == ("image", "inspect"):
            return app_id if args[-1].startswith("vision-bill:") else native_id
        if "ps" in args:
            return "active-ocr-id" if args[-1] == "paddleocr-service" else ""
        if args[1:3] == ("volume", "ls"):
            return ""
        if args[1:3] == ("container", "inspect"):
            return json.dumps([{"Image": "sha256:" + "a" * 64}])
        pytest.fail(f"Unexpected subprocess command: {args}")

    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    monkeypatch.setattr(ocr_source_up, "acquire_project_claim", lambda *args: None)
    monkeypatch.setattr(ocr_source_up, "has_project_claim", lambda *args: False)
    with pytest.raises(ValueError, match="different OCR image"):
        ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)


def test_prepare_rejects_the_other_accelerator_record_in_its_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    prepared = tmp_path / "cuda"
    prepared.mkdir()
    overlay = {"x-vision-bill-source": {"project": "original"}}
    (prepared / "compose.local.json").write_text(json.dumps(overlay))
    state = tmp_path / "rocm"
    state.mkdir()

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "config" in args:
            return json.dumps({"name": "original"})
        pytest.fail(f"Unexpected subprocess command: {args}")

    for key in ("VB_IMAGE_TAG", "OCR_IMAGE_TAG"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    with pytest.raises(ValueError, match="another accelerator"):
        ocr_source_up.deploy(("docker", "compose"), state, "rocm", tmp_path / "approved")


@pytest.mark.parametrize(
    ("version", "expected"),
    [(100304, "gfx1034"), (110001, "gfx1101"), (120001, "gfx1201"), (90010, "gfx90a")],
)
def test_rocm_target_matches_the_selected_render_minor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int, expected: str
) -> None:
    from scripts import ocr_source_up

    for node, minor, target in ((0, 0, 0), (1, 128, 100300), (2, 129, version)):
        directory = tmp_path / str(node)
        directory.mkdir()
        (directory / "properties").write_text(
            f"drm_render_minor {minor}\nvendor_id 4098\ngfx_target_version {target}\n"
        )
    original_stat = Path.stat

    def device_stat(path: Path, *args, **kwargs):
        if path == Path("/dev/dri/renderD129"):
            return SimpleNamespace(st_rdev=os.makedev(226, 129))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", device_stat)
    assert ocr_source_up.rocm_render_target(Path("/dev/dri/renderD129"), tmp_path) == expected


@pytest.mark.parametrize("nodes", [(), (100300, 110000), (0,), (110099,)])
def test_rocm_target_refuses_missing_ambiguous_or_invalid_kfd_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nodes: tuple[int, ...]
) -> None:
    from scripts import ocr_source_up

    for node, version in enumerate(nodes):
        directory = tmp_path / str(node)
        directory.mkdir()
        (directory / "properties").write_text(
            f"drm_render_minor 129\nvendor_id 4098\ngfx_target_version {version}\n"
        )
    original_stat = Path.stat
    render = Path("/dev/dri/renderD129")

    def device_stat(path: Path, *args, **kwargs):
        if path == render:
            return SimpleNamespace(st_rdev=os.makedev(226, 129))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", device_stat)
    with pytest.raises(ValueError, match="set OCR_AMDGPU_TARGETS"):
        ocr_source_up.rocm_render_target(render, tmp_path)


def test_automatic_rocm_build_uses_the_selected_target_and_keeps_explicit_build_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    class BuildReached(Exception):
        pass

    target = ""
    selected_builds = []
    detected = []

    def command(*args: str, capture: bool = False) -> str:
        if "config" in args:
            return json.dumps({
                "name": "test",
                "services": {"paddleocr-service": {
                    "image": "vision-bill-ocr-rocm:candidate",
                    "build": {"args": {"AMDGPU_TARGETS": target}},
                }},
            })
        if args[-2:] == ("build", "paddleocr-service"):
            selected_builds.append(os.environ["OCR_AMDGPU_TARGETS"])
            raise BuildReached
        pytest.fail(f"Unexpected subprocess command: {args}")

    def detect(render: Path) -> str:
        detected.append(render)
        return "gfx1201"

    monkeypatch.setattr(ocr_source_up, "command", command)
    monkeypatch.setattr(ocr_source_up, "gpu_args", lambda *_: ([], {"OCR_RENDER_NODE": "/dev/dri/renderD129"}))
    monkeypatch.setattr(ocr_source_up, "rocm_render_target", detect, raising=False)
    for name in ("reject_other_profile", "reject_existing_ocr", "verify_existing_app", "stage_first_run_inputs"):
        monkeypatch.setattr(ocr_source_up, name, lambda *_, **_kwargs: None)
    for key in ("VB_IMAGE_TAG", "OCR_IMAGE_TAG", "OCR_AMDGPU_TARGETS", "OCR_RENDER_NODE"):
        monkeypatch.setenv(key, "")

    for selected_targets in ("", "gfx1100;gfx1201"):
        target = selected_targets
        with pytest.raises(BuildReached):
            ocr_source_up.deploy(("docker", "compose"), tmp_path, "rocm", None)
    assert selected_builds == ["gfx1201", "gfx1100;gfx1201"]
    assert detected == [Path("/dev/dri/renderD129")]


def test_project_claim_cannot_be_relabelled_by_another_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    volumes: dict[str, dict[str, Any]] = {}

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if args[:3] == ("docker", "volume", "create"):
            key, value = args[args.index("--label") + 1].split("=", 1)
            volumes.setdefault(args[-1], {"Labels": {key: value}})
            return args[-1]
        if args[:3] == ("docker", "volume", "inspect"):
            return json.dumps([volumes[args[-1]]])
        pytest.fail(f"Unexpected Docker call: {args}")

    identity = {"project": "old", "bundle_sha256": "a" * 64, "image_ids": {}}
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    monkeypatch.setattr(ocr_source_up, "ROOT", tmp_path / "first")
    ocr_source_up.acquire_project_claim(identity, "cuda", tmp_path / "first" / "release")
    ocr_source_up.acquire_project_claim(identity, "cuda", tmp_path / "first" / "release")
    monkeypatch.setattr(ocr_source_up, "ROOT", tmp_path / "second")
    with pytest.raises(ValueError, match="already claimed"):
        ocr_source_up.acquire_project_claim(identity, "rocm", tmp_path / "second" / "release")
    assert len(volumes) == 1


def test_existing_app_checks_actual_data_binds_even_with_matching_compose_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    monkeypatch.setattr(ocr_source_up, "ROOT", tmp_path)
    targets = ["logs", "uploads", "uploads_tmp", "config"]
    mounts = [
        {
            "Type": "bind",
            "Source": str(tmp_path / "server_data" / name),
            "Destination": f"/app/{name}",
            "RW": True,
        }
        for name in targets
    ]
    app = {
        "Config": {"Labels": {"com.docker.compose.project.working_dir": str(tmp_path)}},
        "Mounts": mounts,
    }

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "ps" in args:
            return "old-app"
        if args[:3] == ("docker", "container", "inspect"):
            return json.dumps([app])
        if "config" in args:
            volumes = [
                {
                    "type": "bind",
                    "source": str(tmp_path / "server_data" / name),
                    "target": f"/app/{name}",
                }
                for name in targets
            ]
            return json.dumps({"services": {"vision_bill": {"volumes": volumes}}})
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    ocr_source_up.verify_existing_app(("docker", "compose"), "original")
    mounts[1]["Source"] = "/mnt/c/original/server_data/uploads"
    with pytest.raises(ValueError, match="data bind origins"):
        ocr_source_up.verify_existing_app(("docker", "compose"), "original")


@pytest.mark.parametrize("has_db_container", [False, True])
def test_orphan_database_requires_known_prior_ownership(
    monkeypatch: pytest.MonkeyPatch, has_db_container: bool
) -> None:
    from scripts import ocr_source_up

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "ps" in args:
            return "existing-db" if has_db_container and args[-1] == "db" else ""
        if args[:3] == ("docker", "container", "inspect"):
            return json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project.working_dir": str(ocr_source_up.ROOT)
                            }
                        }
                    }
                ]
            )
        if args[:3] == ("docker", "volume", "ls"):
            assert "label=com.docker.compose.project=original" in args
            assert "label=com.docker.compose.volume=db_data" in args
            return "original_db_data"
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    with pytest.raises(ValueError, match="existing database volume"):
        ocr_source_up.verify_existing_app(("docker", "compose"), "original")
    ocr_source_up.verify_existing_app(("docker", "compose"), "original", allow_orphan=True)


def test_wsl_bind_translation_uses_unstarted_probe_and_removes_only_its_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import ocr_source_up

    source = "/home/alex/vision-bill/server_data/config"
    observed = "/run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/known"
    container_id = "a" * 64
    calls: list[tuple[str, ...]] = []

    def docker_boundary(*args: str, capture: bool = False) -> str:
        calls.append(args)
        if args[:2] == ("docker", "create"):
            assert "--pull" in args and "never" in args
            assert "--network" in args and "none" in args
            assert f"type=bind,source={source},target=/app/config" in args
            return container_id
        if args[:3] == ("docker", "container", "inspect"):
            assert args[-1] == container_id
            return json.dumps(
                [
                    {
                        "Mounts": [
                            {
                                "Destination": "/app/config",
                                "Source": observed,
                                "Type": "bind",
                                "RW": True,
                            }
                        ]
                    }
                ]
            )
        if args[:3] == ("docker", "container", "rm"):
            assert args[3:] == (container_id,)
            return ""
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(platform, "release", lambda: "6.6.0-microsoft-standard-WSL2")
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    expected = {"/app/config": source}
    assert ocr_source_up.bind_sources_match(
        expected, {"/app/config": observed}, "sha256:" + "c" * 64
    )
    assert not ocr_source_up.bind_sources_match(
        expected, {"/app/config": observed + "other"}, "sha256:" + "c" * 64
    )
    assert len(calls) == 6


def test_reserved_claim_cannot_adopt_database_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, managed_release: tuple[Path, str, str]
) -> None:
    from scripts import ocr_source_up

    release, native_id, bundle_digest = managed_release
    app_id = "sha256:" + "e" * 64
    overlay = ocr_source_up.managed_overlay(
        "cuda",
        release,
        native_id,
        app_id,
        "original",
        {"OCR_CUDA_DEVICE_ID": "2"},
        bundle_digest,
        config_id=native_id,
    )
    (tmp_path / "compose.local.json").write_text(json.dumps(overlay))

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if args[1:3] == ("image", "inspect"):
            return app_id if args[-1].startswith("vision-bill:") else native_id
        if "ps" in args:
            return ""
        if args[1:3] == ("volume", "ls"):
            return "original_db_data"
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(ocr_source_up, "has_project_claim", lambda *args: True)
    monkeypatch.setattr(ocr_source_up, "acquire_project_claim", lambda *args: None)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    with pytest.raises(ValueError, match="existing database volume"):
        ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)


def test_upgrade_keeps_original_claim_across_successive_image_changes(tmp_path):
    from scripts import ocr_source_up as source

    first = tmp_path / "release-first"
    second = tmp_path / "release-second"
    original = {
        "project": "receipts",
        "bundle_sha256": "a" * 64,
        "image_ids": {"vision_bill": "sha256:" + "1" * 64, source.NATIVE: "sha256:" + "2" * 64},
    }
    replacement = {**original, "bundle_sha256": "b" * 64}
    replacement["claim_origin"] = source.claim_origin(original, first)
    assert source.project_claim(replacement, "cuda", second) == source.project_claim(
        original, "cuda", first
    )
    assert source.claim_origin(replacement, second) == replacement["claim_origin"]
    assert source.project_claim(replacement, "rocm", second) != source.project_claim(
        original, "cuda", first
    )


def test_upgrade_verifies_the_admitted_original_before_reusing_volumes(
    tmp_path, managed_release, monkeypatch
):
    from scripts import ocr_source_up as source

    release, config_id, digest = managed_release
    overlay = source.managed_overlay(
        "cuda",
        release,
        config_id,
        config_id,
        "saved",
        {"OCR_CUDA_DEVICE_ID": "2"},
        digest,
        config_id=config_id,
    )
    managed = tmp_path / "compose.local.json"
    managed.write_text(json.dumps(overlay))
    state = {"claim": False, "running": ""}
    monkeypatch.setattr(source, "has_project_claim", lambda *args: state["claim"])
    monkeypatch.setattr(source, "command", lambda *args, **kwargs: state["running"])
    for recorded in (False, True):
        if recorded:
            source.record_admission(managed)
        with pytest.raises(ValueError, match="admitted"):
            source.deploy(("docker", "compose"), tmp_path, "cuda", None, upgrade=True)
    state.update(claim=True, running="old-container")
    with pytest.raises(ValueError, match="compose down"):
        source.deploy(("docker", "compose"), tmp_path, "cuda", None, upgrade=True)


@pytest.mark.parametrize(("configured", "expected"), [("", "2"), ("0", "0"), ("1", "1")])
def test_wsl_upgrade_resolves_explicit_adapter_before_rebinding(
    configured, expected, tmp_path, managed_release, monkeypatch
):
    from scripts import ocr_source_up as source

    release, image_id, _ = managed_release
    bundle = release / "bundle.json"
    manifest = json.loads(bundle.read_bytes())
    manifest["accelerator"] = "rocm"
    bundle.write_text(json.dumps(manifest))
    overlay = source.managed_overlay(
        "rocm", release, image_id, image_id, "saved", {"OCR_ROCM_DEVICE_ID": "2"},
        source.path_digest(bundle), config_id=image_id,
    )
    managed = tmp_path / "compose.local.json"
    managed.write_text(json.dumps(overlay))
    source.record_admission(managed)
    monkeypatch.setattr(source, "has_project_claim", lambda *args: True)
    monkeypatch.delenv("OCR_ROCM_DEVICE_ID", raising=False)

    def command(*args, capture=False):
        if args[-2:] == ("config", "--environment"):
            return f"OCR_ROCM_DEVICE_ID={configured}"
        assert "ps" in args and os.environ["OCR_ROCM_DEVICE_ID"] == expected
        return "running"

    monkeypatch.setattr(source, "command", command)
    with pytest.raises(ValueError, match="compose down"):
        source.deploy(("docker", "compose"), tmp_path, "rocm", None, upgrade=True)


def test_cutover_keeps_admission_if_interrupted_before_or_after_pointer_change(tmp_path):
    from scripts import ocr_source_up as source

    managed = tmp_path / "compose.local.json"
    managed.write_text("before")
    pending = tmp_path / "next.json"
    pending.write_text("after")
    source.record_admission(managed, pending)
    assert source.was_admitted(managed)
    pending.replace(managed)
    assert source.was_admitted(managed)
    managed.write_text("foreign")
    assert not source.was_admitted(managed)


@pytest.mark.parametrize(
    ("shell", "dotenv", "expected"),
    [(None, "", "cu126"), (None, "cu129", "cu129"), ("cu129", "cu126", "cu129")],
)
def test_source_upgrade_channel_only_changes_on_explicit_selection(
    monkeypatch, shell, dotenv, expected
):
    from scripts import ocr_source_up as source

    def compose(*args, capture=False):
        assert args == ("docker", "compose", "config", "--environment") and capture
        return f"OCR_CUDA_CHANNEL={dotenv}\n"

    monkeypatch.setattr(source, "command", compose)
    monkeypatch.delenv("OCR_CUDA_CHANNEL", raising=False)
    if shell:
        monkeypatch.setenv("OCR_CUDA_CHANNEL", shell)
    overlay = {"build": {"args": {"OCR_CUDA_CHANNEL": "cu129"}}}
    assert (
        source.cuda_build_channel(("docker", "compose"), overlay, {"cuda_channel": "cu126"})
        == expected
    )


@pytest.mark.parametrize("failure", [None, "build", "pending_ocr", "volume_mismatch", "late_pending"])
@pytest.mark.parametrize("reuse", [True, False])
def test_upgrade_builds_and_restarts_the_saved_project_without_dropping_data(
    tmp_path, managed_release, monkeypatch, failure, reuse
):
    from scripts import ocr_source_up as source

    old_release, config_id, digest = managed_release
    monkeypatch.delenv("OCR_CUDA_CHANNEL", raising=False)
    new_id = "sha256:" + "c" * 64
    project = "saved-project"
    prior = source.managed_overlay(
        "cuda",
        old_release,
        config_id,
        config_id,
        project,
        {"OCR_CUDA_DEVICE_ID": "2"},
        digest,
        config_id=config_id,
    )
    if not reuse:
        prior["services"][source.NATIVE]["environment"]["OCR_CUDA_EXPECTED_UUID"] = "a" * 32
    managed = tmp_path / "compose.local.json"
    managed.write_text(json.dumps(prior))
    source.record_admission(managed)
    commands = []
    service = source.NATIVE
    built_config = {
        "name": project,
        "services": {
            service: {
                "image": "native-new",
                "deploy": {"resources": {"reservations": {"devices": [{"device_ids": ["2"]}]}}},
            },
            "vision_bill": {"image": "app-new"},
        },
    }

    volume_name = project + "_ocr-coordination"
    built_config["volumes"] = {"ocr-coordination": {"name": volume_name}}
    for name in (service, "vision_bill"):
        built_config["services"][name]["volumes"] = [
            {"type": "volume", "source": "ocr-coordination", "target": "/coord"}
        ]
    built_config["services"][service]["environment"] = {
        "OCR_HANDOFF_LOCK_PATH": "/coord/gpu-handoff.lock"
    }
    built_config["services"]["vision_bill"]["environment"] = {
        "OCR_ACTIVATION_HANDOFF_LOCK_PATH": "/coord/gpu-handoff.lock",
    }

    candidate = "vision-bill-ocr-cuda:source-build-" + "a" * 32 if reuse else None

    def deploy():
        source.deploy(
            ("docker", "compose"), tmp_path, "cuda", None, upgrade=True, reuse_native_image=candidate
        )

    def docker(*args, capture=False):
        commands.append(args)
        if (
            args[:3] == ("docker", "volume", "ls")
            and args[-1] == "label=com.docker.compose.volume=ocr-coordination"
        ):
            assert f"label=com.docker.compose.project={project}" in args
            return volume_name if failure != "volume_mismatch" else "other-volume"
        if args[:2] == ("docker", "run") and "--mount" in args:
            assert source.JOURNAL_INSPECTOR in args and config_id not in args and "--gpus" not in args
            assert "--pull=missing" in args and "--network=none" in args and "--read-only" in args
            assert not any("type=bind" in item for item in args)
            assert "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=1m" in args
            assert args[args.index("--mount") + 1] == (
                f"type=volume,source={volume_name},target=/coord,readonly,volume-nocopy"
            )
            if failure == "pending_ocr" or (
                failure == "late_pending" and sum(source.JOURNAL_INSPECTOR in call for call in commands) == 2
            ):
                return "pending gpu-handoff.ocr-only.json"
            return "clean"
        if "--environment" in args:
            return "COMPOSE_PROJECT_NAME=saved-project\n"
        if args[:2] == ("docker", "run") and "--volume" not in args:
            assert config_id in args and "--gpus" not in args and "none" in args
            return "cu126\n"
        if "--check" in args:
            assert args[args.index("--channel") + 1] == "cu126"
            if "--build-context" in args:
                assert not reuse and args[-2:] == ("--prior-image", config_id)
                return str(SCRIPT.parents[1] / "paddleocr-service")
            assert "--reuse-image" in args
            return "f" * 64
        if "config" in args:
            assert args[args.index("--project-name") + 1] == project
            return json.dumps(built_config)
        if (
            "ps" in args
            or "ls" in args
            or args[:2] == ("make", "ocr-build-cuda")
            or args[0] == "uv"
        ):
            return ""
        if args[:3] == ("docker", "image", "inspect"):
            return new_id
        if args[:2] == ("docker", "run"):
            assert (("OCR_CUDA_EXPECTED_UUID=" + "a" * 32) in args) is (not reuse)
            staged = Path(args[args.index("--volume") + 1].removesuffix(":/release:ro"))
            manifest = json.loads((staged / "bundle.json").read_text())
            if reuse:
                expected = json.loads(args[args.index("-c") + 5])
                assert expected == source.checkout_sources()
                manifest["source_files"] = expected
            return json.dumps(dict(manifest, _bound_cuda_uuid="a" * 32))
        if "build" in args and failure == "build":
            raise subprocess.CalledProcessError(2, args)
        if "tag" in args or "build" in args or "up" in args:
            return ""
        pytest.fail(args)

    monkeypatch.setattr(source, "command", docker)
    monkeypatch.setattr(source, "has_project_claim", lambda *args: True)
    monkeypatch.setattr(source, "acquire_project_claim", lambda *args: None)

    def image_config(release, image_id):
        (release / "image-config.json").write_bytes(
            (old_release / "image-config.json").read_bytes()
        )
        return config_id

    monkeypatch.setattr(source, "export_config", image_config)
    if failure in {"pending_ocr", "volume_mismatch"}:
        expected = "gpu-handoff.ocr-only.json" if failure == "pending_ocr" else "coordination volume"
        with pytest.raises(ValueError, match=expected):
            deploy()
        assert json.loads(managed.read_bytes()) == prior
        assert list(tmp_path.glob("release-*")) == [old_release]
        assert not any(
            "build" in call or "up" in call or call[:1] == ("make",) for call in commands
        )
        return
    if failure == "late_pending":
        with pytest.raises(ValueError, match="gpu-handoff.ocr-only.json"):
            deploy()
        assert json.loads(managed.read_bytes()) == prior
        assert source.was_admitted(managed)
        assert list(tmp_path.glob("release-*")) == [old_release]
        assert not any("up" in call for call in commands)
        return
    if failure == "build":
        with pytest.raises(subprocess.CalledProcessError):
            deploy()
        assert json.loads(managed.read_bytes()) == prior
        assert list(tmp_path.glob("release-*")) == [old_release]
    else:
        deploy()
        saved = json.loads(managed.read_bytes())["x-vision-bill-source"]
        assert (
            saved["project"] == project
            and saved["device_environment"] == prior["x-vision-bill-source"]["device_environment"]
        )
        assert saved["image_ids"][service] == new_id
        assert saved["cuda_channel"] == "cu126"
        assert json.loads(managed.read_bytes())["services"][service]["environment"]["OCR_CUDA_EXPECTED_UUID"] == "a" * 32
        assert saved["claim_origin"]["release"] == old_release.name
        assert "--no-build" in commands[-1] and project in commands[-1]
    assert source.was_admitted(managed)
    assert (("make", "ocr-build-cuda") in commands) is (not reuse)
    assert all("rm" not in args and "down" not in args for args in commands)


@pytest.mark.parametrize(
    "journal,pending,verdict",
    [
        (None, False, "clean"),
        ("gpu-handoff.ocr-only.json", False, "clean"),
        ("gpu-handoff.ocr-only.json", True, "pending gpu-handoff.ocr-only.json"),
        ("gpu-handoff.json", True, "pending gpu-handoff.json"),
    ],
)
def test_offline_upgrade_preflight_uses_both_real_journal_checks_without_writes(
    tmp_path, journal, pending, verdict
):
    from scripts import ocr_source_up as source

    if journal:
        data = {
            "version": 2,
            "coordinator": "vision-bill-model-handoff-v2",
            "generation": 1,
            "resources": {"ocr": "paddle:example", "llm": "external-unmanaged"},
            "ocr_pending": pending,
            "unsettled": False,
            "models": [],
        }
        record = tmp_path / journal
        record.write_text(json.dumps(data))
        record.chmod(0o600)
        (tmp_path / "gpu-handoff.lock").touch(mode=0o600)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            source.CHECK_UPGRADE_HANDOFF,
            str(tmp_path / "gpu-handoff.lock"),
        ],
        cwd=source.ROOT / "paddleocr-service",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == verdict
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()}


@pytest.mark.parametrize("channel", ["cu126", "cu129"])
def test_legacy_cuda_channel_uses_verified_install_run_after_image_deleted(
    tmp_path, monkeypatch, channel
):
    from scripts import ocr_source_up as source

    config = tmp_path / "image-config.json"
    default = {"created_by": "ARG OCR_CUDA_CHANNEL=cu129"}
    instruction = (
        f"RUN |1 OCR_CUDA_CHANNEL={channel} /bin/sh -c python /opt/ocr/build/resolve_cuda.py "
        '--check /opt/locks/requirements.cuda.lock --channel "$OCR_CUDA_CHANNEL" '
        "&& python -m pip install --require-hashes --no-deps -r /opt/locks/requirements.cuda.lock"
    )
    previous = {"image_ids": {source.NATIVE: "sha256:" + "a" * 64}}

    def command(*args, capture=False):
        if args[-1] == "--environment":
            return ""
        raise subprocess.CalledProcessError(125, args)

    monkeypatch.setattr(source, "command", command)
    monkeypatch.delenv("OCR_CUDA_CHANNEL", raising=False)
    config.write_text(json.dumps({"history": [default, {"created_by": instruction}]}))
    assert source.cuda_build_channel(("docker", "compose"), {}, previous, config) == channel
    config.write_text(json.dumps({"history": [default]}))
    with pytest.raises(ValueError, match="Set OCR_CUDA_CHANNEL"):
        source.cuda_build_channel(("docker", "compose"), {}, previous, config)


def test_upgrade_inspector_respects_another_owner_and_rejects_unsafe_evidence(tmp_path):
    import fcntl

    from scripts import ocr_source_up as source

    lock = tmp_path / "gpu-handoff.lock"

    def inspect():
        return subprocess.check_output(
            [sys.executable, "-B", "-c", source.CHECK_UPGRADE_HANDOFF, str(lock)],
            cwd=source.ROOT / "paddleocr-service",
            text=True,
        ).strip()

    with lock.open("w") as owner:
        lock.chmod(0o600)
        fcntl.flock(owner, fcntl.LOCK_EX)
        assert inspect() == "busy"
    journal = lock.with_suffix(".json")
    journal.symlink_to(tmp_path / "missing")
    assert inspect() == "invalid gpu-handoff.json"
    assert journal.is_symlink()


@pytest.mark.parametrize(
    ("initial", "after", "expect_recovery", "error"),
    [
        ("clean", None, False, None),
        ("pending gpu-handoff.ocr-only.json", "clean", True, None),
        (
            "pending gpu-handoff.ocr-only.json",
            "invalid gpu-handoff.json",
            True,
            "failed validation",
        ),
        ("pending gpu-handoff.json", None, False, "shared GPU handoff"),
        ("invalid gpu-handoff.ocr-only.json", None, False, "failed validation"),
        ("busy", None, False, "lock is held"),
    ],
)
def test_recovery_dispatch_requires_specific_journal_and_rechecks_clean(
    monkeypatch, initial, after, expect_recovery, error
):
    from scripts import ocr_source_up as source

    volume = "original_ocr-coordination"
    lock = "/coord/gpu-handoff.lock"
    mounts = [{"target": "/coord", "source": "ocr-coordination", "type": "volume"}]
    config = {
        "volumes": {"ocr-coordination": {"name": volume}},
        "services": {
            "vision_bill": {
                "environment": {"OCR_ACTIVATION_HANDOFF_LOCK_PATH": lock},
                "volumes": mounts,
            },
            source.NATIVE: {"environment": {"OCR_HANDOFF_LOCK_PATH": lock}, "volumes": mounts},
        },
    }
    previous = {"project": "original"}
    verdicts = iter([initial, after] if after is not None else [initial])

    def docker(*args, capture=False):
        assert source.JOURNAL_INSPECTOR in args and "--user=0:0" in args
        assert "--read-only" in args and "--network=none" in args
        return next(verdicts)

    calls = []

    def recover(config_arg, previous_arg, *, run, helper_image):
        assert (config_arg, previous_arg, run, helper_image) == (
            config,
            previous,
            docker,
            source.JOURNAL_INSPECTOR,
        )
        calls.append(True)

    monkeypatch.setattr(source, "command", docker)
    monkeypatch.setattr(source, "compose_volumes", lambda *_: [volume])
    monkeypatch.setattr(
        source.importlib,
        "import_module",
        lambda *_, **kw: SimpleNamespace(
            recover_stopped_ocr=recover, inspection_program=lambda code: code
        ),
    )
    if error:
        with pytest.raises(ValueError, match=error):
            source.verify_upgrade_handoff(config, previous, recover_stopped_ocr=True)
    else:
        source.verify_upgrade_handoff(config, previous, recover_stopped_ocr=True)
    assert calls == ([True] if expect_recovery else [])
    assert list(verdicts) == []


def test_stopped_ocr_recovery_flag_requires_upgrade():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "cuda", "--recover-stopped-ocr"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--recover-stopped-ocr requires --upgrade" in result.stderr

@pytest.mark.parametrize("reason", ["missing", "hardlink", "executable"])
def test_preflight_journals_require_original_private_lock(tmp_path, reason):
    from ocr_service.initialization_lock import inspect_stopped_handoff

    lock = tmp_path / "gpu-handoff.lock"
    record = lock.with_suffix(".ocr-only.json")
    record.touch(mode=0o600)
    record.write_text(json.dumps({
        "version": 2, "coordinator": "vision-bill-model-handoff-v2", "generation": 1,
        "resources": {"llm": "external-unmanaged", "ocr": "local"},
        "models": [], "ocr_pending": False, "unsettled": False,
    }))
    if reason == "hardlink":
        lock.touch(mode=0o600)
        os.link(lock, tmp_path / "alias")
    elif reason == "executable":
        lock.touch(mode=0o700)
    assert inspect_stopped_handoff(lock) == "invalid gpu-handoff.lock"
