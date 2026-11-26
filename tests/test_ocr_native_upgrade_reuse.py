import json

import pytest
from test_ocr_source_up import managed_release  # noqa: F401

from scripts import ocr_cuda_probe, ocr_native_reuse
from scripts import ocr_source_up as source


def service_build(accelerator, root):
    build = {"context": str(root), "dockerfile": f"Dockerfile.{accelerator}"}
    if accelerator == "cuda":
        build["additional_contexts"] = {
            "cuda-locks": str(root),
            "cuda-wheel": str(root / "empty-wheel-context"),
        }
        build["args"] = {"OCR_CUDA_CHANNEL": "cu129"}
    else:
        build["args"] = {
            "AMDGPU_TARGETS": "gfx1100",
            "OCR_ROCM_TRANSPORT": "kfd",
            "BUILD_JOBS": "auto",
        }
    return build


@pytest.mark.parametrize(
    ("accelerator", "change"),
    [
        (accelerator, change)
        for accelerator in ("cuda", "rocm")
        for change in ("none", "gpu", "tag", "inputs", "legacy", "channel_or_transport")
    ]
    + [("rocm", "wsl_without_probe")],
)
def test_admitted_upgrade_reuses_only_proven_native_image(
    tmp_path,
    monkeypatch,
    managed_release,  # noqa: F811
    accelerator,
    change,
):
    old, config_id, _ = managed_release
    root = source.ROOT / source.NATIVE
    build = service_build(accelerator, root)
    selected = {key: value for key, value in build["args"].items() if key != "BUILD_JOBS"}
    devices = (
        {"OCR_CUDA_DEVICE_ID": "0"}
        if accelerator == "cuda"
        else {"OCR_RENDER_NODE": "/dev/dri/renderD128"}
    )
    if change == "wsl_without_probe":
        devices = {"OCR_ROCM_DEVICE_ID": "1"}
        selected["OCR_ROCM_TRANSPORT"] = build["args"]["OCR_ROCM_TRANSPORT"] = "wsl"
        build["args"]["AMDGPU_TARGETS"] = ""
    monkeypatch.delenv("OCR_CUDA_CHANNEL", raising=False)
    sources = source.checkout_sources()
    image = "sha256:" + "c" * 64
    bundle_file = old / "bundle.json"
    manifest = json.loads(bundle_file.read_bytes())
    manifest.update(
        accelerator=accelerator,
        image_identity=image,
        image_config_sha256=config_id[7:],
        source_files=sources,
    )
    bundle_file.write_text(json.dumps(manifest))
    overlay = source.managed_overlay(
        accelerator,
        old,
        image,
        config_id,
        "project",
        devices,
        source.path_digest(bundle_file),
        config_id=config_id,
    )
    identity = overlay["x-vision-bill-source"]
    if change != "legacy":
        fingerprint = ocr_native_reuse.build_fingerprint(root, accelerator, build, selected)
        assert fingerprint
        identity["native_build"] = {
            "inputs_sha256": fingerprint if change != "inputs" else "d" * 64,
            "selected_args": selected,
            "device_uuid": "1" * 32,
        }
    if accelerator == "cuda":
        identity["cuda_channel"] = "cu129"
        if "native_build" in identity:
            identity["native_build"]["cuda_lock_sha256"] = "a" * 64
    if change == "channel_or_transport":
        if accelerator == "cuda":
            monkeypatch.setenv("OCR_CUDA_CHANNEL", "cu126")
        else:
            build["args"]["OCR_ROCM_TRANSPORT"] = "dual"
    managed = tmp_path / "compose.local.json"
    managed.write_text(json.dumps(overlay))
    source.record_admission(managed)
    tag = overlay["services"][source.NATIVE]["image"]
    monkeypatch.setattr(ocr_cuda_probe, "preflight", lambda *args: "")
    monkeypatch.setattr(source, "gpu_args", lambda *args: ([], devices))
    monkeypatch.setattr(source, "has_project_claim", lambda *args: True)
    for name in (
        "acquire_project_claim",
        "verify_upgrade_handoff",
        "verify_existing_app",
        "reject_other_profile",
        "existing_controllers",
    ):
        monkeypatch.setattr(source, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        source, "export_config", lambda *args: pytest.fail("Image config re-exported")
    )
    calls = []

    class NeedNewBuild(Exception):
        pass

    def command(*args, capture=False):
        calls.append(args)
        if "--environment" in args or "ps" in args:
            return ""
        if "config" in args:
            return json.dumps(
                {
                    "name": "project",
                    "services": {
                        source.NATIVE: {"image": "native:new", "build": build},
                        "vision_bill": {"image": "app:new"},
                    },
                }
            )
        if args[:3] == ("docker", "image", "inspect"):
            if args[-1] == tag:
                return config_id if change == "tag" else image
            assert args[-1] == "app:new"
            return config_id
        if "--build-context" in args or args[-2:] == ("build", source.NATIVE):
            raise NeedNewBuild
        if "--reuse-image" in args:
            pytest.fail("Automatic reuse should work without a host-side lock cache")
        if source.BIND_BUNDLE in args:
            assert "--network" in args and "none" in args
            assert args[args.index(source.BIND_BUNDLE) + 1 :][:2] == (image, config_id)
            assert json.loads(args[args.index(source.BIND_BUNDLE) + 4]) == sources
            return json.dumps(
                {**manifest, f"_bound_{accelerator}_uuid": ("2" if change == "gpu" else "1") * 32}
            )
        if "tag" in args or args[-2:] == ("build", "vision_bill") or "up" in args:
            return ""
        pytest.fail(args)

    monkeypatch.setattr(source, "command", command)
    if change in {"tag", "inputs", "legacy", "channel_or_transport"}:
        with pytest.raises(NeedNewBuild):
            source.deploy(("docker", "compose"), tmp_path, accelerator, None, upgrade=True)
    elif change == "gpu":
        with pytest.raises(ValueError, match="Selected GPU differs"):
            source.deploy(("docker", "compose"), tmp_path, accelerator, None, upgrade=True)
    else:
        source.deploy(("docker", "compose"), tmp_path, accelerator, None, upgrade=True)
        saved = json.loads(managed.read_bytes())
        assert saved["x-vision-bill-source"]["image_ids"][source.NATIVE] == image
        assert saved["x-vision-bill-source"]["native_build"] == identity["native_build"]
        assert any(args[-2:] == ("build", "vision_bill") for args in calls)
        assert source.was_admitted(managed)
        return
    assert json.loads(managed.read_bytes()) == overlay
    assert source.was_admitted(managed)


def test_build_context_change_invalidates_reuse(tmp_path):
    root = tmp_path / source.NATIVE
    root.mkdir()
    (root / ".dockerignore").write_text("*\n!Dockerfile.rocm\n!locks/\nlocks/**\n!locks/*.lock\n")
    (root / "Dockerfile.rocm").write_text("FROM scratch\n")
    locks = root / "locks"
    locks.mkdir()
    lock = locks / "input.lock"
    lock.write_text("A")
    build = service_build("rocm", root)
    before = ocr_native_reuse.build_fingerprint(root, "rocm", build, {})
    assert before
    (tmp_path / "app.py").write_text("An application-only change")
    assert ocr_native_reuse.build_fingerprint(root, "rocm", build, {}) == before
    lock.write_text("B")
    assert ocr_native_reuse.build_fingerprint(root, "rocm", build, {}) != before
    lock.write_text("A")
    assert (
        ocr_native_reuse.build_fingerprint(root, "rocm", build, {"AMDGPU_TARGETS": "gfx1201"})
        != before
    )
    (root / "Dockerfile.rocm.dockerignore").touch()
    assert ocr_native_reuse.build_fingerprint(root, "rocm", build, {}) is None
