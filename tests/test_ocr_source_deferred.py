import json
from pathlib import Path

import pytest

from scripts import ocr_deferred
from scripts import ocr_source_up as source


def test_deferred_source_starts_app_and_database_before_waiting_for_activation(monkeypatch, tmp_path):
    app_id = "sha256:" + "b" * 64
    project = "receipt-deferred"
    commands = []
    phase = []
    orphan_checks = []
    revision = "a" * 40

    def command(*args, capture=False):
        commands.append(args)
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return revision
        if "config" in args:
            return json.dumps({"name": project, "services": {"vision_bill": {"image": "app-build"}}})
        return ""

    def wait(status_path, read_generation, build):
        assert status_path.is_file()
        assert any(cmd[-2:] == ("vision_bill", "db") and "up" in cmd for cmd in commands)
        assert not phase
        build()

    monkeypatch.setattr(source, "command", command)
    monkeypatch.setattr(source, "reject_other_profile", lambda *args: None)
    monkeypatch.setattr(source, "reject_existing_ocr", lambda *args: None)
    monkeypatch.setattr(source, "verify_existing_app", lambda *args, **kwargs: orphan_checks.append(kwargs))
    monkeypatch.setattr(source, "acquire_volume_claim", lambda *args, **kwargs: None)
    monkeypatch.setattr(source, "volume_has_claim", lambda *args: True)
    monkeypatch.setattr(source, "inspect_image", lambda *args: app_id)
    monkeypatch.setattr(source, "need_host", lambda *args, **kwargs: None)
    monkeypatch.setattr(source, "select_rocm_render_node", lambda *args: None)
    monkeypatch.setattr(source, "existing_controllers", lambda *args: [])
    monkeypatch.setattr(source, "deferred_app_container", lambda *args: "b" * 64)
    monkeypatch.setattr(ocr_deferred, "wait_for_ocr", wait)
    monkeypatch.setattr(source, "deploy", lambda *args, **kwargs: phase.append(kwargs))
    source.start_deferred(("docker", "compose"), tmp_path, "rocm", None)

    assert len(phase) == 1 and phase[0]["deferred_app"].image_id == app_id
    assert not any(cmd[-2:] == ("build", source.NATIVE) for cmd in commands)

    def build_calls():
        return [cmd for cmd in commands if cmd[-2:] == ("build", "vision_bill")]

    monkeypatch.setattr(ocr_deferred, "wait_for_ocr", lambda *args: None)
    source.start_deferred(("docker", "compose"), tmp_path, "rocm", None)
    assert len(build_calls()) == 1
    revision = "d" * 40
    source.start_deferred(("docker", "compose"), tmp_path, "rocm", None)
    assert len(build_calls()) == 2
    saved = json.loads((tmp_path / "deferred.app.json").read_bytes())
    assert saved["x-vision-bill-deferred"]["revision"] == revision
    assert len(phase) == 1
    assert [kwargs["allow_orphan"] for kwargs in orphan_checks] == [False, True, True]
    monkeypatch.setattr(source, "volume_has_claim", lambda *args: False)
    with pytest.raises(ValueError, match="on this Docker engine"):
        source.start_deferred(("docker", "compose"), tmp_path, "rocm", None)


def test_deferred_request_uses_only_verified_container_and_canonical_uuid(monkeypatch):
    image_id = "sha256:" + "b" * 64
    app = source.DeferredApp("test", image_id, Path("/safe/status"), "a" * 40)
    container_id = "c" * 64
    generation = "799eb605-20f0-43d1-ae60-dfe828be67eb"
    monkeypatch.setattr(source, "deferred_app_container", lambda *args: container_id)

    def command(*args, capture=False):
        assert args[:4] == ("docker", "exec", container_id, "python")
        return generation + "\n"

    monkeypatch.setattr(source, "command", command)
    assert source.deferred_generation(("docker", "compose"), app) == generation
    monkeypatch.setattr(source, "command", lambda *args, **kwargs: generation + "\nextra\n")
    with pytest.raises(ValueError, match="activation generation"):
        source.deferred_generation(("docker", "compose"), app)


def test_deferred_handoff_starts_only_the_native_controller(monkeypatch, tmp_path):
    native_id = "sha256:" + "a" * 64
    app = source.DeferredApp("handoff", "sha256:" + "b" * 64, tmp_path / "status", "c" * 40)
    calls = []

    def command(*args, capture=False):
        calls.append(args)
        if "config" in args:
            return json.dumps({"name": app.project, "services": {source.NATIVE: {"image": "native-build"}}})
        if args[:2] == ("docker", "run"):
            return json.dumps({"_bound_rocm_uuid": "d" * 32})
        return ""

    monkeypatch.setenv("VB_IMAGE_TAG", "before")
    monkeypatch.setenv("OCR_IMAGE_TAG", "before")
    monkeypatch.setattr(source, "command", command)
    monkeypatch.setattr(source, "gpu_args", lambda *args: ([], {}))
    monkeypatch.setattr(source, "configure_rocm_build", lambda *args: None)
    monkeypatch.setattr(source, "stage_first_run_inputs", lambda *args: None)
    monkeypatch.setattr(source, "export_config", lambda *args: native_id)
    monkeypatch.setattr(source, "inspect_image", lambda *args, **kwargs: native_id)
    monkeypatch.setattr(source, "deferred_app_container", lambda *args: "b" * 64)
    monkeypatch.setattr(source, "verify_existing_app", lambda *args, **kwargs: None)
    monkeypatch.setattr(source, "acquire_project_claim", lambda *args: None)

    source.deploy(("docker", "compose"), tmp_path, "rocm", None, deferred_app=app)

    assert [call[-1] for call in calls if "build" in call] == [source.NATIVE]
    up, = [call for call in calls if "up" in call]
    assert "--no-deps" in up and up[-1] == source.NATIVE
    managed = json.loads((tmp_path / "compose.local.json").read_bytes())
    assert managed["services"]["vision_bill"] == app.overlay()["services"]["vision_bill"]
