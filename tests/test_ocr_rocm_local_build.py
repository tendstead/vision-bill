"""Host selection is metadata-only; these tests run without GPU drivers or Docker."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import ocr_build_rocm
from scripts import ocr_source_up as source


def agent(number, kind, name, vendor="AMD"):
    return f"Agent {number}\n***\n  Name: {name}\n  Vendor Name: {vendor}\n  Device Type: {kind}\n"


GPU = agent(2, "GPU", "gfx1100")
CPU = agent(1, "CPU", "AMD Ryzen")


@pytest.mark.parametrize(
    ("output", "index", "target"),
    [
        (CPU + GPU, "0", "gfx1100"),
        (CPU, "0", None),
        (CPU + GPU + agent(3, "GPU", "gfx1201"), "0", None),
        (GPU, "1", None),
        (agent(1, "GPU", "gfx1100", "other"), "0", None),
    ],
)
def test_wsl_only_uses_one_unmasked_gpu(monkeypatch, output, index, target):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")

    def rocminfo(args, **options):
        assert "HIP_VISIBLE_DEVICES" not in options["env"]
        assert options["timeout"] > 0
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(source.subprocess, "run", rocminfo)
    if target:
        assert source.wsl_rocm_target(index) == target
    else:
        with pytest.raises(ValueError, match="OCR_AMDGPU_TARGETS"):
            source.wsl_rocm_target(index)


def test_wsl_missing_probe_reports_manual_target(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(source.subprocess, "run", missing)
    with pytest.raises(ValueError, match="OCR_ROCM_DEVICE_ID"):
        source.wsl_rocm_target("0")


def test_render_selection_uses_only_amd_and_honors_compose_env(tmp_path, monkeypatch):
    def add_node(number, vendor, minor):
        folder = tmp_path / str(number)
        folder.mkdir()
        (folder / "properties").write_text(
            f"vendor_id {vendor}\ndrm_render_minor {minor}\ngfx_target_version 110000\n"
        )

    monkeypatch.delenv("OCR_RENDER_NODE", raising=False)
    add_node(0, 0, 0)
    add_node(1, 4098, 129)
    monkeypatch.setattr(source, "command", lambda *args, **kwargs: "")
    source.select_rocm_render_node(("docker", "compose"), tmp_path)
    assert os.environ.pop("OCR_RENDER_NODE") == "/dev/dri/renderD129"
    add_node(2, 4098, 130)
    with pytest.raises(ValueError, match="Multiple AMD render nodes"):
        source.select_rocm_render_node(("docker", "compose"), tmp_path)
    monkeypatch.setattr(
        source, "command", lambda *args, **kwargs: "OCR_RENDER_NODE=/dev/dri/renderD130\n"
    )
    (tmp_path / "1/properties").write_text("malformed-property\n")
    source.select_rocm_render_node(("docker", "compose"), tmp_path)
    assert "OCR_RENDER_NODE" not in os.environ


def test_source_refuses_wrong_transport_before_build():
    native = {"build": {"args": {"AMDGPU_TARGETS": "gfx1100", "OCR_ROCM_TRANSPORT": "kfd"}}}
    with pytest.raises(ValueError, match="requires OCR_ROCM_TRANSPORT=wsl"):
        source.configure_rocm_build(native, {"OCR_ROCM_DEVICE_ID": "0"})


@pytest.mark.parametrize(
    ("explicit", "probe", "expected"),
    [
        ("gfx1201;gfx1030", None, "gfx1030;gfx1201"),
        ("", "gfx1100", "gfx1100"),
        ("", "gfx9999", None),
        ("", None, None),
    ],
)
def test_make_cross_or_auto_wsl_build(monkeypatch, explicit, probe, expected):
    calls = []
    monkeypatch.setenv("OCR_AMDGPU_TARGETS", "")
    monkeypatch.setenv("OCR_ROCM_TRANSPORT", "")
    monkeypatch.setattr(
        source,
        "source_compose",
        lambda accelerator: ("docker", "compose", "-f", "docker-compose.rocm-wsl.yml"),
    )
    transport = "dual" if explicit else "wsl"
    native = {"build": {"args": {"AMDGPU_TARGETS": explicit, "OCR_ROCM_TRANSPORT": transport}}}

    def command(*args, **kwargs):
        if "config" in args:
            return json.dumps({"services": {source.NATIVE: native}})
        assert args[-2:] == ("build", source.NATIVE)
        calls.append((os.environ["OCR_AMDGPU_TARGETS"], os.environ["OCR_ROCM_TRANSPORT"]))

    def detect(index):
        if not probe:
            raise ValueError("No selected target")
        return probe

    def devices(*args):
        assert not explicit
        return [], {"OCR_ROCM_DEVICE_ID": "0"}

    monkeypatch.setattr(source, "command", command)
    monkeypatch.setattr(source, "need_host", lambda *args, **kwargs: None)
    monkeypatch.setattr(source, "gpu_args", devices)
    monkeypatch.setattr(source, "wsl_rocm_target", detect)
    assert ocr_build_rocm.main() == (0 if expected else 1)
    assert calls == ([(expected, transport)] if expected else [])


def test_native_render_node_from_other_vendor_is_rejected(monkeypatch):
    render = "/dev/dri/renderD129"
    devices = [{"source": item, "target": item} for item in ("/dev/kfd", render)]
    vendor = Path("/sys/class/drm/renderD129/device/vendor")
    monkeypatch.setattr(Path, "is_file", lambda path: path == vendor)
    monkeypatch.setattr(Path, "read_text", lambda path: "0x8086\n")
    with pytest.raises(ValueError, match="not an AMD device"):
        source.gpu_args("rocm", {"devices": devices})
