"""Explicit source qualification shares the serving fences before touching evidence."""

from pathlib import Path
from unittest.mock import Mock

import pytest


def arguments(tmp_path: Path) -> list[str]:
    return [
        "--accelerator", "cuda",
        "--runtime", str(tmp_path / "candidate.json"),
        "--cases", str(tmp_path / "cases.toml"),
        "--output", str(tmp_path / "qualification.json"),
    ]


@pytest.mark.parametrize("intent", [[], ["--maintenance-intent", "external-llm-quiesced"]])
def test_requires_intent_and_original_pairing_mode_before_qualification(
    tmp_path, monkeypatch, intent
):
    from ocr_service import source_maintenance

    launch = Mock()
    monkeypatch.setattr(source_maintenance, "run_owned_qualification", launch)
    with pytest.raises(SystemExit, match="2"):
        source_maintenance.main([*arguments(tmp_path), *intent])
    launch.assert_not_called()
    assert not (tmp_path / "qualification.json").exists()


@pytest.mark.parametrize("fence", ["running_ocr", "crashed_ocr", "outstanding_app", "running_app"])
def test_source_qualification_leaves_previous_outputs_untouched_while_fenced(
    tmp_path, monkeypatch, fence
):
    from contextlib import ExitStack

    from ocr_service import source_maintenance, supervision

    native = tmp_path / "installation.lock"
    app = tmp_path / "config" / "gpu-handoff.lock"
    output = tmp_path / "qualification.json"
    runtime = tmp_path / "qualification.runtime.json"
    output.write_bytes(b"previous report")
    runtime.write_bytes(b"previous verified runtime")
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(native))
    monkeypatch.setattr(source_maintenance, "APP_CONFIG", app.parent)
    launch = Mock()
    monkeypatch.setattr(source_maintenance, "run_owned_qualification", launch)
    with ExitStack() as held:
        if fence == "running_ocr":
            held.enter_context(supervision.installation_lock(native, exclusive=False))
        elif fence == "crashed_ocr":
            supervision.marker_path(native, "ocr").write_text("{}")
        elif fence == "running_app":
            held.enter_context(supervision.installation_lock(app, exclusive=True))
        else:
            app.parent.mkdir()
            journal = app.with_suffix(".json")
            journal.write_text("{}")
            journal.chmod(0o600)
        with pytest.raises(SystemExit, match="1"):
            source_maintenance.main([
                *arguments(tmp_path),
                "--maintenance-intent", "external-llm-quiesced",
                "--handoff-lock", str(app),
            ])
    launch.assert_not_called()
    assert output.read_bytes() == b"previous report"
    assert runtime.read_bytes() == b"previous verified runtime"


def test_source_qualifier_holds_both_original_locks_through_its_child(tmp_path, monkeypatch):
    from ocr_service import source_maintenance, supervision

    native = tmp_path / "installation.lock"
    app = tmp_path / "config" / "same-original-app.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(native))
    monkeypatch.setattr(source_maintenance, "APP_CONFIG", app.parent)
    monkeypatch.setattr(supervision.os, "getpid", lambda: 1)
    monkeypatch.setattr(supervision, "_descendants", lambda: set())

    def child(command):
        assert command[1:3] == ["-m", "ocr_service.qualify"]
        assert supervision.marker_path(native, "initializer").is_file()
        with pytest.raises(supervision.InstallationBusy), supervision.installation_lock(
            native, exclusive=False
        ):
            pass
        with pytest.raises(supervision.InstallationBusy), supervision.installation_lock(
            app, exclusive=True
        ):
            pass
        return 0

    launch = Mock(side_effect=child)
    monkeypatch.setattr(source_maintenance, "run_owned_qualification", launch)
    with pytest.raises(SystemExit, match="0"):
        source_maintenance.main([
            *arguments(tmp_path),
            "--maintenance-intent", "separate-ocr-device",
            "--handoff-lock", str(app),
        ])
    launch.assert_called_once()
    assert not supervision.marker_path(native, "initializer").exists()
    with (
        supervision.installation_lock(native, exclusive=True),
        supervision.installation_lock(app, exclusive=True),
    ):
        pass


def test_old_direct_qualification_leaves_outputs_untouched(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    from ocr_service import qualify

    (tmp_path / "candidate.json").write_text("{}")
    output = tmp_path / "qualification.json"
    output.write_bytes(b"existing qualification")
    monkeypatch.setattr(sys, "argv", ["qualify", *arguments(tmp_path)])
    monkeypatch.setattr(
        qualify.RuntimeSettings, "model_validate_json", lambda _: SimpleNamespace(accelerator="cuda")
    )
    launch = Mock(return_value=SimpleNamespace(qualified=True))
    monkeypatch.setattr(qualify, "qualify_runtime", launch)
    with pytest.raises(SystemExit, match="2"):
        qualify.main()
    assert "source_maintenance" in capsys.readouterr().err
    launch.assert_not_called()
    assert output.read_bytes() == b"existing qualification"


def test_native_qualifier_accepts_only_a_live_initializer_inode(tmp_path, monkeypatch):
    from ocr_service import supervision

    lock = tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    monkeypatch.setattr(supervision.os, "getppid", lambda: 1)
    monkeypatch.setattr(supervision.os, "getpid", lambda: 1)
    monkeypatch.setattr(supervision, "_descendants", lambda: set())
    launch = Mock(return_value=0)
    monkeypatch.setattr(supervision, "run_owned", launch)
    with supervision.managed_installation("initializer", lock):
        assert supervision.run_owned_qualification(["fake-qualification"]) == 0
        command = launch.call_args.args[0]
        descriptor, = launch.call_args.kwargs["pass_fds"]
        assert command == ["fake-qualification", "--supervisor-fd", str(descriptor)]
        assert supervision.qualification_is_supervised(descriptor)
        assert not supervision.qualification_is_supervised(None)
        other = tmp_path / "different.lock"
        with other.open("w") as unrelated:
            assert not supervision.qualification_is_supervised(unrelated.fileno())
    assert not supervision.qualification_is_supervised(descriptor)


def test_native_qualification_child_and_local_stage_routes_remain_available(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from ocr_service import qualify, supervision

    lock = tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    monkeypatch.setattr(supervision.os, "getppid", lambda: 1)
    (tmp_path / "candidate.json").write_text("{}")
    monkeypatch.setattr(
        qualify.RuntimeSettings, "model_validate_json", lambda _: SimpleNamespace(accelerator="cuda")
    )
    launch = Mock(return_value=SimpleNamespace(qualified=True))
    monkeypatch.setattr(qualify, "qualify_runtime", launch)
    with supervision.installation_lock(lock, exclusive=True) as descriptor:
        supervision.marker_path(lock, "initializer").write_text("{}")
        monkeypatch.setattr(sys, "argv", [
            "qualify", *arguments(tmp_path), "--supervisor-fd", str(descriptor),
        ])
        with pytest.raises(SystemExit, match="0"):
            qualify.main()
    launch.assert_called_once()

    output = tmp_path / "stage.json"
    monkeypatch.setattr(qualify, "collect", lambda _: {"synthetic": True})
    monkeypatch.setattr(sys, "argv", [
        "qualify", "--stage", "collect", "--stage-input", str(tmp_path / "candidate.json"),
        "--output", str(output),
    ])
    qualify.main()
    assert '"synthetic": true' in output.read_text()


def test_qualifier_child_receives_the_live_descriptor(tmp_path, monkeypatch):
    import sys

    from ocr_service import supervision

    lock = tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    monkeypatch.setattr(supervision.os, "getpid", lambda: 1)
    monkeypatch.setattr(supervision, "_descendants", lambda: set())
    package = Path(supervision.__file__).resolve().parents[1]
    code = """
import os
import sys
sys.path.insert(0, sys.argv[1])
from ocr_service.supervision import qualification_is_supervised
os.getppid = lambda: 1
assert sys.argv[2] == "--supervisor-fd"
assert qualification_is_supervised(int(sys.argv[3]))
"""
    with supervision.managed_installation("initializer", lock):
        assert supervision.run_owned_qualification([sys.executable, "-c", code, str(package)]) == 0
