"""CPU container evidence for PID-1 process ownership and durable crash fences."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "paddleocr-service/ocr_service/supervision.py"
IMAGE = "python:3.12-slim"


def test_host_automatic_native_supervision_is_rejected_before_marker(tmp_path):
    from ocr_service.supervision import managed_installation

    if os.getpid() == 1:
        pytest.skip("Host refusal is exercised outside the managed PID namespace")
    with (
        pytest.raises(RuntimeError, match="PID 1"),
        managed_installation("initializer", tmp_path / "installation.lock"),
    ):
        pytest.fail("Host execution must not enter the native scope")
    assert not list(tmp_path.glob("*.json"))


@pytest.fixture
def cpu_containers(tmp_path):
    if (
        not shutil.which("docker")
        or subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("Owned CPU Docker image unavailable")
    containers = []

    def start(command, *, role="ocr", supervise=True, ocr_entrypoint=False):
        name = "vb-task6-" + uuid4().hex
        containers.append(name)
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "-v",
            f"{HELPER}:/supervision.py:ro",
            "-v",
            f"{tmp_path}:/coord",
        ]
        if ocr_entrypoint:
            args += [
                "-v",
                f"{ROOT / 'paddleocr-service'}:/app:ro",
                "-e",
                "PYTHONPATH=/app",
                "-e",
                "OCR_INSTALLATION_LOCK_PATH=/coord/installation.lock",
                "-e",
                "OCR_RUNTIME_SETTINGS_PATH=/manual/runtime.json",
                "-e",
                "OCR_QUALIFICATION_IMAGE_PATH=/manual/smoke.png",
            ]
        args += [IMAGE, "python"]
        if ocr_entrypoint:
            args += ["-m", "ocr_service.entrypoint", "--"] + command
        elif supervise:
            args += [
                "/supervision.py",
                "--role",
                role,
                "--lock",
                "/coord/installation.lock",
                "--",
            ] + command
        else:
            args += command
        subprocess.run(args, check=True, capture_output=True, text=True)
        return name

    yield tmp_path, start
    for name in reversed(containers):
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


def wait_file(path, *, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text()
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for owned CPU artifact {path}")


def exit_code(name):
    return int(
        subprocess.run(
            ["docker", "wait", name], capture_output=True, text=True, check=True, timeout=15
        ).stdout
    )


@pytest.mark.parametrize("signum", ["HUP", "QUIT", "TERM", "INT"])
def test_supervisor_reaps_detached_child_before_clean_marker_removal(cpu_containers, signum):
    coord, start = cpu_containers
    child = "import os,time;from pathlib import Path;Path('/coord/grandchild').write_text(str(os.getpid()));time.sleep(60)"
    parent = (
        "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"
        + repr(child)
        + "],start_new_session=True);time.sleep(60)"
    )
    name = start(["python", "-c", parent], role="initializer")
    wait_file(coord / "grandchild")
    assert (coord / "installation.active-initializer.json").is_file()
    subprocess.run(["docker", "kill", "--signal", signum, name], check=True, capture_output=True)
    assert exit_code(name) != 0
    assert not (coord / "installation.active-initializer.json").exists()
    replacement = start(
        ["python", "-c", "from pathlib import Path;Path('/coord/replaced').write_text('clean')"],
        role="initializer",
    )
    assert exit_code(replacement) == 0
    assert wait_file(coord / "replaced") == "clean"


@pytest.mark.parametrize("marker", ["initializer", "ocr", "ollama"])
@pytest.mark.parametrize("role", ["initializer", "ocr"])
def test_uncleared_markers_block_all_competing_native_roles(cpu_containers, marker, role):
    coord, start = cpu_containers
    evidence = coord / f"installation.active-{marker}.json"
    evidence.write_text('{"run_id":"hard-dead-private-test"}')
    before = evidence.read_bytes()
    name = start(
        ["python", "-c", "from pathlib import Path;Path('/coord/native-started').touch()"],
        role=role,
    )
    assert exit_code(name) != 0
    assert not (coord / "native-started").exists()
    assert evidence.read_bytes() == before


def test_live_same_role_is_excluded(cpu_containers):
    coord, start = cpu_containers
    ocr = start(
        [
            "python",
            "-c",
            "from pathlib import Path;import time;Path('/coord/ocr-ready').touch();time.sleep(60)",
        ]
    )
    wait_file(coord / "ocr-ready")
    same = start(["python", "-c", "from pathlib import Path;Path('/coord/wrong-ocr').touch()"])
    assert exit_code(same) != 0
    assert not (coord / "wrong-ocr").exists()
    for name in (ocr,):
        subprocess.run(["docker", "stop", "--time", "5", name], check=True, capture_output=True)
        assert exit_code(name) != 0
    assert not list(coord.glob("*.json"))


def test_hardkill_fence_refuses_competing_start_during_namespace_teardown(cpu_containers):
    coord, start = cpu_containers
    # The contender is already running as PID 1 before the hard kill begins, and
    # retries exclusion continuously; it does not wait for docker kill to return.
    contender_script = """
import sys,time
from pathlib import Path
sys.path.insert(0,'/')
from supervision import managed_installation,InstallationBusy,UncleanInstallation
Path('/coord/contender-ready').touch()
while not Path('/coord/kill-begins').exists(): time.sleep(.001)
while True:
    try:
        with managed_installation('initializer',Path('/coord/installation.lock')):
            Path('/coord/competing-native-start').touch()
        raise SystemExit(0)
    except InstallationBusy: time.sleep(.001)
    except UncleanInstallation:
        Path('/coord/refused-during-teardown').touch()
        raise SystemExit(42)
"""
    victim = start(
        [
            "python",
            "-c",
            "from pathlib import Path;import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);Path('/coord/victim-ready').touch();time.sleep(60)",
        ]
    )
    wait_file(coord / "victim-ready")
    contender = start(["-c", contender_script], supervise=False)
    wait_file(coord / "contender-ready")
    marker = coord / "installation.active-ocr.json"
    original = marker.stat().st_mtime_ns
    (coord / "kill-begins").touch()
    killer = subprocess.Popen(
        ["docker", "kill", "--signal", "KILL", victim], stdout=subprocess.DEVNULL
    )
    try:
        wait_file(coord / "refused-during-teardown")
        assert exit_code(contender) == 42
        assert marker.stat().st_mtime_ns == original
        assert not (coord / "competing-native-start").exists()
    finally:
        killer.wait(timeout=15)
    assert exit_code(victim) == 137
    # Recovery is deliberately external and follows positive owned removal.
    subprocess.run(["docker", "rm", victim, contender], check=True, capture_output=True)
    marker.unlink()
    recovered = start(
        ["python", "-c", "from pathlib import Path;Path('/coord/recovered').touch()"],
        role="initializer",
    )
    assert exit_code(recovered) == 0
    assert (coord / "recovered").exists()


def test_ocr_entrypoint_retains_pid1_and_manual_paths(cpu_containers):
    coord, start = cpu_containers
    code = "import os,json,time;from pathlib import Path;Path('/coord/ocr-entrypoint').write_text(json.dumps([os.getppid(),os.environ['OCR_RUNTIME_SETTINGS_PATH'],os.environ['OCR_QUALIFICATION_IMAGE_PATH']]));time.sleep(60)"
    name = start(["python", "-c", code], ocr_entrypoint=True)
    assert json.loads(wait_file(coord / "ocr-entrypoint")) == [
        1,
        "/manual/runtime.json",
        "/manual/smoke.png",
    ]
    assert (coord / "installation.active-ocr.json").exists()
    subprocess.run(["docker", "kill", "--signal", "HUP", name], check=True, capture_output=True)
    assert exit_code(name) == 129
    assert not (coord / "installation.active-ocr.json").exists()


def test_retired_ollama_supervisor_role_rejects_before_native_command(tmp_path):
    started = tmp_path / "started"
    result = subprocess.run(
        [
            os.sys.executable,
            str(HELPER),
            "--role",
            "ollama",
            "--lock",
            str(tmp_path / "installation.lock"),
            "--",
            os.sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(started)!r}).touch()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("role", ["initializer", "ocr"])
@pytest.mark.parametrize("live", [False, True])
def test_legacy_ollama_marker_blocks_host_without_mutation(tmp_path, role, live):
    from contextlib import nullcontext

    from ocr_service.supervision import (
        UncleanInstallation,
        installation_lock,
        managed_installation,
        role_lock_path,
    )

    lock = tmp_path / "installation.lock"
    marker = tmp_path / "installation.active-ollama.json"
    marker.write_text('{"run_id":"legacy-private"}')
    os.utime(marker, (1, 1))
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    peer = (
        installation_lock(role_lock_path(lock, "ollama"), exclusive=True) if live else nullcontext()
    )
    with peer, pytest.raises(UncleanInstallation), managed_installation(role, lock):
        pytest.fail("Legacy Ollama ownership cannot authorize any native work")
    assert before == (marker.read_bytes(), marker.stat().st_mtime_ns)
    assert not (tmp_path / f"installation.active-{role}.json").exists()
