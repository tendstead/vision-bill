import json
import os
from uuid import uuid4

import pytest
from ocr_contract.activation import (
    ActivationPhase,
    ActivationStatus,
    ActivationStore,
    MaintenanceIntent,
)

from scripts import ocr_stopped_recovery as recovery


@pytest.fixture
def stopped(tmp_path):
    coord = tmp_path / "coord"
    coord.mkdir(mode=0o700)
    store = ActivationStore(*(tmp_path / part for part in ("request", "status", "locks")))
    with store.controller(), store.admission(exclusive=True):
        request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
        store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
        store.set_shared_gpu(False)
    for name in ("installation.lock", "gpu-handoff.lock"):
        (coord / name).touch(mode=0o600)
    journal = coord / "gpu-handoff.ocr-only.json"
    journal.touch(mode=0o600)
    saved = {
        "version": 2,
        "coordinator": "vision-bill-model-handoff-v2",
        "resources": {"llm": "external-unmanaged", "ocr": recovery.OCR_RESOURCE},
        "generation": 39,
        "unsettled": False,
        "models": [],
        "ocr_pending": True,
    }
    journal.write_text(json.dumps(saved, indent=4))
    return coord, store, journal, saved


def test_stopped_recovery_preserves_generation_bytes_and_locks(stopped):
    coord, store, journal, saved = stopped
    before = journal.read_bytes()
    inode = (coord / "gpu-handoff.lock").stat().st_ino
    backup = recovery.settle_stopped_owner(coord, store, lambda: None)
    assert backup.parent == coord and backup.read_bytes() == before
    assert backup.stat().st_mode & 0o777 == 0o600
    assert json.loads(journal.read_bytes()) == {**saved, "ocr_pending": False}
    assert (coord / "gpu-handoff.lock").stat().st_ino == inode
    assert not store.shared_gpu()


@pytest.mark.parametrize(
    "issue",
    [
        "other-ocr",
        "other-llm",
        "owned-model",
        "unsettled",
        "corrupt",
        "shared",
        "pending-peer",
        "foreign-peer",
        "marker",
        "direct",
        "unsafe-direct",
        "lock",
        "missing-lock",
        "unsafe-journal",
        "Docker owner",
        "status",
        "hardlink",
    ],
)
def test_stopped_recovery_refuses_ambiguous_state(stopped, issue):
    import fcntl

    coord, store, journal, saved = stopped
    if issue == "other-ocr":
        saved["resources"]["ocr"] = "paddleocr:foreign"
    if issue == "other-llm":
        saved["resources"]["llm"] = "external"
    if issue == "owned-model":
        saved["models"] = ["private-model"]
    if issue == "unsettled":
        saved["unsettled"] = True
    journal.write_text(json.dumps(saved))
    if issue == "corrupt":
        journal.write_text("{")
    if issue == "shared":
        store.set_shared_gpu(True)
    if issue in ("pending-peer", "foreign-peer"):
        peer = coord / "gpu-handoff.json"
        peer.touch(mode=0o600)
        peer.write_text(
            json.dumps(
                {
                    **saved,
                    "ocr_pending": issue == "pending-peer",
                    "resources": {
                        "llm": "ollama:abc",
                        "ocr": recovery.OCR_RESOURCE if issue == "pending-peer" else "foreign",
                    },
                }
            )
        )
    if issue == "marker":
        (coord / "installation.active-ocr.json").touch()
    if issue == "direct":
        (store.request_dir / "direct-calls").mkdir(mode=0o700)
        (store.request_dir / "direct-calls" / uuid4().hex).touch()
    if issue == "unsafe-direct":
        (store.request_dir / "direct-calls").mkdir(mode=0o755)
        (store.request_dir / "direct-calls").chmod(0o755)
    if issue == "missing-lock":
        (coord / "gpu-handoff.lock").unlink()
    if issue == "unsafe-journal":
        journal.chmod(0o644)
    if issue == "status":
        (store.status_dir / "status.json").unlink()
    if issue == "hardlink":
        os.link(journal, coord / "alias")
    before = journal.read_bytes()
    with (coord / "installation.lock").open("rb") as stream:
        if issue == "lock":
            fcntl.flock(stream, fcntl.LOCK_SH)

        def confirm():
            if issue == "Docker owner":
                raise ValueError("owner remains")

        with pytest.raises((ValueError, OSError, RuntimeError)):
            recovery.settle_stopped_owner(coord, store, confirm)
    assert journal.read_bytes() == before
    assert not list(coord.glob("*.recovered-*.json"))


def test_inspection_program_uses_only_transported_source(tmp_path):
    import subprocess
    import sys

    program = recovery.inspection_program(
        "from ocr_service.initialization_lock import OutstandingHandoff\nprint(OutstandingHandoff.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "OutstandingHandoff"


def stock_config(project="sample"):
    res = {
        "services": {},
        "networks": {
            "default": {"name": project + "_default", "ipam": {}},
            "postgres_net": {"name": "postgres_net", "driver": "bridge"},
        },
        "volumes": {k: {"name": project + "_" + k} for k in recovery.MOUNTS},
    }
    for name in ("vision_bill", recovery.SERVING):
        res["services"][name] = {
            "volumes": [
                {"type": "volume", "source": k, "target": v}
                for k, v in recovery.MOUNTS.items()
                if name == recovery.SERVING or k not in ("ocr-cache", "ocr-state")
            ]
        }
    res["services"][recovery.SERVING].update(
        environment={
            "OCR_HANDOFF_LOCK_PATH": "/coord/gpu-handoff.lock",
            "OCR_INSTALLATION_LOCK_PATH": "/coord/installation.lock",
            "OCR_PORT": "8082",
            "OCR_STATE_DIR": "/var/lib/ocr",
        },
        command=["python", "-m", "ocr_service.activation"],
        networks={"default": {}},
    )
    res["services"]["vision_bill"].update(
        environment={
            "OCR_ACTIVATION_ENDPOINT": recovery.ENDPOINT,
            "VISION_BILL_OCR_ACTIVATION": "true",
            "OCR_ACTIVATION_HANDOFF_LOCK_PATH": "/coord/gpu-handoff.lock",
        },
        networks={"default": None, "postgres_net": None},
        extra_hosts=["host.docker.internal=host-gateway"],
    )
    return res


@pytest.mark.parametrize(
    "override",
    [
        "extra_hosts",
        "external-network",
        "alias",
        "host-mode",
        "port",
        "install-lock",
        "nonlocal",
        "redirect",
    ],
)
def test_recovery_validates_original_route(override):
    config = stock_config()
    assert set(recovery._volume_names(config, "sample")) == set(recovery.MOUNTS)
    services = config["services"]
    app, native = services["vision_bill"], services[recovery.SERVING]
    if override == "extra_hosts":
        app["extra_hosts"] = ["paddleocr-service=127.9.9.9"]
    if override == "external-network":
        config["networks"]["default"]["external"] = True
    if override == "alias":
        native["networks"]["default"] = {"aliases": ["other"]}
    if override == "host-mode":
        app["network_mode"] = "host"
    if override == "port":
        native["environment"]["OCR_PORT"] = "9999"
    if override == "install-lock":
        native["environment"]["OCR_INSTALLATION_LOCK_PATH"] = "/other/lock"
    if override == "nonlocal":
        app["environment"]["OCR_ACTIVATION_ENDPOINT"] = "http://external.invalid"
    if override == "redirect":
        app["volumes"].append({"type": "bind", "source": "/foreign/hosts", "target": "/etc/hosts"})
    with pytest.raises(ValueError):
        recovery._volume_names(config, "sample")


@pytest.mark.parametrize("foreign", ["network", "mount", "bind", "alias-volume"])
def test_recovery_checks_foreign_container_access(foreign):
    names = {"ocr-state": "sample_ocr-state"}
    roots = "/var/lib/docker/volumes/"
    records = {
        name: {"Name": name, "Mountpoint": roots + name + "/_data"} for name in names.values()
    }
    original = records["sample_ocr-state"]["Mountpoint"]
    mount = {"Type": "volume", "Name": "sample_ocr-state"}
    if foreign == "bind":
        mount = {"Type": "bind", "Source": original + "/models"}
    if foreign == "alias-volume":
        mount = {"Type": "volume", "Name": "foreign"}
        records["foreign"] = {
            "Name": "foreign",
            "Mountpoint": roots + "foreign/_data",
            "Driver": "local",
            "Options": {"o": "bind", "device": original},
        }

    def run(*args, capture):
        if args[1:3] == ("network", "ls"):
            return "sample_default" if foreign == "network" else ""
        if args[1] == "ps":
            return "helper\nstranger"
        if args[1:3] == ("volume", "inspect"):
            return json.dumps([records[n] for n in args[3:]])
        if args[1:3] == ("container", "inspect"):
            return json.dumps({"mounts": [mount], "labels": {}})
        pytest.fail(args)

    with pytest.raises(ValueError, match="network|volume"):
        recovery._verify_stopped(names, "sample", "helper", run)


def crashed_native(coord):
    marker = coord / "installation.active-ocr.json"
    marker.touch(mode=0o600)
    marker.write_text(json.dumps({"schema": 1, "role": "ocr", "run_id": uuid4().hex}))
    (coord / "installation.role-ocr.lock").touch(mode=0o600)
    return marker


def test_valid_native_crash_marker_archived_only_after_removal(stopped):
    coord, store, journal, saved = stopped
    marker = crashed_native(coord)
    original_marker, original_journal = marker.read_bytes(), journal.read_bytes()
    role_inode = (coord / "installation.role-ocr.lock").stat().st_ino

    def confirm():
        assert marker.read_bytes() == original_marker
        assert journal.read_bytes() == original_journal
        assert not list(coord.glob("*.recovered-*.json"))

    journal_backup = recovery.settle_stopped_owner(coord, store, confirm)
    (backup,) = coord.glob("installation.ocr.recovered-*.json")
    assert backup.read_bytes() == original_marker
    assert backup.stat().st_mode & 0o777 == 0o600
    assert backup.name.removeprefix("installation.ocr.") == journal_backup.name.removeprefix(
        "gpu-handoff.ocr-only."
    )
    assert not marker.exists()
    assert (coord / "installation.role-ocr.lock").stat().st_ino == role_inode
    assert json.loads(journal.read_bytes()) == dict(saved, ocr_pending=False)


@pytest.mark.parametrize(
    "case",
    [
        "initializer",
        "ollama",
        "unknown",
        "wrong-role",
        "not-uuid4",
        "duplicate",
        "schema-bool",
        "shape",
        "mode",
        "symlink",
        "hardlink",
        "busy-role",
        "missing-role-lock",
        "changed",
        "new-marker",
        "container",
    ],
)
def test_crash_marker_ambiguous_ownership_preserves_journal(stopped, case):
    import fcntl

    coord, store, journal, _ = stopped
    marker = crashed_native(coord)
    role_lock = coord / "installation.role-ocr.lock"
    with role_lock.open("rb") as role:
        saved = json.loads(marker.read_bytes())
        if case in ("initializer", "ollama", "unknown"):
            (coord / f"installation.active-{case}.json").touch(mode=0o600)
        if case == "wrong-role":
            saved["role"] = "initializer"
        if case == "not-uuid4":
            saved["run_id"] = "a" * 32
        if case == "schema-bool":
            saved["schema"] = True
        if case == "shape":
            saved["container_id"] = "unknown"
        marker.write_text(json.dumps(saved))
        if case == "duplicate":
            marker.write_text(marker.read_text().replace('"schema": 1', '"schema": 2, "schema": 1'))
        if case == "mode":
            marker.chmod(0o644)
        if case == "symlink":
            target = coord / "unrelated"
            marker.rename(target)
            marker.symlink_to(target)
        if case == "hardlink":
            os.link(marker, coord / "other-file")
        if case == "missing-role-lock":
            role_lock.unlink()
        if case == "busy-role":
            fcntl.flock(role, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = journal.read_bytes()

        def confirm():
            if case == "container":
                raise ValueError("attached container")
            if case == "changed":
                marker.write_text(json.dumps({**saved, "run_id": uuid4().hex}))
            if case == "new-marker":
                (coord / "installation.active-old.json").touch()

        with pytest.raises((ValueError, OSError, RuntimeError)):
            recovery.settle_stopped_owner(coord, store, confirm)
        assert journal.read_bytes() == before
        assert marker.exists()
        assert not list(coord.glob("*.recovered-*.json"))


@pytest.mark.parametrize("failed", [False, True])
def test_crash_marker_interruption_after_archive_is_retryable(stopped, monkeypatch, failed):
    coord, store, journal, saved = stopped
    marker = crashed_native(coord)
    if failed:
        rec = store.status_dir / "status.json"
        rec.write_text(
            json.dumps(
                {
                    **json.loads(rec.read_text()),
                    "phase": "failed",
                    "reason_code": "native_activation_recovery_required",
                }
            )
        )
    prior = marker.read_bytes()
    original = journal.read_bytes()
    replace = os.replace

    def fail_at_journal(src, dst):
        if dst == journal:
            raise OSError("disk failure")
        return replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_at_journal)
        with pytest.raises(OSError):
            recovery.settle_stopped_owner(coord, store, lambda: None)
    assert journal.read_bytes() == original
    assert not marker.exists()
    (backup,) = coord.glob("installation.ocr.recovered-*.json")
    assert backup.read_bytes() == prior
    recovery.settle_stopped_owner(coord, store, lambda: None)
    assert json.loads(journal.read_bytes()) == dict(saved, ocr_pending=False)


@pytest.mark.parametrize(
    "phase,reason,has_marker,allowed",
    [
        (ActivationPhase.FAILED, "native_activation_recovery_required", True, True),
        (ActivationPhase.FAILED, "native_activation_recovery_required", False, False),
        (ActivationPhase.FAILED, "native_installation_busy", True, False),
        (ActivationPhase.PREPARING, None, True, False),
    ],
)
def test_matching_restoration_failure_requires_crash_evidence(
    stopped, phase, reason, has_marker, allowed
):
    coord, store, journal, _ = stopped
    if has_marker:
        crashed_native(coord)
    path = store.status_dir / "status.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "phase": phase.value, "reason_code": reason}))
    if allowed:
        recovery.settle_stopped_owner(coord, store, lambda: None)
        assert not json.loads(journal.read_bytes())["ocr_pending"]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            recovery.settle_stopped_owner(coord, store, lambda: None)
        assert json.loads(journal.read_bytes())["ocr_pending"]


def test_retry_syncs_retired_marker_before_settling_journal(stopped, monkeypatch):
    import stat

    coord, store, journal, _ = stopped
    marker = crashed_native(coord)
    fsync, replace = os.fsync, os.replace

    def interrupted_sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) and not marker.exists():
            raise OSError("marker rename not yet durably acknowledged")
        fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", interrupted_sync)
        with pytest.raises(OSError):
            recovery.settle_stopped_owner(coord, store, lambda: None)
    assert json.loads(journal.read_bytes())["ocr_pending"]
    assert not marker.exists()
    synced = False

    def sync(fd):
        nonlocal synced
        fsync(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced = True

    def publish(src, dst):
        if dst == journal:
            assert synced, "must sync archived marker before retiring journal"
        replace(src, dst)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", publish)
    recovery.settle_stopped_owner(coord, store, lambda: None)
