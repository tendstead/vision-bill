"""Initializer software evidence; the native process boundary is simulated."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_ocr_provision import NAMES, sha
from test_ocr_provision import release as _release

release = _release

DEVICE = {
    "accelerator": "cuda",
    "gpu_index": 0,
    "name": "CPU-DOUBLE",
    "uuid": "000102030405060708090a0b0c0d0e0f",
    "pci_bus_id": "0000:01:00.0",
    "driver_version": 12000,
    "compute_capability": [8, 0],
}


@pytest.fixture(autouse=True)
def metadata_only_device(monkeypatch):
    monkeypatch.setattr(
        "ocr_service.initialize.inspect_device_identity", lambda *args: DEVICE.copy(), raising=False
    )


@pytest.fixture
def native_pid1():
    if os.getpid() != 1:
        pytest.skip(
            "Native coordinator lifecycle is verified as PID 1 in the owned CPU test container"
        )


def external_qualifier(monkeypatch, *, success=True):
    """Replace only the external native executable, keeping coordinator behavior real."""
    from ocr_service import supervision

    real_popen = subprocess.Popen

    def spawn(command, **kwargs):
        assert command[1:3] == ["-m", "ocr_service.qualify"]
        runtime = Path(command[command.index("--runtime") + 1])
        cases = Path(command[command.index("--cases") + 1])
        output = Path(command[command.index("--output") + 1])
        script = """
import json,sys,pathlib
runtime,cases,output=map(pathlib.Path,sys.argv[1:4])
settings=json.loads(runtime.read_text())
success=sys.argv[4]=='yes'
manifest=pathlib.Path(settings['qualification_manifest'])
manifest.write_text(json.dumps({'image_identity':settings['image_identity'],'model_content':{},'qualified_target':{'name':'CPU-TEST-ONLY'}}))
report={'device':json.loads(sys.argv[5]),'qualified':success,'stage':'complete' if success else 'native_collection','accelerator':settings['accelerator'],'image_identity':settings['image_identity'],'runtime_fingerprint':'cpu-test-only','profile_id':'cpu-test-only','manifest_sha256':__import__('hashlib').sha256(manifest.read_bytes()).hexdigest(),'cases':[{'kind':kind,'fresh_worker':repetition} for kind in ['upright','rotated','warped','table','formula'] for repetition in range(2)]}
output.write_text(json.dumps(report))
if success: output.with_suffix('.runtime.json').write_text(json.dumps(settings))
sys.exit(0 if success else 1)
"""
        return real_popen(
            [
                sys.executable,
                "-c",
                script,
                str(runtime),
                str(cases),
                str(output),
                "yes" if success else "no",
                json.dumps(DEVICE),
            ],
            **kwargs,
        )

    monkeypatch.setattr(supervision.subprocess, "Popen", spawn)


def initialize_release(tmp_path, release, **kwargs):
    from ocr_service.initialize import initialize

    bundle, _, config = release
    return initialize(
        bundle,
        tmp_path / "state",
        config,
        "sha256:" + sha(config.read_bytes()),
        maintenance_intent="external-llm-quiesced",
        **kwargs,
    )


@pytest.mark.usefixtures("native_pid1")
def test_failed_native_process_cannot_publish_settings(tmp_path, release, monkeypatch):
    from ocr_service.initialize import InitializationFailed

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch, success=False)
    with pytest.raises(InitializationFailed):
        initialize_release(tmp_path, release)
    assert not (tmp_path / "state/selected").exists()


@pytest.mark.usefixtures("native_pid1")
def test_success_publishes_verified_settings_then_cached_start_never_spawns_or_rewrites(
    tmp_path, release, monkeypatch
):
    from ocr_service.initialization_lock import installation_lock

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch)
    result = initialize_release(tmp_path, release)
    selected = tmp_path / "state/selected"
    assert selected.resolve() == result
    runtime = json.loads((selected / "runtime.json").read_bytes())
    assert set(runtime["model_dirs"]) == set(NAMES)
    smoke = Path(json.loads((selected / "selection.json").read_bytes())["smoke_path"])
    assert smoke.read_bytes() == b"tiny-private-test-upright"
    before = {
        path: path.lstat().st_mtime_ns
        for path in (selected, selected / "runtime.json", selected / "complete.json")
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Cached complete initialization must not run native work")

    from ocr_service import supervision

    monkeypatch.setattr(supervision.subprocess, "Popen", forbidden)
    with installation_lock(tmp_path / "installation.lock", exclusive=False):
        bundle, _, config = release
        from ocr_service.initialize import check_installation

        assert (
            check_installation(
                bundle, tmp_path / "state", config, "sha256:" + sha(config.read_bytes())
            )
            == result
        )
    assert before == {path: path.lstat().st_mtime_ns for path in before}


@pytest.mark.usefixtures("native_pid1")
def test_upgrade_preserves_explicit_qualification_and_installs_without_native_work(
    tmp_path, release, monkeypatch
):
    from ocr_contract.activation import MaintenanceIntent
    from ocr_service import initialize as module

    state = tmp_path / "state"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch)
    old = initialize_release(tmp_path, release)
    path, bundle, config = release
    for name, asset in bundle["models"].items():
        asset["source"] = f"https://example.invalid/{name}.tar"
    path.write_text(json.dumps(bundle))
    monkeypatch.setattr(module, "verify_package_pins", lambda bundle: None)
    monkeypatch.setattr(module, "owned_qualification_process",
                        lambda *args: pytest.fail("Migration ran qualification"))
    args = (path, state, config, "sha256:" + sha(config.read_bytes()))
    with pytest.raises(module.PreparationRequired):
        module.check_prepared_installation(*args)
    new = module.prepare_installation(
        *args, maintenance_intent=MaintenanceIntent.SEPARATE_OCR_DEVICE,
        handoff_lock=tmp_path / "handoff.lock",
    )
    assert (state / "selected").resolve() == new != old
    assert (old / "qualification.json").is_file()
    assert not (new / "qualification.json").exists()
    assert module.check_prepared_installation(*args) == new
    with pytest.raises(module.InitializationFailed):
        module.check_installation(*args)


@pytest.mark.usefixtures("native_pid1")
@pytest.mark.parametrize("published", [False, True], ids=["pre-swap", "post-swap"])
def test_interrupted_publication_preserves_selected_installation(
    tmp_path, release, monkeypatch, published
):
    from ocr_service import initialize as module
    from ocr_service.supervision import ManagedInterrupted

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch)
    previous = initialize_release(tmp_path, release)
    state = tmp_path / "state"
    selected = state / "selected"
    original_replace = Path.replace

    def interrupt_replace(path, target):
        if target != selected:
            return original_replace(path, target)
        if published:
            original_replace(path, target)
        signal.raise_signal(signal.SIGTERM)

    with monkeypatch.context() as publication:
        publication.setattr(Path, "replace", interrupt_replace)
        with pytest.raises(ManagedInterrupted) as interrupted:
            initialize_release(tmp_path, release, force=True)
    assert interrupted.value.signum == signal.SIGTERM
    assert previous.is_dir()
    retained = selected.resolve()
    assert (retained != previous) is published
    assert set(previous.parent.iterdir()) == {previous, retained}
    assert list(state.glob(".selected-*")) == []

    def forbidden(*args, **kwargs):
        pytest.fail("Retry should reuse the verified installation")

    monkeypatch.setattr(module, "owned_qualification_process", forbidden)
    monkeypatch.setattr(module, "provision_bundle", forbidden)
    bundle, _, config = release
    assert (
        module.check_installation(bundle, state, config, "sha256:" + sha(config.read_bytes()))
        == retained
    )
    assert initialize_release(tmp_path, release) == retained


@pytest.mark.usefixtures("native_pid1")
def test_live_runtime_prevents_first_and_forced_native_initialization(
    tmp_path, release, monkeypatch
):
    from ocr_service.initialization_lock import InstallationBusy, installation_lock

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    with (
        installation_lock(tmp_path / "installation.lock", exclusive=False),
        pytest.raises(InstallationBusy),
    ):
        initialize_release(tmp_path, release)
    external_qualifier(monkeypatch)
    initialize_release(tmp_path, release)
    before = (tmp_path / "state/selected").readlink()
    with (
        installation_lock(tmp_path / "installation.lock", exclusive=False),
        pytest.raises(InstallationBusy),
    ):
        initialize_release(tmp_path, release, force=True)
    assert (tmp_path / "state/selected").readlink() == before


@pytest.mark.parametrize(
    "alter", ["runtime", "report", "model", "smoke", "manifest", "completion", "cases"]
)
@pytest.mark.usefixtures("native_pid1")
def test_cached_installation_integrity_is_fully_rechecked(tmp_path, release, monkeypatch, alter):
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch)
    result = initialize_release(tmp_path, release)
    selected = tmp_path / "state/selected"
    settings = json.loads((selected / "runtime.json").read_bytes())
    paths = {
        "cases": Path(settings["model_dirs"][NAMES[0]]).parents[1] / "cases.toml",
        "runtime": result / "runtime.json",
        "report": result / "qualification.json",
        "manifest": Path(settings["qualification_manifest"]),
        "model": Path(settings["model_dirs"][NAMES[0]]) / "weights",
        "smoke": Path(json.loads((selected / "selection.json").read_bytes())["smoke_path"]),
        "completion": result / "complete.json",
    }
    paths[alter].write_bytes(b"tampered")
    with pytest.raises(ValueError):
        initialize_release(tmp_path, release)


@pytest.mark.parametrize("identity", ["id", "config", "architecture"])
def test_actual_image_configuration_identity_required_before_native(
    tmp_path, release, monkeypatch, identity
):
    from ocr_service.initialize import initialize

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    bundle, data, config = release
    image_id = "sha256:" + sha(config.read_bytes())
    if identity == "id":
        image_id = "sha256:" + "0" * 64
    elif identity == "config":
        config.write_text("{}")
    else:
        config.write_text('{"architecture":"arm64","os":"linux"}')
        image_id = "sha256:" + sha(config.read_bytes())
        data["image_config_sha256"] = sha(config.read_bytes())
        bundle.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        initialize(
            bundle, tmp_path / "state", config, image_id, maintenance_intent="external-llm-quiesced"
        )
    assert not (tmp_path / "state/selected").exists()


def test_entrypoint_preserves_explicit_manual_paths(monkeypatch):
    from ocr_service.entrypoint import configure_selected

    monkeypatch.setenv("OCR_RUNTIME_SETTINGS_PATH", "/manual/runtime.json")
    monkeypatch.setenv("OCR_QUALIFICATION_IMAGE_PATH", "/manual/smoke.png")
    configure_selected()
    assert os.environ["OCR_RUNTIME_SETTINGS_PATH"] == "/manual/runtime.json"
    assert os.environ["OCR_QUALIFICATION_IMAGE_PATH"] == "/manual/smoke.png"


def test_entrypoint_resolves_selected_paths_without_native_import(tmp_path, monkeypatch):
    from ocr_service.entrypoint import configure_selected

    root = tmp_path / "installations" / "private-test"
    root.mkdir(parents=True)
    (root / "complete.json").write_text("{}")
    (tmp_path / "selected").symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("OCR_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("OCR_RUNTIME_SETTINGS_PATH", raising=False)
    monkeypatch.delenv("OCR_QUALIFICATION_IMAGE_PATH", raising=False)
    configure_selected()
    assert os.environ["OCR_RUNTIME_SETTINGS_PATH"] == str(root / "runtime.json")
    assert "OCR_QUALIFICATION_IMAGE_PATH" not in os.environ


@pytest.mark.parametrize("role", ["initializer", "ocr", "ollama"])
@pytest.mark.parametrize("force", [False, True])
def test_uncleared_marker_blocks_actual_initialization_including_force(
    tmp_path, release, monkeypatch, role, force
):
    from ocr_service.supervision import UncleanInstallation

    lock = tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    marker = tmp_path / f"installation.active-{role}.json"
    marker.write_text('{"run_id":"hard-dead-test"}')
    before = marker.read_bytes()
    with pytest.raises(UncleanInstallation):
        initialize_release(tmp_path, release, force=force)
    assert marker.read_bytes() == before
    assert not (tmp_path / "state/selected").exists()


@pytest.mark.usefixtures("native_pid1")
@pytest.mark.parametrize("role", ["initializer", "ocr", "ollama", "unknown"])
def test_cache_hit_still_honors_crash_fences(tmp_path, release, monkeypatch, role):
    from ocr_service.initialize import check_installation
    from ocr_service.supervision import UncleanInstallation

    lock = tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    external_qualifier(monkeypatch)
    initialize_release(tmp_path, release)
    marker = tmp_path / f"installation.active-{role}.json"
    marker.write_text('{"run_id":"unclean-test"}')
    original_marker = marker.read_bytes()
    bundle, _, config = release
    with pytest.raises(UncleanInstallation):
        check_installation(bundle, tmp_path / "state", config, "sha256:" + sha(config.read_bytes()))
    assert marker.read_bytes() == original_marker


def test_cache_miss_never_qualifies_on_check(tmp_path, release, monkeypatch):
    from ocr_service import initialize as module

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    qualify_mock = Mock(side_effect=AssertionError("check must never qualify"))
    monkeypatch.setattr(module, "owned_qualification_process", qualify_mock)
    monkeypatch.setattr(module, "provision_bundle", qualify_mock)
    bundle, _, config = release
    with pytest.raises(module.InitializationFailed) as cache_miss:
        module.check_installation(
            bundle, tmp_path / "state", config, "sha256:" + sha(config.read_bytes())
        )
    assert cache_miss.value.reason_code == "native_initialization_required"
    qualify_mock.assert_not_called()
    assert not (tmp_path / "state").exists()


def test_maintenance_intent_bound_to_invocation(tmp_path, release, monkeypatch):
    from ocr_service.initialize import initialize

    monkeypatch.setenv("OCR_MAINTENANCE_INTENT", "external-llm-quiesced")
    bundle, _, config = release
    args = bundle, tmp_path / "state", config, "sha256:" + sha(config.read_bytes())
    with pytest.raises(TypeError, match="maintenance_intent"):
        initialize(*args)
    with pytest.raises(ValueError, match="maintenance"):
        initialize(*args, maintenance_intent="api-unreachable")
    assert not (tmp_path / "state").exists()


def test_app_lock_blocks_maintenance(tmp_path, release, monkeypatch):
    from ocr_service.initialize import initialize
    from ocr_service.supervision import InstallationBusy, installation_lock

    handoff = tmp_path / "gpu-handoff.lock"
    inode = None
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    bundle, _, config = release
    with installation_lock(handoff, exclusive=True):
        inode = handoff.stat().st_ino
        with pytest.raises(InstallationBusy):
            initialize(
                bundle,
                tmp_path / "state",
                config,
                "sha256:" + sha(config.read_bytes()),
                maintenance_intent="external-llm-quiesced",
                handoff_lock=handoff,
            )
    assert inode == handoff.stat().st_ino
    assert not (tmp_path / "installation.lock").exists()


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
def test_metadata_probe_uses_driver_only(tmp_path, monkeypatch, accelerator):
    import builtins
    import ctypes

    from ocr_service import device_inventory

    real_import = builtins.__import__

    def no_native_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"paddle", "paddleocr", "paddlex"}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_native_import)
    expected = {**DEVICE, "accelerator": accelerator}
    if accelerator == "rocm":
        expected = {key: value for key, value in expected.items() if key != "compute_capability"}
        expected["uuid"] = b"0123456789abcdef".hex()
        expected["gfx_target"] = "gfx1030"

        def run(command, **kwargs):
            assert command == ["/opt/ocr/bin/hip-device-info", "0"]
            return SimpleNamespace(stdout=json.dumps(expected))

        monkeypatch.setattr(device_inventory.subprocess, "run", run)
    else:

        def integer(value):
            def call(*args):
                ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_int))[0] = value
                return 0

            return call

        def text_value(value):
            def call(buffer, *args):
                ctypes.memmove(buffer, value.encode() + b"\0", len(value) + 1)
                return 0

            return call

        def capability(major, minor, device):
            ctypes.cast(major, ctypes.POINTER(ctypes.c_int))[0] = 8
            ctypes.cast(minor, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0

        def uuid(buffer, device):
            ctypes.memmove(buffer, bytes(range(16)), 16)
            return 0

        driver = SimpleNamespace(
            cuInit=lambda flags: 0,
            cuDeviceGet=lambda ptr, index: integer(0)(ptr),
            cuDriverGetVersion=integer(12000),
            cuDeviceGetName=text_value("CPU-DOUBLE"),
            cuDeviceGetPCIBusId=text_value("0000:01:00.0"),
            cuDeviceComputeCapability=capability,
            cuDeviceGetUuid_v2=uuid,
        )
        monkeypatch.setattr(device_inventory.ctypes, "CDLL", lambda name: driver)
        expected["uuid"] = "000102030405060708090a0b0c0d0e0f"
    assert device_inventory.inspect_device_identity(accelerator, 0) == expected


@pytest.mark.parametrize(
    "uuid", [b"XX" + b"\0" * 14, b"0" * 16], ids=["gpu-xx-sentinel", "zero-id"]
)
def test_rocm_metadata_rejects_unavailable_uuid(monkeypatch, uuid):
    from ocr_service import device_inventory

    metadata = {key: value for key, value in DEVICE.items() if key != "compute_capability"}
    metadata.update(accelerator="rocm", gfx_target="gfx1030", uuid=uuid.hex())
    monkeypatch.setattr(
        device_inventory.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(metadata)),
    )
    with pytest.raises(device_inventory.DeviceIdentityUnavailable) as error:
        device_inventory.inspect_device_identity("rocm", 0)
    assert error.value.reason_code == "native_device_identity_unavailable"


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
def test_metadata_unavailable_is_specific_safe_failure(monkeypatch, accelerator):
    from ocr_service import device_inventory

    monkeypatch.setattr(
        device_inventory.ctypes, "CDLL", Mock(side_effect=OSError("missing driver"))
    )
    monkeypatch.setattr(
        device_inventory.subprocess,
        "run",
        lambda *a, **k: type("Result", (), {"stdout": "not-json"})(),
    )
    with pytest.raises(device_inventory.DeviceIdentityUnavailable) as error:
        device_inventory.inspect_device_identity(accelerator, 0)
    assert error.value.reason_code == "native_device_identity_unavailable"


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"version": 1},
        {"models": ["owned-model"]},
        {"unsettled": True},
        {"ocr_pending": True},
        {"resources": {"llm": "", "ocr": "native"}},
        {"generation": True},
        {"extra": False},
    ],
)
@pytest.mark.parametrize("shared_gpu", [True, False])
def test_maintenance_handoff_journal_strict_clean_v2_only(tmp_path, release, monkeypatch, change, shared_gpu):
    from contextlib import contextmanager

    from ocr_service import initialize as module

    handoff = tmp_path / "gpu-handoff.lock"
    journal = handoff.with_suffix(".json" if shared_gpu else ".ocr-only.json")
    state = {
        "version": 2,
        "coordinator": "vision-bill-model-handoff-v2",
        "resources": {"llm": "explicit-test-llm" if shared_gpu else "external-unmanaged", "ocr": "explicit-test-ocr"},
        "generation": 1,
        "models": [],
        "unsettled": False,
        "ocr_pending": False,
    }
    if change:
        state.update(change)
    journal.write_text(json.dumps(state))
    journal.chmod(0o600)
    before = journal.read_bytes(), journal.stat().st_mtime_ns
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))

    # Stop before actual PID1/native work: entering the native scope proves the
    # strict read-only handoff gate accepted; no process boundary is simulated.
    @contextmanager
    def stop_before_native(*args):
        raise RuntimeError("accepted clean journal")
        yield

    monkeypatch.setattr(module, "managed_installation", stop_before_native)
    bundle, _, config = release
    with pytest.raises(RuntimeError, match="handoff" if change else "accepted clean journal"):
        module.initialize(
            bundle,
            tmp_path / "state",
            config,
            "sha256:" + sha(config.read_bytes()),
            maintenance_intent="separate-ocr-device",
            handoff_lock=handoff,
        )
    assert before == (journal.read_bytes(), journal.stat().st_mtime_ns)


@pytest.mark.parametrize(
    "raw",
    [
        '{"models":[]}',
        "not-json",
        '{"version":2,"coordinator":"vision-bill-model-handoff-v2","resources":{"llm":"test","ocr":"test"},"generation":1,"models":[],"unsettled":true,"unsettled":false,"ocr_pending":false}',
    ],
)
def test_maintenance_preserves_legacy_malformed_duplicate_journals(tmp_path, raw):
    from ocr_service.initialization_lock import maintenance_handoff

    journal = tmp_path / "gpu-handoff.json"
    journal.write_text(raw)
    journal.chmod(0o600)
    before = journal.read_bytes()
    with (
        pytest.raises(RuntimeError, match="handoff"),
        maintenance_handoff(journal.with_suffix(".lock")),
    ):
        pytest.fail("Unsafe journal must fence maintenance")
    assert journal.read_bytes() == before


@pytest.mark.usefixtures("native_pid1")
def test_maintenance_holds_both_locks_through_drain_and_bound_publication(
    tmp_path, release, monkeypatch
):
    from ocr_service import initialize as module
    from ocr_service import supervision
    from ocr_service.supervision import InstallationBusy, installation_lock

    handoff, native = tmp_path / "gpu-handoff.lock", tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(native))
    external_qualifier(monkeypatch)
    original_drain = supervision._drain_owned
    drains = []

    def verify_drain():
        for lock in (handoff, native):
            with pytest.raises(InstallationBusy), installation_lock(lock, exclusive=True):
                pytest.fail("Maintenance exclusion released before child drain")
        drains.append(True)
        original_drain()

    monkeypatch.setattr(supervision, "_drain_owned", verify_drain)
    bundle, _, config = release
    # Fresh daemon volumes mount with root-owned 0755 directories.
    (tmp_path / "state").mkdir(mode=0o755)
    root = module.initialize(
        bundle,
        tmp_path / "state",
        config,
        "sha256:" + sha(config.read_bytes()),
        maintenance_intent="separate-ocr-device",
        handoff_lock=handoff,
    )
    assert len(drains) >= 2
    evidence = json.loads((root / "maintenance.json").read_bytes())
    assert evidence["intent"] == "separate-ocr-device"
    assert evidence["device_identity"] == DEVICE
    assert evidence["image_id"] == "sha256:" + sha(config.read_bytes())
    assert evidence["bundle_id"] == json.loads((root / "complete.json").read_bytes())["bundle_id"]
    assert evidence["invocation_id"]
    for path in (tmp_path / "state", root.parent, root):
        assert path.stat().st_mode & 0o777 == 0o700
    monkeypatch.setattr(module, "inspect_device_identity", lambda *a: {**DEVICE, "uuid": "changed"})
    with pytest.raises(module.InitializationFailed, match="native_initialization_required"):
        module.check_installation(
            bundle, tmp_path / "state", config, "sha256:" + sha(config.read_bytes())
        )


def test_paired_maintenance_requires_explicit_original_handoff_path(
    tmp_path, release, monkeypatch, capsys
):
    from ocr_service import initialize as module

    bundle, _, config = release
    monkeypatch.setenv("OCR_MAINTENANCE_REQUIRE_HANDOFF", "true")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ocr-init",
            "maintain",
            "--bundle",
            str(bundle),
            "--image-config",
            str(config),
            "--image-id",
            "sha256:" + sha(config.read_bytes()),
            "--maintenance-intent",
            "external-llm-quiesced",
        ],
    )
    native = Mock(side_effect=AssertionError("paired CLI must fail before native work"))
    monkeypatch.setattr(module, "initialize", native)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "original app --handoff-lock" in capsys.readouterr().err
    native.assert_not_called()


@pytest.mark.parametrize("invalid", [None, "unknown_model", "oversized"])
def test_failed_collection_logs_only_a_bounded_static_hook_diagnostic(
    tmp_path, release, monkeypatch, caplog, invalid
):
    from contextlib import nullcontext

    from ocr_service import initialize as module

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    monkeypatch.setattr(module, "managed_installation", lambda *_args: nullcontext())
    private_text = "private tensor name and path"

    def fail(command):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({
            "stage": "native_collection",
            "failure": "static_hook_collection_failed",
            "qualified": False,
            "static_hook_failure": {
                "model_name": private_text if invalid == "unknown_model" else "UVDoc",
                "reason_code": "hook_cpu_float",
                "input_count": 7,
                "output_count": 5,
            },
            "private_fixture_output": private_text * (100_000 if invalid == "oversized" else 1),
        }))
        return 1

    monkeypatch.setattr(module, "owned_qualification_process", fail)
    with pytest.raises(module.InitializationFailed):
        initialize_release(tmp_path, release)

    messages = [r.getMessage() for r in caplog.records]
    expected = "static_hook_collection_failed model=UVDoc reason=hook_cpu_float input=7 output=5"
    assert (expected in messages) == (invalid is None)
    assert private_text not in caplog.text
    assert not (tmp_path / "state" / "selected").exists()
    assert not list((tmp_path / "state" / "installations").iterdir())


@pytest.mark.parametrize(
    ("first", "expected_attempts"),
    [
        ("collection_timeout", 2),
        ("recovered", 2),
        ("generic_exit", 1),
        ("unmarked", 1),
        ("other_stage", 1),
        ("other_image", 1),
        ("oom", 1),
        ("static_hooks", 1),
        ("malformed", 1),
        ("symlink", 1),
        ("hardlink", 1),
        ("manual", 1),
    ],
)
def test_managed_initializer_retries_only_verified_collection_wait_timeout(
    tmp_path, release, monkeypatch, caplog, first, expected_attempts
):
    from contextlib import nullcontext

    from ocr_service import initialize as module

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    monkeypatch.setattr(module, "managed_installation", lambda *_args: nullcontext())
    state = tmp_path / "state"
    old = state / "installations" / "previous"
    old.mkdir(parents=True)
    (old / "preserve").write_bytes(b"previous installation")
    selected = state / "selected"
    selected.symlink_to(old.relative_to(state))
    outside = tmp_path / "outside.json"
    attempts = []

    def qualify(command):
        runtime = json.loads(Path(command[command.index("--runtime") + 1]).read_bytes())
        report = Path(command[command.index("--output") + 1])
        cache = Path(runtime["cache_dir"])
        artifacts = (report, report.with_suffix(".runtime.json"), Path(runtime["qualification_manifest"]))
        assert not cache.exists()
        assert all(not path.exists() for path in artifacts)
        assert selected.resolve() == old
        if first == "recovered" and attempts:
            attempts.append(report.parent)
            artifacts[1].write_text(json.dumps(runtime))
            artifacts[2].write_text("{}")
            report.write_text(json.dumps({
                "qualified": True,
                "stage": "complete",
                "accelerator": runtime["accelerator"],
                "image_identity": runtime["image_identity"],
                "profile_id": "test-profile",
                "runtime_fingerprint": "test-runtime",
                "manifest_sha256": sha(artifacts[2].read_bytes()),
                "device": DEVICE,
                "cases": [
                    {"kind": kind, "fresh_worker": repetition}
                    for kind in module.KINDS
                    for repetition in range(2)
                ],
            }))
            return 0
        cache.mkdir()
        (cache / "stale").write_bytes(b"candidate only")
        for path in artifacts[1:]:
            path.write_bytes(b"candidate only")
        attempt = first if not attempts else "collection_timeout"
        expected = {
            "qualified": False,
            "accelerator": runtime["accelerator"],
            "image_identity": runtime["image_identity"] if attempt != "other_image" else "other",
            "stage": "native_collection" if attempt != "other_stage" else "fresh_replay",
            "failure": "static_hook_collection_failed" if attempt == "static_hooks" else "TimeoutError",
            "collection_wait_timed_out": attempt != "unmarked",
        }
        if attempt in ("symlink", "hardlink"):
            outside.write_text(json.dumps(expected))
            if attempt == "symlink":
                report.symlink_to(outside)
            else:
                report.hardlink_to(outside)
        elif attempt == "malformed":
            report.write_text("{")
        else:
            report.write_text(json.dumps(expected))
        attempts.append(report.parent)
        return {"generic_exit": 1, "oom": 42, "static_hooks": 1}.get(attempt, 44)

    monkeypatch.setattr(module, "owned_qualification_process", qualify)
    kwargs = {"retry_collection_timeout": True} if first != "manual" else {}
    if first == "recovered":
        install = initialize_release(tmp_path, release, force=True, **kwargs)
        assert selected.resolve() == install
        assert (install / "complete.json").is_file()
    else:
        with pytest.raises(module.InitializationFailed):
            initialize_release(tmp_path, release, force=True, **kwargs)
        assert not any(candidate.exists() for candidate in attempts)
        assert selected.resolve() == old
    assert len(attempts) == expected_attempts
    assert (old / "preserve").read_bytes() == b"previous installation"
    if first in ("symlink", "hardlink"):
        assert outside.is_file()
    assert sum(r.getMessage() == "native_qualification_retry_collection_timeout" for r in caplog.records) == (
        expected_attempts - 1
    )
