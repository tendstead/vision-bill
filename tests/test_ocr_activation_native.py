"""CPU evidence for activation admission and the native process boundary."""

import io
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path

import pytest
from ocr_contract.activation import (
    ActivationBusy,
    ActivationPhase,
    ActivationStatus,
    ActivationStore,
    MaintenanceIntent,
)
from test_ocr_provision import release as _release
from test_ocr_provision import sha

release = _release


@pytest.fixture
def store(tmp_path):
    return ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "coord")


def test_idle_controller_serves_liveness_without_release_inputs_or_native_state(tmp_path):
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    service_dir = Path(__file__).parents[1] / "paddleocr-service"
    environment = {
        **os.environ,
        "OCR_PORT": str(port),
        "OCR_STATE_DIR": str(tmp_path / "no-weights"),
        "OCR_IMAGE_ID": "",
        "OCR_INSTALLATION_LOCK_PATH": str(tmp_path / "installation.lock"),
        "OCR_HANDOFF_LOCK_PATH": str(tmp_path / "handoff.lock"),
    }
    command = (
        "import sys; from pathlib import Path; from ocr_contract.activation import ActivationStore; "
        "from ocr_service.activation import main; root = Path(sys.argv[1]); "
        "main(ActivationStore(root / 'request', root / 'status', root / 'coord'))"
    )
    with subprocess.Popen(
        [sys.executable, "-c", command, str(tmp_path)], cwd=service_dir, env=environment
    ) as controller:
        try:
            until = time.monotonic() + 5
            while controller.poll() is None and time.monotonic() < until:
                conn = HTTPConnection("127.0.0.1", port, timeout=0.25)
                try:
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    assert response.status == 200
                    assert json.loads(response.read()) == {"status": "ok"}
                    conn.request("GET", "/ready")
                    response = conn.getresponse()
                    assert response.status == 503
                    assert json.loads(response.read())["ready"] is False
                    break
                except ConnectionError:
                    time.sleep(0.05)
                finally:
                    conn.close()
            else:
                pytest.fail(f"Idle health never responded (exit={controller.poll()})")
            assert not (tmp_path / "no-weights").exists()
            assert not (tmp_path / "installation.lock").exists()
            assert not (tmp_path / "handoff.lock").exists()
            assert not (tmp_path / "status/status.json").exists()
        finally:
            if controller.poll() is None:
                controller.terminate()
                assert controller.wait(timeout=5) == 143


@pytest.fixture
def inputs(tmp_path, release):
    from ocr_service.activation import ReleaseInputs

    bundle, _, image_config = release
    return ReleaseInputs(
        bundle,
        tmp_path / "native-state",
        image_config,
        "sha256:" + sha(image_config.read_bytes()),
        tmp_path / "handoff.lock",
    )


@pytest.fixture
def idle_health():
    from ocr_service.activation import IdleHealth

    with IdleHealth(0) as health:
        yield health


def test_native_waits_for_app_admissions_and_publishes_only_after_park(
    store, inputs, idle_health, monkeypatch
):
    from ocr_service import activation

    request = store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    scopes = []
    installation = inputs.state_dir / "installation"

    def prepare(*args, **kwargs):
        assert args == (inputs.bundle_path, inputs.state_dir, inputs.image_config, inputs.image_id)
        assert kwargs == {
            "maintenance_intent": "external-llm-quiesced",
            "handoff_lock": inputs.handoff_lock,
            "repair": True,
        }
        assert store.read_status(request).phase is ActivationPhase.PREPARING
        with pytest.raises(ActivationBusy), store.admission(exclusive=False):
            pass
        scopes.append("initialized")
        return installation

    @contextmanager
    def serving_scope(role, _path):
        assert role == "ocr"
        assert scopes == ["initialized"]
        scopes.append("serving")
        yield
        scopes.append("drained")

    class Process:
        def wait(self):
            with store.admission(exclusive=False):
                assert store.read_status(request).phase is ActivationPhase.READY
            return 0

    def start(release_inputs, selected, port):
        assert release_inputs == inputs
        assert selected == installation
        assert port == idle_health.port
        assert scopes == ["initialized", "serving"]
        return Process()

    def verify_park(process, port):
        assert isinstance(process, Process)
        assert port == idle_health.port
        assert store.read_status(request).phase is ActivationPhase.PREPARING
        with pytest.raises(ActivationBusy), store.admission(exclusive=False):
            pass

    monkeypatch.setattr(activation, "prepare_installation", prepare)
    monkeypatch.setattr(activation, "managed_installation", serving_scope)
    monkeypatch.setattr(activation, "start_server", start)
    monkeypatch.setattr(activation, "wait_for_park", verify_park)

    with store.admission(exclusive=False):
        assert activation.process_request(store, request, inputs, idle_health) is False
        assert store.read_status(request).phase is ActivationPhase.WAITING_FOR_ADMISSION
    assert scopes == []
    with pytest.raises(activation.ServingExited):
        activation.process_request(store, request, inputs, idle_health)
    assert scopes == ["initialized", "serving", "drained"]
    assert store.read_status(request) == ActivationStatus(request.generation, ActivationPhase.READY)


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("archive", "native_model_integrity_failed"),
        ("download", "native_model_download_failed"),
    ],
)
def test_activation_failure_requires_an_explicit_new_generation(
    store, inputs, idle_health, release, monkeypatch, failure, reason
):
    from ocr_service import activation, initialize, provision

    path, bundle, _ = release
    for name, row in bundle["models"].items():
        row["source"] = f"https://example.invalid/{name}.tar"
    path.write_text(json.dumps(bundle))
    attempts = []

    def fetch_archive(url, *, timeout):
        attempts.append(url)
        if failure == "download":
            raise ConnectionError("private upstream location")

        class ArchiveResponse(io.BytesIO):
            def geturl(self):
                return url

        return ArchiveResponse(b"incorrect archive bytes")

    @contextmanager
    def initializer_scope(role, _path):
        assert role == "initializer"
        yield

    monkeypatch.setattr(provision.urllib.request, "urlopen", fetch_archive)
    monkeypatch.setattr(initialize, "managed_installation", initializer_scope)
    monkeypatch.setattr(initialize, "inspect_device_identity", lambda *args: {})
    monkeypatch.setattr(initialize, "verify_package_pins", lambda bundle: None, raising=False)
    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    assert activation.process_request(store, request, inputs, idle_health) is True
    assert store.read_status(request) == ActivationStatus(
        request.generation, ActivationPhase.FAILED, reason
    )
    assert not (inputs.state_dir / "selected").exists()
    assert len(attempts) == 1
    assert activation.process_request(store, request, inputs, idle_health) is False
    assert len(attempts) == 1

    retry = store.submit_request(request.maintenance_intent, retry_generation=request.generation)
    assert activation.process_request(store, request, inputs, idle_health) is False
    assert activation.process_request(store, retry, inputs, idle_health) is True
    assert attempts == [attempts[0], attempts[0]]


def test_local_model_archives_cannot_start_automatic_activation(
    store, inputs, idle_health, monkeypatch
):
    from ocr_service import activation, initialize

    @contextmanager
    def initializer_scope(role, _path):
        assert role == "initializer"
        yield

    def forbidden(*args, **kwargs):
        pytest.fail("A local archive manifest reached device detection or provisioning")

    monkeypatch.setattr(initialize, "managed_installation", initializer_scope)
    monkeypatch.setattr(initialize, "inspect_device_identity", forbidden)
    monkeypatch.setattr(initialize, "provision_bundle", forbidden)
    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    assert activation.process_request(store, request, inputs, idle_health) is True
    assert store.read_status(request) == ActivationStatus(
        request.generation, ActivationPhase.FAILED, "native_activation_inputs_invalid"
    )
    assert not inputs.state_dir.exists()


def test_a_published_generation_restarts_from_verified_cache_without_requalifying(
    store, inputs, idle_health, monkeypatch
):
    from ocr_service import activation

    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    ready = ActivationStatus(request.generation, ActivationPhase.READY)
    assert store.publish_status(ready)

    def forbidden(*args, **kwargs):
        pytest.fail("A previously activated generation must use the verified cache")

    def invalid_cache(*args):
        assert args == (inputs.bundle_path, inputs.state_dir, inputs.image_config, inputs.image_id)
        with pytest.raises(ActivationBusy), store.admission(exclusive=False):
            pass
        raise ValueError("Invalid cache")

    monkeypatch.setattr(activation, "prepare_installation", forbidden)
    monkeypatch.setattr(activation, "start_server", forbidden)
    monkeypatch.setattr(activation, "check_prepared_installation", invalid_cache)
    assert activation.process_request(store, request, inputs, idle_health) is True
    assert store.read_status(request) == ActivationStatus(
        request.generation, ActivationPhase.FAILED, "native_activation_recovery_required"
    )


@pytest.mark.parametrize("busy_scope", ["cache", "serving"])
def test_cached_restart_retries_busy_installation_lock(
    store, inputs, idle_health, tmp_path, monkeypatch, busy_scope
):
    from ocr_service import activation
    from ocr_service.supervision import installation_lock, role_lock_path

    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    ready = ActivationStatus(request.generation, ActivationPhase.READY)
    assert store.publish_status(ready)
    lock = tmp_path / "installation.lock"
    lock_to_block = lock if busy_scope == "cache" else role_lock_path(lock, "ocr")
    installation = tmp_path / "installation"
    parked = []

    def cached(*args):
        with installation_lock(lock, exclusive=False):
            return installation

    @contextmanager
    def serving_scope(role, path):
        assert role == "ocr"
        assert path == lock
        with installation_lock(role_lock_path(lock, "ocr"), exclusive=True):
            yield

    def forbidden(*args, **kwargs):
        pytest.fail("A busy cached restart must never requalify")

    class Process:
        def wait(self):
            return 0

    def wait_for_park(_process, port):
        parked.append(port)

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    monkeypatch.setattr(activation, "prepare_installation", forbidden)
    monkeypatch.setattr(activation, "check_prepared_installation", cached)
    monkeypatch.setattr(activation, "managed_installation", serving_scope)
    monkeypatch.setattr(activation, "start_server", lambda *args: Process())
    monkeypatch.setattr(activation, "wait_for_park", wait_for_park)

    with installation_lock(lock_to_block, exclusive=True):
        assert activation.process_request(store, request, inputs, idle_health) is False
    assert parked == []
    assert store.read_status(request) == ready

    with pytest.raises(activation.ServingExited):
        activation.process_request(store, request, inputs, idle_health)
    assert parked == [idle_health.port]
    assert store.read_status(request) == ready


def test_failed_native_cleanup_reports_existing_recovery_fence(
    store, inputs, idle_health, tmp_path, monkeypatch
):
    from ocr_service import activation
    from ocr_service.supervision import marker_path

    lock = tmp_path / "installation.lock"
    marker = marker_path(lock, "initializer")
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))

    def failed_cleanup(*args, **kwargs):
        marker.write_text("retained crash fence")
        raise RuntimeError("Cannot confirm initializer process exit")

    monkeypatch.setattr(activation, "prepare_installation", failed_cleanup)
    request = store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    assert activation.process_request(store, request, inputs, idle_health) is True
    assert marker.is_file()
    assert store.read_status(request) == ActivationStatus(
        request.generation, ActivationPhase.FAILED, "native_activation_recovery_required"
    )


def test_existing_native_api_parks_without_initializing_the_gpu(inputs, idle_health, monkeypatch):
    from ocr_service import activation

    installation = inputs.state_dir / "installations" / "cpu-test-only"
    installation.mkdir(parents=True)
    (inputs.state_dir / "selected").symlink_to(installation, target_is_directory=True)
    (installation / "selection.json").write_text(json.dumps({"smoke_path": "/absent.png"}))
    (installation / "runtime.json").write_text(
        json.dumps(
            {
                "accelerator": "cuda",
                "gpu_index": 0,
                "model_dirs": {},
                "cache_dir": "/absent-cache",
                "qualification_manifest": "/absent-qualification.json",
                "expected_packages": {},
                "expected_paddle_commit": "cpu-test-only",
                "image_identity": "cpu-test-only",
            }
        )
    )
    monkeypatch.setenv("OCR_RUNTIME_SETTINGS_PATH", str(installation / "absent.json"))
    monkeypatch.setenv("OCR_SERVICE_MODE", "standalone")
    monkeypatch.setattr(activation, "PARK_TIMEOUT_SECONDS", 5)

    with (
        idle_health.paused(),
        activation.start_server(inputs, installation, idle_health.port) as child,
    ):
        try:
            activation.wait_for_park(child, idle_health.port)
            assert child.poll() is None
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)


def test_prepare_and_restore_use_verified_models_without_qualification_or_fixtures(
    inputs,
    release,
    idle_health,
    monkeypatch,
):
    from ocr_service import activation, initialize, provision
    from test_ocr_initialize import DEVICE

    path, raw, _ = release
    legacy_bundle = provision.load_release_bundle(path)
    # An upgraded image can reuse archives populated before a failed qualification.
    old = provision.provision_bundle(legacy_bundle, inputs.state_dir)
    for name, asset in raw["models"].items():
        asset["source"] = f"https://example.invalid/{name}.tar"
    path.write_text(json.dumps(raw))
    Path(raw["fixtures"]["source"]).unlink()

    @contextmanager
    def initializer(role, lock):
        assert role == "initializer"
        yield

    def forbidden(*args, **kwargs):
        pytest.fail("Preparation or restart ran inference, fetched a model, or staged a fixture")

    monkeypatch.setattr(initialize, "managed_installation", initializer)
    monkeypatch.setattr(initialize, "inspect_device_identity", lambda *args: DEVICE)
    monkeypatch.setattr(initialize, "verify_package_pins", lambda bundle: None, raising=False)
    monkeypatch.setattr(initialize, "owned_qualification_process", forbidden)
    monkeypatch.setattr(provision.urllib.request, "urlopen", forbidden)
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(inputs.state_dir / "installation.lock"))
    args = (inputs.bundle_path, inputs.state_dir, inputs.image_config, inputs.image_id)
    root = initialize.prepare_installation(
        *args,
        maintenance_intent=MaintenanceIntent.SEPARATE_OCR_DEVICE,
        handoff_lock=inputs.handoff_lock,
    )
    assert (inputs.state_dir / "selected").resolve() == root
    assert not (root / "qualification.json").exists()
    runtime = json.loads((root / "runtime.json").read_bytes())
    assert runtime["schema_version"] == 2 and runtime["device_identity"] == DEVICE
    assert "qualification_manifest" not in runtime
    for name, directory in runtime["model_dirs"].items():
        assert (Path(directory) / "weights").read_bytes() == name.encode()
    assert initialize.check_prepared_installation(*args) == root
    # The actual server can announce itself parked on a CPU-only host with no ML packages;
    # it starts from the published runtime alone, before the operator has a real receipt.
    with idle_health.paused(), activation.start_server(inputs, root, idle_health.port) as child:
        try:
            activation.wait_for_park(child, idle_health.port)
            connection = HTTPConnection("127.0.0.1", idle_health.port, timeout=3)
            try:
                connection.request("GET", "/info")
                info = json.loads(connection.getresponse().read())
                assert info["service_mode"] == "managed"
                assert info["worker"]["reason_code"] == "ocr_parked"
            finally:
                connection.close()
        finally:
            child.terminate()
            child.wait(timeout=5)
    assert old.smoke_path.is_file()
    (Path(runtime["model_dirs"]["UVDoc"]) / "weights").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        initialize.check_prepared_installation(*args)

    # A failed replacement preserves the existing selection and every valid model.
    archive = inputs.state_dir / "archives" / raw["models"]["UVDoc"]["sha256"]
    good_archive = archive.read_bytes()
    archive.write_bytes(b"bad cached archive")
    requests = []

    class Response(io.BytesIO):
        def geturl(self):
            return raw["models"]["UVDoc"]["source"]

    def replacement(url, *, timeout):
        assert url == raw["models"]["UVDoc"]["source"]
        requests.append(url)
        return Response(b"bad download" if len(requests) == 1 else good_archive)

    monkeypatch.setattr(provision.urllib.request, "urlopen", replacement)
    with pytest.raises(initialize.PreparationFailed, match="native_model_integrity_failed"):
        initialize.prepare_installation(
            *args,
            maintenance_intent=MaintenanceIntent.SEPARATE_OCR_DEVICE,
            handoff_lock=inputs.handoff_lock,
            repair=True,
        )
    assert (inputs.state_dir / "selected").resolve() == root
    assert archive.read_bytes() == b"bad cached archive"
    repaired = initialize.prepare_installation(
        *args,
        maintenance_intent=MaintenanceIntent.SEPARATE_OCR_DEVICE,
        handoff_lock=inputs.handoff_lock,
        repair=True,
    )
    assert initialize.check_prepared_installation(*args) == repaired
    assert len(requests) == 2
    assert archive.read_bytes() == good_archive
    assert (Path(runtime["model_dirs"]["UVDoc"]) / "weights").read_bytes() == b"UVDoc"

    # A redirected cache root can present perfectly matching bytes outside native ownership.
    models = inputs.state_dir / "models"
    moved = inputs.state_dir.parent / "external-models"
    models.rename(moved)
    models.symlink_to(moved, target_is_directory=True)
    with pytest.raises(initialize.PreparationFailed, match="native_model_integrity_failed"):
        initialize.check_prepared_installation(*args)


def test_unclean_handoff_journal_remains_fenced_after_setup_failure(store, inputs, idle_health):
    from ocr_service import activation

    inputs.handoff_lock.with_suffix(".json").write_text('{"unsettled": true}')
    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    assert activation.process_request(store, request, inputs, idle_health) is True
    assert store.read_status(request) == ActivationStatus(
        request.generation, ActivationPhase.FAILED, "native_activation_recovery_required"
    )
