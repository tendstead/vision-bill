import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from ocr_contract import activation
from ocr_contract.activation import (
    ActivationBusy,
    ActivationConflict,
    ActivationPhase,
    ActivationStatus,
    ActivationStore,
    MaintenanceIntent,
)


@pytest.fixture
def store(tmp_path):
    return ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")


def test_passive_read_leaves_unrequested_storage_absent(store, tmp_path):
    assert store.read_request() is None
    assert list(tmp_path.iterdir()) == []


def test_duplicate_submissions_create_one_versioned_request(store):
    intent = MaintenanceIntent.EXTERNAL_LLM_QUIESCED
    with ThreadPoolExecutor(max_workers=4) as pool:
        requests = list(pool.map(store.submit_request, [intent] * 8))

    request = requests[0]
    assert requests == [request] * 8
    assert isinstance(request.generation, UUID)
    assert store.read_request() == request
    request_file = store.request_dir / "request.json"
    assert json.loads(request_file.read_text()) == {
        "version": 1,
        "generation": str(request.generation),
        "maintenance_intent": "external-llm-quiesced",
    }
    assert stat.S_IMODE(request_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.request_dir.stat().st_mode) == 0o700
    assert store.read_status(request) is None
    assert not store.status_dir.exists()
    with pytest.raises(ActivationConflict):
        store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)


def test_failed_setup_requires_explicit_retry_and_fences_stale_status(store):
    intent = MaintenanceIntent.SEPARATE_OCR_DEVICE
    first = store.submit_request(intent)
    failed = ActivationStatus(first.generation, ActivationPhase.FAILED, "ocr_setup_failed")
    assert store.publish_status(failed)
    assert store.read_status(first) == failed
    assert store.submit_request(intent) == first

    with ThreadPoolExecutor(max_workers=4) as pool:
        retries = list(
            pool.map(lambda _: store.submit_request(intent, retry_generation=first.generation), range(8))
        )

    retry = retries[0]
    assert retries == [retry] * 8
    assert retry.generation != first.generation
    assert store.read_status(retry) is None
    assert not store.publish_status(ActivationStatus(first.generation, ActivationPhase.READY))
    assert store.read_status(retry) is None
    with pytest.raises(ActivationConflict):
        store.submit_request(intent, retry_generation=retry.generation)


def test_status_replacement_keeps_interrupted_reader_whole_and_ready_terminal(store):
    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    initial = ActivationStatus(request.generation, ActivationPhase.PREPARING, "ocr_downloading")
    assert store.publish_status(initial)
    status_file = store.status_dir / "status.json"
    resumed = ActivationStore(store.request_dir, store.status_dir, store.lock_dir)
    assert resumed.read_request() == request
    assert resumed.read_status(request) == initial

    ready = ActivationStatus(request.generation, ActivationPhase.READY)
    with status_file.open() as previous:
        assert resumed.publish_status(ready)
        assert json.load(previous) == {
            "version": 1,
            "generation": str(request.generation),
            "phase": "preparing",
            "reason_code": "ocr_downloading",
        }
    assert stat.S_IMODE(status_file.stat().st_mode) == 0o600
    assert resumed.read_status(request) == ready
    assert resumed.publish_status(ready)
    assert not resumed.publish_status(initial)
    assert resumed.read_status(request) == ready


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "fifo", "permissions", "unknown-version"])
def test_untrusted_request_records_fail_closed(store, tmp_path, unsafe):
    store.request_dir.mkdir(mode=0o700)
    record = store.request_dir / "request.json"
    external = tmp_path / "external.json"
    external.write_text("external content")
    external.chmod(0o600)
    if unsafe == "symlink":
        record.symlink_to(external)
    elif unsafe == "hardlink":
        record.hardlink_to(external)
    elif unsafe == "fifo":
        os.mkfifo(record, 0o600)
    else:
        record.write_text(json.dumps({
            "version": 2 if unsafe == "unknown-version" else 1,
            "generation": str(uuid4()),
            "maintenance_intent": "separate-ocr-device",
        }))
        record.chmod(0o600 if unsafe == "unknown-version" else 0o644)

    with pytest.raises(activation.ActivationStateError):
        store.read_request()
    with pytest.raises(activation.ActivationStateError):
        store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    assert external.read_text() == "external content"


def test_activation_records_cannot_choose_external_requests_or_progress_text(store):
    with pytest.raises(ValueError):
        store.submit_request("https://assets.example/weights.tar")
    assert store.read_request() is None

    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    with pytest.raises(ValueError):
        store.publish_status(
            ActivationStatus(request.generation, ActivationPhase.FAILED, "/private/path")
        )
    assert store.read_status(request) is None


def test_activity_lock_contends_across_processes_on_one_inode(store):
    def try_exclusive_in_new_process():
        return subprocess.run(
            [sys.executable, "-c", """
from pathlib import Path
import sys
from ocr_contract.activation import ActivationBusy, ActivationStore

try:
    with ActivationStore(lock_dir=Path(sys.argv[1])).admission(exclusive=True):
        pass
except ActivationBusy:
    sys.exit(17)
""", str(store.lock_dir)],
            env={**os.environ, "PYTHONPATH": str(Path(activation.__file__).resolve().parents[1])},
            check=False,
            timeout=5,
        ).returncode

    with store.admission(exclusive=False):
        original = (store.lock_dir / "admission.lock").stat()
        with store.admission(exclusive=False):
            assert try_exclusive_in_new_process() == 17
    assert try_exclusive_in_new_process() == 0
    with store.admission(exclusive=True):
        with pytest.raises(ActivationBusy), store.admission(exclusive=False):
            pytest.fail("inference entered while activation was exclusive")
        current = (store.lock_dir / "admission.lock").stat()
        assert (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)


def test_controller_lock_is_independent_of_admission(store):
    with (
        store.controller(),
        store.admission(exclusive=False),
        pytest.raises(ActivationBusy),
        store.controller(),
    ):
        pytest.fail("another controller entered while the first was present")
    with store.controller(), store.admission(exclusive=True):
        assert store.read_request() is None


def test_runtime_intent_is_generation_bound_and_missing_defaults_enabled(tmp_path):
    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    request = store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    assert store.runtime_enabled(request)
    store.set_runtime_enabled(request, False)
    assert not store.runtime_enabled(request)
    assert ActivationStore(store.request_dir, store.status_dir, store.lock_dir).runtime_enabled(request) is False
    store.publish_status(ActivationStatus(request.generation, ActivationPhase.FAILED))
    replacement = store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED, retry_generation=request.generation)
    assert store.runtime_enabled(replacement)
