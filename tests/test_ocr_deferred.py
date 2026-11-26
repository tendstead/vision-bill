import json
import threading
import time
from uuid import uuid4

import pytest

from scripts.ocr_deferred import DeferredHandoffError, initialize_status, wait_for_ocr


def test_waits_and_ignores_malformed_requests(tmp_path):
    path = tmp_path / "deferred" / "status.json"
    initialize_status(path)
    generation = str(uuid4())
    requests = iter([None, "not-a-uuid", None, generation])
    observed = []

    def read():
        assert json.loads(path.read_bytes())["phase"] == "waiting"
        return next(requests)

    def build():
        observed.append(json.loads(path.read_bytes()))

    wait_for_ocr(path, read, build, interval=0.002)
    assert len(observed) == 1
    assert observed[0] == {
        "version": 1,
        "phase": "building",
        "generation": generation,
        "updated_at": observed[0]["updated_at"],
    }


def test_failed_generation_waits_for_a_fresh_request_even_after_restart(tmp_path):
    path = tmp_path / "status.json"
    first, second, third = (str(uuid4()) for _ in range(3))
    calls = []
    requests = iter([first, first, second, None, first, second, third])
    phases = []

    def read():
        phases.append(json.loads(path.read_bytes())["phase"])
        request = next(requests)
        if request is None:
            raise KeyboardInterrupt
        return request

    def build():
        generation = json.loads(path.read_bytes())["generation"]
        calls.append(generation)
        if generation != third:
            raise OSError("build exited")

    with pytest.raises(KeyboardInterrupt):
        wait_for_ocr(path, read, build, interval=0.002)
    initialize_status(path)
    wait_for_ocr(path, read, build, interval=0.002)
    assert calls == [first, second, third]
    assert phases == ["waiting", *(["failed"] * 6)]


def test_heartbeat_advances_while_synchronous_build_is_busy(tmp_path):
    path = tmp_path / "status.json"
    generation = str(uuid4())
    advanced = threading.Event()

    def build():
        initial = json.loads(path.read_bytes())["updated_at"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            record = json.loads(path.read_bytes())
            if record["updated_at"] > initial and record["generation"] == generation:
                advanced.set()
                return
            time.sleep(0.002)

    wait_for_ocr(path, lambda: generation, build, interval=0.015)
    assert advanced.is_set()


def test_possible_native_handoff_requires_launcher_reconciliation(tmp_path):
    path = tmp_path / "status.json"
    generation = str(uuid4())

    def build():
        raise DeferredHandoffError("native controller may be running")

    with pytest.raises(DeferredHandoffError, match="controller"):
        wait_for_ocr(path, lambda: generation, build, interval=0.01)
    original = path.read_bytes()
    assert json.loads(original)["phase"] == "building"
    with pytest.raises(DeferredHandoffError, match="Reconcile"):
        initialize_status(path)
    assert path.read_bytes() == original
    initialize_status(path, interrupted_safe=True)
    assert json.loads(path.read_bytes())["phase"] == "failed"
