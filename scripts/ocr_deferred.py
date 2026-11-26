"""Wait on the source host for an activation request while holding the launcher lock."""

import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)


class DeferredHandoffError(RuntimeError):
    """Native deployment may own the activation; only the launcher can reconcile it."""


def _generation(value: object) -> str:
    if not isinstance(value, str) or len(value) != 36 or str(UUID(value)) != value:
        raise ValueError("Expected a canonical activation UUID")
    return value


def _replace(path: Path, data: str, mode: int) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def _publish(path: Path, phase: str, generation: str | None) -> None:
    record = {"version": 1, "phase": phase, "generation": generation, "updated_at": time.time()}
    _replace(path, json.dumps(record) + "\n", 0o644)


def _last_status(path: Path) -> tuple[str, str | None]:
    try:
        record = json.loads(path.read_bytes())
    except FileNotFoundError:
        return "waiting", None
    if record["version"] != 1 or record["phase"] not in {"waiting", "building", "failed"}:
        raise ValueError("Invalid deferred OCR status")
    if record["phase"] == "waiting":
        if record["generation"] is not None:
            raise ValueError("Waiting status cannot carry an activation UUID")
        return "waiting", None
    return record["phase"], _generation(record["generation"])


def initialize_status(status_path: Path, *, interrupted_safe: bool = False) -> None:
    """Publish availability before starting the app; interrupted attempts require a new UUID."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    phase, generation = _last_status(status_path)
    if phase == "building" and not interrupted_safe:
        raise DeferredHandoffError("Reconcile interrupted native deployment before retrying")
    _publish(status_path, "failed" if generation else "waiting", generation)


def wait_for_ocr(
    status_path: Path,
    read_generation: Callable[[], str | None],
    build: Callable[[], None],
    *,
    interval: float = 5,
) -> None:
    """Run the trusted host callback once per generation and return on success."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Expected a positive heartbeat interval")
    initialize_status(status_path)
    phase, generation = _last_status(status_path)
    history = status_path.with_name(status_path.name + ".attempts")
    try:
        attempted = {_generation(line) for line in history.read_text().splitlines()}
    except FileNotFoundError:
        attempted = set()
    if generation:
        attempted.add(generation)
    stopped = threading.Event()
    state_lock = threading.Lock()

    def heartbeat() -> None:
        while not stopped.wait(interval):
            try:
                with state_lock:
                    _publish(status_path, phase, generation)
            except OSError:
                logger.exception("Could not refresh deferred OCR status")

    thread = threading.Thread(target=heartbeat, name="ocr-build-heartbeat", daemon=True)
    thread.start()
    try:
        while True:
            try:
                request = read_generation()
                if request is not None:
                    request = _generation(request)
            except Exception:
                logger.exception("Could not read deferred OCR activation")
                request = None
            if request is not None and request not in attempted:
                with state_lock:
                    generation, phase = request, "building"
                    _publish(status_path, phase, generation)
                    attempted.add(request)
                    _replace(history, "\n".join(sorted(attempted)) + "\n", 0o600)
                try:
                    build()
                except DeferredHandoffError:
                    raise
                except Exception:
                    logger.exception("Deferred OCR build failed")
                    with state_lock:
                        phase = "failed"
                        _publish(status_path, phase, generation)
                else:
                    return
            time.sleep(interval)
    finally:
        stopped.set()
        thread.join()
