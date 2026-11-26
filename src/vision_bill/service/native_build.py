"""Read-only source-launcher heartbeat and the app's pre-controller request transitions."""

import json
import os
import stat
import time
from dataclasses import dataclass
from uuid import UUID, uuid4

from ocr_contract.activation import (
    _MAX_RECORD_BYTES,
    _REQUEST,
    _REQUEST_LOCK,
    ActivationConflict,
    ActivationRequest,
    ActivationStore,
    MaintenanceIntent,
    _lock,
    _private_directory,
    _record_string,
    _unique_fields,
    _write_record,
)


@dataclass(frozen=True)
class NativeBuildStatus:
    phase: str
    generation: UUID | None


def native_build_status() -> NativeBuildStatus | None:
    """Read the source launcher's bounded heartbeat from its read-only mount."""
    path = os.environ.get("VISION_BILL_OCR_NATIVE_BUILD_STATUS")
    if not path:
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECORD_BYTES:
                return None
            data = stream.read(_MAX_RECORD_BYTES + 1)
        if len(data) > _MAX_RECORD_BYTES:
            return None
        record = json.loads(data, object_pairs_hook=_unique_fields)
        if not isinstance(record, dict) or set(record) != {"version", "updated_at", "generation", "phase"}:
            return None
        if type(record["version"]) is not int or record["version"] != 1:
            return None
        timestamp = record["updated_at"]
        if type(timestamp) not in (int, float) or not 0 <= time.time() - timestamp <= 30:
            return None
        if record["phase"] == "waiting" and record["generation"] is None:
            return NativeBuildStatus("waiting", None)
        if record["phase"] in ("building", "failed"):
            generation = UUID(_record_string(record["generation"]))
            if str(generation) == record["generation"]:
                return NativeBuildStatus(record["phase"], generation)
    except (OSError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    return None


def native_build_failed(request: ActivationRequest) -> bool:
    return native_build_status() == NativeBuildStatus("failed", request.generation)


def resolve_failed_build(
    store: ActivationStore, request: ActivationRequest, *, retry_intent: MaintenanceIntent | None = None,
) -> None:
    """Fence explicit OFF/retry against native status publication on the same request inode."""
    with _lock(store.request_dir, _REQUEST_LOCK, exclusive=True, blocking=True):
        if (
            store.read_request() != request or store.read_status(request) is not None
            or not native_build_failed(request)
        ):
            raise ActivationConflict("OCR build failure is no longer current")
        if retry_intent is not None:
            _write_record(store.request_dir, _REQUEST, uuid4(), maintenance_intent=retry_intent)
        else:
            with _private_directory(store.request_dir, create=False) as directory:
                os.replace(_REQUEST, "last-failed-build.json", src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
