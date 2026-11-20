"""Project-local activation records and Linux admission locks shared by app and OCR.

The app owns the request mount. The native controller owns the status mount.
Both may use the lock mount. Deployments expose each record mount to its reader
as read-only and run the two peers with the same UID. A new request represents
explicit intent and completed app-side admission checks.
"""

import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

_VERSION = 1
_MAX_RECORD_BYTES = 4096
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}")
_REQUEST = "request.json"
_STATUS = "status.json"
_REQUEST_LOCK = "request.lock"


class MaintenanceIntent(StrEnum):
    EXTERNAL_LLM_QUIESCED = "external-llm-quiesced"
    SEPARATE_OCR_DEVICE = "separate-ocr-device"


class ActivationPhase(StrEnum):
    PREPARING = "preparing"
    WAITING_FOR_ADMISSION = "waiting-for-admission"
    QUALIFYING = "qualifying"
    READY = "ready"
    FAILED = "failed"


class ActivationStateError(RuntimeError):
    """Activation storage is unavailable or contains untrusted state."""


class ActivationConflict(RuntimeError):
    """The requested transition conflicts with the current generation."""


class ActivationBusy(RuntimeError):
    """A peer holds conflicting admission or controller ownership."""


@dataclass(frozen=True)
class ActivationRequest:
    generation: UUID
    maintenance_intent: MaintenanceIntent


@dataclass(frozen=True)
class ActivationStatus:
    generation: UUID
    phase: ActivationPhase
    reason_code: str | None = None


@dataclass(frozen=True)
class ActivationStore:
    request_dir: Path = Path("/activation/request")
    status_dir: Path = Path("/activation/status")
    lock_dir: Path = Path("/activation/lock")

    def read_request(self) -> ActivationRequest | None:
        record = _read_record(self.request_dir, _REQUEST)
        if record is None:
            return None
        try:
            generation = _generation(record, {"maintenance_intent"})
            intent = MaintenanceIntent(_record_string(record["maintenance_intent"]))
            return ActivationRequest(generation, intent)
        except (TypeError, ValueError) as exc:
            raise ActivationStateError("Invalid OCR activation request") from exc

    def read_status(self, request: ActivationRequest) -> ActivationStatus | None:
        """Read the status for a request snapshot without creating any state."""
        record = _read_record(self.status_dir, _STATUS)
        if record is None:
            return None
        try:
            generation = _generation(record, {"phase", "reason_code"})
            phase = ActivationPhase(_record_string(record["phase"]))
            status = ActivationStatus(generation, phase, _validated_reason(record["reason_code"]))
        except (TypeError, ValueError) as exc:
            raise ActivationStateError("Invalid OCR activation status") from exc
        return status if status.generation == request.generation else None

    def submit_request(
        self, maintenance_intent: MaintenanceIntent, *, retry_generation: UUID | None = None
    ) -> ActivationRequest:
        """Record app-approved intent. A matching failed generation allows an explicit retry."""
        intent = MaintenanceIntent(maintenance_intent)
        with _lock(self.request_dir, _REQUEST_LOCK, exclusive=True, blocking=True):
            current = self.read_request()
            if current is None:
                if retry_generation is not None:
                    raise ActivationConflict("No OCR activation request to retry")
            elif current.generation == retry_generation:
                status = self.read_status(current)
                if status is None or status.phase != ActivationPhase.FAILED:
                    raise ActivationConflict("Only a failed OCR activation can be retried")
            else:
                if current.maintenance_intent != intent:
                    raise ActivationConflict("OCR activation already has a different maintenance intent")
                return current

            request = ActivationRequest(uuid4(), intent)
            _write_record(self.request_dir, _REQUEST, request.generation, maintenance_intent=intent)
            return request

    def runtime_enabled(self, request: ActivationRequest) -> bool:
        """Missing intent preserves legacy activation; old generations do not apply."""
        record = _read_record(self.request_dir, "runtime.json")
        if record is None:
            return True
        try:
            generation = _generation(record, {"enabled"})
            enabled = record["enabled"]
            if enabled not in ("true", "false"):
                raise ValueError("Invalid runtime intent")
            return generation != request.generation or enabled == "true"
        except (TypeError, ValueError) as exc:
            raise ActivationStateError("Invalid OCR runtime intent") from exc

    def set_runtime_enabled(self, request: ActivationRequest, enabled: bool) -> None:
        """Caller holds exclusive admission through routing publication."""
        if self.read_request() != request:
            raise ActivationConflict("OCR activation generation changed")
        _write_record(self.request_dir, "runtime.json", request.generation,
                      enabled="true" if enabled else "false")

    def shared_gpu(self) -> bool:
        """Existing and unconfigured deployments default to a shared GPU."""
        record = _read_record(self.request_dir, "topology.json")
        if record is None:
            return True
        try:
            _generation(record, {"shared_gpu"})
            shared_gpu = record["shared_gpu"]
            if shared_gpu not in ("true", "false"):
                raise ValueError("Invalid GPU topology")
            return shared_gpu == "true"
        except (TypeError, ValueError) as exc:
            raise ActivationStateError("Invalid GPU topology") from exc

    def set_shared_gpu(self, shared_gpu: bool) -> None:
        """Caller holds exclusive activation admission through routing publication."""
        _write_record(self.request_dir, "topology.json", uuid4(),
                      shared_gpu="true" if shared_gpu else "false")

    def publish_status(self, status: ActivationStatus, *, restoration_failed: bool = False) -> bool:
        """Publish for the current generation. The controller owns the status writer."""
        phase = ActivationPhase(status.phase)
        reason = _validated_reason(status.reason_code)
        # Take a read-only lock on the app's inode to fence app retry through
        # the status write. Only the app creates that inode.
        try:
            with _lock(
                self.request_dir, _REQUEST_LOCK, exclusive=False, blocking=True, create=False
            ):
                request = self.read_request()
                if request is None or request.generation != status.generation:
                    return False
                current = self.read_status(request)
                if current is not None:
                    if current == status:
                        return True
                    if current.phase is ActivationPhase.FAILED:
                        return False
                    if current.phase is ActivationPhase.READY and not (
                        restoration_failed and phase is ActivationPhase.FAILED
                        and reason == "native_activation_recovery_required"
                    ):
                        return False
                _write_record(self.status_dir, _STATUS, status.generation, phase=phase, reason_code=reason)
                return True
        except FileNotFoundError:
            return False

    def admission(self, *, exclusive: bool, blocking: bool = False) -> AbstractContextManager[None]:
        """Hold shared inference or exclusive qualification admission before any handoff lock."""
        return _lock(self.lock_dir, "admission.lock", exclusive=exclusive, blocking=blocking)

    def controller(self) -> AbstractContextManager[None]:
        """Own the native controller through the full serving process lifetime."""
        return _lock(self.lock_dir, "controller.lock", exclusive=True, blocking=False)


def _generation(record: dict[str, object], fields: set[str]) -> UUID:
    if (
        set(record) != fields | {"version", "generation"}
        or type(record["version"]) is not int
        or record["version"] != _VERSION
    ):
        raise ValueError("Unknown OCR activation record schema")
    value = _record_string(record["generation"])
    generation = UUID(value)
    if str(generation) != value:
        raise ValueError("Invalid OCR activation generation")
    return generation


def _record_string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected an OCR activation record string")
    return value


def _validated_reason(value: object) -> str | None:
    if value is None or isinstance(value, str) and _REASON_CODE.fullmatch(value):
        return value
    raise ValueError("Invalid OCR activation reason code")


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("Duplicate OCR activation record field")
        fields[key] = value
    return fields


@contextmanager
def _private_directory(path: Path, *, create: bool) -> Iterator[int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        if create:
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
            else:
                parent = os.open(path.parent, flags)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ActivationStateError("Unsafe OCR activation directory") from exc
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise ActivationStateError("Unsafe OCR activation directory permissions")
        yield descriptor
    finally:
        os.close(descriptor)


def _check_file(descriptor: int, *, lock: bool = False) -> None:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o177
        or info.st_nlink > 1
        or (lock and info.st_nlink != 1)
    ):
        raise ActivationStateError("Unsafe OCR activation file")


def _read_record(path: Path, filename: str) -> dict[str, object] | None:
    try:
        with _private_directory(path, create=False) as directory:
            descriptor = os.open(
                filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            with os.fdopen(descriptor, "rb") as stream:
                _check_file(stream.fileno())
                data = stream.read(_MAX_RECORD_BYTES + 1)
        if len(data) > _MAX_RECORD_BYTES:
            raise ValueError("OCR activation record exceeds size limit")
        record = json.loads(data, object_pairs_hook=_unique_fields)
        if not isinstance(record, dict):
            raise TypeError("OCR activation record must be an object")
        return record
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        raise ActivationStateError("Cannot read OCR activation record") from exc


def _write_record(path: Path, filename: str, generation: UUID, **fields: str | None) -> None:
    record = {"version": _VERSION, "generation": str(generation), **fields}
    temporary = f".{filename}.{uuid4().hex}.tmp"
    try:
        with _private_directory(path, create=True) as directory:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
                dir_fd=directory,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    json.dump(record, stream, separators=(",", ":"))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, filename, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
    except OSError as exc:
        raise ActivationStateError("Cannot publish OCR activation record") from exc


@contextmanager
def _lock(
    path: Path, filename: str, *, exclusive: bool, blocking: bool, create: bool = True
) -> Iterator[None]:
    # Keep the contract importable on app-only Windows installations.
    import fcntl

    with _private_directory(path, create=create) as directory:
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= os.O_CREAT | os.O_RDWR if create else os.O_RDONLY
        try:
            descriptor = os.open(filename, flags, 0o600, dir_fd=directory)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ActivationStateError("Cannot open OCR activation lock") from exc
        try:
            _check_file(descriptor, lock=True)
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            if not blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, operation)
            except BlockingIOError as exc:
                raise ActivationBusy("OCR activation lock is busy") from exc
            except OSError as exc:
                raise ActivationStateError("Cannot acquire OCR activation lock") from exc
            yield
        finally:
            os.close(descriptor)
