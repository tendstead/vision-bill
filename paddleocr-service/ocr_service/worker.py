"""Single native process with bounded admission and request-owned deadlines."""

import json
import math
import multiprocessing as mp
import os
import queue
import socket
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ocr_contract import GpuMemorySnapshot, OcrResponse, WorkerReadiness
from ocr_contract.document_image import decode_working_png, validate_working_png
from ocr_contract.limits import MAX_OCR_RESPONSE_BYTES

from .gpu_memory import GPU_OOM_CODE, is_gpu_out_of_memory, probe_gpu_memory
from .profile import (
    InvalidReceiptOutput,
    PreparedRuntimeSettings,
    RuntimeIdentityMismatch,
    RuntimeSettings,
    load_profile,
    serving_identity,
)

MAX_CONTROL_BYTES = 4096


class ControlChannel:
    """Atomic small datagrams: receive never waits for a second frame fragment."""

    def __init__(self, endpoint: socket.socket):
        self.socket = endpoint

    @property
    def closed(self) -> bool:
        return self.socket.fileno() < 0

    def put(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, default=str).encode()
        if len(payload) > MAX_CONTROL_BYTES:
            raise ValueError("Control notification exceeds its fixed bound")
        self.socket.settimeout(0.1)
        self.socket.send(payload)

    def get(self, *, timeout: float) -> dict[str, Any]:
        self.socket.settimeout(timeout)
        try:
            payload, _, flags, _ = self.socket.recvmsg(MAX_CONTROL_BYTES)
        except (TimeoutError, BlockingIOError):
            raise queue.Empty from None
        if flags & socket.MSG_TRUNC:
            raise ValueError("Truncated control datagram")
        message = json.loads(payload)
        if not isinstance(message, dict):
            raise TypeError("Control notification is not an object")
        return message

    def cancel_join_thread(self) -> None:
        pass

    def close(self) -> None:
        self.socket.close()


def publish_result(result: OcrResponse, directory: Path) -> str:
    result.check_wire_bounds()
    view = result.working_image
    validate_working_png(decode_working_png(view.png_base64), view)
    payload = result.dump_checked_wire()
    destination = directory / (uuid4().hex + ".json")
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as temporary:
        temporary.write(payload)
        pending = Path(temporary.name)
    pending.replace(destination)
    return destination.name


def read_result(directory: Path, name: Any) -> OcrResponse:
    if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".json":
        raise ValueError("Invalid result artifact identity")
    UUID(Path(name).stem)
    path = directory / name
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as artifact:
        info = os.fstat(artifact.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_OCR_RESPONSE_BYTES:
            raise ValueError("Invalid result artifact type or size")
        payload = artifact.read(info.st_size + 1)
        if len(payload) != info.st_size:
            raise ValueError("Result artifact size changed while reading")
    path.unlink()
    return OcrResponse.parse_wire(payload)


@dataclass(frozen=True)
class OcrRequest:
    request_id: UUID
    original_path: Path
    photo_index: int
    source_sha256: str


class OcrWorkerError(Exception):
    def __init__(self, code: str, status_code: int, *, cleanup_safe: bool = True):
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.cleanup_safe = cleanup_safe


def _worker_main(
    input_channel: Any,
    output_channel: Any,
    generation: int,
    profile_path: Path,
    runtime: PreparedRuntimeSettings | RuntimeSettings | None,
    reply_directory: Path,
) -> None:
    # No exception text or receipt content crosses the general error/log boundary.
    try:
        from .pipeline import build_serving_pipeline

        if runtime is None:
            raise ValueError("Explicit local runtime settings are required")
        adapter = build_serving_pipeline(load_profile(profile_path), runtime)
        output_channel.put(
            {
                "kind": "ready",
                "generation": generation,
                "profile_id": adapter.profile_id,
                "runtime_fingerprint": adapter.runtime_fingerprint,
            }
        )
    except Exception as exc:  # noqa: BLE001 - native boundary must hide exception details
        code = "ocr_model_load_failed"
        if is_gpu_out_of_memory(exc):
            code = GPU_OOM_CODE
        elif isinstance(exc, RuntimeIdentityMismatch):
            code = "ocr_runtime_identity_mismatch"
        output_channel.put({"kind": "init_failed", "code": code, "generation": generation})
        return
    while True:
        request = input_channel.get()
        if request is None:
            return
        message: dict[str, Any] = {
            "generation": generation,
            "request_id": request.request_id,
        }
        try:
            result = adapter.predict(
                request.original_path,
                request_id=request.request_id,
                photo_index=request.photo_index,
                source_sha256=request.source_sha256,
            )
            try:
                artifact = publish_result(result, reply_directory)
            except (ValueError, TypeError, AttributeError) as exc:
                raise InvalidReceiptOutput("Invalid OCR response artifact") from exc
            message.update(kind="result", artifact=artifact)
        except Exception as exc:  # noqa: BLE001 - native boundary must hide exception details
            # A failed native execution can leave the accelerator context unusable.
            code = "ocr_inference_failed"
            if is_gpu_out_of_memory(exc):
                code = GPU_OOM_CODE
            elif isinstance(exc, InvalidReceiptOutput):
                code = "ocr_invalid_result"
            message.update(kind="worker_failed", code=code)
            output_channel.put(message)
            return
        output_channel.put(message)


class OcrWorkerManager:
    def __init__(
        self,
        *,
        runtime: PreparedRuntimeSettings | RuntimeSettings | None = None,
        profile_path: Path = Path(__file__).parents[1] / "profile.json",
        startup_timeout: float = 600,
        mp_context: Any = None,
    ):
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("Startup timeout must be finite and positive")
        self._ctx = mp_context if mp_context is not None else mp.get_context("spawn")
        self._runtime = runtime
        self._profile_path = profile_path
        self._startup_timeout = startup_timeout
        self._control = threading.RLock()
        self._state = threading.Lock()
        self._active = threading.Lock()
        self._admission = threading.Lock()
        self._standalone_admitted = 0
        self._standalone_session = False
        self._stopped = threading.Event()
        self._generation = 0
        self._idle_recovery_generation: int | None = None
        self._recovery_attempted_generation: int | None = None
        self._reason: str | None = "ocr_parked"
        self._parked = True
        self._owner: str | None = None
        self._owner_generation = 0
        self._owner_load_allowed = False
        self.last_identity: dict[str, str | None] = {}
        if isinstance(runtime, PreparedRuntimeSettings):
            self.last_identity.update(
                serving_identity(
                    load_profile(profile_path),
                    runtime,
                    runtime.model_content,
                    runtime.device_identity,
                )
            )
        self._profile_id: str | None = None
        self._runtime_fingerprint: str | None = None
        self._preload_memory: GpuMemorySnapshot | None = None
        self._ready = False
        self._admitted = 0
        self._proc: Any = None
        self._in_q: Any = None
        self._out_q: Any = None
        self._reply_sender: ControlChannel | None = None
        self._reply_directory: tempfile.TemporaryDirectory[str] | None = None

    @property
    def pending_count(self) -> int:
        with self._state:
            return max(0, self._admitted - 1)

    def readiness(self) -> WorkerReadiness:
        recover_generation = None
        with self._state:
            try:
                alive = self._ready and self._proc is not None and self._proc.is_alive()
            except (OSError, ValueError):
                alive = False
            ready = self._ready and alive and not self._stopped.is_set()
            reason = self._reason if not ready else None
            if self._stopped.is_set():
                reason = "ocr_stopped"
            elif self._ready and not alive and not self._parked:
                reason = "ocr_worker_crashed"
                self._ready, self._reason = False, reason
                if (
                    not self._standalone_session
                    and self._idle_recovery_generation != self._generation
                ):
                    self._idle_recovery_generation = self._generation
                    recover_generation = self._generation
            observation = WorkerReadiness(
                ready=ready,
                generation=max(1, self._generation),
                profile_id=self._profile_id,
                reason_code=reason,
                runtime_fingerprint=self._runtime_fingerprint if ready else None,
                preload_memory=self._preload_memory,
            )

        if recover_generation is not None:
            threading.Thread(
                target=self._recover_idle,
                args=(recover_generation,),
                name="ocr-idle-recovery",
                daemon=True,
            ).start()
        return observation

    def _recover_idle(self, generation: int) -> None:
        # A polling thread never owns an in-flight request's artifacts. Wait
        # for its owner to finish, then recheck generation under lifecycle lock:
        # request recovery or shutdown may already have retired this process.
        with self._active:
            self._restart_generation(generation)

    def _restart_generation(self, generation: int) -> None:
        with self._control:
            if (
                self._stopped.is_set()
                or self._parked
                or self._standalone_session
                or generation != self._generation
                or self._recovery_attempted_generation == generation
            ):
                return
            # Poll and request callbacks consume the same recovery right before
            # teardown/startup. A failed replacement must not grant another try.
            self._recovery_attempted_generation = generation
            try:
                self.start()
            except OcrWorkerError:
                # Unconfirmed death preserves ownership and remains unready.
                pass

    def _set_unready(self, reason: str, *, keep_preload: bool = False) -> None:
        with self._state:
            self._ready, self._reason, self._profile_id = False, reason, None
            self._runtime_fingerprint = None
            if not keep_preload:
                self._preload_memory = None

    def start(
        self,
        *,
        deadline_monotonic: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Bounded initialization, called off the HTTP event loop."""
        with self._control:
            if self._stopped.is_set() or self.readiness().ready:
                return
            self._parked = False
            self._set_unready("ocr_starting")
            if self._proc is not None:
                self._teardown()
            self._generation += 1
            deadline = time.monotonic() + self._startup_timeout
            if deadline_monotonic is not None:
                deadline = min(deadline, deadline_monotonic)

            def interruption(now: float) -> str | None:
                if self._stopped.is_set():
                    return "ocr_stopped"
                if cancel_event is not None and cancel_event.is_set():
                    return "ocr_cancelled"
                if now < deadline:
                    return None
                if deadline_monotonic is not None and now >= deadline_monotonic:
                    return "ocr_deadline_exceeded"
                return "ocr_startup_timeout"

            try:
                now = time.monotonic()
                if reason := interruption(now):
                    self._set_unready(reason)
                    return
                if self._runtime is not None:
                    # Keep ownership while the helper runs and is reaped. Park
                    # must not release the GPU while this startup owns a context.
                    snapshot = probe_gpu_memory(
                        self._runtime.accelerator, self._runtime.gpu_index, timeout=deadline - now
                    )
                    if reason := interruption(time.monotonic()):
                        self._set_unready(reason)
                        return
                    with self._state:
                        self._preload_memory = snapshot
                self._in_q = self._ctx.Queue(maxsize=1)
                receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
                self._out_q = ControlChannel(receiver)
                self._reply_sender = ControlChannel(sender)
                self._reply_directory = tempfile.TemporaryDirectory(prefix="vision-bill-replies-")
                self._proc = self._ctx.Process(
                    target=_worker_main,
                    args=(
                        self._in_q,
                        self._reply_sender,
                        self._generation,
                        self._profile_path,
                        self._runtime,
                        Path(self._reply_directory.name),
                    ),
                    daemon=True,
                )
                self._proc.start()
                while True:
                    now = time.monotonic()
                    if reason := interruption(now):
                        self._set_unready(reason)
                        break
                    remaining = deadline - now
                    child_exited = False
                    try:
                        message = self._out_q.get(timeout=min(0.05, remaining))
                    except queue.Empty:
                        if self._proc.is_alive():
                            continue
                        # The child may have posted its final datagram between
                        # the timed receive and the liveness observation.
                        child_exited = True
                        message = self._out_q.get(timeout=0)
                    if message.get("generation") != self._generation:
                        continue
                    if message.get("kind") == "ready" and not child_exited:
                        with self._state:
                            self._ready = not self._stopped.is_set()
                            self._reason = None if self._ready else "ocr_stopped"
                            self._profile_id = message["profile_id"]
                            self._runtime_fingerprint = message.get("runtime_fingerprint")
                            self.last_identity = {
                                "profile_id": self._profile_id,
                                "runtime_fingerprint": self._runtime_fingerprint,
                            }
                        if self._ready:
                            return
                    code = "ocr_init_failed"
                    if message.get("kind") == "init_failed" and message.get("code") in (
                        GPU_OOM_CODE,
                        "ocr_model_load_failed",
                        "ocr_runtime_identity_mismatch",
                    ):
                        code = message["code"]
                    self._set_unready(code, keep_preload=True)
                    break
            except Exception:  # noqa: BLE001 - native boundary must hide exception details
                self._set_unready("ocr_init_failed")
            self._teardown()

    def _teardown(self) -> None:
        """Never release a process's input ownership until death is confirmed."""
        with self._state:
            self._ready = False
            self._profile_id = None
            self._runtime_fingerprint = None
        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=0.1)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1)
                if proc.is_alive():
                    raise OSError("Worker death could not be confirmed")
            except (OSError, ValueError):
                self._set_unready("ocr_worker_unstoppable")
                raise OcrWorkerError("ocr_worker_unstoppable", 503, cleanup_safe=False) from None
            proc.close()
        for channel in (self._in_q, self._out_q, self._reply_sender):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self._proc = self._in_q = self._out_q = None
        self._reply_sender = None
        if self._reply_directory is not None:
            self._reply_directory.cleanup()
            self._reply_directory = None

    def _recover(self, reason: str, *, restart: bool = True) -> None:
        with self._control:
            generation = self._generation
            if not restart:
                # A readiness poll may already have queued recovery for this child.
                self._recovery_attempted_generation = generation
            self._set_unready(reason, keep_preload=reason == GPU_OOM_CODE)
            self._teardown()
        if not restart:
            return
        # A replacement loads the runtime without replaying the failed request.
        threading.Thread(
            target=self._restart_generation,
            args=(generation,),
            name="ocr-worker-recovery",
            daemon=True,
        ).start()

    def claim(self, owner: str) -> None:
        """Fence old HTTP requests only after their native child is reaped."""
        with self._active, self._control:
            generation = int(owner.split(":", 1)[0])
            if generation <= self._owner_generation:
                raise OcrWorkerError("ocr_stale_owner", 409)
            self._parked = True
            self._teardown()
            self._set_unready("ocr_parked")
            self._owner = owner
            self._owner_generation = generation
            self._owner_load_allowed = True

    def load(
        self,
        owner: str,
        *,
        deadline_monotonic: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        with self._control:
            if self._owner != owner:
                raise OcrWorkerError("ocr_stale_owner", 409)
            if not self._owner_load_allowed:
                raise OcrWorkerError("ocr_owner_phase_closed", 409)
            self.start(deadline_monotonic=deadline_monotonic, cancel_event=cancel_event)
            ready = self.readiness()
            if not ready.ready:
                raise OcrWorkerError(
                    ready.reason_code or "ocr_not_ready",
                    504 if ready.reason_code == "ocr_deadline_exceeded" else 503,
                )

    def park(self, owner: str) -> None:
        with self._active, self._control:
            if self._owner != owner:
                raise OcrWorkerError("ocr_stale_owner", 409)
            # Revoke before teardown, including a failed reap. A delayed load
            # HTTP request must never resurrect this phase after park returns.
            self._owner_load_allowed = False
            self._parked = True
            self._teardown()
            self._set_unready("ocr_parked")

    @staticmethod
    def _check_request(deadline_monotonic: float, cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise OcrWorkerError("ocr_cancelled", 499)
        if time.monotonic() >= deadline_monotonic:
            raise OcrWorkerError("ocr_deadline_exceeded", 504)

    def infer_standalone(
        self,
        request: OcrRequest,
        *,
        deadline_monotonic: float,
        cancel_event: threading.Event | None = None,
    ) -> OcrResponse:
        """Own a cold request through positively confirmed native teardown."""
        self._check_request(deadline_monotonic, cancel_event)
        with self._state:
            if self._standalone_admitted >= 5:
                raise OcrWorkerError("ocr_queue_full", 429)
            self._standalone_admitted += 1
        acquired = False
        try:
            # This lock covers claim/load/infer/park. infer and park acquire
            # _active themselves; nesting _active here would deadlock.
            while not acquired:
                self._check_request(deadline_monotonic, cancel_event)
                acquired = self._admission.acquire(
                    timeout=min(0.05, max(0, deadline_monotonic - time.monotonic()))
                )
            self._check_request(deadline_monotonic, cancel_event)
            with self._control, self._state:
                # Polls may mark a crash unready, but only this request owns
                # teardown. Block both fresh and previously queued recovery.
                self._standalone_session = True
                owner = f"{self._owner_generation + 1}:{uuid4()}"
            self.claim(owner)
            try:
                self._check_request(deadline_monotonic, cancel_event)
                self.load(owner, deadline_monotonic=deadline_monotonic, cancel_event=cancel_event)
                result = self.infer(
                    request,
                    deadline_monotonic=deadline_monotonic,
                    owner=owner,
                    cancel_event=cancel_event,
                    recover=False,
                )
                self._check_request(deadline_monotonic, cancel_event)
            finally:
                # A failed park must preserve request artifacts, even if an
                # earlier inference error appeared safe to clean up.
                try:
                    self.park(owner)
                except Exception:  # noqa: BLE001 - unconfirmed death forbids artifact cleanup
                    raise OcrWorkerError(
                        "ocr_worker_unstoppable", 503, cleanup_safe=False
                    ) from None
            self._check_request(deadline_monotonic, cancel_event)
            return result
        finally:
            if acquired:
                with self._control, self._state:
                    self._standalone_session = False
                self._admission.release()
            with self._state:
                self._standalone_admitted -= 1

    def infer(
        self,
        request: OcrRequest,
        *,
        deadline_monotonic: float,
        owner: str | None = None,
        cancel_event: threading.Event | None = None,
        recover: bool = True,
    ) -> OcrResponse:
        if time.monotonic() >= deadline_monotonic:
            raise OcrWorkerError("ocr_deadline_exceeded", 504)
        with self._state:
            if self._admitted >= 5:
                raise OcrWorkerError("ocr_queue_full", 429)
            self._admitted += 1
        acquired = False
        try:
            acquired = self._active.acquire(timeout=max(0, deadline_monotonic - time.monotonic()))
            if not acquired or time.monotonic() >= deadline_monotonic:
                # Waiting callers do not own the worker and cannot terminate its owner.
                raise OcrWorkerError("ocr_deadline_exceeded", 504)
            ready = self.readiness()
            if not ready.ready:
                raise OcrWorkerError(ready.reason_code or "ocr_not_ready", 503)
            with self._control:
                # The earlier readiness observation is advisory. This check and
                # dispatch share ownership with stop's channel/process retirement.
                if self._owner != owner:
                    raise OcrWorkerError("ocr_stale_owner", 409)
                if self._stopped.is_set():
                    raise OcrWorkerError("ocr_stopped", 503)
                ready = self.readiness()
                if not ready.ready:
                    raise OcrWorkerError(ready.reason_code or "ocr_not_ready", 503)
                generation, proc, in_q, out_q = (
                    self._generation,
                    self._proc,
                    self._in_q,
                    self._out_q,
                )
                assert self._reply_directory is not None
                reply_directory = Path(self._reply_directory.name)
                try:
                    in_q.put(request, block=False)
                except (OSError, ValueError, queue.Full):
                    self._recover("ocr_worker_crashed", restart=recover)
                    raise OcrWorkerError("ocr_worker_crashed", 503) from None
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._recover("ocr_cancelled", restart=recover)
                    raise OcrWorkerError("ocr_cancelled", 499)
                if self._stopped.is_set():
                    with self._control:
                        self._teardown()
                    raise OcrWorkerError("ocr_stopped", 503)
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    self._recover("ocr_deadline_exceeded", restart=recover)
                    raise OcrWorkerError("ocr_deadline_exceeded", 504)
                try:
                    try:
                        message = out_q.get(timeout=min(0.05, remaining))
                    except queue.Empty:
                        with self._control:
                            if self._stopped.is_set() or proc is not self._proc:
                                raise
                            if proc.is_alive():
                                continue
                            message = out_q.get(timeout=0)
                except (queue.Empty, OSError, ValueError, TypeError, RecursionError):
                    if self._stopped.is_set():
                        with self._control:
                            self._teardown()
                        raise OcrWorkerError("ocr_stopped", 503) from None
                    self._recover("ocr_worker_crashed", restart=recover)
                    raise OcrWorkerError("ocr_worker_crashed", 503) from None
                if message.get("generation") != generation or message.get("request_id") != str(
                    request.request_id
                ):
                    continue
                if message.get("kind") == "worker_failed":
                    code = message.get("code")
                    if code == GPU_OOM_CODE:
                        self._recover(GPU_OOM_CODE, restart=False)
                        raise OcrWorkerError(GPU_OOM_CODE, 503)
                    if code not in ("ocr_inference_failed", "ocr_invalid_result"):
                        code = "ocr_invalid_result"
                    self._recover(code, restart=recover)
                    raise OcrWorkerError(code, 502)
                if message.get("kind") != "result":
                    raise OcrWorkerError("ocr_inference_failed", 502)
                try:
                    result = read_result(reply_directory, message["artifact"])
                    if (
                        result.request_id != request.request_id
                        or result.photo_index != request.photo_index
                        or result.source_sha256 != request.source_sha256
                    ):
                        raise ValueError("Response identity mismatch")
                except (OSError, ValueError, TypeError, KeyError):
                    if self._stopped.is_set():
                        with self._control:
                            self._teardown()
                        raise OcrWorkerError("ocr_stopped", 503) from None
                    self._recover("ocr_invalid_result", restart=recover)
                    raise OcrWorkerError("ocr_invalid_result", 502) from None
                if time.monotonic() >= deadline_monotonic:
                    self._recover("ocr_deadline_exceeded", restart=recover)
                    raise OcrWorkerError("ocr_deadline_exceeded", 504)
                return result
        finally:
            if acquired:
                self._active.release()
            with self._state:
                self._admitted -= 1

    def stop(self) -> None:
        self._stopped.set()
        with self._control:
            self._set_unready("ocr_stopped")
            self._teardown()
