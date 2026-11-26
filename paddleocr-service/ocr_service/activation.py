"""Idle native activation controller for an app-paired Compose project."""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import FrameType
from typing import Self
from uuid import UUID

from ocr_contract.activation import (
    ActivationBusy,
    ActivationPhase,
    ActivationRequest,
    ActivationStateError,
    ActivationStatus,
    ActivationStore,
)

from .initialize import (
    PreparationFailed,
    PreparationRequired,
    check_prepared_installation,
    prepare_installation,
)
from .logging_config import configure_native_logging
from .supervision import (
    InstallationBusy,
    ManagedInterrupted,
    UncleanInstallation,
    check_fences,
    lock_path,
    managed_installation,
)

logger = logging.getLogger("ocr_service.activation")
ACTIVATION_STORE = ActivationStore()
PARK_TIMEOUT_SECONDS = 30
POLL_SECONDS = 1


class ServingExited(RuntimeError):
    """A parked serving process exited and Compose may restart the controller."""


@dataclass(frozen=True)
class ReleaseInputs:
    bundle_path: Path
    state_dir: Path
    image_config: Path
    image_id: str
    handoff_lock: Path


class _IdleHealthHandler(BaseHTTPRequestHandler):
    timeout = 1

    def do_GET(self) -> None:
        if self.path == "/health":
            status, body = 200, b'{"status":"ok"}'
        elif self.path == "/ready":
            status, body = 503, b'{"ready":false,"reason_code":"ocr_activation_idle"}'
        else:
            self.send_error(404)
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class IdleHealth:
    """Keep liveness available while model files are being prepared."""

    def __init__(self, port: int):
        self.port = port

    def __enter__(self) -> Self:
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), _IdleHealthHandler)
        self.port = self._server.server_port
        self._thread = Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="ocr-idle-health",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    @contextmanager
    def paused(self) -> Iterator[None]:
        self.__exit__()
        try:
            yield
        finally:
            self.__enter__()


def start_server(inputs: ReleaseInputs, installation: Path, port: int) -> subprocess.Popen[bytes]:
    # Match the installation we just verified after acquiring the serving lock.
    # A separate maintenance owner might have changed it between lock scopes.
    if (inputs.state_dir / "selected").resolve(strict=True) != installation:
        raise ValueError("Native installation changed before serving")
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "server.py")],
        env={
            **os.environ,
            "OCR_PORT": str(port),
            "OCR_SERVICE_MODE": "managed",
            "OCR_RUNTIME_SETTINGS_PATH": str(installation / "runtime.json"),
            "OCR_QUALIFICATION_IMAGE_PATH": "",
        },
        start_new_session=True,
    )


def wait_for_park(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + PARK_TIMEOUT_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        connection = HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/ready")
            response = connection.getresponse()
            if response.status == 503:
                payload = json.loads(response.read(4096))
                if (
                    isinstance(payload, dict)
                    and payload.get("ready") is False
                    and payload.get("reason_code") == "ocr_parked"
                    and process.poll() is None
                ):
                    return
        except (HTTPException, OSError, ValueError):
            pass
        finally:
            connection.close()
        time.sleep(0.1)
    raise RuntimeError("Native managed service did not reach parked readiness")


def process_request(
    store: ActivationStore, request: ActivationRequest, inputs: ReleaseInputs, health: IdleHealth
) -> bool:
    """Settle one current request once admission is available. Keep serving until exit."""
    with ExitStack() as admission:
        try:
            admission.enter_context(store.admission(exclusive=True))
        except ActivationBusy:
            status = store.read_status(request)
            if status is None or status.phase in (
                ActivationPhase.PREPARING,
                ActivationPhase.WAITING_FOR_ADMISSION,
            ):
                store.publish_status(
                    ActivationStatus(request.generation, ActivationPhase.WAITING_FOR_ADMISSION)
                )
            return False

        if store.read_request() != request:
            return False
        status = store.read_status(request)
        if status is not None and status.phase is ActivationPhase.FAILED:
            return False
        restarting = status is not None and status.phase is ActivationPhase.READY

        reason = "native_activation_inputs_invalid"
        try:
            args = (inputs.bundle_path, inputs.state_dir, inputs.image_config, inputs.image_id)
            if not restarting and not store.publish_status(
                ActivationStatus(request.generation, ActivationPhase.PREPARING)
            ):
                return False
            installation = None
            if restarting:
                reason = "native_activation_recovery_required"
                try:
                    installation = check_prepared_installation(*args)
                except PreparationRequired:
                    pass  # Migrate a previously requested installation under maintenance exclusion.
            if installation is None:
                logger.info("ocr_activation generation=%s state=preparing", request.generation)
                installation = prepare_installation(
                    *args,
                    maintenance_intent=request.maintenance_intent,
                    handoff_lock=inputs.handoff_lock,
                    repair=not restarting,
                )

            reason = "native_serving_startup_failed"
            # Keep the existing PID1 marker/descendant cleanup around the child.
            # The initializer has already released its exclusive installation lock.
            with health.paused(), managed_installation("ocr", lock_path()):
                process = start_server(inputs, installation, health.port)
                wait_for_park(process, health.port)
                if not restarting and not store.publish_status(
                    ActivationStatus(request.generation, ActivationPhase.READY)
                ):
                    raise ActivationStateError("Activation request changed before parked readiness")
                logger.info("ocr_activation generation=%s state=parked", request.generation)
                admission.close()
                process.wait()
            raise ServingExited("Native serving process exited")
        except (ActivationStateError, ServingExited):
            raise
        except UncleanInstallation:
            reason = "native_activation_recovery_required"
        except InstallationBusy:
            if restarting:
                return False
            reason = "native_installation_busy"
        except PreparationFailed as exc:
            reason = exc.reason_code
        except Exception as exc:  # noqa: BLE001 - fail the generation with a stable reason
            logger.debug("Native activation failed with %s", type(exc).__name__)

        # A failed managed cleanup deliberately retains its original fence.
        try:
            check_fences(lock_path(), "initializer")
        except UncleanInstallation:
            reason = "native_activation_recovery_required"
        if restarting:
            logger.warning("native_restoration_failed reason=%s", reason)
            reason = "native_activation_recovery_required"
        store.publish_status(
            ActivationStatus(request.generation, ActivationPhase.FAILED, reason),
            restoration_failed=restarting,
        )
        logger.warning("%s", reason)
        return True


def run_controller(store: ActivationStore, inputs: ReleaseInputs, port: int) -> None:
    attempted: UUID | None = None
    storage_unavailable = False
    with store.controller(), IdleHealth(port) as health:
        while True:
            try:
                request = store.read_request()
                if request is not None and request.generation != attempted:
                    status = store.read_status(request)
                    failed = status is not None and status.phase is ActivationPhase.FAILED
                    if failed or process_request(store, request, inputs, health):
                        attempted = request.generation
                storage_unavailable = False
            except ActivationStateError:
                if not storage_unavailable:
                    logger.warning("%s", "native_activation_state_unavailable")
                storage_unavailable = True
            time.sleep(POLL_SECONDS)


def main(store: ActivationStore = ACTIVATION_STORE) -> None:
    configure_native_logging()

    def stop(signum: int, _frame: FrameType | None) -> None:
        raise ManagedInterrupted(signum)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(number, stop)
    try:
        run_controller(
            store,
            ReleaseInputs(
                bundle_path=Path("/release/bundle.json"),
                state_dir=Path(os.getenv("OCR_STATE_DIR", "/var/lib/ocr")),
                image_config=Path("/release/image-config.json"),
                image_id=os.getenv("OCR_IMAGE_ID", ""),
                handoff_lock=Path(os.getenv("OCR_HANDOFF_LOCK_PATH", "/coord/gpu-handoff.lock")),
            ),
            port=int(os.getenv("OCR_PORT", "8082")),
        )
    except ManagedInterrupted as exc:
        raise SystemExit(128 + exc.signum) from None
    except ServingExited:
        raise SystemExit("native_serving_exited") from None
    except (ActivationBusy, ActivationStateError):
        raise SystemExit("native_activation_controller_unavailable") from None


if __name__ == "__main__":
    main()
