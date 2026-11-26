"""Coordinate inference with native OCR installation and routing changes."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ocr_contract.activation import (
    ActivationBusy,
    ActivationPhase,
    ActivationRequest,
    ActivationStateError,
    ActivationStatus,
    ActivationStore,
    MaintenanceIntent,
)

from ..model.handoff import HandoffStatus, HandoffWait
from .gpu_handoff import ModelHandoff, current_owner
from .native_build import native_build_status

if TYPE_CHECKING:
    from ..config import OcrSettings
    from ..provider.llm.base import LLMProvider

_MARKERS = "direct-calls"
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class _Admission:
    store: ActivationStore
    shared_gpu: bool | None
    task: asyncio.Task[Any] | None
    thread: int
    direct_only: bool


_current: ContextVar[_Admission | None] = ContextVar("ocr_activation_admission", default=None)


def activation_enabled() -> bool:
    return os.environ.get("VISION_BILL_OCR_ACTIVATION") == "true"


def activation_handoff_path() -> Path:
    return Path(os.environ.get("OCR_ACTIVATION_HANDOFF_LOCK_PATH", "/coord/gpu-handoff.lock"))


def _task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _current_admission() -> _Admission | None:
    admitted = _current.get()
    if admitted is not None and admitted.task is _task() and admitted.thread == get_ident():
        return admitted
    return None


def failed_setup_can_disable(status: ActivationStatus | None) -> bool:
    """Only initial setup failures with confirmed native cleanup permit direct recovery."""
    return (
        status is not None
        and status.phase == ActivationPhase.FAILED
        and status.reason_code
        in {
            "native_model_download_failed",
            "native_model_integrity_failed",
            "native_device_identity_unavailable",
            "native_package_mismatch",
            "native_installation_write_failed",
            "native_activation_inputs_invalid",
            "native_qualification_failed",
            "ocr_gpu_out_of_memory",
            "native_serving_startup_failed",
        }
    )


def _require_current_route(
    store: ActivationStore, request: ActivationRequest, ocr: OcrSettings
) -> bool:
    shared_gpu = store.shared_gpu()
    if store.runtime_enabled(request) != ocr.enabled or shared_gpu != ocr.shared_gpu:
        raise HandoffWait(
            HandoffStatus(state="configuration_required", reason_code="execution_policy_changed")
        )
    return shared_gpu


@contextmanager
def policy_capture(ocr: OcrSettings) -> Iterator[None]:
    """Capture the current routing decision while initial OCR setup may be pending."""
    if not activation_enabled():
        yield
        return
    store = ActivationStore()
    with ExitStack() as lock:
        try:
            locked = True
            try:
                lock.enter_context(store.admission(exclusive=False))
            except ActivationBusy:
                locked = False
            request = store.read_request()
            status = store.read_status(request) if request is not None else None
            if request is not None and status is not None and status.phase == ActivationPhase.READY:
                if not locked:
                    raise HandoffWait(HandoffStatus(state="busy", reason_code="dedicated_ocr_activation_pending"))
                _require_current_route(store, request, ocr)
        except (ActivationStateError, OSError):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="dedicated_ocr_activation_state_invalid",
            )) from None
        yield


def _require_provider(
    provider: LLMProvider | None, shared_gpu: bool | None, *,
    candidate: ModelHandoff | None, uses_ocr: bool, direct_only: bool,
) -> None:
    from ..config import settings

    if direct_only and (
        uses_ocr or candidate is not None or settings.ocr.enabled
        or (provider is not None and provider.handoff is not None)
    ):
        raise HandoffWait(HandoffStatus(state="configuration_required", reason_code="execution_policy_changed"))
    if uses_ocr and not settings.ocr.enabled:
        raise HandoffWait(HandoffStatus(state="configuration_required", reason_code="dedicated_ocr_disabled"))
    if shared_gpu is None or (provider is None and not uses_ocr and candidate is None):
        return
    owner = current_owner() if _task() is not None else None
    attached = provider.handoff if provider is not None else None
    handoff = candidate if candidate is not None else (owner.coordinator if owner is not None else attached)
    endpoint = os.environ.get("OCR_ACTIVATION_ENDPOINT", settings.ocr.service_url).rstrip("/")
    if not shared_gpu and handoff is None and not uses_ocr:
        return
    if (
        isinstance(handoff, ModelHandoff)
        and (not uses_ocr or candidate is not None)
        and handoff.shared_gpu == shared_gpu
        and getattr(handoff.ocr, "managed_lifecycle", False)
        and handoff.path.resolve() == activation_handoff_path().resolve()
        and getattr(handoff.ocr, "service_url", None) == endpoint
        and (owner is None or handoff.matches(owner.coordinator))
        and (
            not shared_gpu and attached is None
            or shared_gpu and provider is not None and provider.supports_model_release
            and handoff.resources.llm == provider.lifecycle_identity
            and (attached is None or isinstance(attached, ModelHandoff) and handoff.matches(attached))
        )
    ):
        return
    raise HandoffWait(HandoffStatus(state="configuration_required", reason_code="execution_policy_changed"))


@contextmanager
def inference_admission(
    provider: LLMProvider | None = None, *, store: ActivationStore | None = None,
    handoff: ModelHandoff | None = None, uses_ocr: bool = False,
) -> Iterator[None]:
    """Hold a shared lock for an attempt, reusing it for that task's nested calls."""
    if not activation_enabled():
        yield
        return
    admitted = _current_admission()
    if admitted is not None and (store is None or store == admitted.store):
        _require_provider(provider, admitted.shared_gpu, candidate=handoff, uses_ocr=uses_ocr, direct_only=admitted.direct_only)
        yield
        return

    selected = store if store is not None else ActivationStore()
    with ExitStack() as lock:
        shared_gpu = None
        try:
            lock.enter_context(selected.admission(exclusive=False, blocking=False))
            request = selected.read_request()
            status = selected.read_status(request) if request is not None else None
            direct_only = request is not None and failed_setup_can_disable(status) and not selected.runtime_enabled(request)
            if (
                request is not None and status is None
                and request.maintenance_intent == MaintenanceIntent.SEPARATE_OCR_DEVICE
                and not selected.shared_gpu()
            ):
                # The native controller also takes exclusive admission. Its first
                # status closes this window while already admitted direct calls drain.
                build = native_build_status()
                direct_only = build is not None and build.generation == request.generation
            if request is not None and status is not None and status.phase == ActivationPhase.READY:
                from ..config import settings
                shared_gpu = _require_current_route(selected, request, settings.ocr)
        except ActivationBusy:
            raise HandoffWait(HandoffStatus(state="busy", reason_code="dedicated_ocr_activation_pending")) from None
        except (ActivationStateError, OSError):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="dedicated_ocr_activation_state_invalid",
            )) from None
        if request is not None and not direct_only and (status is None or status.phase != ActivationPhase.READY):
            raise HandoffWait(HandoffStatus(state="busy", reason_code="dedicated_ocr_activation_pending"))

        _require_provider(provider, shared_gpu, candidate=handoff, uses_ocr=uses_ocr, direct_only=direct_only)
        token = _current.set(_Admission(selected, shared_gpu, _task(), get_ident(), direct_only))
        try:
            yield
        finally:
            _current.reset(token)


@contextmanager
def _marker_directory(store: ActivationStore, *, create: bool) -> Iterator[int]:
    if create:
        store.request_dir.mkdir(mode=0o700, exist_ok=True)
    directory = os.open(store.request_dir, _DIRECTORY_FLAGS)
    try:
        if create:
            try:
                os.mkdir(_MARKERS, 0o700, dir_fd=directory)
            except FileExistsError:
                pass
            os.fsync(directory)
            # The request directory may have been created by another first call.
            parent = os.open(store.request_dir.parent, _DIRECTORY_FLAGS)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        markers = os.open(_MARKERS, _DIRECTORY_FLAGS, dir_fd=directory)
        try:
            info = os.fstat(markers)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ActivationStateError("Unsafe direct inference marker directory")
            yield markers
        finally:
            os.close(markers)
    finally:
        os.close(directory)


def unsettled_direct_calls(store: ActivationStore) -> tuple[Path, ...]:
    """Read markers while holding exclusive admission, including after a client crash."""
    try:
        with _marker_directory(store, create=False) as directory:
            return tuple(store.request_dir / _MARKERS / name for name in sorted(os.listdir(directory)))
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ActivationStateError("Cannot read direct inference markers") from exc


@contextmanager
def direct_inference_call(*, coordinated: bool = False) -> Iterator[None]:
    """Record dispatch before uncoordinated I/O and retain uncertain remote calls."""
    if not activation_enabled():
        yield
        return
    admitted = _current_admission()
    if admitted is None or (admitted.shared_gpu is True and not coordinated):
        raise HandoffWait(HandoffStatus(state="configuration_required", reason_code="execution_policy_changed"))
    if coordinated:
        # The ModelHandoff journal records dispatch and settlement under its lock.
        yield
        return

    with ExitStack() as marker:
        try:
            directory = marker.enter_context(_marker_directory(admitted.store, create=True))
            name = uuid4().hex
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory)
        except (ActivationStateError, OSError):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="dedicated_ocr_activation_state_invalid",
            )) from None
        yield
        os.unlink(name, dir_fd=directory)
        os.fsync(directory)
