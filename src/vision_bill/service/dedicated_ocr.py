"""Admin opt-in for the project's idle, device-specific OCR service."""
import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from uuid import UUID

import httpx
from ocr_contract.activation import (
    ActivationBusy,
    ActivationConflict,
    ActivationPhase,
    ActivationStateError,
    ActivationStore,
    MaintenanceIntent,
)

from ..config import OcrSettings, Settings, runtime_restart_required
from ..distribution.handoff import adopt_binding
from ..model.dedicated_ocr import DedicatedOcrStatus
from ..model.handoff import HandoffStatus, HandoffWait
from ..model.ocr import WorkerReadiness
from ..provider.llm.base import LLMProvider
from ..provider.ocr.paddleocr import PaddleOcrProvider
from .activation_admission import (
    activation_handoff_path,
    failed_setup_can_disable,
    unsettled_direct_calls,
)
from .gpu_handoff import ModelHandoff
from .native_build import native_build_failed, native_build_status, resolve_failed_build


class DedicatedOcrActivation:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        apply_route: Callable[[PaddleOcrProvider | None, OcrSettings], None],
        wake_receipts: Callable[[], None],
        *,
        store: ActivationStore | None = None,
        handoff_path: Path | None = None,
    ):
        self.settings = settings
        self.provider = provider
        self.apply_route = apply_route
        self.wake_receipts = wake_receipts
        self.store = store if store is not None else ActivationStore()
        self.external_mode = settings.ocr.enabled or (
            not settings.ocr.service_url and not os.environ.get("OCR_ACTIVATION_ENDPOINT", "").strip()
        )
        self.prior_handoff = settings.ocr.coordination_path
        self.ocr_settings = settings.ocr.model_copy(update={
            "enabled": True,
            "allow_vision_fallback": False,
            "manage_model_lifecycle": True,
            "service_url": os.environ.get("OCR_ACTIVATION_ENDPOINT", settings.ocr.service_url),
            "coordination_path": handoff_path if handoff_path is not None else activation_handoff_path(),
        })
        self.ocr = PaddleOcrProvider(
            self.ocr_settings.service_url, self.ocr_settings.timeout_seconds
        )
        self.handoff = ModelHandoff(self.ocr_settings.coordination_path, provider, self.ocr)
        self.ocr_only_handoff = ModelHandoff(
            self.ocr_settings.coordination_path, provider, self.ocr, shared_gpu=False,
        )
        self._applied: UUID | None = None
        self._enabled: bool | None = None
        self._shared_gpu: bool | None = None
        self._blocker: str | None = None

    async def _controller_is_live(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                response = await client.get(f"{self.ocr.service_url}/health")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def _setup_unavailable(self, *, native_started: bool = False) -> str | None:
        build = native_build_status()
        if not native_started and build is not None and build.phase in ("waiting", "failed"):
            return None
        if await self._controller_is_live():
            return None
        if os.environ.get("VISION_BILL_OCR_NATIVE_BUILD_STATUS"):
            return "native_controller_unavailable" if native_started else "native_build_host_unavailable"
        return "gpu_profile_required"

    async def _runtime_is_serving(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                response = await client.get(f"{self.ocr.service_url}/info")
                response.raise_for_status()
                info = response.json()
                if not isinstance(info, dict) or info.get("service_mode") != "managed" or info.get("contract_version") != 2:
                    return False
                worker = WorkerReadiness.model_validate(info.get("worker"), strict=True)
                if worker.ready:
                    return worker.reason_code is None and bool(worker.profile_id and worker.runtime_fingerprint)
                return worker.reason_code == "ocr_parked"
        except (httpx.HTTPError, ValueError):
            return False

    def _lifecycle_blocker(self) -> str | None:
        if not self.provider.supports_model_release or runtime_restart_required(self.settings):
            return "llm_lifecycle_required"
        return None

    def _with_selection(self, result: DedicatedOcrStatus) -> DedicatedOcrStatus:
        if self.external_mode:
            return result
        try:
            request = self.store.read_request()
            current = self.store.read_status(request) if request else None
            selected = bool(
                request and current and current.phase == ActivationPhase.READY
                and self.store.runtime_enabled(request)
            )
            return result.model_copy(update={"selected": selected, "shared_gpu": self.store.shared_gpu()})
        except (ActivationStateError, OSError):
            return result

    async def status(self) -> DedicatedOcrStatus:
        return self._with_selection(await self._status())

    async def _status(self) -> DedicatedOcrStatus:
        if self.external_mode:
            return DedicatedOcrStatus(state="unavailable", reason="ocr_config_managed_externally")
        try:
            request = self.store.read_request()
            current = self.store.read_status(request) if request else None
            enabled = self.store.runtime_enabled(request) if request else True
            shared_gpu = self.store.shared_gpu()
        except ActivationStateError:
            return DedicatedOcrStatus(state="action_required", reason="ocr_activation_state_invalid")
        if current and current.phase == ActivationPhase.READY:
            applied = (self._applied, self._enabled, self._shared_gpu) == (current.generation, enabled, shared_gpu)
            if not enabled and applied:
                return DedicatedOcrStatus(state="disabled", installed=True)
            if shared_gpu:
                if blocker := self._lifecycle_blocker():
                    return DedicatedOcrStatus(state="action_required", reason=blocker, installed=True)
                try:
                    if self.settings.llm.model_name not in self.handoff.read_binding().model_ids:
                        return DedicatedOcrStatus(
                            state="action_required", reason="model_handoff_model_not_adopted", installed=True,
                        )
                except (HandoffWait, OSError, ValueError):
                    return DedicatedOcrStatus(state="action_required", reason="ocr_activation_state_invalid", installed=True)
            if applied:
                if await self._runtime_is_serving():
                    return DedicatedOcrStatus(state="active", installed=True)
                return DedicatedOcrStatus(state="action_required", reason="ocr_runtime_unavailable", installed=True)
            if self._blocker:
                return DedicatedOcrStatus(state="action_required", reason=self._blocker, installed=True)
            return DedicatedOcrStatus(state="installing", installed=True)
        if current and current.phase == ActivationPhase.FAILED:
            if not enabled and failed_setup_can_disable(current):
                return DedicatedOcrStatus(state="disabled")
            return DedicatedOcrStatus(state="failed", reason=current.reason_code or "ocr_setup_failed")
        if request:
            if current is None and os.environ.get("VISION_BILL_OCR_NATIVE_BUILD_STATUS"):
                build = native_build_status()
                if build is not None and build.generation == request.generation:
                    if build.phase == "failed":
                        return DedicatedOcrStatus(state="failed", reason="native_image_build_failed")
                    return DedicatedOcrStatus(state="installing", reason="native_image_building")
                if build is None and not await self._controller_is_live():
                    return DedicatedOcrStatus(state="unavailable", reason="native_build_host_unavailable")
            return DedicatedOcrStatus(state="installing" if current else "requested")
        if unavailable := await self._setup_unavailable():
            return DedicatedOcrStatus(state="unavailable", reason=unavailable)
        if blocker := self._blocker or (self._lifecycle_blocker() if shared_gpu else None):
            return DedicatedOcrStatus(state="action_required", reason=blocker)
        return DedicatedOcrStatus(state="disabled")

    async def set_gpu_topology(self, shared_gpu: bool, *, confirm_handoff: bool) -> DedicatedOcrStatus:
        if self.external_mode:
            raise _topology_wait("ocr_config_managed_externally")
        try:
            async with asyncio.timeout(6):
                with self.store.admission(exclusive=True):
                    if shared_gpu != self.store.shared_gpu():
                        await self._change_gpu_topology(shared_gpu, confirm_handoff, monotonic() + 5)
        except (ActivationBusy, ActivationConflict, TimeoutError):
            raise _topology_wait("llm_quiescence_required") from None
        except (ActivationStateError, OSError, ValueError):
            raise _topology_wait("ocr_activation_state_invalid") from None
        except httpx.HTTPError:
            raise _topology_wait("ocr_runtime_unavailable") from None
        except HandoffWait:
            raise
        except RuntimeError as exc:
            if str(exc).partition(":")[0] == "model_release_unconfirmed":
                raise _topology_wait("model_release_unconfirmed") from None
            raise _topology_wait("ocr_runtime_unavailable") from None
        return await self.status()

    async def _change_gpu_topology(self, shared_gpu: bool, confirm_handoff: bool, deadline: float) -> None:
        request = self.store.read_request()
        current = self.store.read_status(request) if request else None
        if (request is None or (current is not None and current.phase == ActivationPhase.FAILED)
                or (current is None and native_build_failed(request))):
            self.store.set_shared_gpu(shared_gpu)
            self._blocker = None
            return
        if current is None or current.phase != ActivationPhase.READY:
            raise _topology_wait("ocr_setup_pending")
        if unsettled_direct_calls(self.store):
            raise _topology_wait("llm_request_recovery_required")
        if not await self._runtime_is_serving():
            raise _topology_wait("ocr_runtime_unavailable")
        enabled = self.store.runtime_enabled(request)
        if shared_gpu:
            if not confirm_handoff:
                raise _topology_wait("ocr_handoff_confirmation_required")
            if blocker := self._lifecycle_blocker():
                raise _topology_wait(blocker)
            async with self.ocr_only_handoff.quiesce(deadline):
                pass
            model = self.settings.llm.model_name
            models = self.handoff.read_binding().model_ids if self.handoff.binding_path.exists() else []
            if model not in models:
                await adopt_binding(
                    self.handoff, [*models, model], dedicated_coordinated_requests=True,
                    external_eager_loading_disabled=True, deadline=deadline,
                )
            async with self.handoff.quiesce(deadline, model_id=model):
                pass
        else:
            try:
                # A clean shared journal needs only native confirmation. A remote
                # endpoint may already be serving unrelated resident models.
                async with self.ocr_only_handoff.quiesce(deadline):
                    pass
            except HandoffWait as exc:
                if exc.status.reason_code != "model_handoff_journal_not_clean":
                    raise
                if not self.provider.supports_model_release:
                    raise _topology_wait("llm_lifecycle_required") from None
                async with self.handoff.quiesce(deadline):
                    pass
        self.store.set_shared_gpu(shared_gpu)
        self._publish_route(request.generation, enabled, shared_gpu)

    def _publish_route(self, generation: UUID, enabled: bool, shared_gpu: bool) -> None:
        selected = self.ocr_settings.model_copy(update={
            "enabled": enabled, "required": False, "shared_gpu": shared_gpu,
        })
        # Keep the shared coordinator on direct vision calls while OCR is OFF.
        self.provider.handoff = self.handoff if shared_gpu else None
        self.settings.ocr = selected
        self.apply_route(self.ocr if enabled else None, selected)
        self._applied = generation
        self._enabled = enabled
        self._shared_gpu = shared_gpu
        self._blocker = None
        self.wake_receipts()

    async def _toggle(self, enabled: bool) -> DedicatedOcrStatus:
        try:
            async with asyncio.timeout(6):
                with self.store.admission(exclusive=True):
                    request = self.store.read_request()
                    current = self.store.read_status(request) if request else None
                    if request is None or current is None or current.phase != ActivationPhase.READY:
                        return DedicatedOcrStatus(state="action_required", reason="ocr_setup_pending")
                    if unsettled_direct_calls(self.store):
                        return DedicatedOcrStatus(state="action_required", reason="llm_request_recovery_required", installed=True)
                    shared_gpu = self.store.shared_gpu()
                    if shared_gpu and enabled and (blocker := self._lifecycle_blocker()):
                        return DedicatedOcrStatus(state="action_required", reason=blocker, installed=True)
                    if shared_gpu and not self.provider.supports_model_release:
                        return DedicatedOcrStatus(
                            state="action_required", reason="llm_lifecycle_required", installed=True,
                        )
                    if enabled and not await self._runtime_is_serving():
                        return DedicatedOcrStatus(state="action_required", reason="ocr_runtime_unavailable", installed=True)
                    # Session refuses unsettled requests, recovers settled journals,
                    # parks the real native worker and releases owned LLM models.
                    handoff = self.handoff if shared_gpu else self.ocr_only_handoff
                    async with handoff.session(
                        monotonic() + 5, model_id=self.settings.llm.model_name if enabled else None,
                    ):
                        pass
                    self.store.set_runtime_enabled(request, enabled)
                    self._publish_route(request.generation, enabled, shared_gpu)
                    return DedicatedOcrStatus(state="active" if enabled else "disabled", installed=True)
        except HandoffWait as exc:
            return DedicatedOcrStatus(state="action_required", reason=exc.status.reason_code, installed=True)
        except (ActivationBusy, ActivationConflict, httpx.HTTPError, TimeoutError):
            return DedicatedOcrStatus(state="action_required", reason="llm_quiescence_required", installed=True)
        except (ActivationStateError, OSError, ValueError):
            return DedicatedOcrStatus(state="action_required", reason="ocr_activation_state_invalid")

    async def submit(self, *, confirm_handoff: bool, enabled: bool = True) -> DedicatedOcrStatus:
        try:
            result = await self._submit(confirm_handoff=confirm_handoff, enabled=enabled)
        except RuntimeError as exc:
            if str(exc).partition(":")[0] != "model_release_unconfirmed":
                raise
            result = DedicatedOcrStatus(
                state="action_required", reason="model_release_unconfirmed", installed=True,
            )
        return self._with_selection(result)

    async def _submit(self, *, confirm_handoff: bool, enabled: bool) -> DedicatedOcrStatus:
        if self.external_mode:
            return await self.status()
        try:
            request = self.store.read_request()
            activation_status = self.store.read_status(request) if request else None
            if activation_status and activation_status.phase == ActivationPhase.READY:
                return await self._toggle(enabled)
        except (ActivationStateError, OSError):
            return DedicatedOcrStatus(state="action_required", reason="ocr_activation_state_invalid")
        try:
            async with asyncio.timeout(6):
                with self.store.admission(exclusive=True):
                    current = self.store.read_request()
                    phase = self.store.read_status(current) if current else None
                    failed_build = current is not None and phase is None and native_build_failed(current)
                    if not enabled:
                        if current is None:
                            return DedicatedOcrStatus(state="disabled")
                        if not failed_build and (phase is None or phase.phase != ActivationPhase.FAILED):
                            return DedicatedOcrStatus(state="action_required", reason="ocr_setup_pending")
                        if not (failed_build or failed_setup_can_disable(phase)) or self.settings.ocr.enabled:
                            return DedicatedOcrStatus(state="action_required", reason="native_activation_recovery_required")
                        if not self.store.runtime_enabled(current):
                            return DedicatedOcrStatus(state="disabled")
                        if unsettled_direct_calls(self.store):
                            return DedicatedOcrStatus(state="action_required", reason="llm_request_recovery_required")
                        # Both journals must be clean, even after a topology change. The
                        # initial native setup ended before creating a serving controller.
                        try:
                            await self.ocr_only_handoff.require_clean_for_maintenance(monotonic() + 2)
                        except HandoffWait as exc:
                            return DedicatedOcrStatus(state="action_required", reason=exc.status.reason_code)
                        if failed_build:
                            resolve_failed_build(self.store, current)
                        else:
                            self.store.set_runtime_enabled(current, False)
                        self.wake_receipts()
                        return DedicatedOcrStatus(state="disabled")
                    if current and not failed_build and (phase is None or phase.phase != ActivationPhase.FAILED):
                        return await self.status()
                    shared_gpu = self.store.shared_gpu()
                    if shared_gpu:
                        if not confirm_handoff:
                            self._blocker = "ocr_handoff_confirmation_required"
                            return DedicatedOcrStatus(state="action_required", reason=self._blocker)
                        if blocker := self._lifecycle_blocker():
                            self._blocker = blocker
                            return DedicatedOcrStatus(state="action_required", reason=blocker)
                    if unavailable := await self._setup_unavailable(native_started=phase is not None):
                        return DedicatedOcrStatus(state="action_required", reason=unavailable)
                    if unsettled_direct_calls(self.store):
                        self._blocker = "llm_request_recovery_required"
                        return DedicatedOcrStatus(state="action_required", reason=self._blocker)
                    if self.prior_handoff != self.handoff.path and (
                        self.prior_handoff.with_suffix(".json").exists()
                        or self.prior_handoff.with_suffix(".binding.json").exists()
                        or self.prior_handoff.with_suffix(".ocr-only.json").exists()
                    ):
                        self._blocker = "ocr_existing_handoff"
                        return DedicatedOcrStatus(state="action_required", reason=self._blocker)
                    intent = MaintenanceIntent.SEPARATE_OCR_DEVICE
                    deadline = monotonic() + 3
                    handoff = self.handoff if shared_gpu else self.ocr_only_handoff
                    try:
                        await handoff.require_clean_for_maintenance(deadline)
                    except HandoffWait as exc:
                        self._blocker = "llm_quiescence_required" if shared_gpu else exc.status.reason_code
                        return DedicatedOcrStatus(state="action_required", reason=self._blocker)
                    if shared_gpu:
                        intent = MaintenanceIntent(os.environ.get(
                            "OCR_ACTIVATION_MAINTENANCE_INTENT",
                            MaintenanceIntent.EXTERNAL_LLM_QUIESCED.value,
                        ))
                        await adopt_binding(
                            self.handoff, [self.settings.llm.model_name],
                            dedicated_coordinated_requests=True,
                            external_eager_loading_disabled=True,
                            deadline=deadline,
                        )
                    if failed_build and current is not None:
                        resolve_failed_build(self.store, current, retry_intent=intent)
                    else:
                        self.store.submit_request(
                            intent, retry_generation=current.generation if current else None,
                        )
                    self._blocker = None
                    return DedicatedOcrStatus(state="requested")
        except (ActivationBusy, ActivationConflict):
            self._blocker = "llm_quiescence_required"
        except (ActivationStateError, OSError, ValueError):
            self._blocker = "ocr_activation_state_invalid"
        except (HandoffWait, httpx.HTTPError, TimeoutError):
            self._blocker = "llm_quiescence_required"
        return DedicatedOcrStatus(state="action_required", reason=self._blocker)

    async def reconcile(self) -> None:
        if self.external_mode:
            return
        try:
            request = self.store.read_request()
            if request is None:
                return
            enabled = self.store.runtime_enabled(request)
            shared_gpu = self.store.shared_gpu()
            if (request.generation, enabled, shared_gpu) == (self._applied, self._enabled, self._shared_gpu):
                return
            current = self.store.read_status(request)
            if current is None or current.phase != ActivationPhase.READY:
                return
            with self.store.admission(exclusive=True):
                request = self.store.read_request()
                current = self.store.read_status(request) if request else None
                if request is None or current is None or current.phase != ActivationPhase.READY:
                    return
                enabled = self.store.runtime_enabled(request)
                shared_gpu = self.store.shared_gpu()
                if shared_gpu:
                    binding = self.handoff.read_binding()
                    if (enabled and self._lifecycle_blocker()) or not self.provider.supports_model_release:
                        self._blocker = "llm_lifecycle_required"
                        return
                    if enabled and self.settings.llm.model_name not in binding.model_ids:
                        self._blocker = "model_handoff_model_not_adopted"
                        return
                # Existing services keep the same provider and receipt store. All
                # route-facing references are replaced together under admission.
                if shared_gpu and unsettled_direct_calls(self.store):
                    self._blocker = "llm_request_recovery_required"
                    return
                self._publish_route(request.generation, enabled, shared_gpu)
        except ActivationBusy:
            return
        except (ActivationStateError, HandoffWait, OSError, ValueError):
            self._blocker = "ocr_activation_state_invalid"

    async def run(self) -> None:
        setup_on_startup = os.environ.get("OCR_ACTIVATION_ON_STARTUP") == "true"
        confirm_handoff = os.environ.get("OCR_ACTIVATION_CONFIRM_HANDOFF") == "true"
        while not self.external_mode:
            if setup_on_startup:
                try:
                    if self.store.read_request() is None:
                        await self.submit(confirm_handoff=confirm_handoff)
                    else:
                        setup_on_startup = False
                except ActivationStateError:
                    self._blocker = "ocr_activation_state_invalid"
            await self.reconcile()
            await asyncio.sleep(1)


def _topology_wait(reason_code: str) -> HandoffWait:
    return HandoffWait(HandoffStatus(state="configuration_required", reason_code=reason_code))
