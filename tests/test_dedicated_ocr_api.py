"""Dedicated OCR can be requested only by an administrator."""
import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vision_bill.api.system.main import router
from vision_bill.security.dependencies import get_current_user, require_admin


def app_with_system() -> FastAPI:
    app = FastAPI()
    app.state.user_db = SimpleNamespace(is_ready=True)
    app.include_router(router, prefix="/system")
    return app


def test_activation_requires_admin() -> None:
    client = TestClient(app_with_system())
    assert client.get("/system/dedicated-ocr").status_code == 401
    assert client.post("/system/dedicated-ocr", json={}).status_code == 401
    assert client.put("/system/dedicated-ocr/gpu-topology", json={"shared_gpu": False}).status_code == 401


def test_app_without_a_gpu_profile_reports_unavailable() -> None:
    app = app_with_system()
    app.dependency_overrides[require_admin] = lambda: None
    client = TestClient(app)
    response = client.get("/system/dedicated-ocr")
    assert response.status_code == 200
    assert response.json() == {"state": "unavailable", "reason": "gpu_profile_required", "installed": False,
                               "selected": False, "shared_gpu": True}
    response = client.post("/system/dedicated-ocr", json={})
    assert response.status_code == 409
    assert response.json()["detail"] == "gpu_profile_required"


def test_a_regular_user_cannot_request_activation() -> None:
    app = app_with_system()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(is_admin=False)
    assert TestClient(app).post(
        "/system/dedicated-ocr", json={"confirm_handoff": True}
    ).status_code == 403
    assert TestClient(app).put(
        "/system/dedicated-ocr/gpu-topology", json={"shared_gpu": False}
    ).status_code == 403


def test_admin_can_persist_separate_gpu_before_installation(tmp_path, monkeypatch):
    from ocr_contract.activation import ActivationStore

    from vision_bill.config import settings
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    monkeypatch.setenv("OCR_ACTIVATION_ENDPOINT", "http://prepared-ocr.test")
    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    path = tmp_path / "gpu.lock"

    def app_with_activation():
        config = settings.model_copy(deep=True)
        config.ocr.enabled = False
        config.ocr.coordination_path = path
        provider = SimpleNamespace(supports_model_release=False)
        activation = DedicatedOcrActivation(
            config, provider, lambda *_args: None, lambda: None,
            store=store, handoff_path=path,
        )

        async def live():
            return True

        monkeypatch.setattr(activation, "_controller_is_live", live)
        app = app_with_system()
        app.dependency_overrides[require_admin] = lambda: None
        app.state.dedicated_ocr = activation
        return TestClient(app)

    client = app_with_activation()
    assert client.get("/system/dedicated-ocr").json()["shared_gpu"] is True
    changed = client.put("/system/dedicated-ocr/gpu-topology", json={"shared_gpu": False})
    assert changed.status_code == 200
    assert changed.json()["shared_gpu"] is False
    assert store.read_request() is None
    client = app_with_activation()
    assert client.get("/system/dedicated-ocr").json()["shared_gpu"] is False
    legacy = path.with_suffix(".json")
    legacy.write_text(json.dumps({
        "version": 2, "coordinator": "vision-bill-model-handoff-v2",
        "resources": {"llm": "openai:legacy", "ocr": "ocr:legacy"},
        "generation": 11, "unsettled": True, "ocr_pending": False, "models": ["old-model"],
    }))
    blocked = client.post("/system/dedicated-ocr", json={})
    assert blocked.status_code == 409 and blocked.json()["detail"] == "model_request_unsettled"
    assert store.read_request() is None
    legacy.unlink()
    requested = client.post("/system/dedicated-ocr", json={})
    assert requested.status_code == 202
    assert requested.json()["state"] == "requested"
    assert store.read_request().maintenance_intent == "separate-ocr-device"


def test_activation_does_not_accept_browser_supplied_download_sources() -> None:
    app = app_with_system()
    app.dependency_overrides[require_admin] = lambda: None
    response = TestClient(app).post(
        "/system/dedicated-ocr", json={"url": "https://example.invalid/models"}
    )
    assert response.status_code == 422


def test_activation_request_is_idempotent_at_the_app_boundary(tmp_path, monkeypatch) -> None:
    from ocr_contract.activation import ActivationPhase, ActivationStatus, ActivationStore

    from vision_bill.config import settings
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    store = ActivationStore(
        tmp_path / "request", tmp_path / "status", tmp_path / "lock"
    )
    config = settings.model_copy(deep=True)
    config.ocr.enabled = False
    config.ocr.coordination_path = tmp_path / "legacy.lock"
    route = []

    class Provider:
        supports_model_release = True
        lifecycle_identity = "openai:test"
        handoff = None

        async def resident_model_ids(self, **_kwargs) -> frozenset[str]:
            return frozenset()

    activation = DedicatedOcrActivation(
        config, Provider(), lambda ocr, settings: route.append((ocr, settings)),
        lambda: None, store=store, handoff_path=tmp_path / "gpu.lock",
    )

    async def controller_is_live() -> bool:
        return True

    monkeypatch.setattr(activation, "_controller_is_live", controller_is_live)
    monkeypatch.setattr(activation, "_runtime_is_serving", controller_is_live, raising=False)

    async def run() -> None:
        assert (await activation.status()).state == "disabled"
        assert store.read_request() is None
        assert (await activation.submit(confirm_handoff=False)).reason == "ocr_handoff_confirmation_required"
        assert store.read_request() is None
        assert (await activation.submit(confirm_handoff=True)).state == "requested"
        request = store.read_request()
        assert request
        assert (await activation.submit(confirm_handoff=True)).state == "requested"
        assert store.read_request() == request
        assert not config.ocr.enabled
        assert route == []
        store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
        await activation.reconcile()
        assert (await activation.status()).state == "active"
        assert config.ocr.enabled and config.ocr.manage_model_lifecycle
        assert len(route) == 1
        await activation.reconcile()
        assert len(route) == 1

        async def offline() -> bool:
            return False

        monkeypatch.setattr(activation, "_runtime_is_serving", offline)
        assert (await activation.status()).reason == "ocr_runtime_unavailable"
        assert config.ocr.enabled

    asyncio.run(run())


def test_startup_request_can_wait_for_an_idle_controller(tmp_path, monkeypatch) -> None:
    import pytest
    from ocr_contract.activation import ActivationPhase, ActivationStatus, ActivationStore

    from vision_bill.config import settings
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    config = settings.model_copy(deep=True)
    config.ocr.enabled = False
    config.ocr.coordination_path = tmp_path / "legacy.lock"

    async def idle_models(**_kwargs) -> frozenset[str]:
        return frozenset()

    provider = SimpleNamespace(
        supports_model_release=True, lifecycle_identity="openai:test", resident_model_ids=idle_models,
    )
    activation = DedicatedOcrActivation(
        config, provider, lambda *_args: None, lambda: None,
        store=store, handoff_path=tmp_path / "gpu.lock",
    )
    checks = iter((False, True))

    async def controller_is_live() -> bool:
        return next(checks)

    monkeypatch.setattr(activation, "_controller_is_live", controller_is_live)
    monkeypatch.setenv("OCR_ACTIVATION_ON_STARTUP", "true")
    monkeypatch.setenv("OCR_ACTIVATION_CONFIRM_HANDOFF", "true")

    async def run() -> None:
        task = asyncio.create_task(activation.run())
        try:
            async with asyncio.timeout(4):
                while (request := store.read_request()) is None:
                    await asyncio.sleep(0.05)
            store.publish_status(ActivationStatus(request.generation, ActivationPhase.FAILED))
            await asyncio.sleep(1)
            assert store.read_request() == request
            failed = await activation.status()
            assert failed.state == "failed" and not failed.selected
            async def live() -> bool:
                return True
            monkeypatch.setattr(activation, "_controller_is_live", live)
            app = app_with_system()
            app.state.dedicated_ocr = activation
            app.dependency_overrides[require_admin] = lambda: None
            client = TestClient(app)
            failed_response = client.get("/system/dedicated-ocr")
            assert failed_response.json()["selected"] is False
            retry = client.post("/system/dedicated-ocr", json={"confirm_handoff": True, "enabled": True})
            assert retry.status_code == 202
            assert retry.json()["state"] == "requested" and not retry.json()["selected"]
            assert store.read_request().generation != request.generation
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_toggle_contract_retains_install_and_legacy_default() -> None:
    from vision_bill.model.dedicated_ocr import DedicatedOcrRequest, DedicatedOcrStatus

    assert DedicatedOcrRequest().enabled is True
    assert DedicatedOcrRequest(enabled=False).enabled is False
    assert DedicatedOcrStatus(state="disabled", installed=True).installed is True
    assert DedicatedOcrStatus(state="active", selected=True).selected is True


async def test_live_topology_round_trip_keeps_one_native_owner_generation(settings, tmp_path, monkeypatch):
    import pytest
    from ocr_contract.activation import ActivationPhase, ActivationStatus, ActivationStore

    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service import gpu_handoff
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    monkeypatch.setenv("OCR_ACTIVATION_ENDPOINT", "http://prepared-ocr.test")
    monkeypatch.setattr(gpu_handoff, "managed_ocr_enabled", lambda: True)
    settings.ocr.coordination_path = tmp_path / "config" / "legacy.lock"
    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    path = tmp_path / "gpu.lock"
    generations = []
    probes = []
    claim_fails = False

    async def resident(**_kwargs):
        probes.append("residency")
        return frozenset()

    async def live():
        return True

    async def claim(token, _deadline):
        nonlocal claim_fails
        generation = int(token.split(":")[0])
        assert not generations or generation > generations[-1]
        generations.append(generation)
        if claim_fails:
            claim_fails = False
            raise RuntimeError("claim result lost")

    async def park(token, _deadline):
        assert int(token.split(":")[0]) == generations[-1]

    provider = SimpleNamespace(supports_model_release=True, lifecycle_identity="openai:test",
                               resident_model_ids=resident, handoff=None)

    def create(config):
        activation = DedicatedOcrActivation(config, provider, lambda *_args: None, lambda: None,
                                            store=store, handoff_path=path)
        monkeypatch.setattr(activation, "_controller_is_live", live)
        monkeypatch.setattr(activation, "_runtime_is_serving", live)
        monkeypatch.setattr(activation.ocr, "claim", claim)
        monkeypatch.setattr(activation.ocr, "park", park)
        return activation

    startup = settings.model_copy(deep=True)
    activation = create(settings)
    assert (await activation.submit(confirm_handoff=True)).state == "requested"
    request = store.read_request()
    assert store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
    await activation.reconcile()
    assert (await activation.submit(confirm_handoff=False)).state == "active"
    original_binding = activation.handoff.binding_path.read_bytes()
    original_journal = activation.handoff.read_journal()
    activation.handoff.journal.write_text(original_journal.model_copy(update={"unsettled": True}).model_dump_json())
    with pytest.raises(HandoffWait, match="model_request_unsettled"):
        await activation.set_gpu_topology(False, confirm_handoff=False)
    assert store.shared_gpu() is True
    activation.handoff.journal.write_text(original_journal.model_dump_json())

    claim_fails = True
    with pytest.raises(HandoffWait, match="ocr_runtime_unavailable"):
        await activation.set_gpu_topology(False, confirm_handoff=False)
    assert store.shared_gpu() is True
    assert (await activation.submit(confirm_handoff=False)).state == "active"

    probes.clear()
    separate = await activation.set_gpu_topology(False, confirm_handoff=False)
    assert separate.state == "active" and separate.shared_gpu is False
    assert provider.handoff is None and probes == []
    activation = create(startup)
    await activation.reconcile()
    assert startup.ocr.enabled and not startup.ocr.shared_gpu
    with pytest.raises(HandoffWait, match="ocr_handoff_confirmation_required"):
        await activation.set_gpu_topology(True, confirm_handoff=False)
    shared = await activation.set_gpu_topology(True, confirm_handoff=True)
    assert shared.state == "active" and shared.shared_gpu is True
    assert provider.handoff is activation.handoff
    assert activation.handoff.binding_path.read_bytes() == original_binding
    assert len(generations) >= 3
    assert store.read_request() == request


async def test_ready_install_can_disable_restart_and_reenable_without_new_generation(tmp_path, monkeypatch):
    from ocr_contract.activation import ActivationPhase, ActivationStatus, ActivationStore

    from vision_bill.config import settings
    from vision_bill.service import gpu_handoff
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    config = settings.model_copy(deep=True)
    config.ocr.enabled = False
    config.ocr.service_url = ""
    config.ocr.coordination_path = tmp_path / "old.lock"
    monkeypatch.setenv("OCR_ACTIVATION_ENDPOINT", "http://prepared-ocr.test")
    monkeypatch.setattr(gpu_handoff, "managed_ocr_enabled", lambda: True)
    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    events = []

    async def idle(**_kwargs):
        return frozenset()

    provider = SimpleNamespace(supports_model_release=True, lifecycle_identity="openai:test",
                               resident_model_ids=idle, handoff=None)

    def create():
        selected = DedicatedOcrActivation(config, provider, lambda *_args: None,
                                          lambda: None, store=store, handoff_path=tmp_path / "gpu.lock")
        async def live():
            return True
        async def claim(token, deadline):
            events.append("claim")
        async def park(token, deadline):
            events.append("park")
        monkeypatch.setattr(selected, "_controller_is_live", live)
        monkeypatch.setattr(selected, "_runtime_is_serving", live)
        monkeypatch.setattr(selected.ocr, "claim", claim)
        monkeypatch.setattr(selected.ocr, "park", park)
        return selected

    activation = create()
    assert (await activation.submit(confirm_handoff=True)).state == "requested"
    request = store.read_request()
    store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
    await activation.reconcile()
    assert (await activation.status()).installed
    assert (await activation.submit(enabled=False, confirm_handoff=False)).state == "disabled"
    assert not config.ocr.enabled
    store.set_runtime_enabled(request, True)
    peer = await activation.status()
    assert peer.state == "installing" and peer.selected
    store.set_runtime_enabled(request, False)
    assert provider.handoff is activation.handoff
    assert events == ["claim", "park"]
    assert store.read_request() == request
    activation = create()
    await activation.reconcile()
    status = await activation.status()
    assert status.state == "disabled" and status.installed
    assert not config.ocr.enabled
    assert (await activation.submit(enabled=True, confirm_handoff=False)).state == "active"
    assert config.ocr.enabled
    assert store.read_request() == request
    assert events == ["claim", "park", "claim", "park"]
    with store.admission(exclusive=False):
        blocked = await activation.submit(enabled=False, confirm_handoff=False)
        assert blocked.reason == "llm_quiescence_required"
    assert store.runtime_enabled(request)
    journal = activation.handoff.read_journal()
    activation.handoff.journal.write_text(journal.model_copy(update={"unsettled": True}).model_dump_json())
    blocked = await activation.submit(enabled=False, confirm_handoff=False)
    assert blocked.reason == "model_request_unsettled"
    assert config.ocr.enabled and store.runtime_enabled(request)
    config.llm.model_name = "unadopted-model"
    assert (await activation.status()).reason == "model_handoff_model_not_adopted"
    assert (await activation.status()).selected is True
    config.llm.model_name = activation.handoff.read_binding().model_ids[0]
    activation.handoff.journal.write_text(journal.model_copy(update={
        "models": [config.llm.model_name], "unsettled": False,
    }).model_dump_json())
    async def refuse_release(_model):
        raise RuntimeError("model_release_unconfirmed: private provider status")
    provider.release_model = refuse_release
    blocked = await activation.submit(enabled=False, confirm_handoff=False)
    assert blocked.reason == "model_release_unconfirmed"
    assert blocked.selected is True and blocked.installed
    assert config.ocr.enabled and store.runtime_enabled(request)
    from vision_bill.service import dedicated_ocr
    monkeypatch.setattr(dedicated_ocr, "runtime_restart_required", lambda _settings: True)
    async def release(_model):
        return None
    provider.release_model = release
    assert (await activation.status()).reason == "llm_lifecycle_required"
    disabled = await activation.submit(enabled=False, confirm_handoff=False)
    assert disabled.state == "disabled" and not disabled.selected


async def test_explicitly_disable_clean_failed_setup_for_direct_vision(settings, tmp_path, monkeypatch):
    import pytest
    from ocr_contract.activation import (
        ActivationPhase,
        ActivationStatus,
        ActivationStore,
        MaintenanceIntent,
    )

    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service.activation_admission import inference_admission
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation
    monkeypatch.setenv("VISION_BILL_OCR_ACTIVATION", "true")
    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    store.set_shared_gpu(False)
    request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
    provider = SimpleNamespace(handoff=None, supports_model_release=False)
    activation = DedicatedOcrActivation(settings, provider, lambda *_: None, lambda: None,
                                        store=store, handoff_path=tmp_path / "gpu.lock")
    assert (await activation.submit(confirm_handoff=False, enabled=False)).state == "action_required"
    store.publish_status(ActivationStatus(request.generation, ActivationPhase.FAILED, "native_qualification_failed"))
    with pytest.raises(HandoffWait), inference_admission(provider, store=store):
        pass
    assert (await activation.submit(confirm_handoff=False, enabled=False)).state == "disabled"
    assert (await activation.status()).state == "disabled"
    with (
        inference_admission(provider, store=store),
        pytest.raises(HandoffWait, match="execution_policy_changed"),
        inference_admission(provider, uses_ocr=True),
    ):
        pass
    assert not settings.ocr.enabled and provider.handoff is None
    assert store.read_request() == request and not store.runtime_enabled(request)
    next_request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE, retry_generation=request.generation)
    assert store.runtime_enabled(next_request)
    store.publish_status(ActivationStatus(next_request.generation, ActivationPhase.FAILED, "native_activation_recovery_required"))
    assert (await activation.submit(confirm_handoff=False, enabled=False)).state == "action_required"
    with pytest.raises(HandoffWait), inference_admission(provider, store=store):
        pass


async def test_managed_mode_does_not_imply_working_runtime(settings, tmp_path, monkeypatch):
    import httpx
    from test_ocr_provider import http_client

    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    activation = DedicatedOcrActivation(settings, SimpleNamespace(), lambda *_: None, lambda: None)
    def handler(request):
        worker = {"ready": False, "generation": 1, "reason_code": "ocr_init_failed"}
        if request.url.path == "/info":
            return httpx.Response(200, json={"service_mode": "managed", "contract_version": 2, "profile": {}, "worker": worker})
        return httpx.Response(503, json=worker)
    http_client(monkeypatch, handler)
    assert not await activation._runtime_is_serving()
