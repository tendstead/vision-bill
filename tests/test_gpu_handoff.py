"""Internal model handoff tests; no hardware or inference is exercised."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import monotonic

import httpx
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def owned_fixture_bindings(monkeypatch):
    from handoff_fixtures import install_owned_bindings
    install_owned_bindings(monkeypatch)


@pytest.fixture
def photos(tmp_path):
    paths = [tmp_path / f"{index}.png" for index in range(2)]
    for path in paths:
        Image.new("RGB", (4, 5)).save(path)
    return paths


def enable_managed_ocr(monkeypatch):
    from vision_bill.config import settings
    monkeypatch.setattr(settings.ocr, "enabled", True)
    monkeypatch.setattr(settings.ocr, "manage_model_lifecycle", True)


class Llm:
    lifecycle_identity = "ollama:test"
    supports_model_release = True

    def __init__(self, events):
        self.events = events
        self.fail = False

    async def resident_model_ids(self, *, deadline):
        return frozenset()

    async def release_model(self, model):
        self.events.append(("unload", model))
        if self.fail:
            raise RuntimeError("release_failed")


class Ocr:
    lifecycle_identity = "ocr:test"

    def __init__(self, events):
        self.events = events

    async def claim(self, token, deadline):
        self.events.append("claim_park")

    async def load(self, token, deadline):
        self.events.append("load_ocr")

    async def park(self, token, deadline):
        self.events.append("park_ocr")


@pytest.mark.asyncio
async def test_changed_ocr_resource_rejects_abandoned_worker_before_side_effects(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    old = Ocr(events)
    handoff = ModelHandoff(tmp_path / "handoff", Llm(events), old)
    with pytest.raises(TimeoutError):
        async with handoff.session(monotonic() + .03) as owner:
            await owner.ocr_phase()
            # An expired session cannot perform its final cleanup.
            await asyncio.sleep(.1)
    state = (tmp_path / "handoff.json").read_text()
    replacement = Ocr(events)
    replacement.lifecycle_identity = "ocr:replacement"
    events.clear()
    with pytest.raises(ValueError, match="model_handoff_ocr_resource_changed"):
        async with ModelHandoff(handoff.path, Llm(events), replacement).session(monotonic() + 1):
            pytest.fail("old native resource was forgotten")
    assert not events
    assert (tmp_path / "handoff.json").read_text() == state


@pytest.mark.asyncio
async def test_ocr_pending_is_durable_before_load_and_failed_park_retains_it(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    ocr = Ocr(events)
    journal = tmp_path / "handoff.json"
    async def load(token, deadline):
        state = json.loads(journal.read_text())
        assert state["ocr_pending"] is True
        assert state["resources"] == {"llm": "ollama:test", "ocr": "ocr:test"}
        assert state["version"] == 2
        assert state["coordinator"] == "vision-bill-model-handoff-v2"
        assert "endpoint" not in state  # Old readers must fail before dispatch.
        events.append("load_ocr")
        raise ConnectionError("load response lost")
    async def park(token, deadline):
        raise RuntimeError("ocr_worker_unstoppable")
    ocr.load, ocr.park = load, park
    with pytest.raises(ConnectionError, match="load response lost"):
        async with ModelHandoff(tmp_path / "handoff", Llm(events), ocr).session(monotonic() + 1) as owner:
            await owner.ocr_phase()
    assert events == ["claim_park", "load_ocr"]
    assert json.loads(journal.read_text())["ocr_pending"] is True
    recovered = Ocr(events)
    async with ModelHandoff(tmp_path / "handoff", Llm(events), recovered).session(monotonic() + 1):
        assert json.loads(journal.read_text())["ocr_pending"] is False
    assert events[-2:] == ["claim_park", "park_ocr"]


@pytest.mark.asyncio
@pytest.mark.parametrize("models", [[], ["old-model"]])
async def test_legacy_journal_never_proves_ocr_parked(tmp_path, models):
    from vision_bill.service.gpu_handoff import ModelHandoff

    journal = tmp_path / "handoff.json"
    original = json.dumps({"endpoint": "ollama:test", "models": models, "generation": 9})
    journal.write_text(original)
    events = []
    with pytest.raises(ValueError, match="model_handoff_legacy_journal.*recover.*OCR.*LLM"):
        async with ModelHandoff(tmp_path / "handoff", Llm(events), Ocr(events)).session(monotonic() + 1):
            pytest.fail("untrusted legacy journal admitted")
    assert not events
    assert journal.read_text() == original


@pytest.mark.asyncio
async def test_nested_session_cannot_change_ocr_resource(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    llm, ocr = Llm(events), Ocr(events)
    other = Ocr(events)
    other.lifecycle_identity = "ocr:other"
    async with ModelHandoff(tmp_path / "handoff", llm, ocr).session(monotonic() + 1):
        with pytest.raises(ValueError, match="model_handoff_identity_changed"):
            async with ModelHandoff(tmp_path / "handoff", llm, other).session(monotonic() + 1):
                pytest.fail("nested resource switched")


@pytest.mark.asyncio
async def test_ocr_identity_must_be_explicit_before_resource_side_effects(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    ocr = Ocr(events)
    ocr.lifecycle_identity = None
    with pytest.raises(ValueError, match="model_handoff_ocr_identity_required"):
        async with ModelHandoff(tmp_path / "handoff", Llm(events), ocr).session(monotonic() + 1):
            pytest.fail("unknown resource admitted")
    assert not events


def test_native_ocr_resource_identity_is_stable_and_hides_endpoint_credentials():
    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider

    first = PaddleOcrProvider("http://user:secret@native:8000/")
    assert first.lifecycle_identity == PaddleOcrProvider("http://user:secret@native:8000").lifecycle_identity
    assert first.lifecycle_identity != PaddleOcrProvider("http://other-native:8000").lifecycle_identity
    assert "secret" not in first.lifecycle_identity
    assert "native" not in first.lifecycle_identity


@pytest.fixture
def ocr_lifecycle_http():
    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider

    parked = {"ready": False, "generation": 2, "reason_code": "ocr_parked",
              "profile_id": None, "runtime_fingerprint": None}
    active = {"ready": True, "generation": 2, "reason_code": None,
              "profile_id": "profile", "runtime_fingerprint": "runtime"}
    responses = {"claim": (200, parked), "load": (200, active), "park": (200, parked)}

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            action = self.path.removeprefix("/lifecycle/")
            status, payload = responses[action]
            body = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield PaddleOcrProvider(f"http://127.0.0.1:{server.server_port}"), responses
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "status", "payload"),
    [
        pytest.param("claim", 200, None, id="claim-empty"),
        pytest.param("park", 200, None, id="park-empty"),
        pytest.param("park", 202, {"ready": False, "generation": 2, "reason_code": "ocr_parked"},
                     id="park-accepted"),
        pytest.param("park", 200, {"ready": True, "generation": 2, "reason_code": "ocr_parked"},
                     id="park-still-active"),
        pytest.param("park", 200, {"ready": "false", "generation": 2, "reason_code": "ocr_parked"},
                     id="park-invalid-type"),
        pytest.param("park", 200, {"ready": False, "generation": 2, "reason_code": "ocr_starting"},
                     id="park-starting"),
        pytest.param("park", 200, {"ready": False, "reason_code": "ocr_parked"},
                     id="park-missing-generation"),
        pytest.param("load", 200, {"ready": False, "generation": 2, "reason_code": "ocr_parked"},
                     id="load-still-parked"),
    ],
)
async def test_invalid_ocr_lifecycle_acknowledgement_blocks_llm(
    tmp_path, monkeypatch, ocr_lifecycle_http, action, status, payload,
):
    from vision_bill.service.gpu_handoff import ModelHandoff

    enable_managed_ocr(monkeypatch)
    provider, responses = ocr_lifecycle_http
    handoff = ModelHandoff(tmp_path / "handoff", Llm([]), provider)
    async with handoff.session(monotonic() + 2) as owner:
        await owner.ocr_phase()
        await owner.llm_phase("formatter")
    assert handoff.read_journal().ocr_pending is False

    previous = responses[action]
    responses[action] = (status, payload)
    with pytest.raises(httpx.RemoteProtocolError, match=f"ocr_invalid_lifecycle_acknowledgement:{action}"):
        async with handoff.session(monotonic() + 2) as owner:
            await owner.ocr_phase()
            await owner.llm_phase("formatter")
            pytest.fail("invalid OCR acknowledgement allowed LLM dispatch")

    journal = handoff.read_journal()
    assert journal.models == []
    # A rejected load can be followed by a confirmed cleanup park.
    assert journal.ocr_pending is (action != "load")

    responses[action] = previous
    async with handoff.session(monotonic() + 2) as owner:
        await owner.llm_phase("formatter")
        assert handoff.read_journal().ocr_pending is False


@pytest.mark.asyncio
async def test_lost_load_response_must_park_before_any_llm_tracking(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    ocr = Ocr(events)
    async def load(token, deadline):
        raise ConnectionError("load response lost")
    async def park(token, deadline):
        raise RuntimeError("ocr_worker_unstoppable")
    ocr.load, ocr.park = load, park
    blocked = False
    with pytest.raises(RuntimeError, match="ocr_worker_unstoppable"):
        async with ModelHandoff(tmp_path / "handoff", Llm(events), ocr).session(monotonic() + 1) as owner:
            with pytest.raises(ConnectionError):
                await owner.ocr_phase()
            try:
                await owner.llm_phase("formatter")
            except RuntimeError as error:
                blocked = str(error) == "ocr_worker_unstoppable"
    assert blocked
    assert json.loads((tmp_path / "handoff.json").read_text())["models"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["coordinator", "generation", "unsettled", "ocr_pending", "resources", "json"])
async def test_invalid_versioned_journal_fails_before_resource_calls(tmp_path, change):
    from vision_bill.service.gpu_handoff import ModelHandoff

    state = {"version": 2, "coordinator": "vision-bill-model-handoff-v2",
             "resources": {"llm": "ollama:test", "ocr": "ocr:test"},
             "models": [], "generation": 1, "unsettled": False, "ocr_pending": True}
    if change == "coordinator":
        del state["coordinator"]
    elif change == "generation":
        state[change] = True
    elif change in ("unsettled", "ocr_pending"):
        state[change] = "false"
    elif change == "resources":
        state["resources"].pop("ocr")
    raw = "{" if change == "json" else json.dumps(state)
    journal = tmp_path / "handoff.json"
    journal.write_text(raw)
    events = []
    with pytest.raises(ValueError, match="model_handoff_invalid_journal"):
        async with ModelHandoff(tmp_path / "handoff", Llm(events), Ocr(events)).session(monotonic() + 1):
            pytest.fail("malformed journal admitted")
    assert not events
    assert journal.read_text() == raw


@pytest.mark.asyncio
async def test_changed_llm_resource_fails_before_even_claiming_ocr(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    llm, ocr = Llm(events), Ocr(events)
    async with ModelHandoff(tmp_path / "handoff", llm, ocr).session(monotonic() + 1):
        pass
    original = (tmp_path / "handoff.json").read_text()
    llm.lifecycle_identity = "ollama:other"
    events.clear()
    with pytest.raises(ValueError, match="model_handoff_llm_resource_changed"):
        async with ModelHandoff(tmp_path / "handoff", llm, ocr).session(monotonic() + 1):
            pytest.fail("changed LLM admitted")
    assert not events
    assert (tmp_path / "handoff.json").read_text() == original


@pytest.mark.asyncio
async def test_matching_resource_recovery_reaps_abandoned_native_before_load(tmp_path):
    from ocr_service.worker import OcrWorkerManager
    from test_ocr_worker_lifecycle import FakeContext

    from vision_bill.service.gpu_handoff import ModelHandoff

    context = FakeContext()
    worker = OcrWorkerManager(mp_context=context, startup_timeout=.2)
    class NativeOcr(Ocr):
        async def claim(self, token, deadline):
            worker.claim(token)

        async def load(self, token, deadline):
            worker.load(token)

        async def park(self, token, deadline):
            worker.park(token)

    events = []
    handoff = ModelHandoff(tmp_path / "handoff", Llm(events), NativeOcr(events))
    try:
        with pytest.raises(TimeoutError):
            async with handoff.session(monotonic() + .03) as owner:
                await owner.ocr_phase()
                await asyncio.sleep(.1)
        abandoned = context.processes[0]
        assert abandoned.is_alive()
        assert json.loads(handoff.journal.read_text())["ocr_pending"] is True
        async with handoff.session(monotonic() + 1) as owner:
            assert not abandoned.is_alive()
            assert worker.readiness().reason_code == "ocr_parked"
            assert json.loads(handoff.journal.read_text())["ocr_pending"] is False
            await owner.ocr_phase()
            assert len(context.processes) == 2
        assert all(not process.is_alive() for process in context.processes)
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_llm_phase_closes_ocr_reentry_until_fresh_session(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    handoff = ModelHandoff(tmp_path / "handoff", Llm(events), Ocr(events))
    async with handoff.session(monotonic() + 1) as owner:
        await owner.ocr_phase()
        await owner.ocr_phase()  # Multiple photos reuse one OCR residency.
        assert events.count("load_ocr") == 1
        await owner.llm_phase("formatter")
        before = list(events)
        with pytest.raises(RuntimeError, match="model_handoff_ocr_phase_closed.*fresh session"):
            await owner.ocr_phase()
        assert events == before
        assert owner.models == ["formatter"]
    async with handoff.session(monotonic() + 1) as owner:
        await owner.ocr_phase()
    assert events.count("load_ocr") == 2


@pytest.mark.asyncio
async def test_phases_serialize_multiple_photos_and_all_llm_calls(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff
    events = []
    llm, ocr = Llm(events), Ocr(events)
    handoff = ModelHandoff(tmp_path / "handoff", llm, ocr)
    async with handoff.session(monotonic() + 3) as owner:
        await owner.ocr_phase()
        events.extend(["photo0", "photo1"])
        await owner.llm_phase("formatter")
        events.extend(["format", "repair", "classify"])
    assert events == ["claim_park", "load_ocr", "photo0", "photo1", "park_ocr",
                      "format", "repair", "classify", "park_ocr", ("unload", "formatter")]


@pytest.mark.asyncio
async def test_failed_cleanup_is_retried_before_successor_load(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff
    events = []
    llm, ocr = Llm(events), Ocr(events)
    handoff = ModelHandoff(tmp_path / "handoff", llm, ocr)
    with pytest.raises(RuntimeError, match="release_failed"):
        async with handoff.session(monotonic() + 3) as owner:
            await owner.llm_phase("own")
            llm.fail = True
    assert json.loads((tmp_path / "handoff.json").read_text())["models"] == ["own"]
    with pytest.raises(RuntimeError, match="release_failed"):
        async with handoff.session(monotonic() + 3) as owner:
            await owner.ocr_phase()
    assert "load_ocr" not in events
    llm.fail = False
    async with handoff.session(monotonic() + 3) as owner:
        await owner.ocr_phase()
    assert events.index(("unload", "own")) < events.index("load_ocr")


@pytest.mark.asyncio
async def test_independent_coordinators_wait_and_cancellation_leaves_cleanup(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service.gpu_handoff import ModelHandoff
    events = []
    llm, ocr = Llm(events), Ocr(events)
    one = ModelHandoff(tmp_path / "handoff", llm, ocr)
    two = ModelHandoff(tmp_path / "handoff", llm, ocr)
    entered = asyncio.Event()
    async def first():
        async with one.session(monotonic() + 3) as owner:
            await owner.llm_phase("own")
            entered.set()
            await asyncio.Event().wait()
    task = asyncio.create_task(first())
    await entered.wait()
    with pytest.raises(HandoffWait) as caught:
        async with two.session(monotonic() + .03):
            pytest.fail("overlapped")
    assert caught.value.status.reason_code == "model_handoff_busy"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with two.session(monotonic() + 3) as owner:
        await owner.ocr_phase()
    assert events.index(("unload", "own")) < events.index("load_ocr")


@pytest.mark.asyncio
async def test_unsupported_provider_fails_without_loading(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff
    events = []
    llm = Llm(events)
    llm.supports_model_release = False
    with pytest.raises(ValueError, match="model_release_unsupported"):
        async with ModelHandoff(tmp_path / "handoff", llm, Ocr(events)).session(monotonic() + 3):
            pytest.fail("unsupported provider admitted")
    assert not events


def test_worker_park_is_reversible_and_fences_stale_owner():
    from ocr_service.worker import OcrWorkerError, OcrWorkerManager
    from test_ocr_worker_lifecycle import FakeContext
    context = FakeContext()
    worker = OcrWorkerManager(mp_context=context, startup_timeout=.2)
    assert worker.readiness().reason_code == "ocr_parked"
    worker.claim("1:first")
    worker.load("1:first")
    assert worker.readiness().ready
    worker.park("1:first")
    assert worker.readiness().reason_code == "ocr_parked"
    assert not context.processes[-1].is_alive()
    worker.claim("2:second")
    with pytest.raises(OcrWorkerError, match="ocr_stale_owner"):
        worker.load("1:first")
    worker.load("2:second")
    worker.park("2:second")
    assert len(context.processes) == 2
    worker.stop()


def test_failed_native_park_revokes_delayed_load_permission_before_teardown():
    from ocr_service.worker import OcrWorkerError, OcrWorkerManager
    from test_ocr_worker_lifecycle import FakeContext

    context = FakeContext()
    worker = OcrWorkerManager(mp_context=context, startup_timeout=.2)
    worker.claim("1:first")
    worker.load("1:first")
    process = context.processes[-1]
    context.ignore_terminate = True
    process.kill = lambda: None
    try:
        with pytest.raises(OcrWorkerError, match="ocr_worker_unstoppable"):
            worker.park("1:first")
        with pytest.raises(OcrWorkerError, match="ocr_owner_phase_closed"):
            worker.load("1:first")
        assert len(context.processes) == 1
    finally:
        process.dead.set()
        worker.stop()


@pytest.mark.asyncio
async def test_ollama_unloads_only_exact_owned_id_and_verifies_foreign_models():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from vision_bill.provider.llm.ollama import OllamaProvider
    provider = OllamaProvider("http://local")
    provider._client.generate = AsyncMock()
    provider._client.ps = AsyncMock(return_value=SimpleNamespace(model_fields_set={"models"}, models=[SimpleNamespace(model="foreign:latest")]))
    await provider.release_model("own")
    provider._client.generate.assert_awaited_once_with(model="own", keep_alive=0)
    provider._client.ps.return_value.models.append(SimpleNamespace(model="own:latest"))
    with pytest.raises(RuntimeError, match="model_release_unconfirmed"):
        await provider.release_model("own")


@pytest.mark.asyncio
async def test_ambiguous_model_call_blocks_successor_ocr(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff
    events = []
    handoff = ModelHandoff(tmp_path / "handoff", Llm(events), Ocr(events))
    async def ambiguous():
        raise ConnectionError("response lost")
    with pytest.raises(ConnectionError, match="response lost"):
        async with handoff.session(monotonic() + 2) as owner:
            await owner.llm_phase("own")
            await owner.model_call(ambiguous())
    with pytest.raises(RuntimeError, match="model_request_unsettled"):
        async with handoff.session(monotonic() + 2) as owner:
            await owner.ocr_phase()
    assert "load_ocr" not in events
    assert json.loads((tmp_path / "handoff.json").read_text())["unsettled"]


@pytest.mark.asyncio
async def test_expired_call_cancellation_keeps_late_completion_from_touching_successor(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff
    events = []
    handoff = ModelHandoff(tmp_path / "handoff", Llm(events), Ocr(events))
    entered, finish = asyncio.Event(), asyncio.Event()
    async def late():
        entered.set()
        await finish.wait()
    async def first():
        async with handoff.session(monotonic() + .04) as owner:
            await owner.llm_phase("own")
            await owner.model_call(late())
    task = asyncio.create_task(first())
    await entered.wait()
    await asyncio.sleep(.05)
    task.cancel()
    with pytest.raises(TimeoutError):
        await task
    before = (tmp_path / "handoff.json").read_text()
    finish.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert (tmp_path / "handoff.json").read_text() == before
    with pytest.raises(RuntimeError, match="model_request_unsettled"):
        async with handoff.session(monotonic() + 2):
            pytest.fail("unsafe admission")


@pytest.mark.asyncio
async def test_process_shared_lock_excludes_an_independent_process(tmp_path):
    import sys

    from vision_bill.service.gpu_handoff import ModelHandoff
    events = []
    lock = tmp_path / "handoff"
    program = """import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(73)
sys.exit(0)
"""
    async with ModelHandoff(lock, Llm(events), Ocr(events)).session(monotonic() + 3):
        proc = await asyncio.create_subprocess_exec(sys.executable, "-c", program, str(lock))
        assert await proc.wait() == 73
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", program, str(lock))
    assert await proc.wait() == 0


@pytest.mark.asyncio
async def test_pipeline_real_handoff_orders_all_photos_before_formatter(photos, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.service.extraction_pipeline import ExtractionPipeline
    events = []
    llm = formatter()
    enable_managed_ocr(monkeypatch)
    llm.release_model = AsyncMock(side_effect=lambda model: events.append(("unload", model)))
    ocr = sidecar()
    ocr.managed_lifecycle = True
    ocr.claim = AsyncMock(side_effect=lambda *args: events.append("claim_park"))
    ocr.load = AsyncMock(side_effect=lambda *args: events.append("load_ocr"))
    ocr.park = AsyncMock(side_effect=lambda *args: events.append("park_ocr"))
    extract = ocr.extract_page
    async def photo(*args, **kwargs):
        events.append("photo" + str(kwargs["photo_index"]))
        return await extract(*args, **kwargs)
    ocr.extract_page = photo
    chat = llm._client.chat
    async def message(*args, **kwargs):
        events.append("llm")
        return await chat(*args, **kwargs)
    llm._client.chat = message
    settings = OcrSettings(enabled=True, coordination_path=tmp_path / "handoff")
    result = await ExtractionPipeline(llm, ocr, settings).extract(
        "fixture", photos, options=ExperimentOptions(include_classification=False),
        deadline=monotonic() + 3, tags=[],
    )
    assert result.receipt is not None
    assert events == ["claim_park", "load_ocr", "photo0", "photo1",
                      "park_ocr", "llm", "park_ocr", ("unload", "fixture")]


@pytest.mark.asyncio
async def test_direct_pipeline_parks_ocr_without_loading_it(photos, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.service.extraction_pipeline import ExtractionPipeline
    llm, ocr = formatter(), sidecar()
    enable_managed_ocr(monkeypatch)
    llm.release_model = AsyncMock()
    ocr.managed_lifecycle = True
    ocr.claim, ocr.load, ocr.park = AsyncMock(), AsyncMock(), AsyncMock()
    settings = OcrSettings(enabled=True, coordination_path=tmp_path / "handoff")
    await ExtractionPipeline(llm, ocr, settings).extract(
        "fixture", photos, options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
        deadline=monotonic() + 3, tags=[],
    )
    ocr.claim.assert_awaited_once()
    ocr.load.assert_not_awaited()
    llm.release_model.assert_awaited_once_with("fixture")


@pytest.mark.asyncio
async def test_failed_park_never_allows_fallback_or_formatter(photos, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.service.extraction_pipeline import ExtractionPipeline
    llm, ocr = formatter(), sidecar()
    enable_managed_ocr(monkeypatch)
    llm.release_model = AsyncMock()
    ocr.managed_lifecycle = True
    ocr.claim, ocr.load = AsyncMock(), AsyncMock()
    ocr.park = AsyncMock(side_effect=RuntimeError("ocr_worker_unstoppable"))
    settings = OcrSettings(enabled=True, allow_vision_fallback=True, coordination_path=tmp_path / "handoff")
    with pytest.raises(RuntimeError, match="ocr_worker_unstoppable"):
        await ExtractionPipeline(llm, ocr, settings).extract(
            "fixture", photos, options=ExperimentOptions(allow_ocr_fallback=True),
            deadline=monotonic() + 3, tags=[],
        )
    llm._client.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_gpu", [True, False])
async def test_explicit_benchmark_identity_qualifies_once_and_is_stable_while_parked(tmp_path, monkeypatch, shared_gpu):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from test_extraction_pipeline import formatter

    from vision_bill.config import OcrSettings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.service.experiment_execution import pipeline_identity

    class ManagedOcr(Ocr):
        managed_lifecycle = True
        service_url = "http://native"
        timeout = 600
        ready = False
        runtime = None

        async def load(self, token, deadline):
            await super().load(token, deadline)
            self.ready, self.runtime = True, "qualified-runtime"

        async def park(self, token, deadline):
            await super().park(token, deadline)
            self.ready = False

        async def get_capabilities(self):
            return OcrCapabilities(enabled=True, configured=True, reachable=True,
                                   ready=self.ready, can_load=not self.ready,
                                   profile_id="qualified-profile" if self.runtime else None,
                                   runtime_fingerprint=self.runtime, ocr_max_new_tokens=4096,
                                   reason_code=None if self.ready else "ocr_parked")

    events = []
    llm = formatter()
    llm.supports_model_release = shared_gpu
    enable_managed_ocr(monkeypatch)
    llm.release_model = AsyncMock()
    ocr = ManagedOcr(events)
    service = SimpleNamespace(_provider=llm, _ocr_provider=ocr, _ocr_settings=OcrSettings(
        enabled=True, coordination_path=tmp_path / "handoff", shared_gpu=shared_gpu,
    ))
    options = ExperimentOptions(pipeline="ocr")
    before = await pipeline_identity(service, options)
    assert before["ocr_runtime"]["runtime_fingerprint"] is None
    assert not events
    frozen = await pipeline_identity(service, options, qualify=True, managed_model_ids=["fixture"])
    assert not ocr.ready
    assert frozen["ocr_runtime"]["runtime_fingerprint"] == "qualified-runtime"
    assert frozen == await pipeline_identity(service, options)
    assert events.count("load_ocr") == 1
    if shared_gpu:
        assert "shared_gpu" not in frozen["ocr_settings"]
        assert frozen["managed_binding_sha256"]
    else:
        assert frozen["ocr_settings"]["shared_gpu"] is False
        assert frozen["managed_binding_sha256"] is None
        events.clear()
        direct = await pipeline_identity(service, ExperimentOptions(pipeline="direct_vision"), qualify=True)
        assert direct["managed_binding_sha256"] is None and direct["ocr_runtime"] is None
        assert not events
    ocr.runtime = "different-runtime"
    assert frozen != await pipeline_identity(service, options)


@pytest.mark.asyncio
async def test_legacy_repairs_keep_one_owner(tmp_path):
    from unittest.mock import AsyncMock

    from test_extraction_pipeline import formatter

    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    llm = formatter()
    valid = llm._client.chat.return_value
    invalid = valid.model_copy(deep=True)
    invalid.message.content = "not json"
    llm._client.chat.side_effect = [invalid, valid]
    llm.release_model = AsyncMock()
    llm.handoff = ModelHandoff(tmp_path / "handoff", llm, Ocr(events))
    await llm.analyse_receipt_from_ocr_text("fixture", {"text": "SHOP"})
    assert llm._client.chat.await_count == 2
    assert events == ["claim_park", "park_ocr"]
    llm.release_model.assert_awaited_once_with("fixture")


def test_unconfirmed_native_death_blocks_park_and_poll_recovery():
    from ocr_service.worker import OcrWorkerError
    from test_ocr_worker_lifecycle import FakeContext, manager

    context = FakeContext()
    worker = manager(context)
    worker._owner = "1:test"
    context.ignore_terminate = True
    process = context.processes[-1]
    process.kill = lambda: None
    with pytest.raises(OcrWorkerError, match="ocr_worker_unstoppable"):
        worker.park("1:test")
    assert worker.readiness().reason_code == "ocr_worker_unstoppable"
    assert len(context.processes) == 1
    process.dead.set()
    worker.stop()


def test_supported_budget_probe_uses_managed_handoff(tmp_path, monkeypatch):
    from scripts.context_budget import Budget
    from vision_bill.config import settings
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    provider = Llm(events)
    provider.handoff = ModelHandoff(tmp_path / "handoff", provider, Ocr(events))
    monkeypatch.setattr(settings.ocr, "enabled", True)
    monkeypatch.setattr(settings.ocr, "manage_model_lifecycle", True)
    monkeypatch.setattr("vision_bill.provider.llm.ollama.OllamaProvider", lambda *args, **kwargs: provider)
    budget = Budget("http://local", "own", 100, 2)
    budget.ctx = 100
    def generate(path, payload):
        events.append("generate")
        return {"prompt_eval_count": 10}
    monkeypatch.setattr(budget, "_post", generate)
    assert budget.count("test") == 10
    assert events == ["claim_park", "generate", "park_ocr", ("unload", "own")]


@pytest.mark.asyncio
async def test_review_disabled_production_rejects_forced_ocr_before_concurrent_direct_work(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings, settings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service.experiment_execution import pipeline_identity

    monkeypatch.setattr(settings.ocr, "enabled", False)
    llm, ocr = formatter(), sidecar()
    ocr.managed_lifecycle = True
    ocr.claim, ocr.load, ocr.park = AsyncMock(), AsyncMock(), AsyncMock()
    llm.release_model = AsyncMock()
    monkeypatch.setattr("vision_bill.provider.factory.get_ocr_provider", lambda config: ocr)
    service = SimpleNamespace(_provider=llm, _ocr_provider=None,
                              _ocr_settings=OcrSettings(enabled=False, coordination_path=tmp_path / "handoff"))
    entered, finish = asyncio.Event(), asyncio.Event()
    response = llm._client.chat.return_value
    async def direct_chat(**kwargs):
        entered.set()
        await finish.wait()
        return response
    llm._client.chat.side_effect = direct_chat
    direct = asyncio.create_task(llm.send_message("fixture", []))
    await entered.wait()
    try:
        assert llm.handoff is None
        with pytest.raises(HandoffWait, match="model_handoff_managed_ocr_required"):
            await pipeline_identity(service, ExperimentOptions(pipeline="ocr"),
                                    qualify=True, managed_model_ids=["fixture"])
        ocr.claim.assert_not_awaited()
        ocr.load.assert_not_awaited()
        llm.release_model.assert_not_awaited()
    finally:
        finish.set()
        await direct


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_stage", ["park", "unload"])
async def test_review_body_expiry_keeps_absolute_deadline_and_pending_cleanup(tmp_path, slow_stage):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events = []
    llm, ocr = Llm(events), Ocr(events)
    if slow_stage == "park":
        original = ocr.park
        async def slow_park(token, deadline):
            await asyncio.sleep(.2)
            await original(token, deadline)
        ocr.park = slow_park
    else:
        original_release = llm.release_model
        async def slow_release(model):
            await asyncio.sleep(.2)
            await original_release(model)
        llm.release_model = slow_release
    handoff = ModelHandoff(tmp_path / "handoff", llm, ocr)
    started = monotonic()
    with pytest.raises(TimeoutError):
        async with handoff.session(started + .04) as owner:
            await owner.llm_phase("own")
            await asyncio.sleep(.3)
    assert monotonic() - started < .15
    assert json.loads((tmp_path / "handoff.json").read_text())["models"] == ["own"]
    assert ("unload", "own") not in events
    async with handoff.session(monotonic() + 1) as owner:
        await owner.ocr_phase()
    assert events.index(("unload", "own")) < events.index("load_ocr")


@pytest.mark.asyncio
async def test_review_frozen_managed_snapshot_waits_before_claim_when_production_disabled(settings, tmp_path, monkeypatch):
    from copy import deepcopy
    from unittest.mock import AsyncMock

    from test_submission_service import context

    from vision_bill.config import OcrSettings
    from vision_bill.config import settings as runtime_settings

    monkeypatch.setattr(runtime_settings.ocr, "enabled", False)
    service, receipts, db, claim, _ = context(settings, tmp_path)
    claim.image.effective_options["ocr"] = OcrSettings(enabled=True, required=True).model_dump(mode="json")
    saved = deepcopy(claim.image.effective_options)
    receipts.check_connection = AsyncMock(return_value=True)
    receipts.ocr_capabilities = AsyncMock()
    assert not await service.ready_for(claim.image)
    db.set_wait_reason.assert_awaited_once_with(claim.image, "execution_policy_changed")
    receipts.check_connection.assert_not_awaited()
    receipts.ocr_capabilities.assert_not_awaited()
    receipts.extract_receipt.assert_not_awaited()
    db.finish_attempt.assert_not_awaited()
    assert claim.image.effective_options == saved


def test_review_capabilities_distinguish_disabled_managed_and_cpu_modes(monkeypatch):
    from types import SimpleNamespace

    from vision_bill.config import OcrSettings, settings
    from vision_bill.service.benchmark_service import BenchmarkService

    monkeypatch.setattr(settings.ocr, "enabled", False)
    service = object.__new__(BenchmarkService)
    service._provider = SimpleNamespace(supports_thinking_control=True)
    service._receipts = SimpleNamespace(_ocr_settings=OcrSettings(enabled=False))
    assert not service.capabilities()["ocr_available"]
    service._receipts._ocr_settings.manage_model_lifecycle = False
    assert service.capabilities()["ocr_available"]


@pytest.mark.asyncio
async def test_review_cleanup_passes_remaining_absolute_deadline_on_body_error(tmp_path):
    from vision_bill.service.gpu_handoff import ModelHandoff

    events, park_deadlines = [], []
    llm, ocr = Llm(events), Ocr(events)
    async def slow_park(token, deadline):
        park_deadlines.append(deadline)
        await asyncio.sleep(.2)
    ocr.park = slow_park
    started = monotonic()
    deadline = started + .04
    with pytest.raises(ValueError, match="body failed"):
        async with ModelHandoff(tmp_path / "handoff", llm, ocr).session(deadline) as owner:
            await owner.llm_phase("own")
            raise ValueError("body failed")
    assert monotonic() - started < .15
    assert park_deadlines == [deadline]
    assert json.loads((tmp_path / "handoff.json").read_text())["models"] == ["own"]


@pytest.mark.parametrize("blocked_probe", ["residency", "ocr_mode"])
@pytest.mark.parametrize("caller_cancels", [False, True])
async def test_extraction_admission_probe_expires_before_outer_work_timeout(
    photos, tmp_path, monkeypatch, blocked_probe, caller_cancels,
):
    from unittest.mock import AsyncMock

    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service.extraction_pipeline import ExtractionPipeline
    from vision_bill.service.gpu_handoff import ModelHandoff

    llm, ocr = formatter(), sidecar()
    enable_managed_ocr(monkeypatch)
    ocr.managed_lifecycle = True
    ocr.claim, ocr.load, ocr.park = AsyncMock(), AsyncMock(), AsyncMock()
    settings = OcrSettings(enabled=True, coordination_path=tmp_path / "handoff.lock")
    llm.handoff = ModelHandoff(settings.coordination_path, llm, ocr)
    entered = asyncio.Event()

    async def stall(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    if blocked_probe == "residency":
        llm.resident_model_ids = stall
        reason = "model_handoff_residency_unknown"
    else:
        ocr.get_capabilities = stall
        reason = "ocr_unavailable"

    pipeline = ExtractionPipeline(llm, ocr, settings)
    deadline = monotonic() + 0.6

    async def extract():
        async with asyncio.timeout(0.5):
            await pipeline.extract(
                "fixture", photos, options=ExperimentOptions(), deadline=deadline,
                handoff_deadline=deadline - 0.1, tags=[],
            )

    if caller_cancels:
        task = asyncio.create_task(extract())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(HandoffWait, match=reason):
            await extract()
    assert entered.is_set() and monotonic() < deadline
    assert not llm.handoff.journal.exists()
    for effect in (ocr.claim, ocr.load, ocr.park, llm._client.chat):
        effect.assert_not_awaited()


def test_prepared_off_retains_coordinator_without_advertising_ocr(monkeypatch, tmp_path):
    from vision_bill.config import OcrSettings, settings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.service.experiment_execution import effective_policy
    from vision_bill.service.gpu_handoff import ModelHandoff
    from vision_bill.service.receipt_service import ReceiptService

    monkeypatch.setenv("VISION_BILL_OCR_ACTIVATION", "true")
    monkeypatch.setattr(settings.ocr, "enabled", False)
    provider = Llm([])
    provider.supports_thinking_control = True
    provider.handoff = ModelHandoff(tmp_path / "handoff", provider, Ocr([]))
    receipts = object.__new__(ReceiptService)
    receipts._provider = provider
    receipts.set_ocr_route(None, OcrSettings(enabled=False, coordination_path=provider.handoff.path))
    assert receipts.execution_handoff(receipts._ocr_settings) is provider.handoff
    service = object.__new__(BenchmarkService)
    service._provider, service._receipts = provider, receipts
    assert not service.capabilities()["ocr_available"]
    with pytest.raises(HandoffWait, match="dedicated_ocr_disabled"):
        effective_policy(receipts, ExperimentOptions(pipeline="ocr"))


async def test_prepared_off_session_permits_parking_but_refuses_native_load(monkeypatch, tmp_path):
    from vision_bill.config import settings
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service.gpu_handoff import ModelHandoff

    monkeypatch.setenv("VISION_BILL_OCR_ACTIVATION", "true")
    monkeypatch.setattr(settings.ocr, "enabled", False)
    events = []
    ocr = Ocr(events)
    ocr.managed_lifecycle = True
    handoff = ModelHandoff(tmp_path / "handoff", Llm(events), ocr)
    async with handoff.session(monotonic() + 1) as owner:
        with pytest.raises(HandoffWait, match="dedicated_ocr_disabled"):
            await owner.ocr_phase()
    assert events == ["claim_park", "park_ocr"]


def test_prepared_profile_preserves_explicit_external_ocr_benchmark(monkeypatch):
    from types import SimpleNamespace

    from vision_bill.config import OcrSettings, settings
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.service.experiment_execution import effective_policy

    monkeypatch.setenv("VISION_BILL_OCR_ACTIVATION", "true")
    external = object()
    ocr_settings = OcrSettings(enabled=True, service_url="http://external-ocr.test", manage_model_lifecycle=False)
    monkeypatch.setattr(settings, "ocr", ocr_settings)
    receipts = SimpleNamespace(_ocr_settings=ocr_settings, _ocr_provider=external)
    service = object.__new__(BenchmarkService)
    service._provider = SimpleNamespace(supports_thinking_control=True)
    service._receipts = receipts
    assert service.capabilities()["ocr_available"]
    selected, effective, _ = effective_policy(receipts, ExperimentOptions(pipeline="ocr"))
    assert selected is external and effective.enabled
