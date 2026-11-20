"""External dedication and residency barriers, with owned software doubles only."""
import asyncio
import fcntl
import json
import stat
import subprocess
import sys
from time import monotonic

import httpx
import pytest
import pytest_asyncio
from ollama import AsyncClient
from test_gpu_handoff import Llm, Ocr

from vision_bill.model.handoff import HandoffWait
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.service.gpu_handoff import ModelHandoff


class ResidentLlm(Llm):
    def __init__(self, events):
        super().__init__(events)
        self.resident = frozenset()

    async def resident_model_ids(self, *, deadline):
        self.events.append("residency")
        return self.resident


async def adopt(handoff, models=("own",), **kwargs):
    from vision_bill.distribution.handoff import adopt_binding
    return await adopt_binding(
        handoff, models, dedicated_coordinated_requests=True,
        external_eager_loading_disabled=True, deadline=monotonic() + 1, **kwargs,
    )


@pytest.mark.asyncio
async def test_fresh_empty_journal_cannot_load_ocr_without_binding(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    llm = ResidentLlm(events)
    llm.resident = frozenset({"foreign"})
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, Ocr(events))
    with pytest.raises(HandoffWait) as caught:
        async with handoff.session(monotonic() + 1) as owner:
            await owner.ocr_phase()
    assert caught.value.status.state == "configuration_required"
    assert "load_ocr" not in events
    assert not handoff.journal.exists()
    with pytest.raises(HandoffWait):
        await adopt(handoff)
    assert all(not isinstance(event, tuple) for event in events)
    assert not handoff.binding_path.exists()


@pytest.mark.asyncio
async def test_adoption_persists_exact_secret_free_binding_and_inspection_is_read_only(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    binding = await adopt(handoff)
    assert binding.model_ids == ["own"]
    assert binding.coordination_path == str(handoff.path.resolve())
    assert stat.S_IMODE(handoff.binding_path.stat().st_mode) == 0o600
    assert handoff.binding_path == tmp_path / "handoff.binding.json"
    assert not handoff.journal.exists()
    before = handoff.binding_path.read_bytes()
    status = await handoff.inspect(model_id="own", deadline=monotonic() + 1)
    assert status.state == "ready"
    assert handoff.binding_path.read_bytes() == before
    assert events == ["residency", "residency"]
    assert not handoff.journal.exists()


@pytest.mark.asyncio
async def test_model_outside_explicit_binding_is_held_before_tracking_or_dispatch(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    with pytest.raises(HandoffWait) as caught:
        async with handoff.session(monotonic() + 1) as owner:
            await owner.llm_phase("other")
    assert caught.value.status.state == "configuration_required"
    assert owner.models == []
    assert all(not isinstance(event, tuple) for event in events)


@pytest.mark.asyncio
async def test_foreign_residency_after_adoption_never_claims_or_unloads(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    llm = ResidentLlm(events)
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1):
        pass
    events.clear()
    llm.resident = frozenset({"foreign"})
    before = handoff.binding_path.read_bytes()
    journal_before = handoff.journal.read_bytes()
    with pytest.raises(HandoffWait) as caught:
        async with handoff.session(monotonic() + 1) as owner:
            await owner.ocr_phase()
    assert caught.value.status.state == "recovery_required"
    assert events == ["residency"]
    assert handoff.binding_path.read_bytes() == before
    assert handoff.journal.read_bytes() == journal_before


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["legacy", "unsettled", "ocr_pending", "models", "mismatch", "malformed"])
async def test_adoption_preserves_unresolved_journal_bytes(tmp_path, kind):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    state = {"version": 2, "coordinator": "vision-bill-model-handoff-v2", "resources": handoff.resources.model_dump(),
             "models": [], "generation": 8, "unsettled": False, "ocr_pending": False}
    if kind in ("unsettled", "ocr_pending"):
        state[kind] = True
    elif kind == "models":
        state["models"] = ["own"]
    elif kind == "mismatch":
        state["resources"]["llm"] = "old"
    raw = b'{"models":[]}' if kind == "legacy" else b'{' if kind == "malformed" else json.dumps(state).encode()
    handoff.journal.write_bytes(raw)
    with pytest.raises((ValueError, RuntimeError)):
        await adopt(handoff)
    assert handoff.journal.read_bytes() == raw
    assert not handoff.binding_path.exists()
    assert events == []


@pytest.mark.asyncio
async def test_live_owner_cannot_be_adopted_and_binding_is_preserved(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1):
        original = handoff.binding_path.read_bytes()
        journal_before = handoff.journal.read_bytes()
        with pytest.raises(HandoffWait) as caught:
            await adopt(handoff)
        assert caught.value.status.state == "busy"
        assert handoff.binding_path.read_bytes() == original
        assert handoff.journal.read_bytes() == journal_before


@pytest.mark.asyncio
@pytest.mark.parametrize("models,dedicated,eager", [([], True, True), (["own"], False, True), (["own"], True, False), ([" "], True, True)])
async def test_invalid_attestations_and_ids_reject_before_effects(tmp_path, models, dedicated, eager):
    from vision_bill.distribution.handoff import adopt_binding
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    with pytest.raises(ValueError):
        await adopt_binding(handoff, models, dedicated_coordinated_requests=dedicated,
                            external_eager_loading_disabled=eager, deadline=monotonic() + 1)
    assert events == []
    assert not handoff.path.exists()


@pytest.mark.asyncio
async def test_lost_chat_reply_keeps_original_error_and_fences_journal(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async def lost():
        raise ConnectionError("original chat reply lost")
    with pytest.raises(ConnectionError, match="original chat reply lost"):
        async with handoff.session(monotonic() + 1) as owner:
            await owner.llm_phase("own")
            await owner.model_call(lost())
    state = json.loads(handoff.journal.read_bytes())
    assert state["unsettled"] is True
    assert state["models"] == ["own"]
    assert ("unload", "own") not in events


def test_generic_inference_identity_is_independent_of_lifecycle_and_credentials():
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.openai import OpenAIProvider
    first = OpenAIProvider("HTTP://user:secret@EXAMPLE.local:80/v1/?token=private", "key-one")
    second = OpenAIProvider("http://example.local/v1", "key-two")
    assert first.lifecycle_identity == "unsupported"
    assert first.inference_identity == second.inference_identity
    assert first.inference_identity != OpenAIProvider("http://other.local/v1", "key-two").inference_identity
    assert all(secret not in first.inference_identity for secret in ("secret", "private", "key-one", "example"))
    assert OllamaProvider("http://EXAMPLE.local:80/").inference_identity == OllamaProvider("http://example.local").inference_identity


@pytest.mark.asyncio
async def test_initial_router_status_failure_is_retryable_without_unsettled(monkeypatch, tmp_path):
    import httpx
    from test_llamacpp_lifecycle import configured
    events = []
    fail = False
    def handler(request):
        assert request.method == "GET"
        if fail:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        return httpx.Response(200, json={"data": [{"id": "own", "status": {"value": "unloaded"}}]})
    async with configured(monkeypatch, handler) as provider:
        handoff = ModelHandoff(tmp_path / "handoff.lock", provider, Ocr(events))
        await adopt(handoff)
        async with handoff.session(monotonic() + 1, model_id="own") as owner:
            owner.track("own")
            fail = True
            with pytest.raises(httpx.HTTPStatusError):
                await provider.release_model("own")
            assert json.loads(handoff.journal.read_bytes())["unsettled"] is False
            assert owner.dispatch_disposition == "not_dispatched"
            fail = False
            await owner.release_models()
            assert owner.models == []


@pytest.mark.asyncio
async def test_selected_model_is_checked_before_claim_and_inspection_preserves_generation(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    events.clear()
    with pytest.raises(HandoffWait):
        async with handoff.session(monotonic() + 1, model_id="other"):
            pytest.fail("unadopted selection claimed resources")
    assert events == []
    async with handoff.session(monotonic() + 1, model_id="own"):
        pass
    before = handoff.journal.read_bytes()
    events.clear()
    assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "ready"
    assert (await handoff.inspect(model_id="other", deadline=monotonic() + 1)).state == "configuration_required"
    assert handoff.journal.read_bytes() == before
    assert events == ["residency"]


@pytest.mark.asyncio
async def test_clean_journal_readoption_changes_only_explicit_model_set(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1):
        pass
    before = handoff.journal.read_bytes()
    binding = await adopt(handoff, models=("new-exact-id",))
    assert binding.model_ids == ["new-exact-id"]
    assert handoff.journal.read_bytes() == before
    assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "configuration_required"


@pytest.mark.asyncio
async def test_busy_inspect_and_adoption_do_not_change_existing_artifacts(tmp_path):
    import fcntl

    from vision_bill.model.handoff import HandoffWait
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm([]), Ocr([]))
    await adopt(handoff)
    before = handoff.binding_path.read_bytes()
    with handoff.path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "busy"
        with pytest.raises(HandoffWait) as caught:
            await adopt(handoff)
        assert caught.value.status.state == "busy"
    assert handoff.binding_path.read_bytes() == before
    assert not handoff.journal.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {}, {"models": [None]}, {"models": [{"name": "foreign"}]},
    {"models": [{"model": "foreign"}, {"model": "foreign"}]},
])
async def test_malformed_ollama_residency_cannot_prove_empty(body):
    import httpx
    from ollama import AsyncClient

    from vision_bill.provider.llm.ollama import OllamaProvider
    provider = OllamaProvider("http://local")
    provider._client = AsyncClient(host="http://local", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body),
    ))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            await provider.resident_model_ids(deadline=monotonic() + 1)
    finally:
        await provider._client._client.aclose()


@pytest.mark.asyncio
async def test_lost_router_unload_reply_preserves_owned_unknown_request(monkeypatch, tmp_path):
    import httpx
    from test_llamacpp_lifecycle import configured
    state = "unloaded"
    posts = []
    def handler(request):
        if request.method == "POST":
            posts.append(json.loads(request.content))
            raise httpx.ReadError("original unload reply lost")
        return httpx.Response(200, json={"data": [{"id": "own", "status": {"value": state}}]})
    async with configured(monkeypatch, handler) as provider:
        handoff = ModelHandoff(tmp_path / "handoff.lock", provider, Ocr([]))
        await adopt(handoff)
        with pytest.raises(httpx.ReadError, match="original unload reply lost"):
            async with handoff.session(monotonic() + 1, model_id="own") as owner:
                owner.track("own")
                state = "loaded"
                await provider.release_model("own")
        assert owner.dispatch_disposition == "unknown"
        saved = json.loads(handoff.journal.read_bytes())
        assert saved["unsettled"] is True
        assert saved["models"] == ["own"]
        assert posts == [{"model": "own"}]


@pytest.mark.asyncio
async def test_cli_status_and_adopt_use_settings_and_do_not_disclose_credentials(monkeypatch, tmp_path, capsys):
    import argparse

    from vision_bill.distribution.handoff import _run
    events = []
    llm, ocr = ResidentLlm(events), Ocr(events)
    ocr.managed_lifecycle = True
    monkeypatch.setattr("vision_bill.provider.factory.get_llm_provider", lambda config: llm)
    monkeypatch.setattr("vision_bill.provider.factory.get_ocr_provider", lambda config: ocr)
    args = argparse.Namespace(command="adopt", handoff_lock=tmp_path / "handoff.lock",
                              model_id=["own"], dedicated_coordinated_requests=True,
                              external_eager_loading_disabled=True)
    assert await _run(args) == 0
    args.command, args.model_id = "status", "own"
    assert await _run(args) == 0
    assert capsys.readouterr().out == '{"state":"ready","reason_code":null}\n' * 2
    assert events == ["residency", "residency"]


@pytest.mark.asyncio
async def test_provider_cannot_release_a_different_endpoint_under_another_owner(tmp_path):
    from unittest.mock import AsyncMock

    from ollama import ProcessResponse

    from vision_bill.provider.llm.ollama import OllamaProvider
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm([]), Ocr([]))
    await adopt(handoff)
    other = OllamaProvider("http://different-endpoint.local")
    other._client.generate = AsyncMock()
    other._client.ps = AsyncMock(return_value=ProcessResponse(models=[]))
    async with handoff.session(monotonic() + 1) as owner:
        owner.track("own")
        with pytest.raises(ValueError, match="model_handoff_identity_changed"):
            await other.release_model("own")
        other._client.generate.assert_not_awaited()


@pytest.mark.parametrize("field", ["version", "dedicated_coordinated_requests", "external_eager_loading_disabled"])
def test_binding_rejects_numeric_attestations_and_boolean_version(field):
    from pydantic import ValidationError

    from vision_bill.model.handoff import HandoffBinding
    data = {"version": 1, "coordination_path": "/tmp/test.lock", "resources": {"llm": "test", "ocr": "test"},
            "model_ids": ["own"], "dedicated_coordinated_requests": True, "external_eager_loading_disabled": True}
    data[field] = True if field == "version" else 1
    with pytest.raises(ValidationError):
        HandoffBinding.model_validate(data)


@pytest.mark.asyncio
async def test_cli_foreign_residency_asks_operator_to_quiesce_without_unloading(monkeypatch, tmp_path, capsys):
    import argparse

    from vision_bill.distribution.handoff import _run
    events = []
    llm, ocr = ResidentLlm(events), Ocr(events)
    llm.resident = frozenset({"foreign"})
    ocr.managed_lifecycle = True
    monkeypatch.setattr("vision_bill.provider.factory.get_llm_provider", lambda config: llm)
    monkeypatch.setattr("vision_bill.provider.factory.get_ocr_provider", lambda config: ocr)
    args = argparse.Namespace(command="adopt", handoff_lock=tmp_path / "handoff.lock", model_id=["own"],
                              dedicated_coordinated_requests=True, external_eager_loading_disabled=True)
    assert await _run(args) == 1
    output = capsys.readouterr()
    assert "quiesce" in output.err.lower()
    assert json.loads(output.out)["reason_code"] == "model_handoff_foreign_residency"
    assert events == ["residency"]


@pytest.mark.asyncio
async def test_later_unadopted_model_after_ocr_load_is_not_an_admission_wait(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1, model_id="own") as owner:
        await owner.ocr_phase()
        before = handoff.journal.read_bytes()
        with pytest.raises(RuntimeError, match="model_handoff_model_not_adopted") as caught:
            await owner.llm_phase("unadopted")
        assert not isinstance(caught.value, HandoffWait)
        assert handoff.journal.read_bytes() == before
        assert events.count("load_ocr") == 1
        assert ("unload", "unadopted") not in events


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence", ["failed", "foreign"])
async def test_residency_failure_after_ocr_load_is_not_an_admission_wait(tmp_path, evidence):
    from vision_bill.model.handoff import HandoffWait
    events = []
    llm = ResidentLlm(events)
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1, model_id="own") as owner:
        await owner.ocr_phase()
        before = handoff.journal.read_bytes()
        if evidence == "failed":
            async def failed_residency(*, deadline):
                raise ConnectionError("read-only evidence unavailable")
            llm.resident_model_ids = failed_residency
        else:
            llm.resident = frozenset({"foreign"})
        with pytest.raises(RuntimeError, match="model_handoff_.*residency") as caught:
            await owner.ocr_phase()
        assert not isinstance(caught.value, HandoffWait)
        assert handoff.journal.read_bytes() == before
        assert events.count("load_ocr") == 1
        assert all(not isinstance(event, tuple) for event in events)


@pytest_asyncio.fixture
async def ollama_client():
    provider = OllamaProvider("http://ollama.local")
    await provider._client._client.aclose()

    def use_transport(handler):
        provider._client = AsyncClient(host="http://ollama.local", transport=httpx.MockTransport(handler))
        return provider

    yield use_transport
    await provider._client._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_reply,ordinary_error", [(False, False), (True, False), (True, True)])
async def test_session_classifies_body_failure_after_cleanup_dispatch(tmp_path, ollama_client, lost_reply, ordinary_error):
    entered, finish = asyncio.Event(), asyncio.Event()
    unloads = []

    async def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        assert request.url.path == "/api/generate"
        unloads.append(json.loads(request.content))
        entered.set()
        await finish.wait()
        if lost_reply:
            raise httpx.ReadError("lost cleanup reply")
        return httpx.Response(200, json={})

    handoff = ModelHandoff(tmp_path / "handoff.lock", ollama_client(handler), Ocr([]))
    await adopt(handoff)
    original = ValueError("original body failure")

    async def run_session():
        async with handoff.session(monotonic() + 2, model_id="own") as owner:
            owner.track("own")
            if ordinary_error:
                raise original
            owner.track("unadopted")

    task = asyncio.create_task(run_session())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        pending = handoff.journal.read_bytes()
        assert json.loads(pending)["unsettled"] is True
        assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "busy"
        assert handoff.journal.read_bytes() == pending
    finally:
        finish.set()
        with pytest.raises(ValueError if ordinary_error else RuntimeError) as caught:
            await task

    assert [(call["model"], call["keep_alive"]) for call in unloads] == [("own", 0)]
    saved = json.loads(handoff.journal.read_bytes())
    assert saved["unsettled"] is lost_reply
    assert saved["models"] == (["own"] if lost_reply else [])
    status = await handoff.inspect(model_id="own", deadline=monotonic() + 1)
    assert status.state == ("recovery_required" if lost_reply else "ready")
    if ordinary_error:
        assert caught.value is original
    else:
        assert not isinstance(caught.value, HandoffWait)
        assert str(caught.value) == "model_handoff_model_not_adopted"
        assert isinstance(caught.value.__cause__, httpx.ReadError if lost_reply else HandoffWait)


@pytest.mark.parametrize("setting", ["LLM__API_KEY", "AUTH__SECRET_KEY"])
def test_cli_invalid_settings_subprocess_does_not_disclose_credentials(tmp_path, setting):
    marker = "vb-test-key-49"
    result = subprocess.run(
        [sys.executable, "-m", "vision_bill.distribution.handoff", "status", "--model-id", "own"],
        cwd=tmp_path, env={setting: marker, "VISION_BILL_CONFIG_PATH": str(tmp_path / "absent.yaml")},
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert marker not in result.stdout + result.stderr
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"state": "recovery_required", "reason_code": "model_handoff_setup_failed"}
    assert result.stderr == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact,nested", [("journal", False), ("binding", False), ("journal", True), ("binding", True)])
async def test_duplicate_handoff_members_fence_all_readers_and_preserve_bytes(tmp_path, artifact, nested):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1, model_id="own"):
        pass
    events.clear()
    path = handoff.journal if artifact == "journal" else handoff.binding_path
    members = path.read_bytes()
    if nested:
        ambiguous = members.replace(b'"llm"', b'"l\\u006cm":"other", "llm"', 1)
    elif artifact == "journal":
        ambiguous = members.replace(b'"unsettled"', b'"unsettled":true, "unsettled"', 1)
    else:
        ambiguous = members.replace(b'"dedicated_coordinated_requests"',
                                    b'"dedicated_coordinated_requests":false, "dedicated_coordinated_requests"', 1)
    assert ambiguous != members
    path.write_bytes(ambiguous)
    before = [p.read_bytes() for p in (handoff.journal, handoff.binding_path)]

    assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "recovery_required"
    with pytest.raises((ValueError, HandoffWait)):
        async with handoff.session(monotonic() + 1, model_id="own"):
            pytest.fail("ambiguous handoff authorized session entry")
    with pytest.raises((ValueError, HandoffWait)):
        await adopt(handoff, models=("another",))
    assert events == []
    assert [p.read_bytes() for p in (handoff.journal, handoff.binding_path)] == before


@pytest.mark.asyncio
async def test_session_lock_deadline_is_typed_busy(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    events.clear()
    with handoff.path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(HandoffWait) as caught:
            async with handoff.session(monotonic() + .03, model_id="own"):
                pytest.fail("busy lock authorized session entry")
    assert caught.value.status.state == "busy"
    assert events == []
    assert not handoff.journal.exists()


@pytest.mark.asyncio
async def test_session_lock_preserves_external_cancellation(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    events.clear()

    async def acquire():
        async with handoff.session(monotonic() + 5, model_id="own"):
            pytest.fail("busy lock authorized session entry")

    with handoff.path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        waiter = asyncio.create_task(acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    assert events == []
    assert not handoff.journal.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("resident_tag", ["latest", "foreign"])
async def test_ollama_restart_recovers_only_recorded_implicit_latest(tmp_path, ollama_client, resident_tag):
    selected = "registry.local:5000/team/own"
    resident, unloads = [], []

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"model": model} for model in resident]})
        assert request.url.path == "/api/generate"
        unloads.append(json.loads(request.content)["model"])
        resident.clear()
        return httpx.Response(200, json={})

    events = []
    provider = ollama_client(handler)
    handoff = ModelHandoff(tmp_path / "handoff.lock", provider, Ocr(events))
    await adopt(handoff, models=(selected,))
    handoff.journal.write_text(json.dumps({
        "version": 2, "coordinator": "vision-bill-model-handoff-v2",
        "resources": handoff.resources.model_dump(), "models": [selected],
        "generation": 8, "unsettled": False, "ocr_pending": False,
    }))
    resident.append(f"{selected}:{resident_tag}")
    before = handoff.journal.read_bytes()
    assert (await handoff.inspect(model_id=selected + ":latest", deadline=monotonic() + 1)).state == "configuration_required"
    status = await handoff.inspect(model_id=selected, deadline=monotonic() + 1)
    assert handoff.journal.read_bytes() == before
    assert unloads == events == []
    if resident_tag == "latest":
        assert status.state == "ready"
        async with handoff.session(monotonic() + 1, model_id=selected) as owner:
            await owner.ocr_phase()
        assert unloads == [selected]
        assert "load_ocr" in events
        assert json.loads(handoff.journal.read_bytes())["models"] == []
    else:
        assert status.reason_code == "model_handoff_foreign_residency"
        with pytest.raises(HandoffWait):
            async with handoff.session(monotonic() + 1, model_id=selected):
                pytest.fail("foreign resident authorized session entry")
        assert handoff.journal.read_bytes() == before
        assert unloads == events == []


@pytest.mark.asyncio
async def test_unconfirmed_cleanup_cannot_escape_as_refundable_wait(tmp_path):
    from unittest.mock import AsyncMock

    events = []
    ocr = Ocr(events)
    ocr.park = AsyncMock(side_effect=RuntimeError("cleanup uncertain"))
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), ocr)
    await adopt(handoff)
    with pytest.raises(RuntimeError, match="model_handoff_model_not_adopted") as caught:
        async with handoff.session(monotonic() + 1) as owner:
            owner.require_model("not-adopted")
    assert not isinstance(caught.value, HandoffWait)
    assert str(caught.value.__cause__) == "cleanup uncertain"


async def test_recovery_session_confirms_owned_cleanup_before_publication(tmp_path):
    from vision_bill.service.gpu_handoff import HandoffOwner

    events = []
    llm = ResidentLlm(events)
    ocr = Ocr(events)

    async def release(model):
        events.append(("unload", model))
        llm.resident = frozenset()

    async def unavailable_capabilities():
        pytest.fail("Completion must not discover OCR capabilities")

    llm.release_model = release
    ocr.get_capabilities = unavailable_capabilities
    handoff = ModelHandoff(tmp_path / "completion.lock", llm, ocr)
    binding = await adopt(handoff)
    producer = HandoffOwner(handoff, monotonic() + 1)
    producer.models = ["own"]
    producer.save()
    llm.resident = frozenset({"own"})
    events.clear()
    async with handoff.recover_checkpoint(
        monotonic() + 1, model_id="own", expected_binding_sha256=handoff.binding_sha256(binding),
        before_dispatch=lambda: asyncio.sleep(0),
    ):
        events.append("publish")
        saved = handoff.read_journal()
        assert not saved.models and not saved.ocr_pending and not saved.unsettled
    assert events == ["residency", "claim_park", ("unload", "own"), "residency", "publish"]


@pytest.mark.asyncio
async def test_session_allows_metadata_slower_than_the_contended_lock_window(tmp_path):
    from vision_bill.service.gpu_handoff import admission_deadline

    events = []
    llm = ResidentLlm(events)
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, Ocr(events))
    await adopt(handoff)
    resident_models = llm.resident_model_ids

    async def slow_residency(*, deadline):
        assert deadline - monotonic() > 0.25
        await asyncio.sleep(0.3)
        return await resident_models(deadline=deadline)

    llm.resident_model_ids = slow_residency
    deadline = monotonic() + 2
    async with handoff.session(
        deadline, model_id="own", lock_deadline=admission_deadline(deadline)
    ):
        assert handoff.journal.exists()


def test_ollama_inference_identity_follows_effective_sdk_ports_without_rebinding_lifecycle():
    from vision_bill.provider.llm.ollama import OllamaProvider

    bare = OllamaProvider("localhost")
    explicit = OllamaProvider("http://localhost:11434/")
    standard = OllamaProvider("http://localhost")
    assert str(bare._client._client.base_url) == "http://localhost:11434"
    assert str(standard._client._client.base_url) == "http://localhost"
    assert bare.inference_identity == explicit.inference_identity
    assert bare.inference_identity != standard.inference_identity
    assert bare.lifecycle_identity != explicit.lifecycle_identity
