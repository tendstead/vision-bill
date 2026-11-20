from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from vision_bill.model.experiment import ExperimentOptions
from vision_bill.model.workflow import AccessScope
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.ocr.base import compact_ocr_text


def test_empty_primary_text_uses_recognized_blocks():
    text = compact_ocr_text({"text": "", "blocks": [{"text": "Mate 12.90"}]})
    assert text == "Mate 12.90"


def test_layout_pixels_are_not_sent_to_llm():
    with pytest.raises(ValueError, match="recognized text"):
        compact_ocr_text({"layout_info": {"input_img": [[[255, 255, 255]]]}})


def test_experiment_provider_settings_do_not_mutate_live_provider():
    live = OllamaProvider("http://localhost:11434", temperature=0.7)
    probe = live.for_experiment(ExperimentOptions(temperature=0, include_classification=False))
    assert live._temperature == 0.7
    assert probe._temperature == 0
    prompt = probe.build_prompt(["deposit", "food"])
    assert "spending_category" not in prompt
    assert "Tag every deposit" not in prompt
    assert "spending_category" in live.build_prompt(["deposit", "food"])


async def test_standalone_qualifies_with_production_llamacpp_binding(settings, tmp_path, monkeypatch):
    import tempfile
    from argparse import Namespace
    from pathlib import Path
    from time import monotonic

    from scripts import standalone_benchmark as script
    from vision_bill import config
    from vision_bill.distribution.handoff import adopt_binding
    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.provider.llm.openai import OpenAIProvider
    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider
    from vision_bill.service.gpu_handoff import ModelHandoff

    llm_settings = config.LLMSettings(
        provider="openai",
        host="http://router.test/v1",
        api_key="fixture-key",
        model_name="selected",
        temperature=0.37,
        model_lifecycle="llamacpp",
        max_output_tokens=2234,
        analysis_timeout_seconds=42.5,
    )
    ocr_settings = config.OcrSettings(
        enabled=True, service_url="http://ocr.test:8082", coordination_path=tmp_path / "shared.lock",
    )
    monkeypatch.setattr(
        config, "settings", settings.model_copy(update={"llm": llm_settings, "ocr": ocr_settings}),
    )
    monkeypatch.setenv("LLM__PROVIDER", "openai")
    monkeypatch.setenv("LLM__HOST", llm_settings.host)
    monkeypatch.setenv("LLM__MODEL_NAME", llm_settings.model_name)
    monkeypatch.setenv("OCR__SERVICE_URL", ocr_settings.service_url)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    real_mkdir = Path.mkdir
    image_dirs = set()

    def host_mkdir(path, *args, **kwargs):
        if path.is_relative_to("/app"):
            raise PermissionError(f"Host CLI cannot write to {path}")
        real_mkdir(path, *args, **kwargs)
        if path.is_relative_to(scratch):
            image_dirs.add(path)

    monkeypatch.setattr(Path, "mkdir", host_mkdir)

    monkeypatch.setattr(OpenAIProvider, "get_available_models", AsyncMock(
        return_value=[ModelInfo("selected"), ModelInfo("unadopted")],
    ))
    monkeypatch.setattr(OpenAIProvider, "resident_model_ids", AsyncMock(return_value=frozenset()))
    monkeypatch.setattr(OpenAIProvider, "release_model", AsyncMock())
    monkeypatch.setattr(PaddleOcrProvider, "_lifecycle", AsyncMock())
    monkeypatch.setattr(PaddleOcrProvider, "get_capabilities", AsyncMock(return_value=OcrCapabilities(
        enabled=True, configured=True, reachable=True, ready=True,
        profile_id="profile", runtime_fingerprint="runtime", ocr_max_new_tokens=4096,
    )))

    application_llm = script.get_llm_provider(llm_settings)
    application_ocr = script.get_ocr_provider(ocr_settings)
    handoff = ModelHandoff(ocr_settings.coordination_path, application_llm, application_ocr)
    args = Namespace(model=None, llm_host=None, ocr_url=None)
    cli_settings = script.get_llm_settings(args)
    cli_llm = script.get_llm_provider(cli_settings)
    cli_ocr_settings = script.get_ocr_settings(args)
    cli_ocr = script.get_ocr_provider(cli_ocr_settings)
    try:
        await adopt_binding(
            handoff, ["selected"], dedicated_coordinated_requests=True,
            external_eager_loading_disabled=True, deadline=monotonic() + 5,
        )
        binding_bytes = handoff.binding_path.read_bytes()
        identity = await script.execution_identity(
            cli_llm, cli_ocr, settings.pg, cli_ocr_settings, cli_settings.model_name, qualify=True,
        )
        assert image_dirs and not list(scratch.iterdir())
        with pytest.raises(ValueError, match="formatter_model_unavailable"):
            await script.execution_identity(
                cli_llm, cli_ocr, settings.pg, cli_ocr_settings, "missing-model", qualify=True,
            )
        assert list(scratch.iterdir()) == []
    finally:
        await application_llm._client.close()
        await cli_llm._client.close()

    assert identity["requested_model"]["id"] == "selected"
    assert identity["pipeline"]["formatter_output_tokens"] == 2234
    assert identity["pipeline"]["analysis_timeout_seconds"] == 42.5
    assert identity["pipeline"]["ocr_runtime"]["runtime_fingerprint"] == "runtime"
    assert handoff.binding_path.read_bytes() == binding_bytes
    journal = handoff.read_journal()
    assert journal is not None and journal.models == [] and not journal.ocr_pending


async def test_thinking_override_is_sent_as_top_level_option():
    live = OllamaProvider("http://localhost:11434", temperature=0.7)
    probe = live.for_experiment(ExperimentOptions(think=False))
    response = AsyncMock()
    response.message.content = "{}"
    probe._client.chat = AsyncMock(return_value=response)
    assert await probe.send_message("model", [{"role": "user", "content": "read"}]) == "{}"
    kwargs = probe._client.chat.call_args.kwargs
    assert kwargs["think"] is False
    assert "think" not in kwargs["options"]


async def test_upload_execution_preserves_all_photo_paths_and_frozen_tags(settings, tmp_path):
    import json

    from ollama import ChatResponse
    from test_experiment_scoring import receipt

    from vision_bill.service.experiment_execution import execute_submission
    from vision_bill.service.receipt_service import ReceiptService

    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    from PIL import Image

    from vision_bill.provider.llm.base import ModelInfo

    for path, color in ((first, "red"), (second, "blue")):
        Image.new("RGB", (5, 7), color).save(path, format="JPEG")
    provider = OllamaProvider("http://localhost:11434", temperature=0.7)
    available = ModelInfo("model")
    available.capabilities = ["vision", "completion"]
    provider.get_available_models = AsyncMock(return_value=[available])
    payload = receipt(("Milk",)).model_dump(mode="json")
    del payload["line_items"][0]["tags"]
    payload["line_items"][0]["source_image"] = 0
    payload["duplicate_candidates"] = []
    provider._client.chat = AsyncMock(
        side_effect=[
            ChatResponse(
                message={"role": "assistant", "content": json.dumps(payload)},
                eval_count=10,
                eval_duration=20_000_000,
            ),
            ChatResponse(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "category": "grocery",
                            "line_items": [{"spending_category": "milk", "tags": ["frozen_tag"]}],
                        }
                    ),
                }
            ),
        ]
    )
    service = ReceiptService(settings.images, settings.pg, provider)
    result = await execute_submission(
        service, "model", [first, second], ExperimentOptions(temperature=0), ["frozen_tag"]
    )
    assert result.receipt.line_items[0].description == "Milk"
    calls = provider._client.chat.call_args_list
    request = calls[0].kwargs
    assert len(request["messages"][0]["images"]) == 2
    assert "frozen_tag" not in request["messages"][0]["content"]
    assert "frozen_tag" in calls[1].kwargs["messages"][0]["content"]
    assert result.receipt.line_items[0].tags == ["frozen_tag"]
    assert result.receipt.category == "other"
    assert result.metadata["classification_enrichment"]["state"] == "completed"
    assert request["options"]["temperature"] == 0
    assert provider._temperature == 0.7
    assert result.metadata["image_count"] == 2
    assert result.metadata["calls"][0]["eval_ms"] == 20


@pytest.mark.parametrize("explicit_null", [False, True])
async def test_experiment_rejects_unobserved_taxes_before_scoring(settings, tmp_path, explicit_null):
    import json

    from PIL import Image
    from test_experiment_scoring import receipt
    from test_extraction_pipeline import formatter

    from vision_bill.service.experiment_execution import ExperimentFailure, execute_submission
    from vision_bill.service.receipt_service import ReceiptService

    path = tmp_path / "receipt.png"
    Image.new("RGB", (3, 4)).save(path)
    provider = formatter()
    payload = receipt().model_dump(mode="json")
    if explicit_null:
        payload["taxes"] = None
    else:
        del payload["taxes"]
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    service = ReceiptService(settings.images, settings.pg, provider)
    with pytest.raises(ExperimentFailure, match="draft_incomplete") as exc:
        await execute_submission(service, "fixture", [path], ExperimentOptions(), [])
    assert exc.value.metadata["draft"]["taxes"] is None
    assert exc.value.metadata["classification_enrichment"]["reason_code"] == "incomplete_observations"
    assert provider.send_message.await_count == 1


async def test_forced_ocr_is_not_silently_replaced_with_vision(settings, tmp_path, monkeypatch):
    from vision_bill.service.experiment_execution import execute_submission
    from vision_bill.service.receipt_service import ReceiptService

    path = tmp_path / "receipt.jpg"
    from PIL import Image
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.provider import factory

    Image.new("RGB", (3, 4)).save(path)
    ocr = sidecar()
    ocr.get_capabilities.return_value.ready = False
    monkeypatch.setattr(factory, "get_ocr_provider", lambda settings: ocr)
    service = ReceiptService(settings.images, settings.pg, formatter())
    with pytest.raises(ValueError, match="ocr_unavailable"):
        await execute_submission(service, "model", [path], ExperimentOptions(pipeline="ocr"), [])


def test_ocr_fingerprint_detects_endpoint_and_timeout_changes():
    from types import SimpleNamespace

    from vision_bill.service.experiment_execution import ocr_fingerprint

    provider = SimpleNamespace(service_url="http://ocr-one:8080", timeout=300)
    original = ocr_fingerprint(provider)
    assert original == ocr_fingerprint(provider)
    provider.service_url = "http://ocr-two:8080"
    assert ocr_fingerprint(provider) != original
    provider.service_url = "http://ocr-one:8080"
    provider.timeout = 600
    assert ocr_fingerprint(provider) != original
    assert ocr_fingerprint(None) is None


@pytest.mark.parametrize("current_digest", ["replacement", None, "missing_model"])
async def test_reevaluation_rejects_changed_or_missing_model(current_digest):
    from types import SimpleNamespace
    from uuid import uuid4

    from vision_bill.service.benchmark_service import BenchmarkService

    run_id, receipt_id = uuid4(), uuid4()
    run = SimpleNamespace(receipt_ids=[receipt_id], model_ids=["model"])
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(
        run=AsyncMock(return_value=run), manifest=AsyncMock(return_value={"options": {}})
    )
    current = SimpleNamespace(id="model", digest=current_digest)
    service._provider = SimpleNamespace(
        get_available_models=AsyncMock(return_value=[] if current_digest == "missing_model" else [current])
    )
    service._run_digests = AsyncMock(return_value={"model": "original"})
    service._run_experiment = AsyncMock()
    with pytest.raises(ValueError, match="digest changed"):
        await service.reevaluate(run_id, receipt_id, "model", scope=AccessScope(user_id=uuid4()))
    service._run_experiment.assert_not_awaited()


async def test_reevaluation_enforces_saved_run_deadline():
    import asyncio
    from types import SimpleNamespace
    from uuid import uuid4

    from vision_bill.service.benchmark_service import BenchmarkService

    run_id, receipt_id = uuid4(), uuid4()
    run = SimpleNamespace(receipt_ids=[receipt_id], model_ids=["model"])
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(
        run=AsyncMock(return_value=run), manifest=AsyncMock(return_value={"options": {}})
    )
    current = SimpleNamespace(id="model", digest="original")
    service._provider = SimpleNamespace(get_available_models=AsyncMock(return_value=[current]))
    service._run_digests = AsyncMock(return_value={"model": "original"})
    service._timeout = AsyncMock(return_value=0.01)

    async def slow(*args, **kwargs):
        await asyncio.Event().wait()

    service._run_experiment = slow
    with pytest.raises(TimeoutError):
        await service.reevaluate(run_id, receipt_id, "model", scope=AccessScope(user_id=uuid4()))


async def test_untyped_replay_cannot_bypass_complete_envelope(settings, tmp_path):
    from types import SimpleNamespace

    from PIL import Image

    from vision_bill.service.experiment_execution import ExperimentFailure, execute_submission
    from vision_bill.service.receipt_service import ReceiptService

    path = tmp_path / "receipt.png"
    Image.new("RGB", (3, 4)).save(path)
    provider = OllamaProvider("http://localhost:11434")
    provider._client.chat = AsyncMock()
    service = ReceiptService(settings.images, settings.pg, provider)
    service._ocr_provider = SimpleNamespace(extract_text=AsyncMock(return_value={"text": "old"}))
    with pytest.raises(ExperimentFailure):
        await execute_submission(service, "model", [path], ExperimentOptions(pipeline="ocr"), [])
    provider._client.chat.assert_not_awaited()


async def test_typed_experiment_ocr_uses_complete_photo_pipeline(settings, tmp_path):
    import time

    from PIL import Image
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.service.experiment_execution import execute_submission
    from vision_bill.service.receipt_service import ReceiptService

    paths = [tmp_path / f"part-{i}.png" for i in range(2)]
    for path in paths:
        Image.new("RGB", (4, 5)).save(path)
    llm, ocr = formatter(False), sidecar()
    service = ReceiptService(settings.images, settings.pg, llm, ocr_provider=ocr)
    result = await execute_submission(
        service,
        "fixture",
        paths,
        ExperimentOptions(pipeline="ocr", include_classification=False),
        [],
    )
    assert result.metadata["used_pipeline"] == "ocr"
    assert result.metadata["image_count"] == 2
    assert ocr.extract_page.await_count == 2
    assert [call.kwargs["photo_index"] for call in ocr.extract_page.await_args_list] == [0, 1]
    assert all(
        call.kwargs["deadline"] > time.monotonic() - 1 for call in ocr.extract_page.await_args_list
    )


async def test_effective_pipeline_identity_binds_profile_runtime_budgets_and_fallback(settings):
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.service import experiment_execution as execution
    from vision_bill.service.receipt_service import ReceiptService

    service = ReceiptService(settings.images, settings.pg, formatter(), ocr_provider=sidecar())
    assert hasattr(execution, "pipeline_identity"), "effective pipeline identity missing"
    options = ExperimentOptions(pipeline="ocr", include_classification=False)
    original = await execution.pipeline_identity(service, options)
    caps = service._ocr_provider.get_capabilities.return_value
    caps.profile_id = "different"
    assert await execution.pipeline_identity(service, options) != original
    caps.profile_id = "fixed-profile"
    caps.runtime_fingerprint = "changed"
    assert await execution.pipeline_identity(service, options) != original
    caps.runtime_fingerprint = None
    service._provider.max_output_tokens += 1
    assert await execution.pipeline_identity(service, options) != original
    assert (
        await execution.pipeline_identity(
            service, options.model_copy(update={"allow_ocr_fallback": True})
        )
        != original
    )


@pytest.mark.parametrize("requested", [["fixture"], ["second", "fixture"]])
async def test_run_manifest_preserves_upload_policy_instead_of_forcing_configured_ocr(
    settings, requested,
):
    from types import SimpleNamespace

    from test_experiment_scoring import receipt
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.service.benchmark_service import BenchmarkService

    service = object.__new__(BenchmarkService)
    service._provider = formatter()
    from vision_bill.provider.llm.base import ModelInfo

    fixture = service._provider.get_available_models.return_value[0]
    second = ModelInfo("second")
    second.capabilities = ["vision"]
    service._provider.get_available_models.return_value = [fixture, second]
    service._receipts = SimpleNamespace(
        _provider=service._provider,
        _ocr_provider=sidecar(),
        _ocr_settings=OcrSettings(enabled=False),
        list_tags=AsyncMock(return_value=[]),
        execution_handoff=lambda _: None,
    )
    rid = uuid4()
    service._select_receipts = AsyncMock(return_value=[rid])
    service._capture_input = AsyncMock(
        return_value={
            "receipt_id": str(rid),
            "expected": receipt().model_dump(mode="json"),
            "images": [{"path": "source", "sha256": "a" * 64}],
        }
    )
    service._db = SimpleNamespace(create_run=AsyncMock())
    await service.create_run(
        BenchmarkCreate(model_ids=requested), scope=AccessScope(user_id=uuid4())
    )
    recorded = service._db.create_run.call_args.kwargs
    assert recorded["model_ids"] == requested
    assert list(recorded["model_digests"]) == requested
    manifest = recorded["manifest"]
    assert manifest["inputs"][0]["pipeline"] == "upload"
    assert manifest["identity"]["ocr_settings"]["enabled"] is False


async def test_ocr_benchmark_discovers_text_only_formatter_models(settings):
    from types import SimpleNamespace

    from test_experiment_scoring import receipt
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.service.benchmark_service import BenchmarkService

    service = object.__new__(BenchmarkService)
    service._provider = formatter(False)
    model = (await service._provider.get_available_models())[0]

    async def models(*, capability="vision"):
        return [model] if capability == "text" else []

    service._provider.get_available_models = models
    service._receipts = SimpleNamespace(
        _provider=service._provider,
        _ocr_provider=sidecar(),
        _ocr_settings=OcrSettings(enabled=True, manage_model_lifecycle=False),
        list_tags=AsyncMock(return_value=[]),
        execution_handoff=lambda _: None,
    )
    rid = uuid4()
    service._select_receipts = AsyncMock(return_value=[rid])
    service._capture_input = AsyncMock(
        return_value={
            "receipt_id": str(rid),
            "expected": receipt().model_dump(mode="json"),
            "images": [{"path": "source", "sha256": "a" * 64}],
        }
    )
    service._db = SimpleNamespace(create_run=AsyncMock())
    await service.create_run(
        BenchmarkCreate(model_ids=["fixture"], experiment=ExperimentOptions(pipeline="ocr")),
        scope=AccessScope(user_id=uuid4()),
    )
    assert service._db.create_run.call_args.kwargs["model_ids"] == ["fixture"]


@pytest.mark.parametrize(
    "intent,enabled,fail_second,expected",
    [
        ("upload", False, False, "direct_vision"),
        ("upload", True, False, "ocr"),
        ("upload", True, True, None),
        ("direct_vision", True, False, "direct_vision"),
        ("ocr", False, False, "ocr"),
        ("ocr", False, True, None),
    ],
)
async def test_serialized_experiment_intent_preserves_effective_ocr_policy(
    settings, tmp_path, monkeypatch, intent, enabled, fail_second, expected
):
    from PIL import Image
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.provider import factory
    from vision_bill.service.experiment_execution import (
        ExperimentFailure,
        execute_submission,
        pipeline_identity,
    )
    from vision_bill.service.receipt_service import ReceiptService

    paths = [tmp_path / f"{i}.png" for i in range(2)]
    for path in paths:
        Image.new("RGB", (3, 4)).save(path)
    ocr = sidecar(1 if fail_second else None)
    monkeypatch.setattr(factory, "get_ocr_provider", lambda settings: ocr)
    service = ReceiptService(
        settings.images,
        settings.pg,
        formatter(),
        ocr_provider=ocr if enabled else None,
        ocr_settings=OcrSettings(enabled=enabled, required=False, allow_vision_fallback=True),
    )
    # Exactly the JSON options serialized in a manifest and reconstructed for replay.
    original = ExperimentOptions(
        pipeline=intent, include_classification=False, allow_ocr_fallback=True
    )
    options = ExperimentOptions.model_validate_json(original.model_dump_json())
    identity = await pipeline_identity(service, options)
    if expected is None:
        with pytest.raises(ExperimentFailure):
            await execute_submission(service, "fixture", paths, options, [])
        service._provider._client.chat.assert_not_awaited()
    else:
        result = await execute_submission(service, "fixture", paths, options, [])
        assert result.metadata["used_pipeline"] == expected
        assert result.metadata["fallback_used"] is False
        if intent == "direct_vision":
            assert len(service._provider._client.chat.call_args.kwargs["messages"][0]["images"]) == 2
    if intent == "direct_vision" or (intent == "upload" and not enabled):
        ocr.extract_page.assert_not_awaited()
    if intent == "ocr":
        assert identity["ocr_settings"]["enabled"] and identity["ocr_settings"]["required"]
        assert not identity["ocr_settings"]["allow_vision_fallback"]
    assert service._ocr_settings.enabled is enabled and not service._ocr_settings.required


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionRefusedError("database reconnect failed"),
        TimeoutError("database acquisition timed out"),
    ],
)
async def test_reference_selection_propagates_infrastructure_errors(failure):
    from types import SimpleNamespace

    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.service.benchmark_service import BenchmarkService

    candidates = [uuid4(), uuid4()]
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(
        reference_candidates=AsyncMock(return_value=candidates),
        current_reference=AsyncMock(side_effect=[failure, object()]),
    )
    with pytest.raises(type(failure), match=str(failure)):
        await service._select_receipts(BenchmarkCreate(model_ids=['fixture'], limit=1), scope=AccessScope(user_id=uuid4()))
    assert service._db.current_reference.await_count == 1


@pytest.mark.parametrize("global_permission", [False, True])
@pytest.mark.parametrize("permission", [None, False, True])
def test_upload_experiment_does_not_invent_frozen_fallback_permission(
    global_permission, permission
):
    from types import SimpleNamespace

    from vision_bill.config import OcrSettings
    from vision_bill.service.experiment_execution import effective_policy

    service = SimpleNamespace(
        _ocr_settings=OcrSettings(enabled=True, allow_vision_fallback=global_permission),
        _ocr_provider=object(),
    )
    options = ExperimentOptions(
        pipeline="upload", **({} if permission is None else {"allow_ocr_fallback": permission})
    )
    before = options.model_dump()
    _, _, effective = effective_policy(service, options)
    assert effective.allow_ocr_fallback is False
    assert options.model_dump() == before


async def test_pipeline_identity_ignores_unused_legacy_alternate_digest(settings):
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.experiment_execution import pipeline_identity
    from vision_bill.service.receipt_service import ReceiptService
    provider = formatter(False)
    alternate = ModelInfo("alternate")
    alternate.digest = "first"
    alternate.capabilities = ["vision"]
    provider.get_available_models.return_value.append(alternate)
    service = ReceiptService(
        settings.images, settings.pg, provider, ocr_provider=sidecar(),
        ocr_settings=OcrSettings(enabled=True, allow_vision_fallback=True,
                                 vision_fallback_model_id="alternate"),
    )
    options = ExperimentOptions(pipeline="upload", allow_ocr_fallback=True)
    first = await pipeline_identity(service, options)
    alternate.digest = "changed"
    second = await pipeline_identity(service, options)
    assert first == second
    assert "vision_fallback_model" not in first
    provider.get_available_models.assert_not_awaited()


async def test_unmanifested_benchmark_reevaluation_requires_explicit_new_run():
    from types import SimpleNamespace

    from vision_bill.service.benchmark_service import BenchmarkService

    run_id, receipt_id = uuid4(), uuid4()
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(receipt_ids=[receipt_id], model_ids=["model"])),
        manifest=AsyncMock(return_value=None),
    )
    service._provider = SimpleNamespace(analyse_receipt_with_metadata=AsyncMock())
    service._expected_receipt_and_path = AsyncMock(side_effect=AssertionError("must not infer"))
    with pytest.raises(ValueError, match="create a new run"):
        await service.reevaluate(run_id, receipt_id, "model", scope=AccessScope(user_id=uuid4()))
    service._provider.analyse_receipt_with_metadata.assert_not_awaited()


async def test_unmanifested_benchmark_worker_terminalizes_without_inference(settings):
    import asyncio
    from types import SimpleNamespace

    from vision_bill.service.benchmark_service import BenchmarkService

    service = object.__new__(BenchmarkService)
    service._settings = settings.worker
    service._db = SimpleNamespace(finish_task=AsyncMock(), retry_task=AsyncMock())
    service._provider = SimpleNamespace(get_available_models=AsyncMock(),
                                        analyse_receipt_with_metadata=AsyncMock())
    task = {"inference_attempt_count": 0, "inference_started": False}
    started = asyncio.get_running_loop().time()
    await service._execute_reserved(task, SimpleNamespace(), None, started, started + 600)
    service._db.finish_task.assert_awaited_once_with(task, success=False, deadline=started + 600)
    service._db.retry_task.assert_not_awaited()
    service._provider.get_available_models.assert_not_awaited()
    service._provider.analyse_receipt_with_metadata.assert_not_awaited()


@pytest.mark.parametrize(
    "change", ["rebind", "rebind_and_timeout", "llm_endpoint", "ocr_endpoint", "ocr_timeout", "standalone"]
)
async def test_managed_benchmark_distinguishes_topology_from_frozen_client_drift(
    settings, tmp_path, monkeypatch, change,
):
    import hashlib
    import json
    from time import monotonic

    from PIL import Image
    from test_experiment_scoring import receipt
    from test_extraction_pipeline import formatter, sidecar
    from test_handoff_binding import adopt

    from vision_bill import config
    from vision_bill.config import OcrSettings
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.model.receipt import Receipt
    from vision_bill.provider.llm.experiment import endpoint_fingerprint
    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.service.experiment_execution import (
        image_digest,
        ocr_fingerprint,
        pipeline_identity,
    )
    from vision_bill.service.gpu_handoff import ModelHandoff
    from vision_bill.service.receipt_service import ReceiptService

    settings.ocr = OcrSettings(
        enabled=True, required=True, service_url="http://fixture.invalid",
        coordination_path=tmp_path / "handoff.lock",
    )
    monkeypatch.setattr(config, "settings", settings)
    llm, ocr = formatter(), sidecar()
    ocr.managed_lifecycle = True
    ocr.service_url, ocr.timeout = settings.ocr.service_url, settings.ocr.timeout_seconds
    ocr.lifecycle_identity = PaddleOcrProvider(ocr.service_url).lifecycle_identity
    ocr.get_capabilities.return_value.runtime_fingerprint = "cpu-runtime"
    ocr.claim, ocr.park, ocr.load = AsyncMock(), AsyncMock(), AsyncMock()
    llm.handoff = ModelHandoff(settings.ocr.coordination_path, llm, ocr)
    await adopt(llm.handoff, ("fixture",))
    service = ReceiptService(settings.images, settings.pg, llm, ocr, settings.ocr)
    benchmark = object.__new__(BenchmarkService)
    benchmark._provider = llm
    benchmark._receipts = service

    photo = tmp_path / "original.png"
    Image.new("RGB", (3, 4)).save(photo)
    receipt_id = uuid4()
    options = ExperimentOptions(pipeline="ocr", include_classification=False)
    manifest = {
        "provider": type(llm).__name__,
        "endpoint_fingerprint": endpoint_fingerprint(llm),
        "ocr_fingerprint": ocr_fingerprint(ocr),
        "validation_schema_sha256": hashlib.sha256(
            json.dumps(Receipt.model_json_schema(), sort_keys=True).encode()
        ).hexdigest(),
        "inputs": [{
            "receipt_id": str(receipt_id),
            "pipeline": "ocr",
            "expected": receipt().model_dump(mode="json"),
            "images": [{"path": str(photo), "sha256": image_digest(photo)}],
        }],
        "options": options.model_dump(mode="json"),
        "identity": await pipeline_identity(service, options),
        "tags": [],
        "prompt_hashes": {"1": hashlib.sha256(
            llm.for_experiment(options).build_prompt([], image_count=1).encode()
        ).hexdigest()},
    }
    if change in ("ocr_timeout", "rebind_and_timeout"):
        ocr.timeout += 1
        service._ocr_settings = settings.ocr.model_copy(update={"timeout_seconds": ocr.timeout})
        assert llm.handoff.binding_sha256() == manifest["identity"]["managed_binding_sha256"]
    if change in ("rebind", "rebind_and_timeout"):
        inspect = ModelHandoff.inspect

        async def inspect_then_readopt(coordinator, **kwargs):
            status = await inspect(coordinator, **kwargs)
            assert status.state == "ready"
            await adopt(coordinator, ("fixture", "replacement"))
            return status

        monkeypatch.setattr(ModelHandoff, "inspect", inspect_then_readopt)
    elif change == "llm_endpoint":
        llm._client._client.base_url = "http://replacement.invalid"
    elif change == "ocr_endpoint":
        ocr.service_url = "http://ocr-replacement.invalid"
        ocr.lifecycle_identity = PaddleOcrProvider(ocr.service_url).lifecycle_identity
        service._ocr_settings = settings.ocr.model_copy(update={"service_url": ocr.service_url})
    elif change == "standalone":
        service._ocr_settings = settings.ocr.model_copy(update={"manage_model_lifecycle": False})
        ocr.managed_lifecycle = False
        llm.handoff = None

    begin = AsyncMock()
    expected_exception, expected_reason = (
        (ValueError, "OCR provider endpoint or client settings changed")
        if change == "ocr_timeout" else (HandoffWait, "execution_policy_changed")
    )
    with pytest.raises(expected_exception, match=expected_reason) as caught:
        await benchmark._run_experiment(
            manifest, receipt_id, "fixture", before_inference=begin,
            inference_deadline=monotonic() + 2, managed_model_ids=["fixture"],
        )
    if change == "ocr_timeout":
        assert not benchmark._retryable(caught.value)
    if change in ("rebind", "rebind_and_timeout"):
        assert llm.handoff.read_binding().model_ids == ["fixture", "replacement"]
    begin.assert_not_awaited()
    for effect in (ocr.claim, ocr.park, ocr.load, ocr.extract_page, llm._client.chat):
        effect.assert_not_awaited()


@pytest.mark.parametrize(
    "body", [{}, {"model_ids": []}, {"model_ids": ["a", "a"]}, {"model_ids": ["   "]}]
)
def test_benchmark_create_requires_explicit_unique_models(body):
    from pydantic import ValidationError

    from vision_bill.model.benchmark import BenchmarkCreate

    with pytest.raises(ValidationError, match="model_ids"):
        BenchmarkCreate.model_validate(body)


@pytest.mark.parametrize("phase", ["create", "replay_readiness", "replay_load", "replay_inference"])
async def test_benchmark_http_reports_gpu_oom_and_parks_worker(settings, tmp_path, monkeypatch, phase):
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI
    from PIL import Image
    from test_experiment_scoring import receipt
    from test_extraction_pipeline import formatter
    from test_handoff_binding import adopt
    from test_ocr_provider import http_client

    from vision_bill import config
    from vision_bill.api.benchmarks import router
    from vision_bill.api.helper.helper import get_benchmark_service
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider
    from vision_bill.security.dependencies import get_current_user
    from vision_bill.security.models import User
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.service.experiment_execution import image_digest
    from vision_bill.service.gpu_handoff import ModelHandoff
    from vision_bill.service.receipt_service import ReceiptService

    settings.ocr = config.OcrSettings(
        enabled=True, service_url="http://native.test", coordination_path=tmp_path / "gpu.lock",
    )
    monkeypatch.setattr(config, "settings", settings)
    requests = []
    loaded = False
    fail = phase == "create"
    ready = {
        "ready": True, "generation": 1, "profile_id": "fixed", "runtime_fingerprint": "runtime",
    }
    parked = {"ready": False, "generation": 1, "reason_code": "ocr_parked"}

    def native(request):
        nonlocal loaded
        path = request.url.path
        requests.append(path)
        if path == "/info":
            return httpx.Response(200, json={"contract_version": 2, "profile": {}, "worker": ready})
        if path == "/ready":
            if fail and phase == "replay_readiness":
                return httpx.Response(503, json={**parked, "reason_code": "ocr_gpu_out_of_memory"})
            return httpx.Response(200 if loaded else 503, json=ready if loaded else parked)
        if path in ("/lifecycle/claim", "/lifecycle/park"):
            loaded = False
            return httpx.Response(200, json=parked)
        if path == "/lifecycle/load" and (not fail or phase == "replay_inference"):
            loaded = True
            return httpx.Response(200, json=ready)
        assert path in ("/ocr", "/lifecycle/load")
        return httpx.Response(503, json={"detail": "ocr_gpu_out_of_memory"})

    client_type = httpx.AsyncClient
    llm = formatter()
    http_client(monkeypatch, native)
    ocr = PaddleOcrProvider(settings.ocr.service_url)
    llm.handoff = ModelHandoff(settings.ocr.coordination_path, llm, ocr)
    await adopt(llm.handoff, ("fixture",))
    receipt_id, run_id = uuid4(), uuid4()
    user = User(id=uuid4(), username="admin", is_admin=True)
    photo = tmp_path / "photo.png"
    Image.new("RGB", (3, 4)).save(photo)
    benchmark = object.__new__(BenchmarkService)
    benchmark._provider = llm
    benchmark._receipts = ReceiptService(settings.images, settings.pg, llm, ocr, settings.ocr)
    benchmark._receipts.list_tags = AsyncMock(return_value=[])
    benchmark._select_receipts = AsyncMock(return_value=[receipt_id])
    benchmark._capture_input = AsyncMock(return_value={
        "receipt_id": str(receipt_id), "expected": receipt().model_dump(mode="json"),
        "images": [{"path": str(photo), "sha256": image_digest(photo)}],
    })
    benchmark._db = SimpleNamespace(
        create_run=AsyncMock(),
        run=AsyncMock(return_value=SimpleNamespace(id=run_id, model_ids=["fixture"], receipt_ids=[receipt_id])),
        manifest=AsyncMock(),
    )
    benchmark._run_digests = AsyncMock(return_value={})
    benchmark._timeout = AsyncMock(return_value=2)
    create = {"model_ids": ["fixture"], "experiment": {"pipeline": "ocr", "include_classification": False}}
    if phase != "create":
        await benchmark.create_run(BenchmarkCreate.model_validate(create), scope=AccessScope(user_id=user.id))
        benchmark._db.manifest.return_value = benchmark._db.create_run.call_args.kwargs["manifest"]
        benchmark._db.create_run.reset_mock()
        requests.clear()
        fail = True

    app = FastAPI()
    app.include_router(router, prefix="/benchmarks")
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_benchmark_service] = lambda: benchmark
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with client_type(transport=transport, base_url="http://app.test") as client:
        if phase == "create":
            response = await client.post("/benchmarks", json=create)
        else:
            response = await client.post(f"/benchmarks/{run_id}/reevaluate", json={
                "receipt_id": str(receipt_id), "model_id": "fixture",
            })

    if phase == "replay_readiness":
        assert not any(request.startswith("/lifecycle/") for request in requests)
    else:
        assert requests[-1] == "/lifecycle/park"
    assert not llm.handoff.read_journal().ocr_pending
    assert requests.count("/ocr") == (1 if phase == "replay_inference" else 0)
    llm._client.chat.assert_not_awaited()
    benchmark._db.create_run.assert_not_awaited()
    assert response.status_code == 503 and response.json() == {"detail": "ocr_gpu_out_of_memory"}


@pytest.mark.parametrize("body", [{}, {"model_ids": []}, {"model_ids": ["a", "a"]}])
async def test_benchmark_service_rejects_invalid_model_selection_before_discovery(body):
    from types import SimpleNamespace

    from vision_bill.config import OcrSettings
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.service.benchmark_service import BenchmarkService

    service = object.__new__(BenchmarkService)
    discover = AsyncMock(return_value=[])
    service._provider = SimpleNamespace(get_available_models=discover)
    service._receipts = SimpleNamespace(_ocr_settings=OcrSettings())
    with pytest.raises(ValueError, match="Select at least one|unique"):
        await service.create_run(
            BenchmarkCreate.model_construct(**body), scope=AccessScope(user_id=uuid4())
        )
    discover.assert_not_awaited()
