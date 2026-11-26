"""Durable execution uses the saved snapshot and one complete attempt budget."""

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from PIL import Image

from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImageRow
from vision_bill.model.workflow import AnalysisClaim, CommitResult, PersistedSourcePart

ENDPOINT = "ollama:" + "d" * 64


def context(settings, tmp_path):
    from vision_bill.service.image_service import ImageService
    from vision_bill.service.submission_service import SubmissionService

    path = tmp_path / "original.png"
    Image.new("RGB", (4, 5)).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    image = ImageRow(
        id=uuid4(),
        image_path=str(path),
        model_id="saved-model",
        effective_options={
            "execution_policy_version": 4,
            "managed_binding_sha256": None,
            "allow_ocr_fallback": False,
            "inference_identity": ENDPOINT,
            "ocr": OcrSettings().model_dump(mode="json"),
            "temperature": 0.13,
            "max_output_tokens": 1234,
        },
    )
    claim = AnalysisClaim(
        image=image,
        generation=1,
        attempt_number=1,
        claim_token=uuid4(),
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=900),
    )
    part = PersistedSourcePart(
        part_id=uuid4(), photo_index=0, source_sha256=digest, image_path=str(path)
    )
    db = SimpleNamespace(
        set_wait_reason=AsyncMock(return_value=True),
        bind_source_parts=AsyncMock(return_value=[part]),
        get_ocr_checkpoint=AsyncMock(return_value=None),
        finish_attempt=AsyncMock(return_value=CommitResult(stale=False)),
        hold_claim_for_policy=AsyncMock(return_value=CommitResult(stale=False)),
        hold_claim_for_readiness=AsyncMock(return_value=CommitResult(stale=False)),
        clear_ocr_checkpoint=AsyncMock(return_value=True),
        save_ocr_checkpoint=AsyncMock(return_value=True),
    )
    receipt = SimpleNamespace(
        inference_identity=ENDPOINT,
        workflow_db=db, extract_receipt=AsyncMock(), list_tags=AsyncMock(return_value=[]),
        execution_handoff=Mock(return_value=None),
    )
    service = SubmissionService(settings, receipt, ImageService(settings.images))
    return service, receipt, db, claim, part


async def test_original_mismatch_terminal_and_safe(settings, tmp_path):
    service, receipt, db, claim, _part = context(settings, tmp_path)
    db.bind_source_parts.side_effect = ValueError("sensitive original mismatch")
    await service.process_claim(claim)
    receipt.extract_receipt.assert_not_awaited()
    outcome = db.finish_attempt.call_args.args[1]
    assert outcome.kind == "terminal" and outcome.result_code == "source_integrity_failed"


async def test_saved_snapshot_is_forwarded_without_current_defaults(settings, tmp_path):
    service, receipt, db, claim, part = context(settings, tmp_path)
    receipt.list_tags.return_value = ["coffee", "food"]
    receipt.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    args, kwargs = receipt.extract_receipt.call_args
    assert args[0] == "saved-model"
    assert kwargs["tags"] == ["coffee", "food"]
    assert kwargs["options"].temperature == 0.13
    assert kwargs["max_output_tokens"] == 1234
    assert kwargs["ocr_settings"].enabled is False
    assert kwargs["source_refs"][0].part_id == part.part_id
    assert db.finish_attempt.call_args.args[1].kind == "retry"


async def test_hung_attempt_includes_binding_in_original_budget(settings, tmp_path):
    service, _receipt, db, claim, _part = context(settings, tmp_path)
    # Exercise the real timer with enough reserve for error persistence under load.
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": 2})
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    finished = []

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def finish(*args, deadline):
        finished.append(loop.time())
        assert finished[-1] < deadline
        return CommitResult(stale=False)

    db.bind_source_parts.side_effect = hang
    db.finish_attempt.side_effect = finish
    began = loop.time()
    async with asyncio.timeout(2.8):
        result = await service.process_claim(claim)
    assert entered.is_set()
    assert not result.stale and len(finished) == 1
    assert finished[0] < began + 2
    assert db.finish_attempt.call_args.args[1].result_code == "timed_out"


async def test_working_artifact_validates_png_hash_dimensions_and_submission(settings, tmp_path):
    import base64
    from time import monotonic

    from test_ocr_provider import wire_page

    from vision_bill.model.ocr import WorkingImage, WorkingImageMetadata
    from vision_bill.service.image_service import ImageService

    service = ImageService(settings.images)
    image_id = uuid4()
    working = WorkingImage.model_validate(wire_page()["working_image"])
    path = await service.save_working_image(image_id, working, deadline=monotonic() + 2)
    metadata = WorkingImageMetadata.model_validate(working.model_dump())
    assert service.read_working_image(image_id, str(path), metadata) == base64.b64decode(
        working.png_base64
    )
    with pytest.raises(ValueError):
        service.read_working_image(uuid4(), str(path), metadata)
    with pytest.raises(ValueError):
        service.read_working_image(image_id, str(path), metadata.model_copy(update={"width": 999}))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        service.read_working_image(image_id, str(path), metadata)


async def test_stale_checkpoint_cleans_only_unreferenced_new_artifact(settings, tmp_path):
    from test_ocr_provider import wire_page

    from vision_bill.model.ocr import OcrResponse

    service, receipt, db, claim, part = context(settings, tmp_path)
    db.save_ocr_checkpoint.return_value = False
    db.unreferenced_views = AsyncMock(side_effect=lambda image_id, views, **kwargs: views)

    async def extract(*args, **kwargs):
        await kwargs["checkpoint"](OcrResponse.model_validate(wire_page(0, part.source_sha256)))
        raise AssertionError("must stop before formatting")

    receipt.extract_receipt.side_effect = extract
    result = await service.process_claim(claim)
    assert result.stale
    assert not list((tmp_path / "uploads" / "working").glob("*/*.png"))
    db.finish_attempt.assert_not_awaited()


async def test_hung_outcome_commit_cannot_extend_complete_attempt_budget(settings, tmp_path):
    service, receipt, db, claim, _part = context(settings, tmp_path)
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": 0.02})
    receipt.extract_receipt.side_effect = TimeoutError()

    async def hang(*args, **kwargs):
        await asyncio.Event().wait()

    db.finish_attempt.side_effect = hang
    async with asyncio.timeout(0.3):
        result = await service.process_claim(claim)
    assert result.stale


async def test_preclaim_ocr_text_model_is_eligible_but_missing_model_waits(settings, tmp_path):
    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.provider.llm.base import ModelInfo

    service, receipts, _db, claim, _part = context(settings, tmp_path)
    claim.image.effective_options["ocr"] = OcrSettings(enabled=True, required=True, manage_model_lifecycle=False).model_dump(
        mode="json"
    )
    settings.ocr = OcrSettings.model_validate(claim.image.effective_options["ocr"])
    receipts.check_connection = AsyncMock(return_value=True)
    model = ModelInfo("saved-model")
    model.capabilities = ["vision", "completion"]
    receipts.get_available_models = AsyncMock(return_value=[model])
    receipts.ocr_capabilities = AsyncMock(
        return_value=OcrCapabilities(
            enabled=True,
            configured=True,
            reachable=True,
            ready=True,
            profile_id="profile",
            runtime_fingerprint="runtime",
        )
    )
    assert await service.ready_for(claim.image)
    receipts.get_available_models.assert_awaited_once_with(capability="text")
    receipts.get_available_models.return_value = []
    assert not await service.ready_for(claim.image)


@pytest.mark.parametrize("unreadable", [True, False])
async def test_worker_errors_follow_durable_retry_policy(settings, tmp_path, unreadable):
    from vision_bill.provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError

    service, receipt, db, claim, _part = context(settings, tmp_path)
    receipt.extract_receipt.side_effect = (
        UnreadableReceiptError("sensitive text")
        if unreadable
        else AnalysisTimeoutError("sensitive text")
    )
    await service.process_claim(claim)
    outcome = db.finish_attempt.call_args.args[1]
    assert outcome.kind == ("terminal" if unreadable else "retry")
    assert outcome.result_code == ("unreadable" if unreadable else "timed_out")


async def test_worker_passes_every_original_in_capture_order(settings, tmp_path):
    from vision_bill.model.db.image import ImagePart

    service, receipt, db, claim, part = context(settings, tmp_path)
    second = tmp_path / "second.png"
    Image.new("RGB", (6, 7)).save(second)
    digest = hashlib.sha256(second.read_bytes()).hexdigest()
    claim.image.additional_images = [ImagePart(image_path=str(second))]
    db.bind_source_parts.return_value.append(
        PersistedSourcePart(
            part_id=uuid4(), photo_index=1, source_sha256=digest, image_path=str(second)
        )
    )
    receipt.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    assert receipt.extract_receipt.call_args.args[1] == [Path(part.image_path), second]


async def test_preclaim_malformed_provider_response_does_not_spend_attempt(settings, tmp_path):
    service, receipts, _db, claim, _part = context(settings, tmp_path)
    receipts.check_connection = AsyncMock(return_value=True)
    receipts.get_available_models = AsyncMock(side_effect=ValueError("malformed provider JSON"))
    assert not await service.ready_for(claim.image)


async def test_controlled_clock_exhausts_original_600_seconds_before_commit(
    settings, tmp_path, monkeypatch
):
    import vision_bill.helper.deadline as deadlines
    import vision_bill.service.submission_service as submissions

    service, receipt, db, claim, _part = context(settings, tmp_path)
    clock = [100.0]
    monkeypatch.setattr(submissions, "monotonic", lambda: clock[0])
    monkeypatch.setattr(deadlines, "monotonic", lambda: clock[0])

    async def expired(*args, **kwargs):
        assert kwargs["deadline"] == 695.0
        clock[0] = 700.01
        raise TimeoutError()

    receipt.extract_receipt.side_effect = expired
    result = await service.process_claim(claim)
    assert result.stale
    db.finish_attempt.assert_not_awaited()


async def test_checkpoint_transport_failure_does_not_clear_cache_or_run_ocr(settings, tmp_path):
    import asyncpg

    service, receipts, db, claim, _part = context(settings, tmp_path)
    db.get_ocr_checkpoint.side_effect = asyncpg.ConnectionDoesNotExistError("connection lost")
    await service.process_claim(claim)
    db.clear_ocr_checkpoint.assert_not_awaited()
    receipts.extract_receipt.assert_not_awaited()
    outcome = db.finish_attempt.call_args.args[1]
    assert outcome.kind == "retry" and outcome.result_code == "analysis_unavailable"


@pytest.mark.parametrize("permission", [None, False, True, "true"])
async def test_saved_fallback_permission_cannot_enable_vision_readiness_or_execution(
    settings, tmp_path, permission
):
    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.provider.llm.base import ModelInfo

    service, receipts, db, claim, _ = context(settings, tmp_path)
    claim.image.effective_options["ocr"] = OcrSettings(
        enabled=True, allow_vision_fallback=True, manage_model_lifecycle=False
    ).model_dump(mode="json")
    settings.ocr = OcrSettings.model_validate(claim.image.effective_options["ocr"])
    for key in ("inference_identity", "execution_policy_version", "managed_binding_sha256", "allow_ocr_fallback"):
        claim.image.effective_options.pop(key)
    if permission is not None:
        claim.image.effective_options["allow_ocr_fallback"] = permission
    snapshot = dict(claim.image.effective_options)
    receipts.check_connection = AsyncMock(return_value=True)
    receipts.ocr_capabilities = AsyncMock(
        return_value=OcrCapabilities(enabled=True, configured=True, reachable=True, ready=False)
    )
    model = ModelInfo("saved-model")
    model.capabilities = ["vision", "completion"]
    receipts.get_available_models = AsyncMock(return_value=[model])
    assert await service.ready_for(claim.image) is isinstance(permission, str)
    receipts.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    if isinstance(permission, str):
        receipts.extract_receipt.assert_not_awaited()
        assert db.finish_attempt.call_args.args[1].result_code == "analysis_contract_failed"
    else:
        db.hold_claim_for_policy.assert_awaited_once()
        receipts.extract_receipt.assert_not_awaited()
        db.finish_attempt.assert_not_awaited()
    assert claim.image.effective_options == snapshot


@pytest.mark.parametrize("vision,alternate", [(True, False), (False, False), (False, True)])
async def test_ocr_readiness_never_uses_a_vision_alternate(settings,tmp_path,vision,alternate):
    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.provider.llm.base import ModelInfo
    service, receipts, _db, claim, _ = context(settings,tmp_path)
    for key in ("inference_identity", "execution_policy_version", "managed_binding_sha256"):
        claim.image.effective_options.pop(key)
    claim.image.effective_options.update(allow_ocr_fallback=True,
        ocr=OcrSettings(enabled=True,allow_vision_fallback=True,vision_fallback_model_id="alternate",manage_model_lifecycle=False).model_dump(mode="json"))
    receipts.check_connection = AsyncMock(return_value=True)
    receipts.ocr_capabilities = AsyncMock(return_value=OcrCapabilities(enabled=True,configured=True,reachable=False,ready=False))
    requested=ModelInfo("saved-model");requested.capabilities=["vision"] if vision else ["completion"]
    other=ModelInfo("alternate");other.capabilities=["vision"]
    receipts.get_available_models=AsyncMock(return_value=[requested]+([other] if alternate else []))
    assert not await service.ready_for(claim.image)
    receipts.get_available_models.assert_not_awaited()


async def test_current_snapshot_reaches_attempt_with_frozen_formatter_settings(settings, tmp_path):
    from vision_bill.service.execution_policy import current_snapshot

    service, receipts, db, claim, _ = context(settings, tmp_path)
    settings.llm.temperature = 0.27
    settings.llm.max_output_tokens = 4321
    claim.image.effective_options = current_snapshot(settings, inference_identity=ENDPOINT)
    settings.llm.temperature = 0.91
    receipts.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    assert receipts.extract_receipt.call_args.kwargs["options"].temperature == 0.27
    assert receipts.extract_receipt.call_args.kwargs["max_output_tokens"] == 4321
    assert db.finish_attempt.call_args.args[1].kind == "retry"


async def test_readiness_wait_is_persisted_and_cleared_without_attempt(settings,tmp_path):
    service, receipts, db, claim, _ = context(settings,tmp_path)
    waits=[]
    async def persist(image, reason):
        waits.append((image.id, image.generation, image.status, reason))
        return True
    db.set_wait_reason=persist
    receipts.check_connection=AsyncMock(return_value=False)
    assert not await service.ready_for(claim.image)
    assert waits == [(claim.image.id,1,'pending','provider_unavailable')]
    from vision_bill.provider.llm.base import ModelInfo
    model=ModelInfo('saved-model');model.capabilities=['vision']
    receipts.check_connection.return_value=True
    receipts.get_available_models=AsyncMock(return_value=[model])
    assert await service.ready_for(claim.image)
    assert waits[-1][-1] is None
    assert claim.image.attempt_count == 0


@pytest.mark.parametrize("version", [None, 2])
async def test_policy_change_holds_before_claim_and_dispatch(settings, tmp_path, version):
    from copy import deepcopy

    from vision_bill.service.execution_policy import current_snapshot

    service, receipts, db, claim, _ = context(settings, tmp_path)
    snapshot = current_snapshot(settings, inference_identity=ENDPOINT)
    snapshot.pop("inference_identity")
    snapshot.pop("managed_binding_sha256")
    if version is None:
        snapshot.pop("execution_policy_version")
    else:
        snapshot["execution_policy_version"] = version
    claim.image.effective_options = snapshot
    before = deepcopy(snapshot)
    settings.ocr = OcrSettings(enabled=True)
    db.hold_claim_for_policy = AsyncMock(return_value=CommitResult(stale=False))
    assert not await service.ready_for(claim.image)
    db.set_wait_reason.assert_awaited_once_with(claim.image, "execution_policy_changed")
    assert not (await service.process_claim(claim)).stale
    db.hold_claim_for_policy.assert_awaited_once()
    receipts.extract_receipt.assert_not_awaited()
    db.bind_source_parts.assert_not_awaited()
    db.finish_attempt.assert_not_awaited()
    assert claim.image.effective_options == before


async def test_policy_change_during_preparation_holds_before_extraction(settings, tmp_path):
    service, receipts, db, claim, part = context(settings, tmp_path)

    async def bind(*args, **kwargs):
        settings.ocr = OcrSettings(enabled=True)
        return [part]

    db.bind_source_parts.side_effect = bind
    await service.process_claim(claim)
    receipts.extract_receipt.assert_not_awaited()
    db.finish_attempt.assert_not_awaited()
    db.hold_claim_for_policy.assert_awaited_once()


async def test_adopted_default_resolves_once_without_changing_frozen_policy(settings, tmp_path):
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.execution_policy import current_snapshot

    service, receipts, db, claim, _ = context(settings, tmp_path)
    claim.image.model_id = None
    claim.image.effective_options = current_snapshot(settings, inference_identity=ENDPOINT)
    saved = claim.image.model_copy(update={"model_id": settings.llm.model_name})
    db.ensure_execution_snapshot = AsyncMock(return_value=saved)
    model = ModelInfo(settings.llm.model_name)
    model.capabilities = ["vision"]
    receipts.get_available_models = AsyncMock(return_value=[model])
    receipts.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    assert db.ensure_execution_snapshot.call_args.kwargs["effective_options"] == claim.image.effective_options
    assert receipts.extract_receipt.call_args.args[0] == settings.llm.model_name


async def test_adopted_default_waits_for_exact_configured_model(settings, tmp_path):
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.execution_policy import current_snapshot

    service, receipts, db, claim, _ = context(settings, tmp_path)
    claim.image.model_id = None
    claim.image.effective_options = current_snapshot(settings, inference_identity=ENDPOINT)
    receipts.check_connection = AsyncMock(return_value=True)
    model = ModelInfo("different")
    model.capabilities = ["vision"]
    receipts.get_available_models = AsyncMock(return_value=[model])
    assert not await service.ready_for(claim.image)
    db.set_wait_reason.assert_awaited_once_with(claim.image, "vision_model_unavailable")


async def test_policy_change_after_a_real_dispatch_keeps_the_attempt_counted(settings, tmp_path):
    from vision_bill.service.extraction_pipeline import OcrRuntimeChangedError
    service, receipts, db, claim, _ = context(settings, tmp_path)

    async def extract(*args, **kwargs):
        settings.ocr = OcrSettings(enabled=True)
        raise OcrRuntimeChangedError("ocr_runtime_changed")

    receipts.extract_receipt.side_effect = extract
    await service.process_claim(claim)
    receipts.extract_receipt.assert_awaited_once()
    db.hold_claim_for_policy.assert_not_awaited()
    assert db.finish_attempt.call_args.args[1].kind == "retry"


async def test_second_extraction_wait_does_not_refund_an_earlier_dispatch(settings, tmp_path):
    from vision_bill.model.handoff import HandoffStatus, HandoffWait
    from vision_bill.service.extraction_pipeline import OcrRuntimeChangedError

    service, receipts, db, claim, _ = context(settings, tmp_path)
    receipts.extract_receipt.side_effect = [
        OcrRuntimeChangedError("ocr_runtime_changed"),
        HandoffWait(HandoffStatus(state="busy", reason_code="model_handoff_busy")),
    ]

    await service.process_claim(claim)

    assert receipts.extract_receipt.await_count == 2
    db.hold_claim_for_readiness.assert_not_awaited()
    outcome = db.finish_attempt.call_args.args[1]
    assert outcome.kind == "retry" and outcome.result_code == "ocr_runtime_changed"


async def test_bound_direct_model_disappearance_holds_without_dispatch(settings, tmp_path):
    from test_extraction_pipeline import formatter

    from vision_bill.service.receipt_service import ReceiptService

    service, _, db, claim, _ = context(settings, tmp_path)
    llm = formatter()
    selected = llm.get_available_models.return_value[0]
    selected.id = claim.image.model_id
    receipts = ReceiptService(settings.images, settings.pg, llm)
    receipts.list_tags = AsyncMock(return_value=[])
    receipts.check_connection = AsyncMock(return_value=True)
    service.receipts = receipts
    claim.image.effective_options["inference_identity"] = receipts.inference_identity
    llm.get_available_models.side_effect = [[selected], [], [selected]]

    assert await service.ready_for(claim.image)
    await service.process_claim(claim)
    db.hold_claim_for_readiness.assert_awaited_once()
    assert db.hold_claim_for_readiness.await_args.args == (claim, "vision_model_unavailable")
    db.finish_attempt.assert_not_awaited()
    llm._client.chat.assert_not_awaited()
    assert await service.ready_for(claim.image)


@pytest.mark.parametrize("route", ["direct", "separate_gpu"])
async def test_changed_formatter_endpoint_holds_before_discovery_and_dispatch(settings, tmp_path, route):
    from vision_bill.service.execution_policy import current_snapshot

    service, receipt, db, claim, _part = context(settings, tmp_path)
    if route == "separate_gpu":
        settings.ocr = OcrSettings(enabled=True, shared_gpu=False)
    receipt.inference_identity = "ollama:" + "b" * 64
    claim.image.effective_options = current_snapshot(settings, inference_identity="ollama:" + "a" * 64)
    assert not await service.ready_for(claim.image)
    db.set_wait_reason.assert_awaited_once_with(claim.image, "execution_policy_changed")
    await service.process_claim(claim)
    db.hold_claim_for_policy.assert_awaited_once()
    receipt.extract_receipt.assert_not_awaited()


async def test_endpoint_change_during_preparation_refunds_claim_before_extraction(settings, tmp_path):
    service, receipt, db, claim, part = context(settings, tmp_path)

    async def bind(*args, **kwargs):
        receipt.inference_identity = "openai:" + "e" * 64
        return [part]

    db.bind_source_parts.side_effect = bind
    await service.process_claim(claim)
    db.hold_claim_for_policy.assert_awaited_once()
    receipt.extract_receipt.assert_not_awaited()
    db.finish_attempt.assert_not_awaited()


@pytest.mark.parametrize("reason", ["ocr_unavailable", "ocr_starting", "ocr_cancelled"])
async def test_resume_preserves_pages_while_native_identity_is_temporarily_unreachable(settings, tmp_path, reason):
    from time import monotonic

    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.service.extraction_pipeline import ReadinessHoldError

    service, receipts, db, claim, part = context(settings, tmp_path)
    db.get_ocr_checkpoint.return_value = SimpleNamespace(ordered_source_sha256=[part.source_sha256])
    receipts.ocr_capabilities = AsyncMock(return_value=OcrCapabilities(
        enabled=True, configured=True, reachable=reason != "ocr_unavailable", ready=False, reason_code=reason,
    ))
    with pytest.raises(ReadinessHoldError, match=reason):
        await service._resume(claim, [part], OcrSettings(enabled=True), monotonic() + 5,
                              db_deadline=asyncio.get_running_loop().time() + 5)
    db.clear_ocr_checkpoint.assert_not_awaited()
