"""All-photo extraction policy and source-aware formatter integration."""

import asyncio
import hashlib
import importlib.util
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ollama import ChatResponse, ProcessResponse
from PIL import Image
from test_experiment_scoring import receipt
from test_ocr_provider import wire_page

from vision_bill.config import OcrSettings
from vision_bill.model.experiment import ExperimentOptions
from vision_bill.model.ocr import OcrCapabilities, OcrResponse
from vision_bill.provider.llm.base import ModelInfo
from vision_bill.provider.llm.ollama import OllamaProvider


@pytest.fixture
def photos(tmp_path):
    paths = [tmp_path / f"{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        Image.new("RGB", (4, 5), (i, 0, 0)).save(path)
    return paths


def formatter(vision=True, payload=None):
    provider = OllamaProvider("http://local")
    provider._client.ps = AsyncMock(return_value=ProcessResponse(models=[]))
    model = ModelInfo("fixture")
    model.capabilities = ["completion", "vision"] if vision else ["completion"]
    provider.get_available_models = AsyncMock(return_value=[model])
    payload = payload or receipt().model_dump(mode="json")
    for row in payload["line_items"]:
        row.setdefault("source_image", 0)
    payload.setdefault("duplicate_candidates", [])
    provider._client.chat = AsyncMock(
        return_value=ChatResponse(
            model="actual-model",
            done_reason="stop",
            eval_count=42,
            message={"role": "assistant", "content": json.dumps(payload)},
        )
    )
    return provider


def pipeline(provider, ocr=None, **settings):
    assert importlib.util.find_spec("vision_bill.service.extraction_pipeline"), (
        "shared pipeline missing"
    )
    from vision_bill.service.extraction_pipeline import ExtractionPipeline

    return ExtractionPipeline(provider, ocr, OcrSettings(enabled=ocr is not None, **settings))


def sidecar(fail_at=None, blank=False):
    async def page(path, *, photo_index, source_sha256, deadline, expected_profile_id=None):
        if photo_index == fail_at:
            if blank:
                return OcrResponse.model_validate(wire_page(photo_index, source_sha256, text=" "))
            raise ValueError("sensitive receipt content must not be exposed")
        return OcrResponse.model_validate(wire_page(photo_index, source_sha256))

    return SimpleNamespace(
        lifecycle_identity="ocr:extraction-fixture",
        get_capabilities=AsyncMock(
            return_value=OcrCapabilities(
                enabled=True,
                configured=True,
                reachable=True,
                ready=True,
                profile_id="fixed-profile",
            )
        ),
        extract_page=AsyncMock(side_effect=page),
    )


async def test_explicit_direct_never_calls_ocr(photos):
    ocr, llm = sidecar(), formatter()
    result = await pipeline(llm, ocr).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is not None
    ocr.extract_page.assert_not_awaited()
    ocr.get_capabilities.assert_not_awaited()
    assert result.provenance.source_image_count == 2
    assert len(llm._client.chat.call_args.kwargs["messages"][0]["images"]) == 2


async def test_ocr_formatting_correction_preserves_separate_purchases_and_printed_vat(photos):
    from decimal import Decimal
    from uuid import uuid4

    from vision_bill.model.ocr import SourceRef
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.service.submission_service import SubmissionService

    # Two independently priced identical purchases have the same two-line layout.
    transcript = (
        "Hat 19%\nSKU 37 / M / blue   12,00\nHat 19%\nSKU 37 / M / blue   12,00\n"
        "Scarf 19%\nSKU 89 / red   11,70\nNet total 30,00   VAT 5,70   Total 35,70"
    )
    source_hash = hashlib.sha256(photos[0].read_bytes()).hexdigest()
    ref = SourceRef(
        part_id=uuid4(), photo_index=0, source_sha256=source_hash, coordinate_space="original"
    )
    ocr = sidecar()
    page = OcrResponse.model_validate(wire_page(digest=source_hash, text=transcript))
    ocr.extract_page = AsyncMock(return_value=page)
    product_names = ["Hat M blue", "Hat M blue", "Scarf red"]
    prices = ["12.00", "12.00", "11.70"]
    observed = receipt(product_names).model_dump(mode="json")
    for item, price in zip(observed["line_items"], prices, strict=True):
        item.update(quantity=None, unit_price=None, total_price=price)
    observed.update(
        subtotal="30.00", tax_total="5.70", total="35.70",
        taxes=[{"name": "VAT", "rate": 19, "amount": "5.70"}],
        amount_policy={"tax_basis": "gross", "subtotal_origin": "printed_net", "tax_information": "present"},
    )
    llm = formatter(vision=False, payload=observed)
    llm._client.chat.side_effect = [
        llm._client.chat.return_value,
        ChatResponse(message={"role": "assistant", "content": '{"corrections":{"taxes.0.rate":"0.19"}}'}),
    ]

    result = await pipeline(llm, ocr).extract(
        "fixture", photos[:1], options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 5, tags=[], source_refs=[ref],
    )
    originals = [OriginalAvailability(part_id=ref.part_id, photo_index=0, available=True, sha256_matches=True)]
    outcome = SubmissionService._outcome(result, [], originals)

    ocr.extract_page.assert_awaited_once()
    assert llm._client.chat.await_count == 2
    first, correction = [call.kwargs["messages"] for call in llm._client.chat.await_args_list]
    assert first[0] == correction[0]
    assert "SKU 37" in first[0]["content"]
    assert all("images" not in message for message in correction)
    assert "taxes.0.rate" in correction[-1]["content"]
    assert outcome.kind == "complete" and outcome.review.findings == []
    assert [item.description for item in result.receipt.line_items] == product_names
    assert [item.total_price for item in result.receipt.line_items] == [Decimal(price) for price in prices]
    assert result.receipt.taxes[0].rate == Decimal("0.19")
    assert result.receipt.subtotal == Decimal("30.00") and result.receipt.total == Decimal("35.70")


@pytest.mark.parametrize("blank", [False, True])
async def test_second_photo_failure_is_not_a_complete_prefix(photos, blank):
    ocr, llm = sidecar(1, blank), formatter()
    checkpoint = AsyncMock()
    result = await pipeline(llm, ocr, required=True, allow_vision_fallback=True).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", allow_ocr_fallback=True),
        deadline=time.monotonic() + 2,
        tags=[],
        checkpoint=checkpoint,
    )
    assert result.receipt is None
    assert result.provenance.source_image_count == 2
    assert len(result.ocr_pages) == 1
    checkpoint.assert_awaited_once()
    llm._client.chat.assert_not_awaited()
    assert not result.provenance.fallback_used


@pytest.mark.parametrize("failure_phase,status,code", [
    ("load", 503, "ocr_gpu_out_of_memory"), ("inference", 503, "ocr_gpu_out_of_memory"),
    ("load", 503, "ocr_model_load_failed"), ("inference", 502, "ocr_invalid_result"),
])
async def test_native_error_parks_worker_and_records_receipt_retry(
    photos, monkeypatch, failure_phase, status, code,
):
    import re

    import httpx
    from test_handoff_binding import adopt
    from test_ocr_provider import http_client

    from vision_bill import config
    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider
    from vision_bill.service.submission_service import SubmissionService

    requests = []
    ready = {"ready": True, "generation": 1, "profile_id": "fixed-profile"}
    parked = {"ready": False, "generation": 1, "reason_code": "ocr_parked"}

    def native(request):
        path = request.url.path
        requests.append(path)
        if path == "/info":
            return httpx.Response(200, json={"contract_version": 2, "profile": {}, "worker": ready})
        if path in ("/lifecycle/claim", "/lifecycle/park"):
            return httpx.Response(200, json=parked)
        if path == "/ready" or (path == "/lifecycle/load" and failure_phase == "inference"):
            return httpx.Response(200, json=ready)
        if path == "/ocr" and requests.count("/ocr") == 1:
            request_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', request.content)
            assert request_id is not None
            return httpx.Response(200, json=wire_page(
                0, hashlib.sha256(photos[0].read_bytes()).hexdigest(), request_id[1].decode(),
            ))
        assert path in ("/ocr", "/lifecycle/load")
        return httpx.Response(status, json={"detail": code})

    llm = formatter()
    http_client(monkeypatch, native)
    ocr = PaddleOcrProvider("http://native")
    extraction = pipeline(
        llm, ocr, manage_model_lifecycle=True, coordination_path=photos[0].parent / "gpu.lock",
    )
    monkeypatch.setattr(config.settings, "ocr", extraction.settings)
    handoff = extraction.coordinator()
    assert handoff is not None
    await adopt(handoff, ("fixture",))
    checkpoint = AsyncMock()
    result = await extraction.extract(
        "fixture", photos, options=ExperimentOptions(pipeline="ocr"),
        deadline=time.monotonic() + 2, tags=[], checkpoint=checkpoint,
    )

    outcome = SubmissionService._outcome(result, [], [])
    assert outcome.kind == "retry" and outcome.result_code == code
    assert result.findings[-1].code == code
    assert requests[-1] == "/lifecycle/park" and not handoff.read_journal().ocr_pending
    assert len(result.ocr_pages) == checkpoint.await_count == (1 if failure_phase == "inference" else 0)
    assert requests.count("/ocr") == (2 if failure_phase == "inference" else 0)
    llm._client.chat.assert_not_awaited()


@pytest.mark.parametrize("failure", ["blank", "transport", "timeout"])
async def test_optional_failure_stops_without_images_despite_legacy_grants(photos, failure):
    import httpx

    llm = formatter()
    ocr = sidecar(1, blank=failure == "blank")
    if failure == "timeout":
        ocr.extract_page.side_effect = httpx.ReadTimeout("OCR timed out")
    result = await pipeline(llm, ocr, allow_vision_fallback=True).extract(
        "fixture",
        photos,
        options=ExperimentOptions(
            pipeline="ocr", allow_ocr_fallback=True, include_classification=False
        ),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is None
    assert not result.provenance.fallback_used
    assert result.provenance.fallback_reason == "ocr_page_failed"
    llm._client.chat.assert_not_awaited()


async def test_text_only_ocr_formats_all_marked_photos_without_png(photos):
    llm = formatter(False)
    result = await pipeline(llm, sidecar()).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=["private_tag"],
    )
    assert result.receipt is not None
    message = llm._client.chat.call_args.kwargs["messages"][0]
    prompt = message["content"]
    assert "PHOTO 0" in prompt and "PHOTO 1" in prompt
    assert "png_base64" not in prompt and "private_tag" not in prompt
    assert "images" not in message
    assert len(result.ocr_pages) == 2
    assert result.provenance.model_id == "fixture"
    assert result.telemetry.calls[0]["actual_model"] == "actual-model"
    assert result.telemetry.output_tokens == 42


async def test_text_only_model_never_receives_fallback_images(photos):
    llm = formatter(False)
    result = await pipeline(llm, sidecar(1), allow_vision_fallback=True).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", allow_ocr_fallback=True),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is None
    assert result.provenance.fallback_reason == "ocr_page_failed"
    llm._client.chat.assert_not_awaited()


async def test_checkpoint_failure_is_not_swallowed_or_fallback(photos):
    llm = formatter()
    checkpoint = AsyncMock(side_effect=OSError("storage failed"))
    with pytest.raises(OSError, match="storage failed"):
        await pipeline(llm, sidecar(), allow_vision_fallback=True).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="ocr", allow_ocr_fallback=True),
            deadline=time.monotonic() + 2,
            tags=[],
            checkpoint=checkpoint,
        )
    llm._client.chat.assert_not_awaited()


async def test_one_deadline_covers_ocr_and_formatter(photos):
    llm, ocr = formatter(), sidecar()

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.04)
        return await sidecar().extract_page(*args, **kwargs)

    ocr.extract_page.side_effect = slow
    with pytest.raises(TimeoutError):
        await pipeline(llm, ocr).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="ocr"),
            deadline=time.monotonic() + 0.02,
            tags=[],
        )
    llm._client.chat.assert_not_awaited()


async def test_real_gif_direct_uses_rgb_png_and_preserves_original_provenance(tmp_path):
    path = tmp_path / "fake.png"
    Image.new("RGB", (3, 5), "red").save(
        path, format="GIF", save_all=True, append_images=[Image.new("RGB", (3, 5), "blue")]
    )
    original = path.read_bytes()
    llm = formatter()

    async def inspect(**kwargs):
        transported = kwargs["messages"][0]["images"][0]
        from io import BytesIO

        assert isinstance(transported, bytes)
        with Image.open(BytesIO(transported)) as decoded:
            assert decoded.format == "PNG" and decoded.mode == "RGB"
        return ChatResponse(message={"role": "assistant", "content": receipt().model_dump_json()})

    llm._client.chat.side_effect = inspect
    result = await pipeline(llm).extract(
        "fixture",
        [path],
        options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert path.read_bytes() == original
    assert result.provenance.ordered_source_sha256 == [hashlib.sha256(original).hexdigest()]
    assert result.field_sources["input"] == []
    assert result.receipt is not None
    assert result.input_views[0].working_image.transforms[0]["frame_count"] == 2
    assert result.input_views[0].source_sha256 == hashlib.sha256(original).hexdigest()


async def test_ocr_discovery_accepts_text_model_and_records_known_context(photos):
    llm = formatter(False)
    llm.get_available_models.return_value[0].context_window_tokens = 32768
    llm.get_available_models.return_value[0].digest = "model-digest"
    result = await pipeline(llm, sidecar()).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    llm.get_available_models.assert_awaited_once_with(capability="text")
    assert result.telemetry.context_window_tokens == 32768
    assert result.provenance.model_digest == "model-digest"


async def test_legacy_vision_alternate_is_never_dispatched_after_ocr_failure(photos):
    llm = formatter(False)
    vision = ModelInfo("vision-fallback")
    vision.capabilities = ["vision"]
    vision.digest = "alternate-digest"
    llm.get_available_models.return_value.append(vision)
    result = await pipeline(
        llm, sidecar(1), allow_vision_fallback=True, vision_fallback_model_id="vision-fallback"
    ).extract(
        "fixture",
        photos,
        options=ExperimentOptions(
            pipeline="ocr", allow_ocr_fallback=True, include_classification=False
        ),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is None
    assert result.provenance.model_id == "fixture"
    assert result.provenance.requested_model_id == "fixture"
    assert result.provenance.model_digest is None
    llm._client.chat.assert_not_awaited()
    llm.get_available_models.assert_not_awaited()
    assert result.input_views == []


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "GIF", "HEIF"])
async def test_ocr_transport_keeps_every_original_format(monkeypatch, tmp_path, fmt):
    import re

    import httpx
    from pillow_heif import register_heif_opener
    from test_ocr_provider import http_client

    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider

    register_heif_opener()
    path = tmp_path / "misleading.dat"
    Image.new("RGB", (5, 7), "red").save(path, format=fmt)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "GIF": "gif", "HEIF": "heif"}[fmt]

    def handler(request):
        assert raw in request.content
        assert f"Content-Type: image/{mime}".encode() in request.content
        request_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', request.content)[1].decode()
        return httpx.Response(200, json=wire_page(digest=digest, request_id=request_id))

    http_client(monkeypatch, handler)
    result = await PaddleOcrProvider("http://local", managed_lifecycle=False).extract_page(
        path, photo_index=0, source_sha256=digest, deadline=time.monotonic() + 2
    )
    assert result.source_sha256 == digest and path.read_bytes() == raw


def test_saved_historical_fallback_finding_survives_review(photos):
    from pathlib import Path

    from vision_bill.model.review import AmountPolicy, OriginalAvailability, ReviewFinding
    from vision_bill.service.receipt_review import evaluate_receipt, persistent_findings

    # The UI fixture is the actual retained saved-review wire shape. The ID is
    # material-dependent, so assert its producer/review lifecycle separately.
    expected = json.loads((Path(__file__).parents[1] / "frontend/src/lib/test-fixtures/retained-ocr-finding.json").read_text())
    historical = receipt()
    findings = [ReviewFinding(id="ocr_page_failed", **expected)]
    originals = [
        OriginalAvailability(part_id=None, photo_index=i, available=True, sha256_matches=True)
        for i in range(len(photos))
    ]
    review = evaluate_receipt(
        historical, AmountPolicy(), {}, [],
        originals,
        retained_findings=findings,
    )
    retained = next(f for f in persistent_findings(review.findings) if f.code == "ocr_page_failed")
    assert retained.model_dump(mode="json", exclude={"id"}) == expected
    assert len(retained.id) == 64 and retained.id != "ocr_page_failed"
    repeated = evaluate_receipt(
        historical, AmountPolicy(), {}, [],
        originals,
        retained_findings=persistent_findings(review.findings),
    )
    assert next(f for f in repeated.findings if f.code == "ocr_page_failed") == retained


async def test_ocr_and_formatter_limits_remain_separate(photos):
    from vision_bill.model.ocr import OcrCapabilities

    ocr = sidecar()
    assert "ocr_max_new_tokens" in OcrCapabilities.model_fields, "typed OCR limit missing"
    ocr.get_capabilities.return_value = OcrCapabilities(
        enabled=True,
        configured=True,
        reachable=True,
        ready=True,
        profile_id="fixed-profile",
        ocr_max_new_tokens=4096,
    )
    result = await pipeline(formatter(False), ocr).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.telemetry.ocr_max_new_tokens == 4096
    assert result.telemetry.max_output_tokens == 32768
    assert result.telemetry.context_window_tokens is None


@pytest.mark.parametrize("fallback", [False, True])
async def test_configured_standard_openai_vision_model_reaches_inference(
    settings, photos, fallback
):
    from vision_bill.config import LLMProviderEnum
    from vision_bill.provider.factory import get_llm_provider

    declared = settings.llm.model_copy(update={"provider": LLMProviderEnum.OPENAI})
    # Assignment must be a supported settings field, not an ignored extra.
    assert "vision_model_ids" in type(declared).model_fields
    declared.vision_model_ids = ["configured-vision"]
    llm = get_llm_provider(declared)
    llm._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {"data": [{"id": "configured-vision", "object": "model"}]}
                ).encode()
            )
        )
    )
    payload = receipt().model_dump(mode="json")
    for row in payload["line_items"]:
        row["source_image"] = 0
    payload["duplicate_candidates"] = []
    llm._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            model="actual-vision",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload))
                )
            ],
        )
    )
    result = await pipeline(
        llm, sidecar(1) if fallback else None, allow_vision_fallback=fallback
    ).extract(
        "configured-vision",
        photos,
        options=ExperimentOptions(
            pipeline="ocr" if fallback else "direct_vision",
            allow_ocr_fallback=fallback,
            include_classification=False,
        ),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.provenance.fallback_used is False
    if fallback:
        assert result.receipt is None
        llm._client.chat.completions.create.assert_not_awaited()
    else:
        assert result.receipt is not None
        assert len(llm._client.chat.completions.create.call_args.kwargs["messages"][0]["content"]) == 3


@pytest.mark.parametrize("stage", ["hash", "normalization"])
async def test_blocked_preparation_obeys_deadline_and_cleans_owned_files(
    monkeypatch, tmp_path, photos, stage
):
    import tempfile
    import threading
    from pathlib import Path

    from PIL import ImageOps

    gate, entered, cleaned = threading.Event(), threading.Event(), threading.Event()
    original_read, original_transpose = Path.read_bytes, ImageOps.exif_transpose
    cleanup = tempfile.TemporaryDirectory.cleanup

    def cleanup_observed(instance):
        cleanup(instance)
        cleaned.set()

    def read(path):
        if stage == "hash" and path == photos[0]:
            entered.set()
            assert gate.wait(5), "test did not release blocked hashing"
        return original_read(path)

    def transpose(image, *args, **kwargs):
        entered.set()
        assert gate.wait(5), "test did not release blocked normalization"
        return original_transpose(image, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read)
    if stage == "normalization":
        monkeypatch.setattr(ImageOps, "exif_transpose", transpose)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", cleanup_observed)
    llm = formatter()
    began = time.monotonic()
    deadline = began + 1
    try:
        with pytest.raises(TimeoutError):
            await pipeline(llm).extract(
                "fixture",
                photos,
                options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
                deadline=deadline,
                tags=[],
            )
        assert entered.is_set()
        assert time.monotonic() < deadline + 0.3
        assert not gate.is_set()
        llm._client.chat.assert_not_awaited()
    finally:
        gate.set()
    if stage == "normalization":
        assert await asyncio.to_thread(cleaned.wait, 0.5)
        assert not list(tmp_path.glob("vision-bill-vision-*"))
    assert llm.experiment_trace is None


@pytest.mark.parametrize("late_stage", ["repair", "enrichment"])
async def test_budget_exhausted_after_all_ocr_during_late_call(photos, late_stage):
    llm, ocr = formatter(False), sidecar()
    first = llm._client.chat.return_value
    if late_stage == "repair":
        # A nullable partial receipt is valid; malformed JSON must enter repair.
        first = ChatResponse(
            message={"role": "assistant", "content": '{"merchant_name":'}
        )
    entered = asyncio.Event()

    async def late(**kwargs):
        if llm._client.chat.await_count == 1:
            return first
        entered.set()
        await asyncio.Event().wait()

    llm._client.chat.side_effect = late
    checkpoint = AsyncMock()
    began = time.monotonic()
    # Include real hashing/parsing and executor scheduling before the late call.
    # The assertions below still require both OCR checkpoints and late-stage entry.
    deadline = began + (61 if late_stage == "enrichment" else 1)
    with pytest.raises(TimeoutError):
        await pipeline(llm, ocr).extract(
            "fixture",
            photos,
            options=ExperimentOptions(
                pipeline="ocr", include_classification=late_stage == "enrichment"
            ),
            deadline=deadline,
            tags=["food"],
            checkpoint=checkpoint,
        )
    assert checkpoint.await_count == 2
    assert entered.is_set() and llm._client.chat.await_count == 2
    assert time.monotonic() < began + 1.35


async def test_ollama_serializes_exact_prepared_png_after_temporary_cleanup(
    monkeypatch, tmp_path, photos
):
    import base64
    import tempfile
    from io import BytesIO

    import httpx

    transported = []
    payload = receipt().model_dump(mode="json")
    for row in payload["line_items"]:
        row["source_image"] = 0
    payload["duplicate_candidates"] = []

    def handler(request):
        assert not list(tmp_path.glob("vision-bill-vision-*"))
        body = json.loads(request.content)
        transported.extend(base64.b64decode(data) for data in body["messages"][0]["images"])
        for png in transported:
            with Image.open(BytesIO(png)) as image:
                assert image.format == "PNG" and image.mode == "RGB"
        return httpx.Response(
            200,
            json={
                "model": "actual",
                "message": {"role": "assistant", "content": json.dumps(payload)},
                "done": True,
            },
        )

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    llm = OllamaProvider("http://local")
    model = ModelInfo("fixture")
    model.capabilities = ["vision"]
    llm.get_available_models = AsyncMock(return_value=[model])
    async with httpx.AsyncClient(
        base_url="http://local", transport=httpx.MockTransport(handler)
    ) as client:
        llm._client._client = client
        result = await pipeline(llm).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
            deadline=time.monotonic() + 2,
            tags=[],
        )
    assert result.receipt is not None
    assert transported == [
        base64.b64decode(view.working_image.png_base64) for view in result.input_views
    ]
    assert [view.source_sha256 for view in result.input_views] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in photos
    ]


@pytest.mark.parametrize("count", [1, 2])
async def test_resume_prefix_uses_live_runtime_and_all_originals(photos, count):
    from uuid import uuid4

    from vision_bill.model.ocr import SourceRef

    ocr, llm = sidecar(), formatter(False)
    ocr.get_capabilities.return_value.runtime_fingerprint = "runtime"
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in photos]
    pages = [OcrResponse.model_validate(wire_page(i, hashes[i])) for i in range(count)]
    refs = [
        SourceRef(part_id=uuid4(), photo_index=i, source_sha256=h, coordinate_space="original")
        for i, h in enumerate(hashes)
    ]
    callback = AsyncMock()
    result = await pipeline(llm, ocr).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
        resume_pages=pages,
        source_refs=refs,
        checkpoint=callback,
    )
    assert len(result.ocr_pages) == 2
    assert ocr.extract_page.await_count == 2 - count
    assert callback.await_count == 2 - count
    assert result.provenance.runtime_fingerprint == "runtime"
    prompt = llm._client.chat.call_args.kwargs["messages"][0]["content"]
    assert "PHOTO 0" in prompt and "PHOTO 1" in prompt
    assert str(refs[0].part_id) in prompt


@pytest.mark.parametrize("runtime", [None, "other"])
async def test_resume_never_trusts_unknown_or_changed_live_runtime(photos, runtime):
    ocr = sidecar()
    ocr.get_capabilities.return_value.runtime_fingerprint = runtime
    page = OcrResponse.model_validate(
        wire_page(0, hashlib.sha256(photos[0].read_bytes()).hexdigest())
    )
    with pytest.raises(ValueError, match="resume"):
        await pipeline(formatter(False), ocr).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="ocr"),
            deadline=time.monotonic() + 2,
            tags=[],
            resume_pages=[page],
        )
    ocr.extract_page.assert_not_awaited()


async def test_changed_runtime_between_readiness_and_page_never_falls_back(photos):
    ocr = sidecar()
    ocr.get_capabilities.return_value.runtime_fingerprint = "other"
    llm = formatter()
    with pytest.raises(ValueError, match="runtime_changed"):
        await pipeline(llm, ocr, allow_vision_fallback=True).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="ocr", allow_ocr_fallback=True),
            deadline=time.monotonic() + 2,
            tags=[],
        )
    llm._client.chat.assert_not_awaited()


@pytest.mark.parametrize("intent", ["direct_vision", "ocr"])
async def test_saved_formatter_controls_repair_and_enrichment_without_shared_mutation(photos, intent):
    llm = formatter()
    llm.update_runtime_settings(temperature=0.99)
    llm.max_output_tokens = 9999
    valid = llm._client.chat.return_value

    async def reply(**kwargs):
        if llm._client.chat.await_count == 1:
            return ChatResponse(message={"role": "assistant", "content": "invalid JSON"})
        return valid

    llm._client.chat.side_effect = reply
    result = await pipeline(llm, sidecar() if intent == "ocr" else None).extract(
        "fixture",
        photos,
        options=ExperimentOptions(
            pipeline=intent, temperature=0.13, include_classification=True
        ),
        deadline=time.monotonic() + 90,
        tags=["food"],
        max_output_tokens=1234,
    )
    assert result.receipt is not None
    assert llm._client.chat.await_count >= 3  # formatter, repair, enrichment
    for call in llm._client.chat.call_args_list:
        assert call.kwargs["options"] == {"temperature": 0.13, "num_predict": 1234}
        if intent == "ocr":
            assert all("images" not in message for message in call.kwargs["messages"])
    assert llm._temperature == 0.99 and llm.max_output_tokens == 9999
    assert llm.experiment_options is None and llm.experiment_trace is None


async def test_changed_original_binding_is_explicit_integrity_error(photos):
    from uuid import uuid4

    import vision_bill.service.extraction_pipeline as module
    from vision_bill.model.ocr import SourceRef

    refs = [
        SourceRef(
            part_id=uuid4(), photo_index=i, source_sha256="a" * 64, coordinate_space="original"
        )
        for i in range(2)
    ]
    with pytest.raises(ValueError) as failure:
        await pipeline(formatter()).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="direct_vision"),
            deadline=time.monotonic() + 2,
            tags=[],
            source_refs=refs,
        )
    assert hasattr(module, "OriginalSourceChangedError")
    assert isinstance(failure.value, module.OriginalSourceChangedError)


async def test_dedicated_off_real_route_executes_direct_upload_and_holds_explicit_ocr(photos, tmp_path, monkeypatch):
    from ocr_contract.activation import ActivationPhase, ActivationStatus, ActivationStore

    from vision_bill.config import settings
    from vision_bill.model.handoff import HandoffWait
    from vision_bill.service import activation_admission
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation
    from vision_bill.service.experiment_execution import execute_submission
    from vision_bill.service.receipt_service import ReceiptService

    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    monkeypatch.setenv("VISION_BILL_OCR_ACTIVATION", "true")
    monkeypatch.setenv("OCR_ACTIVATION_ENDPOINT", "http://prepared-ocr.test")
    monkeypatch.setenv("OCR_ACTIVATION_HANDOFF_LOCK_PATH", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(activation_admission, "ActivationStore", lambda: store)
    monkeypatch.setattr(settings, "ocr", OcrSettings(enabled=False, coordination_path=tmp_path / "old.lock"))
    monkeypatch.setattr(settings.llm, "model_name", "fixture")
    llm = formatter()
    monkeypatch.setattr(llm, "resident_model_ids", AsyncMock(return_value=frozenset()))
    monkeypatch.setattr(llm, "release_model", AsyncMock())
    receipts = object.__new__(ReceiptService)
    receipts._provider = llm
    receipts.set_ocr_route(None, settings.ocr)
    activation = DedicatedOcrActivation(settings, llm, receipts.set_ocr_route, lambda: None,
                                         store=store, handoff_path=tmp_path / "gpu.lock")
    monkeypatch.setattr(activation, "_controller_is_live", AsyncMock(return_value=True))
    monkeypatch.setattr(activation, "_runtime_is_serving", AsyncMock(return_value=True))
    monkeypatch.setattr(activation.ocr, "claim", AsyncMock())
    monkeypatch.setattr(activation.ocr, "park", AsyncMock())
    monkeypatch.setattr(activation.ocr, "load", AsyncMock())
    monkeypatch.setattr(activation.ocr, "get_capabilities", AsyncMock(return_value=OcrCapabilities(enabled=True, configured=True, reachable=True, ready=True)))
    assert (await activation.submit(confirm_handoff=True)).state == "requested"
    request = store.read_request()
    store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
    await activation.reconcile()
    assert (await activation.submit(confirm_handoff=False, enabled=False)).state == "disabled"
    assert receipts.execution_handoff(settings.ocr) is activation.handoff
    result = await receipts.extract_receipt(
        "fixture", photos, ocr_settings=settings.ocr,
        options=ExperimentOptions(include_classification=False), deadline=time.monotonic() + 5, tags=[],
    )
    assert result.receipt is not None
    assert len(llm._client.chat.call_args.kwargs["messages"][0]["images"]) == 2
    with pytest.raises(HandoffWait, match="dedicated_ocr_disabled"):
        await execute_submission(receipts, "fixture", photos, ExperimentOptions(pipeline="ocr"), [])
    activation.ocr.load.assert_not_awaited()
    assert not activation.handoff.read_journal().unsettled
    from uuid import uuid4

    from vision_bill.service.benchmark_scoring import PROMPT_VERSION
    from vision_bill.service.benchmark_service import BenchmarkService

    queued = object.__new__(BenchmarkService)
    queued._provider, queued._receipts = llm, receipts
    queued._settings = SimpleNamespace(max_attempts=3)
    queued._db = SimpleNamespace(hold_before_inference=AsyncMock(return_value=True),
                                 begin_attempt=AsyncMock(), finish_task=AsyncMock())
    queued._run_digests = AsyncMock(return_value={"fixture": None})
    task = {"receipt_id": uuid4(), "model_id": "fixture", "inference_attempt_count": 0,
            "inference_started": False}
    manifest = {"inputs": [{"receipt_id": str(task["receipt_id"]), "pipeline": "ocr"}],
                "options": ExperimentOptions(pipeline="ocr").model_dump(mode="json")}
    started = asyncio.get_running_loop().time()
    run = SimpleNamespace(model_ids=["fixture"], prompt_version=PROMPT_VERSION)
    await queued._execute_reserved(task, run, manifest, started, started + 5)
    queued._db.hold_before_inference.assert_awaited_once_with(
        task, "dedicated_ocr_disabled", deadline=started + 5,
    )
    queued._db.begin_attempt.assert_not_awaited()
    queued._db.finish_task.assert_not_awaited()
    activation.ocr.load.assert_not_awaited()


async def test_complete_cache_formats_with_parked_owner_without_native_load(photos, monkeypatch):
    from vision_bill.service import extraction_pipeline

    ocr, llm = sidecar(), formatter(False)
    caps = ocr.get_capabilities.return_value
    caps.runtime_fingerprint = "runtime"
    caps.ready = False
    caps.can_load = True
    owner = SimpleNamespace(deadline=time.monotonic() + 5, llm_phase=AsyncMock(), ocr_phase=AsyncMock())
    monkeypatch.setattr(extraction_pipeline, "current_owner", lambda: owner)
    pages = [
        OcrResponse.model_validate(wire_page(i, hashlib.sha256(path.read_bytes()).hexdigest()))
        for i, path in enumerate(photos)
    ]
    result = await pipeline(llm, ocr).extract(
        "fixture", photos, options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=owner.deadline, tags=[], resume_pages=pages,
    )
    assert result.receipt is not None
    ocr.extract_page.assert_not_awaited()
    owner.ocr_phase.assert_not_awaited()
    owner.llm_phase.assert_awaited_once_with("fixture")
