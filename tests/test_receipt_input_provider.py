"""The shared parser and repair path keeps nullable values and private attempts."""

import json
import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from ollama import ChatResponse, ListResponse, ShowResponse
from test_ocr_provider import wire_page

from vision_bill.model.extraction import ReceiptInput
from vision_bill.model.ocr import OcrResponse, SourceRef
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider


def ocr_input():
    page = OcrResponse.model_validate(wire_page())
    ref = SourceRef(
        part_id=uuid4(),
        photo_index=0,
        source_sha256=page.source_sha256,
        view_id=page.working_image.view_id,
        coordinate_space="working",
    )
    return ReceiptInput(
        mode="ocr", image_paths=["original.gif"], ocr_pages=[page], source_refs=[ref]
    )


@pytest.mark.parametrize("backend", ["ollama", "openai"])
async def test_partial_null_draft_retains_actual_model_finish_and_bounded_attempts(backend, caplog):
    provider = (
        OllamaProvider("http://local")
        if backend == "ollama"
        else OpenAIProvider("http://local", "local")
    )
    payload = json.dumps({"merchant_name": "Private Shop", "total": None, "line_items": None})
    if backend == "ollama":
        provider._client.chat = AsyncMock(
            return_value=ChatResponse(
                model="actual",
                done_reason="length",
                message={"role": "assistant", "content": payload},
            )
        )
    else:
        provider._client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                model="actual",
                choices=[
                    SimpleNamespace(
                        finish_reason="length", message=SimpleNamespace(content=payload)
                    )
                ],
                usage=None,
            )
        )
    assert hasattr(provider, "extract_receipt_input"), "shared typed provider path missing"
    output = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert output.receipt is None and output.draft.merchant_name == "Private Shop"
    assert output.draft.total is None and output.draft.discount_total is None
    assert output.telemetry.calls[-1]["actual_model"] == "actual"
    assert output.telemetry.finish_reason == "length"
    assert output.telemetry.output_tokens is None
    assert len(output.telemetry.calls) == 1
    assert output.findings == []
    assert "Private Shop" not in caplog.text
    call = (
        provider._client.chat.call_args
        if backend == "ollama"
        else provider._client.chat.completions.create.call_args
    )
    assert (
        call.kwargs["options"]["num_predict"] if backend == "ollama" else call.kwargs["max_tokens"]
    ) == 8192


async def test_ollama_text_discovery_includes_completion_without_changing_default():
    provider = OllamaProvider("http://local")
    provider._client.list = AsyncMock(
        return_value=ListResponse(models=[{"model": "text"}, {"model": "vision"}])
    )
    provider._client.show = AsyncMock(
        side_effect=lambda name: ShowResponse(
            capabilities=["completion"] + (["vision"] if name == "vision" else []),
            model_info={"llama.context_length": 32768},
        )
    )
    assert [m.id for m in await provider.get_available_models()] == ["vision"]
    assert (
        "capability" in __import__("inspect").signature(provider.get_available_models).parameters
    ), "explicit discovery filter missing"
    models = await provider.get_available_models(capability="text")
    assert [m.id for m in models] == ["text", "vision"]
    assert models[0].context_window_tokens == 32768


async def test_openai_text_discovery_keeps_unknown_text_models():
    provider = OpenAIProvider("http://local", "local")
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {
                        "data": [
                            {"id": "text"},
                            {
                                "id": "vision",
                                "architecture": {"input_modalities": ["image", "text"]},
                            },
                        ]
                    }
                ).encode()
            )
        )
    )
    assert [m.id for m in await provider.get_available_models()] == ["vision"]
    assert (
        "capability" in __import__("inspect").signature(provider.get_available_models).parameters
    ), "explicit discovery filter missing"
    assert [m.id for m in await provider.get_available_models(capability="text")] == [
        "text",
        "vision",
    ]


async def test_source_aware_overlap_keeps_original_draft_and_verified_decisions():
    from test_multi_image_extraction import extraction, item

    from vision_bill.model.experiment import ExperimentOptions

    inp = ocr_input()
    page = OcrResponse.model_validate(wire_page(1, "b" * 64))
    ref = SourceRef(
        part_id=uuid4(),
        photo_index=1,
        source_sha256="b" * 64,
        view_id=page.working_image.view_id,
        coordinate_space="working",
    )
    inp = ReceiptInput(
        mode="ocr",
        image_paths=["a", "b"],
        ocr_pages=[*inp.ocr_pages, page],
        source_refs=[*inp.source_refs, ref],
    )
    payload = extraction([item(), item(), item(source=1), item("Bread", source=1)], [(1, 2)])
    payload.update(currency="EUR", discount_total="0", tax_total="0", taxes=[], payment_method="cash")
    payload["overlap_decisions"] = [{"reason": "printed suffix and prefix"}]
    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    assert hasattr(provider, "extract_receipt_input"), "typed path missing"
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert len(result.draft.line_items) == 4
    assert len(result.receipt.line_items) == 3
    assert result.field_sources["line_items.2"] == [ref]
    assert result.overlap_decisions[-1]["verified"] is True
    assert result.overlap_decisions[-1]["removed_index"] == 2

    from vision_bill.service.observation_admission import admit_observations

    assert [row.description for row in admit_observations(result).line_items] == ["Milk", "Milk", "Bread"]


@pytest.mark.parametrize("field", ["taxes", "payment_method", "currency", "tax_total", "subtotal", "total"])
@pytest.mark.parametrize("explicit_null", [False, True])
async def test_unobserved_mandatory_values_do_not_dispatch_classification(field, explicit_null):
    from test_experiment_scoring import receipt

    values = receipt().model_dump(mode="json")
    if explicit_null:
        values[field] = None
    else:
        del values[field]
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(return_value=json.dumps(values))
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None
    assert result.draft is not None
    assert getattr(result.draft, field) is None
    assert provider.send_message.await_count == 1
    assert result.classification_enrichment.state == "skipped"
    assert result.classification_enrichment.reason_code == "incomplete_observations"


@pytest.mark.parametrize("explicit_null", [False, True])
async def test_unobserved_optional_values_survive_formatting(explicit_null):
    from test_experiment_scoring import receipt

    values = receipt(("Milk",)).model_dump(mode="json")
    for obj, fields in (
        (values, ("discount_total", "tip")),
        (values["line_items"][0], ("quantity", "unit_price")),
    ):
        for field in fields:
            if explicit_null:
                obj[field] = None
            else:
                del obj[field]
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(return_value=json.dumps(values))

    result = await provider.extract_observations(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )

    assert result.receipt is not None and result.draft is not None
    for observed in [result.receipt, result.draft]:
        assert observed.discount_total is None and observed.tip is None
        assert observed.line_items[0].quantity is None
        assert observed.line_items[0].unit_price is None
        assert observed.line_items[0].total_price == Decimal("2.00")


async def test_observation_stage_preserves_receipt_and_empty_tag_enrichment():
    from test_experiment_scoring import receipt

    values = receipt(("Milk",)).model_dump(mode="json")
    del values["category"]
    del values["line_items"][0]["spending_category"]
    del values["line_items"][0]["tags"]
    labels = {"category": "grocery", "line_items": [{"spending_category": "milk", "tags": []}]}
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(side_effect=[json.dumps(values), json.dumps(labels)])
    deadline = time.monotonic() + 90
    observed = await provider.extract_observations("requested", ocr_input(), tags=[], deadline=deadline)
    assert provider.send_message.await_count == 1
    assert observed.classification_enrichment.state == "pending"
    assert observed.classification_enrichment.missing_fields == [
        "category", "line_items.0.spending_category", "line_items.0.tags",
    ]
    enriched = await provider.enrich_observations("requested", observed, [], work_deadline=deadline)
    assert enriched.classification_enrichment.state == "completed"
    assert enriched.classification_enrichment.missing_fields == []
    assert enriched.receipt.category == "grocery"
    assert enriched.receipt.line_items[0].tags == []
    assert enriched.draft.line_items[0].tags is None
    assert observed.classification_enrichment.state == "pending"


async def test_fully_observed_labels_complete_without_dispatch():
    from test_experiment_scoring import receipt

    values = receipt(("Milk",)).model_dump(mode="json")
    values["category"] = "grocery"
    values["line_items"][0].update(spending_category="milk", tags=[])
    conflicting = {"category": "electronics", "line_items": [{"spending_category": "cheese", "tags": ["changed"]}]}
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(side_effect=[json.dumps(values), json.dumps(conflicting)])
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt.category == "grocery"
    assert result.receipt.line_items[0].spending_category == "milk"
    assert result.receipt.line_items[0].tags == []
    assert result.classification_enrichment.model_dump() == {
        "state": "completed", "reason_code": None, "missing_fields": [],
    }
    assert provider.send_message.await_count == 1
    assert len(result.telemetry.calls) == 1


@pytest.mark.parametrize("include_conflicts", [False, True])
async def test_enrichment_fills_only_missing_labels_on_partly_observed_rows(include_conflicts):
    from test_experiment_scoring import receipt

    values = receipt().model_dump(mode="json")
    values["category"] = "grocery"
    values["line_items"][0].update(spending_category="milk", tags=[])
    values["line_items"][1].update(spending_category="bread_bakery", tags=None)
    labels = {"line_items": [{}, {"tags": ["bread"]}]}
    if include_conflicts:
        labels["category"] = "electronics"
        labels["line_items"][0].update(spending_category="cheese", tags=["changed"])
        labels["line_items"][1]["spending_category"] = "fish"
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(side_effect=[json.dumps(values), json.dumps(labels)])
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 90
    )
    assert result.receipt.category == "grocery"
    assert [(row.spending_category, row.tags) for row in result.receipt.line_items] == [
        ("milk", []), ("bread_bakery", ["bread"]),
    ]
    assert [row.tags for row in result.draft.line_items] == [[], None]
    assert result.classification_enrichment.model_dump() == {
        "state": "completed", "reason_code": None, "missing_fields": [],
    }
    assert provider.send_message.await_count == 2


async def test_enrichment_accepts_category_only_when_all_row_labels_are_observed():
    from test_experiment_scoring import receipt

    values = receipt(("Milk",)).model_dump(mode="json")
    del values["category"]
    values["line_items"][0].update(spending_category="milk", tags=["dairy"])
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(
        side_effect=[json.dumps(values), '{"category":"grocery"}']
    )
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 90
    )
    assert result.receipt.category == "grocery"
    assert result.receipt.line_items[0].spending_category == "milk"
    assert result.receipt.line_items[0].tags == ["dairy"]
    assert result.classification_enrichment.state == "completed"
    assert result.classification_enrichment.missing_fields == []


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("{", id="malformed-json"),
        pytest.param('{"category":"grocery","line_items":[]}', id="wrong-row-count"),
        pytest.param("[" * 10000 + "]" * 10000, id="nested-json"),
    ],
)
async def test_bad_classification_is_unavailable_without_erasing_labels_or_creating_findings(response):
    from test_experiment_scoring import receipt

    values = receipt(("Milk",)).model_dump(mode="json")
    del values["category"]
    values["line_items"][0]["spending_category"] = "milk"
    values["line_items"][0]["tags"] = ["observed"]
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(side_effect=[json.dumps(values), response])
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 90
    )
    assert result.receipt.line_items[0].spending_category == "milk"
    assert result.receipt.line_items[0].tags == ["observed"]
    assert result.findings == []
    assert result.classification_enrichment.state == "unavailable"
    assert result.classification_enrichment.reason_code == "invalid_response"
    assert result.classification_enrichment.missing_fields == ["category"]


async def test_unknown_source_ref_cannot_survive_as_valid_draft():
    from test_experiment_scoring import receipt

    inp = ocr_input()
    invalid = inp.source_refs[0].model_copy(update={"source_sha256": "c" * 64})
    payload = receipt().model_dump(mode="json")
    payload["field_sources"] = {"merchant_name": [invalid.model_dump(mode="json")]}
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None and result.draft is None
    assert any(f.code == "formatter_invalid_response" for f in result.findings)


async def test_classification_is_separate_cannot_change_observations_and_is_in_telemetry():
    from test_experiment_scoring import receipt

    values = receipt().model_dump(mode="json")
    for row in values["line_items"]:
        del row["tags"]
    provider = OllamaProvider("http://local")
    provider._client.chat = AsyncMock(
        side_effect=[
            ChatResponse(
                model="extractor",
                message={"role": "assistant", "content": json.dumps(values)},
            ),
            ChatResponse(
                model="classifier",
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "total": "99999",
                            "category": "grocery",
                            "line_items": [
                                {
                                    "description": "changed",
                                    "tags": ["food"],
                                    "spending_category": "food",
                                }
                                for _ in receipt().line_items
                            ],
                        }
                    ),
                },
            ),
        ]
    )
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=["food"], deadline=time.monotonic() + 90
    )
    assert result.receipt.total == receipt().total
    assert [r.description for r in result.receipt.line_items] == [
        r.description for r in receipt().line_items
    ]
    assert len(result.telemetry.calls) == 2
    assert result.telemetry.calls[1]["stage"] == "classification"
    assert result.classification_enrichment.state == "unavailable"
    assert result.classification_enrichment.missing_fields == [
        "line_items.0.spending_category", "line_items.1.spending_category",
    ]
    assert all(row.tags == ["food"] for row in result.receipt.line_items)


async def test_supplied_and_explicit_null_fields_are_distinguished():
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(return_value='{"merchant_name":"Shop","total":null}')
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.telemetry.calls[0]["supplied_fields"] == ["merchant_name", "total"]
    assert result.telemetry.calls[0]["null_fields"] == ["total"]
    assert result.draft.discount_total is None


async def test_openai_discovery_records_only_declared_context_capacity():
    provider = OpenAIProvider("http://local", "local")
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {"data": [{"id": "text", "context_length": 16384}, {"id": "unknown"}]}
                ).encode()
            )
        )
    )
    models = await provider.get_available_models(capability="text")
    assert models[0].context_window_tokens == 16384
    assert models[1].context_window_tokens is None


@pytest.mark.parametrize("sources", [{"total": 5}, {"total": None}])
async def test_malformed_field_sources_trigger_bounded_repairs(sources):
    from test_experiment_scoring import receipt

    provider = OllamaProvider("http://local")
    payload = receipt().model_dump(mode="json")
    payload["field_sources"] = sources
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None
    assert len(result.telemetry.calls) == 3


async def test_formatter_does_not_receive_wire_telemetry_or_image_transform_payloads():
    from test_experiment_scoring import receipt

    inp = ocr_input()
    inp.ocr_pages[0].telemetry = {"png_base64": "PRIVATE_IMAGE_PAYLOAD", "pixels": [[[255]]]}
    inp.ocr_pages[0].working_image.transforms = [{"raw_image": "PRIVATE_TRANSFORM_PAYLOAD"}]
    from vision_bill.model.experiment import ExperimentOptions

    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=receipt().model_dump_json())
    await provider.extract_receipt_input("requested", inp, tags=[], deadline=time.monotonic() + 2)
    content = provider.send_message.call_args.args[1][0]["content"]
    assert "PRIVATE_IMAGE_PAYLOAD" not in content
    assert "PRIVATE_TRANSFORM_PAYLOAD" not in content
    assert "pixels" not in content


@pytest.mark.parametrize("backend", ["ollama", "openai"])
async def test_formatter_discovery_excludes_embedding_only_models(backend):
    if backend == "ollama":
        provider = OllamaProvider("http://local")
        provider._client.list = AsyncMock(
            return_value=ListResponse(models=[{"model": "embed"}, {"model": "text"}])
        )
        provider._client.show = AsyncMock(
            side_effect=lambda name: ShowResponse(
                capabilities=["embedding"] if name == "embed" else ["completion"], model_info={}
            )
        )
    else:
        provider = OpenAIProvider("http://local", "local")
        provider._client.models.with_raw_response.list = AsyncMock(
            return_value=SimpleNamespace(
                http_response=SimpleNamespace(
                    content=json.dumps(
                        {
                            "data": [
                                {"id": "embed", "capabilities": ["embedding"]},
                                {"id": "text", "capabilities": ["completion"]},
                            ]
                        }
                    ).encode()
                )
            )
        )
    assert [model.id for model in await provider.get_available_models(capability="text")] == [
        "text"
    ]


async def test_openai_data_capabilities_are_honored_for_direct_discovery():
    provider = OpenAIProvider("http://local", "local")
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {
                        "data": [
                            {"id": "vision", "capabilities": ["vision"]},
                            {"id": "text", "capabilities": ["completion"]},
                        ]
                    }
                ).encode()
            )
        )
    )
    assert [model.id for model in await provider.get_available_models()] == ["vision"]


async def test_null_overlap_candidates_are_repaired_without_crashing():
    from test_experiment_scoring import receipt

    provider = OllamaProvider("http://local")
    payload = receipt().model_dump(mode="json")
    payload["duplicate_candidates"] = None
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None and len(result.telemetry.calls) == 3


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "text", "capabilities": ["completion"]},
        {"id": "text", "capabilities": ["embedding"]},
        {"id": "text", "architecture": {"input_modalities": ["text"]}},
    ],
)
async def test_configured_vision_does_not_override_known_text_metadata(entry):
    assert "vision_model_ids" in __import__("inspect").signature(OpenAIProvider).parameters
    provider = OpenAIProvider("http://local", "local", vision_model_ids=["text"])
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(content=json.dumps({"data": [entry]}).encode())
        )
    )
    assert await provider.get_available_models() == []


async def test_undeclared_standard_openai_model_keeps_vision_discovery_default():
    provider = OpenAIProvider("http://local", "local")
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(content=b'{"data":[{"id":"unknown"}]}')
        )
    )
    assert await provider.get_available_models() == []
    assert [model.id for model in await provider.get_available_models(capability="text")] == [
        "unknown"
    ]


@pytest.mark.parametrize("mode", ["ocr", "direct_vision"])
@pytest.mark.parametrize(
    "added",
    [
        {"bbox": [[0, 0], [1, 0], [1, 1]]},
        {"text_span": [0, 4]},
        {"block_id": "invented"},
    ],
)
async def test_identity_does_not_authorize_new_geometry(mode, added, tmp_path):
    from PIL import Image
    from test_experiment_scoring import receipt

    inp = ocr_input()
    if mode == "direct_vision":
        path = tmp_path / "real.png"
        Image.new("RGB", (3, 4)).save(path)
        inp = ReceiptInput(mode=mode, image_paths=[str(path)], source_refs=inp.source_refs)
    ref = inp.source_refs[0].model_dump(mode="json") | added
    payload = receipt().model_dump(mode="json") | {"field_sources": {"merchant_name": [ref]}}
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None
    assert "merchant_name" not in result.field_sources


@pytest.mark.parametrize("mode", ["ocr", "direct_vision"])
async def test_exact_supplied_geometry_is_preserved(mode, tmp_path):
    from PIL import Image
    from test_experiment_scoring import receipt

    inp = ocr_input()
    known = inp.source_refs[0].model_copy(
        update={"bbox": [[0, 0], [1, 0], [1, 1]], "text_span": (0, 4)}
    )
    path = tmp_path / "real.png"
    Image.new("RGB", (3, 4)).save(path)
    inp = ReceiptInput(
        mode=mode,
        image_paths=[str(path)],
        ocr_pages=inp.ocr_pages if mode == "ocr" else [],
        source_refs=[known],
    )
    payload = receipt().model_dump(mode="json") | {
        "field_sources": {"merchant_name": [known.model_dump(mode="json")]}
    }
    from vision_bill.model.experiment import ExperimentOptions

    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is not None
    assert result.field_sources["merchant_name"] == [known]


@pytest.mark.parametrize("span, accepted", [((0, 9), True), ((1, 5), False), ((0, 99), False)])
async def test_ocr_geometry_must_match_block_and_exact_transcript_span(span, accepted):
    from test_experiment_scoring import receipt

    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.ocr import OcrBlock

    inp = ocr_input()
    block = OcrBlock(
        id="printed", label="text", text=inp.ocr_pages[0].text, bbox=[[0, 0], [1, 0], [1, 1]]
    )
    inp.ocr_pages[0].blocks = [block]
    ref = inp.source_refs[0].model_copy(
        update={"bbox": block.bbox, "block_id": block.id, "text_span": span}
    )
    payload = receipt().model_dump(mode="json") | {
        "field_sources": {"merchant_name": [ref.model_dump(mode="json")]}
    }
    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert (result.receipt is not None) is accepted


@pytest.mark.parametrize("stage", ["extraction", "classification"])
async def test_blocked_parsing_cannot_return_late_success_or_mutate_input(monkeypatch, stage):
    import asyncio
    import threading

    from test_experiment_scoring import receipt

    from vision_bill.model.experiment import ExperimentOptions

    inp = ocr_input()
    before = inp.model_dump()
    response = receipt().model_dump_json()
    classification = json.dumps(
        {"category": "grocery", "line_items": [{"tags": ["late"]} for _ in receipt().line_items]}
    )
    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=stage == "classification")
    )
    provider.send_message = AsyncMock(side_effect=[response, classification])
    gate, entered, finished = threading.Event(), threading.Event(), threading.Event()
    original = json.loads
    blocked = response if stage == "extraction" else classification

    def loads(raw, *args, **kwargs):
        if raw == blocked:
            entered.set()
            gate.wait(0.5)
            try:
                return original(raw, *args, **kwargs)
            finally:
                finished.set()
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(json, "loads", loads)
    timer = threading.Timer(0.5, gate.set)
    timer.start()
    began = time.monotonic()
    try:
        if stage == "classification":
            result = await provider.extract_receipt_input(
                "requested", inp, tags=["food"], deadline=began + 60.15
            )
            assert result.receipt.total == receipt().total
            assert all(row.tags == [] for row in result.receipt.line_items)
            assert result.classification_enrichment.state == "unavailable"
            assert result.classification_enrichment.reason_code == "classification_budget_exhausted"
        else:
            with pytest.raises(TimeoutError):
                await provider.extract_receipt_input(
                    "requested", inp, tags=["food"], deadline=began + 0.08
                )
        assert entered.is_set()
        assert time.monotonic() - began < 0.3
    finally:
        gate.set()
        timer.cancel()
    assert await asyncio.to_thread(finished.wait, 0.5)
    assert inp.model_dump() == before
    assert provider.experiment_trace == []


async def test_ocr_repeated_text_span_cannot_point_to_another_block():
    from test_experiment_scoring import receipt

    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.ocr import OcrBlock

    inp = ocr_input()
    inp.ocr_pages[0].blocks = [
        OcrBlock(
            id=str(index),
            label="text",
            text="Milk 2.00",
            bbox=[[0, index], [1, index], [1, index + 1]],
        )
        for index in range(2)
    ]
    inp.ocr_pages[0].text = "Milk 2.00\nMilk 2.00"
    # Matching text is insufficient: block1 occupies the second occurrence, not the first.
    ref = inp.source_refs[0].model_copy(
        update={"block_id": "1", "bbox": inp.ocr_pages[0].blocks[1].bbox, "text_span": (0, 9)}
    )
    payload = receipt().model_dump(mode="json") | {
        "field_sources": {"merchant_name": [ref.model_dump(mode="json")]}
    }
    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    output = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert output.receipt is None


async def test_final_rows_project_distinct_blocks_after_accepted_boundary_overlap():
    from test_multi_image_extraction import extraction, item

    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.ocr import OcrBlock

    pages = [OcrResponse.model_validate(wire_page(i, str(i + 1) * 64)) for i in range(2)]
    refs = [
        SourceRef(
            part_id=uuid4(),
            photo_index=i,
            source_sha256=page.source_sha256,
            view_id=page.working_image.view_id,
            coordinate_space="working",
        )
        for i, page in enumerate(pages)
    ]
    sources = {}
    for i, names in enumerate((("A", "B"), ("B", "C"))):
        for j, name in enumerate(names):
            block = OcrBlock(
                id=f"{i}-{j}", label="text", text=name, bbox=[[j, 0], [j + 1, 0], [j + 1, 1]]
            )
            pages[i].blocks.append(block)
            source = refs[i].model_copy(update={"block_id": block.id, "bbox": block.bbox})
            sources[f"line_items.{i * 2 + j}.description"] = [source.model_dump(mode="json")]
    inp = ReceiptInput(mode="ocr", image_paths=["a", "b"], ocr_pages=pages, source_refs=refs)
    payload = extraction([item("A"), item("B"), item("B", source=1), item("C", source=1)], [(1, 2)])
    payload.update(
        currency="EUR", discount_total="0", tax_total="0", taxes=[], payment_method="cash"
    )
    payload["field_sources"] = sources
    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert [row.description for row in result.receipt.line_items] == ["A", "B", "C"]
    assert [result.field_sources[f"line_items.{i}.description"][0].block_id for i in range(3)] == [
        "0-0",
        "0-1",
        "1-1",
    ]
    assert "line_items.3.description" not in result.field_sources
    assert [row.description for row in result.draft.line_items] == ["A", "B", "B", "C"]
    assert result.raw_field_sources["line_items.2.description"][0].block_id == "1-0"
    assert result.source_projection_version == "2"
    assert result.overlap_decisions[-1]["removed_index"] == 2


def test_historical_envelope_without_projection_identity_remains_legacy():
    from vision_bill.model.extraction import InferenceEnvelope

    envelope = InferenceEnvelope.model_validate(
        {
            "provenance": {
                "requested_pipeline": "ocr",
                "used_pipeline": "ocr",
                "source_image_count": 1,
                "ordered_source_sha256": ["a" * 64],
                "model_id": "old",
            },
            "telemetry": {"elapsed_ms": 1, "max_output_tokens": 8192},
        }
    )
    assert envelope.source_projection_version == "1"


def test_direct_view_identity_is_fresh_per_preparation_without_changing_pixels(tmp_path):
    from hashlib import sha256

    from PIL import Image

    from vision_bill.provider.llm.openai import OpenAIProvider

    path = tmp_path / "same.png"
    Image.new("RGB", (8, 8), "white").save(path)
    digest = sha256(path.read_bytes()).hexdigest()
    provider = OpenAIProvider("http://unused/v1", "test")
    first_ref = SourceRef(
        part_id=uuid4(), photo_index=0, source_sha256=digest, coordinate_space="original"
    )
    second_ref = first_ref.model_copy(update={"part_id": uuid4()})

    def prepare(ref):
        return provider._prepare_receipt_input(
            ReceiptInput(mode="direct_vision", image_paths=[str(path)], source_refs=[ref])
        )[2][0]

    first, second = prepare(first_ref), prepare(second_ref)
    assert first.working_image.sha256 == second.working_image.sha256
    assert first.working_image.view_id != second.working_image.view_id
    assert prepare(first_ref).working_image.view_id != first.working_image.view_id


@pytest.mark.parametrize("missing", ["null", "omitted"])
@pytest.mark.parametrize("count", [1, 2])
async def test_nullable_observations_stop_without_invalid_repair_or_financial_defaults(
    missing, count
):
    from vision_bill.model.experiment import ExperimentOptions

    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    payload = {
        "confidence": 95,
        "merchant_name": "Observed",
        "date": "2026-01-01",
        "line_items": [
            {
                "description": "A",
                "quantity": 1,
                "unit_price": "1",
                "total_price": "1",
                "source_image": 0,
            }
        ],
        "subtotal": "1",
        "total": "1",
        "payment_method": "cash",
    }
    if missing == "null":
        payload.update(currency=None, discount_total=None, tax_total=None)
    if count == 2:
        payload["duplicate_candidates"] = []
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    input = ocr_input()
    if count == 2:
        second = input.ocr_pages[0].model_copy(deep=True)
        second.photo_index = 1
        input.ocr_pages.append(second)
        input.image_paths.append("second.gif")
    result = await provider.extract_receipt_input(
        "m", input, tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None
    assert (
        result.draft.currency is None
        and result.draft.discount_total is None
        and result.draft.tax_total is None
    )
    assert result.findings == []
    assert len(result.telemetry.calls) == 1
    assert provider.send_message.await_count == 1


async def test_incomplete_observations_do_not_bypass_invalid_overlap_structure():
    from test_multi_image_extraction import extraction, item

    from vision_bill.model.experiment import ExperimentOptions

    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    payload = extraction([item("A"), item("B"), item("B", source=1), item("C", source=1)], [(0, 3)])
    payload.update(currency=None, discount_total=None, tax_total=None)
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    inp = ocr_input()
    page = inp.ocr_pages[0].model_copy(deep=True)
    page.photo_index = 1
    inp.ocr_pages.append(page)
    inp.image_paths.append("second")
    result = await provider.extract_receipt_input("m", inp, tags=[], deadline=time.monotonic() + 2)
    assert result.receipt is None and result.draft is None
    assert [f.code for f in result.findings] == ["formatter_invalid_response"]
    assert len(result.telemetry.calls) == 3


def test_real_original_bindings_supply_exact_working_refs_to_ocr_formatter():
    provider = OllamaProvider("http://local")
    inp = ocr_input()
    original = inp.source_refs[0].model_copy(
        update={"coordinate_space": "original", "view_id": None}
    )
    inp.source_refs = [original]
    prepared, messages, views = provider._prepare_receipt_input(inp)
    working = [
        ref
        for ref in provider._authorized_source_refs(prepared)
        if ref.coordinate_space == "working"
    ]
    assert len(working) == 1
    assert working[0].part_id == original.part_id
    assert working[0].source_sha256 == original.source_sha256
    assert working[0].view_id == inp.ocr_pages[0].working_image.view_id
    assert prepared.source_refs == [original]
    assert inp.source_refs == [original] and views == []
    assert str(working[0].view_id) in messages[0]["content"]
    wrong = original.model_copy(update={"source_sha256": "f" * 64})
    inp.source_refs = [wrong]
    with pytest.raises(ValueError, match="ocr_source_reference_mismatch"):
        provider._prepare_receipt_input(inp)


@pytest.mark.parametrize(
    "changed", [None, "part_id", "photo_index", "source_sha256", "view_id", "bbox"]
)
async def test_original_to_working_binding_accepts_only_exact_saved_page_geometry(changed):
    from test_experiment_scoring import receipt

    from vision_bill.model.experiment import ExperimentOptions

    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    inp = ocr_input()
    original = inp.source_refs[0].model_copy(
        update={"coordinate_space": "original", "view_id": None}
    )
    inp.source_refs = [original]
    from vision_bill.model.ocr import OcrBlock

    block = OcrBlock(id="merchant", label="text", text="Shop", bbox=[[0, 0], [2, 0], [2, 2]])
    inp.ocr_pages[0].blocks = [block]
    ref = original.model_copy(
        update={
            "coordinate_space": "working",
            "view_id": inp.ocr_pages[0].working_image.view_id,
            "block_id": block.id,
            "bbox": block.bbox,
        }
    ).model_dump(mode="json")
    if changed:
        ref[changed] = {
            "part_id": str(uuid4()),
            "photo_index": 1,
            "source_sha256": "e" * 64,
            "view_id": str(uuid4()),
            "bbox": [[0, 0], [1, 0], [1, 1]],
        }[changed]
    payload = receipt().model_dump(mode="json")
    payload["field_sources"] = {"merchant_name": [ref]}
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input("m", inp, tags=[], deadline=time.monotonic() + 2)
    if changed:
        assert result.receipt is None and result.draft is None
        assert len(result.telemetry.calls) == 3
    else:
        assert result.receipt is not None
        assert result.field_sources["merchant_name"][0].model_dump(mode="json") == ref
        assert result.field_sources["input"] == [original]
        assert result.provenance.ordered_source_sha256 == [original.source_sha256]


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
async def test_prompt_compliant_observations_admit_complete_without_raw_classification(
    classification, overlap
):
    from test_multi_image_extraction import extraction, item

    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.service.submission_service import SubmissionService

    inp = ocr_input()
    rows = [item()]
    if overlap:
        page = OcrResponse.model_validate(wire_page(1, "b" * 64))
        ref = inp.source_refs[0].model_copy(
            update={
                "part_id": uuid4(),
                "photo_index": 1,
                "source_sha256": "b" * 64,
                "view_id": page.working_image.view_id,
            }
        )
        inp = ReceiptInput(
            mode="ocr",
            image_paths=["a", "b"],
            ocr_pages=[*inp.ocr_pages, page],
            source_refs=[*inp.source_refs, ref],
        )
        rows += [item(source=1)]
    payload = extraction(rows, [(0, 1)] if overlap else [])
    payload.update(
        currency="EUR",
        payment_method="cash",
        discount_total="0",
        tax_total="0",
        taxes=[],
        subtotal="2.50",
        total="2.50",
        amount_policy={
            "tax_basis": "gross",
            "subtotal_origin": "printed",
            "tax_information": "explicit_zero",
            "discount_treatment": "included_in_lines",
            "tip_treatment": "none",
        },
    )
    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=classification)
    )
    provider.send_message = AsyncMock(
        side_effect=[
            json.dumps(payload),
            json.dumps(
                {
                    "category": "grocery",
                    "line_items": [{"tags": ["food"], "spending_category": "food"}],
                }
            ),
        ]
    )
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 90
    )
    outcome = SubmissionService._outcome(
        result,
        [],
        [
            OriginalAvailability(part_id=r.part_id, available=True, sha256_matches=True)
            for r in inp.source_refs
        ],
    )
    assert outcome.kind == "complete", outcome.review
    assert outcome.review.can_auto_verify
    assert result.draft.category is None
    assert all(row.tags is None for row in result.draft.line_items)
    assert len(result.draft.line_items) == (2 if overlap else 1)
    assert len(result.receipt.line_items) == 1
    assert result.receipt.line_items[0].tags == (["food"] if classification else [])
    prompt = provider.send_message.call_args_list[0].args[1][0]["content"]
    schema = json.loads(prompt.split("Nullable schema: ")[1].split("\nObservations:")[0])
    assert "amount_policy" in schema["properties"]
    policy_schema = schema["$defs"]["AmountPolicy"]["properties"]
    assert set(policy_schema["tax_basis"]["enum"]) == {"gross", "net", "unknown"}
    assert "explicit_zero" in policy_schema["tax_information"]["enum"]
