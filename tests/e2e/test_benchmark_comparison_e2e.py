"""Paired jobs freeze and commit together, but spend attempts independently."""

from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest

from tests.e2e import test_benchmark_references_e2e as references
from tests.test_extraction_pipeline import formatter, sidecar
from vision_bill.config import ImageSettings, OcrSettings
from vision_bill.model.benchmark import BenchmarkCreate
from vision_bill.model.experiment import ExperimentOptions
from vision_bill.model.workflow import AccessScope
from vision_bill.provider.llm.base import ModelInfo
from vision_bill.service.benchmark_service import BenchmarkService
from vision_bill.service.receipt_service import ReceiptService

postgres, db, workflow_db = references.postgres, references.db, references.workflow_db
pytestmark = pytest.mark.e2e


async def setup_comparison(w, scope, tmp_path):
    row, _, _ = await references.human(w, scope, tmp_path)
    provider, ocr = formatter(), sidecar()
    vision = provider.get_available_models.return_value[0]
    text = ModelInfo("formatter")
    text.capabilities = ["completion"]
    models = {"text": [text, vision], "vision": [vision], "all": [text, vision]}
    provider.get_available_models = AsyncMock(side_effect=lambda *, capability: models[capability])
    receipts = ReceiptService(
        ImageSettings(save_dir=str(tmp_path), tmp_dir=str(tmp_path)), w._settings,
        provider, ocr_provider=ocr, ocr_settings=OcrSettings(manage_model_lifecycle=False),
    )
    receipts._db = w._receipts
    service = BenchmarkService(provider, receipts)
    preview = await service.reference_preview(row.id, scope=scope)
    await service.qualify_reference(row.id, preview.review_revision, preview.document_sha256, scope=scope)
    request = BenchmarkCreate.model_validate({
        "comparison": True, "receipt_ids": [row.id], "model_ids": [text.id, vision.id],
        "request_timeout_seconds": 60,
        "experiment": {"include_classification": False, "temperature": 0, "pipeline": "direct_vision"},
    })
    return service, request, provider, ocr


async def test_comparison_freezes_eligible_models_same_gold_and_separate_work(workflow_db, tmp_path):
    w, scope = workflow_db
    service, request, provider, ocr = await setup_comparison(w, scope, tmp_path)
    primary = await service.create_run(request, scope=scope)
    assert primary.comparison_arm == "direct_vision"
    assert primary.comparison_id is not None
    runs = {r.comparison_arm: r for r in await service.list_runs(scope=scope)}
    assert runs["direct_vision"].id == primary.id
    sibling = runs["ocr"]
    assert sibling.comparison_id == primary.comparison_id
    assert sibling.id != primary.id
    assert primary.model_ids == ["fixture"]
    assert sibling.model_ids == ["formatter", "fixture"]
    assert primary.receipt_ids == sibling.receipt_ids == request.receipt_ids
    assert primary.dataset_fingerprint == sibling.dataset_fingerprint
    assert await service.list_runs(scope=AccessScope(user_id=uuid4())) == []
    direct = await service._db.manifest(primary.id)
    ocr_manifest = await service._db.manifest(sibling.id)
    for manifest, pipeline in ((direct, "direct_vision"), (ocr_manifest, "ocr")):
        assert manifest["options"]["pipeline"] == pipeline
        assert manifest["options"]["temperature"] == 0
        assert manifest["inputs"][0]["pipeline"] == pipeline
    assert direct["inputs"][0] == {**ocr_manifest["inputs"][0], "pipeline": "direct_vision"}
    provider._client.chat.assert_not_awaited()
    ocr.extract_page.assert_not_awaited()

    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM benchmark_runs WHERE request_timeout_seconds=60") == 2
    # Exhaust a single OCR task without touching the sibling's attempts or summary.
    async with w.pool.acquire() as conn:
        await conn.execute(
            """UPDATE benchmark_tasks SET inference_attempt_count=3 WHERE run_id=$1 AND model_id='formatter'""",
            sibling.id,
        )
        await conn.execute(
            """UPDATE benchmark_tasks SET retry_at=now()+interval '1 day'
               WHERE run_id=$1 OR model_id='fixture'""", primary.id,
        )
    task = dict(await service._db.lease_task())
    await service._execute(task)
    assert (await service.get_status(sibling.id, scope=scope)).terminal == 1
    assert (await service.get_status(primary.id, scope=scope)).terminal == 0
    assert (await service._db.results(sibling.id))[0]["full_receipt_correct"] is False
    assert {r.id for r in await service.list_runs(scope=scope, comparison_id=primary.comparison_id)} == {primary.id, sibling.id}
    assert await service.list_runs(scope=scope, comparison_id=uuid4()) == []
    query = {"scope": scope, "limit": 50, "offset": 0, "model_id": "formatter", "receipt_ids": request.receipt_ids}
    rows = await service.results(sibling.id, **query)
    assert len(rows) == 1 and rows[0]["success"] is False
    assert await service.results(sibling.id, **{**query, "model_id": "fixture"}) == []
    assert await service.results(sibling.id, **{**query, "receipt_ids": [uuid4()]}) == []
    provider._client.chat.assert_not_awaited()


async def test_comparison_rollback_leaves_no_primary_run(workflow_db, tmp_path):
    w, scope = workflow_db
    service, request, _, _ = await setup_comparison(w, scope, tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute("""ALTER TABLE benchmark_summaries ADD CHECK (model_id <> 'formatter')""")
    with pytest.raises(asyncpg.CheckViolationError):
        await service.create_run(request, scope=scope)
    async with w.pool.acquire() as conn:
        for table in ("benchmark_runs", "benchmark_tasks", "benchmark_summaries"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0


async def test_comparison_requires_pairable_model_and_unambiguous_intent(workflow_db, tmp_path):
    w, scope = workflow_db
    service, request, _, _ = await setup_comparison(w, scope, tmp_path)
    with pytest.raises(ValueError, match="vision-capable"):
        await service.create_run(request.model_copy(update={"model_ids": ["formatter"]}), scope=scope)
    with pytest.raises(ValueError, match="comparison"):
        await service.create_run(request.model_copy(update={"experiment": ExperimentOptions(pipeline="ocr")}), scope=scope)
    assert await service.list_runs(scope=scope) == []


async def test_mixed_inputs_remain_scoped_and_use_the_same_sparse_gold(workflow_db, tmp_path):
    from hashlib import sha256

    from vision_bill.model.benchmark_case import BenchmarkCaseUpdate
    from vision_bill.provider.db.benchmark_cases import BenchmarkCaseDB

    w, scope = workflow_db
    service, request, _, _ = await setup_comparison(w, scope, tmp_path)
    cases = BenchmarkCaseDB(w.pool)
    photo = tmp_path / "benchmark-source"
    photo.write_bytes(b"case photo")
    case = await cases.create(uuid4(), "Counter", [{
        "photo_index": 0, "path": str(photo), "sha256": sha256(photo.read_bytes()).hexdigest(),
    }], scope=scope)
    await cases.update(case.id, BenchmarkCaseUpdate.model_validate({
        "expected_revision": case.revision, "expected": {"total": "17.43"}, "confirm": True,
    }), scope=scope)
    selected = request.model_copy(update={"case_ids": [case.id]})
    primary = await service.create_run(selected, scope=scope)
    runs = await service.list_runs(scope=scope, comparison_id=primary.comparison_id)
    assert len(runs) == 2
    for arm in runs:
        assert arm.receipt_ids == [*selected.receipt_ids, case.id]
        assert arm.case_ids == [case.id] and arm.case_names == {case.id: "Counter"}
        assert arm.scoring_version == "5" and arm.dataset_fingerprint == primary.dataset_fingerprint
        manifest = await service._db.manifest(arm.id)
        assert manifest["scoring_version"] == "5"
        assert manifest["inputs"][1]["expected"]["total"] == "17.43"
        assert manifest["inputs"][1]["scored_fields"] == ["total"]
        async with w.pool.acquire() as conn:
            types = await conn.fetch("SELECT DISTINCT receipt_id,input_kind FROM benchmark_tasks WHERE run_id=$1", arm.id)
            assert {r["receipt_id"]: r["input_kind"] for r in types} == {case.id: "case", selected.receipt_ids[0]: "receipt"}
    # A later reference edit invalidates current qualification; the old run stays frozen.
    await cases.update(case.id, BenchmarkCaseUpdate.model_validate({
        "expected_revision": 2, "expected": {"total": "88.00"},
    }), scope=scope)
    with pytest.raises(ValueError, match="confirmation"):
        await service.create_run(selected, scope=scope)
    assert len(await service.list_runs(scope=scope)) == 2
    assert (await service._db.manifest(primary.id))["inputs"][1]["expected"]["total"] == "17.43"


async def test_downgrade_refuses_loss_of_comparison_identity(db):
    from tests.e2e.test_workflow_db_e2e import migrate

    migrate(db, "0023")
    conn = await asyncpg.connect(db.pg_dsn)
    comparison = uuid4()
    try:
        await conn.executemany(
            """INSERT INTO benchmark_runs(model_ids,receipt_ids,dataset_fingerprint,
               prompt_version,scoring_version,comparison_id,comparison_arm)
               VALUES('["fixture"]','[]','fixture','1','1',$1,$2)""",
            [(comparison, "direct_vision"), (comparison, "ocr")],
        )
        migration = migrate(db, "0022", "downgrade", check=False)
        assert migration.returncode != 0 and "benchmark comparison downgrade refused" in migration.stderr
        arms = await conn.fetch(
            "SELECT comparison_arm FROM benchmark_runs WHERE comparison_id=$1 ORDER BY comparison_arm", comparison
        )
        assert [row["comparison_arm"] for row in arms] == ["direct_vision", "ocr"]
    finally:
        await conn.close()
