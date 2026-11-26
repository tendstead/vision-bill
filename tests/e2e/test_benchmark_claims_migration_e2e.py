"""Forward-only benchmark claim accounting preserves historical evidence at upgrade."""

import json
from unittest.mock import AsyncMock

import asyncpg
import pytest

from tests.e2e import test_workflow_db_e2e as workflow_tests

db, migrate, postgres = workflow_tests.db, workflow_tests.migrate, workflow_tests.postgres

pytestmark = pytest.mark.e2e


async def test_0010_upgrade_preserves_reports_and_unknown_legacy_poll_baseline(db, tmp_path):
    migrate(db, "0010")
    conn = await asyncpg.connect(db.pg_dsn)
    try:
        original = tmp_path / "legacy.png"
        original.write_bytes(b"preserved original")
        image = await conn.fetchval("INSERT INTO images(image_path,status) VALUES($1,'analyzed') RETURNING id", str(original))
        receipt = await conn.fetchval("INSERT INTO receipts(merchant_name,date,total,status,verified,image_id) VALUES('legacy','2020-01-01',12.34,'verified',true,$1) RETURNING id", image)
        runs = []
        for version, status in (("2", "queued"), ("3", "running"), ("3", "completed")):
            run = await conn.fetchval("""INSERT INTO benchmark_runs(model_ids,receipt_ids,dataset_fingerprint,prompt_version,scoring_version,status,experiment_manifest)
                VALUES('["old"]',$1,'legacy','2',$2,$3,$4) RETURNING id""",
                json.dumps([str(receipt)]), version, status,
                json.dumps({"historical": "unchanged", "options": {}}) if version == "3" else None)
            runs.append(run)
            await conn.execute("INSERT INTO benchmark_summaries(run_id,model_id,completed,succeeded,failed,total_score,total_attempts) VALUES($1,'old',7,5,2,3.5,17)", run)
            if status == "completed":
                await conn.execute("INSERT INTO benchmark_results(run_id,model_id,receipt_id,success,payload) VALUES($1,'old',$2,false,'{\"legacy_error\":true}')", run, receipt)
            else:
                await conn.execute("""INSERT INTO benchmark_tasks(run_id,model_id,receipt_id,status,attempts,started_at,leased_until)
                    VALUES($1,'old',$2,$3,431,'2020-01-01',clock_timestamp()+interval '1 hour')""", run, receipt, status)
        reports = [await conn.fetch("SELECT to_jsonb(t)::text AS payload FROM " + table + " t ORDER BY run_id") for table in ("benchmark_results", "benchmark_summaries")]
        historic_run = await conn.fetchval("SELECT (to_jsonb(r)-'owner_id'-'comparison_id'-'comparison_arm')::text FROM benchmark_runs r WHERE id=$1", runs[-1])
        migrate(db)
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0025"
        for table, before in zip(("benchmark_results", "benchmark_summaries"), reports, strict=True):
            assert await conn.fetch("SELECT to_jsonb(t)::text AS payload FROM " + table + " t ORDER BY run_id") == before
        assert await conn.fetchval("SELECT (to_jsonb(r)-'owner_id'-'comparison_id'-'comparison_arm')::text FROM benchmark_runs r WHERE id=$1", runs[-1]) == historic_run
        assert await conn.fetchval("SELECT bool_and(comparison_id IS NULL AND comparison_arm IS NULL) FROM benchmark_runs")
        tasks = await conn.fetch("SELECT * FROM benchmark_tasks ORDER BY status")
        assert {t["status"] for t in tasks} == {"queued", "retrying"}
        for task in tasks:
            assert task["attempts"] == 431  # poll count, NOT the historical true inference count
            assert task["inference_attempt_count"] == 0
            assert task["inference_started"] is False and task["claim_token"] is None
            assert task["started_at"].year == 2020
            if task["status"] == "retrying":
                assert task["leased_until"] is None
                assert task["retry_at"] <= await conn.fetchval("SELECT clock_timestamp()")
        assert await conn.fetchval("SELECT experiment_manifest FROM benchmark_runs WHERE id=$1", runs[1]) == json.dumps({"options": {}, "historical": "unchanged"})
        # Unmanifested historical work stops without inference. Its 431 historical
        # polls remain historical metadata, never an inference count or lifetime proof.
        from vision_bill.config import ImageSettings
        from vision_bill.provider.llm.base import ModelInfo
        from vision_bill.provider.llm.ollama import OllamaProvider
        from vision_bill.service.benchmark_service import BenchmarkService
        from vision_bill.service.receipt_service import ReceiptService
        provider = OllamaProvider("http://unused-local-provider")
        model = ModelInfo("old")
        model.capabilities = ["vision"]
        provider.get_available_models = AsyncMock(return_value=[model])
        provider.analyse_receipt_with_metadata = AsyncMock(side_effect=TimeoutError("transient"))
        receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path),tmp_dir=str(tmp_path)), db, provider)
        await receipts._db.init_db()
        try:
            service = BenchmarkService(provider, receipts)
            await conn.execute("UPDATE benchmark_tasks SET retry_at=clock_timestamp()+interval '1 day' WHERE run_id=$1", runs[1])
            task = await service._db.lease_task()
            assert task["run_id"] == runs[0]
            assert task["attempts"] == 431 and task["inference_attempt_count"] == 0
            await service._execute(dict(task))
            summary = (await service._db.summaries(runs[0]))[0]
            assert summary.completed == 8
            assert await conn.fetchrow("SELECT * FROM benchmark_tasks WHERE run_id=$1", runs[0]) is None
            provider.analyse_receipt_with_metadata.assert_not_awaited()
            assert await service._db.lease_task() is None
            assert summary.succeeded == 5 and summary.failed == 3
            assert summary.average_score == .7 and summary.all_case_average_score is None
            assert await conn.fetchval("SELECT (to_jsonb(r)-'owner_id'-'comparison_id'-'comparison_arm')::text FROM benchmark_runs r WHERE id=$1", runs[-1]) == historic_run
        finally:
            await receipts._db.destroy_db()
    finally:
        await conn.close()


@pytest.mark.parametrize("state", ["claim", "attempt"])
async def test_0017_downgrade_refuses_to_discard_tracked_unfinished_state(db, state):
    migrate(db)
    conn = await asyncpg.connect(db.pg_dsn)
    try:
        receipt = await conn.fetchval("INSERT INTO receipts(merchant_name,date,total) VALUES('legacy','2020-01-01',1) RETURNING id")
        run = await conn.fetchval("""INSERT INTO benchmark_runs(model_ids,receipt_ids,dataset_fingerprint,prompt_version,scoring_version)
            VALUES('["old"]','[]','f','2','2') RETURNING id""")
        if state == "claim":
            await conn.execute("INSERT INTO benchmark_tasks(run_id,model_id,receipt_id,claim_token) VALUES($1,'old',$2,gen_random_uuid())", run, receipt)
        else:
            await conn.execute("INSERT INTO benchmark_tasks(run_id,model_id,receipt_id,inference_attempt_count) VALUES($1,'old',$2,1)", run, receipt)
        result = migrate(db, "0016", "downgrade", check=False)
        assert result.returncode != 0 and "destructive benchmark claim downgrade refused" in result.stderr
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0025"
        assert await conn.fetchval("SELECT count(*) FROM benchmark_tasks") == 1
    finally:
        await conn.close()
