"""PostgreSQL confirmation fences, owner scope and typed benchmark task references."""

import asyncio
import hashlib
import json
from uuid import uuid4

import asyncpg
import pytest

from tests.e2e import test_workflow_db_e2e as workflow_tests
from vision_bill.model.benchmark_case import BenchmarkCaseUpdate
from vision_bill.model.workflow import AccessScope
from vision_bill.provider.db.benchmark_cases import (
    BenchmarkCaseDB,
    CaseNotFound,
    CaseRevisionConflict,
)
from vision_bill.provider.db.benchmark_db import BenchmarkDB

postgres, db, workflow_db = workflow_tests.postgres, workflow_tests.db, workflow_tests.workflow_db
pytestmark = pytest.mark.e2e


async def test_scoped_sources_revision_fence_and_frozen_inputs(workflow_db, tmp_path):
    workflow, owner = workflow_db
    case_db = BenchmarkCaseDB(workflow.pool)
    other = AccessScope(user_id=uuid4())
    admin = other.model_copy(update={"can_see_all": True})
    async with workflow.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users(id,username,hashed_password) VALUES($1,'other','test')",
            other.user_id,
        )
    case_id = uuid4()
    sources = []
    for index, content in enumerate((b"first", b"second")):
        path = tmp_path / f"{index}.png"
        path.write_bytes(content)
        sources.append(
            {
                "path": str(path),
                "photo_index": index,
                "media_type": "image/png",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    case = await case_db.create(case_id, "Private", sources, scope=owner)
    assert not case.confirmed
    assert await case_db.list_cases(scope=other) == []
    with pytest.raises(CaseNotFound):
        await case_db.photo(case_id, 0, scope=other)
    with pytest.raises(ValueError, match="human confirmation"):
        await case_db.freeze(case_id, scope=owner)
    assert await case_db.photo(case_id, 1, scope=owner) == (b"second", "image/png")
    for index in (-1, 2):
        with pytest.raises(CaseNotFound):
            await case_db.photo(case_id, index, scope=owner)
    edit = BenchmarkCaseUpdate.model_validate(
        {
            "expected_revision": 1,
            "expected": {"total": "987654321.01", "tax_total": "0.00"},
            "confirm": True,
        }
    )
    with pytest.raises(CaseNotFound):
        await case_db.update(case_id, edit, scope=other)
    attempts = await asyncio.gather(
        *(case_db.update(case_id, edit, scope=admin) for _ in range(2)), return_exceptions=True
    )
    assert sum(isinstance(item, CaseRevisionConflict) for item in attempts) == 1
    saved = await case_db.get(case_id, scope=owner)
    assert saved.confirmed and saved.confirmed_by == admin.user_id and saved.revision == 2
    item = await case_db.freeze(case_id, scope=owner)
    assert item["scored_fields"] == ["tax_total", "total"]
    assert item["expected"]["tip"] is None and item["expected"]["tax_total"] == "0.00"
    assert item["images"] == sources and item["reference"]["qualified_by"] == str(admin.user_id)
    async with workflow.pool.acquire() as conn, conn.transaction():
        await case_db.validate_frozen(conn, item, owner)
    with pytest.raises(CaseNotFound):
        await case_db.freeze(case_id, scope=other)
    (tmp_path / "0.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="photo changed"):
        await case_db.freeze(case_id, scope=owner)
    (tmp_path / "0.png").write_bytes(b"first")
    replaced = await case_db.update(
        case_id, BenchmarkCaseUpdate(expected_revision=2, name="Updated"), scope=owner
    )
    assert replaced.revision == 3 and not replaced.confirmed and replaced.confirmed_by is None
    async with workflow.pool.acquire() as conn, conn.transaction():
        with pytest.raises(ValueError, match="human confirmation"):
            await case_db.validate_frozen(conn, item, owner)
        for table in (
            "receipts",
            "images",
            "line_items",
            "taxes",
            "benchmark_runs",
            "benchmark_tasks",
        ):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0


async def test_task_foreign_keys_and_run_visibility_follow_input_type(workflow_db, db):
    workflow, owner = workflow_db
    benchmark = BenchmarkDB(workflow.pool)
    other_id, input_id = uuid4(), uuid4()
    # A UUID collision makes the two independent ownership checks observable.
    async with workflow.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users(id,username,hashed_password) VALUES($1,'other','test')", other_id
        )
        await conn.execute(
            "INSERT INTO receipts(id,user_id,merchant_name,date,total) VALUES($1,$2,'legacy','2020-01-01',1)",
            input_id,
            owner.user_id,
        )
        await conn.execute(
            "INSERT INTO benchmark_cases(id,owner_id,name,sources) VALUES($1,$2,'other case','[{}]')",
            input_id,
            other_id,
        )
        run = await conn.fetchval(
            """INSERT INTO benchmark_runs(owner_id,model_ids,receipt_ids,dataset_fingerprint,prompt_version,scoring_version,experiment_manifest)
            VALUES($1,'["m"]',$2,'ds','2','5',$3) RETURNING id""",
            owner.user_id,
            json.dumps([str(input_id)]),
            json.dumps(
                {
                    "inputs": [
                        {
                            "case_id": str(input_id),
                            "input_kind": "case",
                            "receipt_id": str(input_id),
                        }
                    ]
                }
            ),
        )
        await conn.execute(
            "INSERT INTO benchmark_tasks(run_id,model_id,receipt_id,input_kind) VALUES($1,'case',$2,'case'),($1,'normal',$2,'receipt')",
            run,
            input_id,
        )
        for table in ("benchmark_cases", "receipts"):
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await conn.execute(f"DELETE FROM {table} WHERE id=$1", input_id)
        assert await benchmark.run(run, scope=owner) is None
        await conn.execute(
            "UPDATE benchmark_cases SET owner_id=$1 WHERE id=$2", owner.user_id, input_id
        )
        assert await benchmark.run(run, scope=owner)
        assert not await benchmark.runs(scope=AccessScope(user_id=other_id))
        assert await benchmark.runs(scope=AccessScope(user_id=other_id, can_see_all=True))
        assert workflow_tests.migrate(db, "0021", "downgrade", check=False).returncode != 0
