"""Persistence for benchmark control-plane data only.

This module deliberately never writes to receipts, line_items, taxes or images.
"""
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg

from ...model.benchmark import BenchmarkRun, BenchmarkSummary, CouncilPolicy


class BenchmarkDB:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def create_run(self, *, model_ids: list[str], receipt_ids: list[UUID], fingerprint: str,
                         model_digests: dict[str, str | None], timeout: int, policy: CouncilPolicy,
                         absolute: Any, relative: Any, apply_flags: bool, prompt_version: str,
                         scoring_version: str, manifest: dict[str, Any] | None = None) -> BenchmarkRun:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow("""INSERT INTO benchmark_runs
                (model_ids, receipt_ids, dataset_fingerprint, model_digests, request_timeout_seconds,
                 council_policy, council_absolute_threshold, council_relative_threshold, apply_council_flags,
                 prompt_version, scoring_version, experiment_manifest)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING *""",
                json.dumps(model_ids),
                json.dumps([str(receipt_id) for receipt_id in receipt_ids]),
                fingerprint,
                json.dumps(model_digests),
                timeout,
                policy.value,
                absolute,
                relative,
                apply_flags, prompt_version, scoring_version,
                json.dumps(manifest) if manifest is not None else None)
            await conn.executemany("INSERT INTO benchmark_tasks (run_id, model_id, receipt_id) VALUES ($1,$2,$3)", [(row["id"], m, r) for m in model_ids for r in receipt_ids])
            await conn.executemany("INSERT INTO benchmark_summaries (run_id, model_id, model_digest) VALUES ($1,$2,$3)", [(row["id"], m, model_digests.get(m)) for m in model_ids])
        return self._run(row)

    async def run(self, run_id: UUID) -> BenchmarkRun | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM benchmark_runs WHERE id=$1", run_id)
        return self._run(row) if row else None

    async def runs(self) -> list[BenchmarkRun]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM benchmark_runs ORDER BY created_at DESC, id DESC")
        return [self._run(r) for r in rows]

    async def task_counts(self, run_id: UUID) -> dict[str, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT status, count(*) AS n FROM benchmark_tasks WHERE run_id=$1 GROUP BY status", run_id)
            terminal = await conn.fetchval("SELECT COALESCE(sum(completed), 0) FROM benchmark_summaries WHERE run_id=$1", run_id)
        counts = {str(r["status"]): int(r["n"]) for r in rows}
        return {**{name: counts.get(name, 0) for name in ("queued", "running", "waiting_for_model", "retrying")}, "terminal": int(terminal)}

    async def summaries(self, run_id: UUID) -> list[BenchmarkSummary]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM benchmark_summaries WHERE run_id=$1 ORDER BY model_id", run_id)
        return [BenchmarkSummary(model_id=r["model_id"], model_digest=r["model_digest"], completed=r["completed"], succeeded=r["succeeded"], failed=r["failed"], average_score=(r["total_score"] / r["succeeded"] if r["succeeded"] else None), average_confidence=(r["total_confidence"] / r["succeeded"] if r["succeeded"] else None), average_attempts=(r["total_attempts"] / r["completed"] if r["completed"] else None), average_latency_ms=(r["total_latency_ms"] / r["completed"] if r["completed"] else None), council_candidates=r["council_candidates"], council_findings=r["council_findings"]) for r in rows]

    async def lease_task(self) -> Mapping[str, Any] | None:
        now = datetime.now(UTC)
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow("""SELECT * FROM benchmark_tasks WHERE
                (status IN ('queued','retrying','waiting_for_model') AND (retry_at IS NULL OR retry_at <= $1))
                OR (status='running' AND leased_until < $1)
                ORDER BY run_id, receipt_id FOR UPDATE SKIP LOCKED LIMIT 1""", now)
            if not row:
                return None
            await conn.execute("UPDATE benchmark_tasks SET status='running', attempts=attempts+1, started_at=$2, leased_until=$3 WHERE run_id=$1 AND model_id=$4 AND receipt_id=$5", row["run_id"], now, now + timedelta(minutes=6), row["model_id"], row["receipt_id"])
            return dict(row)

    async def retry_task(self, task: Mapping[str, Any], waiting: bool = False) -> None:
        delay = min(5 * 2 ** max(0, int(task["attempts"])), 300)
        async with self._pool.acquire() as conn:
            await conn.execute("UPDATE benchmark_tasks SET status=$1, leased_until=NULL, retry_at=$2 WHERE run_id=$3 AND model_id=$4 AND receipt_id=$5", "waiting_for_model" if waiting else "retrying", datetime.now(UTC) + timedelta(seconds=delay), task["run_id"], task["model_id"], task["receipt_id"])

    async def finish_task(self, task: Mapping[str, Any], *, success: bool, score: float = 0, confidence: int = 0, attempts: int = 1, latency_ms: float = 0, result: dict[str, Any] | None = None) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            deleted = await conn.execute("DELETE FROM benchmark_tasks WHERE run_id=$1 AND model_id=$2 AND receipt_id=$3", task["run_id"], task["model_id"], task["receipt_id"])
            if deleted == "DELETE 0":
                return  # A recovered lease must not count the same task twice.
            if result is not None:
                await conn.execute(
                    "INSERT INTO benchmark_results (run_id, model_id, receipt_id, success, payload) "
                    "VALUES ($1,$2,$3,$4,$5) ON CONFLICT (run_id, model_id, receipt_id) DO NOTHING",
                    task["run_id"], task["model_id"], task["receipt_id"], success, json.dumps(result))
            await conn.execute("""UPDATE benchmark_summaries SET completed=completed+1, succeeded=succeeded+$1, failed=failed+$2, total_score=total_score+$3, total_confidence=total_confidence+$4, total_attempts=total_attempts+$5, total_latency_ms=total_latency_ms+$6 WHERE run_id=$7 AND model_id=$8""", int(success), int(not success), score, confidence, attempts, latency_ms, task["run_id"], task["model_id"])
            await conn.execute("""UPDATE benchmark_runs SET status='completed', completed_at=now()
                WHERE id=$1 AND NOT EXISTS (SELECT 1 FROM benchmark_tasks WHERE run_id=$1)""", task["run_id"])

    @staticmethod
    def _run(row: Mapping[str, Any]) -> BenchmarkRun:
        model_ids = json.loads(row["model_ids"]) if isinstance(row["model_ids"], str) else row["model_ids"]
        receipt_ids = (
            json.loads(row["receipt_ids"])
            if isinstance(row["receipt_ids"], str)
            else row["receipt_ids"]
        )
        manifest = row.get("experiment_manifest")
        if isinstance(manifest, str):
            manifest = json.loads(manifest)
        experiment = {k: v for k, v in manifest.items() if k not in {"inputs", "tags"}} if manifest else None
        return BenchmarkRun(experiment=experiment, id=row["id"], status=row["status"], model_ids=list(model_ids), receipt_ids=list(receipt_ids), dataset_fingerprint=row["dataset_fingerprint"], prompt_version=row["prompt_version"], scoring_version=row["scoring_version"], council_policy=row["council_policy"], apply_council_flags=row["apply_council_flags"], created_at=row["created_at"], completed_at=row["completed_at"])

    async def manifest(self, run_id: UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval("SELECT experiment_manifest FROM benchmark_runs WHERE id=$1", run_id)
        if isinstance(value, str):
            value = json.loads(value)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("Invalid benchmark manifest")
        return dict(value)

    async def results(self, run_id: UUID, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM benchmark_results WHERE run_id=$1 ORDER BY model_id, receipt_id LIMIT $2 OFFSET $3",
                run_id, limit, offset)
        return [{"model_id": row["model_id"], "receipt_id": str(row["receipt_id"]),
                 "success": row["success"],
                 **(json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"])}
                for row in rows]
