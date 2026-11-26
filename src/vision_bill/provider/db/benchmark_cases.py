"""Scoped human-authored reference cases. Only benchmark tables are mutated."""

import asyncio
import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg
from ocr_contract.limits import APP_SOURCE_MAX_BYTES

from ...model.benchmark_case import BenchmarkCase, BenchmarkCaseSummary, BenchmarkCaseUpdate
from ...model.extraction import ExtractionDraft
from ...model.workflow import AccessScope
from ...service.experiment_scoring import ITEM, OBSERVABLE_FIELDS, TAX
from .deadline import connection

_SUMMARY = """id, owner_id, name, revision, created_at, updated_at, confirmed_by, confirmed_at,
    confirmation_id IS NOT NULL AS confirmed, jsonb_array_length(sources) AS photo_count"""


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _detail(row: Mapping[str, Any]) -> BenchmarkCase:
    return BenchmarkCase.model_validate({**row, "expected": _json(row["expected"])})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scored_observations(expected: dict[str, Any]) -> list[str]:
    fields = [key for key in OBSERVABLE_FIELDS if expected.get(key) is not None]
    if not fields:
        raise ValueError("Enter at least one expected observation before confirming")
    for key in fields:
        value = expected[key]
        if isinstance(value, str) and not value.strip():
            raise ValueError("Expected observations must contain a value")
    for group, columns in (("line_items", ITEM), ("taxes", TAX)):
        for row in expected.get(group) or []:
            if (
                group == "taxes"
                and row.get("rate") is not None
                and not 0 <= Decimal(str(row["rate"])) <= 1
            ):
                raise ValueError("Expected tax rates must be fractions between zero and one")
            observed = [row[key] for key in columns if row.get(key) is not None]
            if not observed or any(isinstance(v, str) and not v.strip() for v in observed):
                raise ValueError("Each expected row needs a printed observation")
    return fields


def _read_source(source: dict[str, Any]) -> bytes:
    try:
        with Path(source["path"]).open("rb") as stream:
            content = stream.read(APP_SOURCE_MAX_BYTES + 1)
    except OSError as exc:
        raise ValueError("Benchmark case photo unavailable") from exc
    if (
        len(content) > APP_SOURCE_MAX_BYTES
        or hashlib.sha256(content).hexdigest() != source["sha256"]
    ):
        raise ValueError("Benchmark case photo changed")
    return content


async def _verify_sources(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        await asyncio.to_thread(_read_source, source)


class CaseNotFound(ValueError):
    pass


class CaseRevisionConflict(ValueError):
    pass


class BenchmarkCaseDB:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def create(
        self, case_id: UUID, name: str, sources: list[dict[str, Any]], *, scope: AccessScope
    ) -> BenchmarkCase:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""INSERT INTO benchmark_cases (id,owner_id,name,sources)
                VALUES ($1,$2,$3,$4) RETURNING {_SUMMARY}, expected""",
                case_id,
                scope.user_id,
                name,
                json.dumps(sources),
            )
        return _detail(row)

    async def list_cases(self, *, scope: AccessScope) -> list[BenchmarkCaseSummary]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT {_SUMMARY} FROM benchmark_cases
                WHERE ($1 OR owner_id=$2) ORDER BY created_at DESC, id DESC""",
                scope.can_see_all,
                scope.user_id,
            )
        return [BenchmarkCaseSummary.model_validate(dict(row)) for row in rows]

    @staticmethod
    async def _load(
        conn: asyncpg.Connection, case_id: UUID, scope: AccessScope, *, lock: bool = False
    ) -> Mapping[str, Any]:
        row = await conn.fetchrow(
            f"""SELECT {_SUMMARY}, expected, sources, confirmation_id FROM benchmark_cases
            WHERE id=$1 AND ($2 OR owner_id=$3) {"FOR UPDATE" if lock else ""}""",
            case_id,
            scope.can_see_all,
            scope.user_id,
        )
        if row is None:
            raise CaseNotFound("Benchmark case not found")
        return dict(row)

    async def get(self, case_id: UUID, *, scope: AccessScope) -> BenchmarkCase:
        async with self._pool.acquire() as conn:
            return _detail(await self._load(conn, case_id, scope))

    async def photo(self, case_id: UUID, index: int, *, scope: AccessScope) -> tuple[bytes, str]:
        async with self._pool.acquire() as conn:
            sources = _json((await self._load(conn, case_id, scope))["sources"])
        if not 0 <= index < len(sources):
            raise CaseNotFound("Benchmark case photo not found")
        return await asyncio.to_thread(_read_source, sources[index]), sources[index]["media_type"]

    async def update(
        self, case_id: UUID, update: BenchmarkCaseUpdate, *, scope: AccessScope
    ) -> BenchmarkCase:
        async with self._pool.acquire() as conn, conn.transaction():
            current = await self._load(conn, case_id, scope, lock=True)
            if update.expected_revision != current["revision"]:
                raise CaseRevisionConflict("Benchmark case revision changed")
            draft = update.expected if update.expected is not None else _detail(current).expected
            expected = draft.model_dump(mode="json", exclude={"merchant_display_name"})
            if update.confirm:
                scored_observations(expected)
                await _verify_sources(_json(current["sources"]))
            row = await conn.fetchrow(
                f"""UPDATE benchmark_cases SET name=COALESCE($2,name), expected=$3,
                revision=revision+1, updated_at=clock_timestamp(),
                confirmation_id=CASE WHEN $4 THEN gen_random_uuid() END,
                confirmed_by=CASE WHEN $4 THEN $5::uuid END,
                confirmed_at=CASE WHEN $4 THEN clock_timestamp() END
                WHERE id=$1 RETURNING {_SUMMARY}, expected""",
                case_id,
                update.name,
                json.dumps(expected),
                update.confirm,
                scope.user_id,
            )
            return _detail(row)

    @staticmethod
    async def _frozen(
        conn: asyncpg.Connection, case_id: UUID, scope: AccessScope
    ) -> dict[str, Any]:
        row = await BenchmarkCaseDB._load(conn, case_id, scope, lock=True)
        if not row["confirmed"]:
            raise ValueError("Benchmark case requires a current human confirmation")
        sources = _json(row["sources"])
        await _verify_sources(sources)
        expected = ExtractionDraft.model_validate(_json(row["expected"])).model_dump(
            mode="json", exclude={"merchant_display_name"}
        )
        scored = scored_observations(expected)
        document = {
            "case_id": str(case_id),
            "owner_id": str(row["owner_id"]),
            "revision": row["revision"],
            "expected": expected,
            "scored_fields": scored,
            "sources": [{"photo_index": s["photo_index"], "sha256": s["sha256"]} for s in sources],
        }
        reference = {
            "id": str(row["confirmation_id"]),
            "review_revision": row["revision"],
            "qualified_by": str(row["confirmed_by"]),
            "qualified_at": row["confirmed_at"].isoformat(),
            "owner_id": str(row["owner_id"]),
            "document_sha256": _digest(document),
            "expected_sha256": _digest(expected),
        }
        return {
            "receipt_id": str(case_id),
            "case_id": str(case_id),
            "input_kind": "case",
            "name": row["name"],
            "expected": expected,
            "scored_fields": scored,
            "images": sources,
            "reference": reference,
        }

    async def freeze(
        self, case_id: UUID, *, scope: AccessScope, deadline: float | None = None
    ) -> dict[str, Any]:
        async with connection(self._pool, deadline=deadline, transaction=True) as conn:
            return await self._frozen(conn, case_id, scope)

    @staticmethod
    async def validate_frozen(
        conn: asyncpg.Connection, item: dict[str, Any], scope: AccessScope
    ) -> None:
        current = await BenchmarkCaseDB._frozen(conn, UUID(item["case_id"]), scope)
        if any(value != item.get(key) for key, value in current.items()):
            raise ValueError("Benchmark case changed before run creation")
