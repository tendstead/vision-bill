"""Durable submissions and fenced, single-connection extraction transactions.

Files are saved by the application before supplying PersistedWorkingView. This
provider stores metadata and delegates completion byte checks to the image service
under the publication locks. Optional deadlines use the asyncio loop clock and bound the
whole connection/transaction lifecycle. A failed transaction may leave an unreferenced file for the
application's artifact cleanup, but never a partially committed receipt.
"""

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID, uuid4

import asyncpg
from pydantic import TypeAdapter, ValidationError

from ...config import PGSettings, WorkerSettings
from ...model.db.image import ClaimPurpose, ImageRow
from ...model.db.receipt import ReceiptRow
from ...model.extraction import (
    DraftView,
    ExtractionDraft,
    PersistedInferenceEnvelope,
    assert_metadata_only,
    pack_inference_metadata,
)
from ...model.handoff import HANDOFF_WAIT_REASONS
from ...model.ocr import OcrPageMetadata, Sha256, SourceRef, WorkingImageMetadata
from ...model.ocr_errors import OCR_RESULT_CODES
from ...model.review import AmountPolicy, OriginalAvailability, ReviewEvaluation, SourceView
from ...model.workflow import (
    AccessScope,
    AnalysisClaim,
    AttemptOutcome,
    CheckpointCompletionOutcome,
    CommitResult,
    DeletedResource,
    DraftFinalize,
    DraftPatch,
    EnqueueResult,
    ObservationCheckpoint,
    OcrCheckpoint,
    PersistedSourcePart,
    PersistedWorkingView,
    RetryRequest,
    SourcePartBinding,
    SubmissionCreate,
    SubmissionStatus,
)
from ...service.execution_policy import (
    EXECUTION_POLICY_VERSION,
    managed_ocr,
    semantic_policy_sha256,
    snapshot_ocr,
)
from ...service.observation_admission import admit_observations
from ...service.receipt_review import evaluate_receipt, persistent_findings, receipt_from_draft
from .deadline import connection
from .receipt_db import OriginalChecker, ReceiptDB

OBSERVATION_RECOVERY_WAIT = "observation_recovery_required"
COMPLETION_WAIT_REASONS = HANDOFF_WAIT_REASONS | {
    OBSERVATION_RECOVERY_WAIT, "observation_checkpoint_invalid", "source_integrity_failed",
}
READINESS_WAIT_REASONS = HANDOFF_WAIT_REASONS | OCR_RESULT_CODES | {
    "provider_unavailable",
    "vision_model_unavailable",
    "formatter_model_unavailable",
}

_UNMANAGED_CLASSIFICATION_UNSETTLED = """CASE
    WHEN observation_checkpoint->>'eligible_generation'=generation::text THEN COALESCE((
        SELECT entry->>'event' FROM jsonb_array_elements(legacy_events)
            WITH ORDINALITY AS history(entry, event_number)
        WHERE entry->>'event' IN ('unmanaged_classification_started','unmanaged_classification_settled')
            AND entry->>'checkpoint_id'=observation_checkpoint->>'checkpoint_id'
            AND entry->>'producer_generation'=generation::text
            AND entry->>'producer_attempt_number'=attempt_count::text
            AND entry->>'inference_identity'=observation_checkpoint#>>'{execution_identity,inference_identity}'
        ORDER BY event_number DESC LIMIT 1
    )='unmanaged_classification_started', FALSE) ELSE FALSE END"""


def _json(value: Any) -> str:
    assert_metadata_only(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decoded(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _status(row: Any) -> SubmissionStatus:
    status = SubmissionStatus.model_validate(ImageRow.model_validate(dict(row)).model_dump())
    if status.observations_available:
        status.draft_available = False
    return status


def _eligible_observations(row: Any) -> dict[str, Any] | None:
    checkpoint = _decoded(row["observation_checkpoint"])
    if isinstance(checkpoint, dict) and checkpoint.get("eligible_generation") == row["generation"]:
        return checkpoint
    return None


def _observation_history(checkpoint: dict[str, Any], event: str) -> dict[str, Any]:
    return {
        "event": event,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "producer_generation": checkpoint["producer_generation"],
        "producer_attempt_number": checkpoint["producer_attempt_number"],
    }


def _unmanaged_classification_start(row: Any) -> dict[str, Any] | None:
    checkpoint = _eligible_observations(row)
    if checkpoint is None:
        return None
    try:
        if (
            checkpoint["producer_generation"] != row["generation"]
            or checkpoint["producer_attempt_number"] != row["attempt_count"]
            or checkpoint["envelope"]["classification_enrichment"]["state"] != "pending"
        ):
            return None
        return _observation_history(checkpoint, "unmanaged_classification_started") | {
            "inference_identity": checkpoint["execution_identity"]["inference_identity"],
        }
    except (KeyError, TypeError, ValueError):
        return None


def _is_unmanaged_checkpoint_policy(row: Any) -> bool:
    try:
        snapshot = _decoded(row["effective_options"])
        identity = _decoded(row["observation_checkpoint"])["execution_identity"]
        ocr = snapshot_ocr(snapshot)
        return (not managed_ocr(ocr) or not ocr.shared_gpu) and (
            semantic_policy_sha256(snapshot) == identity["semantic_policy_sha256"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _has_classification_event(row: Any, event: dict[str, Any]) -> bool:
    return any(event.items() <= item.items() for item in _decoded(row["legacy_events"]))


def _classification_settlement_required(row: Any) -> bool:
    started = _unmanaged_classification_start(row)
    return started is not None and _has_classification_event(row, started) and not (
        _has_classification_event(row, started | {"event": "unmanaged_classification_settled"})
    )


class _ExpiredClaim(Exception):
    """Force rollback if a once-live lease expires before the final seal."""


def _nested_source_refs(value: Any) -> list[SourceRef]:
    """Validate references in flexible provenance/overlap metadata too."""
    refs: list[SourceRef] = []
    if isinstance(value, dict):
        if "part_id" in value:
            refs.append(SourceRef.model_validate(value))
        for item in value.values():
            refs.extend(_nested_source_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_nested_source_refs(item))
    return refs


def _nested_view_ids(value: Any) -> list[UUID]:
    """View identities in flexible metadata must belong to this submission too."""
    ids: list[UUID] = []
    if isinstance(value, dict):
        if value.get("view_id") is not None:
            ids.append(UUID(str(value["view_id"])))
        for item in value.values():
            ids.extend(_nested_view_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_nested_view_ids(item))
    return ids


class WorkflowDB:
    def __init__(
        self, settings: PGSettings, *, worker_settings: WorkerSettings | None = None,
        original_checker: OriginalChecker | None = None,
        working_view_checker: Callable[[UUID, str, WorkingImageMetadata], bytes] | None = None,
    ):
        self._settings = settings
        self._worker = worker_settings or WorkerSettings()
        self._pool: asyncpg.Pool | None = None
        self._receipts = ReceiptDB(settings, original_checker=original_checker)
        self._working_view_checker = working_view_checker

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Workflow database is not initialized")
        return self._pool

    @property
    def is_ready(self) -> bool:
        return self._pool is not None

    async def init_db(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self._settings.pg_dsn)

    async def destroy_db(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def get_submission(
        self, image_id: UUID, *, scope: AccessScope
    ) -> SubmissionStatus | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM images WHERE id=$1 AND (user_id=$2 OR $3)",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
            if row is None:
                return None
            status = _status(row)
            status.source_views = await self._source_views(conn, image_id)
            status.latest_extraction_id = await conn.fetchval(
                "SELECT id FROM receipt_extractions WHERE image_id=$1 ORDER BY generation DESC,attempt_number DESC LIMIT 1",
                image_id,
            )
            return status

    async def get_source_parts(
        self,
        image_id: UUID,
        *,
        scope: AccessScope,
    ) -> list[PersistedSourcePart]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT p.* FROM image_source_parts p JOIN images i
                ON i.id=p.image_id WHERE i.id=$1 AND (i.user_id=$2 OR $3)
                ORDER BY p.photo_index""",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
        return [PersistedSourcePart.model_validate(dict(row)) for row in rows]

    async def bind_source_parts(
        self,
        claim: AnalysisClaim,
        bindings: list[SourcePartBinding],
        *,
        deadline: float | None = None,
    ) -> list[PersistedSourcePart] | None:
        """Initialize unknown legacy source identities inside the live claim fence.

        The caller hashes originals under its attempt deadline. Paths must equal
        the stored ordered originals; established identities cannot be changed.
        """
        if claim.purpose != "inference":
            return None
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await self._lock_claim(conn, claim)
            if row is None:
                return None
            originals = ImageRow.model_validate(dict(row)).parts()
            if len(bindings) != len(originals):
                raise ValueError("source binding count mismatch")
            for index, binding in enumerate(bindings):
                if (
                    binding.photo_index != index
                    or binding.image_path != originals[index].image_path
                ):
                    raise ValueError("source binding order/path mismatch")
            manifest = [
                {"photo_index": b.photo_index, "source_sha256": b.source_sha256} for b in bindings
            ]
            old_manifest = _decoded(row["source_manifest"])
            if old_manifest and [(v["photo_index"], v["source_sha256"]) for v in old_manifest] != [
                (b.photo_index, b.source_sha256) for b in bindings
            ]:
                raise ValueError("established source hashes cannot be changed")
            existing = await self._parts(conn, row["id"])
            if existing:
                if [SourcePartBinding.model_validate(p.model_dump()) for p in existing] != bindings:
                    raise ValueError("established source identities cannot be changed")
                return existing
            for index, binding in enumerate(bindings):
                await conn.execute(
                    """INSERT INTO image_source_parts(image_id,photo_index,
                    source_sha256,image_path,metadata) VALUES($1,$2,$3,$4,$5::jsonb)""",
                    row["id"],
                    index,
                    binding.source_sha256,
                    binding.image_path,
                    _json(originals[index].model_dump(mode="json")),
                )
            await conn.execute(
                "UPDATE images SET source_manifest=$2::jsonb WHERE id=$1",
                row["id"],
                _json(manifest),
            )
            return await self._parts(conn, row["id"])

    async def enqueue_submission(
        self,
        submission: SubmissionCreate,
        *,
        scope: AccessScope,
        idempotency_key: str | None,
    ) -> EnqueueResult:
        if idempotency_key is not None and not idempotency_key.strip():
            raise ValueError("idempotency key must not be empty")
        manifest = submission.source_manifest
        if len(manifest) != len(submission.parts):
            raise ValueError("source manifest must match ordered parts")
        for index, source in enumerate(manifest):
            if source.get("photo_index") != index:
                raise ValueError("source photo indices must be ordered and contiguous")
            TypeAdapter(Sha256).validate_python(source.get("source_sha256"))
        async with self.pool.acquire() as conn, conn.transaction():
            if idempotency_key is not None:
                # Serialize absent keys too, rather than racing SELECT then INSERT.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                    f"upload:{scope.user_id}:{idempotency_key}",
                )
                old = await conn.fetchrow(
                    "SELECT * FROM images WHERE user_id=$1 AND idempotency_key=$2",
                    scope.user_id,
                    idempotency_key,
                )
                if old:
                    expected = (
                        submission.request_fingerprint,
                        [item["source_sha256"] for item in manifest],
                        submission.requested_options,
                    )
                    actual = (
                        old["request_fingerprint"],
                        [item["source_sha256"] for item in _decoded(old["source_manifest"])],
                        _decoded(old["requested_options"]),
                    )
                    if actual != expected:
                        raise ValueError("upload idempotency payload conflict")
                    return EnqueueResult(submission=_status(old), created=False)
            primary = submission.parts[0]
            row = await conn.fetchrow(
                """INSERT INTO images(id,original_filename,media_type,
                size_bytes,image_path,thumbnail_path,user_id,bypass_review,additional_images,
                model_id,source_manifest,requested_options,effective_options,request_fingerprint,
                idempotency_key,max_attempts) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11::jsonb,
                $12::jsonb,$13::jsonb,$14,$15,$16) RETURNING *""",
                submission.id,
                primary.original_filename,
                primary.media_type,
                primary.size_bytes,
                primary.image_path,
                primary.thumbnail_path,
                scope.user_id,
                submission.bypass_review,
                _json([part.model_dump(mode="json") for part in submission.parts[1:]]),
                submission.model_id,
                _json(manifest),
                _json(submission.requested_options),
                _json(submission.effective_options),
                submission.request_fingerprint,
                idempotency_key,
                self._worker.max_attempts,
            )
            for index, part in enumerate(submission.parts):
                await conn.execute(
                    """INSERT INTO image_source_parts(image_id,photo_index,
                    source_sha256,image_path,metadata) VALUES($1,$2,$3,$4,$5::jsonb)""",
                    submission.id,
                    index,
                    manifest[index]["source_sha256"],
                    part.image_path,
                    _json(part.model_dump(mode="json")),
                )
            return EnqueueResult(submission=_status(row), created=True)

    @staticmethod
    async def _record_abandonment(conn: asyncpg.Connection, row: Any, *, exhausted: bool) -> None:
        checkpoint = _eligible_observations(row)
        await conn.execute(
            """INSERT INTO receipt_extractions(image_id,generation,attempt_number,outcome_kind,
            result_code,envelope,source_views)
            VALUES($1,$2,$3,$4,'lease_expired',$5::jsonb,$6::jsonb)
            ON CONFLICT(image_id,generation,attempt_number) DO NOTHING""",
            row["id"], row["generation"], row["attempt_count"], "terminal" if exhausted else "retry",
            _json(checkpoint["envelope"]) if checkpoint else None,
            _json(checkpoint["persisted_views"]) if checkpoint else "[]",
        )

    async def reap_expired_claims(self, *, deadline: float | None = None) -> int:
        """Release expired leases. Only inference creates abandonment history."""
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            rows = await conn.fetch(
                f"""SELECT id,generation,attempt_count,max_attempts,claim_purpose,
                observation_checkpoint,wait_reason,
                {_UNMANAGED_CLASSIFICATION_UNSETTLED} AS unmanaged_classification_unsettled FROM images
                WHERE status='processing' AND lease_expires_at<=clock_timestamp() AND receipt_id IS NULL
                AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id)
                ORDER BY id FOR UPDATE SKIP LOCKED"""
            )
            reaped = 0
            for row in rows:
                completion = row["claim_purpose"] == "completion"
                checkpoint = _eligible_observations(row)
                recovering = completion or checkpoint is not None
                exhausted = not recovering and row["attempt_count"] >= row["max_attempts"]
                result_code = None if completion else "attempt_limit" if exhausted else "lease_expired"
                delay = None if recovering or exhausted else self._worker.retry_delay(row["attempt_count"])
                reason = row["wait_reason"]
                if row["unmanaged_classification_unsettled"]:
                    reason = "model_request_unsettled"
                elif recovering and reason != "execution_policy_changed":
                    reason = OBSERVATION_RECOVERY_WAIT
                history = []
                if checkpoint is not None and not completion:
                    history.append(_observation_history(checkpoint, "observation_checkpoint_abandoned"))
                # Recheck expiry and the receipt association while holding the image lock.
                changed = await conn.fetchval(
                    """UPDATE images SET status=$2,
                    result_code=CASE WHEN $7 THEN result_code ELSE $3 END,
                    error=CASE WHEN $7 THEN error ELSE $3 END,
                    claim_token=NULL,claim_purpose=NULL,lease_expires_at=NULL,processing_at=NULL,
                    retry_at=clock_timestamp()+make_interval(secs => $4),
                    wait_reason=$5,legacy_events=legacy_events || $6::jsonb
                    WHERE id=$1 AND status='processing' AND lease_expires_at<=clock_timestamp()
                    AND receipt_id IS NULL
                    AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id)
                    RETURNING id""",
                    row["id"], "failed" if exhausted else "pending", result_code, delay,
                    reason, _json(history), completion,
                )
                if changed is not None:
                    if not completion:
                        await self._record_abandonment(conn, row, exhausted=exhausted)
                    reaped += 1
            return reaped

    async def claim_next(
        self, *, lease_seconds: int = 900, image_id: UUID | None = None,
        purpose: ClaimPurpose = "inference",
    ) -> AnalysisClaim | None:
        if lease_seconds <= self._worker.attempt_timeout_seconds:
            raise ValueError("claim lease must exceed the complete attempt deadline")
        async with self.pool.acquire() as conn, conn.transaction():
            while True:
                row = await conn.fetchrow(
                    f"""SELECT * FROM images WHERE ($1::uuid IS NULL OR id=$1) AND receipt_id IS NULL AND
                    NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id) AND
                    (status='pending' OR ($3='completion' AND status IN ('failed','timed_out','unreadable')))
                    AND (retry_at IS NULL OR retry_at<=clock_timestamp())
                    AND CASE WHEN $3='completion'
                        THEN observation_checkpoint->>'eligible_generation'=generation::text
                            AND NOT ({_UNMANAGED_CLASSIFICATION_UNSETTLED})
                        ELSE observation_checkpoint->>'eligible_generation' IS DISTINCT FROM generation::text
                            AND wait_reason IS DISTINCT FROM $2 END
                    ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1""",
                    image_id, OBSERVATION_RECOVERY_WAIT, purpose,
                )
                if row is None:
                    return None
                if purpose == "inference" and row["attempt_count"] >= row["max_attempts"]:
                    await conn.execute(
                        """UPDATE images SET status='failed', result_code='attempt_limit',
                        claim_token=NULL,claim_purpose=NULL,lease_expires_at=NULL,
                        processing_at=NULL WHERE id=$1""",
                        row["id"],
                    )
                    continue
                checkpoint = _eligible_observations(row) if purpose == "completion" else None
                if checkpoint is not None and checkpoint.get("producer_attempt_number") == row["attempt_count"]:
                    await self._record_abandonment(conn, row, exhausted=False)
                claimed = await conn.fetchrow(
                    """UPDATE images SET status='processing',
                    processing_at=clock_timestamp(),claim_token=$2,claim_purpose=$4,
                    lease_expires_at=clock_timestamp()+$3::double precision*interval '1 second',
                    attempt_count=attempt_count+CASE WHEN $4='inference' THEN 1 ELSE 0 END,
                    retry_at=NULL,wait_reason=NULL
                    WHERE id=$1 RETURNING *""",
                    row["id"],
                    uuid4(),
                    lease_seconds,
                    purpose,
                )
                return AnalysisClaim(
                    image=ImageRow.model_validate(dict(claimed)),
                    generation=claimed["generation"],
                    attempt_number=claimed["attempt_count"],
                    claim_token=claimed["claim_token"],
                    lease_expires_at=claimed["lease_expires_at"],
                    purpose=claimed["claim_purpose"],
                    checkpoint_id=checkpoint["checkpoint_id"] if checkpoint is not None else None,
                )

    @staticmethod
    async def _lock_claim(conn: asyncpg.Connection, claim: AnalysisClaim) -> Any:
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1 FOR UPDATE", claim.image.id)
        if row is None:
            return None
        live = await conn.fetchval(
            "SELECT $1::timestamptz>clock_timestamp()", row["lease_expires_at"]
        )
        if (
            row["status"] != "processing"
            or row["generation"] != claim.generation
            or row["attempt_count"] != claim.attempt_number
            or row["claim_token"] != claim.claim_token
            or row["claim_purpose"] != claim.purpose
            or not live
        ):
            return None
        if claim.purpose == "completion":
            checkpoint = _eligible_observations(row)
            if (
                checkpoint is None or str(claim.checkpoint_id) != checkpoint.get("checkpoint_id")
                or row["user_id"] != claim.image.user_id
            ):
                return None
        return row

    @staticmethod
    async def _parts(conn: asyncpg.Connection, image_id: UUID) -> list[PersistedSourcePart]:
        rows = await conn.fetch(
            "SELECT * FROM image_source_parts WHERE image_id=$1 ORDER BY photo_index", image_id
        )
        return [PersistedSourcePart.model_validate(dict(row)) for row in rows]

    @staticmethod
    async def _save_views(
        conn: asyncpg.Connection,
        image_id: UUID,
        parts: list[PersistedSourcePart],
        views: list[PersistedWorkingView],
    ) -> dict[UUID, PersistedWorkingView]:
        bound: dict[UUID, PersistedWorkingView] = {}
        for view in views:
            if view.photo_index >= len(parts):
                raise ValueError("working view photo index does not match real source parts")
            part = parts[view.photo_index]
            if view.part_id != part.part_id or view.source_sha256 != part.source_sha256:
                raise ValueError("working view does not match real source part")
            metadata = view.metadata.model_dump(mode="json")
            await conn.execute(
                """INSERT INTO extraction_working_views(view_id,image_id,part_id,
                artifact_path,metadata) VALUES($1,$2,$3,$4,$5::jsonb) ON CONFLICT(view_id) DO NOTHING""",
                view.metadata.view_id,
                image_id,
                view.part_id,
                view.artifact_path,
                _json(metadata),
            )
            saved = await conn.fetchrow(
                "SELECT * FROM extraction_working_views WHERE view_id=$1", view.metadata.view_id
            )
            if (
                saved["image_id"] != image_id
                or saved["part_id"] != view.part_id
                or saved["artifact_path"] != view.artifact_path
                or _decoded(saved["metadata"]) != metadata
            ):
                raise ValueError("immutable working artifact conflict")
            bound[view.metadata.view_id] = view
        return bound

    @staticmethod
    def _require_view(
        parts: list[PersistedSourcePart],
        views: dict[UUID, PersistedWorkingView],
        photo_index: int,
        source_sha256: str,
        metadata: WorkingImageMetadata,
    ) -> None:
        if photo_index >= len(parts) or parts[photo_index].source_sha256 != source_sha256:
            raise ValueError("source index/hash mismatch")
        saved = views.get(metadata.view_id)
        projected = WorkingImageMetadata.model_validate(metadata.model_dump()).model_dump(
            mode="json"
        )
        if (
            saved is None
            or saved.photo_index != photo_index
            or saved.source_sha256 != source_sha256
            or saved.metadata.model_dump(mode="json") != projected
        ):
            raise ValueError("missing or mismatched saved working artifact")

    @staticmethod
    async def _validate_refs(
        conn: asyncpg.Connection,
        image_id: UUID,
        parts: list[PersistedSourcePart],
        references: list[SourceRef],
    ) -> None:
        for ref in references:
            if ref.photo_index >= len(parts):
                raise ValueError("source reference photo index mismatch")
            part = parts[ref.photo_index]
            if ref.part_id != part.part_id or ref.source_sha256 != part.source_sha256:
                raise ValueError("source reference must bind to a real source part")
            if ref.view_id is not None:
                exists = await conn.fetchval(
                    """SELECT EXISTS(SELECT 1 FROM extraction_working_views
                    WHERE view_id=$1 AND image_id=$2 AND part_id=$3)""",
                    ref.view_id,
                    image_id,
                    ref.part_id,
                )
                if not exists:
                    raise ValueError("source reference view does not match saved artifact")
            if ref.coordinate_space == "working" and ref.view_id is None:
                raise ValueError("working source reference requires saved view")

    async def _validate_metadata_refs(
        self,
        conn: asyncpg.Connection,
        image_id: UUID,
        parts: list[PersistedSourcePart],
        value: Any,
        *,
        checkpoint_view_ids: set[UUID] | None = None,
    ) -> None:
        view_ids = set(_nested_view_ids(value))
        if checkpoint_view_ids is not None and not view_ids <= checkpoint_view_ids:
            raise ValueError("source reference does not match a checkpoint working view")
        await self._validate_refs(conn, image_id, parts, _nested_source_refs(value))
        for view_id in view_ids:
            if not await conn.fetchval(
                """SELECT EXISTS(SELECT 1 FROM extraction_working_views
                WHERE view_id=$1 AND image_id=$2)""",
                view_id,
                image_id,
            ):
                raise ValueError("source reference view does not match saved artifact")

    async def get_observation_checkpoint(
        self, claim: AnalysisClaim, *, deadline: float | None = None,
    ) -> ObservationCheckpoint | None:
        """Return complete observations only under the live, same-generation lease."""
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await self._lock_claim(conn, claim)
            if row is None:
                return None
            checkpoint = _eligible_observations(row)
            return ObservationCheckpoint.model_validate(checkpoint) if checkpoint else None

    @staticmethod
    async def _lock_unedited_submission(
        conn: asyncpg.Connection, row: Any,
    ) -> None:
        # Keep the shared image → receipt → draft lock order.
        existing = await conn.fetchrow(
            "SELECT id FROM receipts WHERE image_id=$1 FOR UPDATE", row["id"]
        )
        draft = await conn.fetchrow(
            "SELECT * FROM receipt_drafts WHERE image_id=$1 FOR UPDATE", row["id"]
        )
        if row["receipt_id"] is not None or existing is not None:
            raise ValueError("a linked receipt must never be overwritten")
        if draft is not None and (draft["human_edited"] or draft["finalized_receipt_id"] is not None):
            raise ValueError("an edited or finalized draft must never be overwritten")

    @staticmethod
    def _validate_observation_binding(
        row: Any, parts: list[PersistedSourcePart], checkpoint: ObservationCheckpoint,
    ) -> None:
        originals = ImageRow.model_validate(dict(row)).parts()
        manifest = _decoded(row["source_manifest"])
        if (
            checkpoint.source_parts != parts
            or [part.image_path for part in parts] != [part.image_path for part in originals]
            or [(part.photo_index, part.source_sha256) for part in parts]
            != [(item["photo_index"], item["source_sha256"]) for item in manifest]
        ):
            raise ValueError("observation checkpoint source manifest mismatch")
        snapshot = _decoded(row["effective_options"])
        expected_mode = "ocr" if snapshot_ocr(snapshot).enabled else "direct_vision"
        identity = checkpoint.execution_identity
        if (
            row["model_id"] != checkpoint.model_id
            or identity.semantic_policy_sha256 != semantic_policy_sha256(snapshot)
            or snapshot.get("inference_identity", identity.inference_identity) != identity.inference_identity
            or checkpoint.envelope.provenance.used_pipeline != expected_mode
        ):
            raise ValueError("observation checkpoint execution policy mismatch")
        admit_observations(checkpoint.envelope)

    async def save_observation_checkpoint(
        self, claim: AnalysisClaim, checkpoint: ObservationCheckpoint, *, deadline: float | None = None,
    ) -> bool:
        """Commit all-photo observations and their saved artifact references together."""
        if claim.purpose != "inference":
            return False
        # Check raw metadata before JSON conversion and revalidate every nested value
        # using the same representation recovery reads back from PostgreSQL.
        assert_metadata_only(checkpoint)
        checkpoint = ObservationCheckpoint.model_validate(checkpoint.model_dump(mode="json"))
        payload = checkpoint.model_dump(mode="json")
        encoded_payload = _json(payload)
        try:
            async with connection(self.pool, deadline=deadline, transaction=True) as conn:
                row = await self._lock_claim(conn, claim)
                if row is None:
                    return False
                await self._lock_unedited_submission(conn, row)
                if (
                    checkpoint.producer_generation != claim.generation
                    or checkpoint.eligible_generation != claim.generation
                    or checkpoint.producer_attempt_number != claim.attempt_number
                ):
                    raise ValueError("observation checkpoint producer claim mismatch")
                parts = await self._parts(conn, row["id"])
                self._validate_observation_binding(row, parts, checkpoint)
                old = _decoded(row["observation_checkpoint"])
                if old is not None and old["producer_generation"] == claim.generation:
                    if old != payload:
                        raise ValueError("immutable observation checkpoint conflict")
                elif old is not None and (
                    old["eligible_generation"] is not None or await conn.fetchval(
                        "SELECT legacy_events @> $2::jsonb FROM images WHERE id=$1",
                        row["id"], _json([{
                            "event": "observation_checkpoint_retired",
                            "checkpoint_id": str(checkpoint.checkpoint_id),
                        }]),
                    )
                ):
                    raise ValueError("immutable observation checkpoint conflict")

                views = await self._save_views(conn, row["id"], parts, checkpoint.persisted_views)
                await self._validate_metadata_refs(
                    conn, row["id"], parts, payload, checkpoint_view_ids=set(views),
                )
                if not await conn.fetchval(
                    """UPDATE images SET observation_checkpoint=$2::jsonb WHERE id=$1
                    AND status='processing' AND claim_purpose='inference'
                    AND generation=$3 AND attempt_count=$4 AND claim_token=$5
                    AND lease_expires_at>clock_timestamp() RETURNING id""",
                    row["id"], encoded_payload, claim.generation, claim.attempt_number, claim.claim_token,
                ):
                    raise _ExpiredClaim
                return True
        except _ExpiredClaim:
            return False

    async def get_ocr_checkpoint(self, claim: AnalysisClaim, *, deadline: float | None = None) -> OcrCheckpoint | None:
        """Read internal metadata under a live fence, including saved-view paths.

        The application checks the actual OCR profile/runtime before reuse and
        loads saved artifacts through FileProvider. This never reconstructs PNGs.
        """
        if claim.purpose != "inference":
            return None
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await self._lock_claim(conn, claim)
            if row is None or row["ocr_checkpoint"] is None:
                return None
            try:
                return OcrCheckpoint.model_validate(_decoded(row["ocr_checkpoint"]))
            except (json.JSONDecodeError, ValidationError):
                # Invalid cache metadata is recoverable. Keep database/transport errors
                # outside this catch and retain all immutable views and extractions.
                await conn.execute(
                    "UPDATE images SET ocr_checkpoint=NULL WHERE id=$1", claim.image.id
                )
                return None

    async def clear_ocr_checkpoint(self, claim: AnalysisClaim, *, deadline: float | None = None) -> bool:
        """Invalidate only the cache pointer, retaining immutable views and history."""
        if claim.purpose != "inference":
            return False
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            if await self._lock_claim(conn, claim) is None:
                return False
            await conn.execute("UPDATE images SET ocr_checkpoint=NULL WHERE id=$1", claim.image.id)
            return True

    async def ensure_execution_snapshot(
        self,
        claim: AnalysisClaim,
        *,
        model_id: str,
        effective_options: dict[str, Any],
        deadline: float | None = None,
    ) -> ImageRow | None:
        """Bind model selection once under the claim fence and original frozen policy."""
        if claim.purpose != "inference":
            return None
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await self._lock_claim(conn, claim)
            if row is None:
                return None
            snapshot = _decoded(row["effective_options"])
            if not snapshot or snapshot != effective_options:
                raise ValueError("execution_snapshot_invalid")
            if row["model_id"] is None:
                # Explicit current-policy adoption clears only a formerly resolved default.
                # Bind its selection once without changing the already frozen policy.
                row = await conn.fetchrow(
                    "UPDATE images SET model_id=$2 WHERE id=$1 RETURNING *",
                    claim.image.id, model_id,
                )
            return ImageRow.model_validate(dict(row))

    async def hold_claim_for_policy(
        self, claim: AnalysisClaim, reason: str, *, deadline: float | None = None,
    ) -> CommitResult:
        """Release the claim and refund only an inference reservation without observations."""
        if reason != "execution_policy_changed":
            raise ValueError("invalid_wait_reason")
        return await self.hold_claim_for_readiness(claim, reason, deadline=deadline)

    async def hold_claim_for_readiness(
        self, claim: AnalysisClaim, reason: str, *, deadline: float | None = None,
    ) -> CommitResult:
        """Refund only the live claim whose pre-dispatch readiness was disproven."""
        if reason not in READINESS_WAIT_REASONS:
            raise ValueError("invalid_wait_reason")
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            observed = await self._lock_claim(conn, claim)
            if observed is None or observed["receipt_id"] is not None or await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM receipts WHERE image_id=$1)", claim.image.id
            ):
                return CommitResult(stale=True)
            refund = int(claim.purpose == "inference" and _eligible_observations(observed) is None)
            row = await conn.fetchrow(
                """UPDATE images SET status='pending',attempt_count=attempt_count-$3,
                claim_token=NULL,claim_purpose=NULL,lease_expires_at=NULL,processing_at=NULL,retry_at=NULL,
                wait_reason=$2 WHERE id=$1 AND lease_expires_at>clock_timestamp()
                AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id) RETURNING *""",
                claim.image.id, reason, refund,
            )
            if row is None:
                return CommitResult(stale=True)
            return CommitResult(stale=False, submission=_status(row))

    async def set_wait_reason(self, image: ImageRow, reason: str | None) -> bool:
        """A readiness observation cannot mutate a successor generation or claimed state."""
        reasons = COMPLETION_WAIT_REASONS if image.observations_available else READINESS_WAIT_REASONS
        if reason is not None and reason not in reasons:
            raise ValueError("invalid_wait_reason")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""UPDATE images SET wait_reason=$4,
                status=CASE WHEN $7 AND status IN ('failed','timed_out','unreadable') THEN 'pending' ELSE status END
                WHERE id=$1 AND generation=$2 AND status=$3
                AND claim_token IS NOT DISTINCT FROM $5::uuid AND receipt_id IS NULL
                AND claim_purpose IS NOT DISTINCT FROM $6
                AND (observation_checkpoint->>'eligible_generation' IS NOT DISTINCT FROM generation::text)=$7
                AND (NOT $7 OR ({_UNMANAGED_CLASSIFICATION_UNSETTLED})=$8)
                AND (status='pending' OR (status='processing' AND lease_expires_at>clock_timestamp())
                    OR ($7 AND status IN ('failed','timed_out','unreadable')))
                AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id) RETURNING id""",
                image.id, image.generation, image.status, reason, image.claim_token, image.claim_purpose,
                image.observations_available, image.unmanaged_classification_unsettled,
            )
        return row is not None

    async def delete_resource(
        self, resource_id: UUID, *, receipt: bool, scope: AccessScope,
        validate_files: Callable[[DeletedResource], None],
    ) -> DeletedResource | None:
        """Image-before-receipt lock, captured ownership, and one atomic DB deletion."""
        async with self.pool.acquire() as conn, conn.transaction():
            kind = "receipt" if receipt else "image"
            pending = await conn.fetchrow("SELECT * FROM resource_deletions WHERE resource_id=$1 AND resource_kind=$2 AND ($4 OR owner_id=$3) FOR UPDATE", resource_id, kind, scope.user_id, scope.can_see_all)
            if pending is not None:
                return DeletedResource.model_validate(_decoded(pending["files"]))
            receipt_row = None
            if receipt:
                receipt_row = await self._receipts._lock_review_receipt(conn, resource_id, scope.user_id, scope.can_see_all)
                if receipt_row is None:
                    return None
                image_id = receipt_row["image_id"]
            else:
                image_id = resource_id
            row = await conn.fetchrow("SELECT * FROM images WHERE id=$1 AND ($3 OR user_id=$2) FOR UPDATE", image_id, scope.user_id, scope.can_see_all) if image_id else None
            if not receipt and row is None:
                return None
            if receipt and image_id is not None and row is None:
                raise ValueError("invalid_image_owner")
            if row is not None:
                linked = await conn.fetch("SELECT id FROM receipts WHERE image_id=$1", image_id)
                if any(item["id"] != (resource_id if receipt else None) for item in linked):
                    raise ValueError("resource_still_referenced")
                if not receipt and row["status"] not in {"pending", "needs_review", "failed", "timed_out", "unreadable"}:
                    raise ValueError("active_submission")
            deleted = DeletedResource(image_id=image_id, receipt_id=resource_id if receipt else None)
            if row is not None:
                image = ImageRow.model_validate(dict(row))
                originals = {part["photo_index"]: part for part in await conn.fetch(
                    "SELECT * FROM image_source_parts WHERE image_id=$1 ORDER BY photo_index FOR SHARE", image_id)}
                for index, part in enumerate(image.parts()):
                    if original := originals.get(index):
                        if original["image_path"] != part.image_path:
                            raise ValueError("original_artifact_identity_changed")
                        part.part_id = original["part_id"]
                        part.source_sha256 = original["source_sha256"]
                        part.photo_index = index
                    shared = await conn.fetchval("""SELECT EXISTS(SELECT 1 FROM images WHERE id<>$1 AND
                        (image_path=$2 OR additional_images @> jsonb_build_array(jsonb_build_object('image_path',$2::text))))
                        OR EXISTS(SELECT 1 FROM image_source_parts WHERE image_id<>$1 AND image_path=$2)""", image_id, part.image_path)
                    if not shared:
                        deleted.parts.append(part)
                views = await conn.fetch("""SELECT v.*,p.photo_index,p.source_sha256,p.image_id AS part_image_id
                    FROM extraction_working_views v JOIN image_source_parts p ON p.part_id=v.part_id
                    WHERE v.image_id=$1""", image_id)
                for view in views:
                    if view["part_image_id"] != image_id:
                        raise ValueError("invalid_working_artifact_owner")
                    shared = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM extraction_working_views WHERE image_id<>$1 AND artifact_path=$2)", image_id, view["artifact_path"])
                    if not shared:
                        deleted.working_views.append(PersistedWorkingView(
                            part_id=view["part_id"], photo_index=view["photo_index"], source_sha256=view["source_sha256"],
                            artifact_path=view["artifact_path"], metadata=WorkingImageMetadata.model_validate(_decoded(view["metadata"]))))
            validate_files(deleted)
            if receipt_row is not None:
                owner_id = receipt_row["user_id"]
            else:
                assert row is not None
                owner_id = row["user_id"]
            await conn.execute("INSERT INTO resource_deletions(resource_id,resource_kind,owner_id,files) VALUES($1,$2,$3,$4::jsonb)", resource_id, kind, owner_id, _json(deleted.model_dump(mode="json")))
            if receipt:
                await conn.execute("DELETE FROM receipts WHERE id=$1", resource_id)
            if row is not None:
                await conn.execute("DELETE FROM images WHERE id=$1", image_id)
        return deleted

    async def cleanup_deleted_resource(self, resource_id: UUID, *, receipt: bool, scope: AccessScope,
                                       remove_files: Callable[[DeletedResource], None]) -> bool:
        """Serialize retries; retain metadata if cleanup, proof, cancellation or commit fails."""
        async with self.pool.acquire() as conn, conn.transaction():
            kind = "receipt" if receipt else "image"
            row = await conn.fetchrow("SELECT files FROM resource_deletions WHERE resource_id=$1 AND resource_kind=$2 AND ($4 OR owner_id=$3) FOR UPDATE", resource_id, kind, scope.user_id, scope.can_see_all)
            if row is None:
                return False
            deleted = DeletedResource.model_validate(_decoded(row["files"]))
            if await conn.fetchval("SELECT EXISTS(SELECT 1 FROM images WHERE id=$1) OR EXISTS(SELECT 1 FROM receipts WHERE id=$2 OR image_id=$1)", deleted.image_id, deleted.receipt_id):
                raise ValueError("resource_still_referenced")
            # All live file references must be absent at the retry boundary. Failed proof retains
            # the manifest and files; it never broadens the deletion set.
            for path in [p.image_path for p in deleted.parts] + [v.artifact_path for v in deleted.working_views] + list(deleted.thumbnail_sha256):
                if await conn.fetchval("""SELECT EXISTS(SELECT 1 FROM images WHERE image_path=$1 OR thumbnail_path=$1 OR
                    additional_images @> jsonb_build_array(jsonb_build_object('image_path',$1::text)) OR
                    additional_images @> jsonb_build_array(jsonb_build_object('thumbnail_path',$1::text)))
                    OR EXISTS(SELECT 1 FROM image_source_parts WHERE image_path=$1)
                    OR EXISTS(SELECT 1 FROM extraction_working_views WHERE artifact_path=$1)""", path):
                    raise ValueError("resource_still_referenced")
            remove_files(deleted)
            await conn.execute("DELETE FROM resource_deletions WHERE resource_id=$1 AND resource_kind=$2", resource_id, kind)
        return True

    async def due_candidates(self, *, purpose: ClaimPurpose | None = "inference") -> list[ImageRow]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""SELECT *,
                {_UNMANAGED_CLASSIFICATION_UNSETTLED} AS unmanaged_classification_unsettled
                FROM images WHERE receipt_id IS NULL AND
                NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id) AND
                (status='pending' OR (status IN ('failed','timed_out','unreadable')
                    AND observation_checkpoint->>'eligible_generation'=generation::text))
                AND (retry_at IS NULL OR retry_at<=clock_timestamp())
                AND CASE WHEN observation_checkpoint->>'eligible_generation'=generation::text
                    THEN $2::text IS DISTINCT FROM 'inference'
                    ELSE $2::text IS DISTINCT FROM 'completion' AND wait_reason IS DISTINCT FROM $1 END
                ORDER BY created_at,id""", OBSERVATION_RECOVERY_WAIT, purpose)
        return [ImageRow.model_validate(dict(row)) for row in rows]

    async def next_wake_delay(
        self,
        maximum: float,
        *,
        deferred_image_ids: Sequence[UUID] = (),
    ) -> float:
        """Include jobs that became due during work, but defer failed readiness probes."""
        async with self.pool.acquire() as conn:
            # An expired row can be skipped by a reaper because another transaction
            # holds it. Revisit it promptly while giving the lock holder time to finish.
            delay = await conn.fetchval(
                """WITH wake_clock AS (SELECT clock_timestamp() AS sampled_at)
                SELECT min(EXTRACT(EPOCH FROM CASE
                    WHEN status<>'processing' THEN COALESCE(retry_at,sampled_at)
                    WHEN lease_expires_at<=sampled_at THEN sampled_at+interval '1 second'
                    ELSE lease_expires_at END-sampled_at))
                FROM images CROSS JOIN wake_clock
                WHERE receipt_id IS NULL
                AND (status='processing' OR (status='pending' AND wait_reason IS DISTINCT FROM $2)
                    OR (status IN ('pending','failed','timed_out','unreadable')
                        AND observation_checkpoint->>'eligible_generation'=generation::text))
                AND id<>ALL($1::uuid[])
                AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id)""",
                list(deferred_image_ids), OBSERVATION_RECOVERY_WAIT,
            )
        return min(maximum, max(0.0, float(delay))) if delay is not None else maximum

    async def save_ocr_checkpoint(self, claim: AnalysisClaim, checkpoint: OcrCheckpoint, *, deadline: float | None = None) -> bool:
        if claim.purpose != "inference":
            return False
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await self._lock_claim(conn, claim)
            if row is None:
                return False
            parts = await self._parts(conn, row["id"])
            if checkpoint.ordered_source_sha256 != [part.source_sha256 for part in parts]:
                raise ValueError("checkpoint ordered source hashes mismatch")
            views = await self._save_views(conn, row["id"], parts, checkpoint.persisted_views)
            for index, raw in enumerate(checkpoint.pages):
                page = OcrPageMetadata.model_validate(raw)
                if (
                    page.photo_index != index
                    or page.profile_id != checkpoint.profile_id
                    or page.runtime_fingerprint != checkpoint.runtime_fingerprint
                ):
                    raise ValueError("checkpoint page order/profile mismatch")
                self._require_view(
                    parts, views, page.photo_index, page.source_sha256, page.working_image
                )
            payload = checkpoint.model_dump(mode="json")
            await self._validate_metadata_refs(conn, row["id"], parts, payload)
            old = _decoded(row["ocr_checkpoint"])
            if old is not None and (
                old["ordered_source_sha256"] != payload["ordered_source_sha256"]
                or old["profile_id"] != payload["profile_id"]
                or old["runtime_fingerprint"] != payload["runtime_fingerprint"]
                or payload["pages"][: len(old["pages"])] != old["pages"]
            ):
                raise ValueError("immutable checkpoint source/profile/page conflict")
            await conn.execute(
                "UPDATE images SET ocr_checkpoint=$2::jsonb WHERE id=$1", row["id"], _json(payload)
            )
            return True

    @staticmethod
    def _completion_identity(claim: AnalysisClaim, checkpoint_id: UUID) -> dict[str, Any]:
        return {
            "event": "checkpoint_completed",
            "checkpoint_id": str(checkpoint_id),
            "producer_generation": claim.generation,
            "producer_attempt_number": claim.attempt_number,
            "completion_token_sha256": sha256(claim.claim_token.bytes).hexdigest(),
        }

    async def _checkpoint_completion_replay(
        self, conn: asyncpg.Connection, claim: AnalysisClaim, checkpoint_id: UUID,
    ) -> CommitResult:
        row = await conn.fetchrow(
            """SELECT * FROM images WHERE id=$1 AND status='analyzed' AND generation=$2
            AND user_id IS NOT DISTINCT FROM $3::uuid AND legacy_events @> $4::jsonb
            FOR UPDATE""",
            claim.image.id, claim.generation, claim.image.user_id,
            _json([self._completion_identity(claim, checkpoint_id)]),
        )
        if row is None:
            return CommitResult(stale=True)
        receipt = await conn.fetchrow(
            "SELECT * FROM receipts WHERE id=$1 AND image_id=$2 AND user_id IS NOT DISTINCT FROM $3::uuid",
            row["receipt_id"], row["id"], row["user_id"],
        )
        if receipt is None:
            return CommitResult(stale=True)
        return CommitResult(
            stale=False, submission=_status(row), receipt=self._receipts._receipt_row_from_record(receipt),
        )

    async def get_checkpoint_completion_result(
        self, claim: AnalysisClaim, checkpoint_id: UUID, *, deadline: float | None = None,
    ) -> CommitResult:
        if claim.purpose != "completion" or claim.checkpoint_id != checkpoint_id:
            return CommitResult(stale=True)
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            return await self._checkpoint_completion_replay(conn, claim, checkpoint_id)

    async def record_unmanaged_classification_event(
        self, claim: AnalysisClaim, checkpoint_id: UUID, *,
        state: Literal["started", "settled"], deadline: float | None = None,
    ) -> bool:
        """Record that the optional request may start or that its provider returned."""
        if claim.purpose != "inference" or state not in ("started", "settled"):
            return False
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await self._lock_claim(conn, claim)
            if row is None or row["receipt_id"] is not None:
                return False
            started = _unmanaged_classification_start(row)
            if (
                started is None or started["checkpoint_id"] != str(checkpoint_id)
                or not _is_unmanaged_checkpoint_policy(row)
            ):
                return False
            settled = started | {"event": "unmanaged_classification_settled"}
            already_started = _has_classification_event(row, started)
            already_settled = _has_classification_event(row, settled)
            if state == "settled":
                if not already_started:
                    return False
                if already_settled:
                    return True
                event = settled | {"settlement": "provider_returned"}
            else:
                if already_settled:
                    return False
                if already_started:
                    return True
                event = started
            updated = await conn.fetchval(
                """UPDATE images SET legacy_events=legacy_events || $2::jsonb WHERE id=$1
                AND lease_expires_at>clock_timestamp() AND receipt_id IS NULL
                AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id) RETURNING id""",
                row["id"], _json([event]),
            )
            return updated is not None

    async def unmanaged_classification_status(
        self, image_id: UUID, *, inference_identity: str, deadline: float | None = None,
    ) -> dict[str, str | int] | None:
        """Return only the identifiers needed by an operator to reconcile one request."""
        async with connection(self.pool, deadline=deadline) as conn:
            row = await conn.fetchrow(
                "SELECT *, lease_expires_at>clock_timestamp() AS claim_live FROM images WHERE id=$1",
                image_id,
            )
        if row is None:
            return None
        event = _unmanaged_classification_start(row)
        if event is None or event["inference_identity"] != inference_identity:
            return None
        state = "not_required"
        if _classification_settlement_required(row):
            state = "busy" if row["claim_live"] else "recovery_required"
        return {
            "state": state, "image_id": str(image_id), "generation": row["generation"],
            "checkpoint_id": event["checkpoint_id"],
        }

    async def reconcile_unmanaged_classification(
        self, image_id: UUID, checkpoint_id: UUID, *, expected_generation: int,
        inference_identity: str, request_settled: bool, deadline: float | None = None,
    ) -> bool:
        """Accept explicit operator confirmation for one exact, inactive checkpoint."""
        if request_settled is not True:
            return False
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await conn.fetchrow(
                """SELECT * FROM images WHERE id=$1 AND generation=$2 AND claim_token IS NULL
                AND status IN ('pending','failed','timed_out','unreadable') AND receipt_id IS NULL
                AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id) FOR UPDATE""",
                image_id, expected_generation,
            )
            if row is None:
                return False
            event = _unmanaged_classification_start(row)
            if (
                event is None or event["checkpoint_id"] != str(checkpoint_id)
                or event["inference_identity"] != inference_identity
                or not _is_unmanaged_checkpoint_policy(row)
                or not _has_classification_event(row, event)
            ):
                return False
            settled = event | {"event": "unmanaged_classification_settled"}
            if not _has_classification_event(row, settled):
                await conn.execute(
                    """UPDATE images SET status='pending',wait_reason=$2,retry_at=NULL,
                    legacy_events=legacy_events || $3::jsonb WHERE id=$1""",
                    image_id, OBSERVATION_RECOVERY_WAIT,
                    _json([settled | {"settlement": "operator_attested"}]),
                )
            return True

    async def defer_checkpoint_completion(
        self, claim: AnalysisClaim, checkpoint_id: UUID, reason: str,
        *, deadline: float | None = None,
    ) -> CommitResult:
        if claim.purpose != "completion" or claim.checkpoint_id != checkpoint_id:
            return CommitResult(stale=True)
        if reason not in COMPLETION_WAIT_REASONS:
            raise ValueError("invalid_wait_reason")
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            if await self._lock_claim(conn, claim) is None:
                return CommitResult(stale=True)
            row = await conn.fetchrow(
                """UPDATE images SET status='pending',wait_reason=$2,retry_at=NULL,
                claim_token=NULL,claim_purpose=NULL,lease_expires_at=NULL,processing_at=NULL
                WHERE id=$1 AND generation=$3 AND claim_token=$4 AND claim_purpose='completion'
                AND lease_expires_at>clock_timestamp() AND receipt_id IS NULL
                AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id)
                RETURNING *""",
                claim.image.id, reason, claim.generation, claim.claim_token,
            )
            return CommitResult(stale=row is None, submission=_status(row) if row else None)

    async def finish_checkpoint_completion(
        self, claim: AnalysisClaim, checkpoint_id: UUID, outcome: CheckpointCompletionOutcome,
        *, deadline: float | None = None,
    ) -> CommitResult:
        if claim.purpose != "completion" or claim.checkpoint_id != checkpoint_id:
            return CommitResult(stale=True)
        try:
            async with connection(self.pool, deadline=deadline, transaction=True) as conn:
                row = await self._lock_claim(conn, claim)
                if row is None:
                    return await self._checkpoint_completion_replay(conn, claim, checkpoint_id)
                if _classification_settlement_required(row):
                    return CommitResult(stale=True)
                return await self._seal_checkpoint_completion(conn, row, claim, checkpoint_id, outcome)
        except _ExpiredClaim:
            return CommitResult(stale=True)

    async def _seal_checkpoint_completion(
        self, conn: asyncpg.Connection, row: Any, claim: AnalysisClaim, checkpoint_id: UUID,
        outcome: CheckpointCompletionOutcome,
    ) -> CommitResult:
        await self._lock_unedited_submission(conn, row)
        checkpoint = ObservationCheckpoint.model_validate(_eligible_observations(row))
        parts = await self._parts(conn, row["id"])
        self._validate_observation_binding(row, parts, checkpoint)
        outcome = CheckpointCompletionOutcome.model_validate(outcome)
        envelope = outcome.envelope
        if (
            checkpoint.producer_attempt_number != claim.attempt_number
            or outcome.persisted_views != checkpoint.persisted_views
            or envelope != checkpoint.recovered_envelope()
        ):
            raise ValueError("checkpoint completion differs from saved observations")
        if self._working_view_checker is None:
            raise ValueError("checkpoint working artifact checker required")
        for view in checkpoint.persisted_views:
            saved = await conn.fetchrow(
                "SELECT * FROM extraction_working_views WHERE view_id=$1 FOR SHARE", view.metadata.view_id,
            )
            if (
                saved is None or saved["image_id"] != row["id"] or saved["part_id"] != view.part_id
                or saved["artifact_path"] != view.artifact_path
                or _decoded(saved["metadata"]) != view.metadata.model_dump(mode="json")
            ):
                raise ValueError("checkpoint working artifact mismatch")
            try:
                await asyncio.to_thread(
                    self._working_view_checker, row["id"], view.artifact_path, view.metadata,
                )
            except OSError as exc:
                raise ValueError("checkpoint working artifact unavailable") from exc
        await self._validate_metadata_refs(
            conn, row["id"], parts,
            {"envelope": envelope.model_dump(mode="json"), "review": outcome.review.model_dump(mode="json")},
            checkpoint_view_ids={view.metadata.view_id for view in checkpoint.persisted_views},
        )
        try:
            originals = await self._receipts._review_originals(conn, row["id"])
        except ValueError as exc:
            raise ValueError("source_integrity_failed") from exc
        if not originals or not all(original.available and original.sha256_matches for original in originals):
            raise ValueError("source_integrity_failed")
        document = admit_observations(envelope)
        review = evaluate_receipt(
            document, envelope.amount_policy, envelope.field_sources, envelope.overlap_decisions,
            originals, retained_findings=envelope.findings,
        )
        if not review.can_finalize or outcome.review != review:
            raise ValueError("checkpoint review is not finalizable")
        extraction_id = await conn.fetchval(
            """SELECT id FROM receipt_extractions WHERE image_id=$1 AND generation=$2
            AND attempt_number=$3 AND outcome_kind IN ('retry','terminal')""",
            row["id"], checkpoint.producer_generation, checkpoint.producer_attempt_number,
        )
        if extraction_id is None:
            raise ValueError("checkpoint producing outcome missing")
        identity = self._completion_identity(claim, checkpoint_id)
        receipt = await self._publish_extracted_receipt(
            conn, row, extraction_id, envelope, review, checks_version=review.checks_version,
            verified=row["bypass_review"] and review.can_auto_verify,
            completion=checkpoint,
        )
        updated = await conn.fetchrow(
            """UPDATE images SET status='analyzed',receipt_id=$2,analyzed_at=clock_timestamp(),
            result_code=NULL,error=NULL,wait_reason=NULL,retry_at=NULL,draft_available=FALSE,
            claim_token=NULL,claim_purpose=NULL,lease_expires_at=NULL,processing_at=NULL,
            observation_checkpoint=jsonb_set(observation_checkpoint,'{eligible_generation}','null'),
            legacy_events=legacy_events || $6::jsonb
            WHERE id=$1 AND lease_expires_at>clock_timestamp() AND receipt_id IS NULL
            AND generation=$3 AND claim_token=$4 AND status='processing' AND claim_purpose='completion'
            AND observation_checkpoint->>'checkpoint_id'=$5 RETURNING *""",
            row["id"], receipt.id, claim.generation, claim.claim_token, str(checkpoint_id),
            _json([identity | {"receipt_id": str(receipt.id), "extraction_id": str(extraction_id)}]),
        )
        if updated is None:
            raise _ExpiredClaim
        return CommitResult(stale=False, submission=_status(updated), receipt=receipt)

    async def _publish_extracted_receipt(
        self, conn: asyncpg.Connection, image: Any, extraction_id: UUID,
        envelope: PersistedInferenceEnvelope, review: ReviewEvaluation,
        *, checks_version: str, verified: bool, completion: ObservationCheckpoint | None = None,
    ) -> ReceiptRow:
        assert envelope.receipt is not None
        fields = {key: [ref.model_dump(mode="json") for ref in refs]
                  for key, refs in envelope.field_sources.items()}
        receipt = await self._receipts.insert_receipt(
            conn, envelope.receipt, image_id=image["id"], user_id=image["user_id"],
            status="verified" if verified else "unverified", verified=verified,
        )
        record = await conn.fetchrow(
            """UPDATE receipts SET extraction_id=$2,amount_policy=$3::jsonb,
            findings=$4::jsonb,field_sources=$5::jsonb,checks_version=$6,
            verification_source=$7,overlap_decisions=$8::jsonb WHERE id=$1 RETURNING *""",
            receipt.id, extraction_id, envelope.amount_policy.model_dump_json(),
            _json([finding.model_dump(mode="json") for finding in review.findings]),
            _json(fields), checks_version, "automatic_bypass" if verified else None,
            _json(envelope.overlap_decisions),
        )
        receipt = self._receipts._receipt_row_from_record(record)
        snapshot: dict[str, Any] = {
            "receipt": receipt.model_dump(mode="json"),
            "extraction_id": str(extraction_id),
            "document": envelope.receipt.model_dump(mode="json"),
            "field_sources": fields,
            "collection_ids": [str(link["collection_id"]) for link in await conn.fetch(
                "SELECT collection_id FROM receipt_collections WHERE receipt_id=$1 ORDER BY collection_id",
                receipt.id,
            )],
        }
        event = "extracted"
        if completion is not None:
            event = "checkpoint_completed"
            snapshot.update({
                "checkpoint_id": str(completion.checkpoint_id),
                "producer_generation": completion.producer_generation,
                "producer_attempt_number": completion.producer_attempt_number,
                "provenance": envelope.provenance.model_dump(mode="json"),
                "classification_enrichment": envelope.classification_enrichment.model_dump(mode="json"),
            })
        await conn.execute(
            """INSERT INTO receipt_review_revisions(receipt_id,operation_revision,
            review_revision,event_kind,snapshot) VALUES($1,1,1,$2,$3::jsonb)""",
            receipt.id, event, _json(snapshot),
        )
        return receipt

    async def finish_attempt(self, claim: AnalysisClaim, outcome: AttemptOutcome, *, deadline: float | None = None) -> CommitResult:
        if claim.purpose != "inference":
            return CommitResult(stale=True)
        try:
            return await self._finish_attempt_transaction(claim, outcome, deadline=deadline)
        except _ExpiredClaim:
            return CommitResult(stale=True)

    async def _finish_attempt_transaction(
        self,
        claim: AnalysisClaim,
        outcome: AttemptOutcome,
        *,
        deadline: float | None = None,
    ) -> CommitResult:
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            row = await self._lock_claim(conn, claim)
            if row is None:
                return CommitResult(stale=True)
            failed = outcome.kind in {"retry", "terminal"}
            settlement_required = _classification_settlement_required(row)
            if settlement_required and not failed:
                return CommitResult(stale=True)
            await self._lock_unedited_submission(conn, row)
            parts = await self._parts(conn, row["id"])
            views = await self._save_views(conn, row["id"], parts, outcome.persisted_views)
            envelope = outcome.envelope
            packed = None
            payload = None
            if envelope is not None:
                if envelope.provenance.ordered_source_sha256 != [
                    p.source_sha256 for p in parts
                ] or envelope.provenance.source_image_count != len(parts):
                    raise ValueError("envelope ordered source hashes mismatch")
                for page in envelope.ocr_pages:
                    self._require_view(
                        parts, views, page.photo_index, page.source_sha256, page.working_image
                    )
                for view in envelope.input_views:
                    self._require_view(
                        parts, views, view.photo_index, view.source_sha256, view.working_image
                    )
                packed = pack_inference_metadata(envelope)
                payload = packed.model_dump(mode="json")
            review_payload = outcome.review.model_dump(mode="json") if outcome.review else None
            saved_views = [view.model_dump(mode="json") for view in outcome.persisted_views]
            await self._validate_metadata_refs(
                conn,
                row["id"],
                parts,
                {"envelope": payload, "review": review_payload, "saved_views": saved_views},
            )
            checkpoint = _eligible_observations(row)
            history = []
            if checkpoint is not None and failed:
                history.append(_observation_history(checkpoint, "observation_checkpoint_attempt_failed"))
                if payload is None:
                    payload = checkpoint["envelope"]
                    saved_views = checkpoint["persisted_views"]
            extraction_id = await conn.fetchval(
                """INSERT INTO receipt_extractions(image_id,generation,
                attempt_number,outcome_kind,result_code,envelope,review,source_views)
                VALUES($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb) RETURNING id""",
                row["id"],
                claim.generation,
                claim.attempt_number,
                outcome.kind,
                outcome.result_code,
                _json(payload) if payload is not None else None,
                _json(review_payload) if review_payload is not None else None,
                _json(saved_views),
            )
            receipt = None
            status = "failed"
            delay = None
            result_code = outcome.result_code
            wait_reason = row["wait_reason"]
            if outcome.kind == "complete":
                assert (
                    envelope is not None
                    and envelope.receipt is not None
                    and outcome.review is not None
                )
                assert packed is not None
                availability = [OriginalAvailability(part_id=p.part_id, photo_index=p.photo_index, available=True, sha256_matches=True) for p in parts]
                if row["bypass_review"]:
                    availability = await self._receipts._review_originals(conn, row["id"])
                financial = evaluate_receipt(envelope.receipt, envelope.amount_policy, envelope.field_sources,
                                             envelope.overlap_decisions, availability, retained_findings=envelope.findings)
                verified = row["bypass_review"] and outcome.review.can_auto_verify and financial.can_auto_verify
                receipt = await self._publish_extracted_receipt(
                    conn, row, extraction_id, packed,
                    financial, checks_version=outcome.review.checks_version, verified=verified,
                )
                status = "analyzed"
            elif outcome.kind == "partial":
                assert envelope is not None and envelope.draft is not None
                assert payload is not None
                await conn.execute(
                    """INSERT INTO receipt_drafts(image_id,extraction_id,generation,document,
                    amount_policy,findings,field_sources,overlap_decisions) VALUES($1,$2,$3,$4::jsonb,$5::jsonb,$6::jsonb,$7::jsonb,$8::jsonb)
                    ON CONFLICT(image_id) DO UPDATE SET extraction_id=EXCLUDED.extraction_id,
                    generation=EXCLUDED.generation,revision=receipt_drafts.revision+1,
                    document=EXCLUDED.document,amount_policy=EXCLUDED.amount_policy,
                    findings=EXCLUDED.findings,field_sources=EXCLUDED.field_sources,
                    overlap_decisions=EXCLUDED.overlap_decisions""",
                    row["id"],
                    extraction_id,
                    claim.generation,
                    envelope.draft.model_dump_json(),
                    envelope.amount_policy.model_dump_json(),
                    _json(
                        [finding.model_dump(mode="json") for finding in outcome.review.findings]
                        if outcome.review
                        else payload["findings"]
                    ),
                    _json(payload["field_sources"]),
                    _json(payload["overlap_decisions"]),
                )
                status = "needs_review"
            elif failed and checkpoint is not None:
                status = "pending"
                if settlement_required:
                    wait_reason = "model_request_unsettled"
                elif wait_reason != "execution_policy_changed":
                    wait_reason = OBSERVATION_RECOVERY_WAIT
            elif outcome.kind == "retry" and row["attempt_count"] < row["max_attempts"]:
                status = "pending"
                if outcome.retry_delay_seconds is None:
                    delay = self._worker.retry_delay(claim.attempt_number)
                else:
                    delay = min(outcome.retry_delay_seconds, self._worker.retry_backoff_cap_seconds)
            elif outcome.kind == "retry":
                result_code = result_code or "attempt_limit"
            elif outcome.result_code in {"timed_out", "unreadable"}:
                status = outcome.result_code
            updated = await conn.fetchrow(
                """UPDATE images SET status=$2,receipt_id=$3,result_code=$4,
                error=$4,claim_token=NULL,claim_purpose=NULL,lease_expires_at=NULL,processing_at=NULL,
                wait_reason=$9,legacy_events=legacy_events || $10::jsonb,
                observation_checkpoint=CASE WHEN $11 AND observation_checkpoint IS NOT NULL
                    THEN jsonb_set(observation_checkpoint,'{eligible_generation}','null')
                    ELSE observation_checkpoint END,
                retry_at=CASE WHEN $5::double precision IS NULL THEN NULL ELSE
                    clock_timestamp()+$5::double precision*interval '1 second' END,
                analyzed_at=CASE WHEN $2::varchar IN ('analyzed','needs_review') THEN clock_timestamp() ELSE analyzed_at END,
                draft_available=$6 WHERE id=$1 AND lease_expires_at>clock_timestamp()
                AND generation=$7 AND claim_token=$8 AND status='processing'
                AND claim_purpose='inference' RETURNING *""",
                row["id"],
                status,
                receipt.id if receipt else None,
                result_code,
                delay,
                status == "needs_review",
                claim.generation,
                claim.claim_token,
                wait_reason,
                _json(history),
                not failed,
            )
            if updated is None:
                raise _ExpiredClaim
            return CommitResult(stale=False, submission=_status(updated), receipt=receipt)

    async def retry_submission(
        self,
        image_id: UUID,
        request: RetryRequest,
        *,
        scope: AccessScope,
        replacement_snapshot: dict[str, Any] | None = None,
    ) -> SubmissionStatus | None:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM images WHERE id=$1 AND (user_id=$2 OR $3) FOR UPDATE",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
            if row is None:
                return None
            old = await conn.fetchrow(
                "SELECT * FROM image_retry_requests WHERE image_id=$1 AND request_key=$2",
                image_id,
                request.request_key,
            )
            if old is not None:
                old_options = _decoded(old["request_options"])
                if (
                    old["expected_generation"] != request.expected_generation
                    or old_options.get("policy", "preserve") != request.policy
                ):
                    raise ValueError("retry key payload conflict")
                return SubmissionStatus.model_validate(_decoded(old["result"]))
            if row["generation"] != request.expected_generation:
                raise ValueError("retry generation conflict")
            if _classification_settlement_required(row):
                raise ValueError("model_request_unsettled")
            if row["receipt_id"] is not None or await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM receipts WHERE image_id=$1)", image_id
            ):
                raise ValueError("a linked receipt cannot be retried")
            draft = await conn.fetchrow(
                "SELECT * FROM receipt_drafts WHERE image_id=$1 FOR UPDATE", image_id
            )
            if draft and (draft["human_edited"] or draft["finalized_receipt_id"] is not None):
                raise ValueError("an edited draft cannot be retried")
            expired_processing = row["status"] == "processing" and await conn.fetchval(
                "SELECT $1::timestamptz<=clock_timestamp()", row["lease_expires_at"]
            )
            policy_held = row["wait_reason"] == "execution_policy_changed" and (
                row["status"] == "pending" or expired_processing
            )
            if row["status"] not in {"failed", "timed_out", "unreadable", "needs_review"} and not (
                policy_held and request.policy == "current"
            ):
                raise ValueError("submission is not terminal/retryable")
            snapshot = _decoded(row["effective_options"])
            model_id = row["model_id"]
            history: list[dict[str, Any]] = []
            if request.policy == "current":
                if replacement_snapshot is None:
                    raise ValueError("replacement policy required")
                ocr = snapshot_ocr(replacement_snapshot)
                if (
                    replacement_snapshot.get("execution_policy_version") != EXECUTION_POLICY_VERSION
                    or ocr.allow_vision_fallback or ocr.vision_fallback_model_id is not None
                ):
                    raise ValueError("execution_snapshot_invalid")
                history = [{
                    "event": "execution_policy_adopted", "generation": row["generation"],
                    "effective_options": snapshot, "model_id": model_id,
                    "requested_options": _decoded(row["requested_options"]),
                    "request_fingerprint": row["request_fingerprint"],
                }]
                snapshot = replacement_snapshot
                model_id = _decoded(row["requested_options"]).get("model_id")
                if expired_processing and row["claim_purpose"] != "completion":
                    # Preserve the real abandoned attempt exactly as normal lease recovery does.
                    await self._record_abandonment(
                        conn, row, exhausted=_eligible_observations(row) is None
                        and row["attempt_count"] >= row["max_attempts"],
                    )
            checkpoint = _decoded(row["observation_checkpoint"])
            if checkpoint is not None and checkpoint["producer_generation"] == row["generation"]:
                retired = _observation_history(checkpoint, "observation_checkpoint_retired")
                retired["checkpoint"] = checkpoint
                history.append(retired)
            updated = await conn.fetchrow(
                """UPDATE images SET generation=generation+1,attempt_count=0,
                status='pending',retry_at=NULL,claim_token=NULL,claim_purpose=NULL,
                lease_expires_at=NULL,processing_at=NULL,
                analyzed_at=NULL,error=NULL,result_code=NULL,wait_reason=NULL,ocr_checkpoint=NULL,
                observation_checkpoint=jsonb_set(observation_checkpoint,'{eligible_generation}','null'),
                effective_options=$2::jsonb,model_id=$3,legacy_events=legacy_events || $4::jsonb
                WHERE id=$1 RETURNING *""",
                image_id,
                _json(snapshot), model_id, _json(history),
            )
            result = _status(updated)
            await conn.execute(
                """INSERT INTO image_retry_requests(image_id,request_key,expected_generation,result,request_options)
                VALUES($1,$2,$3,$4::jsonb,$5::jsonb)""",
                image_id,
                request.request_key,
                request.expected_generation,
                result.model_dump_json(),
                _json({"policy": request.policy}),
            )
            return result

    @staticmethod
    def _public_view(image_id: UUID, photo_index: int, raw: Any) -> SourceView:
        metadata = WorkingImageMetadata.model_validate(_decoded(raw))
        return SourceView(
            **metadata.model_dump(exclude={"media_type"}),
            photo_index=photo_index,
            coordinate_space="working",
            url=f"/api/v1/images/{image_id}/views/{metadata.view_id}",
        )

    async def _source_views(self, conn: asyncpg.Connection, image_id: UUID) -> list[SourceView]:
        rows = await conn.fetch(
            """SELECT v.metadata,p.photo_index FROM extraction_working_views v
            JOIN image_source_parts p ON p.part_id=v.part_id WHERE v.image_id=$1
            ORDER BY p.photo_index,v.created_at,v.view_id""",
            image_id,
        )
        return [self._public_view(image_id, r["photo_index"], r["metadata"]) for r in rows]

    async def get_working_view(
        self,
        image_id: UUID,
        view_id: UUID,
        *,
        scope: AccessScope,
    ) -> PersistedWorkingView | None:
        """Internal file manifest, accessible only through the owning submission."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT v.*,p.photo_index,p.source_sha256
                FROM extraction_working_views v JOIN images i ON i.id=v.image_id
                JOIN image_source_parts p ON p.part_id=v.part_id
                WHERE i.id=$1 AND v.view_id=$2 AND (i.user_id=$3 OR $4)""",
                image_id,
                view_id,
                scope.user_id,
                scope.can_see_all,
            )
        if row is None:
            return None
        return PersistedWorkingView(
            part_id=row["part_id"],
            photo_index=row["photo_index"],
            source_sha256=row["source_sha256"],
            artifact_path=row["artifact_path"],
            metadata=WorkingImageMetadata.model_validate(_decoded(row["metadata"])),
        )

    async def get_extraction(
        self,
        image_id: UUID,
        extraction_id: UUID,
        *,
        scope: AccessScope,
    ) -> dict[str, Any] | None:
        """Public immutable snapshot: no internal artifact paths or transport payloads."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT e.* FROM receipt_extractions e
                JOIN images i ON i.id=e.image_id WHERE i.id=$1 AND e.id=$2
                AND (i.user_id=$3 OR $4)""",
                image_id,
                extraction_id,
                scope.user_id,
                scope.can_see_all,
            )
        if row is None:
            return None
        payload = dict(row)
        payload["envelope"] = _decoded(payload["envelope"])
        payload["review"] = _decoded(payload["review"])
        views = [PersistedWorkingView.model_validate(v) for v in _decoded(payload["source_views"])]
        public = [
            self._public_view(image_id, v.photo_index, v.metadata.model_dump()).model_dump(
                mode="json"
            )
            for v in views
        ]
        payload["source_views"] = public
        if payload["envelope"] is not None:
            payload["envelope"]["input_views"] = public
        return payload

    async def _draft_view(self, conn: asyncpg.Connection, row: Any) -> DraftView:
        findings = _decoded(row["findings"])
        if row["finalized_receipt_id"] is None:
            # Older drafts stored completeness findings for fields now optional.
            findings = [
                finding for finding in findings
                if finding["code"] != "missing_mandatory_field" or not re.fullmatch(
                    r"discount_total|line_items\.[0-9]+\.(quantity|unit_price)",
                    finding["field_path"],
                )
            ]
        return DraftView(
            image_id=row["image_id"],
            revision=row["revision"],
            document=ExtractionDraft.model_validate(_decoded(row["document"])),
            amount_policy=_decoded(row["amount_policy"]),
            findings=findings,
            field_sources=_decoded(row["field_sources"]),
            overlap_decisions=_decoded(row["overlap_decisions"]),
            finalized_receipt_id=row["finalized_receipt_id"],
            source_views=await self._source_views(conn, row["image_id"]),
        )

    async def get_draft(self, image_id: UUID, *, scope: AccessScope) -> DraftView | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT d.* FROM receipt_drafts d JOIN images i ON i.id=d.image_id
                WHERE i.id=$1 AND (i.user_id=$2 OR $3)
                AND i.observation_checkpoint->>'eligible_generation' IS DISTINCT FROM i.generation::text""",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
            return await self._draft_view(conn, row) if row else None

    async def patch_draft(
        self,
        image_id: UUID,
        patch: DraftPatch,
        *,
        scope: AccessScope,
    ) -> DraftView | None:
        async with self.pool.acquire() as conn, conn.transaction():
            # Same lock order as finish/retry/finalize prevents worker/human races.
            image = await conn.fetchrow(
                """SELECT * FROM images WHERE id=$1
                AND (user_id=$2 OR $3) FOR UPDATE""",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
            if image is None:
                return None
            row = await conn.fetchrow(
                "SELECT * FROM receipt_drafts WHERE image_id=$1 FOR UPDATE", image_id
            )
            if row is None:
                return None
            if row["revision"] != patch.expected_revision:
                raise ValueError("draft_revision_conflict")
            if row["finalized_receipt_id"] is not None or image["receipt_id"] is not None:
                raise ValueError("draft_already_finalized")
            if image["status"] == "processing" or _eligible_observations(image) is not None:
                raise ValueError("draft_processing_conflict")
            metadata_fields = {"amount_policy", "field_sources", "overlap_decisions"}
            if set(patch.changes) - (set(ExtractionDraft.model_fields) | metadata_fields):
                raise ValueError("draft_unknown_field")
            assert_metadata_only(patch.changes)
            document = ExtractionDraft.model_validate(
                {**_decoded(row["document"]), **{key: value for key, value in patch.changes.items()
                                              if key not in metadata_fields}}
            )
            policy = AmountPolicy.model_validate(patch.changes.get("amount_policy", _decoded(row["amount_policy"])))
            sources = TypeAdapter(dict[str, list[SourceRef]]).validate_python(
                patch.changes.get("field_sources", _decoded(row["field_sources"]))
            )
            overlaps = TypeAdapter(list[dict[str, Any]]).validate_python(
                patch.changes.get("overlap_decisions", _decoded(row["overlap_decisions"]))
            )
            parts = await self._parts(conn, image_id)
            # Validate flexible nested evidence before typed projection can drop extra keys.
            await self._validate_metadata_refs(conn, image_id, parts, patch.changes)
            metadata = {"field_sources": {key: [ref.model_dump(mode="json") for ref in refs]
                                          for key, refs in sources.items()},
                        "overlap_decisions": overlaps}
            await self._validate_metadata_refs(conn, image_id, parts, metadata)
            originals = await self._receipts._review_originals(conn, image_id)
            review = evaluate_receipt(document, policy, sources, overlaps, originals,
                                      retained_findings=persistent_findings(
                                          (await self._draft_view(conn, row)).findings))
            row = await conn.fetchrow(
                """UPDATE receipt_drafts SET document=$2::jsonb,amount_policy=$3::jsonb,
                field_sources=$4::jsonb,overlap_decisions=$5::jsonb,findings=$6::jsonb,
                revision=revision+1,human_edited=TRUE WHERE image_id=$1 RETURNING *""",
                image_id,
                document.model_dump_json(),
                policy.model_dump_json(),
                _json(metadata["field_sources"]),
                _json(overlaps),
                _json([finding.model_dump(mode="json") for finding in review.findings]),
            )
            # Editing a draft retained during a pending explicit retry prevents later claims.
            await conn.execute(
                """UPDATE images SET status='needs_review',retry_at=NULL,
                draft_available=TRUE WHERE id=$1""",
                image_id,
            )
            return await self._draft_view(conn, row)

    async def finalize_draft(
        self,
        image_id: UUID,
        request: DraftFinalize,
        *,
        scope: AccessScope,
        originals: list[OriginalAvailability],
    ) -> ReceiptRow | None:
        async with self.pool.acquire() as conn, conn.transaction():
            image = await conn.fetchrow(
                """SELECT * FROM images WHERE id=$1
                AND (user_id=$2 OR $3) FOR UPDATE""",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
            if image is None:
                return None
            row = await conn.fetchrow(
                "SELECT * FROM receipt_drafts WHERE image_id=$1 FOR UPDATE", image_id
            )
            if row is None:
                return None
            if row["revision"] != request.expected_revision:
                raise ValueError("draft_revision_conflict")
            if row["finalized_receipt_id"] is not None:
                saved = await conn.fetchrow(
                    "SELECT * FROM receipts WHERE id=$1", row["finalized_receipt_id"]
                )
                return self._receipts._receipt_row_from_record(saved)
            if (
                image["status"] == "processing" or image["receipt_id"] is not None
                or _eligible_observations(image) is not None
            ):
                raise ValueError("draft_state_conflict")
            view = await self._draft_view(conn, row)
            receipt = receipt_from_draft(view.document)
            parts = await self._parts(conn, image_id)
            if (
                not parts
                or len(originals) != len(parts)
                or {o.part_id for o in originals} != {p.part_id for p in parts}
                or not all(o.available and o.sha256_matches for o in originals)
            ):
                raise ValueError("original_integrity_failed")
            originals = await self._receipts._review_originals(conn, image_id)
            if not originals or not all(o.available and o.sha256_matches for o in originals):
                raise ValueError("original_integrity_failed")
            sources = TypeAdapter(dict[str, list[SourceRef]]).validate_python(_decoded(row["field_sources"]))
            overlaps = _decoded(row["overlap_decisions"])
            review = evaluate_receipt(view.document, view.amount_policy, sources, overlaps, originals,
                                      retained_findings=persistent_findings(view.findings))
            if not review.can_finalize:
                raise ValueError("draft_integrity_failed")
            saved_receipt = await self._receipts.insert_receipt(
                conn,
                receipt,
                image_id=image_id,
                user_id=image["user_id"],
                status="unverified",
                verified=False,
            )
            saved = await conn.fetchrow(
                """UPDATE receipts SET extraction_id=$2,amount_policy=$3::jsonb,
                findings=$4::jsonb,field_sources=$5::jsonb,checks_version=$6,overlap_decisions=$7::jsonb WHERE id=$1 RETURNING *""",
                saved_receipt.id,
                row["extraction_id"],
                row["amount_policy"],
                _json([finding.model_dump(mode="json") for finding in review.findings]),
                row["field_sources"],
                review.checks_version,
                _json(overlaps),
            )
            result = self._receipts._receipt_row_from_record(saved)
            await conn.execute(
                """INSERT INTO receipt_review_revisions(receipt_id,operation_revision,
                review_revision,event_kind,actor_user_id,snapshot) VALUES($1,1,1,'draft_finalized',$2,$3::jsonb)""",
                result.id,
                scope.user_id,
                _json(
                    {
                        "receipt": result.model_dump(mode="json"),
                        "draft_revision": row["revision"],
                        "document": receipt.model_dump(mode="json"),
                        "request_key": request.request_key,
                        "collection_ids": [str(link["collection_id"]) for link in await conn.fetch(
                            "SELECT collection_id FROM receipt_collections WHERE receipt_id=$1 ORDER BY collection_id",
                            result.id,
                        )],
                    }
                ),
            )
            await conn.execute(
                "UPDATE receipt_drafts SET finalized_receipt_id=$2 WHERE image_id=$1",
                image_id,
                result.id,
            )
            await conn.execute(
                """UPDATE images SET receipt_id=$2,status='analyzed',draft_available=FALSE,
                retry_at=NULL,result_code=NULL,error=NULL,analyzed_at=clock_timestamp() WHERE id=$1""",
                image_id,
                result.id,
            )
            return result

    async def unreferenced_views(
        self,
        image_id: UUID,
        views: list[PersistedWorkingView],
        *,
        deadline: float | None = None,
    ) -> list[PersistedWorkingView]:
        """Wait for any in-flight commit before authorizing caller-owned orphan cleanup."""
        async with connection(self.pool, deadline=deadline, transaction=True) as conn:
            await conn.fetchrow("SELECT id FROM images WHERE id=$1 FOR UPDATE", image_id)
            saved = await conn.fetch(
                "SELECT view_id FROM extraction_working_views WHERE view_id=ANY($1::uuid[])",
                [v.metadata.view_id for v in views],
            )
            ids = {r["view_id"] for r in saved}
            return [view for view in views if view.metadata.view_id not in ids]
