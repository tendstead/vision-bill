"""Real PostgreSQL coverage for fenced observation checkpoint persistence."""

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import uuid4

import asyncpg
import pytest
from test_observation_checkpoint import observation_payload

from vision_bill.config import Settings
from vision_bill.model import workflow
from vision_bill.model.db.image import ImagePart
from vision_bill.provider.db import workflow_db as provider
from vision_bill.service import execution_policy

from . import test_workflow_db_e2e as shared

postgres, db, workflow_db = shared.postgres, shared.db, shared.workflow_db
pytestmark = pytest.mark.e2e


async def claimed_checkpoint(w, scope, *, ocr=False, max_attempts=3):
    payload = observation_payload(ocr=ocr)
    snapshot = {
        "execution_policy_version": 4,
        "inference_identity": payload["execution_identity"]["inference_identity"],
        "managed_binding_sha256": None,
        "allow_ocr_fallback": False,
        "ocr": {"enabled": ocr},
        "temperature": 0.0,
        "max_output_tokens": 8192,
    }
    parts = payload["source_parts"]
    image = (
        await w.enqueue_submission(
            workflow.SubmissionCreate(
                id=uuid4(),
                parts=[ImagePart(image_path=part["image_path"]) for part in parts],
                source_manifest=[
                    {key: part[key] for key in ("photo_index", "source_sha256")} for part in parts
                ],
                requested_options={"model_id": "test"},
                effective_options=snapshot,
                model_id="test",
                request_fingerprint="checkpoint-fixture",
            ),
            scope=scope,
            idempotency_key=None,
        )
    ).submission
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE images SET max_attempts=$2 WHERE id=$1", image.id, max_attempts)
    claim = await w.claim_next(image_id=image.id)
    assert claim is not None
    actual_parts = await w.get_source_parts(image.id, scope=scope)
    payload = observation_payload(parts=[p.model_dump() for p in actual_parts], ocr=ocr)
    payload["execution_identity"]["semantic_policy_sha256"] = (
        execution_policy.semantic_policy_sha256(snapshot)
    )
    return claim, workflow.ObservationCheckpoint.model_validate(payload)


@pytest.mark.parametrize("ocr", [False, True], ids=["direct", "ocr"])
async def test_save_is_fenced_immutable_and_public_projection_is_safe(workflow_db, ocr):
    w, scope = workflow_db
    claim, checkpoint = await claimed_checkpoint(w, scope, ocr=ocr)
    if not ocr:
        view = checkpoint.persisted_views[0]
        view.metadata.transforms.append(
            {
                "source": {
                    "part_id": str(view.part_id),
                    "photo_index": view.photo_index,
                    "source_sha256": view.source_sha256,
                    "view_id": str(view.metadata.view_id),
                    "coordinate_space": "working",
                },
            }
        )

    for invalid_claim in (
        claim.model_copy(update={"claim_token": uuid4()}),
        claim.model_copy(update={"generation": claim.generation + 1}),
        claim.model_copy(
            update={"purpose": "completion", "checkpoint_id": checkpoint.checkpoint_id}
        ),
    ):
        assert not await w.save_observation_checkpoint(invalid_claim, checkpoint)
        assert await w.get_observation_checkpoint(invalid_claim) is None
    assert await w.save_observation_checkpoint(claim, checkpoint)
    assert await w.save_observation_checkpoint(claim, checkpoint)
    assert await w.get_observation_checkpoint(claim) == checkpoint
    with pytest.raises(ValueError, match="immutable observation checkpoint"):
        await w.save_observation_checkpoint(
            claim, checkpoint.model_copy(update={"checkpoint_id": uuid4()})
        )

    status = await w.get_submission(claim.image.id, scope=scope)
    assert status is not None and status.observations_available
    public = status.model_dump()
    assert {
        "observation_checkpoint",
        "claim_purpose",
        "claim_token",
        "execution_identity",
    }.isdisjoint(public)
    assert str(checkpoint.checkpoint_id) not in status.model_dump_json()
    assert checkpoint.producer_attempt_number == claim.attempt_number


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("source", "source manifest"),
        ("foreign_ref", "bind to a real source"),
        ("unlisted_view", "checkpoint working view"),
        ("view_artifact_path", "artifact_path"),
        ("backend_identity", "inference_identity"),
        ("other_endpoint", "execution policy"),
        ("view_foreign_ref", "bind to a real source"),
        ("view_unlisted_view", "checkpoint working view"),
        ("snapshot", "execution policy"),
        ("missing_observation", "draft_incomplete"),
        ("payload", "belongs in saved artifacts"),
    ],
)
async def test_rejected_checkpoint_keeps_no_artifacts_or_pointer(workflow_db, corruption, message):
    w, scope = workflow_db
    claim, checkpoint = await claimed_checkpoint(w, scope)
    if corruption == "source":
        checkpoint.source_parts[1].image_path = "/private/another-submission.png"
    if corruption == "foreign_ref":
        checkpoint.envelope.overlap_decisions.append(
            {
                "evidence": {
                    "part_id": str(uuid4()),
                    "photo_index": 0,
                    "source_sha256": "a" * 64,
                }
            }
        )
    if corruption == "view_artifact_path":
        checkpoint.persisted_views[0].artifact_path = ""
    if corruption in ("backend_identity", "other_endpoint"):
        value = "invalid-endpoint-identity" if corruption == "backend_identity" else "ollama:" + "c" * 64
        identity = checkpoint.execution_identity.model_copy(update={"inference_identity": value})
        checkpoint = checkpoint.model_copy(update={"execution_identity": identity})
    if corruption == "view_foreign_ref":
        checkpoint.persisted_views[0].metadata.transforms.append(
            {"source": {"part_id": str(uuid4()), "photo_index": 0, "source_sha256": "a" * 64}}
        )
    if corruption in {"unlisted_view", "view_unlisted_view"}:
        # A historical view from this image is outside this checkpoint's manifest.
        view = checkpoint.persisted_views[0]
        meta = view.metadata.model_copy(update={"view_id": uuid4()})
        async with w.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO extraction_working_views(view_id,image_id,part_id,artifact_path,metadata)
                VALUES($1,$2,$3,$4,$5::jsonb)""",
                meta.view_id,
                claim.image.id,
                view.part_id,
                "/private/previous.png",
                meta.model_dump_json(),
            )
        references = (
            checkpoint.persisted_views[0].metadata.transforms
            if corruption == "view_unlisted_view"
            else checkpoint.envelope.overlap_decisions
        )
        references.append({"view_id": str(meta.view_id)})
    if corruption == "snapshot":
        identity = checkpoint.execution_identity.model_copy(
            update={"semantic_policy_sha256": "0" * 64}
        )
        checkpoint = checkpoint.model_copy(update={"execution_identity": identity})
    if corruption == "missing_observation":
        checkpoint.envelope.draft.currency = None
    if corruption == "payload":
        checkpoint.envelope.telemetry.calls.append({"working": {"png_base64": "Ynl0ZXM="}})

    before = await w.get_submission(claim.image.id, scope=scope)
    async with w.pool.acquire() as conn:
        row_query = "SELECT * FROM images WHERE id=$1"
        claim_before = await conn.fetchrow(row_query, claim.image.id)
        with pytest.raises(ValueError, match=message):
            await w.save_observation_checkpoint(claim, checkpoint)
        assert await conn.fetchrow(row_query, claim.image.id) == claim_before
        assert not await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM receipt_extractions WHERE image_id=$1)", claim.image.id
        )

    assert await w.get_submission(claim.image.id, scope=scope) == before
    assert await w.get_observation_checkpoint(claim) is None
    assert await w.unreferenced_views(claim.image.id, checkpoint.persisted_views) == (
        checkpoint.persisted_views
    )


@pytest.mark.parametrize("blocker", ["image", "view"])
async def test_waiting_save_cannot_commit_after_lease_expiry(workflow_db, blocker):
    w, scope = workflow_db
    claim, checkpoint = await claimed_checkpoint(w, scope)

    async with w.pool.acquire() as conn:
        if blocker == "view":
            await conn.execute(
                """CREATE FUNCTION wait_to_save_view() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN PERFORM pg_advisory_xact_lock(9); RETURN NEW; END $$"""
            )
            await conn.execute(
                """CREATE TRIGGER wait_to_save_view BEFORE INSERT ON extraction_working_views
                FOR EACH ROW EXECUTE FUNCTION wait_to_save_view()"""
            )
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp()+interval '1 second' WHERE id=$1",
            claim.image.id,
        )
        async with conn.transaction():
            if blocker == "image":
                await conn.fetchrow("SELECT id FROM images WHERE id=$1 FOR UPDATE", claim.image.id)
                query = "SELECT * FROM images WHERE id=%"
            else:
                # The artifact insertion now waits after acquiring the image/receipt/draft locks.
                await conn.execute("SELECT pg_advisory_xact_lock(9)")
                query = "INSERT INTO extraction_working_views(view_id,image_id,part_id,%"
            save = asyncio.create_task(w.save_observation_checkpoint(claim, checkpoint))
            async with w.pool.acquire() as observer, asyncio.timeout(2):
                while not await observer.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE $1)",
                    query,
                ):
                    if save.done():
                        await save
                        pytest.fail("save did not wait for the controlled lock")
                    await asyncio.sleep(0.01)
            await conn.execute("SELECT pg_sleep(1)")
        assert not await save
        assert not await conn.fetchval(
            "SELECT observation_checkpoint IS NOT NULL FROM images WHERE id=$1",
            claim.image.id,
        )
    unreferenced = await w.unreferenced_views(claim.image.id, checkpoint.persisted_views)
    assert unreferenced == checkpoint.persisted_views


async def test_lost_commit_ack_keeps_files_referenced(workflow_db, monkeypatch):
    w, scope = workflow_db
    claim, checkpoint = await claimed_checkpoint(w, scope)
    connection = provider.connection

    @asynccontextmanager
    async def lost_ack(*args, **kwargs):
        async with connection(*args, **kwargs) as conn:
            yield conn
        raise ConnectionError("commit acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(provider, "connection", lost_ack)
        with pytest.raises(ConnectionError, match="commit acknowledgement lost"):
            await w.save_observation_checkpoint(claim, checkpoint)

    assert await w.unreferenced_views(claim.image.id, checkpoint.persisted_views) == []
    assert await w.get_observation_checkpoint(claim) == checkpoint


@pytest.mark.parametrize("ending", ["expired", "error"])
async def test_last_inference_attempt_keeps_completion_wait_and_retry_retires_checkpoint(
    workflow_db,
    db,
    ending,
):
    w, scope = workflow_db
    claim, checkpoint = await claimed_checkpoint(w, scope, max_attempts=1)
    image_id = claim.image.id
    assert await w.save_observation_checkpoint(claim, checkpoint)
    async with w.pool.acquire() as conn:
        if ending == "expired":
            await conn.execute(
                "UPDATE images SET lease_expires_at=clock_timestamp() WHERE id=$1", image_id
            )
            assert await w.reap_expired_claims() == 1
            assert await w.reap_expired_claims() == 0
        else:
            result = await w.finish_attempt(
                claim,
                workflow.AttemptOutcome(kind="retry", result_code="analysis_unavailable"),
            )
            assert not result.stale

        held = await w.get_submission(image_id, scope=scope)
        assert held.status == "pending" and held.attempt_count == 1
        assert held.wait_reason == "observation_recovery_required" and held.observations_available
        assert not await w.due_candidates()
        assert await w.claim_next(image_id=image_id) is None
        attempt = await conn.fetchrow(
            "SELECT outcome_kind,envelope FROM receipt_extractions WHERE image_id=$1",
            image_id,
        )
        assert attempt["outcome_kind"] == "retry"
        assert json.loads(attempt["envelope"])["receipt"]["total"] == "90071992547.91"

        await conn.execute(
            "UPDATE images SET wait_reason='execution_policy_changed' WHERE id=$1", image_id
        )
        request = workflow.RetryRequest(
            expected_generation=1, request_key="adopt", policy="current"
        )
        snapshot = execution_policy.current_snapshot(
            Settings(auth={"secret_key": "fixture-only-secret"}, pg=db), inference_identity="ollama:" + "d" * 64
        )
        retried = await w.retry_submission(
            image_id, request, scope=scope, replacement_snapshot=snapshot
        )
        assert retried.generation == 2 and retried.attempt_count == 0
        assert not retried.observations_available
        assert (
            await w.retry_submission(image_id, request, scope=scope, replacement_snapshot=snapshot)
            == retried
        )
        assert await conn.fetchval(
            """SELECT EXISTS(SELECT 1 FROM images i, jsonb_array_elements(i.legacy_events) e
            WHERE i.id=$1 AND e->>'event'='observation_checkpoint_retired'
            AND e->'checkpoint'->>'checkpoint_id'=$2)""",
            image_id,
            str(checkpoint.checkpoint_id),
        )
        successor = await w.claim_next(image_id=image_id)
        assert await w.get_observation_checkpoint(successor) is None
        assert await w.unreferenced_views(image_id, checkpoint.persisted_views) == []


async def test_expired_completion_has_no_inference_abandonment_or_budget_change(workflow_db):
    w, scope = workflow_db
    claim, checkpoint = await claimed_checkpoint(w, scope, max_attempts=1)
    image_id = claim.image.id
    assert await w.save_observation_checkpoint(claim, checkpoint)
    async with w.pool.acquire() as conn:
        # Task 11 provides completion admission. Seed that durable lease shape here.
        await conn.execute(
            "UPDATE images SET claim_purpose='completion',claim_token=$2 WHERE id=$1",
            image_id,
            uuid4(),
        )
        count = await conn.fetchval(
            "SELECT count(*) FROM receipt_extractions WHERE image_id=$1", image_id
        )
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp() WHERE id=$1", image_id
        )
        assert await w.reap_expired_claims() == 1
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM receipt_extractions WHERE image_id=$1", image_id
            )
            == count
        )
        assert await conn.fetchval("SELECT claim_purpose FROM images WHERE id=$1", image_id) is None
    status = await w.get_submission(image_id, scope=scope)
    assert status.status == "pending" and status.attempt_count == 1
    assert status.wait_reason == "observation_recovery_required" and status.observations_available


async def test_migration_preserves_0019_receipt_attempt_and_human_draft_rows(db):
    shared.migrate(db, "0019")
    conn = await asyncpg.connect(db.pg_dsn)
    try:
        active = await conn.fetchval(
            """INSERT INTO images(status,attempt_count,claim_token,lease_expires_at)
            VALUES('processing',2,$1, '2026-01-01') RETURNING id""",
            uuid4(),
        )
        completed = await conn.fetchval(
            "INSERT INTO images(status) VALUES('analyzed') RETURNING id"
        )
        receipt = await conn.fetchval(
            """INSERT INTO receipts(merchant_name,date,total,image_id)
            VALUES('Human Receipt','2026-01-01',90071992547.91,$1) RETURNING id""",
            completed,
        )
        await conn.execute("UPDATE images SET receipt_id=$2 WHERE id=$1", completed, receipt)
        await conn.execute("INSERT INTO images(status,error) VALUES('failed','old failure')")
        edited = await conn.fetchval(
            "INSERT INTO images(status,draft_available) VALUES('needs_review',true) RETURNING id"
        )
        extraction = await conn.fetchval(
            """INSERT INTO receipt_extractions(image_id,generation,attempt_number,outcome_kind)
            VALUES($1,1,1,'partial') RETURNING id""",
            edited,
        )
        await conn.execute(
            """INSERT INTO receipt_drafts(image_id,extraction_id,generation,revision,document,
            amount_policy,findings,field_sources,human_edited)
            VALUES($1,$2,1,4,'{"merchant_name":"human correction"}','{}','[]','{}',true)""",
            edited,
            extraction,
        )
        images_before = await conn.fetch("SELECT to_jsonb(i) FROM images i ORDER BY id")
        receipts_before = await conn.fetch("SELECT to_jsonb(r) FROM receipts r ORDER BY id")
        drafts_before = await conn.fetch(
            "SELECT to_jsonb(d) FROM receipt_drafts d ORDER BY image_id"
        )
        extractions_before = await conn.fetch(
            "SELECT to_jsonb(e) FROM receipt_extractions e ORDER BY id"
        )

        shared.migrate(db)
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0021"
        assert (
            await conn.fetch(
                "SELECT to_jsonb(i)-'claim_purpose'-'observation_checkpoint' FROM images i ORDER BY id"
            )
            == images_before
        )
        assert await conn.fetch("SELECT to_jsonb(r) FROM receipts r ORDER BY id") == receipts_before
        assert (
            await conn.fetch("SELECT to_jsonb(d) FROM receipt_drafts d ORDER BY image_id")
            == drafts_before
        )
        assert (
            await conn.fetch("SELECT to_jsonb(e) FROM receipt_extractions e ORDER BY id")
            == extractions_before
        )
        purposes = await conn.fetch(
            "SELECT id,claim_purpose FROM images WHERE claim_purpose IS NOT NULL"
        )
        assert [tuple(row) for row in purposes] == [(active, "inference")]
        assert not await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM images WHERE observation_checkpoint IS NOT NULL)"
        )
        shared.migrate(db, "0019", "downgrade")
        shared.migrate(db)
        await conn.execute("UPDATE images SET claim_purpose='completion' WHERE id=$1", active)
        refused = shared.migrate(db, "0019", "downgrade", check=False)
        assert refused.returncode and "observation checkpoint downgrade refused" in refused.stderr
        await conn.execute(
            """UPDATE images SET claim_purpose='inference',
            observation_checkpoint='{"version":1}'::jsonb WHERE id=$1""",
            active,
        )
        refused = shared.migrate(db, "0019", "downgrade", check=False)
        assert refused.returncode and "observation checkpoint downgrade refused" in refused.stderr
    finally:
        await conn.close()
