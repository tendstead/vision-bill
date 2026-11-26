"""Real PostgreSQL 18: migrations, queue locks, fencing and transaction rollback.

Each test owns a new database inside one isolated, disposable Docker container.
No operator connection settings are used. Run explicitly with -m e2e.
"""

import asyncio
import importlib
import json
import os
import secrets
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import asyncpg
import pytest

from vision_bill.config import PGSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.extraction import InferenceEnvelope, InferenceTelemetry, PipelineProvenance
from vision_bill.model.receipt import Receipt
from vision_bill.model.review import ReviewEvaluation
from vision_bill.model.workflow import AccessScope, AttemptOutcome, RetryRequest, SubmissionCreate

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]

ENDPOINT = "ollama:" + "d" * 64


@pytest.fixture(scope="module")
def postgres():
    name = "vision-bill-task7-" + secrets.token_hex(6)
    password = secrets.token_hex(24)
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "-e",
            "POSTGRES_USER=workflow_test",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "-e",
            "POSTGRES_DB=workflow_admin",
            "-p",
            "127.0.0.1::5432",
            "postgres:18",
        ],
        check=True,
        capture_output=True,
    )
    try:
        port = int(
            subprocess.check_output(["docker", "port", name, "5432/tcp"], text=True)
            .strip()
            .rsplit(":", 1)[1]
        )
        for _ in range(100):
            if (
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        name,
                        "pg_isready",
                        "-h",
                        "127.0.0.1",
                        "-U",
                        "workflow_test",
                        "-d",
                        "workflow_admin",
                    ],
                    check=False,
                    capture_output=True,
                ).returncode
                == 0
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("isolated PostgreSQL not ready")
        yield PGSettings(
            user="workflow_test",
            password=password,
            host="127.0.0.1",
            port=port,
            db="workflow_admin",
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", "--volumes", name], check=True, capture_output=True
        )


def migrate(settings, revision="head", direction="upgrade", check=True):
    env = {
        **os.environ,
        "PG__USER": settings.user,
        "PG__PASSWORD": settings.password,
        "PG__HOST": settings.host,
        "PG__PORT": str(settings.port),
        "PG__DB": settings.db,
        "AUTH__SECRET_KEY": secrets.token_hex(32),
    }
    with TemporaryDirectory(prefix="vision-bill-migration-") as directory:
        env["VISION_BILL_CONFIG_PATH"] = str(Path(directory) / "config.yaml")
        return subprocess.run(
            [sys.executable, "-m", "alembic", direction, revision],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=check,
        )


@pytest.fixture
async def db(postgres):
    admin = await asyncpg.connect(postgres.pg_dsn)
    dbname = "case_" + secrets.token_hex(8)
    await admin.execute(f'CREATE DATABASE "{dbname}"')
    settings = postgres.model_copy(update={"db": dbname})
    try:
        yield settings
    finally:
        await admin.execute(f'DROP DATABASE "{dbname}" WITH (FORCE)')
        await admin.close()


@pytest.fixture
async def workflow_db(db, settings):
    migrate(db)
    module = importlib.import_module("vision_bill.provider.db.workflow_db")
    from vision_bill.service.image_service import ImageService
    images = ImageService(settings.images)
    provider = module.WorkflowDB(
        db, original_checker=images.check_originals, working_view_checker=images.read_working_image,
    )
    await provider.init_db()
    owner = uuid4()
    async with provider.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users(id, username, hashed_password) VALUES($1,$2,'test')",
            owner,
            str(owner),
        )
    try:
        yield provider, AccessScope(user_id=owner)
    finally:
        await provider.destroy_db()


def submission():
    return SubmissionCreate(
        id=uuid4(),
        parts=[ImagePart(image_path="/private/original.png")],
        source_manifest=[{"photo_index": 0, "source_sha256": "a" * 64}],
        requested_options={},
        effective_options={},
        request_fingerprint="same-request",
    )


async def enqueue(w, scope, *, effective_options=None):
    request = submission()
    if effective_options is not None:
        request.effective_options = effective_options
    return (await w.enqueue_submission(request, scope=scope, idempotency_key=None)).submission


def complete(*, total="90071992547.91"):
    receipt = Receipt(
        confidence=90,
        merchant_name="Merchant",
        date="2026-01-01",
        currency="EUR",
        line_items=[
            {"description": "item", "quantity": 1, "unit_price": total, "total_price": total}
        ],
        subtotal=total,
        total=total,
        taxes=[{"name": "VAT", "amount": "0.01"}],
    )
    return AttemptOutcome(
        kind="complete",
        envelope=InferenceEnvelope(
            receipt=receipt,
            provenance=PipelineProvenance(
                requested_pipeline="direct_vision",
                used_pipeline="direct_vision",
                source_image_count=1,
                ordered_source_sha256=["a" * 64],
                model_id="test",
            ),
            telemetry=InferenceTelemetry(elapsed_ms=1, max_output_tokens=8192),
        ),
        review=ReviewEvaluation(can_finalize=True, can_auto_verify=False),
    )


async def test_bootstrap_and_empty_downgrade(db):
    migrate(db)
    conn = await asyncpg.connect(db.pg_dsn)
    assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0025"
    await conn.close()
    migrate(db, "0010", "downgrade")
    migrate(db)


async def test_product_tag_upgrade_preserves_identity_and_matches_translations(db):
    from vision_bill.provider.db.receipt_db import ReceiptDB

    migrate(db, "0023")
    catalog = json.loads((ROOT / "src/vision_bill/model/product_tags.json").read_text())
    assert all(set(labels) == {"en", "de"} and all(labels.values()) for labels in catalog.values())
    conn = await asyncpg.connect(db.pg_dsn)
    provider = ReceiptDB(db)
    try:
        await conn.execute("INSERT INTO tags(name) VALUES ('milk'), ('Berries'), ('wochenmarkt')")
        historical = dict(await conn.fetch("SELECT name, id FROM tags"))
        receipt = await conn.fetchval(
            "INSERT INTO receipts(merchant_name,date,total) VALUES('shop','2026-01-01',1) RETURNING id"
        )
        item = await conn.fetchval(
            "INSERT INTO line_items(receipt_id,description,original_description,total_price,tags) "
            "VALUES($1,'Milch','Milch',1,ARRAY['milk','Berries','wochenmarkt']) RETURNING id", receipt
        )
        for attempt in range(2):
            if attempt:
                migrate(db, "0023", "downgrade")
            migrate(db)
            actual = dict(await conn.fetch("SELECT name, id FROM tags"))
            assert set(actual) == (set(catalog) - {"berries"}) | {"Berries", "wochenmarkt"}
            assert historical.items() <= actual.items()
            assert await conn.fetchval("SELECT tags FROM line_items WHERE id=$1", item) == [
                "milk", "Berries", "wochenmarkt"
            ]
            historical = actual
        await provider.init_db()
        assert set(await provider.list_tags()) == set(historical)
        assert await provider.create_tag("berries") == ("Berries", False)
        assert set(await provider.list_tags()) == set(historical)
    finally:
        await provider.destroy_db()
        await conn.close()


async def test_retry_policy_migration_preserves_legacy_identity_and_refuses_lossy_backout(db):
    import json
    migrate(db, "0018")
    conn = await asyncpg.connect(db.pg_dsn)
    try:
        image = await conn.fetchval("INSERT INTO images(status) VALUES('failed') RETURNING id")
        result = json.dumps({"id": str(image), "generation": 2})
        await conn.execute("INSERT INTO image_retry_requests(image_id,request_key,expected_generation,result) VALUES($1,'legacy',1,$2::jsonb)", image, result)
        migrate(db, "0019")
        assert json.loads(await conn.fetchval("SELECT request_options FROM image_retry_requests")) == {}
        assert json.loads(await conn.fetchval("SELECT result FROM image_retry_requests")) == json.loads(result)
        migrate(db, "0018", "downgrade")
        migrate(db, "0019")
        await conn.execute("UPDATE image_retry_requests SET request_options=$1::jsonb", json.dumps({"policy": "current"}))
        refused = migrate(db, "0018", "downgrade", check=False)
        assert refused.returncode != 0 and "adoption downgrade refused" in refused.stderr
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0019"
    finally:
        await conn.close()


async def test_due_only_skip_locked_concurrent_claims(workflow_db):
    w, scope = workflow_db
    first, second, future = [await enqueue(w, scope) for _ in range(3)]
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET retry_at=now()+interval '1 hour' WHERE id=$1", future.id
        )
        async with conn.transaction():
            await conn.fetchrow("SELECT id FROM images WHERE id=$1 FOR UPDATE", first.id)
            claimed = await asyncio.wait_for(w.claim_next(), timeout=2)
            assert claimed.image.id == second.id
    claimed = await w.claim_next()
    assert claimed.image.id == first.id
    assert await w.claim_next() is None


async def test_concurrent_claims_unique(workflow_db):
    w, scope = workflow_db
    ids = {(await enqueue(w, scope)).id for _ in range(5)}
    claims = await asyncio.gather(*(w.claim_next() for _ in range(8)))
    assert {c.image.id for c in claims if c is not None} == ids
    assert sum(c is not None for c in claims) == 5


async def test_expiry_tokens_generation_and_three_attempt_limit(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim1 = await w.claim_next()
    assert (
        await w.finish_attempt(
            claim1.model_copy(update={"claim_token": uuid4()}), AttemptOutcome(kind="retry")
        )
    ).stale
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET lease_expires_at=now()-interval '1 second' WHERE id=$1", image.id
        )
    assert (await w.finish_attempt(claim1, complete())).stale
    assert await w.reap_expired_claims() == 1
    saved = await w.get_submission(image.id, scope=scope)
    assert saved.status == "pending" and saved.attempt_count == 1 and saved.retry_at is not None
    assert await w.claim_next() is None
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE images SET retry_at=clock_timestamp() WHERE id=$1", image.id)
    claim2 = await w.claim_next()
    assert claim2.attempt_number == 2
    assert (await w.finish_attempt(claim1, complete())).stale
    await w.finish_attempt(claim2, AttemptOutcome(kind="retry", retry_delay_seconds=0))
    claim3 = await w.claim_next()
    assert claim3.attempt_number == 3
    result = await w.finish_attempt(claim3, AttemptOutcome(kind="retry", retry_delay_seconds=0))
    assert result.submission.status == "failed" and result.submission.attempt_count == 3
    assert await w.claim_next() is None
    retried = await w.retry_submission(
        image.id, RetryRequest(expected_generation=1, request_key="retry"), scope=scope
    )
    assert retried.generation == 2
    assert (await w.finish_attempt(claim3, complete())).stale


@pytest.mark.parametrize("code", ["formatter_output_limit", "analysis_incomplete"])
@pytest.mark.parametrize("policy,expected_limit", [("preserve", 8192), ("current", 32768)])
async def test_failed_formatter_retry_uses_selected_generation_budget(
    workflow_db, settings, code, policy, expected_limit
):
    from vision_bill.service.execution_policy import current_snapshot

    w, scope = workflow_db
    current = current_snapshot(settings, inference_identity=ENDPOINT) | {"max_output_tokens": 32768}
    frozen = current | {"max_output_tokens": 8192}
    create = submission().model_copy(update={"effective_options": frozen})
    image = (await w.enqueue_submission(create, scope=scope, idempotency_key=None)).submission
    # The historical worker exhausted its three ordinary attempts; the new worker
    # can stop on the first explicit output-limit finding.
    if code == "analysis_incomplete":
        for _ in range(2):
            await w.finish_attempt(await w.claim_next(), AttemptOutcome(
                kind="retry", result_code=code, retry_delay_seconds=0,
            ))
    kind = "terminal" if code == "formatter_output_limit" else "retry"
    finished = await w.finish_attempt(await w.claim_next(), AttemptOutcome(kind=kind, result_code=code))
    assert finished.submission.status == "failed" and finished.submission.result_code == code
    assert await w.claim_next() is None
    retried = await w.retry_submission(
        image.id, RetryRequest(expected_generation=1, request_key="next", policy=policy),
        scope=scope, replacement_snapshot=current,
    )
    assert retried.generation == 2 and retried.attempt_count == 0
    claim = await w.claim_next()
    assert claim.attempt_number == 1
    assert claim.image.effective_options["max_output_tokens"] == expected_limit


async def test_upload_idempotency_is_owner_scoped_and_conflicts(workflow_db):
    w, scope = workflow_db
    request = submission()
    results = await asyncio.gather(
        *(w.enqueue_submission(request, scope=scope, idempotency_key="upload") for _ in range(3))
    )
    assert sum(r.created for r in results) == 1
    assert len({r.submission.id for r in results}) == 1
    altered = request.model_copy(update={"request_fingerprint": "changed"})
    with pytest.raises(ValueError, match="conflict"):
        await w.enqueue_submission(altered, scope=scope, idempotency_key="upload")
    async with w.pool.acquire() as conn:
        other = await conn.fetchval(
            "INSERT INTO users(username,hashed_password) VALUES('other','test') RETURNING id"
        )
    different = await w.enqueue_submission(
        submission(), scope=AccessScope(user_id=other), idempotency_key="upload"
    )
    assert different.created


async def test_retry_key_replay_survives_later_generations(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    await w.finish_attempt(await w.claim_next(), AttemptOutcome(kind="terminal"))
    request = RetryRequest(expected_generation=1, request_key="r1")
    first = await w.retry_submission(image.id, request, scope=scope)
    await w.finish_attempt(await w.claim_next(), AttemptOutcome(kind="terminal"))
    await w.retry_submission(
        image.id, RetryRequest(expected_generation=2, request_key="r2"), scope=scope
    )
    replay = await w.retry_submission(image.id, request, scope=scope)
    assert replay == first and replay.generation == 2
    with pytest.raises(ValueError, match="conflict"):
        await w.retry_submission(
            image.id, request.model_copy(update={"expected_generation": 3}), scope=scope
        )
    assert await w.retry_submission(image.id, request, scope=AccessScope(user_id=uuid4())) is None


@pytest.mark.parametrize("requested", [{}, {"model_id": "requested-model", "allow_ocr_fallback": True}])
async def test_policy_hold_and_adoption_fence_snapshot_history_and_replay(workflow_db, settings, requested):
    import json

    from vision_bill.config import OcrSettings
    from vision_bill.service.execution_policy import current_snapshot

    w, scope = workflow_db
    old = current_snapshot(settings, inference_identity=ENDPOINT)
    create = submission().model_copy(update={"effective_options": old, "requested_options": requested, "model_id": "old-resolved"})
    image = (await w.enqueue_submission(create, scope=scope, idempotency_key="upload")).submission
    # A previous real attempt must remain counted; a policy race only returns its own reservation.
    await w.finish_attempt(await w.claim_next(), AttemptOutcome(kind="retry", retry_delay_seconds=0))
    claim = await w.claim_next()
    held = await w.hold_claim_for_policy(claim, "execution_policy_changed")
    assert not held.stale and held.submission.status == "pending"
    assert held.submission.attempt_count == 1 and held.submission.generation == 1
    async with w.pool.acquire() as conn:
        assert json.loads(await conn.fetchval("SELECT effective_options FROM images WHERE id=$1", image.id)) == old
    assert (await w.finish_attempt(claim, complete())).stale
    assert (await w.hold_claim_for_policy(claim, "execution_policy_changed")).stale
    settings.ocr = OcrSettings(enabled=True)
    replacement = current_snapshot(settings, inference_identity=ENDPOINT)
    request = RetryRequest(expected_generation=1, request_key="adopt", policy="current")
    adopted = await w.retry_submission(image.id, request, scope=scope, replacement_snapshot=replacement)
    assert adopted.generation == 2 and adopted.attempt_count == 0
    assert adopted.model_id == requested.get("model_id")
    assert "effective_options" not in adopted.model_dump()
    async with w.pool.acquire() as conn:
        saved = await conn.fetchrow("SELECT * FROM images WHERE id=$1", image.id)
        assert json.loads(saved["effective_options"]) == replacement
        assert json.loads(saved["requested_options"]) == requested
        history = json.loads(saved["legacy_events"])[-1]
        assert history["event"] == "execution_policy_adopted"
        assert history["generation"] == 1 and history["model_id"] == "old-resolved"
        assert history["effective_options"] == old and history["requested_options"] == requested
        identity = await conn.fetchval("SELECT request_options FROM image_retry_requests WHERE image_id=$1", image.id)
        assert json.loads(identity) == {"policy": "current"}
    # No current settings or replacement validation can affect an already committed replay.
    assert await w.retry_submission(image.id, request, scope=scope, replacement_snapshot=None) == adopted
    with pytest.raises(ValueError, match="conflict"):
        await w.retry_submission(image.id, request.model_copy(update={"policy": "preserve"}), scope=scope)
    assert (await w.finish_attempt(claim, complete())).stale
    assert (await w.enqueue_submission(create, scope=scope, idempotency_key="upload")).submission.generation == 2
    # Selection binding is one-time and cannot replace the adopted frozen formatter policy.
    next_claim = await w.claim_next()
    bound = await w.ensure_execution_snapshot(next_claim, model_id="current-default", effective_options=replacement)
    assert bound.model_id == requested.get("model_id", "current-default")
    assert bound.effective_options == replacement
    rebound = await w.ensure_execution_snapshot(next_claim, model_id="later-default", effective_options=replacement)
    assert rebound == bound


async def test_legacy_retry_identity_defaults_to_preserve(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    await w.finish_attempt(await w.claim_next(), AttemptOutcome(kind="terminal"))
    request = RetryRequest(expected_generation=1, request_key="legacy")
    first = await w.retry_submission(image.id, request, scope=scope)
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE image_retry_requests SET request_options='{}' WHERE image_id=$1", image.id)
    assert await w.retry_submission(image.id, request, scope=scope) == first
    with pytest.raises(ValueError, match="conflict"):
        await w.retry_submission(image.id, request.model_copy(update={"policy": "current"}), scope=scope)


async def test_current_policy_cannot_replace_live_or_completed_work(workflow_db, settings):
    from vision_bill.service.execution_policy import current_snapshot
    w, scope = workflow_db
    image = await enqueue(w, scope)
    request = RetryRequest(expected_generation=1, request_key="adopt", policy="current")
    with pytest.raises(ValueError, match="retryable"):
        await w.retry_submission(image.id, request, scope=scope, replacement_snapshot=current_snapshot(settings, inference_identity=ENDPOINT))
    claim = await w.claim_next()
    await w.finish_attempt(claim, complete())
    with pytest.raises(ValueError, match="receipt"):
        await w.retry_submission(image.id, request, scope=scope, replacement_snapshot=current_snapshot(settings, inference_identity=ENDPOINT))


async def test_expired_processing_policy_hold_can_adopt_but_live_claim_cannot(workflow_db, settings):
    from vision_bill.service.execution_policy import current_snapshot
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    assert await w.set_wait_reason(claim.image, "execution_policy_changed")
    request = RetryRequest(expected_generation=1, request_key="adopt-expired", policy="current")
    with pytest.raises(ValueError, match="retryable"):
        await w.retry_submission(image.id, request, scope=scope, replacement_snapshot=current_snapshot(settings, inference_identity=ENDPOINT))
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1", image.id)
    adopted = await w.retry_submission(image.id, request, scope=scope, replacement_snapshot=current_snapshot(settings, inference_identity=ENDPOINT))
    assert adopted.generation == 2 and adopted.status == "pending" and adopted.attempt_count == 0
    assert (await w.finish_attempt(claim, complete())).stale
    async with w.pool.acquire() as conn:
        event = await conn.fetchrow("SELECT generation,attempt_number,result_code FROM receipt_extractions WHERE image_id=$1", image.id)
        assert dict(event) == {"generation": 1, "attempt_number": 1, "result_code": "lease_expired"}


async def test_policy_adoption_rechecks_extended_lease_under_row_lock(workflow_db, settings):
    from vision_bill.service.execution_policy import current_snapshot
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    assert await w.set_wait_reason(claim.image, "execution_policy_changed")
    async with w.pool.acquire() as conn, w.pool.acquire() as observer:
        await conn.execute("UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1", image.id)
        async with conn.transaction():
            await conn.fetchrow("SELECT id FROM images WHERE id=$1 FOR UPDATE", image.id)
            attempt = asyncio.create_task(w.retry_submission(
                image.id, RetryRequest(expected_generation=1, request_key="race", policy="current"),
                scope=scope, replacement_snapshot=current_snapshot(settings, inference_identity=ENDPOINT),
            ))
            # Observe the exact blocker from a separate session without a cached activity snapshot.
            for _ in range(100):
                if await observer.fetchval(
                    """SELECT EXISTS (SELECT 1 FROM pg_stat_activity
                    WHERE datname=current_database() AND $1=ANY(pg_blocking_pids(pid)))""",
                    conn.get_server_pid(),
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("retry did not wait on the owned image lock")
            await conn.execute("UPDATE images SET lease_expires_at=clock_timestamp()+interval '5 minutes' WHERE id=$1", image.id)
        with pytest.raises(ValueError, match="retryable"):
            await attempt
        saved = await conn.fetchrow("SELECT generation,status,claim_token FROM images WHERE id=$1", image.id)
        assert dict(saved) == {"generation": 1, "status": "processing", "claim_token": claim.claim_token}
        assert await conn.fetchval("SELECT count(*) FROM image_retry_requests WHERE image_id=$1", image.id) == 0


async def test_claim_policy_hold_does_not_change_a_concurrently_linked_receipt(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    async with w.pool.acquire() as conn:
        receipt = await conn.fetchval("INSERT INTO receipts(merchant_name,date,total,image_id) VALUES('human','2026-01-01',1.23,$1) RETURNING id", image.id)
        await conn.execute("UPDATE images SET receipt_id=$2 WHERE id=$1", image.id, receipt)
    result = await w.hold_claim_for_policy(claim, "execution_policy_changed")
    assert result.stale
    async with w.pool.acquire() as conn:
        saved = await conn.fetchrow("SELECT status,attempt_count,receipt_id FROM images WHERE id=$1", image.id)
        assert dict(saved) == {"status": "processing", "attempt_count": 1, "receipt_id": receipt}
        assert await conn.fetchval("SELECT total FROM receipts WHERE id=$1", receipt) == Decimal("1.23")


async def test_finish_atomic_decimal_and_immutable_history(workflow_db, db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    results = await asyncio.gather(
        w.finish_attempt(claim, complete()), w.finish_attempt(claim, complete())
    )
    assert sum(r.stale for r in results) == 1
    row = next(r.receipt for r in results if not r.stale)
    assert row.total == Decimal("90071992547.91")
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT total_price FROM line_items WHERE receipt_id=$1", row.id)
            == row.total
        )
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 1
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 1
        with pytest.raises(asyncpg.RaiseError):
            await conn.execute("UPDATE receipt_review_revisions SET event_kind='tamper'")
    with pytest.raises(ValueError, match="receipt"):
        await w.retry_submission(
            image.id, RetryRequest(expected_generation=1, request_key="never"), scope=scope
        )
    assert migrate(db, "0010", "downgrade", check=False).returncode != 0
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0025"
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 1


async def test_child_failure_rolls_back_everything(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    outcome = complete()
    outcome.envelope.receipt.line_items[0].description = "x" * 256
    with pytest.raises(asyncpg.StringDataRightTruncationError):
        await w.finish_attempt(claim, outcome)
    async with w.pool.acquire() as conn:
        for table in ("receipts", "receipt_extractions", "receipt_review_revisions"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
        assert (
            await conn.fetchval("SELECT status FROM images WHERE id=$1", image.id) == "processing"
        )


async def test_live_lease_must_still_hold_when_sealing(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    async with w.pool.acquire() as conn:
        await conn.execute("""CREATE FUNCTION delay_child() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_sleep(0.5); RETURN NEW; END $$""")
        await conn.execute(
            "CREATE TRIGGER delay_child BEFORE INSERT ON line_items FOR EACH ROW EXECUTE FUNCTION delay_child()"
        )
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp()+interval '0.2 second' WHERE id=$1",
            image.id,
        )
    assert (await w.finish_attempt(claim, complete())).stale
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 0


def wire_page():
    from vision_bill.model.ocr import OcrResponse

    return OcrResponse(
        request_id=uuid4(),
        photo_index=0,
        source_sha256="a" * 64,
        text="Merchant",
        blocks=[],
        profile_id="profile",
        runtime_fingerprint="runtime",
        working_image={
            "view_id": uuid4(),
            "width": 2,
            "height": 3,
            "sha256": "b" * 64,
            "png_base64": "private-image-bytes",
        },
    )


async def saved_view(w, scope, image, page):
    from vision_bill.model.ocr import WorkingImageMetadata
    from vision_bill.model.workflow import PersistedWorkingView

    parts = await w.get_source_parts(image.id, scope=scope)
    assert len(parts) == 1
    return PersistedWorkingView(
        part_id=parts[0].part_id,
        photo_index=0,
        source_sha256="a" * 64,
        artifact_path="/private/saved-working.png",
        metadata=WorkingImageMetadata.model_validate(page.working_image.model_dump()),
    )


async def test_checkpoints_bind_real_parts_profile_and_fence(workflow_db):
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    checkpoint = OcrCheckpoint(
        ordered_source_sha256=["a" * 64],
        profile_id="profile",
        runtime_fingerprint="runtime",
        pages=[page],
        persisted_views=[view],
    )
    assert await w.save_ocr_checkpoint(claim, checkpoint)
    assert await w.save_ocr_checkpoint(claim, checkpoint)
    with pytest.raises(ValueError, match="profile"):
        await w.save_ocr_checkpoint(claim, checkpoint.model_copy(update={"profile_id": "changed"}))
    with pytest.raises(ValueError, match="real source"):
        await w.save_ocr_checkpoint(
            claim,
            checkpoint.model_copy(
                update={"persisted_views": [view.model_copy(update={"part_id": uuid4()})]}
            ),
        )
    assert not await w.save_ocr_checkpoint(
        claim.model_copy(update={"claim_token": uuid4()}), checkpoint
    )
    async with w.pool.acquire() as conn:
        raw = await conn.fetchval("SELECT ocr_checkpoint::text FROM images WHERE id=$1", image.id)
        assert "png_base64" not in raw and "private-image-bytes" not in raw
    assert await w.get_source_parts(image.id, scope=AccessScope(user_id=uuid4())) == []


async def test_finish_requires_saved_artifacts_and_real_source_refs(workflow_db):
    from vision_bill.model.extraction import DerivedInputView
    from vision_bill.model.ocr import SourceRef

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    outcome = complete()
    outcome.envelope.ocr_pages = [page]
    outcome.envelope.input_views = [
        DerivedInputView(photo_index=0, source_sha256="a" * 64, working_image=page.working_image)
    ]
    with pytest.raises(ValueError, match="saved working artifact"):
        await w.finish_attempt(claim, outcome)
    outcome.persisted_views = [view]
    ref = SourceRef(part_id=uuid4(), photo_index=0, source_sha256="a" * 64)
    outcome.envelope.field_sources = {"total": [ref]}
    with pytest.raises(ValueError, match="real source part"):
        await w.finish_attempt(claim, outcome)
    ref.part_id = view.part_id
    ref.view_id = view.metadata.view_id
    ref.coordinate_space = "working"
    assert not (await w.finish_attempt(claim, outcome)).stale
    async with w.pool.acquire() as conn:
        raw = await conn.fetchval("SELECT envelope::text FROM receipt_extractions")
        assert "png_base64" not in raw and "private-image-bytes" not in raw
        assert str(view.part_id) in raw and str(view.metadata.view_id) in raw


@pytest.mark.parametrize("policy", ["preserve", "current"])
async def test_partial_draft_human_edits_and_receipt_link_are_preserved(workflow_db, settings, policy):
    from vision_bill.model.extraction import ExtractionDraft
    from vision_bill.model.review import ReviewFinding
    from vision_bill.service.execution_policy import current_snapshot

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    envelope = complete().envelope
    envelope.receipt = None
    envelope.draft = ExtractionDraft(merchant_name="uncertain", total="1.23")
    finding = ReviewFinding(
        id="missing", code="missing_value", severity="review", field_path="date"
    )
    review = ReviewEvaluation(findings=[finding], can_finalize=False, can_auto_verify=False)
    result = await w.finish_attempt(
        claim, AttemptOutcome(kind="partial", envelope=envelope, review=review)
    )
    assert result.submission.status == "needs_review" and result.submission.draft_available
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT findings->0->>'id' FROM receipt_drafts WHERE image_id=$1", image.id
            )
            == "missing"
        )
        await conn.execute(
            "UPDATE receipt_drafts SET human_edited=TRUE,revision=2 WHERE image_id=$1", image.id
        )
    with pytest.raises(ValueError, match="edited draft"):
        await w.retry_submission(
            image.id, RetryRequest(expected_generation=1, request_key="human", policy=policy), scope=scope,
            replacement_snapshot=current_snapshot(settings, inference_identity=ENDPOINT) if policy == "current" else None,
        )
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT document->>'total' FROM receipt_drafts WHERE image_id=$1", image.id
            )
            == "1.23"
        )


async def test_upgrade_0010_preserves_legacy_content_paths_and_ambiguity(db):
    migrate(db, "0010")
    conn = await asyncpg.connect(db.pg_dsn)
    ids = []
    for state in ("pending", "failed", "processing", "analyzed", "failed", "processing"):
        ids.append(
            await conn.fetchval(
                "INSERT INTO images(status,error,image_path) VALUES($1,'legacy error','/legacy/path') RETURNING id",
                state,
            )
        )
    receipt_id = await conn.fetchval(
        """INSERT INTO receipts(merchant_name,date,total,status,verified,image_id)
        VALUES('old','2020-01-01',12345678901.23,'verified',TRUE,$1) RETURNING id""",
        ids[4],
    )
    # Simulate historical duplicates explicitly preserved by migration 0002.
    await conn.execute("DROP TRIGGER receipts_image_id_unique ON receipts")
    await conn.execute("DROP INDEX ux_receipts_image_id")
    duplicate = await conn.fetchval(
        """INSERT INTO receipts(merchant_name,date,total,image_id)
        VALUES('duplicate','2020-01-01',9.87,$1) RETURNING id""",
        ids[4],
    )
    reverse = await conn.fetchval(
        "INSERT INTO receipts(merchant_name,date,total) VALUES('reverse','2020-01-01',2.34) RETURNING id"
    )
    await conn.execute("UPDATE images SET receipt_id=$2 WHERE id=$1", ids[5], reverse)
    await conn.close()
    migrate(db)
    conn = await asyncpg.connect(db.pg_dsn)
    states = [await conn.fetchval("SELECT status FROM images WHERE id=$1", image) for image in ids]
    assert states == ["pending", "pending", "pending", "analyzed", "analyzed", "analyzed"]
    receipt = await conn.fetchrow("SELECT * FROM receipts WHERE id=$1", receipt_id)
    assert (
        receipt["total"] == Decimal("12345678901.23")
        and receipt["verified"]
        and receipt["verification_source"] == "legacy"
    )
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM receipts WHERE id=ANY($1::uuid[])",
            [receipt_id, duplicate, reverse],
        )
        == 3
    )
    for image in ids:
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1", image)
        assert row["image_path"] == "/legacy/path" and row["user_id"] is None
        assert row["attempt_count"] == 0 and row["max_attempts"] == 3
    assert (
        await conn.fetchval("SELECT legacy_events->0->>'error' FROM images WHERE id=$1", ids[1])
        == "legacy error"
    )
    assert (
        await conn.fetchval(
            "SELECT legacy_events->0->>'receipt_count' FROM images WHERE id=$1", ids[4]
        )
        == "2"
    )
    await conn.close()


async def test_upgrade_0010_preserves_precise_local_times_and_legacy_evidence(db, tmp_path):
    from datetime import time as Time

    from vision_bill.provider.db.receipt_db import ReceiptDB

    migrate(db, "0010")
    conn = await asyncpg.connect(db.pg_dsn)
    owner = await conn.fetchval("INSERT INTO users(username,hashed_password) VALUES($1,'test') RETURNING id", str(uuid4()))
    records = []
    times = [Time(14, 30), Time(14, 30, 15), Time(14, 30, 15, 100000), Time(14, 30, 0, 1)]
    for index, value in enumerate(times):
        original = tmp_path / f"legacy-{index}.png"
        original.write_bytes(f"original legacy {index}".encode())
        image = await conn.fetchval("INSERT INTO images(user_id,image_path,status) VALUES($1,$2,'analyzed') RETURNING id", owner, str(original))
        receipt = await conn.fetchval("""INSERT INTO receipts(user_id,image_id,merchant_name,date,time,subtotal,total,tax_total,currency,verified,status)
            VALUES($1,$2,'Legacy','2020-01-01',$3,12.34,12.34,1.23,'EUR',TRUE,'verified') RETURNING id""", owner, image, value)
        await conn.execute("UPDATE images SET receipt_id=$2 WHERE id=$1", image, receipt)
        for position in range(2):
            await conn.execute("""INSERT INTO line_items(receipt_id,description,original_description,quantity,unit_price,total_price,position)
                VALUES($1,$2::text,$2::text,1.2345,6.17,6.17,$3)""", receipt, f"line-{position}", position)
            await conn.execute("INSERT INTO taxes(receipt_id,name,rate,amount) VALUES($1,$2,0.1234,0.615)", receipt, f"tax-{position}")
        records.append((receipt, image, original))
    run = await conn.fetchval("""INSERT INTO benchmark_runs(status,model_ids,receipt_ids,dataset_fingerprint,prompt_version,scoring_version,experiment_manifest)
        VALUES('completed','["legacy"]','[]','frozen','old','3','{"frozen_time":"14:30:15.1"}') RETURNING id""")
    await conn.execute("""INSERT INTO benchmark_results(run_id,model_id,receipt_id,success,payload)
        VALUES($1,'legacy',$2,TRUE,'{"time":"14:30:15.1","score":0.875}')""", run, records[0][0])
    queries = [
        "SELECT id,user_id,image_id,time,subtotal,total,tax_total,currency,verified,status FROM receipts ORDER BY id",
        "SELECT id,receipt_id,description,quantity,unit_price,total_price,position FROM line_items ORDER BY receipt_id,position",
        "SELECT id,receipt_id,name,rate,amount FROM taxes ORDER BY receipt_id,id",
        "SELECT id,user_id,image_path,receipt_id FROM images ORDER BY id",
        "SELECT experiment_manifest FROM benchmark_runs WHERE id=$1",
        "SELECT payload FROM benchmark_results WHERE run_id=$1",
    ]
    before = [await conn.fetch(query, *([run] if "$1" in query else [])) for query in queries]
    await conn.close()
    migrate(db)
    conn = await asyncpg.connect(db.pg_dsn)
    try:
        assert [await conn.fetch(query, *([run] if "$1" in query else [])) for query in queries] == before
        reader = ReceiptDB(db)
        reader._pool = await asyncpg.create_pool(db.pg_dsn, min_size=1, max_size=2)
        try:
            for index, (receipt, image, original) in enumerate(records):
                detail = await reader.get_receipt_with_details(receipt, user_id=owner)
                assert detail.receipt.time == times[index].isoformat()
                assert detail.receipt.verification_source == "legacy"
                assert [item.description for item in detail.line_items] == ["line-0", "line-1"]
                assert original.read_bytes() == f"original legacy {index}".encode()
                assert await conn.fetchval("SELECT image_path FROM images WHERE id=$1", image) == str(original)
        finally:
            await reader.destroy_db()
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0025"
    finally:
        await conn.close()


async def test_bind_legacy_parts_is_fenced_stable_and_cannot_rewrite(workflow_db):
    from vision_bill.model import workflow

    w, scope = workflow_db
    async with w.pool.acquire() as conn:
        image_id = await conn.fetchval(
            "INSERT INTO images(user_id,image_path) VALUES($1,'/legacy/original') RETURNING id",
            scope.user_id,
        )
    claim = await w.claim_next()
    assert hasattr(workflow, "SourcePartBinding"), "typed binding contract missing"
    binding = workflow.SourcePartBinding(
        photo_index=0, source_sha256="c" * 64, image_path="/legacy/original"
    )
    parts = await w.bind_source_parts(claim, [binding])
    assert parts[0].image_path == "/legacy/original" and parts[0].source_sha256 == "c" * 64
    assert (await w.bind_source_parts(claim, [binding])) == parts
    assert (
        await w.bind_source_parts(claim.model_copy(update={"claim_token": uuid4()}), [binding])
        is None
    )
    for key, value in (("source_sha256", "d" * 64), ("image_path", "/wrong"), ("photo_index", 1)):
        with pytest.raises(ValueError, match="source"):
            await w.bind_source_parts(claim, [binding.model_copy(update={key: value})])
    assert await w.get_source_parts(image_id, scope=scope) == parts


async def test_legacy_image_mutators_cannot_bypass_durable_fence(workflow_db):
    from vision_bill.provider.db.image_db import ImageDB

    w, scope = workflow_db
    image = await enqueue(w, scope)
    legacy = ImageDB(w._settings)
    legacy._pool = w.pool
    assert await legacy.claim_for_analysis(image.id) is None
    claim = await w.claim_next()
    await legacy.mark_failed(image.id, "old worker")
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT status FROM images WHERE id=$1", image.id) == "processing"
        )
    assert not (await w.finish_attempt(claim, complete())).stale


async def test_expired_third_lease_seals_terminal_without_fourth_claim(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    for attempt in range(1, 4):
        claim = await w.claim_next()
        assert claim.attempt_number == attempt
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
                image.id,
            )
        assert await w.reap_expired_claims() == 1
        assert await w.claim_next() is None
        if attempt < 3:
            saved = await w.get_submission(image.id, scope=scope)
            assert saved.status == "pending" and saved.attempt_count == attempt and saved.retry_at is not None
            async with w.pool.acquire() as conn:
                await conn.execute("UPDATE images SET retry_at=clock_timestamp() WHERE id=$1", image.id)
    async with w.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1", image.id)
        assert row["status"] == "failed" and row["attempt_count"] == 3
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 3


async def test_worker_backoff_configuration_is_honored(workflow_db):
    from vision_bill.config import WorkerSettings
    from vision_bill.provider.db.workflow_db import WorkflowDB

    w, scope = workflow_db
    configured = WorkflowDB(
        w._settings,
        worker_settings=WorkerSettings(retry_backoff_seconds=(3, 7), retry_backoff_cap_seconds=5),
    )
    configured._pool = w.pool
    await enqueue(configured, scope)
    claim = await configured.claim_next()
    result = await configured.finish_attempt(claim, AttemptOutcome(kind="retry"))
    async with w.pool.acquire() as conn:
        delay = await conn.fetchval(
            "SELECT EXTRACT(EPOCH FROM retry_at-clock_timestamp()) FROM images WHERE id=$1",
            result.submission.id,
        )
        assert Decimal(2) < delay <= Decimal(3)
        await conn.execute(
            "UPDATE images SET retry_at=clock_timestamp() WHERE id=$1", result.submission.id
        )
    result = await configured.finish_attempt(
        await configured.claim_next(), AttemptOutcome(kind="retry", retry_delay_seconds=99)
    )
    async with w.pool.acquire() as conn:
        delay = await conn.fetchval(
            "SELECT EXTRACT(EPOCH FROM retry_at-clock_timestamp()) FROM images WHERE id=$1",
            result.submission.id,
        )
        assert Decimal(4) < delay <= Decimal(5)


async def test_sealed_extraction_cannot_be_deleted_directly(workflow_db):
    w, scope = workflow_db
    await enqueue(w, scope)
    result = await w.finish_attempt(await w.claim_next(), complete())
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.RaiseError):
            await conn.execute("DELETE FROM receipt_extractions")
        await conn.execute("DELETE FROM receipts WHERE id=$1", result.receipt.id)
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 0


@pytest.mark.parametrize("path", ["terminal_review", "checkpoint_page", "finish_page"])
@pytest.mark.parametrize("forged", ["part", "view"])
async def test_every_persisted_review_and_page_reference_is_validated(workflow_db, path, forged):
    from vision_bill.model.review import ReviewFinding
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    reference = {
        "part_id": str(uuid4()) if forged == "part" else str(view.part_id),
        "photo_index": 0,
        "source_sha256": "a" * 64,
        "view_id": str(uuid4()) if forged == "view" else str(view.metadata.view_id),
    }
    if path == "terminal_review":
        finding = ReviewFinding(
            id="forged",
            code="source",
            severity="integrity",
            field_path="total",
            source_refs=[reference],
        )
        review = ReviewEvaluation(findings=[finding], can_finalize=False, can_auto_verify=False)
        with pytest.raises(ValueError, match="source reference"):
            await w.finish_attempt(
                claim, AttemptOutcome(kind="terminal", review=review, persisted_views=[view])
            )
    else:
        page.telemetry = {"elapsed_ms": 1, "nested": {"source_refs": [reference]}}
        if path == "checkpoint_page":
            checkpoint = OcrCheckpoint(
                ordered_source_sha256=["a" * 64],
                profile_id="profile",
                runtime_fingerprint="runtime",
                pages=[page],
                persisted_views=[view],
            )
            with pytest.raises(ValueError, match="source reference"):
                await w.save_ocr_checkpoint(claim, checkpoint)
        else:
            outcome = complete()
            outcome.envelope.ocr_pages = [page]
            outcome.persisted_views = [view]
            with pytest.raises(ValueError, match="source reference"):
                await w.finish_attempt(claim, outcome)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 0
        assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == 0
        assert (
            await conn.fetchval("SELECT ocr_checkpoint FROM images WHERE id=$1", image.id) is None
        )


async def test_checkpoint_read_is_fenced_and_preserves_saved_metadata(workflow_db):
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    checkpoint = OcrCheckpoint(
        ordered_source_sha256=["a" * 64],
        profile_id="profile",
        runtime_fingerprint="runtime",
        pages=[page],
        persisted_views=[view],
    )
    await w.save_ocr_checkpoint(claim, checkpoint)
    assert await w.get_ocr_checkpoint(claim) == checkpoint
    assert await w.get_ocr_checkpoint(claim.model_copy(update={"claim_token": uuid4()})) is None
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
            image.id,
        )
    assert await w.get_ocr_checkpoint(claim) is None


async def test_numeric_update_and_owner_admin_scope(workflow_db):
    from vision_bill.api.receipts import ReceiptUpdate
    from vision_bill.provider.db.receipt_db import ReceiptDB
    from vision_bill.service.image_service import ImageService

    w, scope = workflow_db
    image = await enqueue(w, scope)
    outcome = complete()
    result = await w.finish_attempt(await w.claim_next(), outcome)
    receipts = ReceiptDB(w._settings, original_checker=ImageService.check_originals)
    receipts._pool = w.pool
    updated = ReceiptUpdate.model_validate(dict(outcome.envelope.receipt.model_dump(), total="90071992547.93", subtotal="90071992547.93", expected_operation_revision=1))
    saved = await receipts.update_receipt(result.receipt.id, updated, user_id=scope.user_id)
    assert saved.total == Decimal("90071992547.93")
    other = AccessScope(user_id=uuid4())
    assert await w.get_submission(image.id, scope=other) is None
    assert (
        await w.get_submission(image.id, scope=other.model_copy(update={"can_see_all": True}))
    ).id == image.id
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """INSERT INTO receipt_extractions(image_id,generation,attempt_number,outcome_kind)
                VALUES($1,1,1,'complete')""",
                image.id,
            )


async def test_overlap_metadata_cannot_smuggle_fabricated_source_refs(workflow_db):
    w, scope = workflow_db
    await enqueue(w, scope)
    outcome = complete()
    outcome.envelope.overlap_decisions = [
        {"source_refs": [{"part_id": str(uuid4()), "photo_index": 0, "source_sha256": "a" * 64}]}
    ]
    with pytest.raises(ValueError, match="real source part"):
        await w.finish_attempt(await w.claim_next(), outcome)


async def test_upload_replay_ignores_new_storage_paths_and_changed_resolved_defaults(workflow_db):
    w, scope = workflow_db
    first = submission().model_copy(
        update={"effective_options": {"pipeline": "direct_vision"}, "model_id": "first-default"}
    )
    saved = await w.enqueue_submission(first, scope=scope, idempotency_key="stable-request")
    replay = first.model_copy(
        update={
            "id": uuid4(),
            "parts": [ImagePart(image_path="/new/random/path", thumbnail_path="/new/thumb")],
            "effective_options": {"pipeline": "ocr"},
            "model_id": "new-default",
            "bypass_review": True,
        }
    )
    result = await w.enqueue_submission(replay, scope=scope, idempotency_key="stable-request")
    assert not result.created and result.submission.id == saved.submission.id
    assert result.submission.model_id == "first-default" and not result.submission.bypass_review
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT effective_options->>'pipeline' FROM images WHERE id=$1", saved.submission.id
            )
            == "direct_vision"
        )
    explicit_changed = replay.model_copy(
        update={"requested_options": {"model_id": "explicitly-other"}}
    )
    with pytest.raises(ValueError, match="conflict"):
        await w.enqueue_submission(explicit_changed, scope=scope, idempotency_key="stable-request")


async def test_untouched_partial_drafts_allow_repeated_generations_with_monotonic_revision(
    workflow_db,
):
    from vision_bill.model.extraction import ExtractionDraft

    w, scope = workflow_db
    image = await enqueue(w, scope)
    for generation in range(1, 4):
        envelope = complete().envelope
        envelope.receipt = None
        envelope.draft = ExtractionDraft(merchant_name=f"generation {generation}")
        result = await w.finish_attempt(
            await w.claim_next(), AttemptOutcome(kind="partial", envelope=envelope)
        )
        assert result.submission.generation == generation
        async with w.pool.acquire() as conn:
            draft = await conn.fetchrow("SELECT * FROM receipt_drafts WHERE image_id=$1", image.id)
            assert draft["revision"] == generation and not draft["human_edited"]
        retried = await w.retry_submission(
            image.id,
            RetryRequest(expected_generation=generation, request_key=f"generation-{generation}"),
            scope=scope,
        )
        assert retried.generation == generation + 1


@pytest.mark.parametrize("legacy_rows", [False, True])
async def test_downgrade_preserves_exact_0010_terminal_status_contract(db, legacy_rows):
    migrate(db, "0010")
    conn = await asyncpg.connect(db.pg_dsn)
    if legacy_rows:
        await conn.execute(
            "INSERT INTO images(status,image_path) VALUES('timed_out','/legacy/timeout'),('unreadable','/legacy/unreadable')"
        )
    await conn.close()
    migrate(db)
    migrate(db, "0010", "downgrade")
    conn = await asyncpg.connect(db.pg_dsn)
    assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0010"
    for state in ("pending", "processing", "analyzed", "failed", "timed_out", "unreadable"):
        row_id = await conn.fetchval("INSERT INTO images(status) VALUES($1) RETURNING id", state)
        await conn.execute("UPDATE images SET status=$2 WHERE id=$1", row_id, state)
    if legacy_rows:
        assert (
            await conn.fetchval("SELECT status FROM images WHERE image_path='/legacy/timeout'")
            == "timed_out"
        )
        assert (
            await conn.fetchval("SELECT status FROM images WHERE image_path='/legacy/unreadable'")
            == "unreadable"
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await conn.execute("INSERT INTO images(status) VALUES('needs_review')")
    await conn.close()


async def test_review_actor_deletion_nulls_only_actor_preserves_history_and_owner(workflow_db):
    w, scope = workflow_db
    await enqueue(w, scope)
    result = await w.finish_attempt(await w.claim_next(), complete())
    async with w.pool.acquire() as conn:
        actor = await conn.fetchval(
            "INSERT INTO users(username,hashed_password,is_admin) VALUES('reviewer','test',TRUE) RETURNING id"
        )
        revision = await conn.fetchval(
            """INSERT INTO receipt_review_revisions(receipt_id,operation_revision,
            review_revision,event_kind,actor_user_id,snapshot) VALUES($1,2,1,'verified',$2,'{"content":"immutable"}') RETURNING id""",
            result.receipt.id,
            actor,
        )
        for statement in (
            "actor_user_id=NULL",
            "event_kind='tampered'",
            "snapshot='{}'::jsonb",
            "operation_revision=99",
        ):
            with pytest.raises(asyncpg.RaiseError, match="immutable"):
                await conn.execute(
                    f"UPDATE receipt_review_revisions SET {statement} WHERE id=$1", revision
                )
        await conn.execute("DELETE FROM users WHERE id=$1", actor)
        history = await conn.fetchrow(
            "SELECT * FROM receipt_review_revisions WHERE id=$1", revision
        )
        assert (
            history["actor_user_id"] is None and history["snapshot"] == '{"content": "immutable"}'
        )
        assert history["operation_revision"] == 2 and history["event_kind"] == "verified"
        assert (
            await conn.fetchval("SELECT user_id FROM receipts WHERE id=$1", result.receipt.id)
            == scope.user_id
        )
        with pytest.raises(asyncpg.RaiseError, match="immutable"):
            await conn.execute(
                "UPDATE receipt_review_revisions SET snapshot='{}'::jsonb WHERE id=$1", revision
            )
        await conn.execute("DELETE FROM receipts WHERE id=$1", result.receipt.id)
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 0


@pytest.mark.parametrize("terminal_review", [False, True])
async def test_reference_validation_preserves_real_references_and_ordinary_telemetry(
    workflow_db, terminal_review
):
    from vision_bill.model.review import ReviewFinding
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    reference = {
        "part_id": str(view.part_id),
        "photo_index": 0,
        "source_sha256": "a" * 64,
        "view_id": str(view.metadata.view_id),
    }
    review = ReviewEvaluation(
        findings=[
            ReviewFinding(
                id="real",
                code="review",
                severity="review",
                field_path="total",
                source_refs=[reference],
            )
        ],
        can_finalize=False,
        can_auto_verify=False,
    )
    if terminal_review:
        result = await w.finish_attempt(
            claim, AttemptOutcome(kind="terminal", review=review, persisted_views=[view])
        )
        assert not result.stale
    else:
        page.telemetry = {
            "elapsed_ms": 12,
            "provider": {"profile": "native"},
            "reference": reference,
        }
        checkpoint = OcrCheckpoint(
            ordered_source_sha256=["a" * 64],
            profile_id="profile",
            runtime_fingerprint="runtime",
            pages=[page],
            persisted_views=[view],
        )
        assert await w.save_ocr_checkpoint(claim, checkpoint)
        outcome = complete()
        outcome.envelope.ocr_pages = [page]
        outcome.persisted_views = [view]
        assert not (await w.finish_attempt(claim, outcome)).stale


async def partial_draft(w, scope, original_path=None):
    from vision_bill.model.extraction import ExtractionDraft

    request = submission()
    if original_path is not None:
        from hashlib import sha256
        original_path.write_bytes(b"original fixture")
        request.parts[0].image_path = str(original_path)
        request.source_manifest[0]["source_sha256"] = sha256(original_path.read_bytes()).hexdigest()
    image = (await w.enqueue_submission(request, scope=scope, idempotency_key=None)).submission
    outcome = complete()
    outcome.envelope.provenance.ordered_source_sha256 = [request.source_manifest[0]["source_sha256"]]
    outcome.envelope.draft = ExtractionDraft(merchant_name="observed", total=None)
    outcome.envelope.receipt = None
    await w.finish_attempt(
        await w.claim_next(), AttemptOutcome(kind="partial", envelope=outcome.envelope)
    )
    return image


async def test_draft_null_cas_human_edit_and_owner_scope(workflow_db):
    from vision_bill.model.workflow import DraftPatch

    w, scope = workflow_db
    image = await partial_draft(w, scope)
    draft = await w.get_draft(image.id, scope=scope)
    assert draft.document.total is None and draft.revision == 1
    assert await w.get_draft(image.id, scope=AccessScope(user_id=uuid4())) is None
    patched = await w.patch_draft(
        image.id, DraftPatch(expected_revision=1, changes={"merchant_name": None}), scope=scope
    )
    assert patched.revision == 2 and patched.document.merchant_name is None
    assert (await w.get_draft(image.id, scope=scope)).document.merchant_name is None
    with pytest.raises(ValueError, match="revision"):
        await w.patch_draft(
            image.id, DraftPatch(expected_revision=1, changes={"total": "1.20"}), scope=scope
        )
    with pytest.raises(ValueError, match="edited"):
        await w.retry_submission(
            image.id, RetryRequest(expected_generation=1, request_key="retry"), scope=scope
        )
    async with w.pool.acquire() as conn:
        assert await conn.fetchval(
            "SELECT human_edited FROM receipt_drafts WHERE image_id=$1", image.id
        )


@pytest.mark.parametrize("kind", ["complete", "partial"])
async def test_purchase_grouping_keeps_raw_extraction_in_draft_and_receipt(workflow_db, kind):
    from vision_bill.model.extraction import ExtractionDraft

    w, scope = workflow_db
    image = await enqueue(w, scope)
    outcome = complete(total="0.58")
    assert outcome.envelope is not None
    receipt = outcome.envelope.receipt
    assert receipt is not None
    rows = [{"description": "Energy Editions", "total_price": "0.29"}] * 2
    data = receipt.model_dump() | {"line_items": rows}
    outcome.envelope.draft = ExtractionDraft.model_validate(data)
    if kind == "partial":
        outcome.envelope.draft.total = None
        outcome.envelope.receipt = None
        outcome = AttemptOutcome(kind="partial", envelope=outcome.envelope)
    else:
        outcome.envelope.receipt = Receipt.model_validate(data)
    finished = await w.finish_attempt(await w.claim_next(), outcome)
    assert not finished.stale
    if kind == "complete":
        assert finished.receipt is not None
        async with w.pool.acquire() as conn:
            stored = await conn.fetch(
                "SELECT quantity, total_price FROM line_items WHERE receipt_id=$1",
                finished.receipt.id,
            )
        assert [(row["quantity"], row["total_price"]) for row in stored] == [
            (Decimal(2), Decimal("0.58")),
        ]
    else:
        draft = await w.get_draft(image.id, scope=scope)
        assert draft is not None and draft.document.line_items
        items = draft.document.line_items
        assert len(items) == 1
        assert items[0].quantity == 2 and items[0].total_price == Decimal("0.58")
        assert items[0].spending_category == "soft_drinks"
    async with w.pool.acquire() as conn:
        raw_count = await conn.fetchval(
            "SELECT jsonb_array_length(envelope #> '{draft,line_items}') FROM receipt_extractions WHERE image_id=$1",
            image.id,
        )
    assert raw_count == 2


async def test_clear_checkpoint_is_fenced_and_retains_immutable_views(workflow_db):
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    await w.save_ocr_checkpoint(
        claim,
        OcrCheckpoint(
            ordered_source_sha256=["a" * 64],
            profile_id="profile",
            runtime_fingerprint="runtime",
            pages=[page],
            persisted_views=[view],
        ),
    )
    assert not await w.clear_ocr_checkpoint(claim.model_copy(update={"claim_token": uuid4()}))
    assert await w.get_ocr_checkpoint(claim) is not None
    assert await w.clear_ocr_checkpoint(claim)
    assert await w.get_ocr_checkpoint(claim) is None
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == 1


async def test_draft_finalize_missing_then_once_unverified_and_rollback(workflow_db, tmp_path):
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.model.workflow import DraftFinalize, DraftPatch

    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / "original.png")
    parts = await w.get_source_parts(image.id, scope=scope)
    originals = [
        OriginalAvailability(part_id=p.part_id, available=True, sha256_matches=True) for p in parts
    ]
    request = DraftFinalize(expected_revision=1, request_key="final")
    with pytest.raises(ValueError, match="incomplete"):
        await w.finalize_draft(image.id, request, scope=scope, originals=originals)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
    document = complete().envelope.receipt.model_dump(mode="json")
    await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes=document), scope=scope)
    request = DraftFinalize(expected_revision=2, request_key="final")
    with pytest.raises(ValueError, match="original"):
        await w.finalize_draft(image.id, request, scope=scope, originals=[])
    first, repeated = await asyncio.gather(
        *(w.finalize_draft(image.id, request, scope=scope, originals=originals) for _ in range(2))
    )
    assert first.id == repeated.id and not first.verified and first.status == "unverified"
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 1
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 1
        assert (
            await conn.fetchval("SELECT receipt_id FROM images WHERE id=$1", image.id) == first.id
        )


async def test_finalized_draft_preserves_optional_amounts_and_item_values(workflow_db, tmp_path):
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.model.workflow import DraftFinalize, DraftPatch

    workflow, scope = workflow_db
    workflow._receipts._pool = workflow.pool
    image = await partial_draft(workflow, scope, tmp_path / "original.png")
    parts = await workflow.get_source_parts(image.id, scope=scope)
    originals = [
        OriginalAvailability(part_id=part.part_id, available=True, sha256_matches=True)
        for part in parts
    ]
    document = {
        "confidence": 90, "merchant_name": "Shop", "date": "2026-01-01",
        "currency": "EUR", "payment_method": None, "subtotal": "2.33",
        "tax_total": "0.16", "total": "2.49", "tip": None, "discount_total": None,
        "taxes": [{"name": "VAT", "rate": 0.07, "amount": "0.16"}],
        "line_items": [
            {"description": "Item", "total_price": "2.49", "quantity": None, "unit_price": None}
        ],
        "amount_policy": {
            "tax_basis": "gross", "subtotal_origin": "derived_net",
            "tip_treatment": "none", "tax_information": "present",
        },
    }
    draft = await workflow.patch_draft(
        image.id, DraftPatch(expected_revision=1, changes=document), scope=scope,
    )
    assert draft is not None and draft.revision == 2
    assert any(f.code == "missing_mandatory_field" and f.field_path == "payment_method" for f in draft.findings)
    corrected = await workflow.patch_draft(
        image.id, DraftPatch(expected_revision=2, changes={"payment_method": "BAR", "confidence": 100}),
        scope=scope,
    )
    assert corrected is not None and corrected.revision == 3
    assert corrected.document.payment_method == "cash" and corrected.document.confidence == 100
    assert all(f.field_path != "payment_method" for f in corrected.findings)
    result = await workflow.finalize_draft(
        image.id, DraftFinalize(expected_revision=3, request_key="optional-values"),
        scope=scope, originals=originals,
    )
    assert result is not None and not result.verified
    assert result.discount_total is result.tip is None
    assert result.total == Decimal("2.49")
    saved = await workflow._receipts.get_receipt_with_details(result.id, user_id=scope.user_id)
    assert saved is not None
    assert saved.receipt.discount_total is saved.receipt.tip is None
    assert saved.line_items[0].quantity is saved.line_items[0].unit_price is None
    assert saved.line_items[0].total_price == Decimal("2.49")


async def test_draft_finalization_rolls_back_after_receipt_insert(workflow_db, monkeypatch, tmp_path):
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.model.workflow import DraftFinalize, DraftPatch

    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / "original.png")
    document = complete().envelope.receipt.model_dump(mode="json")
    await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes=document), scope=scope)
    originals = [
        OriginalAvailability(part_id=p.part_id, available=True, sha256_matches=True)
        for p in await w.get_source_parts(image.id, scope=scope)
    ]
    insert = w._receipts.insert_receipt

    async def interrupted(*args, **kwargs):
        await insert(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(w._receipts, "insert_receipt", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await w.finalize_draft(
            image.id,
            DraftFinalize(expected_revision=2, request_key="final"),
            scope=scope,
            originals=originals,
        )
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 0
        assert (
            await conn.fetchval(
                "SELECT finalized_receipt_id FROM receipt_drafts WHERE image_id=$1", image.id
            )
            is None
        )
        assert (
            await conn.fetchval("SELECT status FROM images WHERE id=$1", image.id) == "needs_review"
        )


@pytest.mark.parametrize("due_kind", ["retry", "lease"])
async def test_next_wake_includes_overdue_retry_and_defers_unavailable_candidate(
    workflow_db, due_kind
):
    w, scope = workflow_db
    first, second = await enqueue(w, scope), await enqueue(w, scope)
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET retry_at=clock_timestamp()-interval '30 seconds' WHERE id=$1",
            first.id,
        )
        await conn.execute(
            "UPDATE images SET retry_at=clock_timestamp()+interval '60 seconds' WHERE id=$1",
            second.id,
        )
    if due_kind == "lease":
        claim = await w.claim_next(image_id=first.id)
        assert claim is not None
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
                first.id,
            )
    assert await w.next_wake_delay(300) == (1 if due_kind == "lease" else 0)
    delay = await w.next_wake_delay(300, deferred_image_ids=[first.id])
    assert 55 < delay <= 60
    assert await w.next_wake_delay(300, deferred_image_ids=[first.id, second.id]) == 300


async def test_long_cycle_rescans_overdue_retry_using_real_wake_query(
    workflow_db, settings, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import vision_bill.service.analysis_scheduler as module

    w, scope = workflow_db
    first, second = await enqueue(w, scope), await enqueue(w, scope)
    clock = [0.0]
    runs, waits = [], []
    provider_ready = [True]

    async def overdue():
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE images SET retry_at=clock_timestamp()-interval '1 second' WHERE id=$1",
                first.id,
            )

    async def process(claim):
        runs.append((claim.image.id, clock[0]))
        if claim.image.id == second.id:
            clock[0] += 60
            await overdue()
            return await w.finish_attempt(
                claim, AttemptOutcome(kind="terminal", result_code="unreadable")
            )
        result = await w.finish_attempt(
            claim, AttemptOutcome(kind="retry", result_code="timed_out")
        )
        if claim.attempt_number == 2:
            provider_ready[0] = False
            await overdue()
        return result

    async def wait_for(awaitable, *, timeout):
        awaitable.close()
        waits.append(timeout)
        if len(waits) == 4:
            raise asyncio.CancelledError
        if len(waits) > 1:
            clock[0] += timeout
        raise TimeoutError

    service = SimpleNamespace(
        ready_for=AsyncMock(side_effect=lambda image: provider_ready[0]),
        process_claim=process,
    )
    scheduler = module.AnalysisScheduler(
        settings, None, SimpleNamespace(workflow_db=w), None, submission_service=service
    )
    monkeypatch.setattr(
        module, "asyncio", SimpleNamespace(wait_for=wait_for, timeout=asyncio.timeout)
    )
    with pytest.raises(asyncio.CancelledError):
        await scheduler._run_forever()
    assert runs == [(first.id, 0), (second.id, 0), (first.id, 60)]
    assert waits == [300, 0, 0, 300]
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT attempt_count FROM images WHERE id=$1", first.id) == 2
        assert await conn.fetchval("SELECT status FROM images WHERE id=$1", first.id) == "pending"

async def test_draft_policy_and_sources_are_versioned_partial_changes(workflow_db, tmp_path):
    from vision_bill.model.workflow import DraftPatch
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    policy = {'tax_basis': 'gross', 'discount_treatment': 'included_in_lines',
              'tip_treatment': 'none', 'subtotal_origin': 'printed', 'tax_information': 'explicit_zero'}
    saved = await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={
        'merchant_name': None, 'amount_policy': policy, 'field_sources': {}, 'overlap_decisions': []
    }), scope=scope)
    assert saved.revision == 2 and saved.document.merchant_name is None and saved.document.total is None
    assert saved.amount_policy.model_dump() == policy
    assert saved.field_sources == {} and saved.overlap_decisions == []
    assert any(f.code == 'missing_mandatory_field' for f in saved.findings)
    assert not any(f.code == 'unknown_tax_basis' for f in saved.findings)
    with pytest.raises(ValueError, match='revision'):
        await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={'amount_policy': {'tax_basis':'net'}}), scope=scope)
    current = await w.get_draft(image.id, scope=scope)
    assert current.revision == 2 and current.amount_policy.tax_basis == 'gross'
    preserved = await w.patch_draft(image.id, DraftPatch(expected_revision=2, changes={'total':'0.000001'}), scope=scope)
    assert preserved.amount_policy == saved.amount_policy and preserved.document.merchant_name is None


@pytest.mark.parametrize('field', ['field_sources', 'overlap_decisions'])
async def test_draft_foreign_source_replacement_rolls_back(workflow_db, tmp_path, field):
    from vision_bill.model.workflow import DraftPatch
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    foreign = {'part_id':str(uuid4()),'photo_index':0,'source_sha256':'a'*64,
               'view_id':str(uuid4()),'coordinate_space':'working','bbox':[[1,1],[2,1],[2,2]]}
    metadata = {'total':[foreign]} if field == 'field_sources' else [{'source_refs':[foreign]}]
    with pytest.raises(ValueError, match='source'):
        await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={'total':'99',field:metadata}), scope=scope)
    current = await w.get_draft(image.id, scope=scope)
    assert current.revision == 1 and current.document.total is None
    async with w.pool.acquire() as conn:
        assert not await conn.fetchval('SELECT human_edited FROM receipt_drafts WHERE image_id=$1', image.id)

async def test_draft_review_metadata_http_contract(workflow_db, tmp_path):
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI

    from vision_bill.api.drafts import router
    from vision_bill.security.dependencies import get_current_user
    from vision_bill.security.models import User
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    app = FastAPI()
    app.include_router(router, prefix='/api/v1/images')
    app.dependency_overrides[get_current_user] = lambda: User(id=scope.user_id, username='owner', is_admin=False)
    app.state.submission_service = SimpleNamespace(db=w, patch_draft=w.patch_draft, get_draft=w.get_draft)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        path = f'/api/v1/images/{image.id}/draft'
        saved = await client.patch(path, json={'expected_revision':1,'changes':{
            'amount_policy':{'tax_basis':'net'},'merchant_name':None,'field_sources':{},'overlap_decisions':[]}})
        assert saved.status_code == 200
        assert saved.json()['amount_policy']['tax_basis'] == 'net'
        assert saved.json()['document']['merchant_name'] is None
        assert saved.json()['revision'] == 2
        assert (await client.patch(path, json={'expected_revision':1,'changes':{'total':'4'}})).status_code == 409
        foreign = {'part_id':str(uuid4()),'photo_index':0,'source_sha256':'a'*64,'coordinate_space':'unknown'}
        assert (await client.patch(path, json={'expected_revision':2,'changes':{'total':'99','field_sources':{'total':[foreign]}}})).status_code == 422
        current = (await client.get(path)).json()
        assert current['revision'] == 2 and current['document']['total'] is None


async def test_draft_overlap_migration_backfill_and_guard(db):
    import json
    migrate(db, '0014')
    from vision_bill.provider.db.workflow_db import WorkflowDB
    w = WorkflowDB(db)
    await w.init_db()
    try:
        async with w.pool.acquire() as conn:
            owner = await conn.fetchval("INSERT INTO users(username,hashed_password) VALUES('draft-test','hash') RETURNING id")
            image = await conn.fetchval("INSERT INTO images(image_path,user_id) VALUES('/private-original',$1) RETURNING id", owner)
            decisions = [{'decision':'retain', 'positions':[0,1], 'reason':'printed twice'}]
            extraction = await conn.fetchval("""INSERT INTO receipt_extractions(image_id,generation,attempt_number,outcome_kind,envelope,source_views)
                VALUES($1,1,1,'partial',$2::jsonb,'[]') RETURNING id""", image, json.dumps({'overlap_decisions':decisions}))
            await conn.execute("""INSERT INTO receipt_drafts(image_id,extraction_id,generation,document,amount_policy,findings,field_sources)
                VALUES($1,$2,1,'{}','{}','[]','{}')""", image, extraction)
        migrate(db)
        async with w.pool.acquire() as conn:
            assert json.loads(await conn.fetchval('SELECT overlap_decisions FROM receipt_drafts WHERE image_id=$1',image)) == decisions
            assert json.loads(await conn.fetchval('SELECT envelope FROM receipt_extractions WHERE id=$1',extraction)) == {'overlap_decisions':decisions}
        assert migrate(db,'0014','downgrade',check=False).returncode != 0
    finally:
        await w.destroy_db()

async def test_draft_validates_nested_metadata_before_typed_projection(workflow_db, tmp_path):
    from vision_bill.model.workflow import DraftPatch
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    part = (await w.get_source_parts(image.id, scope=scope))[0]
    ref = {'part_id':str(part.part_id),'photo_index':0,'source_sha256':part.source_sha256,
           'coordinate_space':'unknown','extra':{'part_id':str(uuid4()),'photo_index':0,
           'source_sha256':'a'*64,'coordinate_space':'unknown'}}
    with pytest.raises(ValueError, match='source'):
        await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={'field_sources':{'total':[ref]},'total':'9'}), scope=scope)
    current = await w.get_draft(image.id, scope=scope)
    assert current.revision == 1 and current.document.total is None


async def capture_collection(w, owner, *, active=True):
    async with w.pool.acquire() as conn:
        return await conn.fetchval(
            "INSERT INTO collections(user_id,name,active) VALUES($1,'capture',$2) RETURNING id",
            owner, active,
        )


async def capture_creation(w, scope, path, tmp_path):
    """Prepare either real workflow creation path; return a replayable operation."""
    if path == "complete":
        image = await enqueue(w, scope)
        claim = await w.claim_next()

        async def create():
            result = await w.finish_attempt(claim, complete())
            return result.receipt
    else:
        from vision_bill.model.review import OriginalAvailability
        from vision_bill.model.workflow import DraftFinalize, DraftPatch

        image = await partial_draft(w, scope, tmp_path / "capture-original.png")
        await w.patch_draft(
            image.id,
            DraftPatch(expected_revision=1, changes=complete().envelope.receipt.model_dump(mode="json")),
            scope=scope,
        )
        originals = [
            OriginalAvailability(part_id=p.part_id, available=True, sha256_matches=True)
            for p in await w.get_source_parts(image.id, scope=scope)
        ]
        # An administrator acts for the owner; capture must still use image ownership.
        admin = uuid4()
        async with w.pool.acquire() as conn:
            await conn.execute("INSERT INTO users(id,username,hashed_password,is_admin) VALUES($1,$2,'test',TRUE)", admin, str(admin))
        await capture_collection(w, admin)

        async def create():
            return await w.finalize_draft(
                image.id, DraftFinalize(expected_revision=2, request_key="capture-final"),
                scope=AccessScope(user_id=admin, can_see_all=True), originals=originals,
            )
    return image, create


@pytest.mark.parametrize("path", ["complete", "draft"])
@pytest.mark.parametrize("ownership", ["owner", "inactive", "ownerless"])
async def test_workflow_creation_captures_only_owner_active_collection(
    workflow_db, tmp_path, path, ownership
):
    import json

    w, scope = workflow_db
    own = await capture_collection(w, scope.user_id, active=ownership != "inactive")
    foreign = uuid4()
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO users(id,username,hashed_password) VALUES($1,$2,'test')", foreign, str(foreign))
    await capture_collection(w, foreign)
    image, create = await capture_creation(w, scope, path, tmp_path)
    if ownership == "ownerless":
        async with w.pool.acquire() as conn:
            await conn.execute("UPDATE images SET user_id=NULL WHERE id=$1", image.id)
    first = await create()
    assert first is not None and not first.verified
    replay = await create()
    if path == "complete":
        assert replay is None
    else:
        assert replay.id == first.id
    expected = [own] if ownership == "owner" else []
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 1
        links = await conn.fetch("SELECT collection_id FROM receipt_collections WHERE receipt_id=$1", first.id)
        assert [r["collection_id"] for r in links] == expected
        revisions = await conn.fetch("SELECT * FROM receipt_review_revisions WHERE receipt_id=$1", first.id)
        assert len(revisions) == 1
        revision = revisions[0]
        assert revision["event_kind"] == ("extracted" if path == "complete" else "draft_finalized")
        assert revision["operation_revision"] == revision["review_revision"] == 1
        snapshot = json.loads(revision["snapshot"])
        assert snapshot["collection_ids"] == [str(cid) for cid in expected]
        assert snapshot["receipt"]["id"] == str(first.id)
        assert snapshot["document"]["merchant_name"] == "Merchant"


@pytest.mark.parametrize("path", ["complete", "draft"])
async def test_capture_rolls_back_with_receipt_creation(workflow_db, tmp_path, monkeypatch, path):
    w, scope = workflow_db
    await capture_collection(w, scope.user_id)
    image, create = await capture_creation(w, scope, path, tmp_path)
    insert = w._receipts.insert_receipt

    async def interrupted(*args, **kwargs):
        receipt = await insert(*args, **kwargs)
        conn = args[0]
        assert await conn.fetchval("SELECT count(*) FROM receipt_collections WHERE receipt_id=$1", receipt.id) == 1
        raise asyncio.CancelledError()

    monkeypatch.setattr(w._receipts, "insert_receipt", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await create()
    async with w.pool.acquire() as conn:
        for table in ("receipts", "receipt_collections", "receipt_review_revisions"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
        assert await conn.fetchval("SELECT receipt_id FROM images WHERE id=$1", image.id) is None


@pytest.mark.parametrize("path", ["complete", "draft"])
async def test_optional_capture_integrity_error_does_not_abort_receipt(workflow_db, tmp_path, path, caplog):
    import json

    w, scope = workflow_db
    await capture_collection(w, scope.user_id)
    _, create = await capture_creation(w, scope, path, tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute("ALTER TABLE receipt_collections ADD CONSTRAINT capture_rejected CHECK(FALSE) NOT VALID")
    receipt = await create()
    assert receipt is not None
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipt_collections") == 0
        snapshot = await conn.fetchval("SELECT snapshot FROM receipt_review_revisions WHERE receipt_id=$1", receipt.id)
        assert json.loads(snapshot)["collection_ids"] == []
    assert "Active collection capture rejected" in caplog.text
    assert "capture_rejected" not in caplog.text


async def test_capture_serializes_collection_deletion_and_records_later_removal(workflow_db, tmp_path, monkeypatch):
    import json

    from vision_bill.provider.db.collection_db import CollectionDB

    w, scope = workflow_db
    collection = await capture_collection(w, scope.user_id)
    _, create = await capture_creation(w, scope, "complete", tmp_path)
    inserted, release = asyncio.Event(), asyncio.Event()
    insert = w._receipts.insert_receipt

    async def pause_after_capture(*args, **kwargs):
        receipt = await insert(*args, **kwargs)
        assert await args[0].fetchval("SELECT count(*) FROM receipt_collections WHERE receipt_id=$1", receipt.id) == 1
        inserted.set()
        await release.wait()
        return receipt

    monkeypatch.setattr(w._receipts, "insert_receipt", pause_after_capture)
    creation = asyncio.create_task(create())
    deletion = None
    try:
        await asyncio.wait_for(inserted.wait(), 3)
        deletion = asyncio.create_task(CollectionDB(w.pool).delete(collection, scope.user_id, False))
        async with asyncio.timeout(3):
            async with w.pool.acquire() as conn:
                while not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event='advisory')"):
                    await asyncio.sleep(0)
        assert not deletion.done()
        release.set()
        receipt = await creation
        assert await deletion
        async with w.pool.acquire() as conn:
            revisions = await conn.fetch("SELECT event_kind,snapshot,operation_revision FROM receipt_review_revisions WHERE receipt_id=$1 ORDER BY operation_revision", receipt.id)
            assert [r["event_kind"] for r in revisions] == ["extracted", "collections_changed"]
            assert [r["operation_revision"] for r in revisions] == [1, 2]
            assert [json.loads(r["snapshot"])["collection_ids"] for r in revisions] == [[str(collection)], []]
            assert await conn.fetchval("SELECT count(*) FROM receipt_collections") == 0
    finally:
        release.set()
        for task in (creation, deletion):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(t for t in (creation, deletion) if t is not None), return_exceptions=True)


@pytest.mark.parametrize("path, interruption", [
    ("complete", "deadline"), ("complete", "cancel"),
    ("draft", "cancel"), ("complete", "connection"),
])
async def test_capture_blocked_lock_propagates_interruption(
    workflow_db, tmp_path, path, interruption, caplog
):
    from vision_bill.provider.db.receipt_db import MEMBERSHIP_LOCK_SQL

    w, scope = workflow_db
    await capture_collection(w, scope.user_id)
    if interruption == "deadline":
        image = await enqueue(w, scope)
        claim = await w.claim_next()
        deadline = asyncio.get_running_loop().time() + 2

        async def create():
            return await w.finish_attempt(claim, complete(), deadline=deadline)
    else:
        image, create = await capture_creation(w, scope, path, tmp_path)
    async with w.pool.acquire() as blocker, blocker.transaction():
        await blocker.execute(MEMBERSHIP_LOCK_SQL)
        task = asyncio.create_task(create())
        try:
            # Observe the actual blocked capture query before interrupting it.
            async with asyncio.timeout(1):
                async with w.pool.acquire() as observer:
                    while not (pid := await observer.fetchval(
                        "SELECT pid FROM pg_stat_activity WHERE datname=current_database() AND wait_event='advisory' LIMIT 1"
                    )):
                        await asyncio.sleep(0)
            assert not task.done()
            if interruption == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif interruption == "connection":
                async with w.pool.acquire() as observer:
                    assert await observer.fetchval("SELECT pg_terminate_backend($1)", pid)
                with pytest.raises(asyncpg.InterfaceError, match="connection is closed"):
                    await task
            else:
                with pytest.raises(TimeoutError):
                    await task
                assert asyncio.get_running_loop().time() < deadline + 0.35
            assert "Active collection capture rejected" not in caplog.text
            # The blocker remains held: a swallowed error cannot fake progress.
            async with w.pool.acquire() as observer:
                for table in ("receipts", "receipt_collections", "receipt_review_revisions"):
                    assert await observer.fetchval(f"SELECT count(*) FROM {table}") == 0
                assert await observer.fetchval("SELECT receipt_id FROM images WHERE id=$1", image.id) is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_capture_serializes_concurrent_activation(workflow_db, tmp_path, monkeypatch):
    from vision_bill.provider.db.collection_db import CollectionDB

    w, scope = workflow_db
    old = await capture_collection(w, scope.user_id)
    new = await capture_collection(w, scope.user_id, active=False)
    _, create = await capture_creation(w, scope, "complete", tmp_path)
    inserted, release = asyncio.Event(), asyncio.Event()
    insert = w._receipts.insert_receipt

    async def pause_after_capture(*args, **kwargs):
        receipt = await insert(*args, **kwargs)
        inserted.set()
        await release.wait()
        return receipt

    monkeypatch.setattr(w._receipts, "insert_receipt", pause_after_capture)
    creation = asyncio.create_task(create())
    activation = None
    try:
        await asyncio.wait_for(inserted.wait(), 3)
        activation = asyncio.create_task(CollectionDB(w.pool).activate(new, scope.user_id, False))
        async with asyncio.timeout(3):
            async with w.pool.acquire() as conn:
                while not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock')"):
                    await asyncio.sleep(0)
        assert not activation.done()
        release.set()
        first = await creation
        assert (await activation).id == new
        monkeypatch.setattr(w._receipts, "insert_receipt", insert)
        _, create_next = await capture_creation(w, scope, "complete", tmp_path)
        second = await create_next()
        async with w.pool.acquire() as conn:
            assert await conn.fetchval("SELECT collection_id FROM receipt_collections WHERE receipt_id=$1", first.id) == old
            assert await conn.fetchval("SELECT collection_id FROM receipt_collections WHERE receipt_id=$1", second.id) == new
    finally:
        release.set()
        for task in (creation, activation):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(t for t in (creation, activation) if t is not None), return_exceptions=True)


async def test_finalized_draft_receipt_can_be_deleted_without_benchmark(workflow_db,tmp_path):
    w,scope=workflow_db
    image,create=await capture_creation(w,scope,'draft',tmp_path)
    row=await create()
    w._receipts._pool = w.pool
    deleted=await w._receipts.delete_receipt(row.id,user_id=scope.user_id)
    assert deleted.id == row.id
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT finalized_receipt_id FROM receipt_drafts WHERE image_id=$1',image.id) is None
        assert await conn.fetchval('SELECT count(*) FROM receipts WHERE id=$1',row.id)==0


async def test_atomic_finalized_deletion_cannot_refinalize_in_delete_interval(workflow_db,tmp_path):
    from vision_bill.model.workflow import DraftFinalize
    w,scope=workflow_db
    image,create=await capture_creation(w,scope,'draft',tmp_path)
    row=await create()
    async with w.pool.acquire() as conn:
        await conn.execute("""CREATE FUNCTION deletion_barrier() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_advisory_xact_lock(731829); RETURN OLD; END $$;
            CREATE TRIGGER deletion_barrier BEFORE DELETE ON receipts FOR EACH ROW EXECUTE FUNCTION deletion_barrier()""")
    async with w.pool.acquire() as blocker:
        await blocker.execute('SELECT pg_advisory_lock(731829)')
        delete=asyncio.create_task(w.delete_resource(row.id,receipt=True,scope=scope,validate_files=lambda _:None))
        finalize=None
        try:
            async with asyncio.timeout(3):
                async with w.pool.acquire() as observer:
                    while not await observer.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event='advisory')"):
                        await asyncio.sleep(0)
            finalize=asyncio.create_task(w.finalize_draft(image.id,DraftFinalize(expected_revision=2,request_key='interleaved'),scope=scope,originals=[]))
            await blocker.execute('SELECT pg_advisory_unlock(731829)')
            assert (await delete).receipt_id==row.id
            assert await finalize is None
            async with w.pool.acquire() as conn:
                assert await conn.fetchval('SELECT count(*) FROM receipts WHERE image_id=$1',image.id)==0
                assert await conn.fetchval('SELECT count(*) FROM receipt_drafts WHERE image_id=$1',image.id)==0
                assert await conn.fetchval('SELECT count(*) FROM images WHERE id=$1',image.id)==0
        finally:
            await blocker.execute('SELECT pg_advisory_unlock(731829)')
            delete.cancel()
            if finalize: finalize.cancel()
            await asyncio.gather(delete,*([finalize] if finalize else []),return_exceptions=True)
