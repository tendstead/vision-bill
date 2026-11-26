"""Durable readiness gates precede claims; retries wake at their due time."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from vision_bill.model.db.image import ImageRow
from vision_bill.model.workflow import CommitResult, SubmissionStatus
from vision_bill.service.analysis_scheduler import AnalysisScheduler


def context(settings):
    image = ImageRow(id=uuid4())
    claim = SimpleNamespace(image=image)
    db = SimpleNamespace(
        is_ready=True,
        reap_expired_claims=AsyncMock(return_value=0),
        due_candidates=AsyncMock(return_value=[image]),
        claim_next=AsyncMock(return_value=claim),
        next_wake_delay=AsyncMock(return_value=0.01),
    )
    receipts = SimpleNamespace(workflow_db=db)
    service = SimpleNamespace(
        ready_for=AsyncMock(return_value=True),
        process_claim=AsyncMock(
            return_value=CommitResult(
                stale=False,
                submission=SubmissionStatus(id=image.id, status="pending", result_code="timed_out"),
            )
        ),
    )
    scheduler = AnalysisScheduler(settings, None, receipts, None, submission_service=service)
    return scheduler, db, service, image


async def test_provider_down_waits_before_claim_without_attempt(settings):
    scheduler, db, service, _image = context(settings)
    service.ready_for.return_value = False
    assert await scheduler.process_pending() == []
    db.claim_next.assert_not_awaited()
    service.process_claim.assert_not_awaited()


async def test_pending_backoff_is_nonterminal_and_fenced_claim_used(settings):
    scheduler, db, service, image = context(settings)
    result = await scheduler.process_pending()
    assert result[0].status == "pending" and result[0].error == "timed_out"
    db.claim_next.assert_awaited_once_with(lease_seconds=900, image_id=image.id)
    service.process_claim.assert_awaited_once_with(db.claim_next.return_value)


async def test_earliest_retry_wakes_without_waiting_300_seconds(settings):
    scheduler, db, _service, _image = context(settings)
    done = asyncio.Event()
    count = 0

    async def scan():
        nonlocal count
        count += 1
        if count > 1:
            done.set()
        return []

    scheduler.process_pending = scan
    await scheduler.start()
    try:
        async with asyncio.timeout(0.3):
            await done.wait()
    finally:
        await scheduler.stop()
    assert count >= 2
    db.next_wake_delay.assert_awaited()


async def test_wake_trigger_and_stop_are_idempotent(settings):
    scheduler, db, _service, _image = context(settings)
    db.next_wake_delay.return_value = 300
    await scheduler.start()
    task = scheduler._task
    await scheduler.start()
    assert scheduler._task is task
    await asyncio.sleep(0)
    scheduler.trigger()
    await asyncio.sleep(0.01)
    await scheduler.stop()
    await scheduler.stop()
    assert scheduler._task is None


async def test_database_unavailable_never_claims(settings):
    scheduler, db, _service, _image = context(settings)
    db.is_ready = False
    assert await scheduler.process_pending() == []
    db.reap_expired_claims.assert_not_awaited()
    db.due_candidates.assert_not_awaited()


async def test_retry_due_during_long_second_job_rescans_without_provider_down_hot_loop(
    settings, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    import vision_bill.service.analysis_scheduler as module

    scheduler, db, service, first = context(settings)
    second = ImageRow(id=uuid4())
    clock = [0.0]
    due = {first.id: 0.0, second.id: 0.0}
    runs = []
    waits = []
    done = asyncio.Event()
    ready = {first.id: True, second.id: True}

    async def candidates(*, purpose=None):
        return [image for image in [first, second] if due.get(image.id, 999) <= clock[0]]

    async def claim_next(*, image_id, lease_seconds):
        return SimpleNamespace(image=first if image_id == first.id else second)

    async def process(claim):
        runs.append((claim.image.id, clock[0]))
        if claim.image.id == first.id and len(runs) == 1:
            due[first.id] = clock[0] + 30
            return CommitResult(
                stale=False,
                submission=SubmissionStatus(
                    id=first.id,
                    status="pending",
                    retry_at=datetime.now(UTC) + timedelta(seconds=30),
                ),
            )
        if claim.image.id == second.id:
            clock[0] += 60
            del due[second.id]
        else:
            ready[first.id] = False
            due[first.id] = clock[0]  # another overdue retry, but provider then goes down
        return CommitResult(
            stale=False, submission=SubmissionStatus(id=claim.image.id, status="pending")
        )

    async def wake_delay(maximum, *, deferred_image_ids=()):
        eligible = [
            when - clock[0] for image_id, when in due.items() if image_id not in deferred_image_ids
        ]
        return min(maximum, max(0, min(eligible))) if eligible else maximum

    async def wait_for(awaitable, *, timeout):
        waits.append(timeout)
        awaitable.close()
        if len(waits) >= 4:
            done.set()
            raise asyncio.CancelledError()
        if timeout:
            clock[0] += timeout if len(waits) > 1 else 0
        raise TimeoutError()

    db.due_candidates.side_effect = candidates
    db.claim_next.side_effect = claim_next
    db.next_wake_delay.side_effect = wake_delay
    service.ready_for.side_effect = lambda image: ready[image.id]
    service.process_claim.side_effect = process
    monkeypatch.setattr(module.asyncio, "wait_for", wait_for)
    task = asyncio.create_task(scheduler._run_forever())
    await done.wait()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert runs == [(first.id, 0), (second.id, 0), (first.id, 60)]
    assert db.claim_next.await_count == 3
    assert waits == [300, 0, 0, 300]


@pytest.mark.parametrize("worker_ready", [False, True], ids=["parked", "ready"])
@pytest.mark.parametrize("managed_lifecycle", [False, True], ids=["unmanaged", "managed"])
@pytest.mark.parametrize("attempt_count", [0, 2], ids=["new", "last-attempt"])
async def test_incompatible_ocr_mode_waits_before_claim_and_preserves_attempt_budget(
    settings, monkeypatch, tmp_path, worker_ready, managed_lifecycle, attempt_count,
):
    from test_ocr_provider import http_client

    from vision_bill import config
    from vision_bill.config import OcrSettings
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.execution_policy import current_snapshot
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.submission_service import SubmissionService

    settings.ocr = OcrSettings(
        enabled=True, service_url="http://local", manage_model_lifecycle=managed_lifecycle,
        coordination_path=tmp_path / "handoff.lock",
    )
    # Enable the real global managed-ownership guard when this client requires it.
    monkeypatch.setattr(config, "settings", settings)
    image = ImageRow(
        id=uuid4(), model_id="formatter", attempt_count=attempt_count,
        effective_options=current_snapshot(settings, inference_identity="ollama:" + "d" * 64),
    )
    model = ModelInfo("formatter")
    model.capabilities = ["completion"]
    formatter = SimpleNamespace(
        inference_identity="ollama:" + "d" * 64,
        supports_model_release=True,
        lifecycle_identity="cpu-fixture:formatter",
        resident_model_ids=AsyncMock(return_value=frozenset()),
        check_connection=AsyncMock(return_value=True),
        get_available_models=AsyncMock(return_value=[model]),
    )

    async def persist_wait(candidate, reason):
        candidate.wait_reason = reason
        return True

    async def claim_next(**kwargs):
        # Mirror the claim's budget increment so mistaken admission is observable.
        image.attempt_count += 1
        return SimpleNamespace(image=image)

    db = SimpleNamespace(
        is_ready=True,
        reap_expired_claims=AsyncMock(return_value=0),
        due_candidates=AsyncMock(return_value=[image]),
        set_wait_reason=AsyncMock(side_effect=persist_wait),
        claim_next=AsyncMock(side_effect=claim_next),
    )
    receipts = ReceiptService(settings.images, settings.pg, formatter)
    if managed_lifecycle:
        from test_handoff_binding import adopt

        await adopt(receipts.execution_handoff(settings.ocr), ("formatter",))
        image.effective_options = receipts.current_execution_snapshot(settings)
    receipts._workflow_db = db
    submissions = SubmissionService(settings, receipts, receipts._image_service)
    submissions.process_claim = AsyncMock(return_value=CommitResult(stale=False))
    scheduler = AnalysisScheduler(
        settings, formatter, receipts, None, submission_service=submissions,
    )
    requests = []
    worker = {
        "ready": worker_ready, "generation": 1,
        "profile_id": "fixed-profile" if worker_ready else None,
        "reason_code": None if worker_ready else "ocr_parked",
    }

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "contract_version": 2, "profile": {"max_new_tokens": 4096},
                "worker": worker,
                "service_mode": "standalone" if managed_lifecycle else "managed",
            })
        assert request.url.path == "/ready"
        return httpx.Response(200 if worker_ready else 503, json=worker)

    http_client(monkeypatch, handler)
    for _ in range(image.max_attempts + 1):
        assert await scheduler.process_pending() == []
    db.claim_next.assert_not_awaited()
    submissions.process_claim.assert_not_awaited()
    assert image.attempt_count == attempt_count
    assert image.status == "pending" and image.wait_reason == "ocr_mode_mismatch"
    assert db.set_wait_reason.await_count == image.max_attempts + 1
    for call in db.set_wait_reason.await_args_list:
        assert call.args == (image, "ocr_mode_mismatch")
    assert requests == [("GET", "/info")] * (image.max_attempts + 1)
    formatter.check_connection.assert_awaited()


async def test_saved_observations_dispatch_only_completion(settings):
    scheduler, db, service, image = context(settings)
    image.observation_checkpoint = {"eligible_generation": image.generation}
    db.due_candidates.side_effect = lambda *, purpose="inference": (
        [image] if purpose in (None, "completion") else []
    )
    service.ready_for_completion = AsyncMock(return_value=True)
    service.complete_checkpoint = AsyncMock(
        return_value=CommitResult(
            stale=False,
            submission=SubmissionStatus(id=image.id, status="analyzed", attempt_count=3),
        )
    )
    assert [result.status for result in await scheduler.process_pending()] == ["analyzed"]
    db.claim_next.assert_awaited_once_with(
        lease_seconds=900, image_id=image.id, purpose="completion"
    )
    service.ready_for.assert_not_awaited()
    service.process_claim.assert_not_awaited()
