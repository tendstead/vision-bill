"""A saved merchant is suggested on the owner's next draft, across transactions."""

from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from tests.e2e import test_workflow_db_e2e as workflow_tests
from tests.e2e.test_receipt_review_e2e import seed
from vision_bill.api.helper.helper import get_receipt_service
from vision_bill.api.merchants import router as merchants_router
from vision_bill.api.receipts import router as receipts_router
from vision_bill.model.extraction import ExtractionDraft
from vision_bill.model.review import OriginalAvailability
from vision_bill.model.workflow import AccessScope, AttemptOutcome, DraftFinalize, DraftPatch
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User
from vision_bill.service.receipt_service import ReceiptService

db = workflow_tests.db
postgres = workflow_tests.postgres
workflow_db = workflow_tests.workflow_db
pytestmark = pytest.mark.e2e


async def observe_draft(workflow, scope, printed):
    request = workflow_tests.submission()
    submission = await workflow.enqueue_submission(request, scope=scope, idempotency_key=None)
    envelope = workflow_tests.complete().envelope
    envelope.receipt = None
    envelope.draft = ExtractionDraft(merchant_name=printed, total=None)
    await workflow.finish_attempt(
        await workflow.claim_next(), AttemptOutcome(kind="partial", envelope=envelope)
    )
    return submission.submission


@pytest.mark.parametrize("save_path", ["receipt_edit", "draft_finalize"])
async def test_new_merchant_is_suggested_on_next_owner_draft(workflow_db, tmp_path, save_path):
    workflow, owner = workflow_db
    workflow._receipts._pool = workflow.pool
    printed = "WERKSTATT KORN GmbH"
    chosen = "Hofladen Korn"
    other = AccessScope(user_id=uuid4())
    async with workflow.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users(id, username, hashed_password) VALUES ($1,$2,'fixture')",
            other.user_id, str(other.user_id),
        )

    service = ReceiptService.__new__(ReceiptService)
    service._db = workflow._receipts
    app = FastAPI()
    app.include_router(merchants_router, prefix="/api/v1/merchants")
    app.include_router(receipts_router, prefix="/api/v1/receipts")
    app.dependency_overrides[get_receipt_service] = lambda: service
    actor = owner
    app.dependency_overrides[get_current_user] = lambda: User(
        id=actor.user_id, username=str(actor.user_id), is_admin=False, can_see_all=False
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async def suggestions(image):
            response = await client.post(
                "/api/v1/merchants/suggestions", json={"merchant_name": printed, "image_id": str(image.id)}
            )
            assert response.status_code == 200, response.text
            return response.json()

        if save_path == "receipt_edit":
            row, body, _ = await seed(workflow, owner, tmp_path)
            payload = body.model_dump(mode="json", exclude_unset=True)
            payload.update(merchant_name=printed, merchant_display_name=chosen)
            result = await client.put(f"/api/v1/receipts/{row.id}", json=payload)
            assert result.status_code == 200, result.text
            saved_id = row.id
        else:
            first = await workflow_tests.partial_draft(workflow, owner, tmp_path / "first.png")
            document = workflow_tests.complete().envelope.receipt.model_dump(mode="json")
            document.update(merchant_name=printed, merchant_display_name=chosen)
            await workflow.patch_draft(
                first.id, DraftPatch(expected_revision=1, changes=document), scope=owner
            )
            assert chosen not in (await suggestions(first))["merchants"]
            parts = await workflow.get_source_parts(first.id, scope=owner)
            originals = [
                OriginalAvailability(part_id=p.part_id, available=True, sha256_matches=True)
                for p in parts
            ]
            saved = await workflow.finalize_draft(
                first.id, DraftFinalize(expected_revision=2, request_key="save"),
                scope=owner, originals=originals,
            )
            saved_id = saved.id

        detail = await client.get(f"/api/v1/receipts/{saved_id}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["receipt"]["merchant_name"] == printed
        assert detail.json()["receipt"]["merchant_display_name"] == chosen

        repeat = await observe_draft(workflow, owner, printed)
        draft = await workflow.get_draft(repeat.id, scope=owner)
        assert draft.document.merchant_name == printed
        assert draft.document.merchant_display_name == chosen
        options = await suggestions(repeat)
        assert options["suggestion"] == chosen
        assert options["merchants"].count(chosen) == 1

        actor = other
        foreign = await client.post(
            "/api/v1/merchants/suggestions", json={"merchant_name": printed, "image_id": str(repeat.id)}
        )
        assert foreign.status_code == 404
        unrelated = await observe_draft(workflow, other, printed)
        other_draft = await workflow.get_draft(unrelated.id, scope=other)
        assert other_draft.document.merchant_display_name != chosen
        other_options = await suggestions(unrelated)
        assert other_options["suggestion"] != chosen
        assert chosen not in other_options["merchants"]
