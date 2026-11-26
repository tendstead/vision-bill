"""Routes consistently scope all new resources through their submission."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from vision_bill.config import Settings
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User


@pytest.fixture
def setup_api():
    from vision_bill.api.drafts import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/images")
    user = User(id=uuid4(), username="owner", is_admin=False)
    app.dependency_overrides[get_current_user] = lambda: user
    db = SimpleNamespace(
        is_ready=True,
        get_extraction=AsyncMock(return_value=None),
        get_working_view=AsyncMock(return_value=None),
        retry_submission=AsyncMock(return_value=None),
    )
    service = SimpleNamespace(
        db=db,
        settings=Settings(),
        get_draft=AsyncMock(return_value=None),
        patch_draft=AsyncMock(return_value=None),
        finalize_draft=AsyncMock(return_value=None),
    )
    from vision_bill.service.execution_policy import current_snapshot

    service.current_snapshot = lambda: current_snapshot(service.settings, inference_identity="ollama:" + "d" * 64)
    app.state.submission_service = service
    app.state.analysis_scheduler = SimpleNamespace(trigger=lambda: None)
    return app, service, user


@pytest.mark.parametrize(
    "method,suffix,body",
    [
        ("GET", "/draft", None),
        ("PATCH", "/draft", {"expected_revision": 1, "changes": {"total": None}}),
        ("POST", "/draft/finalize", {"expected_revision": 1, "request_key": "key"}),
        ("POST", "/retry", {"expected_generation": 1, "request_key": "key"}),
        ("GET", "/extractions/" + str(uuid4()), None),
        ("GET", "/views/" + str(uuid4()), None),
    ],
)
async def test_cross_user_missing_resources_are_404_and_always_scoped(
    setup_api, method, suffix, body
):
    app, service, user = setup_api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.request(
            method, "/api/v1/images/" + str(uuid4()) + suffix, json=body
        )
    assert response.status_code == 404
    calls = [
        m
        for m in (
            service.get_draft,
            service.patch_draft,
            service.finalize_draft,
            service.db.retry_submission,
            service.db.get_extraction,
            service.db.get_working_view,
        )
        if m.await_count
    ]
    assert len(calls) == 1
    assert calls[0].call_args.kwargs["scope"].user_id == user.id


async def test_stale_draft_conflict_and_incomplete_finalize(setup_api):
    app, service, _user = setup_api
    service.patch_draft.side_effect = ValueError("draft_revision_conflict")
    service.finalize_draft.side_effect = ValueError("draft_incomplete")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        path = "/api/v1/images/" + str(uuid4()) + "/draft"
        assert (
            await client.patch(path, json={"expected_revision": 1, "changes": {}})
        ).status_code == 409
        assert (
            await client.post(
                path + "/finalize", json={"expected_revision": 1, "request_key": "key"}
            )
        ).status_code == 422


@pytest.mark.parametrize("policy", ["preserve", "current"])
async def test_retry_passes_explicit_policy_and_current_snapshot_only_when_adopting(setup_api, policy):
    from vision_bill.service.execution_policy import current_snapshot

    app, service, user = setup_api
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/images/{uuid4()}/retry",
            json={"expected_generation": 1, "request_key": "key", "policy": policy},
        )
    assert response.status_code == 404
    call = service.db.retry_submission.call_args
    assert call.args[1].policy == policy
    assert call.kwargs["scope"].user_id == user.id
    assert call.kwargs["replacement_snapshot"] == (current_snapshot(service.settings, inference_identity="ollama:" + "d" * 64) if policy == "current" else None)


async def test_draft_patch_preserves_numeric_observations_and_metadata_types(setup_api):
    app, service, _user = setup_api
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.patch(
            f"/api/v1/images/{uuid4()}/draft",
            headers={"Content-Type": "application/json"},
            content="""{"expected_revision":1,"changes":{"line_items":[{"quantity":1.00000000000000001}],
                       "taxes":[{"rate":0.10000000000000001}],"overlap_decisions":[{"score":0.75}]}}""",
        )
    assert response.status_code == 404
    changes = service.patch_draft.call_args.args[1].changes
    assert changes["line_items"][0]["quantity"] == "1.00000000000000001"
    assert changes["taxes"][0]["rate"] == "0.10000000000000001"
    assert changes["overlap_decisions"][0]["score"] == 0.75


def test_legacy_draft_cash_edit_rechecks_missing_payment():
    from vision_bill.model.extraction import ExtractionDraft
    from vision_bill.model.review import AmountPolicy
    from vision_bill.model.workflow import DraftPatch
    from vision_bill.service.receipt_review import evaluate_receipt, persistent_findings

    old = ExtractionDraft(confidence=90)
    policy = AmountPolicy()
    findings = evaluate_receipt(old, policy, {}, [], []).findings
    assert any(f.field_path == "payment_method" for f in findings)
    change = DraftPatch(expected_revision=1, changes={"payment_method": "BAR", "confidence": 100})
    edited = ExtractionDraft.model_validate({**old.model_dump(), **change.changes})
    assert edited.payment_method == "cash" and edited.confidence == 100
    current = evaluate_receipt(edited, policy, {}, [], [], retained_findings=persistent_findings(findings))
    assert not any(f.field_path == "payment_method" for f in current.findings)


async def test_invalid_draft_payment_reports_field_without_value(setup_api):
    from vision_bill.model.extraction import ExtractionDraft

    app, service, _ = setup_api
    def validate(_id, patch, **kwargs):
        ExtractionDraft.model_validate(patch.changes)

    service.patch_draft.side_effect = validate
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.patch(
            f"/api/v1/images/{uuid4()}/draft",
            json={"expected_revision": 1, "changes": {"payment_method": "private-invalid-value"}},
        )
    assert response.status_code == 422
    assert response.json()["detail"] == {"code": "draft_invalid_fields", "fields": ["payment_method"]}


def test_validation_error_reports_each_invalid_row_without_private_values():
    from pydantic import ValidationError

    from vision_bill.api.drafts import mutation_error
    from vision_bill.model.extraction import ExtractionDraft

    with pytest.raises(ValidationError) as caught:
        ExtractionDraft.model_validate({
            "line_items": [{"quantity": "private quantity"}, {"unit_price": "private price"}],
            "taxes": [{"amount": -1}, {"rate": "private rate"}],
        })
    assert mutation_error(caught.value).detail == {
        "code": "draft_invalid_fields",
        "fields": ["line_items.0.quantity", "line_items.1.unit_price", "taxes.0.amount", "taxes.1.rate"],
    }


def test_validation_error_redacts_untrusted_locations():
    from pydantic import ValidationError

    from vision_bill.api.drafts import mutation_error

    locations = [
        ("private root",), ("taxes", 2, "private key"), ("line_items", -1, "quantity"),
        ("line_items", "private index", "description"), ("taxes", 0, "description"),
    ]
    error = ValidationError.from_exception_data("ExtractionDraft", [
        {"type": "string_type", "loc": loc, "input": "private text"} for loc in locations
    ])
    assert mutation_error(error).detail == {"code": "draft_invalid_fields", "fields": ["line_items", "taxes"]}
