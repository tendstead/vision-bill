from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from vision_bill.api.benchmarks import router
from vision_bill.api.helper.helper import get_benchmark_service
from vision_bill.model.benchmark import BenchmarkCreate
from vision_bill.model.ocr_errors import OcrWorkerFailure
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User
from vision_bill.service.experiment_execution import ExperimentFailure


@pytest.mark.parametrize('method,suffix,name,body', [
    ('GET', '', 'list_runs', None),
    ('GET', '/references/{id}', 'reference_preview', None),
    ('POST', '/references/{id}', 'qualify_reference', {'expected_review_revision': 1, 'expected_document_sha256': 'a'*64}),
    ('GET', '/{id}', 'get_status', None),
    ('GET', '/{id}/results', 'results', None),
    ('POST', '/{id}/reevaluate', 'reevaluate', {'receipt_id': str(uuid4()), 'model_id': 'm'}),
    ('POST', '', 'create_run', {'model_ids': ['m']}),
])
async def test_every_benchmark_read_write_receives_real_admin_scope(method, suffix, name, body):
    user = User(id=uuid4(), username='restricted-admin', is_admin=True, can_see_all=False)
    app = FastAPI()
    app.include_router(router, prefix='/benchmarks')
    app.dependency_overrides[get_current_user] = lambda: user
    service = SimpleNamespace(**{name: AsyncMock(side_effect=ValueError('test boundary'))})
    app.dependency_overrides[get_benchmark_service] = lambda: service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://test') as client:
        await client.request(method, '/benchmarks' + suffix.replace('{id}', str(uuid4())), json=body)
    call = getattr(service, name)
    assert call.await_count == 1
    scope = call.call_args.kwargs['scope']
    assert scope.user_id == user.id and scope.can_see_all is False


@pytest.mark.parametrize('message', ['Explicit receipt is unavailable or not verified', 'No current qualified references match the selection'])
async def test_create_rejects_ineligible_selection_as_unprocessable(message):
    user = User(id=uuid4(), username='restricted-admin', is_admin=True, can_see_all=False)
    app = FastAPI()
    app.include_router(router, prefix='/benchmarks')
    app.dependency_overrides[get_current_user] = lambda: user
    service = SimpleNamespace(create_run=AsyncMock(side_effect=ValueError(message)))
    app.dependency_overrides[get_benchmark_service] = lambda: service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(
            '/benchmarks', json={'model_ids': ['m'], 'receipt_ids': [str(uuid4()), str(uuid4())]}
        )
    assert response.status_code == 422 and response.json()['detail'] == message
    assert len(service.create_run.call_args.args[0].receipt_ids) == 2
    assert service.create_run.call_args.kwargs['scope'].user_id == user.id


def test_benchmark_selection_deduplicates_receipts_in_first_seen_order():
    first, second = uuid4(), uuid4()
    request = BenchmarkCreate(model_ids=["m"], receipt_ids=[second, first, second])
    assert request.receipt_ids == [second, first]


@pytest.mark.parametrize("method, error, status", [
    ("create_run", OcrWorkerFailure("ocr_model_load_failed"), 503),
    ("reevaluate", ExperimentFailure("ocr_invalid_result", {}), 502),
    ("reevaluate", ExperimentFailure("ocr_deadline_exceeded", {}), 504),
    ("reevaluate", ExperimentFailure("ocr_gpu_out_of_memory", {}), 503),
    ("reevaluate", ValueError("Receipt/model is not part of this benchmark run"), 404),
])
async def test_benchmark_route_preserves_worker_failures(method, error, status):
    app = FastAPI()
    app.include_router(router, prefix="/benchmarks")
    user = User(id=uuid4(), username="admin", is_admin=True, can_see_all=False)
    app.dependency_overrides[get_current_user] = lambda: user
    service = SimpleNamespace(**{method: AsyncMock(side_effect=error)})
    app.dependency_overrides[get_benchmark_service] = lambda: service
    url, body = "/benchmarks", {"model_ids": ["m"]}
    if method == "reevaluate":
        url, body = f"/benchmarks/{uuid4()}/reevaluate", {"receipt_id": str(uuid4()), "model_id": "m"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
        response = await client.post(url, json=body)
    assert response.status_code == status
    assert response.json()["detail"] == str(error)
