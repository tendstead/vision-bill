"""HTTP endpoints for durable, local LLM benchmark runs."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from ..model.benchmark import BenchmarkCreate, BenchmarkRun, BenchmarkStatus
from ..security.dependencies import require_admin
from ..service.benchmark_service import BenchmarkService
from .helper.helper import get_benchmark_service

router = APIRouter(dependencies=[Depends(require_admin)])


class BenchmarkReevaluation(BaseModel):
    """A transient evaluation of one receipt with one model."""

    receipt_id: UUID
    model_id: str = Field(min_length=1)


def _location(run_id: UUID) -> str:
    return f"/api/v1/benchmarks/{run_id}"


@router.post("", response_model=BenchmarkRun, status_code=status.HTTP_202_ACCEPTED)
async def create_benchmark_run(
    request: BenchmarkCreate,
    response: Response,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),  # noqa: B008
) -> BenchmarkRun:
    """Queue a benchmark run and return immediately with its durable run id."""
    try:
        run = await benchmark_service.create_run(request)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    response.headers["Location"] = _location(run.id)
    return run


@router.get("", response_model=list[BenchmarkRun])
async def list_benchmark_runs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),  # noqa: B008
) -> list[BenchmarkRun]:
    """Return benchmark history, newest first."""
    runs = await benchmark_service.list_runs()
    return runs[offset : offset + limit]


@router.get("/capabilities")
async def experiment_capabilities(
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),  # noqa: B008
) -> dict[str, bool]:
    return benchmark_service.capabilities()


@router.get("/{run_id}/results")
async def individual_results(
    run_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),  # noqa: B008
) -> list[dict[str, Any]]:
    try:
        return await benchmark_service.results(run_id, limit=limit, offset=offset)
    except ValueError as error:
        raise HTTPException(404, detail=str(error)) from error


@router.get("/{run_id}", response_model=BenchmarkStatus)
async def get_benchmark_status(
    run_id: UUID,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),  # noqa: B008
) -> BenchmarkStatus:
    """Get live queue state, selected IDs, and aggregate model summaries."""
    result = await benchmark_service.get_status(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return result


@router.post("/{run_id}/reevaluate")
async def reevaluate_receipt(
    run_id: UUID,
    request: BenchmarkReevaluation,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),  # noqa: B008
) -> dict[str, Any]:
    """Evaluate one receipt transiently; no extraction output is persisted."""
    try:
        result = await benchmark_service.reevaluate(
            run_id=run_id,
            receipt_id=request.receipt_id,
            model_id=request.model_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail="Benchmark reevaluation exceeded its run timeout") from error
    return result
