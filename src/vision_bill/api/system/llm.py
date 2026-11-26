from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ...provider.llm.base import LLMProvider, ModelDiscoveryError, discover_models
from ...security.dependencies import get_current_user

router = APIRouter(tags=["LLM", "AI", "Model Evaluation"], dependencies=[Depends(get_current_user)])


@router.get("/models")
async def list_models(request: Request, response: Response, capability: Literal["vision", "text"] = "vision") -> list[dict[str, Any]]:
    provider: LLMProvider | None = getattr(request.app.state, "llm_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="LLM provider is unavailable")
    try:
        models = await discover_models(provider, capability=capability)
    except ModelDiscoveryError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    response.headers["Cache-Control"] = "no-store"
    return [m.__dict__ for m in models]
