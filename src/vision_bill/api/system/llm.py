from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request

from ...provider.llm.base import LLMProvider
from ...security.dependencies import get_current_user

router = APIRouter(tags=["LLM", "AI", "Model Evaluation"], dependencies=[Depends(get_current_user)])


@router.get("/models")
async def list_models(request: Request, capability: Literal["vision", "text"] = "vision") -> list[dict[str, Any]]:
    provider: LLMProvider | None = getattr(request.app.state, "llm_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="LLM provider is unavailable")
    models = await provider.get_available_models(capability=capability)
    return [m.__dict__ for m in models]
