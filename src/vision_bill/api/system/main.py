from collections.abc import Callable, Coroutine
from typing import Any, Literal, Self

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from starlette.responses import Response

from ...config import (
    EDITABLE_ENV_KEYS,
    LlmAuthentication,
    LLMProviderEnum,
    SavedLlmEndpointChanged,
    deployment_llm_credential_source,
    editable_setting_sources,
    has_ambient_openai_credentials,
    has_ambient_openai_headers,
    has_legacy_llm_credentials,
    has_real_api_key,
    llm_origin,
    runtime_restart_required,
    saved_llm_settings,
    settings,
    valid_bearer_key,
    write_config_file,
)
from ...model.dedicated_ocr import (
    DedicatedOcrRequest,
    DedicatedOcrStatus,
    DedicatedOcrTopologyRequest,
)
from ...model.handoff import HandoffWait
from ...provider.llm.connection import ConnectionStatus, check_saved_connection
from ...security.dependencies import require_admin

router = APIRouter(tags=["System"])


class SettingsWriteRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        route = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await route(request)
            except RequestValidationError as exc:
                errors = exc.errors()
                if any("llm_authentication" in error["loc"] for error in errors):
                    raise HTTPException(status_code=422, detail="Invalid LLM authentication settings") from None
                detail = [{key: error[key] for key in ("loc", "msg", "type")} for error in errors]
                raise HTTPException(status_code=422, detail=detail) from None

        return handle


settings_writes = APIRouter(route_class=SettingsWriteRoute)


class LLMSettingsUpdate(BaseModel):
    provider: LLMProviderEnum
    host: str = Field(min_length=1, pattern=r"\S")
    model_name: str = Field(min_length=1, pattern=r"\S")
    temperature: float = Field(ge=0, le=2)


class LlmAuthenticationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    mode: Literal["none", "bearer"]
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if self.mode == "none" and self.api_key is None:
            return self
        if self.mode == "bearer" and self.api_key is not None and valid_bearer_key(self.api_key.get_secret_value()):
            return self
        raise ValueError("Invalid LLM authentication settings")


class LlmAuthenticationView(BaseModel):
    mode: Literal["none", "bearer"]
    source: Literal["default", "environment", "config", "settings"]
    status: Literal["none", "configured", "origin_mismatch"]


class EditableSettings(BaseModel):
    llm: LLMSettingsUpdate
    allow_registration: bool


class SettingsUpdate(EditableSettings):
    changed_paths: frozenset[str] | None = Field(default=None, exclude=True)
    llm_authentication: LlmAuthenticationUpdate | None = Field(default=None, exclude=True, repr=False)


class SettingsView(EditableSettings):
    sources: dict[str, str]
    restart_required: bool
    llm_authentication: LlmAuthenticationView


class LlmConnectionView(BaseModel):
    status: ConnectionStatus


class LlmConnectionRequest(BaseModel):
    host: str


@router.get(
    "/dedicated-ocr", response_model=DedicatedOcrStatus, dependencies=[Depends(require_admin)]
)
async def dedicated_ocr_status(request: Request) -> DedicatedOcrStatus:
    activation = getattr(request.app.state, "dedicated_ocr", None)
    if activation is None:
        return DedicatedOcrStatus(state="unavailable", reason="gpu_profile_required")
    result: DedicatedOcrStatus = await activation.status()
    return result


@router.post(
    "/dedicated-ocr", status_code=202, response_model=DedicatedOcrStatus,
    dependencies=[Depends(require_admin)],
)
async def activate_dedicated_ocr(
    request: Request, body: DedicatedOcrRequest,
) -> DedicatedOcrStatus:
    activation = getattr(request.app.state, "dedicated_ocr", None)
    if activation is None:
        raise HTTPException(status_code=409, detail="gpu_profile_required")
    result: DedicatedOcrStatus = await activation.submit(
        confirm_handoff=body.confirm_handoff, enabled=body.enabled,
    )
    if result.state in {"unavailable", "action_required"}:
        raise HTTPException(status_code=409, detail=result.reason)
    return result


@router.put(
    "/dedicated-ocr/gpu-topology", response_model=DedicatedOcrStatus,
    dependencies=[Depends(require_admin)],
)
async def set_dedicated_ocr_gpu_topology(
    request: Request, body: DedicatedOcrTopologyRequest,
) -> DedicatedOcrStatus:
    activation = getattr(request.app.state, "dedicated_ocr", None)
    if activation is None:
        raise HTTPException(status_code=409, detail="gpu_profile_required")
    try:
        result: DedicatedOcrStatus = await activation.set_gpu_topology(
            body.shared_gpu, confirm_handoff=body.confirm_handoff,
        )
    except HandoffWait as exc:
        raise HTTPException(status_code=409, detail=exc.status.reason_code) from None
    return result


@router.get("/ui-config")
async def get_ui_config() -> dict[str, bool | int]:
    """Return safe runtime defaults needed by the browser UI (public, pre-auth)."""
    return {
        "bypass_review_default": settings.images.bypass_review_default,
        "registration_open": settings.auth.allow_registration,
        "ocr_enabled": settings.ocr.enabled,
        "ocr_required": settings.ocr.required,
        "ocr_allow_vision_fallback": settings.ocr.allow_vision_fallback,
        "formatter_max_output_tokens": settings.llm.max_output_tokens,
    }


def _authentication_view() -> LlmAuthenticationView:
    llm = settings.llm
    auth = llm.authentication
    if auth is None:
        if has_ambient_openai_credentials():
            return LlmAuthenticationView(mode="bearer", source="environment", status="configured")
        if has_real_api_key(llm.api_key):
            return LlmAuthenticationView(mode="bearer", source=deployment_llm_credential_source(), status="configured")
        return LlmAuthenticationView(mode="none", source="default", status="none")
    if auth.mode == "none":
        return LlmAuthenticationView(mode="none", source="settings", status="none")
    status: Literal["configured", "origin_mismatch"] = (
        "configured" if llm.request_api_key() is not None else "origin_mismatch"
    )
    return LlmAuthenticationView(mode="bearer", source="settings", status=status)


def _settings_view() -> SettingsView:
    return SettingsView(
        llm=LLMSettingsUpdate(
            provider=settings.llm.provider,
            host=settings.llm.host,
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
        ),
        allow_registration=settings.auth.allow_registration,
        sources=editable_setting_sources(),
        restart_required=runtime_restart_required(settings),
        llm_authentication=_authentication_view(),
    )


@router.get("/settings", response_model=SettingsView, dependencies=[Depends(require_admin)])
async def get_settings() -> SettingsView:
    """Return editable settings without exposing credentials or secrets."""
    return _settings_view()


@router.post(
    "/settings/llm/connection-test", response_model=LlmConnectionView, dependencies=[Depends(require_admin)],
)
async def test_llm_connection(request: LlmConnectionRequest) -> LlmConnectionView:
    try:
        llm = saved_llm_settings()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Could not read saved LLM settings") from None
    if request.host != llm.host:
        raise HTTPException(status_code=409, detail="The saved connection changed. Reload settings before testing.")
    return LlmConnectionView(status=await check_saved_connection(llm))


def _validated_llm_origin(host: str) -> str:
    try:
        parsed = TypeAdapter(AnyHttpUrl).validate_python(host)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Host must be an HTTP(S) URL") from None
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(status_code=422, detail="Host must not contain credentials")
    try:
        return llm_origin(host)
    except ValueError:
        raise HTTPException(status_code=422, detail="Host must be an HTTP(S) URL without credentials") from None


@settings_writes.put("/settings", response_model=SettingsView, dependencies=[Depends(require_admin)])
async def update_settings(request: Request, update: SettingsUpdate) -> SettingsView:
    """Persist admin-editable settings and apply safe changes immediately."""
    requested = {
        "llm.provider": update.llm.provider,
        "llm.host": update.llm.host,
        "llm.model_name": update.llm.model_name,
        "llm.temperature": update.llm.temperature,
        "auth.allow_registration": update.allow_registration,
    }
    current = {
        "llm.provider": settings.llm.provider,
        "llm.host": settings.llm.host,
        "llm.model_name": settings.llm.model_name,
        "llm.temperature": settings.llm.temperature,
        "auth.allow_registration": settings.auth.allow_registration,
    }
    selected = update.changed_paths if update.changed_paths is not None else frozenset(requested)
    if not selected <= EDITABLE_ENV_KEYS.keys():
        raise HTTPException(status_code=422, detail="Unsupported setting path")
    changed = frozenset(field for field in selected if requested[field] != current[field])
    sources = editable_setting_sources()
    if update.changed_paths is None:
        locked = sorted(field for field in changed if sources[field] == "environment")
        if locked:
            raise HTTPException(
                status_code=409,
                detail="These settings are controlled by environment variables: " + ", ".join(locked),
            )
    if "auth.allow_registration" in changed and sources["auth.allow_registration"] == "environment":
        raise HTTPException(status_code=409, detail="Registration is controlled by environment configuration")
    candidate = settings.model_copy(deep=True)
    for field in changed:
        section, key = field.split(".", 1)
        setattr(getattr(candidate, section), key, requested[field])

    authentication = update.llm_authentication
    credential_origin = None
    llm_authentication_expected_origin = None
    candidate_origin = None
    if "llm.host" in changed:
        candidate_origin = _validated_llm_origin(candidate.llm.host)
        if (
            has_ambient_openai_headers() and not has_real_api_key(settings.llm.api_key)
            and llm_origin(settings.llm.host) != candidate_origin
        ):
            # Previous releases cannot bind custom header credentials on their own.
            raise HTTPException(
                status_code=409,
                detail="Remove the deployment's custom OpenAI headers before changing the endpoint origin.",
            )
        if authentication is None:
            existing = settings.llm.authentication
            if existing is None:
                try:
                    credential_origin = llm_origin(settings.llm.host)
                except ValueError:
                    raise HTTPException(status_code=409, detail="Select authentication for the new endpoint") from None
                protected_origin = credential_origin if has_legacy_llm_credentials(settings.llm.api_key) else None
            else:
                protected_origin = existing.origin
            if protected_origin is not None and candidate_origin != protected_origin:
                raise HTTPException(
                    status_code=409,
                    detail="Endpoint origin is bound to private credentials. Select a new key or None for the new endpoint.",
                )
    if authentication is not None:
        if candidate.llm.provider != LLMProviderEnum.OPENAI:
            raise HTTPException(status_code=422, detail="LLM authentication settings require the OpenAI provider")
        origin = candidate_origin or _validated_llm_origin(candidate.llm.host)
        if origin != _validated_llm_origin(update.llm.host):
            raise HTTPException(status_code=409, detail="The endpoint changed. Reload settings before changing authentication.")
        llm_authentication_expected_origin = llm_origin(settings.llm.host)
        candidate.llm.authentication = LlmAuthentication(
            mode=authentication.mode, api_key=authentication.api_key,
            origin=origin if authentication.mode == "bearer" else None,
        )
    try:
        write_config_file(
            candidate,
            changed_paths=changed,
            credential_origin=credential_origin,
            llm_authentication_expected_origin=llm_authentication_expected_origin,
        )
    except SavedLlmEndpointChanged:
        raise HTTPException(status_code=409, detail="The endpoint changed. Reload settings before changing authentication.") from None
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Could not write application config") from exc

    settings.llm.provider = candidate.llm.provider
    settings.llm.host = candidate.llm.host
    settings.llm.authentication = candidate.llm.authentication
    settings.llm.model_name = candidate.llm.model_name
    settings.llm.temperature = candidate.llm.temperature
    settings.auth.allow_registration = candidate.auth.allow_registration
    provider = getattr(request.app.state, "llm_provider", None)
    if provider is not None:
        provider.update_runtime_settings(temperature=settings.llm.temperature)
    return _settings_view()


router.include_router(settings_writes)


# @router.post("/compare-models")
# async def compare_models_endpoint(prompt: str, models: list[str] | None = None):
#    results = await compare_models(prompt, models)
#    return {"results": [r.__dict__ for r in results]}
