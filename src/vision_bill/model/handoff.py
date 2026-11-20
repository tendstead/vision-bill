"""Credential-free dedication and admission contracts for external models."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ResourceIdentity = Annotated[str, Field(min_length=1)]
DispatchDisposition = Literal["not_dispatched", "settled", "unknown"]

HANDOFF_WAIT_REASONS = frozenset({
    "dedicated_ocr_activation_pending",
    "dedicated_ocr_activation_state_invalid",
    "execution_policy_changed",
    "model_release_unsupported",
    "model_handoff_binding_required",
    "model_handoff_binding_changed",
    "model_handoff_busy",
    "model_handoff_foreign_residency",
    "model_handoff_invalid_binding",
    "model_handoff_invalid_configuration",
    "model_handoff_journal_not_clean",
    "model_handoff_managed_ocr_required",
    "model_handoff_model_not_adopted",
    "model_handoff_recorded_model_not_adopted",
    "model_handoff_residency_unknown",
    "model_request_unsettled",
    "ocr_mode_mismatch",
    "ocr_unavailable",
})


class HandoffResources(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    llm: ResourceIdentity
    ocr: ResourceIdentity


class HandoffBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    coordination_path: ResourceIdentity
    resources: HandoffResources
    model_ids: list[ResourceIdentity] = Field(min_length=1)
    dedicated_coordinated_requests: Literal[True]
    external_eager_loading_disabled: Literal[True]

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("model_handoff_invalid_binding_version")
        return value

    @field_validator("dedicated_coordinated_requests", "external_eager_loading_disabled", mode="before")
    @classmethod
    def explicit_attestation(cls, value: Any) -> Any:
        if value is not True:
            raise ValueError("model_handoff_explicit_attestations_required")
        return value

    @field_validator("model_ids")
    @classmethod
    def exact_model_ids(cls, values: list[str]) -> list[str]:
        if any(value != value.strip() or not value for value in values) or len(set(values)) != len(values):
            raise ValueError("model_handoff_exact_unique_model_ids_required")
        return values


class HandoffStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: Literal["ready", "busy", "recovery_required", "configuration_required"]
    reason_code: str | None = None


class HandoffWait(RuntimeError):
    """Admission was held before new model work was dispatched."""

    def __init__(self, status: HandoffStatus):
        self.status = status
        super().__init__(status.reason_code or status.state)
