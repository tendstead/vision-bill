from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool


class DedicatedOcrStatus(BaseModel):
    state: Literal[
        "unavailable", "disabled", "requested", "installing", "active", "failed", "action_required"
    ]
    reason: str | None = None
    installed: bool = False
    selected: bool = False
    shared_gpu: bool = True


class DedicatedOcrRequest(BaseModel):
    """Activation takes its configuration from the selected GPU profile."""

    model_config = ConfigDict(extra="forbid")
    confirm_handoff: StrictBool = False
    enabled: StrictBool = True


class DedicatedOcrTopologyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shared_gpu: StrictBool
    confirm_handoff: StrictBool = False
