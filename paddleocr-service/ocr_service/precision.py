"""Sealed static-artifact checks and bounded, metadata-only predictor hook audits.

A manifest is build evidence, never runtime success. Each session must execute
its actual compiled predictor after hooks are installed before finish can pass.
"""

import re
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .profile import content_identity, file_digest
from .vlm_precision import floating_dtype

# Paddle 3.3.1 on Linux x86_64 emits a pointer followed by epoch nanoseconds.
_PIR_VARIABLE = re.compile(r"(0x[0-9a-f]{1,16}[1-9][0-9]{18})_inner_var_(0|[1-9][0-9]*)")
_PIR_NAME_PREFIX = "pir-temporary:"


class HookVariableNames:
    """Keep Paddle's variable ordinal while removing its process-local scope prefix."""

    def __init__(self, parameter_names: Collection[str]):
        self.parameter_names = frozenset(parameter_names)
        if any(name.startswith(_PIR_NAME_PREFIX) for name in self.parameter_names):
            raise ValueError("Static parameter uses the reserved PIR identity prefix")
        self.scope: str | None = None

    def canonical(self, name: str) -> str:
        if name.startswith(_PIR_NAME_PREFIX):
            raise ValueError("Native variable uses the reserved PIR identity prefix")
        if name in self.parameter_names:
            return name
        generated = _PIR_VARIABLE.fullmatch(name)
        if generated is None:
            return name
        scope, ordinal = generated.groups()
        # The ordinal is scoped to the interpreter. Bind its scope for the
        # whole predictor lifetime and reject any ambiguous remapping.
        if self.scope is not None and self.scope != scope:
            raise ValueError("Static forward uses multiple generated PIR scopes")
        self.scope = scope
        return f"{_PIR_NAME_PREFIX}{ordinal}"


class ParameterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    shape: list[int]
    dtype: str


class HookRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hook: Literal["input", "output"]
    op: str
    name: str
    dtype: Literal["float32"]
    place: str


class StaticArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_name: str
    graph_file: str
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    params_file: str
    params_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inspector_identity: str = Field(min_length=1)
    inspector_version: str = Field(min_length=1)
    parameters: list[ParameterRecord] = Field(min_length=1)
    # Exported parameter -> actual compiled input-hook variable, established by
    # the real image qualifier. No speculative op-name or CPU allowlists.
    parameter_bindings: dict[str, str]
    coverage: list[HookRecord] = Field(min_length=2)

    @model_validator(mode="after")
    def require_complete_inventory(self) -> "StaticArtifact":
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate static parameter inventory")
        floating = {p.name for p in self.parameters if floating_dtype(p.dtype) is not None}
        if not floating or any(
            floating_dtype(p.dtype) not in (None, "float32") for p in self.parameters
        ):
            raise ValueError("Static artifact requires a nonempty FP32 floating inventory")
        if set(self.parameter_bindings) != floating:
            raise ValueError("Every floating parameter needs a compiled hook binding")
        if len(set(self.parameter_bindings.values())) != len(floating):
            raise ValueError("Compiled parameter bindings must be unambiguous")
        return self


def validate_artifact(artifact: StaticArtifact, model_dir: Path, inspector_identity: str) -> None:
    if artifact.inspector_identity != inspector_identity:
        raise ValueError("Static artifact inspector source identity mismatch")
    for filename, expected_hash in (
        (artifact.graph_file, artifact.graph_sha256),
        (artifact.params_file, artifact.params_sha256),
    ):
        path = model_dir / filename
        if Path(filename).name != filename or not path.is_file():
            raise ValueError("Static artifact must refer to a local sealed model file")
        if file_digest(path) != expected_hash:
            raise ValueError("Sealed static artifact hash mismatch")


class StaticAuditSession:
    def __init__(self, artifact: StaticArtifact, gpu_index: int):
        self.artifact = artifact
        self.gpu_index = gpu_index
        self.enabled = False
        self.counts = {"input": 0, "output": 0}
        self.observed: set[tuple[str, str, str, str, str]] = set()
        self.violations: set[str] = set()
        self.names = HookVariableNames(
            {param.name for param in artifact.parameters} | set(artifact.parameter_bindings.values())
        )
        self.expected = {tuple(item.model_dump().values()) for item in artifact.coverage}
        required = set(artifact.parameter_bindings.values())
        declared = {
            record.name
            for record in artifact.coverage
            if record.hook == "input" and record.place == f"gpu:{gpu_index}"
        }
        if not required <= declared or not any(
            r.hook == "output" and r.place == f"gpu:{gpu_index}" for r in artifact.coverage
        ):
            raise ValueError("Static coverage must include every GPU weight and GPU output")

    def start(self) -> None:
        self.observed.clear()
        self.violations.clear()
        self.counts = {"input": 0, "output": 0}
        self.enabled = True

    def _observe(self, hook: str, op: str, name: str, tensor: Any) -> None:
        if not self.enabled:
            return
        self.counts[hook] += 1
        try:
            dtype = floating_dtype(tensor.dtype)
            if dtype is None:
                return  # CPU integer shape tensors do not imply float fallback.
            place = tensor.place
            if place.is_gpu_place():
                if place.gpu_device_id() != self.gpu_index:
                    self.violations.add("Floating tensor uses another GPU device index")
                    return
                location = f"gpu:{self.gpu_index}"
            elif place.is_cpu_place():
                location = "cpu"
            else:
                self.violations.add("Floating tensor uses an unsupported device place")
                return
            record = (hook, str(op), self.names.canonical(str(name)), dtype, location)
            if dtype != "float32" or record not in self.expected:
                # Bound memory even if a native graph emits an unbounded stream.
                self.violations.add("Unexpected floating precision/device/coverage")
                return
            self.observed.add(record)
        except Exception:  # noqa: BLE001 -- callbacks must retain violations across native bindings
            self.violations.add("Unable to inspect native hook tensor metadata")

    def observe_input(self, op: str, name: str, tensor: Any) -> None:
        self._observe("input", op, name, tensor)

    def observe_output(self, op: str, name: str, tensor: Any) -> None:
        self._observe("output", op, name, tensor)

    def finish(self) -> dict[str, Any]:
        self.enabled = False
        if self.violations or self.observed != self.expected or not all(self.counts.values()):
            raise ValueError("Static runtime coverage incomplete or precision/device violation")
        return {
            "model_name": self.artifact.model_name,
            "input_count": self.counts["input"],
            "output_count": self.counts["output"],
            "floating_parameters": len(self.artifact.parameter_bindings),
            "coverage_digest": content_identity(sorted(self.observed)),
            "graph_sha256": self.artifact.graph_sha256,
            "params_sha256": self.artifact.params_sha256,
        }


def begin_static_audit(
    model: Any,
    *,
    expected_model_name: str,
    gpu_index: int,
    sealed_artifact: StaticArtifact,
    runner_type: type[Any],
    inspector_identity: str,
) -> StaticAuditSession:
    runner = model.runner
    if not isinstance(runner, runner_type):
        raise TypeError("Enabled model must have the pinned PaddleStaticRunner")
    expected = {
        "run_mode": "paddle_fp32",
        "device_type": "gpu",
        "device_id": gpu_index,
        "enable_cinn": False,
    }
    if any(runner._config.get(key) != value for key, value in expected.items()):
        raise ValueError("Static resolved runner must use native GPU FP32 with CINN disabled")
    if (
        runner._model_name != expected_model_name
        or sealed_artifact.model_name != expected_model_name
    ):
        raise ValueError("Static model identity mismatch")
    expected_graph = f"{runner.model_file_prefix}.json"
    if not (runner.model_dir / expected_graph).is_file():
        expected_graph = f"{runner.model_file_prefix}.pdmodel"
    if (
        sealed_artifact.graph_file != expected_graph
        or sealed_artifact.params_file != f"{runner.model_file_prefix}.pdiparams"
    ):
        raise ValueError("Manifest does not describe the resolved native model artifacts")
    validate_artifact(sealed_artifact, Path(runner.model_dir), inspector_identity)
    session = StaticAuditSession(sealed_artifact, gpu_index)
    # Native bindings have no unregister API. Register exactly once, then toggle.
    runner.predictor.register_input_hook(session.observe_input)
    runner.predictor.register_output_hook(session.observe_output)
    return session
