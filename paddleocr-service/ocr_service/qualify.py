"""Executable native release gate. Run in the built image on physical hardware.

No framework import occurs in the coordinator. Inspection, collection and replay
run in separate bounded subprocesses; stdout/stderr never expose fixture content.
"""

import argparse
import importlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from ocr_contract import GpuMemorySnapshot, OcrResponse
from pydantic import BaseModel, ConfigDict, Field

from .bootstrap import (
    HookCollectionError,
    HookCollector,
    HookFailureReason,
    common_coverage,
    inspect_artifact,
    parameter_bindings,
    seal_dependencies,
)
from .device_inventory import execution_target, inspect_device_identity
from .fp32 import audit_dynamic_model, floating_dtype, require_gpu
from .gpu_memory import GPU_OOM_CODE, GPU_OOM_EXIT, GpuOutOfMemory, is_gpu_out_of_memory
from .images import normalize_source
from .pipeline import PipelineAdapter, build_pipeline, construct_native, tree_identity
from .precision import ParameterRecord, StaticArtifact
from .profile import (
    MODEL_NAMES,
    NativeRuntimeSettings,
    RuntimeSettings,
    file_digest,
    load_profile,
    verify_dependencies,
)
from .supervision import qualification_is_supervised

PROFILE = Path(__file__).resolve().parents[1] / "profile.json"
KINDS = {"upright", "rotated", "warped", "table", "formula"}
STATIC_HOOK_FAILURE_CODE = "static_hook_collection_failed"
_STATIC_HOOK_FAILURE_EXIT = 43
COLLECTION_WAIT_TIMEOUT_EXIT = 44
_FAILURE_REPORT_BYTES = 16 * 1024


class QualificationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["upright", "rotated", "warped", "table", "formula"]
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Require recognized structure for table/formula cases, without logging text.
    required_labels: list[str] = Field(default_factory=list)


class StaticHookFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    model_name: Literal["PP-DocLayoutV3", "PP-LCNet_x1_0_doc_ori", "UVDoc"]
    reason_code: HookFailureReason
    input_count: int = Field(ge=0, le=2**63 - 1)
    output_count: int = Field(ge=0, le=2**63 - 1)


class StaticHookCollectionFailed(ValueError):
    def __init__(self, diagnostic: StaticHookFailure):
        super().__init__(STATIC_HOOK_FAILURE_CODE)
        self.diagnostic = diagnostic


class NativeStageWaitTimeout(TimeoutError):
    def __init__(self, stage: str):
        super().__init__("Native qualification stage exceeded 600 seconds")
        self.stage = stage


def _read_failure_report(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size > _FAILURE_REPORT_BYTES
            ):
                return None
            payload = stream.read(info.st_size + 1)
        if len(payload) != info.st_size:
            return None
        data = json.loads(payload)
        return data if isinstance(data, dict) and data.get("qualified") is False else None
    except (OSError, ValueError, RecursionError):
        return None


def read_static_hook_failure(path: Path) -> StaticHookFailure | None:
    """Read only a small failure envelope, never raw native diagnostics."""
    report = _read_failure_report(path)
    if (
        report is None
        or report.get("stage") != "native_collection"
        or report.get("failure") != STATIC_HOOK_FAILURE_CODE
    ):
        return None
    try:
        return StaticHookFailure.model_validate(report.get("static_hook_failure"))
    except ValueError:
        return None


def is_collection_wait_timeout(path: Path, *, accelerator: str, image_identity: str) -> bool:
    report = _read_failure_report(path)
    return (
        report is not None
        and report.get("stage") == "native_collection"
        and report.get("failure") == "TimeoutError"
        and report.get("collection_wait_timed_out") is True
        and report.get("accelerator") == accelerator
        and report.get("image_identity") == image_identity
    )


class QualificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qualified: bool = False
    accelerator: str
    image_identity: str
    device: dict[str, Any] = Field(default_factory=dict)
    preload_memory: GpuMemorySnapshot | None = None
    profile: dict[str, Any] = Field(default_factory=dict)
    model_content: dict[str, str] = Field(default_factory=dict)
    profile_id: str | None = None
    runtime_fingerprint: str | None = None
    manifest_sha256: str | None = None
    cases: list[dict[str, Any]] = Field(default_factory=list)
    qualified_targets: list[str] = Field(default_factory=list)
    failure: str | None = None
    static_hook_failure: StaticHookFailure | None = None
    collection_wait_timed_out: bool = False
    stage: str = "inputs"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        json.dump(value, target, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")
    temporary.replace(path)


def load_cases(path: Path) -> list[QualificationCase]:
    raw = tomllib.loads(path.read_text())
    cases = [QualificationCase.model_validate(row) for row in raw["cases"]]
    if len(cases) != 5 or {case.kind for case in cases} != KINDS:
        raise ValueError("Exactly one real case of every required kind is needed")
    for case in cases:
        resolved = (path.parent / case.path).resolve()
        if not resolved.is_file() or file_digest(resolved) != case.sha256:
            raise ValueError("Qualification input hash mismatch")
        if case.kind in {"table", "formula"} and case.kind not in case.required_labels:
            raise ValueError("Table/formula fixtures require matching recognized labels")
    if len({case.sha256 for case in cases}) != len(cases):
        raise ValueError("Qualification cases must be distinct real inputs")
    return [case.model_copy(update={"path": (path.parent / case.path).resolve()}) for case in cases]


def has_required_labels(case: QualificationCase, response: OcrResponse) -> bool:
    labels = {block.label for block in response.blocks}
    # The native layout model distinguishes display and inline formulas.
    if labels.intersection({"display_formula", "inline_formula"}):
        labels.add("formula")
    return set(case.required_labels) <= labels


def run_stage(stage: str, payload: dict[str, Any], directory: Path) -> dict[str, Any]:
    incoming, outgoing = directory / "stage-input.json", directory / "stage-output.json"
    write_json(incoming, payload)
    outgoing.unlink(missing_ok=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ocr_service.qualify",
            "--stage",
            stage,
            "--stage-input",
            str(incoming),
            "--output",
            str(outgoing),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=600)
    except BaseException as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The child can exit between the deadline and group signal.
        process.wait(timeout=30)
        if isinstance(exc, subprocess.TimeoutExpired):
            raise NativeStageWaitTimeout(stage) from None
        raise
    if code == GPU_OOM_EXIT and stage in ("collect", "replay"):
        raise GpuOutOfMemory
    if code == _STATIC_HOOK_FAILURE_EXIT and stage == "collect":
        diagnostic = read_static_hook_failure(outgoing)
        if diagnostic is not None:
            raise StaticHookCollectionFailed(diagnostic)
    if code:
        raise subprocess.CalledProcessError(code, process.args)
    if outgoing.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Qualification metadata exceeds bounds")
    result: dict[str, Any] = json.loads(outgoing.read_bytes())
    return result


def device_identity(
    paddle: Any, runtime: NativeRuntimeSettings, *, compute: bool = True
) -> dict[str, Any]:
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise ValueError("Architecture spoofing is forbidden")
    result = execution_target(
        paddle, inspect_device_identity(runtime.accelerator, runtime.gpu_index), runtime.accelerator
    )
    if runtime.accelerator == "rocm" and compute:
        result["compute_probe"] = compute_probe(paddle, runtime.gpu_index)
    return result


def compute_probe(paddle: Any, gpu_index: int) -> dict[str, Any]:
    """Target-specific FP32 math evidence inside the bounded collector process."""
    import numpy as np

    paddle.set_device(f"gpu:{gpu_index}")
    started = time.perf_counter()
    with paddle.no_grad():
        left = paddle.ones([4, 4], dtype="float32")
        right = paddle.ones([4, 4], dtype="float32")
        image = paddle.ones([1, 1, 5, 5], dtype="float32")
        kernel = paddle.ones([1, 1, 3, 3], dtype="float32")
        product = paddle.matmul(left, right)
        convolution = paddle.nn.functional.conv2d(image, kernel)
        for value in (left, right, image, kernel, product, convolution):
            require_gpu(value, gpu_index)
            if floating_dtype(value.dtype) != "float32":
                raise ValueError("Native compute probe requires FP32")
        if not np.all(product.numpy() == 4) or not np.all(convolution.numpy() == 9):
            raise ValueError("Native compute probe result mismatch")
    return {
        "passed": True,
        "precision": "float32",
        "device": f"gpu:{gpu_index}",
        "seconds": time.perf_counter() - started,
        "operations": ["matmul", "conv2d"],
    }


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = RuntimeSettings.model_validate(payload["runtime"])
    verify_dependencies(
        runtime.expected_packages, payload["dependencies"], accelerator=runtime.accelerator
    )
    profile = load_profile(PROFILE)
    preload: list[GpuMemorySnapshot] = []
    native, paddle, runner_type = construct_native(profile, runtime, record_preload=preload.append)
    device = device_identity(paddle, runtime)
    preprocessor = native.doc_preprocessor_pipeline
    models = [
        native.layout_det_model,
        preprocessor.doc_ori_classify_model,
        preprocessor.doc_unwarping_model,
    ]
    collectors = []
    for name, model in zip(MODEL_NAMES[:3], models, strict=True):
        runner = model.runner
        expected = {
            "run_mode": "paddle_fp32",
            "device_type": "gpu",
            "device_id": runtime.gpu_index,
            "enable_cinn": False,
        }
        if (
            not isinstance(runner, runner_type)
            or runner._model_name != name
            or any(runner._config.get(key) != value for key, value in expected.items())
        ):
            raise ValueError("Collector resolved runner is not native GPU FP32")
        artifact = payload["inventory"][name]
        if (
            Path(runner.model_dir).resolve() != runtime.model_dirs[name].resolve()
            or artifact["graph_file"].rsplit(".", 1)[0] != runner.model_file_prefix
        ):
            raise ValueError("Inspector and native runner artifacts differ")
        collector = HookCollector(
            runtime.gpu_index, parameter_names={entry["name"] for entry in artifact["parameters"]}
        )
        runner.predictor.register_input_hook(collector.observe_input)
        runner.predictor.register_output_hook(collector.observe_output)
        collectors.append(collector)
    # This temporary adapter supplies exact kwargs and response normalization only.
    # It is never qualified and its gated predict() is never invoked.
    adapter = PipelineAdapter(
        native, profile, "unsealed", "unsealed", [], gpu_index=runtime.gpu_index
    )
    forwards: list[list[Any]] = [[] for _ in collectors]
    for row in payload["cases"]:
        case = QualificationCase.model_validate(row)
        for _ in range(2):
            for collector in collectors:
                collector.start()
            with normalize_source(case.path, case.sha256) as normalized:
                results = iter(native.predict(str(normalized.path), **adapter.predict_kwargs))
                result = next(results)
                if next(results, None) is not None:
                    raise ValueError("Expected single-image native inference")
                response = adapter.convert_result(
                    result,
                    request_id=UUID(int=0),
                    photo_index=0,
                    source_sha256=case.sha256,
                    transforms=normalized.transforms,
                    findings=normalized.findings,
                )
            if not has_required_labels(case, response):
                raise ValueError("Required recognized structure absent")
            audit_dynamic_model(native.vl_rec_model.infer, runtime.gpu_index)
            for index, collector in enumerate(collectors):
                try:
                    forwards[index].append(collector.finish())
                except HookCollectionError as exc:
                    raise StaticHookCollectionFailed(StaticHookFailure.model_validate({
                        "model_name": MODEL_NAMES[index],
                        "reason_code": exc.reason_code,
                        "input_count": exc.input_count,
                        "output_count": exc.output_count,
                    })) from None
    artifacts = {}
    for index, name in enumerate(MODEL_NAMES[:3]):
        coverage = common_coverage(forwards[index])
        inventory = payload["inventory"][name]
        bindings = parameter_bindings(
            [ParameterRecord.model_validate(row) for row in inventory["parameters"]],
            coverage,
            runtime.gpu_index,
        )
        artifacts[name] = StaticArtifact.model_validate(
            {
                **inventory,
                "coverage": [row.model_dump() for row in coverage],
                "parameter_bindings": bindings,
            }
        ).model_dump()
    return {
        "static_artifacts": artifacts,
        "device": device,
        "preload_memory": preload[0].model_dump() if preload else None,
    }


def replay(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = RuntimeSettings.model_validate(payload["runtime"])
    started = time.perf_counter()
    adapter = build_pipeline(load_profile(PROFILE), runtime)
    constructed = time.perf_counter()
    adapter.qualify(Path(payload["smoke"]))
    cold_smoke = time.perf_counter() - constructed
    startup = time.perf_counter() - started
    case = QualificationCase.model_validate(payload["case"])
    timings = []
    for _ in range(2):
        started = time.perf_counter()
        response = adapter.predict(
            case.path, request_id=UUID(int=0), photo_index=0, source_sha256=case.sha256
        )
        timings.append(time.perf_counter() - started)
        if not has_required_labels(case, response):
            raise ValueError("Required recognized structure absent in replay")
    return {
        "kind": case.kind,
        "source_sha256": case.sha256,
        "startup_seconds": startup,
        "cold_smoke_seconds": cold_smoke,
        "first_case_seconds": timings[0],
        "warm_case_seconds": timings[1],
        "passed": True,
        "static_audits": adapter.evidence,
        "dynamic_fp32_parameters": audit_dynamic_model(
            adapter.pipeline.vl_rec_model.infer, runtime.gpu_index
        ),
        "profile_id": adapter.profile_id,
        "runtime_fingerprint": adapter.runtime_fingerprint,
    }


def validate_replay(result: dict[str, Any]) -> None:
    audits = result.get("static_audits", [])
    if (
        result.get("passed") is not True
        or len(audits) != 3
        or {row.get("model_name") for row in audits} != set(MODEL_NAMES[:3])
        or result.get("dynamic_fp32_parameters", 0) <= 0
    ):
        raise ValueError("Native replay evidence is incomplete")
    for row in audits:
        if any(
            row.get(key, 0) <= 0 for key in ("input_count", "output_count", "floating_parameters")
        ):
            raise ValueError("Native replay audit is empty")
    for key in ("startup_seconds", "cold_smoke_seconds", "first_case_seconds", "warm_case_seconds"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("Native replay timing is invalid")


def qualify_runtime(runtime: RuntimeSettings, cases: Path, output: Path) -> QualificationReport:
    report = QualificationReport(
        accelerator=runtime.accelerator, image_identity=runtime.image_identity
    )
    serving = output.with_suffix(".runtime.json")
    try:
        report.stage = "publication_prepare"
        serving.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        report.stage = "inputs"
        inputs = load_cases(cases)
        if not re.search(r"(?:^|@)sha256:[0-9a-f]{64}$", runtime.image_identity):
            raise ValueError("An immutable built image digest is required")
        report.profile = load_profile(PROFILE).model_dump()
        report.stage = "dependencies"
        expected, dependencies = seal_dependencies(accelerator=runtime.accelerator)
        if runtime.expected_packages and runtime.expected_packages != expected:
            raise ValueError("Runtime expected packages differ from installed lock")
        runtime = runtime.model_copy(update={"expected_packages": expected})
        verify_dependencies(expected, dependencies, accelerator=runtime.accelerator)
        report.model_content = {
            name: tree_identity(runtime.model_dirs[name]) for name in MODEL_NAMES
        }
        smoke = next(case for case in inputs if case.kind == "upright")
        base = {
            "runtime": runtime.model_dump(mode="json"),
            "dependencies": dependencies,
            "cases": [case.model_dump(mode="json") for case in inputs],
        }
        with tempfile.TemporaryDirectory(prefix="ocr-qualify-") as scratch:
            directory = Path(scratch)
            report.stage = "artifact_inspection"
            inventory = {
                name: run_stage("inspect", {**base, "model_name": name}, directory)
                for name in MODEL_NAMES[:3]
            }
            report.stage = "native_collection"
            collected = run_stage("collect", {**base, "inventory": inventory}, directory)
            report.device = collected["device"]
            if collected.get("preload_memory") is not None:
                report.preload_memory = GpuMemorySnapshot.model_validate(collected["preload_memory"])
            manifest = {
                "qualified_target": {key: value for key, value in report.device.items() if key != "compute_probe"},
                "dependencies": dependencies,
                "image_identity": runtime.image_identity,
                "paddle_identity": "paddle:" + runtime.expected_paddle_commit,
                "model_content": report.model_content,
                "smoke_sha256": smoke.sha256,
                "static_artifacts": collected["static_artifacts"],
            }
            # Seal candidate evidence before fresh factory validation, never previous success.
            write_json(runtime.qualification_manifest, manifest)
            report.manifest_sha256 = file_digest(runtime.qualification_manifest)
            report.stage = "fresh_replay"
            for case in inputs:
                # Two independent fresh workers check compiled hook identifier stability.
                for repetition in range(2):
                    result = run_stage(
                        "replay",
                        {**base, "smoke": str(smoke.path), "case": case.model_dump(mode="json")},
                        directory,
                    )
                    validate_replay(result)
                    if report.profile_id is not None and (
                        result["profile_id"] != report.profile_id
                        or result["runtime_fingerprint"] != report.runtime_fingerprint
                    ):
                        raise ValueError("Fresh replay identity changed")
                    report.profile_id = result["profile_id"]
                    report.runtime_fingerprint = result["runtime_fingerprint"]
                    report.cases.append({**result, "fresh_worker": repetition})
            if report.model_content != {
                name: tree_identity(runtime.model_dirs[name]) for name in MODEL_NAMES
            }:
                raise ValueError("Model bytes changed during qualification")
            verify_dependencies(expected, dependencies, accelerator=runtime.accelerator)
        report.stage = "publish_runtime"
        write_json(serving, runtime.model_dump(mode="json"))
        target = report.device.get("gfx_target") or report.device.get("name")
        completed = report.model_copy(
            update={
                "qualified": True,
                "qualified_targets": [str(target)],
                "stage": "complete",
            }
        )
        report.stage = "publish_report"
        write_json(output, completed.model_dump())
        return completed  # Both mandatory publications succeeded before committing success.
    except Exception as exc:  # noqa: BLE001 -- unknown native/build errors fail closed
        report.qualified = False
        report.qualified_targets = []
        if isinstance(exc, StaticHookCollectionFailed):
            report.failure = STATIC_HOOK_FAILURE_CODE
            report.static_hook_failure = exc.diagnostic
        elif isinstance(exc, NativeStageWaitTimeout):
            report.failure = "TimeoutError"
            report.collection_wait_timed_out = (
                exc.stage == "collect" and report.stage == "native_collection"
            )
        else:
            report.failure = GPU_OOM_CODE if isinstance(exc, GpuOutOfMemory) else type(exc).__name__
        # A writer can fail after replacing its destination. Revoke both stale and
        # partially published artifacts before attempting a sanitized failure report.
        serving.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        try:
            write_json(output, report.model_dump())
        except OSError:
            # Unwritable report storage must not turn a failed attempt into success.
            # A post-replacement error can leave this false report, never a true one.
            pass
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=["cuda", "rocm"])
    parser.add_argument("--runtime", type=Path, default=os.environ.get("OCR_RUNTIME_SETTINGS_PATH"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["inspect", "collect", "replay"], help=argparse.SUPPRESS)
    parser.add_argument("--stage-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        payload = json.loads(args.stage_input.read_bytes())
        if args.stage == "inspect":
            runtime = RuntimeSettings.model_validate(payload["runtime"])
            verify_dependencies(
                runtime.expected_packages, payload["dependencies"], accelerator=runtime.accelerator
            )
            paddle = importlib.import_module("paddle")
            if str(paddle.version.commit) != runtime.expected_paddle_commit:
                raise ValueError("Inspector Paddle source identity mismatch")
            name = payload["model_name"]
            result = inspect_artifact(
                runtime.model_dirs[name], name, paddle, "paddle:" + runtime.expected_paddle_commit
            )
        else:
            try:
                result = (collect if args.stage == "collect" else replay)(payload)
            except StaticHookCollectionFailed as exc:
                write_json(args.output, {
                    "qualified": False,
                    "stage": "native_collection",
                    "failure": STATIC_HOOK_FAILURE_CODE,
                    "static_hook_failure": exc.diagnostic.model_dump(),
                })
                raise SystemExit(_STATIC_HOOK_FAILURE_EXIT) from None
            except Exception as exc:
                if is_gpu_out_of_memory(exc):
                    raise SystemExit(GPU_OOM_EXIT) from None
                raise
        write_json(args.output, result)
        return
    if not qualification_is_supervised(args.supervisor_fd):
        parser.error(
            "Run full qualification through ocr_service.initialize maintain "
            "or ocr_service.source_maintenance with explicit maintenance intent"
        )
    if not args.runtime or not args.cases or not args.accelerator:
        parser.error("--runtime (or OCR_RUNTIME_SETTINGS_PATH), --accelerator and --cases required")
    runtime = RuntimeSettings.model_validate_json(args.runtime.read_bytes())
    if runtime.accelerator != args.accelerator:
        parser.error("--accelerator differs from explicit runtime settings")
    report = qualify_runtime(runtime, args.cases, args.output)
    if not report.qualified and report.failure == GPU_OOM_CODE:
        raise SystemExit(GPU_OOM_EXIT)
    if not report.qualified and report.collection_wait_timed_out:
        raise SystemExit(COLLECTION_WAIT_TIMEOUT_EXIT)
    raise SystemExit(0 if report.qualified else 1)


if __name__ == "__main__":
    main()
