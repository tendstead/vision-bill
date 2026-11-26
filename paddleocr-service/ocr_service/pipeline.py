"""Lazy native factory, qualification gate, and recognized source-view response."""

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import resource
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from ocr_contract import GpuMemorySnapshot, OcrBlock, OcrResponse

from .device_inventory import execution_target, inspect_device_identity
from .gpu_memory import probe_gpu_memory
from .images import normalize_source, working_image
from .precision import StaticArtifact, StaticAuditSession, begin_static_audit
from .profile import (
    ADAPTER_VERSION,
    MODEL_NAMES,
    NORMALIZER_VERSION,
    STATIC_MODEL_NAMES,
    InvalidReceiptOutput,
    NativeRuntimeSettings,
    OcrProfile,
    PreparedRuntimeSettings,
    RuntimeIdentityMismatch,
    RuntimeSettings,
    check_accelerator,
    content_identity,
    effective_pipeline_config,
    file_digest,
    runtime_identity,
    serving_identity,
    verify_dependencies,
    verify_installed_versions,
)
from .vlm_precision import (
    audit_dynamic_model,
    component_precision,
    native_pipeline_class,
    native_predictor_class,
    requires_host_assembly,
)

# Official PaddleX v3.7.2 source bytes. Modifications require a new audited adapter.
PINNED_SOURCES = {
    "pipelines/paddleocr_vl/pipeline.py": "32966d14aad6ac1bc043737c96352a4f0b4efcbead96f9354254dd4b9d1c6675",
    "pipelines/base.py": "f955c34caa6e630b6853c10a4ad189157def0d547fecd262a2e2f9868d175a30",
    "models/doc_vlm/modeling/paddleocr_vl/_paddleocr_vl.py": "2cb9ab3d39c21e776babd20ec34bace122ae2e6fdd6fd8edf1f5789db6d38347",
    "models/common/transformers/transformers/model_utils.py": "eb6c36a17fb52d484331338565e6315cf423218174a309a5d3b42ccbb823c557",
    "models/doc_vlm/predictor.py": "9353a5fa81c1bd57a1e5f2e6bdad0e24c4ed1caccd765f4b5bb127e906e1c45c",
    "models/predictors/local_model_predictor.py": "f796661ba5a658dcdb7cc1378e2d2348edba3532136c91e6757954b21ce643dd",
    "models/__init__.py": "b4b5e5cfd94de3fa5b0edd5999226c202160c926f6b5450fac574371eb470812",
    "models/runners/paddle_static/runner.py": "943d590638a4fb71e8d8ec6a61b0e913e9a6bc4a6e41fd586d466c7938982aa3",
    "pipelines/doc_preprocessor/pipeline.py": "9d75e4d0c628c8e01bd23c5637b8bce218cab19fc4c1bac5ec898a04b66b23f2",
}


def verify_sources(root: Path) -> None:
    for relative, expected in PINNED_SOURCES.items():
        if file_digest(root / "inference" / relative) != expected:
            raise RuntimeIdentityMismatch(f"Unsupported PaddleX source identity: {relative}")


def tree_identity(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a local model directory")
    content = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Unexpected nonregular entry in model directory")
        if path.is_file():
            content[str(path.relative_to(root))] = file_digest(path)
    if not content:
        raise ValueError("Model directory contains no sealed content")
    return content_identity(content)


def recognized_blocks(result: Any, width: int, height: int) -> list[OcrBlock]:
    blocks = []
    for index, block in enumerate(result.get("parsing_res_list", []) or []):
        if isinstance(block, dict):
            text, label, box = (
                block.get("block_content"),
                block.get("block_label", "text"),
                block.get("block_bbox"),
            )
        else:
            text, label, box = (
                getattr(block, "content", None),
                getattr(block, "label", "text"),
                getattr(block, "bbox", None),
            )
        if not isinstance(text, str) or not text.strip():
            continue
        # Only the recognized block's geometry is converted, never native result arrays.
        if box is not None and hasattr(box, "tolist"):
            box = box.tolist()
        if not isinstance(box, (list, tuple)):
            raise TypeError("Recognized block lacks working-space geometry")
        if len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            x1, y1, x2, y2 = map(float, box)
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            points = [[float(x), float(y)] for x, y in box]
        if len(points) < 3 or any(
            not math.isfinite(x)
            or not math.isfinite(y)
            or not 0 <= x <= width
            or not 0 <= y <= height
            for x, y in points
        ):
            raise ValueError("Recognized block exceeds working-image bounds")
        blocks.append(OcrBlock(id=str(index), label=str(label), text=text.strip(), bbox=points))
    if not blocks:
        raise ValueError("OCR returned no recognized text")
    return blocks


logger = logging.getLogger(__name__)


def receipt_memory_fields(gpu_index: int) -> str:
    # The native worker runs on Linux. These process-local counters neither launch
    # another process nor synchronize GPU work. Some ROCm builds omit the counter.
    fields = f"worker_peak_rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}"
    try:
        cuda = getattr(getattr(sys.modules.get("paddle"), "device", None), "cuda", None)
        allocated = getattr(cuda, "memory_allocated", None)
        if callable(allocated):
            fields += f" gpu_allocated_bytes={int(allocated(gpu_index))}"
    except Exception:  # noqa: BLE001 - optional allocator statistics cannot fail a page
        return fields
    return fields


@contextmanager
def trace_receipt_stages(native: Any, gpu_index: int) -> Iterator[None]:
    """Measure real synchronous predictor calls, without logging their inputs."""
    if not logger.isEnabledFor(logging.INFO):
        yield
        return
    preprocessor = native.doc_preprocessor_pipeline
    stages = (
        ("orientation", preprocessor.doc_ori_classify_model),
        ("layout", native.layout_det_model),
        ("vlm_region", native.vl_rec_model),
    )
    regions_completed = 0
    page_started = time.monotonic()
    cpu_started = time.process_time()
    region_seconds = 0.0

    def trace(stage: str, process: Callable[..., Any]) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            nonlocal regions_completed, region_seconds
            is_region = stage == "vlm_region"
            # Bound INFO output. DEBUG exposes the exact last region on interruption.
            level = logging.INFO if not is_region or regions_completed == 0 else logging.DEBUG
            if level == logging.INFO:
                logger.info("ocr_receipt_stage stage=%s state=start regions_completed=%d %s",
                            stage, regions_completed, receipt_memory_fields(gpu_index))
            else:
                logger.debug("ocr_receipt_stage stage=%s state=start regions_completed=%d",
                             stage, regions_completed)
            started = time.monotonic()
            try:
                result = process(*args, **kwargs)
            except Exception:
                logger.warning("ocr_receipt_stage stage=%s state=failed regions_completed=%d elapsed_ms=%d",
                               stage, regions_completed, round((time.monotonic() - started) * 1000))
                raise
            elapsed = time.monotonic() - started
            if is_region:
                regions_completed += 1  # The pinned native VLM uses batch size one.
                region_seconds += elapsed
            logger.log(level, "ocr_receipt_stage stage=%s state=done regions_completed=%d elapsed_ms=%d",
                       stage, regions_completed, round(elapsed * 1000))
            if is_region and regions_completed % 16 == 0:
                logger.info("ocr_receipt_stage stage=vlm_regions state=continue regions_completed=%d elapsed_ms=%d %s",
                            regions_completed, round(region_seconds * 1000), receipt_memory_fields(gpu_index))
            return result
        return measured

    with ExitStack() as cleanup:
        for stage, model in stages:
            # Keep the native predictor, sampler, runner and lazy iterator intact.
            if "process" in vars(model):
                cleanup.callback(setattr, model, "process", vars(model)["process"])
            else:
                cleanup.callback(delattr, model, "process")
            model.process = trace(stage, model.process)
        yield
    logger.info(
        "ocr_receipt_stage stage=vlm_regions state=done regions_completed=%d elapsed_ms=%d "
        "page_elapsed_ms=%d worker_cpu_ms=%d %s",
        regions_completed, round(region_seconds * 1000), round((time.monotonic() - page_started) * 1000),
        round((time.process_time() - cpu_started) * 1000), receipt_memory_fields(gpu_index),
    )


class ReceiptPipeline:
    """Serve real inputs using an explicitly constructed native pipeline."""

    def __init__(
        self, pipeline: Any, profile: OcrProfile, profile_id: str, runtime_fingerprint: str,
        *, gpu_index: int = 0,
    ):
        self.pipeline = pipeline
        self.gpu_index = gpu_index
        self.profile = profile
        self.profile_id = profile_id
        self.runtime_fingerprint = runtime_fingerprint
        self.evidence: list[dict[str, Any]] = []

    @property
    def predict_kwargs(self) -> dict[str, Any]:
        return {
            "use_doc_orientation_classify": self.profile.use_doc_orientation_classify,
            "use_doc_unwarping": self.profile.use_doc_unwarping,
            "use_queues": False,
            "min_pixels": self.profile.min_pixels,
            "max_pixels": self.profile.max_pixels,
            "max_new_tokens": self.profile.max_new_tokens,
        }

    def _execute(self, image: Path) -> Any:
        results = iter(self.pipeline.predict(str(image), **self.predict_kwargs))
        result = next(results, None)
        if result is None or next(results, None) is not None:
            raise InvalidReceiptOutput("Native OCR requires one result per uploaded image")
        return result

    def convert_result(
        self,
        result: Any,
        *,
        request_id: UUID,
        photo_index: int,
        source_sha256: str,
        transforms: list[dict[str, Any]],
        findings: list[str],
    ) -> OcrResponse:
        view = working_image(result["doc_preprocessor_res"], transforms)
        blocks = recognized_blocks(result, view.width, view.height)
        return OcrResponse(
            request_id=request_id,
            photo_index=photo_index,
            source_sha256=source_sha256,
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
            working_image=view,
            profile_id=self.profile_id,
            runtime_fingerprint=self.runtime_fingerprint,
            telemetry={
                "findings": findings,
                "coordinate_space": "working",
                **({"precision_audits": self.evidence} if self.evidence else {}),
            },
        )

    def predict(
        self, image: Path, *, request_id: UUID, photo_index: int, source_sha256: str
    ) -> OcrResponse:
        started = time.monotonic()
        logger.info("ocr_receipt_stage stage=normalize state=start")
        with normalize_source(image, source_sha256) as normalized:
            logger.info("ocr_receipt_stage stage=normalize state=done elapsed_ms=%d",
                        round((time.monotonic() - started) * 1000))
            with trace_receipt_stages(self.pipeline, self.gpu_index):
                result = self._execute(normalized.path)
            try:
                started = time.monotonic()
                logger.info("ocr_receipt_stage stage=convert state=start")
                response = self.convert_result(
                    result,
                    request_id=request_id,
                    photo_index=photo_index,
                    source_sha256=source_sha256,
                    transforms=normalized.transforms,
                    findings=normalized.findings,
                )
                logger.info("ocr_receipt_stage stage=convert state=done elapsed_ms=%d blocks=%d",
                            round((time.monotonic() - started) * 1000), len(response.blocks))
                return response
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
                raise InvalidReceiptOutput("Invalid OCR result") from exc


class PipelineAdapter(ReceiptPipeline):
    """Explicit hardware qualification with live static-model evidence."""

    def __init__(
        self,
        pipeline: Any,
        profile: OcrProfile,
        profile_id: str,
        runtime_fingerprint: str,
        static_audits: list[StaticAuditSession],
        *,
        smoke_sha256: str | None = None,
        gpu_index: int = 0,
        text_dtype: str = "float32",
    ):
        super().__init__(pipeline, profile, profile_id, runtime_fingerprint)
        self.static_audits = static_audits
        self.smoke_sha256 = smoke_sha256
        self.gpu_index = gpu_index
        self.text_dtype = text_dtype
        self.qualified = False

    def _execute(self, image: Path) -> Any:
        for session in self.static_audits:
            session.start()
        try:
            result = super()._execute(image)
            evidence = [session.finish() for session in self.static_audits]
            audit_dynamic_model(self.pipeline.vl_rec_model.infer, self.gpu_index, self.text_dtype)
            self.evidence = evidence
            return result
        except Exception:
            self.qualified = False
            raise
        finally:
            for session in self.static_audits:
                session.enabled = False

    def qualify(self, image: Path) -> list[dict[str, Any]]:
        """Execute fresh native forward without going through the serving gate."""
        self.qualified = False
        if len(self.static_audits) != len(STATIC_MODEL_NAMES) or {
            s.artifact.model_name for s in self.static_audits
        } != set(STATIC_MODEL_NAMES):
            raise ValueError("Qualification requires two static model audits")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if not self.smoke_sha256 or digest != self.smoke_sha256:
            raise ValueError("Qualification image hash differs from sealed smoke coverage")
        with normalize_source(image, digest) as normalized:
            result = self._execute(normalized.path)
            self.convert_result(
                result,
                request_id=UUID(int=0),
                photo_index=0,
                source_sha256=digest,
                transforms=normalized.transforms,
                findings=normalized.findings,
            )
        self.qualified = True
        return list(self.evidence)

    def predict(
        self, image: Path, *, request_id: UUID, photo_index: int, source_sha256: str
    ) -> OcrResponse:
        if not self.qualified:
            raise ValueError("Native pipeline has not been qualified")
        return super().predict(
            image, request_id=request_id, photo_index=photo_index, source_sha256=source_sha256
        )


def construct_native(
    profile: OcrProfile,
    runtime: NativeRuntimeSettings,
    *,
    device: dict[str, Any],
    record_preload: Callable[[GpuMemorySnapshot], None] | None = None,
) -> tuple[Any, Any, type[Any]]:
    """Construct the pinned native profile; no coverage or serving permission is implied."""
    # PaddleX captures this flag at import time. Refuse a pre-imported process,
    # where setting the environment would create a false fallback guarantee.
    if any(name == "paddlex" or name.startswith("paddlex.") for name in sys.modules):
        raise ValueError("Build native pipeline in a fresh worker before importing PaddleX")
    os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "1"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(runtime.cache_dir)
    if set(runtime.model_dirs) != set(MODEL_NAMES):
        raise ValueError("All three explicit native model directories are required")
    paddle = importlib.import_module("paddle")
    paddlex = importlib.import_module("paddlex")
    normalize_engine_config = importlib.import_module(
        "paddlex.inference.models"
    ).normalize_engine_config
    DocVLMLocalPredictor = importlib.import_module(
        "paddlex.inference.models.doc_vlm.predictor"
    ).DocVLMLocalPredictor
    PaddleStaticRunner = importlib.import_module(
        "paddlex.inference.models.runners.paddle_static"
    ).PaddleStaticRunner
    BasePipeline = importlib.import_module("paddlex.inference.pipelines.base").BasePipeline
    _PaddleOCRVLPipeline = importlib.import_module(
        "paddlex.inference.pipelines.paddleocr_vl.pipeline"
    )._PaddleOCRVLPipeline

    verify_sources(Path(str(paddlex.__file__)).parent)
    if str(paddle.version.commit) != runtime.expected_paddle_commit:
        raise RuntimeIdentityMismatch("Paddle source commit mismatch")
    check_accelerator(paddle, runtime.accelerator, runtime.gpu_index)
    if str(inspect.signature(DocVLMLocalPredictor._build)) != "(self, **kwargs)":
        raise RuntimeIdentityMismatch("Unsupported native predictor _build signature")
    signature = inspect.signature(BasePipeline.create_model)
    if (
        list(signature.parameters) != ["self", "config", "kwargs"]
        or signature.parameters["kwargs"].kind != inspect.Parameter.VAR_KEYWORD
    ):
        raise RuntimeIdentityMismatch("Unsupported native pipeline create_model signature")
    config = effective_pipeline_config(profile)
    static_config = {
        "run_mode": "paddle_fp32",
        "device_type": "gpu",
        "device_id": runtime.gpu_index,
        "enable_cinn": False,
    }

    def resolve(value: Any) -> None:
        if isinstance(value, dict):
            if "model_name" in value:
                value["model_dir"] = str(runtime.model_dirs[value["model_name"]])
                value["engine"] = (
                    "paddle_dynamic"
                    if value["module_name"] == "vl_recognition"
                    else "paddle_static"
                )
                value["engine_config"] = (
                    {"device_type": "gpu", "device_id": runtime.gpu_index}
                    if value["module_name"] == "vl_recognition"
                    else dict(static_config)
                )
            for child in value.values():
                resolve(child)
        elif isinstance(value, list):
            for child in value:
                resolve(child)

    resolve(config)
    predictor = native_predictor_class(
        DocVLMLocalPredictor,
        runtime.gpu_index,
        text_dtype=component_precision(runtime.accelerator, device.get("compute_capability"))["text"],
        assemble_on_host=requires_host_assembly(
            runtime.accelerator, device.get("compute_capability")
        ),
    )
    pipeline_type = native_pipeline_class(_PaddleOCRVLPipeline, predictor, normalize_engine_config)
    if record_preload is not None:
        snapshot = probe_gpu_memory(runtime.accelerator, runtime.gpu_index)
        if snapshot is not None:
            record_preload(snapshot)
    native = pipeline_type(config, device=runtime.device, use_hpip=False)
    return native, paddle, PaddleStaticRunner


def verified_qualification_manifest(
    runtime: RuntimeSettings,
) -> tuple[dict[str, Any], dict[str, StaticArtifact], dict[str, str], dict[str, str]]:
    if any(name == "paddlex" or name.startswith("paddlex.") for name in sys.modules):
        raise ValueError("Build native pipeline in a fresh worker before importing PaddleX")
    manifest = json.loads(runtime.qualification_manifest.read_text())
    versions, dependency_content = verify_dependencies(
        runtime.expected_packages,
        manifest.get("dependencies", {}),
        accelerator=runtime.accelerator,
    )
    inspector = f"paddle:{runtime.expected_paddle_commit}"
    if (
        manifest["image_identity"] != runtime.image_identity
        or manifest["paddle_identity"] != inspector
    ):
        raise ValueError("Qualification manifest build/source identity mismatch")
    artifacts = {
        name: StaticArtifact.model_validate(value)
        for name, value in manifest["static_artifacts"].items()
    }
    if set(artifacts) != set(STATIC_MODEL_NAMES):
        raise ValueError("Qualification manifest must cover every enabled static model")
    model_content = {name: tree_identity(path) for name, path in runtime.model_dirs.items()}
    if model_content != manifest["model_content"]:
        raise ValueError("Native model content identity mismatch")
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise ValueError("Architecture spoofing is forbidden")
    paddle = importlib.import_module("paddle")
    check_accelerator(paddle, runtime.accelerator, runtime.gpu_index)
    device = inspect_device_identity(runtime.accelerator, runtime.gpu_index)
    actual_target = execution_target(paddle, device, runtime.accelerator)
    if not manifest.get("qualified_target") or actual_target != manifest["qualified_target"]:
        raise ValueError("Native qualified target identity mismatch; requalification required")
    return manifest, artifacts, versions, dependency_content


def build_serving_pipeline(
    profile: OcrProfile,
    runtime: PreparedRuntimeSettings | RuntimeSettings,
) -> ReceiptPipeline:
    """Verify the selected runtime, then load local models without running a sample."""
    started = time.monotonic()
    logger.info("ocr_worker_load stage=verify state=start accelerator=%s gpu_index=%d",
                runtime.accelerator, runtime.gpu_index)
    try:
        if isinstance(runtime, PreparedRuntimeSettings):
            verify_installed_versions(runtime.expected_packages, accelerator=runtime.accelerator)
            model_content = {name: tree_identity(path) for name, path in runtime.model_dirs.items()}
            if model_content != runtime.model_content:
                raise ValueError("Prepared model content changed")
            paddle = importlib.import_module("paddle")
            check_accelerator(paddle, runtime.accelerator, runtime.gpu_index)
            device = inspect_device_identity(runtime.accelerator, runtime.gpu_index)
            execution_target(paddle, device, runtime.accelerator)
            if device != runtime.device_identity:
                raise ValueError("Prepared device identity changed")
        else:
            manifest, _, _, _ = verified_qualification_manifest(runtime)
            model_content = manifest["model_content"]
            device = manifest["qualified_target"]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("ocr_worker_load stage=verify state=failed error_type=%s", type(exc).__name__)
        raise RuntimeIdentityMismatch("Native runtime identity mismatch") from exc

    logger.info(
        "ocr_worker_load stage=verify state=done elapsed_ms=%d decoder_dtype=%s "
        "visual_dtype=float32 projector_dtype=float32 static_dtype=float32 max_pixels=%d",
        int((time.monotonic() - started) * 1000),
        component_precision(runtime.accelerator, device.get("compute_capability"))["text"], profile.max_pixels,
    )
    native, _, runner_type = construct_native(profile, runtime, device=device)
    preprocessor = native.doc_preprocessor_pipeline
    models = (
        native.layout_det_model,
        preprocessor.doc_ori_classify_model,
    )
    for name, model in zip(STATIC_MODEL_NAMES, models, strict=True):
        runner = model.runner
        if (
            not isinstance(runner, runner_type)
            or runner._model_name != name
            or Path(runner.model_dir).resolve() != runtime.model_dirs[name].resolve()
            or runner._config.get("run_mode") != "paddle_fp32"
            or runner._config.get("device_type") != "gpu"
            or runner._config.get("device_id") != runtime.gpu_index
            or runner._config.get("enable_cinn") is not False
        ):
            raise RuntimeIdentityMismatch("Unexpected native static runner settings")
    identity = serving_identity(profile, runtime, model_content, device)
    logger.info("ocr_worker_load stage=ready total_ms=%d", int((time.monotonic() - started) * 1000))
    return ReceiptPipeline(native, profile, identity["profile_id"], identity["runtime_fingerprint"],
                           gpu_index=runtime.gpu_index)


def build_pipeline(profile: OcrProfile, runtime: RuntimeSettings) -> PipelineAdapter:
    manifest, artifacts, versions, dependency_content = verified_qualification_manifest(runtime)
    inspector = f"paddle:{runtime.expected_paddle_commit}"
    model_content = manifest["model_content"]
    native, paddle, PaddleStaticRunner = construct_native(
        profile, runtime, device=manifest["qualified_target"]
    )
    preprocessor = native.doc_preprocessor_pipeline
    models = (
        native.layout_det_model,
        preprocessor.doc_ori_classify_model,
    )
    audits = [
        begin_static_audit(
            model,
            expected_model_name=name,
            gpu_index=runtime.gpu_index,
            sealed_artifact=artifacts[name],
            runner_type=PaddleStaticRunner,
            inspector_identity=inspector,
        )
        for name, model in zip(STATIC_MODEL_NAMES, models, strict=True)
    ]
    profile_id = content_identity(
        {
            "effective_config": effective_pipeline_config(profile),
            "limits": profile.model_dump(),
            "adapter": ADAPTER_VERSION,
            "component_precision": manifest["qualified_target"]["component_precision"],
            "normalizer": NORMALIZER_VERSION,
            "model_content": model_content,
            "dependencies": versions,
            "dependency_content": dependency_content,
            "paddlex_sources": PINNED_SOURCES,
            "paddle_commit": runtime.expected_paddle_commit,
        }
    )
    fingerprint = runtime_identity(
        profile_id,
        {
            "accelerator": runtime.accelerator,
            "gpu_index": runtime.gpu_index,
            "device_name": paddle.device.cuda.get_device_name(runtime.gpu_index),
            "paddle_build": paddle.version.full_version,
            "image_identity": runtime.image_identity,
            "qualification_manifest": content_identity(manifest),
        },
    )
    return PipelineAdapter(
        native,
        profile,
        profile_id,
        fingerprint,
        audits,
        text_dtype=manifest["qualified_target"]["component_precision"]["text"],
        smoke_sha256=manifest["smoke_sha256"],
        gpu_index=runtime.gpu_index,
    )
