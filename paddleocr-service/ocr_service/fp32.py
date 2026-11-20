"""Native VLM substitution with an explicit loader dtype and parameter audit."""

from collections.abc import Callable
from typing import Any


def floating_dtype(dtype: Any) -> str | None:
    normalized = str(dtype).lower().rsplit(".", 1)[-1]
    floating = "float" in normalized or normalized in {"fp16", "bf16", "fp32", "fp64"}
    return normalized if floating else None


def require_gpu(tensor: Any, gpu_index: int) -> None:
    location = tensor.place
    if not location.is_gpu_place():
        raise ValueError("Floating tensor is outside the requested GPU")
    if location.gpu_device_id() != gpu_index:
        raise ValueError("Floating tensor is on a different GPU index")


def audit_dynamic_model(model: Any, gpu_index: int) -> int:
    inspected = 0
    for weight in model.parameters():
        precision = floating_dtype(weight.dtype)
        if precision is None:
            continue
        if precision != "float32":
            raise ValueError("Native dynamic floating parameter precision must be FP32")
        require_gpu(weight, gpu_index)
        inspected += 1
    if inspected == 0:
        raise ValueError("Native dynamic model has no verified floating parameters")
    return inspected


def fp32_predictor_class(upstream: type[Any], gpu_index: int) -> type[Any]:
    class ExplicitFp32Predictor(upstream):  # type: ignore[misc]
        def _build(self, **kwargs: Any) -> Any:
            # The pinned upstream constructor chooses BF16 immediately before
            # dispatching here. Override before its original from_pretrained.
            self.dtype = "float32"
            built = super()._build(**kwargs)
            audit_dynamic_model(built[0], gpu_index)
            return built

    return ExplicitFp32Predictor


def native_pipeline_class(
    upstream: type[Any], predictor: Callable[..., Any], normalize: Callable[..., Any]
) -> type[Any]:
    class ExplicitNativePipeline(upstream):  # type: ignore[misc]
        def create_model(self, config: dict[str, Any], **kwargs: Any) -> Any:
            if config.get("module_name") == "vl_recognition":
                name = config["model_name"]
                directory = config.get("model_dir")
                if name != "PaddleOCR-VL-1.6-0.9B" or not directory:
                    raise ValueError("Native VLM requires its pinned name and explicit model path")
                normalized = normalize(
                    "paddle_dynamic",
                    config.get("engine_config", {}),
                    model_name=name,
                    device=self.device,
                )
                return predictor(
                    model_name=name,
                    model_dir=directory,
                    batch_size=1,
                    engine_config=normalized,
                    **kwargs,
                )
            return super().create_model(config, **kwargs)

    return ExplicitNativePipeline
