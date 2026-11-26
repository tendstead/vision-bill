"""Preserve uploaded bytes while deriving RGB inputs and working-space PNG views."""

import base64
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from ocr_contract import WorkingImage
from ocr_contract.document_image import NORMALIZER_VERSION, document_rgb, encode_working_png
from ocr_contract.limits import NATIVE_SOURCE_MAX_BYTES, check_working_image_dimensions
from PIL import Image, ImageOps


@dataclass(frozen=True)
class NormalizedSource:
    path: Path
    transforms: list[dict[str, Any]]
    findings: list[str]


@contextmanager
def normalize_source(source: Path, source_sha256: str) -> Iterator[NormalizedSource]:
    with source.open("rb") as stream:
        uploaded_bytes = stream.read(NATIVE_SOURCE_MAX_BYTES + 1)
    if len(uploaded_bytes) > NATIVE_SOURCE_MAX_BYTES:
        raise ValueError("OCR source exceeds the native upload limit")
    actual_sha256 = hashlib.sha256(uploaded_bytes).hexdigest()
    if actual_sha256 != source_sha256:
        raise ValueError("Uploaded source hash differs from the supplied identity")

    from pillow_heif.as_plugin import register_heif_opener

    register_heif_opener()
    with TemporaryDirectory(prefix="vision-bill-ocr-") as temporary:
        derived_path = Path(temporary, "input.png")
        with Image.open(BytesIO(uploaded_bytes)) as decoded:
            check_working_image_dimensions(*decoded.size)
            frame_count = getattr(decoded, "n_frames", 1)
            decoded.seek(0)
            transform = {
                "operation": "normalize_rgb",
                "normalizer_version": NORMALIZER_VERSION,
                "alpha_background": "#ffffff",
                "source_format": decoded.format,
                "exif_orientation": decoded.getexif().get(274, 1),
                "selected_frame": 0,
                "frame_count": frame_count,
            }
            document_rgb(ImageOps.exif_transpose(decoded)).save(derived_path, format="PNG")
        findings = ["multiframe_first_frame_only"] if frame_count > 1 else []
        yield NormalizedSource(derived_path, [transform], findings)


def working_image(preprocessor: Any, transforms: list[dict[str, Any]]) -> WorkingImage:
    import numpy as np

    output = preprocessor.get("output_img")
    if (
        not isinstance(output, np.ndarray)
        or output.ndim != 3
        or output.shape[2] != 3
        or output.dtype != np.uint8
    ):
        raise ValueError("DocPreprocessor output_img must contain uint8 BGR pixels")
    settings = preprocessor.get("model_settings", {})
    if (
        not isinstance(settings, dict)
        or settings.get("use_doc_orientation_classify") is not True
        or settings.get("use_doc_unwarping") is not False
    ):
        raise ValueError("Working image requires orientation-only document preprocessing")
    angle = preprocessor.get("angle")
    if angle not in {0, 90, 180, 270}:
        raise ValueError("Enabled document orientation requires a recognized rotation angle")
    height, width, _ = output.shape
    check_working_image_dimensions(width, height)
    with Image.fromarray(np.ascontiguousarray(output[..., ::-1])) as rgb:
        png_bytes = encode_working_png(rgb)
    preprocessing_transform = {
        "operation": "doc_preprocessor",
        "angle": int(angle),
        "orientation_model": "PP-LCNet_x1_0_doc_ori",
        "unwarping_model": None,
        "coordinate_space": "working",
        "inverse_mapping": None,
    }
    return WorkingImage(
        view_id=uuid4(),
        width=width,
        height=height,
        sha256=hashlib.sha256(png_bytes).hexdigest(),
        png_base64=base64.b64encode(png_bytes).decode("ascii"),
        transforms=transforms + [preprocessing_transform],
        inverse_mapping=None,
    )


# Compatibility for pre-v2 diagnostic callers; serving uses PipelineAdapter.
def _to_plain(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain JSON-safe Python."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    # Primitives (and None) pass through untouched; this also narrows the type
    # so the numpy checks below are only reached for non-primitive objects.
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    # numpy scalar (has .item) or numpy array (has .tolist)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except (ValueError, AttributeError, TypeError):
            pass
    return obj


def _normalize_bbox(coordinate: Any) -> list[list[float]]:
    """Normalise a box coordinate to a list of ``[x, y]`` points.

    PaddleOCR may return either a flat list ``[x1, y1, x2, y2, ...]`` or a list
    of points ``[[x1, y1], [x2, y2], ...]``. Both are handled.
    """
    coordinate = _to_plain(coordinate)
    if not coordinate:
        return []
    first = coordinate[0]
    if isinstance(first, (int, float)) or hasattr(first, "item"):
        nums = [float(v) for v in coordinate]
        return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]
    return [[float(x) for x in point] for point in coordinate]


def process_ocrvl_result(result: Any) -> dict[str, Any]:
    """Return recognized content in reading order, without image/model internals."""
    res_dict = result.res if hasattr(result, "res") else result
    blocks: list[dict[str, Any]] = []
    for block in (res_dict or {}).get("parsing_res_list", []) or []:
        # Native PaddleX results contain ParsingBlock objects; exported JSON
        # contains dictionaries. Both preserve the pipeline's reading order.
        if isinstance(block, dict):
            text = block.get("block_content", "")
            label = block.get("block_label", "text")
            coordinate = block.get("block_bbox", [])
        else:
            text = getattr(block, "content", "")
            label = getattr(block, "label", "text")
            coordinate = getattr(block, "bbox", [])
        if not isinstance(text, str) or not text.strip():
            continue
        blocks.append(
            {
                "label": str(label),
                "text": text.strip(),
                "bbox": _normalize_bbox(coordinate),
            }
        )
    if not blocks:
        raise ValueError("OCR returned no recognized text; layout boxes are not transcription")
    return {"text": "\n".join(block["text"] for block in blocks), "blocks": blocks}
