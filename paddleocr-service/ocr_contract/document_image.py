"""White-page document composition and bounded working PNG handling."""

import base64
from hashlib import sha256
from io import BytesIO

from PIL import Image

from . import WorkingImageMetadata, limits

NORMALIZER_VERSION = "white-document-rgb-v2"


def document_rgb(image: Image.Image) -> Image.Image:
    """Composite RGBA, LA and palette/color-key transparency without losing ink."""
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def encode_working_png(image: Image.Image) -> bytes:
    """Encode one metadata-free RGB8 frame using the bounded Pillow settings."""
    limits.check_working_image_dimensions(*image.size)
    if image.mode != "RGB":
        raise ValueError("A working image requires RGB8 pixels")
    output = BytesIO()
    # Pillow's PNG writer only inherits ICC and transparency from RGB image.info.
    # Passing None overrides both. Other ancillary chunks need explicit options.
    image.save(
        output,
        format="PNG",
        save_all=False,
        icc_profile=None,
        transparency=None,
        optimize=False,
        compress_level=-1,
    )
    png = output.getvalue()
    if len(png) > limits.MAX_WORKING_PNG_BYTES:
        raise ValueError("ocr_working_png_too_large")
    return png


def decode_working_png(encoded: str) -> bytes:
    """Check encoded and decoded sizes before materializing the PNG."""
    limits.check_working_png_base64(encoded)
    return base64.b64decode(encoded, validate=True)


def validate_working_png(raw: bytes, metadata: WorkingImageMetadata) -> None:
    """Check PNG identity, frame geometry, chunk CRCs and pixel decoding."""
    limits.check_working_image_dimensions(metadata.width, metadata.height)
    if len(raw) > limits.MAX_WORKING_PNG_BYTES or sha256(raw).hexdigest() != metadata.sha256:
        raise ValueError("invalid_working_artifact_hash")
    try:
        with BytesIO(raw) as stream:
            with Image.open(stream) as decoded:
                if (
                    decoded.format != "PNG"
                    or decoded.size != (metadata.width, metadata.height)
                    or getattr(decoded, "n_frames", 1) != 1
                ):
                    raise ValueError("invalid_working_artifact_dimensions")
                decoded.verify()
            # Pillow verify checks chunk CRCs without inflating IDAT. Reopen
            # the same bounded stream to validate the compressed pixels too.
            stream.seek(0)
            with Image.open(stream) as decoded:
                decoded.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid_working_artifact_png") from exc
