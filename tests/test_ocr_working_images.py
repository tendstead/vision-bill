import base64
import hashlib
from io import BytesIO

import pytest
from PIL import Image


def test_normalization_preserves_source_and_rotates_exif(tmp_path):
    from ocr_service.images import normalize_source

    source = tmp_path / "photo.jpg"
    image = Image.new("RGB", (9, 5), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    original = source.read_bytes()
    with normalize_source(source, hashlib.sha256(original).hexdigest()) as normalized:
        assert Image.open(normalized.path).size == (5, 9)
        assert Image.open(normalized.path).mode == "RGB"
        assert normalized.transforms[0]["exif_orientation"] == 6
        derived = normalized.path
    assert source.read_bytes() == original
    assert not derived.exists()


def test_multiframe_gif_reports_first_frame(tmp_path):
    from ocr_service.images import normalize_source

    source = tmp_path / "photo.gif"
    Image.new("RGB", (4, 7), "red").save(
        source, save_all=True, append_images=[Image.new("RGB", (4, 7), "blue")]
    )
    with normalize_source(source, hashlib.sha256(source.read_bytes()).hexdigest()) as normalized:
        assert normalized.transforms[0]["selected_frame"] == 0
        assert normalized.transforms[0]["frame_count"] == 2
        assert normalized.findings == ["multiframe_first_frame_only"]
        assert Image.open(normalized.path).convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_source_hash_mismatch_rejected_before_decode(tmp_path):
    from ocr_service.images import normalize_source

    source = tmp_path / "bad.png"
    source.write_bytes(b"bad")
    with pytest.raises(ValueError, match="source hash"), normalize_source(source, "0" * 64):
        pass


def test_actual_preprocessor_bgr_output_becomes_rgb_png():
    import numpy as np
    from ocr_service.images import working_image

    original_bgr = np.arange(3 * 8 * 3, dtype=np.uint8).reshape(3, 8, 3)
    pixels = np.rot90(original_bgr)
    result = {
        "output_img": pixels,
        "input_img": original_bgr,
        "angle": 90,
        "model_settings": {"use_doc_orientation_classify": True, "use_doc_unwarping": False},
    }
    view = working_image(result, [{"input": "RGB"}])
    image = Image.open(BytesIO(base64.b64decode(view.png_base64)))
    assert image.size == (view.width, view.height) == (3, 8)
    assert image.tobytes() == pixels[:, :, ::-1].tobytes()
    assert view.sha256 == hashlib.sha256(base64.b64decode(view.png_base64)).hexdigest()
    assert view.inverse_mapping is None
    assert view.transforms[-1] == {
        "operation": "doc_preprocessor",
        "angle": 90,
        "orientation_model": "PP-LCNet_x1_0_doc_ori",
        "unwarping_model": None,
        "coordinate_space": "working",
        "inverse_mapping": None,
    }
    result["model_settings"]["use_doc_unwarping"] = True
    with pytest.raises(ValueError, match="orientation-only"):
        working_image(result, [])


def test_working_image_requires_actual_native_output():
    from ocr_service.images import working_image

    with pytest.raises(ValueError, match="output_img"):
        working_image({}, [])


def test_heif_source_uses_actual_decoder_and_preserves_bytes(tmp_path):
    from ocr_service.images import normalize_source
    from pillow_heif import from_pillow

    source = tmp_path / "photo.heic"
    from_pillow(Image.new("RGB", (12, 8), "white")).save(source)
    original = source.read_bytes()
    with normalize_source(source, hashlib.sha256(original).hexdigest()) as normalized:
        with Image.open(normalized.path) as derived:
            assert derived.mode == "RGB"
            assert derived.size == (12, 8)
        assert normalized.transforms[0]["source_format"] == "HEIF"
    assert source.read_bytes() == original


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
@pytest.mark.parametrize("native", [True, False])
def test_transparent_document_preserves_black_ink_on_white_background(tmp_path, mode, native):
    from ocr_service.images import normalize_source
    from vision_bill.model.extraction import ReceiptInput
    from vision_bill.provider.llm.ollama import OllamaProvider

    source = tmp_path / "transparent.png"
    if mode == "RGBA":
        image = Image.new(mode, (2, 1), (0, 0, 0, 0))
        image.putpixel((0, 0), (0, 0, 0, 255))
    elif mode == "LA":
        image = Image.new(mode, (2, 1), (0, 0))
        image.putpixel((0, 0), (0, 255))
    else:
        image = Image.new(mode, (2, 1), 0)
        image.putpalette([0, 0, 0, 0, 0, 0] + [0] * 762)
        image.info["transparency"] = 0
        image.putpixel((0, 0), 1)
    image.save(source)
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if native:
        with normalize_source(source, digest) as normalized:
            derived = normalized.path.read_bytes()
            transforms = normalized.transforms
    else:
        _, _, views = OllamaProvider("http://local")._prepare_receipt_input(
            ReceiptInput(mode="direct_vision", image_paths=[str(source)])
        )
        view = views[0]
        derived = base64.b64decode(view.working_image.png_base64)
        transforms = view.working_image.transforms
        assert view.source_sha256 == digest
        assert view.working_image.sha256 == hashlib.sha256(derived).hexdigest()
        assert view.working_image.inverse_mapping is None
    with Image.open(BytesIO(derived)) as rgb:
        assert rgb.size == (2, 1)
        assert [rgb.getpixel((x, 0)) for x in range(2)] == [(0, 0, 0), (255, 255, 255)]
    assert transforms[0]["alpha_background"] == "#ffffff"
    assert source.read_bytes() == original


async def test_accepted_seeded_12mp_jpeg_round_trips_full_resolution_working_png(
    settings, tmp_path
):
    import time
    from uuid import uuid4

    import numpy as np
    from ocr_contract import OcrResponse, WorkingImageMetadata
    from ocr_service.api import validate_result, validate_source
    from ocr_service.images import working_image
    from ocr_service.worker import publish_result, read_result

    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider
    from vision_bill.service.image_service import ImageService

    pixels = np.random.default_rng(1729).integers(
        0, 256, size=(3000, 4000, 3), dtype=np.uint8
    )
    encoded = BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=90)
    source = encoded.getvalue()
    assert len(source) < 20 * 1024 * 1024
    validate_source(source, "image/jpeg", limit=25 * 1024 * 1024)

    # The decoded JPEG stands in for the preprocessor's BGR output.
    with Image.open(BytesIO(source)) as decoded:
        expected_rgb = decoded.convert("RGB").tobytes()
    expected_pixels = np.frombuffer(expected_rgb, dtype=np.uint8).reshape(3000, 4000, 3)
    view = working_image(
        {
            "output_img": expected_pixels[:, :, ::-1], "angle": 0,
            "model_settings": {"use_doc_orientation_classify": True, "use_doc_unwarping": False},
        },
        [],
    )
    png = base64.b64decode(view.png_base64)
    assert len(png) > 25 * 1024 * 1024
    print(f"source={len(source)} bytes, working PNG={len(png)} bytes")
    assert (view.width, view.height) == (4000, 3000)
    with Image.open(BytesIO(png)) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.size == (4000, 3000)
        assert decoded.tobytes() == expected_rgb
    result = OcrResponse(
        request_id=uuid4(),
        photo_index=0,
        source_sha256=hashlib.sha256(source).hexdigest(),
        text="SHOP",
        blocks=[],
        profile_id="fixed-profile",
        runtime_fingerprint="cpu-test",
        working_image=view,
    )
    validate_result(result)

    payload = result.model_dump_json().encode()
    print(f"wire={len(payload)} bytes")
    page = PaddleOcrProvider._validated_response(
        payload, result.request_id, 0, result.source_sha256, result.profile_id
    )
    assert read_result(tmp_path, publish_result(result, tmp_path)) == page
    service = ImageService(settings.images)
    image_id = uuid4()
    path = await service.save_working_image(image_id, page.working_image, deadline=time.monotonic() + 10)
    metadata = WorkingImageMetadata.model_validate(page.working_image.model_dump())
    assert service.read_working_image(image_id, str(path), metadata) == png


@pytest.mark.parametrize("size", [(1, 25_000), (401, 331)])
def test_bounded_rgb_encoder_writes_only_standard_chunks_and_rgb8_scanlines(size):
    import random
    import struct
    import zlib

    from ocr_contract.document_image import encode_working_png
    from ocr_contract.limits import working_png_bound

    width, height = size
    pixels = random.Random(1729).randbytes(3 * width * height)
    image = Image.frombytes("RGB", size, pixels)
    image.info.update({"icc_profile": b"untrusted profile", "transparency": (0, 0, 0)})
    png = encode_working_png(image)
    assert len(png) <= working_png_bound(width * height)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    chunks = []
    offset = 8
    while offset < len(png):
        length, name = struct.unpack_from(">I4s", png, offset)
        chunks.append((name, png[offset + 8 : offset + 8 + length]))
        offset += 12 + length
    assert offset == len(png)
    assert chunks[0] == (b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    assert chunks[-1] == (b"IEND", b"")
    idat = [payload for name, payload in chunks[1:-1] if name == b"IDAT"]
    assert len(idat) == len(chunks) - 2
    assert len(idat) >= 2
    assert all(len(chunk) == 65_536 for chunk in idat[:-1])
    assert len(zlib.decompress(b"".join(idat))) == (3 * width + 1) * height
    with Image.open(BytesIO(png)) as decoded:
        assert decoded.size == size
        assert decoded.tobytes() == pixels
