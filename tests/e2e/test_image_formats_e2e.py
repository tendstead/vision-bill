"""Image formats against the real stack: HEIC/HEIF acceptance, byte-exact originals and
orientation-aware WebP thumbnails (needs the image helper, which the image always has)."""

import struct
from pathlib import Path

import httpx
import pytest

from e2e.helpers import API, register_user

pytestmark = pytest.mark.e2e

ASSETS = Path(__file__).parent / "assets"


def webp_size(data: bytes) -> tuple[int, int]:
    """Pixel size of a WebP file (lossy, lossless or extended)."""
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", "not a WebP file"
    chunk = data[12:16]
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        bits = struct.unpack("<I", data[21:25])[0]
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        return (
            int.from_bytes(data[24:27], "little") + 1,
            int.from_bytes(data[27:30], "little") + 1,
        )
    raise AssertionError(f"unknown WebP chunk {chunk!r}")


def upload(client: httpx.Client, name: str, content_type: str) -> dict:
    data = (ASSETS / name).read_bytes()
    response = client.post(f"{API}/images", files={"receipt": (name, data, content_type)})
    assert response.status_code in (201, 202), response.text
    return response.json()


def test_ui_config_reports_heic_support(http_factory) -> None:
    assert http_factory().get(f"{API}/system/ui-config").json()["heic_supported"] is True


@pytest.mark.parametrize(
    ("name", "content_type", "size"),
    [
        ("synthetic.heic", "image/heic", (200, 100)),
        ("synthetic-exif-rotated.jpg", "image/jpeg", (100, 200)),  # EXIF orientation 6 applied
    ],
)
def test_original_is_kept_and_thumbnail_is_upright(
    http_factory, name: str, content_type: str, size: tuple[int, int]
) -> None:
    client, _ = register_user(http_factory, "formats")
    image_id = upload(client, name, content_type)["image_id"]

    original = client.get(f"{API}/images/{image_id}/file")
    assert original.status_code == 200
    assert original.content == (ASSETS / name).read_bytes()  # stored byte for byte
    assert original.headers["content-type"] == content_type

    thumb = client.get(f"{API}/images/{image_id}/thumb")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/webp"
    assert webp_size(thumb.content) == size


def test_corrupt_heic_is_rejected_and_nothing_is_stored(http_factory) -> None:
    client, _ = register_user(http_factory, "formats")
    broken = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1" + b"\x00" * 64
    response = client.post(f"{API}/images", files={"receipt": ("broken.heic", broken, "image/heic")})
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "The image could not be decoded"
    assert client.get(f"{API}/images").json() == []
