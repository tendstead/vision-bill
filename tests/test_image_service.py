import os
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from PIL import Image

from vision_bill.config import Settings
from vision_bill.model.image import ImageInfo
from vision_bill.service.image_service import ImageService, UnsupportedImageTypeError

RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000001")


@pytest.fixture
def mock_magic_service() -> Generator[MagicMock, None, None]:
    """Mock for the magic_service singleton."""
    with patch("vision_bill.service.image_service.magic_service") as mocked:
        # The service uses magic_service.magic.from_buffer
        mocked.magic.from_buffer = MagicMock()
        yield mocked


@pytest.fixture
def image_service(settings: Settings, mock_magic_service: MagicMock) -> ImageService:
    """Provides an ImageService instance."""
    return ImageService(settings.images)


@pytest.mark.parametrize(
    "content, expected_type",
    [
        (b"fake_jpeg_bytes", "image/jpeg"),
        (b"fake_png_bytes", "image/png"),
        (b"fake_webp_bytes", "image/webp"),
    ],
)
async def test_get_media_type_success(
    image_service: ImageService,
    mock_magic_service: MagicMock,
    content: bytes,
    expected_type: str,
) -> None:
    # Arrange
    mock_magic_service.magic.from_buffer.return_value = expected_type

    # Act
    result = image_service.get_media_type(content)

    # Assert
    assert result == expected_type
    mock_magic_service.magic.from_buffer.assert_called_once()


@pytest.mark.asyncio
async def test_validate_and_inspect_success(
    image_service: ImageService,
    mock_magic_service: MagicMock,
) -> None:
    # Arrange
    content = b"fake_png_bytes"
    mock_magic_service.magic.from_buffer.return_value = "image/png"

    # Act
    result = image_service.validate_and_inspect(content)

    # Assert
    assert isinstance(result, ImageInfo)
    assert result.media_type == "image/png"
    assert result.size_bytes == len(content)
    assert result.content == content


@pytest.mark.asyncio
async def test_validate_and_inspect_failure(
    image_service: ImageService,
    mock_magic_service: MagicMock,
) -> None:
    # Arrange
    content = b"fake_pdf_bytes"
    mock_magic_service.magic.from_buffer.return_value = "application/pdf"

    # Act & Assert
    with pytest.raises(UnsupportedImageTypeError) as exc_info:
        image_service.validate_and_inspect(content)

    assert "application/pdf" in str(exc_info.value)


def test_store_tmp_image(image_service: ImageService, settings: Settings) -> None:
    # Arrange
    content = b"fake-image-content"
    tmp_dir = Path(settings.images.tmp_dir)
    assert tmp_dir.exists()

    # Act
    saved_path = image_service.store_tmp_image(content)

    # Assert
    assert saved_path.is_absolute()
    assert saved_path.parent == tmp_dir
    assert saved_path.suffix == ".png"
    assert saved_path.exists()


def test_store_perm_image_moves_file(image_service: ImageService, settings: Settings) -> None:
    """store_perm_image moves the tmp file into the save dir under a stable name."""
    content = b"fake-image-content"
    tmp_path = image_service.store_tmp_image(content)
    save_dir = Path(settings.images.save_dir)

    original, thumb = image_service.store_perm_image(tmp_path, RECEIPT_ID)

    assert original is not None
    assert original == save_dir / f"receipt_{RECEIPT_ID}.png"
    assert original.exists()
    assert not tmp_path.exists()
    assert original.read_bytes() == content
    assert thumb is None  # fake (undecodable) content -> no thumbnail was made


def test_store_perm_image_missing_returns_none(
    image_service: ImageService,
    settings: Settings,
) -> None:
    """store_perm_image returns (None, None) when the tmp file no longer exists."""
    missing = Path(settings.images.tmp_dir) / "does_not_exist.png"

    original, thumb = image_service.store_perm_image(missing, RECEIPT_ID)

    assert original is None
    assert thumb is None


def test_heif_opener_registered() -> None:
    """Importing image_service registers the pillow-heif opener; a real PNG opens."""
    import vision_bill.service.image_service  # noqa: F401  (registers on import)

    buf = BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    buf.seek(0)
    assert Image.open(buf).size == (4, 4)
    # register_heif_opener() registers 'HEIF' in Image.OPEN and extensions in Image.EXTENSION
    assert "HEIF" in Image.OPEN
    assert ".heif" in Image.EXTENSION
    assert ".heic" in Image.EXTENSION


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def test_generate_thumbnail_creates_webp_in_thumbnails_dir(
    image_service: ImageService, settings: Settings, tmp_path: Path
) -> None:
    src = _write_image(tmp_path / "photo.png", (1000, 800), (10, 20, 30))

    result = image_service.generate_thumbnail(src)

    assert result is not None
    assert result == tmp_path / "thumbnails" / "photo.thumb.webp"
    assert result.exists()
    with Image.open(result) as im:
        w, h = im.size
    assert max(w, h) <= 512  # long edge capped
    assert abs((w / h) - (1000 / 800)) < 0.02  # aspect ratio preserved


def test_generate_thumbnail_never_upscales(image_service: ImageService, tmp_path: Path) -> None:
    src = _write_image(tmp_path / "tiny.png", (10, 10), (255, 0, 0))

    result = image_service.generate_thumbnail(src)

    assert result is not None
    with Image.open(result) as im:
        w, h = im.size
    assert max(w, h) <= 10  # small image is not enlarged to 512


def test_generate_thumbnail_returns_none_on_corrupt(
    image_service: ImageService, tmp_path: Path
) -> None:
    src = tmp_path / "bad.png"
    src.write_bytes(b"this is not an image")

    assert image_service.generate_thumbnail(src) is None
    assert not (tmp_path / "thumbnails" / "bad.thumb.webp").exists()


def test_store_perm_image_moves_thumb(image_service: ImageService, settings: Settings) -> None:
    """When a thumbnail exists, store_perm_image moves it to save_dir/thumbnails/."""
    from PIL import Image as PILImage

    tmp_src = Path(settings.images.tmp_dir) / "real.png"
    PILImage.new("RGB", (80, 60), (0, 0, 255)).save(tmp_src, format="PNG")
    image_service.generate_thumbnail(tmp_src)  # creates real.thumb.webp
    save_dir = Path(settings.images.save_dir)

    original, thumb = image_service.store_perm_image(tmp_src, RECEIPT_ID)

    assert original == save_dir / f"receipt_{RECEIPT_ID}.png"
    assert thumb == save_dir / "thumbnails" / f"receipt_{RECEIPT_ID}.thumb.webp"
    assert original is not None and original.exists()
    assert thumb is not None and thumb.exists()
    assert not tmp_src.exists()
    assert not (Path(settings.images.tmp_dir) / "thumbnails" / "real.thumb.webp").exists()


def test_delete_image_removes_thumb(image_service: ImageService, tmp_path: Path) -> None:
    """delete_image removes the thumbnail (in the thumbnails/ subfolder) when present."""
    from PIL import Image as PILImage

    src = tmp_path / "gone.png"
    PILImage.new("RGB", (20, 20), (1, 1, 1)).save(src, format="PNG")
    thumb = image_service.generate_thumbnail(src)
    assert thumb is not None and thumb.exists()

    assert image_service.delete_image(src) is True
    assert not src.exists()
    assert not thumb.exists()


def test_store_original_keeps_actual_bytes_hash_order_and_immutable_path(image_service, settings):
    from hashlib import sha256

    content = b"original bytes"
    part = image_service.store_original(content, RECEIPT_ID, 1, "image/jpeg")
    target = Path(settings.images.save_dir) / "submissions" / str(RECEIPT_ID) / "1.jpg"
    assert Path(part.image_path) == target
    assert target.read_bytes() == content
    assert part.source_sha256 == sha256(content).hexdigest()
    assert part.photo_index == 1
    assert part.part_id is None  # Only WorkflowDB assigns actual source-part identities.
    with pytest.raises(FileExistsError):
        image_service.store_original(b"changed", RECEIPT_ID, 1, "image/jpeg")
    assert target.read_bytes() == content
    assert list(target.parent.iterdir()) == [target]


def test_deleted_original_retry_syncs_distinct_existing_parents(image_service, tmp_path, monkeypatch):
    from vision_bill.model.db.image import ImagePart
    from vision_bill.model.workflow import DeletedResource

    monkeypatch.chdir(tmp_path)
    originals = [Path("originals/original.png"), tmp_path / "originals/second.png"]
    originals[0].parent.mkdir()
    for original in originals:
        original.write_bytes(b"original")
    thumbnail = image_service._thumb_for(originals[0])
    thumbnail.parent.mkdir()
    thumbnail.write_bytes(b"thumbnail")
    absent = tmp_path / "absent" / "original.png"
    deleted = DeletedResource(parts=[
        ImagePart(image_path=str(original)) for original in [*originals, absent]
    ])
    image_service.validate_deleted_files(deleted, initial=True)
    originals[0].unlink()  # The first attempt removed it before an unlink failure.
    synced = []
    real_fsync = os.fsync

    def track_fsync(fd):
        synced.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", track_fsync)
    image_service.remove_deleted_files(deleted)

    assert not any(path.exists() for path in [*originals, thumbnail, absent])
    assert sorted(synced) == sorted([
        originals[0].parent.stat().st_ino, thumbnail.parent.stat().st_ino
    ])


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P'])
def test_thumbnail_composites_transparent_document_over_white(image_service, tmp_path, mode):
    from PIL import Image as PILImage
    source = tmp_path / f'{mode}.png'
    if mode == 'RGBA':
        image = PILImage.new(mode, (40, 40), (0, 0, 0, 0))
        ink = (0, 0, 0, 255)
    elif mode == 'LA':
        image = PILImage.new(mode, (40, 40), (0, 0))
        ink = (0, 255)
    else:
        image = PILImage.new('P', (40, 40), 0)
        image.putpalette([0, 0, 0, 0, 0, 0] + [0] * 762)
        image.info['transparency'] = 0
        ink = 1
    for x in range(15, 25):
        for y in range(15, 25):
            image.putpixel((x, y), ink)
    image.save(source)
    original = source.read_bytes()
    thumb = image_service.generate_thumbnail(source)
    assert thumb is not None
    with PILImage.open(thumb) as decoded:
        assert min(decoded.getpixel((3, 3))[:3]) > 235
        assert max(decoded.getpixel((20, 20))[:3]) < 20
    assert source.read_bytes() == original


def test_relative_thumbnail_deletion_seals_paths_and_retries(image_service, tmp_path, monkeypatch):
    from vision_bill.model.db.image import ImagePart
    from vision_bill.model.workflow import DeletedResource
    monkeypatch.chdir(tmp_path)
    original = Path('uploads/original.png')
    original.parent.mkdir(exist_ok=True)
    original.write_bytes(b'original')
    thumb = image_service._thumb_for(original)
    thumb.parent.mkdir()
    thumb.write_bytes(b'thumbnail')
    deleted = DeletedResource(parts=[ImagePart(image_path=str(original), thumbnail_path=str(thumb))])
    image_service.validate_deleted_files(deleted, initial=True)
    original.unlink()
    image_service.remove_deleted_files(deleted)
    assert not thumb.exists()
    image_service.remove_deleted_files(deleted)


@pytest.mark.parametrize('bad', ['wrong', 'symlink', 'changed'])
def test_relative_thumbnail_deletion_retains_identity_guards(image_service, tmp_path, monkeypatch, bad):
    from vision_bill.model.db.image import ImagePart
    from vision_bill.model.workflow import DeletedResource
    monkeypatch.chdir(tmp_path)
    original = Path('original.png'); original.write_bytes(b'original')
    thumb = image_service._thumb_for(original); thumb.parent.mkdir(); thumb.write_bytes(b'thumbnail')
    deleted = DeletedResource(parts=[ImagePart(image_path=str(original), thumbnail_path=str(thumb))])
    if bad == 'wrong': deleted.parts[0].thumbnail_path = 'other.webp'
    elif bad == 'symlink':
        other = Path('other.webp'); other.write_bytes(b'thumbnail'); thumb.unlink(); thumb.symlink_to(other.absolute())
    else:
        image_service.validate_deleted_files(deleted, initial=True)
        thumb.write_bytes(b'changed')
    with pytest.raises(ValueError, match='thumbnail_artifact'):
        image_service.validate_deleted_files(deleted, initial=bad != 'changed')
    assert original.exists() and thumb.exists()


async def test_working_artifact_save_and_read_apply_exact_png_byte_cap(
    image_service, monkeypatch
):
    import base64
    import hashlib
    import time
    from uuid import uuid4

    from ocr_contract import WorkingImage, WorkingImageMetadata, limits

    with BytesIO() as buffer:
        Image.new("RGB", (3, 4), "white").save(buffer, "PNG")
        png = buffer.getvalue()
    view = WorkingImage(
        view_id=uuid4(),
        width=3,
        height=4,
        sha256=hashlib.sha256(png).hexdigest(),
        png_base64=base64.b64encode(png).decode(),
    )
    metadata = WorkingImageMetadata.model_validate(view.model_dump())
    monkeypatch.setattr(limits, "MAX_WORKING_PNG_BYTES", len(png))
    path = await image_service.save_working_image(
        RECEIPT_ID, view, deadline=time.monotonic() + 2
    )
    assert image_service.read_working_image(RECEIPT_ID, str(path), metadata) == png

    monkeypatch.setattr(limits, "MAX_WORKING_PNG_BYTES", len(png) - 1)
    with pytest.raises(ValueError, match="invalid_working_artifact_type"):
        image_service.read_working_image(RECEIPT_ID, str(path), metadata)
    with pytest.raises(ValueError, match="ocr_working_png_too_large"):
        await image_service.save_working_image(
            uuid4(), view, deadline=time.monotonic() + 2
        )
    assert path.read_bytes() == png


async def test_working_artifact_pixel_cap_precedes_pixel_loading(image_service, monkeypatch):
    import base64
    import hashlib
    import time
    from uuid import uuid4

    from ocr_contract import WorkingImage, WorkingImageMetadata, limits

    with BytesIO() as buffer:
        Image.new("RGB", (3, 4), "white").save(buffer, "PNG")
        png = buffer.getvalue()
    view = WorkingImage(
        view_id=uuid4(), width=3, height=4, sha256=hashlib.sha256(png).hexdigest(),
        png_base64=base64.b64encode(png).decode(),
    )
    metadata = WorkingImageMetadata.model_validate(view.model_dump())
    path = await image_service.save_working_image(RECEIPT_ID, view, deadline=time.monotonic() + 2)
    monkeypatch.setattr(limits, "MAX_IMAGE_PIXELS", 11)

    def unexpected_load(*args, **kwargs):
        pytest.fail("working artifact exceeded the pixel envelope before loading")

    monkeypatch.setattr(Image.Image, "load", unexpected_load)
    with pytest.raises(ValueError, match="ocr_invalid_working_image_dimensions"):
        image_service.read_working_image(RECEIPT_ID, str(path), metadata)
    with pytest.raises(ValueError, match="ocr_invalid_working_image_dimensions"):
        await image_service.save_working_image(uuid4(), view, deadline=time.monotonic() + 2)
