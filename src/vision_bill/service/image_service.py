from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import magic
import pillow_heif
from ocr_contract import limits
from ocr_contract.document_image import decode_working_png, validate_working_png
from PIL import Image, ImageOps

from ..config import ImageSettings
from ..helper.deadline import check_deadline
from ..helper.document_image import document_rgb
from ..model.db.image import ImagePart
from ..model.image import ImageInfo
from ..model.ocr import WorkingImage, WorkingImageMetadata
from ..model.review import OriginalAvailability
from ..model.workflow import DeletedResource

pillow_heif.register_heif_opener()  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/gif",
}


class UnsupportedImageTypeError(Exception):
    """Raised when uploaded file is not an accepted image type."""

    def __init__(self, detected_type: str):
        self.detected_type = detected_type
        super().__init__(f"Unsupported image type: {detected_type}")


class MagicService:
    """Wrapper for magic to only be loaded once"""

    def __init__(self) -> None:
        self.magic = magic.Magic(mime=True)


# Singleton instance — reuse across requests instead of
# reconstructing magic.Magic() every call
magic_service = MagicService()


class ImageService:
    """Handles image inspection and validation."""

    @staticmethod
    async def check_originals(parts: list[ImagePart]) -> list[OriginalAvailability]:
        """Read every original, including legacy paths, without moving or modifying it."""

        def inspect() -> list[OriginalAvailability]:
            result = []
            for part in parts:
                available = matches = False
                try:
                    digest = sha256()
                    with Path(part.image_path).open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    available = True
                    matches = part.source_sha256 is None or digest.hexdigest() == part.source_sha256
                except OSError:
                    pass
                result.append(
                    OriginalAvailability(
                        part_id=part.part_id, photo_index=part.photo_index, available=available, sha256_matches=matches
                    )
                )
            return result

        return await asyncio.to_thread(inspect)

    def __init__(self, settings: ImageSettings, sniff_chunk_size: int = 4096):
        self._sniff_chunk_size = sniff_chunk_size
        self._tmp_dir = Path(settings.tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir = Path(settings.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._max_edge = settings.thumbnail_max_edge
        self._quality = settings.thumbnail_quality

    def get_media_type(self, content: bytes) -> str:
        """
        Detect MIME type from raw bytes. Only needs a small chunk,
        so this is cheap even for large files.
        """
        chunk = content[: self._sniff_chunk_size]
        media_type = magic_service.magic.from_buffer(chunk)
        logger.debug("Detected media type: %s", media_type)
        return media_type

    def validate_and_inspect(self, content: bytes) -> ImageInfo:
        """
        Detects the media type and raises if it's not an accepted image type.
        Returns an ImageInfo with the detected type and byte size.
        """
        logger.info("Validating image content")
        media_type = self.get_media_type(content)

        if media_type not in ALLOWED_IMAGE_TYPES:
            logger.warning("Unsupported media type detected: %s", media_type)
            raise UnsupportedImageTypeError(media_type)

        logger.info("Image successfully validated as %s", media_type)
        return ImageInfo(
            media_type=media_type,
            size_bytes=len(content),
            content=content,
        )

    def store_original(
        self, content: bytes, submission_id: UUID, photo_index: int, media_type: str
    ) -> ImagePart:
        """Publish original bytes atomically without ever replacing an existing file.

        The caller supplies the detected MIME type. DB assigns source-part UUIDs
        after acceptance; this method supplies only the original hash and order.
        """
        extensions = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
            "image/heic": "heic",
            "image/heif": "heif",
            "image/gif": "gif",
        }
        if media_type not in extensions:
            raise UnsupportedImageTypeError(media_type)
        if photo_index < 0:
            raise ValueError("photo index must be nonnegative")
        directory = self._save_dir / "submissions" / str(submission_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{photo_index}.{extensions[media_type]}"
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        # Persist the new directory entries as well as the file contents.
        for parent in (directory, directory.parent, self._save_dir):
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return ImagePart(
            image_path=str(target),
            media_type=media_type,
            size_bytes=len(content),
            source_sha256=sha256(content).hexdigest(),
            photo_index=photo_index,
        )

    def store_tmp_image(self, content: bytes) -> Path:
        """Writes content to a unique temporary file and returns the path."""
        file_id = str(uuid.uuid4())
        # Use a consistent extension or derive it if needed; .png is safe for vision models
        tmp_path = self._tmp_dir / f"temp_{file_id}.png"

        with open(tmp_path, "wb") as f:
            f.write(content)

        return tmp_path

    def store_perm_image(self, tmp_path: Path, receipt_id: UUID) -> tuple[Path | None, Path | None]:
        """Move a tmp image (and its thumbnail) to permanent storage.

        The original goes to ``save_dir`` and its thumbnail to ``save_dir/thumbnails/``.
        Returns ``(original_path, thumbnail_path)``; either is ``None`` when absent.
        """
        if not tmp_path.exists():
            logger.warning("Tmp image %s does not exist - nothing to store", tmp_path)
            return (None, None)

        destination = self._save_dir / f"receipt_{receipt_id}{tmp_path.suffix}"
        if destination.exists():
            destination = (
                self._save_dir / f"{uuid.uuid4().hex}_receipt_{receipt_id}{tmp_path.suffix}"
            )

        shutil.move(str(tmp_path), str(destination))

        thumb_src = self._thumb_for(tmp_path)
        if thumb_src.exists():
            thumb_dest = self._thumb_for(destination)
            thumb_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(thumb_src), str(thumb_dest))
            return (destination, thumb_dest)
        return (destination, None)

    def delete_image(self, image_path: Path | None) -> bool:
        """Remove an image file from disk.

        Best-effort: returns True if a file was deleted, False if the path was
        missing or falsy. Never raises for a missing file.
        """
        if not image_path:
            return False
        if not image_path.exists():
            logger.warning("Image file %s does not exist - nothing to delete", image_path)
            return False
        image_path.unlink()
        logger.info("Deleted image file %s", image_path)
        thumb = self._thumb_for(image_path)
        if thumb.exists():
            thumb.unlink()
            logger.info("Deleted thumbnail %s", thumb)
        return True

    @staticmethod
    def _thumb_for(original: Path) -> Path:
        """Deterministic thumbnail path: a `thumbnails/` subfolder next to the
        original's directory, named <stem>.thumb.webp."""
        return original.parent / "thumbnails" / f"{original.stem}.thumb.webp"

    def generate_thumbnail(self, src_path: Path) -> Path | None:
        """Create <dir>/thumbnails/<stem>.thumb.webp for src_path; return it, or
        None on any failure.

        Fails soft: a thumbnail problem must never break upload/verify/analysis.
        Decodes by content (Pillow magic bytes), so the tmp file's .png name is fine.
        """
        try:
            src = Image.open(src_path)
            src.load()
            img = ImageOps.exif_transpose(src)
            img = document_rgb(img)
            img.thumbnail((self._max_edge, self._max_edge), Image.Resampling.LANCZOS)
            dest = self._thumb_for(src_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, format="WEBP", quality=self._quality)
            return dest
        except Exception as exc:  # noqa: BLE001
            logger.warning("Thumbnail generation failed for %s: %s", src_path, exc)
            return None

    def _working_path(self, image_id: UUID, view_id: UUID) -> Path:
        root = (self._save_dir / "working").absolute()
        target = root / str(image_id) / f"{view_id}.png"
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("invalid_working_artifact_path")
        return target

    def read_working_image(
        self,
        image_id: UUID,
        artifact_path: str,
        metadata: WorkingImageMetadata,
    ) -> bytes:
        limits.check_working_image_dimensions(metadata.width, metadata.height)
        expected = self._working_path(image_id, metadata.view_id)
        if Path(artifact_path).absolute() != expected or expected.is_symlink():
            raise ValueError("invalid_working_artifact_path")
        fd = os.open(expected, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limits.MAX_WORKING_PNG_BYTES:
                raise ValueError("invalid_working_artifact_type")
            raw = handle.read(info.st_size + 1)
            if len(raw) != info.st_size:
                raise ValueError("invalid_working_artifact_size")
        validate_working_png(raw, metadata)
        return raw

    async def save_working_image(
        self,
        image_id: UUID,
        working: WorkingImage,
        *,
        deadline: float,
    ) -> Path:
        """Isolated immutable publication; cancelled writers clean only their own file."""
        from time import monotonic

        limits.check_working_image_dimensions(working.width, working.height)
        metadata = WorkingImageMetadata.model_validate(
            working.model_dump(exclude={"png_base64"})
        )
        encoded = working.png_base64

        def save() -> tuple[Path, bool]:
            target = self._working_path(image_id, metadata.view_id)
            raw = decode_working_png(encoded)
            validate_working_png(raw, metadata)
            check_deadline(deadline)
            target.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    try:
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                        check_deadline(deadline)
                        os.link(temporary, target)
                        created = True
                    finally:
                        temporary.unlink(missing_ok=True)
                for parent in (target.parent, target.parent.parent, self._save_dir):
                    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                check_deadline(deadline)
                return target, created
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise

        def discard(task: asyncio.Task[tuple[Path, bool]]) -> None:
            if not task.cancelled() and task.exception() is None:
                target, created = task.result()
                if created:
                    target.unlink(missing_ok=True)

        task = asyncio.create_task(asyncio.to_thread(save))
        try:
            async with asyncio.timeout(max(0, deadline - monotonic())):
                target, _ = await asyncio.shield(task)
            return target
        except BaseException:
            task.add_done_callback(discard)
            raise

    def validate_deleted_files(self, deleted: DeletedResource, *, initial: bool = False) -> None:
        """Seal before deletion; recheck the committed owned identities before every retry."""
        for part in deleted.parts:
            path = Path(part.image_path)
            if path.absolute() != path.resolve():
                raise ValueError("invalid_original_artifact_path")
            if path.exists():
                actual = sha256(path.read_bytes()).hexdigest()
                if part.source_sha256 != actual and (not initial or part.source_sha256 is not None):
                    raise ValueError("original_artifact_identity_changed")
                part.source_sha256 = actual
            if initial:
                thumbnail = self._thumb_for(path)
                if part.thumbnail_path and Path(part.thumbnail_path).absolute() != thumbnail.absolute():
                    raise ValueError("invalid_thumbnail_artifact_path")
                if thumbnail.absolute() != thumbnail.resolve():
                    raise ValueError("invalid_thumbnail_artifact_path")
                deleted.thumbnail_sha256[str(thumbnail)] = sha256(thumbnail.read_bytes()).hexdigest() if thumbnail.exists() else None
        for name, expected_hash in deleted.thumbnail_sha256.items():
            thumbnail = Path(name)
            if thumbnail.absolute() != thumbnail.resolve():
                raise ValueError("invalid_thumbnail_artifact_path")
            if thumbnail.exists() and sha256(thumbnail.read_bytes()).hexdigest() != expected_hash:
                raise ValueError("thumbnail_artifact_identity_changed")
        for view in deleted.working_views:
            if deleted.image_id is None:
                raise ValueError("invalid_working_artifact_owner")
            expected = self._working_path(deleted.image_id, view.metadata.view_id)
            if Path(view.artifact_path).absolute() != expected or expected.absolute() != expected.resolve():
                raise ValueError("invalid_working_artifact_path")
            if expected.exists():
                self.read_working_image(deleted.image_id, view.artifact_path, view.metadata)

    def remove_deleted_files(self, deleted: DeletedResource) -> None:
        """Only a confirmed committed deletion authorizes unlink; failures retain metadata."""
        self.validate_deleted_files(deleted)
        paths = []
        for view in deleted.working_views:
            assert deleted.image_id is not None
            self.remove_unreferenced_working_image(deleted.image_id, view.artifact_path, view.metadata)
            paths.append(Path(view.artifact_path))
        for name in [*(part.image_path for part in deleted.parts), *deleted.thumbnail_sha256]:
            path = Path(name)
            path.unlink(missing_ok=True)
            paths.append(path)
        self._sync_deleted_parents(paths)

    @staticmethod
    def _sync_deleted_parents(paths: Sequence[Path]) -> None:
        # A prior attempt may have unlinked these files before syncing the directory entries.
        for parent in dict.fromkeys(path.absolute().parent for path in paths):
            try:
                parent.lstat()
            except FileNotFoundError:
                continue
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def remove_unreferenced_working_image(
        self,
        image_id: UUID,
        artifact_path: str,
        metadata: WorkingImageMetadata,
        *,
        deadline: float | None = None,
    ) -> None:
        """Caller must hold DB evidence that its own newly created file is unreferenced."""
        expected = self._working_path(image_id, metadata.view_id)
        if Path(artifact_path).absolute() != expected or expected.is_symlink():
            raise ValueError("invalid_working_artifact_path")
        if deadline is not None:
            check_deadline(deadline)
        expected.unlink(missing_ok=True)
