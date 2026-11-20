"""Fault injection confined to one Compose project's explicitly mounted test volume."""

from pathlib import Path
from uuid import UUID


def remove_test_file(root: Path, image_id: str, kind: str, photo_index: int = 0) -> None:
    identity = str(UUID(image_id))
    if type(photo_index) is not int or photo_index < 0 or kind not in ("original", "thumbnail"):
        raise ValueError("invalid owned file request")
    directory = root.resolve() / "submissions" / identity
    if kind == "thumbnail":
        candidates = [directory / "thumbnails" / f"{photo_index}.thumb.webp"]
    else:
        candidates = [
            directory / f"{photo_index}.{extension}"
            for extension in ("png", "jpg", "webp", "heic", "heif", "gif")
        ]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1 or not existing[0].resolve().is_relative_to(directory):
        raise ValueError("owned file missing or ambiguous")
    existing[0].unlink()
