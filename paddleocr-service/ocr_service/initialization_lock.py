"""CPU-compatible installation lock API, shared with the standalone supervisor."""

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .supervision import InstallationBusy, UncleanInstallation, installation_lock, lock_path

__all__ = ["InstallationBusy", "installation_lock", "lock_path", "maintenance_handoff"]


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate handoff journal key")
        result[key] = value
    return result


@contextmanager
def maintenance_handoff(path: Path | None) -> Iterator[None]:
    """Hold the original app lock; clean v2 evidence never proves external residency.

    This is only a no-outstanding-owned-work check. Legacy/unknown journals require
    external recovery and are never rewritten, adopted, or cleared here.
    """
    if path is None:
        yield
        return
    with installation_lock(path, exclusive=True):
        _require_clean_journal(path.with_suffix(".json"))
        _require_clean_journal(path.with_suffix(".ocr-only.json"), ocr_only=True)
        yield


def _require_clean_journal(journal: Path, *, ocr_only: bool = False) -> None:
    try:
        descriptor = os.open(journal, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise UncleanInstallation(
            "Unsafe handoff journal path; preserve and externally recover before maintenance"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or info.st_size > 1024 * 1024
        ):
            raise ValueError("Unsafe handoff journal file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            saved = json.load(stream, object_pairs_hook=_unique_pairs)
        keys = {
            "version",
            "coordinator",
            "resources",
            "generation",
            "models",
            "unsettled",
            "ocr_pending",
        }
        if (
            not isinstance(saved, dict)
            or set(saved) != keys
            or type(saved["version"]) is not int
            or saved["version"] != 2
            or saved["coordinator"] != "vision-bill-model-handoff-v2"
        ):
            raise ValueError("Unknown handoff journal schema")
        resources = saved["resources"]
        if (
            not isinstance(resources, dict)
            or set(resources) != {"llm", "ocr"}
            or any(
                not isinstance(value, str) or not value.strip() or value == "unsupported"
                for value in resources.values()
            )
        ):
            raise ValueError("Invalid handoff resource identities")
        if ocr_only and resources["llm"] != "external-unmanaged":
            raise ValueError("Invalid OCR-only handoff resource identity")
        if (
            type(saved["generation"]) is not int
            or saved["generation"] < 1
            or saved["models"] != []
            or saved["unsettled"] is not False
            or saved["ocr_pending"] is not False
        ):
            raise ValueError("Outstanding handoff work")
    except (OSError, ValueError) as exc:
        raise UncleanInstallation(
            "Unsafe handoff journal; preserve and externally recover recorded resources before maintenance"
        ) from exc
    finally:
        os.close(descriptor)
