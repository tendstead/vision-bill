"""CPU-compatible installation lock API, shared with the standalone supervisor."""

import fcntl
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .supervision import InstallationBusy, UncleanInstallation, installation_lock, lock_path

__all__ = ["InstallationBusy", "installation_lock", "lock_path", "maintenance_handoff"]


class OutstandingHandoff(UncleanInstallation):
    """A handoff journal records owned work that has not been settled."""


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


def inspect_stopped_handoff(lock: Path) -> str:
    """Inspect both journals under the existing lock, even on a read-only volume."""
    fd = None
    try:
        try:
            fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            for suffix in (".json", ".ocr-only.json"):
                try:
                    lock.with_suffix(suffix).lstat()
                except FileNotFoundError:
                    continue
                return f"invalid {lock.name}"
            return "busy" if os.path.lexists(lock) else "clean"
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o177
            or info.st_nlink != 1
        ):
            return f"invalid {lock.name}"
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        for suffix, ocr_only in ((".json", False), (".ocr-only.json", True)):
            journal = lock.with_suffix(suffix)
            try:
                _require_clean_journal(journal, ocr_only=ocr_only)
            except OutstandingHandoff:
                return f"pending {journal.name}"
            except UncleanInstallation:
                return f"invalid {journal.name}"
        if not os.path.samestat(os.fstat(fd), lock.lstat()):
            return "busy"
        return "clean"
    except BlockingIOError:
        return "busy"
    except OSError:
        return f"invalid {lock.name}"
    finally:
        if fd is not None:
            os.close(fd)


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
            or not isinstance(saved["models"], list)
            or type(saved["unsettled"]) is not bool
            or type(saved["ocr_pending"]) is not bool
        ):
            raise ValueError("Invalid handoff journal state")
        if saved["models"] or saved["unsettled"] or saved["ocr_pending"]:
            raise OutstandingHandoff("Outstanding handoff work")
    except (OSError, ValueError) as exc:
        raise UncleanInstallation(
            "Unsafe handoff journal; preserve and externally recover recorded resources before maintenance"
        ) from exc
    finally:
        os.close(descriptor)
