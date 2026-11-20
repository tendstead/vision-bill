"""Pure-stdlib PID-1 supervision and durable fences for managed native OCR.

An uncleared marker requires external recovery after every managed container has
been positively removed. No timeout, force option or ordinary startup clears one.
Legacy Ollama markers remain unconditional recovery fences after runtime removal.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Literal
from uuid import uuid4

Role = Literal["initializer", "ocr"]
ROLES = ("initializer", "ocr")
_ACTIVE = False
_INITIALIZER_LOCK_FD: int | None = None
_BASELINE: set[int] = set()


class InstallationBusy(RuntimeError):
    """An owned managed process still holds installation exclusion."""


class UncleanInstallation(RuntimeError):
    """External stopped-container confirmation is required before marker recovery."""


class ManagedInterrupted(BaseException):
    """Termination must finish owned descendant cleanup before releasing exclusion."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Managed process interrupted by signal {signum}")


def lock_path() -> Path:
    return Path(os.environ.get("OCR_INSTALLATION_LOCK_PATH", "/coord/installation.lock"))


@contextmanager
def installation_lock(path: Path, *, exclusive: bool, inheritable: bool = False) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InstallationBusy(
                "Managed installation is in use; stop the serving pair"
            ) from None
        os.set_inheritable(descriptor, inheritable)
        yield descriptor
    finally:
        os.close(descriptor)


def marker_path(lock: Path, role: str) -> Path:
    return lock.with_name(lock.stem + f".active-{role}.json")


def role_lock_path(lock: Path, role: str) -> Path:
    return lock.with_name(lock.stem + f".role-{role}.lock")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_fences(lock: Path, role: Literal["initializer", "ocr", "check"]) -> None:
    for marker in lock.parent.glob(lock.stem + ".active-*.json"):
        marker_role = marker.name.removeprefix(lock.stem + ".active-").removesuffix(".json")
        if (
            role == "initializer"
            or marker_role in {"initializer", role}
            or marker_role not in ROLES
        ):
            raise UncleanInstallation(
                "Uncleared managed marker; confirm all managed containers removed before external recovery"
            )
        # A serving peer may coexist only while that peer still holds its own
        # lifetime lock. A marker alone cannot prove its namespace is alive.
        try:
            with installation_lock(role_lock_path(lock, marker_role), exclusive=True):
                if not marker.exists():
                    continue  # The peer completed cleanup between listing and lock acquisition.
                raise UncleanInstallation(
                    "Uncleared peer marker; confirm all managed containers removed before external recovery"
                )
        except InstallationBusy:
            pass


def _descendants() -> set[int]:
    parents = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                parents[int(entry.name)] = int(
                    (entry / "stat").read_text().rsplit(")", 1)[1].split()[1]
                )
            except (FileNotFoundError, ProcessLookupError):
                pass
    pending = {os.getpid()}
    result: set[int] = set()
    while pending:
        children = {pid for pid, parent in parents.items() if parent in pending} - result
        result |= children
        pending = children
    return result


def _drain_owned() -> None:
    # PID 1 adopts session-detached orphans. Never clear the durable marker while
    # any owned descendant (including an adopted zombie) can still exist.
    while remaining := _descendants() - _BASELINE:
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in remaining:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        time.sleep(0.01)


@contextmanager
def managed_installation(role: Role, lock: Path) -> Iterator[None]:
    global _ACTIVE, _BASELINE, _INITIALIZER_LOCK_FD
    if role not in ROLES:
        raise ValueError("Unknown managed native role")
    with (
        installation_lock(lock, exclusive=role == "initializer") as lock_fd,
        installation_lock(role_lock_path(lock, role), exclusive=True),
    ):
        check_fences(lock, role)
        if os.getpid() != 1:
            raise RuntimeError(
                "Automatic native work requires the managed container supervisor as PID 1"
            )
        if _ACTIVE:
            raise RuntimeError("A managed native scope is already active")
        _BASELINE = _descendants()
        handlers = {
            number: signal.getsignal(number)
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
        }

        def interrupted(signum: int, frame: FrameType | None) -> None:
            raise ManagedInterrupted(signum)

        marker = marker_path(lock, role)
        created = False
        try:
            for number in handlers:
                signal.signal(number, interrupted)
            descriptor = os.open(
                marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
            )
            created = True
            with os.fdopen(descriptor, "w") as output:
                json.dump({"schema": 1, "role": role, "run_id": uuid4().hex}, output)
                output.flush()
                os.fsync(output.fileno())
            _sync_directory(lock.parent)
            _ACTIVE = True
            if role == "initializer":
                _INITIALIZER_LOCK_FD = lock_fd
            yield
        finally:
            for number in handlers:
                signal.signal(number, signal.SIG_IGN)
            # Any failure of positive cleanup deliberately leaves the marker.
            _drain_owned()
            _ACTIVE = False
            _INITIALIZER_LOCK_FD = None
            if created:
                marker.unlink()
                _sync_directory(lock.parent)
            for number, handler in handlers.items():
                signal.signal(number, handler)


def run_owned(
    command: list[str], *, quiet: bool = False, pass_fds: tuple[int, ...] = (),
) -> int:
    if not _ACTIVE or os.getpid() != 1:
        raise RuntimeError("Native command requires an active PID 1 managed installation scope")
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        return process.wait()
    finally:
        _drain_owned()
        if process is not None:
            process.wait()


def run_owned_qualification(command: list[str], *, quiet: bool = False) -> int:
    """Make the initializer's original exclusive lock available to its child."""
    descriptor = _INITIALIZER_LOCK_FD
    if descriptor is None:
        raise RuntimeError("Native qualification requires an active initializer")
    return run_owned(
        [*command, "--supervisor-fd", str(descriptor)],
        quiet=quiet,
        pass_fds=(descriptor,),
    )


def qualification_is_supervised(descriptor: int | None) -> bool:
    """Check the child still has the parent's locked installation inode."""
    if descriptor is None or os.getppid() != 1:
        return False
    path = lock_path()
    try:
        if not os.path.samestat(os.fstat(descriptor), path.stat()):
            return False
        if not marker_path(path, "initializer").is_file():
            return False
        with installation_lock(path, exclusive=False):
            return False  # The initializer's exclusive lock was not held.
    except InstallationBusy:
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--lock", type=Path, default=lock_path())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("An owned native executable must follow --")
    try:
        with managed_installation(args.role, args.lock):
            code = run_owned(command)
        raise SystemExit(code if code >= 0 else 128 - code)
    except ManagedInterrupted as exc:
        raise SystemExit(128 + exc.signum) from None


if __name__ == "__main__":
    main()
