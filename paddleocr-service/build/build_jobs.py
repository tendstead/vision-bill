"""Choose native build parallelism inside the Linux builder."""

import os
import re
import sys
from pathlib import Path

BYTES_PER_JOB = 4 * 1024**3


def read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def build_jobs(requested: str, proc: Path = Path("/proc"), groups: Path = Path("/sys/fs/cgroup")) -> int:
    if requested != "auto":
        if not re.fullmatch(r"[1-9][0-9]*", requested):
            raise ValueError("BUILD_JOBS must be auto or a positive integer")
        return int(requested)

    limits = [len(os.sched_getaffinity(0))]
    memory = []
    available = re.search(r"^MemAvailable:\s+(\d+) kB$", read(proc / "meminfo"), re.MULTILINE)
    if available:
        memory.append(int(available[1]) * 1024)

    for entry in read(proc / "self/cgroup").splitlines():
        _, controllers, group = entry.split(":", 2)
        # A private cgroup namespace can expose only its root, with membership
        # expressed relative to a different mount. Always inspect that root too.
        relative = Path(group.lstrip("/"))
        hierarchy = [Path(".")] if ".." in relative.parts else [relative, *relative.parents]
        for parent in hierarchy:
            if not controllers:
                node = groups / parent
                quota = read(node / "cpu.max").split()
                bound = read(node / "memory.max")
                used = read(node / "memory.current")
            else:
                names = controllers.split(",")
                quota = []
                bound = used = ""
                if "cpu" in names:
                    for folder in (controllers, "cpu"):
                        node = groups / folder / parent
                        if (node / "cpu.cfs_quota_us").is_file():
                            quota = [read(node / "cpu.cfs_quota_us"), read(node / "cpu.cfs_period_us")]
                            break
                if "memory" in names:
                    node = groups / "memory" / parent
                    bound = read(node / "memory.limit_in_bytes")
                    used = read(node / "memory.usage_in_bytes")
            if len(quota) == 2 and all(part.isdecimal() for part in quota) and int(quota[1]) > 0:
                limits.append(int(quota[0]) // int(quota[1]))
            if bound.isdecimal() and used.isdecimal():
                memory.append(int(bound) - int(used))
    limits.append(min(memory, default=BYTES_PER_JOB) // BYTES_PER_JOB)
    return max(1, min(limits))


if __name__ == "__main__":
    try:
        print(build_jobs(sys.argv[1]))
    except ValueError as exc:
        sys.exit(str(exc))
