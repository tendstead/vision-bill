#!/usr/bin/env python3
"""Conservative native inventory: opaque packed/JIT payloads remain unverified.

Checks actual LLVM offloading metadata, not filenames. Optional upstream-excluded
Lt/CK components are recorded by source provenance, not asserted RDNA2-capable.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_service.targets import DEFAULT_TARGETS, validate_profile

TARGETS = set(DEFAULT_TARGETS)


def require_targets(found: set[str], required: set[str]) -> None:
    if not required <= found:
        raise ValueError(
            "Required native target kernels missing: " + ",".join(sorted(required - found))
        )


def validate_compile_targets(path: Path, required_targets: set[str] = TARGETS) -> None:
    if path.suffix == ".json":
        commands = [
            row.get("command", " ".join(row.get("arguments", [])))
            for row in json.loads(path.read_text())
        ]
    else:
        commands = path.read_text().splitlines()
    matched = 0
    for command in commands:
        if "hipcc" not in command and "clang++" not in command:
            continue
        found = set(re.findall(r"--offload-arch=(gfx[0-9a-f]+)", command))
        if not found:
            continue
        if found != required_targets:
            raise ValueError("Native compile target flags differ from selected candidate set")
        matched += 1
    if not matched:
        raise ValueError("Native compile target commands are unavailable")


def inventory(roots: list[Path], tool: Path, required_targets: set[str] = TARGETS) -> dict[str, Any]:
    targets: set[str] = set()
    files: dict[str, str] = {}
    if not roots or not any(path.is_file() for path in roots):
        raise ValueError("Required host library is missing")
    roots = [path for path in roots if path.exists()]
    paths: set[Path] = set()
    for root in roots:
        if root.is_dir():
            paths.update(path.resolve() for path in root.rglob("*") if path.is_file())
        else:
            paths.add(root.resolve())
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError("Required host library is missing")
        with path.open("rb") as stream:
            files[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        # LLVM inspects code-object ELF notes and host fatbinary offloading metadata.
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic != b"\x7fELF":
            continue
        result = subprocess.run(
            [str(tool), "--offloading", str(path)],
            capture_output=True,
            check=True,
            text=True,
            timeout=120,
        )
        notes = subprocess.run(
            [str(tool.parent / "llvm-readelf"), "--notes", str(path)],
            capture_output=True,
            check=True,
            text=True,
            timeout=120,
        )
        found = set(
            re.findall(
                r"(?m)^\s*(?:arch\s+|amdhsa\.target:\s*amdgcn-amd-amdhsa--)(gfx[0-9a-f]+)(?=[:\s]|$)",
                result.stdout + notes.stdout,
            )
        )
        targets.update(found)
    if targets:
        require_targets(targets, required_targets)
    return {
        "targets": sorted(targets),
        "files": files,
        "offline_status": "verified" if required_targets <= targets else "unverified",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddle", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-log", type=Path, required=True)
    parser.add_argument("--target-profile", type=Path, required=True)
    args = parser.parse_args()
    target_profile = validate_profile(json.loads(args.target_profile.read_text()))
    selected = set(target_profile["candidate_targets"])
    tool = args.sdk / "lib/llvm/bin/llvm-objdump"
    validate_compile_targets(args.compile_log, selected)
    library = args.sdk / "lib"
    result = {
        "target_profile": target_profile,
        "paddle": inventory(list(args.paddle.rglob("*.so")), tool, selected),
        "rocblas": inventory(list(library.glob("librocblas.so*")) + [library / "rocblas"], tool, selected),
        "miopen": inventory(
            list(library.glob("libMIOpen.so*")) + [args.sdk / "share/miopen"], tool, selected
        ),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
