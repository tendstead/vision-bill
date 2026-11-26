"""Verify TheRock's generated GPU toolchains immediately after root configure."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_service.targets import validate_profile

CORE = {"comm-libs/rccl", "math-libs/BLAS/rocBLAS", "ml-libs/MIOpen"}
GPU_KEYS = ("AMDGPU_TARGETS", "GPU_TARGETS", "CMAKE_HIP_ARCHITECTURES")


def verify(build: Path, selected: set[str]) -> None:
    cache = dict(
        re.findall(
            r"^(THEROCK_\w+):[^=\n]+=(.*)$", (build / "CMakeCache.txt").read_text(), re.MULTILINE
        )
    )
    for scope in ("", "DIST_", "TEST_"):
        key = f"THEROCK_{scope}AMDGPU_TARGETS"
        if set(cache.get(key, "").split(";")) != selected:
            raise ValueError(f"{key}={cache.get(key)!r}, expected {sorted(selected)}")
        family = f"THEROCK_{scope}AMDGPU_FAMILIES"
        if cache.get(family) != "":
            raise ValueError(f"{family} must be empty for an explicit target list")
        print(f"{key}={cache[key]}")

    checked: set[str] = set()
    for group in ("comm-libs", "math-libs", "ml-libs"):
        for path in sorted((build / group).rglob("_toolchain.cmake")):
            project = path.parent.relative_to(build).as_posix()
            # This install-time consumer unbundles a fixed architecture even
            # though its toolchain uses DISABLE_AMDGPU_TARGETS.
            if project == "comm-libs/rccl-translate" and "gfx1250" not in selected:
                raise ValueError("rccl-translate requires a selected RCCL gfx1250 payload")
            contents = path.read_text()
            # DISABLE_AMDGPU_TARGETS is used for host-only projects. Some optional
            # GPU toolchains deliberately get an empty or filtered list at this pin.
            if not any(f"set({key} " in contents for key in GPU_KEYS):
                continue
            values: dict[str, str] = {}
            for key in GPU_KEYS:
                matches = re.findall(rf"^set\({key} (\S*) +CACHE STRING ", contents, re.MULTILINE)
                if len(matches) != 1:
                    raise ValueError(f"{project} has no unambiguous generated {key} assignment")
                targets = set(matches[0].split(";")) if matches[0] else set()
                if targets - selected or (project in CORE and targets != selected):
                    raise ValueError(f"{project} {key}={matches[0]!r}, selected {sorted(selected)}")
                values[key] = matches[0]
            if len(set(values.values())) != 1:
                raise ValueError(f"{project} generated inconsistent target lists: {values}")
            print(f"{project} GPU_TARGETS={values['GPU_TARGETS']}")
            checked.add(project)
    if missing := CORE - checked:
        raise ValueError(f"Missing required GPU toolchain(s): {', '.join(sorted(missing))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    try:
        profile = validate_profile(json.loads(args.profile.read_text()))
        verify(args.build, set(profile["candidate_targets"]))
    except (OSError, ValueError) as err:
        sys.exit(str(err))
