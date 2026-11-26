"""Audit RCCL's generated Ninja commands before building its device objects."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_service.targets import validate_profile

# RCCL's custom language invokes this driver directly for both phases. Match
# executable and argument boundaries, so e.g. a host source named gfx942.cpp
# or a path for a generated object cannot count as a requested architecture.
DRIVER = re.compile(r"""(?:^|[\s/"'])rccl-device-compile["']?\s+--(compile|link)\b""")
COMPILER = re.compile(r"""(?:^|[\s/"'])(?:hipcc|(?:amd)?clang(?:\+\+)?(?:-\d+)?)["']?\s""")
ARCH = re.compile(r"""(?:^|\s)["']?--arch(?:=|\s+)["']?([^\s"']+)""")
OFFLOAD = re.compile(r"""(?:^|\s)["']?--(?:offload-arch|amdgpu-target)(?:=|\s+)["']?([^\s"']+)""")


def verify(build: Path, selected: set[str]) -> None:
    observed: dict[str, set[str]] = {"compile": set(), "link": set(), "HIP": set()}
    # Same default targets as TheRock's cmake --build. Ninja expands rule and
    # edge variables even for custom commands absent from compile_commands.json.
    # Stream the graph: a multi-target RCCL build has thousands of large commands.
    with subprocess.Popen(
        ["ninja", "-t", "commands"],
        cwd=build / "comm-libs/rccl/build",
        stdout=subprocess.PIPE,
        text=True,
    ) as ninja:
        assert ninja.stdout is not None
        for line in ninja.stdout:
            for command in line.split(" && "):
                if driver := DRIVER.search(command):
                    arguments = ARCH.findall(command)
                    observed[driver[1]].update(arguments or ["<missing --arch>"])
                elif "--offload-host-only" not in command and "--cuda-host-only" not in command:
                    if (
                        "--offload-arch" in command or "--amdgpu-target" in command
                    ) and COMPILER.search(command):
                        observed["HIP"].update(OFFLOAD.findall(command))
        if ninja.wait():
            raise ValueError("Ninja could not read the RCCL device build plan")

    for phase, targets in observed.items():
        print(f"RCCL {phase} targets={';'.join(sorted(targets))}")
        if targets - selected or (phase != "HIP" and targets != selected):
            raise ValueError(f"RCCL {phase} targets={sorted(targets)}, selected {sorted(selected)}")


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
