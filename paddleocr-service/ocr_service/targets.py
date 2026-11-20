"""Versioned build candidates; syntax admission is never hardware qualification."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = ("gfx1030", "gfx1031", "gfx1032", "gfx1034")
_TARGET = re.compile(r"gfx[0-9a-f]{3,8}")


def _validated(targets: Any) -> list[str]:
    if (
        not isinstance(targets, list)
        or not 1 <= len(targets) <= 32
        or any(not isinstance(item, str) or not _TARGET.fullmatch(item) for item in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("Invalid candidate target profile")
    return sorted(targets)


def parse_targets(argument: str) -> list[str]:
    return _validated(argument.split(";") if argument else list(DEFAULT_TARGETS))


def validate_profile(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "candidate_targets"}
        or type(value.get("schema")) is not int
        or value["schema"] != 1
    ):
        raise ValueError("Invalid candidate target profile")
    return {"schema": 1, "candidate_targets": _validated(value["candidate_targets"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    targets = parse_targets(args.targets)
    args.output.write_text(json.dumps({"schema": 1, "candidate_targets": targets}, sort_keys=True) + "\n")
    print(";".join(targets))
