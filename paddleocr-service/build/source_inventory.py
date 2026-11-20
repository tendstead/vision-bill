#!/usr/bin/env python3
"""Record actual recursive gitlinks, patched checkouts, and native build inputs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def record(root: Path, output: Path) -> None:
    paths = [root]
    status = git(root, "submodule", "status", "--recursive")
    for row in status.splitlines():
        if row.startswith("-"):
            raise ValueError(f"Uninitialized recursive source gitlink: {row[1:]}")
        paths.append(root / row[1:].split()[1])
    checkouts = {}
    for path in paths:
        checkouts[str(path.relative_to(root))] = {
            "head": git(path, "rev-parse", "HEAD").strip(),
            "gitlinks": [
                row
                for row in git(path, "ls-files", "--stage").splitlines()
                if row.startswith("160000 ")
            ],
            "patch": git(path, "diff", "--binary", "HEAD"),
        }
    patches = {}
    for pattern in ("*.patch", "*.smrev"):
        for path in root.rglob(pattern):
            if path.is_file():
                patches[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.write_text(
        json.dumps(
            {"checkouts": checkouts, "recursive_status": status, "patches_and_smrev": patches},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record(args.root, args.output)
