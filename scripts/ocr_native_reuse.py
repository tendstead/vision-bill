"""Evidence for reusing a locally built OCR image across app-only source upgrades."""

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


def build_fingerprint(
    root: Path, accelerator: str, build: dict[str, Any], selected_args: dict[str, str]
) -> str | None:
    """Hash the files admitted by the native context's deny-by-default .dockerignore.

    Unsupported context layouts disable reuse. Directory traversal rules must reopen
    individual files explicitly, so unrelated local model caches are never scanned.
    """
    if set(build) - {"context", "dockerfile", "args", "additional_contexts"}:
        return None
    if "context" not in build or Path(build["context"]).resolve() != root.resolve():
        return None
    if build.get("dockerfile") != f"Dockerfile.{accelerator}":
        return None
    # CUDA revalidates the exact lock and wheel from its resolver-managed contexts.
    expected_contexts = {"cuda-locks", "cuda-wheel"} if accelerator == "cuda" else set()
    if set(build.get("additional_contexts", {})) != expected_contexts:
        return None
    if (root / f"Dockerfile.{accelerator}.dockerignore").exists():
        return None
    ignore = root / ".dockerignore"
    if not ignore.is_file():
        return None
    rules = [
        line
        for raw in ignore.read_text().splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    ]
    if not rules or rules[0] != "*":
        return None
    paths = {ignore, root / f"Dockerfile.{accelerator}"}
    for index, rule in enumerate(rules):
        if not rule.startswith("!"):
            continue
        pattern = rule[1:]
        if (
            not re.fullmatch(r"[\w.*-]+(?:/[\w.*-]+)*/?", pattern)
            or ".." in pattern
            or "**" in pattern
        ):
            return None
        if pattern.endswith("/"):
            if rules[index + 1 : index + 2] != [f"{pattern}**"]:
                return None
            paths.add(root / pattern)
            continue
        for path in root.glob(pattern):
            if path.is_dir():
                return None
            paths.add(path)
    inputs = {}
    for path in paths:
        mode = path.lstat().st_mode
        if (not stat.S_ISREG(mode) and not stat.S_ISDIR(mode)) or path.resolve() != path.absolute():
            return None
        digest = ""
        if stat.S_ISREG(mode):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        inputs[path.relative_to(root).as_posix()] = (stat.S_IMODE(mode), digest)
    if accelerator == "cuda":
        # The CUDA build goes through the checkout's Makefile.
        recipe = root.parent / "Makefile"
        if not recipe.is_file() or recipe.is_symlink():
            return None
        inputs["../Makefile"] = (
            stat.S_IMODE(recipe.stat().st_mode),
            hashlib.sha256(recipe.read_bytes()).hexdigest(),
        )
    settings = {**build, "args": {**build.get("args", {}), **selected_args}}
    proof = {"version": 1, "accelerator": accelerator, "inputs": inputs, "build": settings}
    return hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()


def admitted_candidate(
    previous: dict[str, Any],
    fingerprint: str | None,
    device_environment: dict[str, str],
    release: Path,
    sources: dict[str, str],
    tag: str,
) -> str | None:
    proof = previous.get("native_build")
    if (
        not isinstance(proof, dict)
        or not fingerprint
        or proof.get("inputs_sha256") != fingerprint
        or not re.fullmatch(r"[0-9a-f]{32}", str(proof.get("device_uuid", "")))
        or previous.get("device_environment") != device_environment
    ):
        return None
    bundle = json.loads((release / "bundle.json").read_bytes())
    if bundle.get("accelerator") == "cuda" and not re.fullmatch(
        r"[0-9a-f]{64}", str(proof.get("cuda_lock_sha256", ""))
    ):
        return None
    image_id = previous["image_ids"]["paddleocr-service"]
    if (
        bundle.get("image_identity") != image_id
        or bundle.get("source_files") != sources
        or bundle.get("image_config_sha256") != previous["image_config_sha256"]
    ):
        return None
    if tag != f"vision-bill-ocr-{bundle['accelerator']}:source-{image_id[7:]}":
        return None
    return tag


def copy_proven_config(previous: dict[str, Any], source: Path, destination: Path) -> str:
    config = (source / "image-config.json").read_bytes()
    digest = hashlib.sha256(config).hexdigest()
    if digest != previous["image_config_sha256"]:
        raise ValueError("Previously admitted native image configuration changed")
    (destination / "image-config.json").write_bytes(config)
    return f"sha256:{digest}"
