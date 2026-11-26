"""Build a local ROCm candidate with the same target selection used by source setup."""

import json
import subprocess
import sys
from typing import Any

from scripts import ocr_source_up as source


def build_service(compose: tuple[str, ...]) -> dict[str, Any]:
    config = source.command(
        *compose, "config", "--no-env-resolution", "--format", "json", capture=True
    )
    service: dict[str, Any] = json.loads(config)["services"][source.NATIVE]
    return service


def main() -> int:
    compose = source.source_compose("rocm")
    try:
        native = build_service(compose)
        if native["build"]["args"].get("AMDGPU_TARGETS"):
            # An explicit cross-build does not require a local GPU.
            devices: dict[str, str] = {}
        else:
            # Host device metadata only describes the local Docker engine.
            source.need_host("rocm", build=True, automatic_inputs=False)
            if "docker-compose.rocm-wsl.yml" not in compose:
                source.select_rocm_render_node(compose)
                native = build_service(compose)
            _, devices = source.gpu_args("rocm", native)
        source.configure_rocm_build(native, devices)
        source.command(*compose, "build", source.NATIVE)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Local ROCm build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
