"""Run explicit standalone or paired source qualification under the native supervisor."""

import argparse
import sys
from pathlib import Path

from .initialization_lock import maintenance_handoff
from .supervision import (
    ManagedInterrupted,
    lock_path,
    managed_installation,
    run_owned_qualification,
)

APP_CONFIG = Path("/app/config")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", required=True, choices=["cuda", "rocm"])
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--maintenance-intent",
        required=True,
        choices=["external-llm-quiesced", "separate-ocr-device"],
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--handoff-lock",
        type=Path,
        help="Paired service's original app lock inside its shared /app/config mount",
    )
    mode.add_argument("--standalone", action="store_true", help="No app handoff coordinator")
    args = parser.parse_args(argv)
    handoff = args.handoff_lock
    if handoff is not None and (
        not handoff.is_absolute()
        or not handoff.parent.resolve().is_relative_to(APP_CONFIG.resolve())
        or handoff == APP_CONFIG
    ):
        parser.error("Paired maintenance requires the original absolute app lock under /app/config")

    command = [
        sys.executable,
        "-m",
        "ocr_service.qualify",
        "--accelerator",
        args.accelerator,
        "--runtime",
        str(args.runtime),
        "--cases",
        str(args.cases),
        "--output",
        str(args.output),
    ]
    try:
        with maintenance_handoff(handoff), managed_installation("initializer", lock_path()):
            code = run_owned_qualification(command)
    except ManagedInterrupted as exc:
        raise SystemExit(128 + exc.signum) from None
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    except OSError:
        parser.exit(1, "Cannot open the original maintenance coordination files\n")
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
