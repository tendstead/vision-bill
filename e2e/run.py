"""Own a unique Compose project, private dotenv/config, dynamic loopback ports and cleanup."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from e2e.prepare_fixtures import prepare
else:
    from prepare_fixtures import prepare

ROOT = Path(__file__).resolve().parents[1]
MARKER = "vision-bill-owned-e2e-v1"


class OwnedState(TypedDict):
    owner: str
    project: str
    directory: str
    env_file: str
    root: str
    policy: str
    image: NotRequired[str]
    override: NotRequired[str]
    state_file: NotRequired[str]
    admin_can_see_all: NotRequired[bool]


def validate_state(value: object) -> OwnedState:
    if not isinstance(value, dict):
        raise ValueError("invalid owned Compose state")  # noqa: TRY004 - invalid serialized state
    required = ("owner", "project", "directory", "env_file", "root", "policy")
    optional = ("image", "override", "state_file")
    if (any(not isinstance(value.get(key), str) or not value[key] for key in required)
        or any(key in value and not isinstance(value[key], str) for key in optional)
        or ("admin_can_see_all" in value and not isinstance(value["admin_can_see_all"], bool))
        or value["policy"] not in {"off", "enabled", "required", "fallback"}):
        raise ValueError("invalid owned Compose state fields")
    return cast(OwnedState, value)


def create_state(directory: Path) -> OwnedState:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config").mkdir()
    shutil.copyfile(ROOT / "e2e/config/config.yaml", directory / "config/config.yaml")
    prepare(ROOT, directory)
    env_file = directory / "compose.env"
    env_file.write_text("")
    return {
        "owner": MARKER,
        "project": "vb-e2e-" + uuid.uuid4().hex,
        "directory": str(directory),
        "env_file": str(env_file),
        "root": str(ROOT),
        "policy": "off",
    }


def compose(state: OwnedState, *arguments: str) -> tuple[list[str], dict[str, str]]:
    state = validate_state(state)
    if (
        state.get("owner") != MARKER
        or state.get("root") != str(ROOT)
        or not state.get("project", "").startswith("vb-e2e-")
    ):
        raise ValueError("refusing unowned Compose state")
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("COMPOSE_", "VB_E2E_"))}
    environment["VB_E2E_APP_IMAGE"] = state.get("image", state["project"] + "-app")
    environment["VB_E2E_FIXTURE_ROOT"] = state["directory"]
    environment["VB_E2E_CONFIG_DIR"] = str(Path(state["directory"]) / "config")
    policy = state["policy"]
    environment["VB_E2E_OCR_SERVICE_URL"] = "" if policy == "off" else "http://ocr-stub:9124"
    environment["VB_E2E_ADMIN_CAN_SEE_ALL"] = str(state.get("admin_can_see_all", True)).lower()
    environment.update(
        VB_E2E_OCR_ENABLED=str(policy != "off").lower(),
        VB_E2E_OCR_REQUIRED=str(policy == "required").lower(),
        VB_E2E_OCR_FALLBACK=str(policy == "fallback").lower(),
    )
    command = [
        "docker",
        "compose",
        "--env-file",
        state["env_file"],
        "--project-name",
        state["project"],
        "--project-directory",
        str(ROOT),
        "-f",
        str(ROOT / "docker-compose.e2e.yml"),
    ]
    if state.get("override"):
        command += ["-f", state["override"]]
    return command + list(arguments), environment


def call(state: OwnedState, *arguments: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    command, environment = compose(state, *arguments)
    return subprocess.run(
        command, env=environment, cwd=ROOT, check=True, text=True, capture_output=capture
    )


def endpoints(state: OwnedState) -> dict[str, str]:
    result = {
        "VB_E2E_STATE": state.get("state_file", str(Path(state["directory"]) / "state.json")),
        "VB_E2E_PROJECT": state["project"],
        "VB_E2E_EXTERNAL_STACK": "1",
        "VB_E2E_POLICY": state["policy"],
    }
    for name, service, port in [
        ("BASE_URL", "app", "8080"),
        ("STUB_URL", "llm-stub", "9123"),
        ("OCR_URL", "ocr-stub", "9124"),
        ("PG_URL", "pg", "5432"),
    ]:
        address = call(state, "port", service, port, capture=True).stdout.strip()
        result["VB_E2E_" + name] = (
            ("http://" if name != "PG_URL" else "postgresql://vision_bill:vision_bill@")
            + address
            + ("/vision_bill" if name == "PG_URL" else "")
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "up", "down", "exec", "env"])
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--policy", choices=["off", "enabled", "required", "fallback"], default="off"
    )
    parser.add_argument("--override", type=Path)
    parser.add_argument("--no-build", action="store_true")
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command.pop(0)
    if args.action in ("run", "up"):
        if args.action == "up" and args.state is None:
            parser.error("up requires --state in a new private directory")
        directory = args.state.parent if args.state else Path(tempfile.mkdtemp(prefix="vb-e2e-"))
        state = create_state(directory)
        state["policy"] = args.policy
        if args.override:
            state["override"] = str(args.override.resolve())
        path = args.state or directory / "state.json"
        state["state_file"] = str(path.resolve())
        path.write_text(json.dumps(state, indent=2))
    else:
        if args.state is None:
            parser.error("an explicit owned --state is required")
        state = validate_state(json.loads(args.state.read_text()))

    def terminate(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.action == "down":
            call(state, "down", "--volumes", "--remove-orphans")
            return 0
        if args.action in ("up", "run"):
            call(state, "up", "--detach", *([] if args.no_build else ["--build"]))
        environment = {**os.environ, **endpoints(state)}
        if args.action in ("run", "up"):
            subprocess.run(
                [sys.executable, str(ROOT / "e2e/wait_ready.py")], env=environment, check=True
            )
        if args.action in ("env", "up"):
            print(
                json.dumps(
                    {k: v for k, v in environment.items() if k.startswith("VB_E2E_")}, indent=2
                )
            )
            return 0
        if not command:
            parser.error("run/exec requires a command after --")
        return subprocess.run(command, env=environment, cwd=ROOT, check=False).returncode
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    finally:
        if args.action == "run":
            try:
                call(state, "logs", "--no-color")
            finally:
                call(state, "down", "--volumes", "--remove-orphans")


if __name__ == "__main__":
    raise SystemExit(main())
