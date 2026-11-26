"""The built image migrates offline before replacing PID 1 with the server."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"


@pytest.mark.parametrize("migration_exit", [0, 7])
def test_installed_migration_precedes_exec_without_package_resolution(tmp_path, migration_exit):
    commands = tmp_path / "bin"
    commands.mkdir()
    events = tmp_path / "events.jsonl"
    for name, exit_expression in [
        ("uv", "97"),
        ("alembic", "int(os.environ['MIGRATION_EXIT'])"),
        ("server", "23"),
    ]:
        executable = commands / name
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['ENTRYPOINT_EVENTS'], 'a') as log:\n"
            f"    log.write(json.dumps({{'tool': {name!r}, 'args': sys.argv[1:], "
            "'pid': os.getpid()}) + '\\n')\n"
            f"sys.exit({exit_expression})\n"
        )
        executable.chmod(0o700)

    arguments = ["two words", "", "literal$HOME;$(do-not-run)"]
    process = subprocess.Popen(
        [str(ENTRYPOINT), str(commands / "server"), *arguments],
        cwd=tmp_path,
        env={
            "PATH": str(commands),
            "ENTRYPOINT_EVENTS": str(events),
            "MIGRATION_EXIT": str(migration_exit),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    observed = [json.loads(line) for line in events.read_text().splitlines()]
    assert "uv" not in [event["tool"] for event in observed]
    assert observed[0]["tool"] == "alembic"
    assert observed[0]["args"] == ["upgrade", "head"]
    if migration_exit:
        assert [event["tool"] for event in observed] == ["alembic"]
        assert process.returncode == 7, (stdout, stderr)
    else:
        assert [event["tool"] for event in observed] == ["alembic", "server"]
        assert observed[1]["args"] == arguments
        assert observed[1]["pid"] == process.pid
        assert process.returncode == 23, (stdout, stderr)


@pytest.mark.parametrize(
    "mode,module",
    [
        ("init", "vision_bill.distribution.bootstrap"),
    ],
)
def test_prepared_modes_bypass_migrations(tmp_path, mode, module):
    commands = tmp_path / "bin"
    commands.mkdir()
    events = tmp_path / "events"
    python = commands / "python"
    python.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        f'Path({str(events)!r}).write_text("\\n".join(sys.argv[1:]))\n'
    )
    python.chmod(0o700)
    result = subprocess.run(
        [str(ENTRYPOINT), mode, "--help"],
        env={"PATH": str(commands)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert events.read_text().splitlines() == ["-m", module, "--help"]


def test_retired_provision_model_fails_without_any_command(tmp_path):
    result = subprocess.run(
        [str(ENTRYPOINT), "provision-model", "--bundled-model", "old-model"],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 64
    assert "external Ollama or OpenAI endpoint" in result.stderr
    assert not list(tmp_path.iterdir())
