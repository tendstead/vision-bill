"""Exercise the operational script without starting Docker or contacting a database."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pg-backup.sh"


@pytest.fixture
def backup_cli(tmp_path: Path):
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "entry = {'args': args}\n"
        "if 'pg_restore' in args: entry['stdin'] = sys.stdin.read()\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as log:\n"
        "    log.write(json.dumps(entry) + '\\n')\n"
        "if args[0] == 'compose': print('postgres:test')\n"
        "elif 'pg_dump' in args:\n"
        "    print('test-owned-dump')\n"
        "    sys.exit(int(os.environ.get('DUMP_EXIT', '0')))\n"
    )
    docker.chmod(0o700)
    (tmp_path / ".env").write_text(
        "PG__USER=test_user\nPG__PASSWORD=test_password\nPG__DB=test_db\n"
        "PG__HOST=test.invalid\nPG__PORT=5432\n"
    )
    calls = tmp_path / "calls.jsonl"
    env = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
    env.pop("backup_image", None)
    env.update(PATH=f"{tmp_path}:/usr/bin:/bin", DOCKER_CALLS=str(calls))

    def run(*args: str, **overrides: str):
        result = subprocess.run(
            ["bash", str(SCRIPT), *args], cwd=tmp_path, env=env | overrides,
            text=True, capture_output=True, check=False,
        )
        entries = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, entries

    return run


@pytest.mark.parametrize("operation", ["--backup", "--restore"])
def test_default_operation_resolves_image_before_running(backup_cli, tmp_path, operation):
    backup = tmp_path / "receipt.dump"
    if operation == "--restore":
        backup.write_text("test-owned-dump\n")
    result, calls = backup_cli(operation, str(backup))

    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    args = calls[1]["args"]
    assert args[:4] == ["run", "--rm", "--network", "host"]
    assert "postgres:test" in args
    assert "test_password" not in args
    if operation == "--backup":
        assert "pg_dump" in args
        assert backup.read_text() == "test-owned-dump\n"
        assert backup.stat().st_mode & 0o777 == 0o600
    else:
        assert "pg_restore" in args
        assert calls[1]["stdin"] == "test-owned-dump\n"


@pytest.mark.parametrize("option", ["--env-file", "--network"])
def test_documented_option_is_accepted(backup_cli, tmp_path, option):
    value = "private-net"
    if option == "--env-file":
        alternate = tmp_path / "custom.env"
        (tmp_path / ".env").rename(alternate)
        value = str(alternate)
    result, calls = backup_cli("--backup", str(tmp_path / "out.dump"), option, value)

    assert result.returncode == 0, result.stderr
    if option == "--env-file":
        assert calls[0]["args"][1:3] == ["--env-file", value]
    else:
        assert calls[1]["args"][2:4] == ["--network", value]


def test_failed_dump_is_not_published(backup_cli, tmp_path):
    backup = tmp_path / "failed.dump"
    result, _ = backup_cli("--backup", str(backup), DUMP_EXIT="3")

    assert result.returncode == 3
    assert not backup.exists()
    assert not list(tmp_path.glob("failed.dump.tmp.*"))


def test_existing_backup_is_preserved(backup_cli, tmp_path):
    backup = tmp_path / "existing.dump"
    backup.write_text("keep existing bytes")
    result, calls = backup_cli("--backup", str(backup))

    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    assert backup.read_text() == "keep existing bytes"
    assert all(entry["args"][0] != "run" for entry in calls)


def test_restore_replace_is_explicit(backup_cli, tmp_path):
    backup = tmp_path / "restore.dump"
    backup.write_text("test-owned-dump\n")
    result, calls = backup_cli("--restore", str(backup), "--replace")

    assert result.returncode == 0, result.stderr
    assert "--clean" in calls[1]["args"]
    assert "--if-exists" in calls[1]["args"]
