"""Prepared installation files are private, persistent, and fail closed."""

import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def initialize(tmp_path, *extra, runtime_environment=None):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AUTH__", "LLM__", "PG__", "VISION_BILL_CONFIG"))
    }
    env["PYTHONPATH"] = str(ROOT / "src")
    env.update(
        {
            "LLM__PROVIDER": "ollama",
            "LLM__HOST": "http://external:11434",
            "LLM__MODEL_NAME": "operator-model",
        }
        if runtime_environment is None
        else runtime_environment
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "vision_bill.distribution.bootstrap",
            "--config-dir",
            str(tmp_path / "config"),
            "--postgres-password-file",
            str(tmp_path / "pg-secret" / "password"),
            *extra,
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_new_install_is_private_stable_and_settings_import_free(tmp_path):
    result = initialize(tmp_path)
    assert result.returncode == 0, result.stderr
    config_file = tmp_path / "config/config.yaml"
    config = yaml.safe_load(config_file.read_text())
    admin = yaml.safe_load((tmp_path / "config/initial-admin.yaml").read_text())
    pg_password = (tmp_path / "pg-secret/password").read_text().strip()
    assert len(config["auth"]["secret_key"]) >= 40
    assert len(pg_password) >= 40
    assert len(admin["password"]) >= 40
    assert len({config["auth"]["secret_key"], pg_password, admin["password"]}) == 3
    assert config["auth"]["bootstrap_username"] == admin["username"]
    assert config["auth"]["bootstrap_password"] == admin["password"]
    assert config["pg"]["password"] == pg_password
    assert config["pg"]["user"] == "vision_bill"
    assert config["llm"]["model_name"] == "operator-model"
    assert config["llm"]["host"] == "http://external:11434"
    assert "distribution" not in config
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert initialize(tmp_path).returncode == 0
    assert {p: p.read_bytes() for p in before} == before
    for path in tmp_path.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    for secret in [config["auth"]["secret_key"], pg_password, admin["password"]]:
        assert secret not in result.stdout + result.stderr
    second = tmp_path / "other"
    second.mkdir()
    assert initialize(second).returncode == 0
    assert yaml.safe_load((second / "config/config.yaml").read_text())["auth"] != config["auth"]


@pytest.mark.parametrize("artifact", [".bootstrap-pending", "initial-admin.yaml"])
def test_interrupted_install_does_not_reset(tmp_path, artifact):
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / artifact).write_text("partial-secret")
    result = initialize(tmp_path)
    assert result.returncode != 0
    assert (directory / artifact).read_text() == "partial-secret"
    assert not (directory / "config.yaml").exists()
    assert "partial-secret" not in result.stdout + result.stderr


@pytest.mark.parametrize("api_key", ["legacy-api-secret", ""])
def test_existing_legacy_yaml_preserves_credentials_without_inventing_admin(tmp_path, api_key):
    directory = tmp_path / "config"
    directory.mkdir()
    content = yaml.safe_dump(
        {
            "llm": {
                "provider": "openai",
                "host": "http://external",
                "api_key": api_key,
                "model_name": "operator-model",
                "temperature": 0.1,
            },
            "pg": {"user": "runtime-login", "password": "legacy-db-secret", "db": "legacy"},
            "auth": {"secret_key": "legacy-auth-secret", "allow_registration": False},
        }
    )
    (directory / "config.yaml").write_text(content)
    assert initialize(tmp_path).returncode == 0
    assert (directory / "config.yaml").read_text() == content
    assert (tmp_path / "pg-secret/password").read_text().strip() == "legacy-db-secret"
    assert not (directory / "initial-admin.yaml").exists()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert initialize(tmp_path).returncode == 0
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("kind,api_key", [("openai", "none"), ("ollama", "")])
async def test_fresh_generated_llm_settings_construct_real_provider_without_http(
    tmp_path,
    settings,
    monkeypatch,
    kind,
    api_key,
):
    from vision_bill import config
    from vision_bill.config import LLMSettings
    from vision_bill.provider.factory import get_llm_provider
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.openai import OpenAIProvider

    result = initialize(
        tmp_path,
        runtime_environment={
            "LLM__PROVIDER": kind,
            "LLM__HOST": "http://local/v1",
            "LLM__MODEL_NAME": "operator-model",
        },
    )
    assert result.returncode == 0, result.stderr
    generated = yaml.safe_load((tmp_path / "config/config.yaml").read_text())
    llm_settings = LLMSettings.model_validate(generated["llm"])
    monkeypatch.setattr(config, "settings", settings)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    def unexpected_connection(*args, **kwargs):
        pytest.fail("provider construction must not make a network connection")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connection)
    provider = get_llm_provider(llm_settings)
    try:
        assert llm_settings.api_key == api_key
        assert isinstance(provider, OpenAIProvider if kind == "openai" else OllamaProvider)
        if kind == "openai":
            assert provider._client.api_key == api_key
        before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        assert initialize(tmp_path).returncode == 0
        assert {path: path.read_bytes() for path in before} == before
    finally:
        if kind == "openai":
            await provider._client.close()
        else:
            await provider._client._client.aclose()


@pytest.mark.parametrize("damage", ["password", "admin", "missing", "invalid", "symlink"])
def test_conflicting_or_invalid_state_is_rejected_without_reset(tmp_path, damage):
    assert initialize(tmp_path).returncode == 0
    config_file = tmp_path / "config/config.yaml"
    if damage == "password":
        (tmp_path / "pg-secret/password").write_text("conflicting-secret")
    elif damage == "admin":
        (tmp_path / "config/initial-admin.yaml").write_text("username: admin\npassword: conflict")
    elif damage == "missing":
        config_file.unlink()
    elif damage == "invalid":
        config_file.write_text("auth: [not, a, mapping]")
    else:
        original = config_file.read_bytes()
        config_file.unlink()
        target = tmp_path / "outside"
        target.write_bytes(original)
        config_file.symlink_to(target)
    before = config_file.read_bytes() if config_file.exists() else None
    result = initialize(tmp_path)
    assert result.returncode != 0
    assert (config_file.read_bytes() if config_file.exists() else None) == before
    assert "conflicting-secret" not in result.stdout + result.stderr


@pytest.mark.asyncio
async def test_generated_credentials_use_existing_empty_database_admin_bootstrap(tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4

    from vision_bill.config import Settings
    from vision_bill.main import bootstrap_admin
    from vision_bill.security.models import User
    from vision_bill.security.password import verify_password

    assert initialize(tmp_path).returncode == 0
    data = yaml.safe_load((tmp_path / "config/config.yaml").read_text())
    settings = Settings(**data)
    database = MagicMock()
    database.count_users = AsyncMock(return_value=0)
    admin = User(id=uuid4(), username=data["auth"]["bootstrap_username"], is_admin=True)
    database.create_user = AsyncMock(return_value=admin)
    database.set_owner_of_orphan_rows = AsyncMock()
    await bootstrap_admin(database, settings)
    args, kwargs = database.create_user.await_args
    assert verify_password(args[1], data["auth"]["bootstrap_password"], settings.auth)
    assert kwargs["is_admin"] is True
    database.set_owner_of_orphan_rows.assert_awaited_once_with(admin.id)
    database.count_users.return_value = 1
    await bootstrap_admin(database, settings)
    database.create_user.assert_awaited_once()


def test_nested_secret_directories_are_all_private(tmp_path):
    previous_umask = os.umask(0)
    try:
        result = initialize(
            tmp_path, "--postgres-password-file", str(tmp_path / "nested/deep/pg/password")
        )
    finally:
        os.umask(previous_umask)
    assert result.returncode == 0, result.stderr
    for name in ["nested", "nested/deep", "nested/deep/pg"]:
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o700


def test_symlink_ancestor_does_not_redirect_secret_creation(tmp_path):
    destination = tmp_path / "outside"
    destination.mkdir()
    link = tmp_path / "link"
    link.symlink_to(destination, target_is_directory=True)
    result = initialize(tmp_path, "--postgres-password-file", str(link / "pg/password"))
    assert result.returncode != 0
    assert list(destination.iterdir()) == []


def test_concurrent_initializers_share_one_stable_installation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: initialize(tmp_path), range(4)))
    assert all(result.returncode == 0 for result in results)
    assert initialize(tmp_path).returncode == 0


def test_restored_config_needs_no_llm_seeds(tmp_path):
    assert initialize(tmp_path).returncode == 0
    config_file = tmp_path / "config/config.yaml"
    data = yaml.safe_load(config_file.read_text())
    data["llm"]["host"] = "http://ollama:11434"
    data["llm"]["model_name"] = "previously-selected-model"
    data["distribution"] = {"expected_model_digest": "retained-legacy-metadata"}
    config_file.write_text(yaml.safe_dump(data))
    original_secrets = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert initialize(tmp_path, runtime_environment={}).returncode == 0
    assert {p: p.read_bytes() for p in original_secrets} == original_secrets


@pytest.mark.parametrize("missing", ["LLM__PROVIDER", "LLM__HOST", "LLM__MODEL_NAME"])
def test_missing_fresh_model_does_not_publish_secrets(tmp_path, missing):
    environment = {
        "LLM__PROVIDER": "ollama",
        "LLM__HOST": "http://external:11434",
        "LLM__MODEL_NAME": "chosen",
    }
    del environment[missing]
    result = initialize(tmp_path, runtime_environment=environment)
    assert result.returncode != 0
    assert not (tmp_path / "config/config.yaml").exists()
    assert not (tmp_path / "config/initial-admin.yaml").exists()
    assert not (tmp_path / "pg-secret/password").exists()
    assert missing in result.stderr


@pytest.mark.parametrize(
    "change",
    [
        {"LLM__PROVIDER": "anthropic"},
        {"LLM__API_KEY": "", "LLM__PROVIDER": "openai"},
        {"LLM__TEMPERATURE": "nan"},
        {"LLM__TEMPERATURE": "3"},
        {"LLM__HOST": " "},
    ],
)
def test_invalid_fresh_inputs_do_not_publish_secrets(tmp_path, change):
    environment = {
        "LLM__PROVIDER": "ollama",
        "LLM__HOST": "http://external:11434",
        "LLM__MODEL_NAME": "chosen",
        **change,
    }
    result = initialize(tmp_path, runtime_environment=environment)
    assert result.returncode != 0
    assert not (tmp_path / "config/config.yaml").exists()
    assert not (tmp_path / "pg-secret/password").exists()


def test_literal_runtime_key_round_trip(tmp_path):
    key = "literal '$HOME #key: \\ value\nnext line"
    result = initialize(
        tmp_path,
        runtime_environment={
            "LLM__PROVIDER": "openai",
            "LLM__HOST": "http://external/v1",
            "LLM__MODEL_NAME": "chosen",
            "LLM__API_KEY": key,
            "LLM__TEMPERATURE": "0.25",
        },
    )
    assert result.returncode == 0, result.stderr
    data = yaml.safe_load((tmp_path / "config/config.yaml").read_text())
    assert data["llm"]["api_key"] == key
    assert data["llm"]["temperature"] == 0.25
    assert key not in result.stdout + result.stderr


def test_explicit_runtime_mapping_does_not_inherit_process_seeds(tmp_path, monkeypatch):
    from vision_bill.distribution.bootstrap import BootstrapError
    from vision_bill.distribution.bootstrap import initialize as bootstrap

    monkeypatch.setenv("LLM__PROVIDER", "openai")
    monkeypatch.setenv("LLM__HOST", "http://ignored.example/v1")
    monkeypatch.setenv("LLM__MODEL_NAME", "ignored-model")
    with pytest.raises(BootstrapError, match="LLM__PROVIDER"):
        bootstrap(tmp_path / "config", tmp_path / "pg/password", runtime_environment={})
    assert not (tmp_path / "config/config.yaml").exists()


def test_unsupported_restored_provider_has_actionable_diagnostic(tmp_path):
    assert initialize(tmp_path).returncode == 0
    path = tmp_path / "config/config.yaml"
    data = yaml.safe_load(path.read_text())
    data["llm"]["provider"] = "anthropic"
    path.write_text(yaml.safe_dump(data))
    before = path.read_bytes()
    result = initialize(tmp_path, runtime_environment={})
    assert result.returncode != 0
    assert "ollama or openai" in result.stderr
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["host", "model_name"])
def test_restored_whitespace_identity_cannot_publish_companions(tmp_path, field):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    content = (
        "llm: {host: 'http://external', api_key: '', model_name: 'selected', temperature: 0}\n"
        "pg: {user: 'runtime-login', password: 'db-secret'}\n"
        "auth: {secret_key: 'auth-secret'}\n"
    )
    saved = yaml.safe_load(content)
    saved["llm"][field] = " \t "
    saved_bytes = yaml.safe_dump(saved).encode()
    path = config_dir / "config.yaml"
    path.write_bytes(saved_bytes)

    result = initialize(tmp_path, runtime_environment={})

    assert result.returncode != 0
    assert not (config_dir / ".bootstrap-complete.yaml").exists()
    assert not (tmp_path / "pg-secret/password").exists()
    assert path.read_bytes() == saved_bytes


@pytest.mark.parametrize("duplicate", [False, True])
def test_restore_yaml_merge_uses_explicit_override_and_rejects_literal_duplicates(
    tmp_path, duplicate
):
    from vision_bill.distribution.bootstrap import read_config

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    content = (
        "llm:\n"
        "  <<: [{<<: &profile {host: 'http://first', model_name: inherited"
        + (", model_name: duplicate" if duplicate else "")
        + "}}, {host: 'http://second', model_name: other}]\n"
        "  model_name: selected\n"
        "  api_key: ''\n"
        "  temperature: 0\n"
        "alternate: {<<: *profile, model_name: alternate}\n"
        "pg: {user: 'runtime-login', password: 'db-secret'}\n"
        "auth: {secret_key: 'auth-secret'}\n"
    )
    path = config_dir / "config.yaml"
    path.write_text(content)

    result = initialize(tmp_path, runtime_environment={})

    assert path.read_text() == content
    if duplicate:
        assert result.returncode != 0
        assert "duplicate keys" in result.stderr
        assert not (config_dir / ".bootstrap-complete.yaml").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (config_dir / ".bootstrap-complete.yaml").exists()
        llm = read_config(path)["llm"]
        assert llm["host"] == "http://first"
        assert llm["model_name"] == "selected"
