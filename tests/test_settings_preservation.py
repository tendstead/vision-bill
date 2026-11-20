"""Admin saves preserve private persisted state and reject unsupported changes."""

import stat

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vision_bill import config
from vision_bill.api.system import main as system
from vision_bill.security.dependencies import require_admin


@pytest.fixture
def saved_settings(settings, tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    data = settings.model_dump(mode="json", exclude={"pg": {"database_url", "pg_dsn"}})
    data["llm"].update(
        host="http://ollama:11434",
        api_key="persisted-api-secret",
        metadata={"future": [1, {"x": "y"}]},
    )
    data["pg"]["password"] = "persisted-db-secret"
    data["auth"].update(secret_key="persisted-auth-secret", pepper="persisted-pepper")
    data["distribution"] = {"expected_model_digest": "legacy-identity", "other": {"retain": True}}
    path.write_text(yaml.safe_dump(data))
    for name in config.EDITABLE_ENV_KEYS.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    settings.llm.host = data["llm"]["host"]
    # Effective private overrides must never overwrite unrelated persisted keys.
    settings.llm.api_key = "environment-api-secret"
    settings.pg.password = "environment-db-secret"
    settings.auth.secret_key = "environment-auth-secret"
    monkeypatch.setattr(system, "settings", settings)
    monkeypatch.setattr(config, "_startup_llm_identity", None)
    config.mark_startup_settings(settings)
    app = FastAPI()
    app.include_router(system.router, prefix="/system")
    app.dependency_overrides[require_admin] = lambda: None
    with TestClient(app) as client:
        yield path, data, settings, client


def payload(settings, **llm):
    return {
        "llm": {
            "provider": settings.llm.provider,
            "host": settings.llm.host,
            "model_name": settings.llm.model_name,
            "temperature": settings.llm.temperature,
            **llm,
        },
        "allow_registration": settings.auth.allow_registration,
    }


def explicit_payload(settings, **llm):
    body = payload(settings, **llm)
    body['changed_paths'] = [f'llm.{field}' for field in llm]
    return body


def test_settings_save_preserves_nested_unknown_metadata(saved_settings):
    path, original, settings, client = saved_settings
    response = client.put("/system/settings", json=payload(settings, model_name="new-selection"))
    assert response.status_code == 200
    saved = yaml.safe_load(path.read_text())
    original["llm"]["model_name"] = "new-selection"
    assert saved == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert response.json()["sources"]["llm.model_name"] == "config"


def test_unsupported_provider_update_is_atomic(saved_settings):
    path, _, settings, client = saved_settings
    before_update = path.read_bytes()
    previous_runtime = settings.model_dump()
    unsupported_update = client.put(
        "/system/settings", json=payload(settings, provider="anthropic")
    )
    assert unsupported_update.status_code == 422
    assert path.read_bytes() == before_update
    assert settings.model_dump() == previous_runtime


@pytest.mark.parametrize("field", ["host", "model_name"])
def test_blank_provider_identity_update_is_atomic(saved_settings, field):
    path, _, settings, client = saved_settings
    before = path.read_bytes()
    previous_runtime = settings.model_dump()
    response = client.put("/system/settings", json=payload(settings, **{field: " "}))
    assert response.status_code == 422
    assert path.read_bytes() == before
    assert settings.model_dump() == previous_runtime


@pytest.mark.parametrize(
    "content", ["auth: [broken]\n", "- not-a-mapping\n", "llm: [\n", "llm: {}\nllm: {}\n"]
)
def test_malformed_existing_document_is_not_replaced(saved_settings, content):
    path, _, settings, client = saved_settings
    path.write_text(content)
    previous_runtime = settings.model_dump()
    response = client.put("/system/settings", json=payload(settings, model_name="changed"))
    assert response.status_code == 503
    assert path.read_text() == content
    assert settings.model_dump() == previous_runtime


def test_unchanged_environment_setting_stays_effective_until_explicit_edit(
    saved_settings, monkeypatch
):
    path, original, settings, client = saved_settings
    monkeypatch.setenv("LLM__HOST", "http://effective.example")
    settings.llm.host = "http://effective.example"
    result = client.put("/system/settings", json=explicit_payload(settings, model_name="new-selection"))
    assert result.status_code == 200
    saved = yaml.safe_load(path.read_text())
    assert saved["llm"]["host"] == original["llm"]["host"]
    assert result.json()["llm"]["host"] == "http://effective.example"
    assert result.json()["sources"]["llm.host"] == "environment"
    assert result.json()["restart_required"] is True
    before = path.read_bytes()
    settings.llm.api_key = "none"
    assert (
        client.put("/system/settings", json=explicit_payload(settings, host="http://other")).status_code
        == 200
    )
    assert yaml.safe_load(path.read_text())["llm"]["host"] == "http://other"


def test_full_save_merges_unknown_nested_keys(saved_settings):
    path, original, settings, _ = saved_settings
    config.write_config_file(settings)
    saved = yaml.safe_load(path.read_text())
    assert saved["distribution"] == original["distribution"]
    assert saved["llm"]["metadata"] == {"future": [1, {"x": "y"}]}


@pytest.mark.parametrize(
    "artifact", [".bootstrap-pending", ".bootstrap-complete.yaml", "initial-admin.yaml"]
)
def test_settings_cannot_recreate_partial_bootstrap_state(saved_settings, artifact):
    path, _, settings, client = saved_settings
    path.unlink()
    (path.parent / artifact).write_text("retain-existing-state")
    response = client.put("/system/settings", json=payload(settings, model_name="changed"))
    assert response.status_code == 503
    assert not path.exists()
    assert (path.parent / artifact).read_text() == "retain-existing-state"


def test_settings_publication_uses_bootstrap_lock(saved_settings):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    from vision_bill.distribution.bootstrap import config_lock

    path, _, settings, _ = saved_settings
    before = path.read_bytes()
    started = Event()

    def save():
        started.set()
        config.write_config_file(settings, changed_paths=frozenset({"llm.model_name"}))

    with ThreadPoolExecutor(max_workers=1) as pool:
        with config_lock(path.parent):
            future = pool.submit(save)
            assert started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            assert path.read_bytes() == before
        future.result(timeout=2)


def test_invalid_save_diagnostics_do_not_log_secret_lines(saved_settings, caplog):
    path, _, settings, client = saved_settings
    path.write_text("private-secret: [malformed-secret-value")
    response = client.put("/system/settings", json=payload(settings, model_name="changed"))
    assert response.status_code == 503
    assert "malformed-secret-value" not in caplog.text + response.text


@pytest.mark.parametrize(
    ("saved_key", "runtime_key", "expected_key"),
    [("", None, "none"), ("saved-key", None, "saved-key"), ("", "runtime-key", "runtime-key")],
)
async def test_ollama_to_openai_update_constructs_fresh_provider(
    saved_settings, monkeypatch, saved_key, runtime_key, expected_key
):
    from vision_bill.provider.factory import get_llm_provider

    path, original, _, client = saved_settings
    original["llm"].update(api_key=saved_key, model_lifecycle="llamacpp")
    path.write_text(yaml.safe_dump(original))
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    monkeypatch.delenv("LLM__API_KEY", raising=False)
    if runtime_key is not None:
        monkeypatch.setenv("LLM__API_KEY", runtime_key)
    before_restart = config.Settings()
    monkeypatch.setattr(system, "settings", before_restart)

    response = client.put(
        "/system/settings",
        json=payload(before_restart, provider="openai", host="http://ollama:11434/v1"),
    )
    assert response.status_code == 200
    original["llm"].update(provider="openai", host="http://ollama:11434/v1")
    original["llm_credential_origin"] = "http://ollama:11434"
    assert yaml.safe_load(path.read_text()) == original
    after_restart = config.Settings()
    provider = get_llm_provider(after_restart.llm)
    try:
        assert provider._client.api_key == expected_key
        async with provider._lifecycle._client() as lifecycle:
            assert lifecycle.headers["Authorization"] == f"Bearer {expected_key}"
    finally:
        await provider._client.close()


@pytest.mark.parametrize("source", ["lowercase env", "JSON env", "JSON dotenv", "dotenv local"])
def test_actual_settings_sources_allow_explicit_adoption(saved_settings, monkeypatch, source):
    path, _, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    if source == "lowercase env":
        monkeypatch.setenv("llm__host", "http://effective.example")
    elif source == "JSON env":
        monkeypatch.setenv("LLM", '{"HOST":"http://effective.example"}')
    elif source == "JSON dotenv":
        (path.parent / ".env").write_text('llm=\'{"HOST":"http://effective.example"}\'\n')
    else:
        (path.parent / ".env").write_text("LLM__HOST=http://overridden.example\n")
        (path.parent / ".env.local").write_text("export llm__host=http://effective.example\n")
    settings = config.Settings()
    monkeypatch.setattr(system, "settings", settings)
    before = path.read_bytes()

    response = client.get("/system/settings")
    assert response.status_code == 200
    assert response.json()["llm"]["host"] == "http://effective.example"
    assert response.json()["sources"]["llm.host"] == "environment"
    assert response.json()["sources"]["llm.model_name"] == "config"
    blocked = client.put("/system/settings", json=explicit_payload(settings, host="http://other"))
    assert blocked.status_code == 200
    assert config.Settings().llm.host == "http://other"


def test_settings_save_accepts_yaml_merge_with_explicit_override(saved_settings):
    path, original, settings, client = saved_settings
    inherited = {**original.pop("llm"), "model_name": "inherited"}
    path.write_text(
        yaml.safe_dump(original)
        + "llm_profile: &profile\n"
        + "".join(f"  {line}\n" for line in yaml.safe_dump(inherited).splitlines())
        + f"llm:\n  <<: *profile\n  model_name: {settings.llm.model_name!r}\n"
    )
    expected = yaml.safe_load(path.read_text())
    expected["llm"]["model_name"] = "new-selection"

    response = client.put("/system/settings", json=payload(settings, model_name="new-selection"))

    assert response.status_code == 200
    assert yaml.safe_load(path.read_text()) == expected


def test_first_start_does_not_overwrite_locked_winner(saved_settings, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from vision_bill.distribution import bootstrap

    path, original, settings, _ = saved_settings
    path.unlink()
    real_lock = bootstrap.config_lock
    at_lock = Event()

    def observe_lock(directory):
        at_lock.set()
        return real_lock(directory)

    monkeypatch.setattr(bootstrap, "config_lock", observe_lock)
    winner = yaml.safe_dump(original)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with real_lock(path.parent):
            loser = pool.submit(config.initialize_config_file, settings)
            assert at_lock.wait(timeout=2)
            bootstrap.write_private(path, winner)
        loser.result(timeout=2)

    preserved = path.read_text() == winner
    assert preserved


@pytest.mark.parametrize(
    ("saved_ocr", "status"),
    [
        ({"required": True}, 200),
        ({"required": True, "enabled": "invalid-boolean"}, 503),
        ({"timeout_seconds": 0}, 503),
    ],
)
def test_settings_save_preserves_partial_ocr_with_canonical_runtime_endpoint(
    saved_settings, monkeypatch, saved_ocr, status
):
    path, original, _, client = saved_settings
    original["ocr"] = saved_ocr
    path.write_text(yaml.safe_dump(original))
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    monkeypatch.setenv("OCR__ENDPOINT", "http://ocr:8082")
    monkeypatch.setenv("OCR__TIMEOUT_SECONDS", "600")
    active = config.Settings()
    monkeypatch.setattr(system, "settings", active)
    assert active.ocr.enabled and active.ocr.timeout_seconds == 600
    before = path.read_bytes()

    response = client.put("/system/settings", json=payload(active, model_name="new-selection"))

    assert response.status_code == status
    if status == 200:
        original["llm"]["model_name"] = "new-selection"
        assert yaml.safe_load(path.read_text()) == original
    else:
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("persisted", "prepared", "status"),
    [
        ({"temperature": 0.3, "metadata": {"retain": True}}, False, 200),
        ({"temperature": 0.3, "host": ["invalid"]}, False, 503),
        ({"temperature": 0.3, "api_key": 42}, False, 503),
        ({"temperature": 0.3}, True, 503),
    ],
)
def test_edits_sparse_runtime_config_without_saving_environment_credentials(
    saved_settings, monkeypatch, persisted, prepared, status
):
    path, _, settings, client = saved_settings
    original = {"llm": persisted, "extension": {"keep": "private metadata"}}
    path.write_text(yaml.safe_dump(original))
    before = path.read_bytes()
    for field in ("provider", "host", "model_name", "api_key"):
        monkeypatch.setenv(f"LLM__{field.upper()}", str(settings.llm.model_dump(mode="json")[field]))
    monkeypatch.setenv("PG__PASSWORD", settings.pg.password)
    monkeypatch.setenv("AUTH__SECRET_KEY", settings.auth.secret_key)
    if prepared:
        (path.parent / ".bootstrap-complete.yaml").write_text("existing: true")

    response = client.put("/system/settings", json=payload(settings, temperature=0.61))

    assert response.status_code == status
    if status == 200:
        original["llm"]["temperature"] = 0.61
        assert yaml.safe_load(path.read_text()) == original
        assert response.json()["sources"]["llm.host"] == "environment"
        assert response.json()["sources"]["llm.temperature"] == "config"
    else:
        assert path.read_bytes() == before


def test_explicit_admin_edit_adopts_only_changed_safe_environment_fields(saved_settings, monkeypatch):
    path, original, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    monkeypatch.setenv("LLM__HOST", "http://environment.example")
    monkeypatch.setenv("LLM__MODEL_NAME", "environment-model")
    monkeypatch.setenv("LLM__API_KEY", "private-env-key")
    active = config.Settings()
    monkeypatch.setattr(system, "settings", active)
    response = client.put('/system/settings', json=explicit_payload(active, host='http://environment.example/v1', temperature=0.72))
    assert response.status_code == 200
    restarted = config.Settings()
    assert restarted.llm.host == 'http://environment.example/v1'
    assert restarted.llm.temperature == 0.72
    assert restarted.llm.model_name == 'environment-model'
    assert restarted.llm.api_key == 'private-env-key'
    assert yaml.safe_load(path.read_text())['llm']['model_name'] == original['llm']['model_name']
    assert response.json()['sources']['llm.host'] == 'config'


@pytest.mark.parametrize('host', ['file:///tmp/model', 'not a URL', 'http://', 'https://user:secret@example.com'])
def test_invalid_host_update_is_atomic(saved_settings, host):
    path, _, settings, client = saved_settings
    before = path.read_bytes()
    response = client.put('/system/settings', json=payload(settings, host=host))
    assert response.status_code == 422
    assert path.read_bytes() == before


def test_non_admin_cannot_adopt_environment_defaults(saved_settings):
    from types import SimpleNamespace
    from vision_bill.security.dependencies import get_current_user

    path, _, settings, client = saved_settings
    client.app.dependency_overrides.pop(require_admin)
    client.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(is_admin=False)
    before = path.read_bytes()
    assert client.put('/system/settings', json=payload(settings, host='https://admin.example')).status_code == 403
    assert path.read_bytes() == before


def test_noop_save_does_not_adopt_environment_tuple(saved_settings, monkeypatch):
    path, _, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.setenv('LLM__HOST', 'http://environment.example')
    monkeypatch.setenv('LLM__MODEL_NAME', 'environment-model')
    active = config.Settings()
    monkeypatch.setattr(system, 'settings', active)
    before = yaml.safe_load(path.read_text())
    assert client.put('/system/settings', json=payload(active)).status_code == 200
    assert yaml.safe_load(path.read_text()) == before
    monkeypatch.setenv('LLM__MODEL_NAME', 'new-environment-model')
    assert config.Settings().llm.model_name == 'new-environment-model'


def test_environment_hot_fields_adopt_on_save(saved_settings, monkeypatch):
    path, _, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.setenv('LLM__MODEL_NAME', 'environment-model')
    monkeypatch.setenv('LLM__TEMPERATURE', '0.2')
    monkeypatch.setenv('AUTH__ALLOW_REGISTRATION', 'true')
    active = config.Settings()
    monkeypatch.setattr(system, 'settings', active)
    body = explicit_payload(active, model_name='adopted-model', temperature=0.6)
    response = client.put('/system/settings', json=body)
    assert response.status_code == 200
    assert (active.llm.model_name, active.llm.temperature, active.auth.allow_registration) == ('adopted-model', 0.6, True)
    restarted = config.Settings()
    assert (restarted.llm.model_name, restarted.llm.temperature, restarted.auth.allow_registration) == ('adopted-model', 0.6, True)


def test_override_marker_cannot_adopt_credentials_or_database_fields(saved_settings, monkeypatch):
    path, original, _, _ = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    original['admin_setting_overrides'] = ['llm.api_key', 'pg.password', 'auth.secret_key', 'distribution.other']
    path.write_text(yaml.safe_dump(original))
    monkeypatch.setenv('LLM__API_KEY', 'runtime-key')
    monkeypatch.setenv('PG__PASSWORD', 'runtime-db')
    monkeypatch.setenv('AUTH__SECRET_KEY', 'runtime-secret')
    restarted = config.Settings()
    assert (restarted.llm.api_key, restarted.pg.password, restarted.auth.secret_key) == ('runtime-key', 'runtime-db', 'runtime-secret')


def test_legacy_host_remains_readable_and_allows_unrelated_edits(saved_settings):
    path, original, settings, client = saved_settings
    settings.llm.host = 'localhost'
    original['llm']['host'] = 'localhost'
    path.write_text(yaml.safe_dump(original))
    assert client.get('/system/settings').status_code == 200
    response = client.put('/system/settings', json=payload(settings, temperature=0.8))
    assert response.status_code == 200
    assert yaml.safe_load(path.read_text())['llm']['host'] == 'localhost'


def test_real_key_cannot_follow_browser_host_to_another_origin(saved_settings):
    path, _, settings, client = saved_settings
    before = path.read_bytes()
    assert client.put('/system/settings', json=payload(settings, host='https://unapproved.example')).status_code == 409
    assert path.read_bytes() == before


def test_changed_paths_preserve_another_tabs_model_edit(saved_settings):
    _, _, settings, client = saved_settings
    stale = payload(settings, temperature=0.8)
    assert client.put('/system/settings', json=payload(settings, model_name='other-tab')).status_code == 200
    stale['changed_paths'] = ['llm.temperature']
    assert client.put('/system/settings', json=stale).status_code == 200
    assert settings.llm.model_name == 'other-tab'
    assert settings.llm.temperature == 0.8


def test_missing_selective_config_never_copies_runtime_secrets(saved_settings):
    path, _, settings, client = saved_settings
    path.unlink()
    assert client.put('/system/settings', json=payload(settings, temperature=0.8)).status_code == 200
    assert yaml.safe_load(path.read_text()) == {'llm': {'temperature': 0.8}}


def test_new_key_cannot_follow_previously_adopted_no_key_endpoint(saved_settings, monkeypatch):
    path, _, settings, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.setenv('LLM__API_KEY', 'none')
    settings.llm.api_key = 'none'
    response = client.put('/system/settings', json=explicit_payload(settings, host='https://unapproved.example'))
    assert response.status_code == 200
    assert config.Settings().llm.host == 'https://unapproved.example'
    monkeypatch.setenv('LLM__API_KEY', 'new-private-key')
    with pytest.raises(ValueError, match='operator approval'):
        config.Settings()


def test_real_key_same_origin_path_change_survives_restart(saved_settings, monkeypatch):
    path, _, settings, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.setenv('LLM__API_KEY', 'real-private-key')
    settings.llm.api_key = 'real-private-key'
    assert client.put('/system/settings', json=payload(settings, host='http://ollama:11434/v1')).status_code == 200
    assert config.Settings().llm.host == 'http://ollama:11434/v1'


def test_registration_environment_control_remains_locked(saved_settings, monkeypatch):
    path, _, settings, client = saved_settings
    monkeypatch.setenv('AUTH__ALLOW_REGISTRATION', 'true')
    settings.auth.allow_registration = True
    body = payload(settings)
    body['allow_registration'] = False
    before = path.read_bytes()
    assert client.put('/system/settings', json=body).status_code == 409
    assert path.read_bytes() == before


def test_credential_origin_distinguishes_explicit_zero_port():
    assert config.llm_origin('http://trusted.example:0') != config.llm_origin('http://trusted.example')
    assert config.llm_origin('https://trusted.example') == config.llm_origin('https://trusted.example:443/v1')


def test_operator_endpoint_and_key_rotation_supersedes_stale_host_adoption(saved_settings, monkeypatch):
    path, _, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.setenv('LLM__HOST', 'https://approved-a.example/v1')
    monkeypatch.setenv('LLM__API_KEY', 'key-a')
    active = config.Settings()
    monkeypatch.setattr(system, 'settings', active)
    assert client.put('/system/settings', json=explicit_payload(active, host='https://approved-a.example/v2')).status_code == 200
    monkeypatch.setenv('LLM__HOST', 'https://approved-b.example/v1')
    monkeypatch.setenv('LLM__API_KEY', 'key-b')
    restarted = config.Settings()
    assert restarted.llm.host == 'https://approved-b.example/v1'
    assert restarted.llm.api_key == 'key-b'
    monkeypatch.setattr(system, 'settings', restarted)
    assert client.put('/system/settings', json=explicit_payload(restarted, host='https://approved-b.example/v2')).status_code == 200
    assert config.Settings().llm.host == 'https://approved-b.example/v2'
    assert config.Settings().llm.api_key == 'key-b'


def test_per_instance_dotenv_rotation_supersedes_stale_host_adoption(saved_settings, monkeypatch):
    path, original, _, _ = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.delenv('LLM__API_KEY', raising=False)
    original['llm']['host'] = 'https://approved-a.example/v2'
    original['admin_setting_overrides'] = ['llm.host']
    original['llm_credential_origin'] = 'https://approved-a.example:443'
    path.write_text(yaml.safe_dump(original))
    dotenv = path.parent / 'operator.env'
    dotenv.write_text('LLM__HOST=https://approved-b.example/v1\nLLM__API_KEY=key-b\n')
    restarted = config.Settings(_env_file=dotenv)
    assert restarted.llm.host == 'https://approved-b.example/v1'
    assert restarted.llm.api_key == 'key-b'


def test_explicit_init_host_retains_precedence_over_unkeyed_adoption(saved_settings, monkeypatch):
    path, original, _, _ = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.delenv('LLM__API_KEY', raising=False)
    original['llm']['host'] = 'https://unkeyed-b.example/v2'
    original['admin_setting_overrides'] = ['llm.host']
    original['llm_credential_origin'] = 'https://approved-a.example:443'
    path.write_text(yaml.safe_dump(original))
    restarted = config.Settings(llm={'host': 'https://approved-a.example/v1', 'api_key': 'real-key'})
    assert restarted.llm.host == 'https://approved-a.example/v1'
    assert restarted.llm.api_key == 'real-key'


def test_per_instance_file_secret_key_cannot_follow_unkeyed_adoption(saved_settings, monkeypatch):
    import json

    path, original, _, _ = saved_settings
    monkeypatch.setitem(config.Settings.model_config, 'yaml_file', str(path))
    monkeypatch.delenv('LLM__API_KEY', raising=False)
    original['llm'].pop('api_key')
    original['llm']['host'] = 'https://unapproved-b.example/v2'
    original['admin_setting_overrides'] = ['llm.host']
    original['llm_credential_origin'] = 'https://approved-a.example:443'
    path.write_text(yaml.safe_dump(original))
    secret_dir = path.parent / 'secrets'
    secret_dir.mkdir()
    (secret_dir / 'llm').write_text(json.dumps({'api_key': 'private-secret-key'}))
    with pytest.raises(ValueError, match='operator approval'):
        config.Settings(_env_file=None, _secrets_dir=secret_dir)


def test_legacy_full_put_keeps_environment_fields_locked(saved_settings, monkeypatch):
    path, _, settings, client = saved_settings
    monkeypatch.setenv('LLM__PROVIDER', settings.llm.provider.value)
    body = payload(settings, provider='openai')
    before = path.read_bytes()
    response = client.put('/system/settings', json=body)
    assert response.status_code == 409
    assert 'llm.provider' in response.json()['detail']
    assert path.read_bytes() == before
    body['changed_paths'] = ['llm.provider']
    response = client.put('/system/settings', json=body)
    assert response.status_code == 200
    assert response.json()['sources']['llm.provider'] == 'config'


def test_explicit_none_overrides_legacy_key_without_copying_or_deleting_it(saved_settings, monkeypatch):
    path, _, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    monkeypatch.setenv("LLM__PROVIDER", "openai")
    monkeypatch.setenv("LLM__API_KEY", "deployment-private-key")
    active = config.Settings()
    monkeypatch.setattr(system, "settings", active)
    config.mark_startup_settings(active)

    initial = client.get("/system/settings")
    assert initial.json()["llm_authentication"] == {
        "mode": "bearer", "source": "environment", "status": "configured",
    }
    body = {**explicit_payload(active), "llm_authentication": {"mode": "none"}}
    response = client.put("/system/settings", json=body)
    assert response.status_code == 200
    assert response.json()["llm_authentication"] == {"mode": "none", "source": "settings", "status": "none"}
    assert response.json()["restart_required"] is True
    assert "deployment-private-key" not in response.text
    saved = yaml.safe_load(path.read_text())
    assert saved["llm"]["api_key"] == "persisted-api-secret"
    assert saved["llm_authentication"] == {"mode": "none"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    restarted = config.Settings()
    assert restarted.llm.request_api_key() is None
    config.mark_startup_settings(restarted)
    assert not config.runtime_restart_required(restarted)


def test_bearer_replacement_cannot_follow_unrelated_origins(saved_settings, monkeypatch):
    path, _, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    monkeypatch.setenv("LLM__API_KEY", "legacy-deployment-key")
    active = config.Settings()
    monkeypatch.setattr(system, "settings", active)
    body = {
        **explicit_payload(active, provider="openai", host="https://one.example/v1"),
        "llm_authentication": {"mode": "bearer", "api_key": "first-private-key"},
    }
    assert client.put("/system/settings", json=body).status_code == 200
    blocked = client.put("/system/settings", json=explicit_payload(active, host="https://two.example/v1"))
    assert blocked.status_code == 409
    assert config.Settings().llm.request_api_key() == "first-private-key"

    body = {
        **explicit_payload(active, host="https://two.example/v1"),
        "llm_authentication": {"mode": "bearer", "api_key": "second-private-key"},
    }
    replaced = client.put("/system/settings", json=body)
    assert replaced.status_code == 200
    assert replaced.json()["llm_authentication"]["status"] == "configured"
    assert "private-key" not in replaced.text + client.get("/system/settings").text
    assert config.Settings().llm.request_api_key() == "second-private-key"
    private = yaml.safe_load(path.read_text())
    private["llm"]["host"] = "https://three.example/v1"
    path.write_text(yaml.safe_dump(private))
    changed = config.Settings()
    assert changed.llm.request_api_key() is None
    monkeypatch.setattr(system, "settings", changed)
    assert client.get("/system/settings").json()["llm_authentication"]["status"] == "origin_mismatch"


@pytest.mark.parametrize("invalid", [123, "canary\ninvalid", {"canary": "private"}])
def test_invalid_bearer_input_is_never_reflected(saved_settings, invalid, caplog):
    path, _, active, client = saved_settings
    previous = path.read_bytes()
    body = {
        **explicit_payload(active, provider="openai"),
        "llm_authentication": {"mode": "bearer", "api_key": invalid},
    }
    response = client.put("/system/settings", json=body)
    assert response.status_code == 422
    assert "canary" not in response.text + caplog.text
    assert response.json()["detail"] == "Invalid LLM authentication settings"
    assert path.read_bytes() == previous


@pytest.mark.parametrize(
    ("status_code", "encoding", "result"),
    [(200, None, "connected"), (401, None, "authentication_failed"), (503, None, "unreachable"),
     (200, "gzip", "invalid_response"), (200, "long_integer", "invalid_response"),
     (200, "rotated_key", "connected")],
)
def test_saved_connection_probe_only_reads_bounded_model_discovery(saved_settings, status_code, encoding, result):
    import gzip
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    path, _, active, client = saved_settings
    calls = []

    class Stub(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.command, self.path, self.headers.get_all("Authorization"), self.headers.get("Accept-Encoding")))
            response_body = b'{"data": [], "private_message": "selected-secret"}'
            self.send_response(status_code)
            if encoding == "gzip":
                self.send_header("Content-Encoding", "gzip")
                response_body = gzip.compress(response_body)
            elif encoding == "long_integer":
                response_body = b'{"data": [' + b"9" * 6000 + b"]}"
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Stub) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host = f"http://127.0.0.1:{server.server_port}/llama/v1"
            body = {
                **explicit_payload(active, provider="openai", host=host),
                "llm_authentication": {"mode": "bearer", "api_key": "selected-secret"},
            }
            assert client.put("/system/settings", json=body).status_code == 200
            if encoding == "rotated_key":
                other_replica = yaml.safe_load(path.read_text())
                other_replica["llm_authentication"]["api_key"] = "rotated-secret"
                path.write_text(yaml.safe_dump(other_replica))
            previous = path.read_bytes()
            response = client.post("/system/settings/llm/connection-test", json={"host": host})
            assert response.status_code == 200
            assert response.json() == {"status": result}
            assert "selected-secret" not in response.text
            assert "rotated-secret" not in response.text
            assert path.read_bytes() == previous
        finally:
            server.shutdown()
            thread.join()
    key = "rotated-secret" if encoding == "rotated_key" else "selected-secret"
    assert calls == [("GET", "/llama/v1/models", [f"Bearer {key}"], "identity")]


@pytest.mark.parametrize("stale_replica", [False, True])
def test_stale_none_cannot_disable_credentials_for_a_changed_origin(saved_settings, monkeypatch, stale_replica):
    path, _, active, client = saved_settings
    active.llm.provider = config.LLMProviderEnum.OPENAI
    replica = active.model_copy(deep=True)
    stale_body = {**explicit_payload(active), "llm_authentication": {"mode": "none"}}
    new_body = {
        **explicit_payload(active, host="https://new.example/v1"),
        "llm_authentication": {"mode": "bearer", "api_key": "new-endpoint-key"},
    }
    assert client.put("/system/settings", json=new_body).status_code == 200
    previous = path.read_bytes()
    if stale_replica:
        monkeypatch.setattr(system, "settings", replica)
    assert client.put("/system/settings", json=stale_body).status_code == 409
    assert path.read_bytes() == previous


@pytest.mark.parametrize("selection", [{"mode": "none"}, {"mode": "bearer", "api_key": "new-key"}])
@pytest.mark.parametrize("key_source", ["config", "environment"])
def test_moving_origin_with_explicit_authentication_fences_legacy_key_on_rollback(
    saved_settings, monkeypatch, selection, key_source,
):
    path, _, active, client = saved_settings
    monkeypatch.delenv("LLM__API_KEY", raising=False)
    if key_source == "environment":
        monkeypatch.setenv("LLM__API_KEY", "deployment-key")
    old_host = active.llm.host
    body = {**explicit_payload(active, provider="openai", host="https://new.example/v1"), "llm_authentication": selection}
    assert client.put("/system/settings", json=body).status_code == 200
    document = yaml.safe_load(path.read_text())
    assert document["llm"]["api_key"] == "persisted-api-secret"
    assert document.get("llm_credential_origin") == old_host
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    document.pop("llm_authentication")
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="operator approval"):
        config.Settings()


def test_connection_probe_rejects_a_different_saved_endpoint(saved_settings):
    _, _, active, client = saved_settings
    active.llm.provider = config.LLMProviderEnum.OPENAI
    active.llm.host = "http://127.0.0.1:1/v1"
    assert client.post("/system/settings/llm/connection-test", json={"host": "http://127.0.0.1:2/v1"}).status_code == 409


def test_connection_probe_rejects_a_new_endpoint_saved_by_another_replica(saved_settings):
    path, document, active, client = saved_settings
    active.llm.provider = config.LLMProviderEnum.OPENAI
    active.llm.host = "http://127.0.0.1:1/v1"
    document["llm"]["host"] = "http://127.0.0.1:2/v1"
    path.write_text(yaml.safe_dump(document))
    assert client.post("/system/settings/llm/connection-test", json={"host": active.llm.host}).status_code == 409


def test_legacy_ambient_authorization_prevents_cross_origin_host_edit(saved_settings, monkeypatch):
    path, _, active, client = saved_settings
    active.llm.provider = config.LLMProviderEnum.OPENAI
    active.llm.api_key = "none"
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "aUtHoRiZaTiOn: Bearer legacy-ambient-key")
    before = path.read_bytes()
    response = client.put("/system/settings", json=explicit_payload(active, host="https://other.example/v1"))
    assert response.status_code == 409
    assert path.read_bytes() == before


@pytest.mark.parametrize("header", ["Authorization", "X-API-Key", "X-Private-Credential"])
def test_explicit_none_does_not_expose_retained_ambient_credentials_on_rollback(saved_settings, monkeypatch, header):
    path, _, active, client = saved_settings
    active.llm.provider = config.LLMProviderEnum.OPENAI
    active.llm.api_key = "none"
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", f"{header}: legacy-ambient-key")
    monkeypatch.setenv("LLM__API_KEY", "none")
    new_origin = "https://other.example/v1"
    choice = {"mode": "none"}
    before = path.read_bytes()
    move = {**explicit_payload(active, host=new_origin), "llm_authentication": choice}
    assert client.put("/system/settings", json=move).status_code == 409
    assert path.read_bytes() == before
    same_origin = {**explicit_payload(active), "llm_authentication": choice}
    assert client.put("/system/settings", json=same_origin).status_code == 200
    assert client.put("/system/settings", json=explicit_payload(active, host=new_origin)).status_code == 409


@pytest.mark.parametrize("header", ["Authorization", "API-Key", "X-API-Key"])
def test_inherited_ambient_credentials_report_environment_without_values(saved_settings, monkeypatch, header):
    _, _, active, client = saved_settings
    active.llm.provider = config.LLMProviderEnum.OPENAI
    active.llm.api_key = "none"
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", f"{header}: private-ambient-credential")
    response = client.get("/system/settings")
    assert response.status_code == 200
    assert response.json()["llm_authentication"] == {
        "mode": "bearer", "source": "environment", "status": "configured",
    }
    assert "private-ambient-credential" not in response.text


def test_explicit_bearer_with_the_same_key_still_requires_restart(saved_settings, monkeypatch):
    _, _, active, client = saved_settings
    active.llm.provider = config.LLMProviderEnum.OPENAI
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Host: legacy-route.example")
    config.mark_startup_settings(active)
    body = {
        **explicit_payload(active),
        "llm_authentication": {"mode": "bearer", "api_key": active.llm.api_key},
    }
    response = client.put("/system/settings", json=body)
    assert response.status_code == 200
    assert response.json()["restart_required"] is True


def test_ordinary_model_discovery_uses_the_running_provider(monkeypatch):
    from types import SimpleNamespace

    from vision_bill.api.system import llm
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.security.dependencies import get_current_user

    async def running_models(*, capability):
        assert capability == "text"
        return [ModelInfo("live-model")]

    async def pending_models(*, capability):
        return [ModelInfo("pending-model")]

    monkeypatch.setattr(llm, "get_llm_provider", lambda _settings: SimpleNamespace(
        get_available_models=pending_models,
    ), raising=False)
    app = FastAPI()
    app.state.llm_provider = SimpleNamespace(get_available_models=running_models)
    app.include_router(llm.router)
    app.dependency_overrides[get_current_user] = lambda: None
    with TestClient(app) as client:
        assert [model["id"] for model in client.get("/models?capability=text").json()] == ["live-model"]
