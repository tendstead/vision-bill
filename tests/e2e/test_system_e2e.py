"""System API: public ui-config, model list, admin settings."""

import pytest

from e2e.helpers import API, register_user

pytestmark = pytest.mark.e2e


def _admin_client(http_factory):
    client = http_factory()
    response = client.post(
        f"{API}/auth/login",
        json={"username": "admin", "password": "admin-e2e-pass"},
    )
    assert response.status_code == 200
    return client


def test_ui_config_is_public(http_factory) -> None:
    client = http_factory()
    response = client.get(f"{API}/system/ui-config")
    assert response.status_code == 200
    body = response.json()
    assert body["registration_open"] is True
    assert body["bypass_review_default"] is False


def test_llm_models_lists_stub_model(http_factory) -> None:
    client, _ = register_user(http_factory, "sys")
    response = client.get(f"{API}/llm/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()]
    assert "e2e-vision" in ids


def test_settings_sources_and_env_conflict(http_factory) -> None:
    client = _admin_client(http_factory)
    view = client.get(f"{API}/system/settings").json()
    assert view["llm"]["provider"] == "openai"
    assert view["sources"]["llm.provider"] == "environment"
    assert view["sources"]["llm.model_name"] == "environment"
    assert view["sources"]["llm.temperature"] != "environment"

    # Legacy full PUT keeps environment-controlled keys locked.
    conflict = client.put(
        f"{API}/system/settings",
        json={
            "llm": {
                "provider": "ollama",
                "host": view["llm"]["host"],
                "model_name": view["llm"]["model_name"],
                "temperature": view["llm"]["temperature"],
            },
            "allow_registration": view["allow_registration"],
        },
    )
    assert conflict.status_code == 409
    assert "llm.provider" in conflict.json()["detail"]

    baseline = {"llm": dict(view["llm"]), "allow_registration": view["allow_registration"]}
    new_temp = 0.1 if view["llm"]["temperature"] != 0.1 else 0.2
    new_model = view["llm"]["model_name"] + "-explicit-settings-adoption"
    try:
        # Legacy full PUT still supports ordinary config-backed fields.
        ok = client.put(
            f"{API}/system/settings",
            json={**baseline, "llm": {**view["llm"], "temperature": new_temp}},
        )
        assert ok.status_code == 200
        assert ok.json()["llm"]["temperature"] == new_temp

        # Browser-style explicit adoption can change an environment default.
        # Its stale temperature is deliberately outside the dirty field mask.
        adopted = client.put(
            f"{API}/system/settings",
            json={
                **baseline,
                "llm": {**view["llm"], "model_name": new_model},
                "changed_paths": ["llm.model_name"],
            },
        )
        assert adopted.status_code == 200
        assert adopted.json()["llm"]["model_name"] == new_model
        assert adopted.json()["llm"]["temperature"] == new_temp
        assert adopted.json()["sources"]["llm.model_name"] == "config"
        assert adopted.json()["sources"]["llm.provider"] == "environment"
        persisted = client.get(f"{API}/system/settings").json()
        assert persisted["llm"]["model_name"] == new_model
        assert persisted["llm"]["temperature"] == new_temp
    finally:
        restored = client.put(
            f"{API}/system/settings",
            json={**baseline, "changed_paths": ["llm.model_name", "llm.temperature"]},
        )
        assert restored.status_code == 200
        assert restored.json()["llm"] == view["llm"]
