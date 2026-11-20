"""Managed bundle wiring validates actual Settings precedence without contacting services."""

import os
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize(
    "saved,runtime,expected",
    [
        ({}, {"OCR__ENDPOINT": "http://ocr:8082"}, 0),
        pytest.param(
            {"required": True},
            {"OCR__ENDPOINT": "http://ocr:8082"},
            0,
            id="canonical-runtime-endpoint-completes-partial-saved-ocr",
        ),
        pytest.param(
            {"enabled": True},
            {"OCR__ENDPOINT": "http://ocr:8082"},
            0,
            id="canonical-runtime-endpoint-supplies-missing-saved-service-url",
        ),
        pytest.param(
            {"timeout_seconds": 0},
            {"OCR__ENDPOINT": "http://ocr:8082", "OCR__TIMEOUT_SECONDS": "600"},
            1,
            id="runtime-timeout-cannot-hide-invalid-saved-timeout",
        ),
        ({"endpoint": "http://ocr:8082"}, {}, 0),
        ({"enabled": True, "service_url": "http://ocr:8082"}, {}, 0),
        ({}, {}, 1),
        ({}, {"OCR__ENDPOINT": ""}, 1),
        ({}, {"OCR__ENDPOINT": "http://external:8082"}, 1),
        ({"endpoint": "http://external:8082"}, {"OCR__ENDPOINT": "http://ocr:8082"}, 1),
        (
            {"enabled": True, "service_url": "http://external:8082"},
            {"OCR__ENDPOINT": "http://ocr:8082"},
            1,
        ),
        ({}, {"OCR__ENDPOINT": "http://ocr:8082", "OCR__MANAGE_MODEL_LIFECYCLE": "false"}, 1),
        pytest.param(
            {},
            {"OCR__ENDPOINT": "http://ocr:8082", "OCR__SERVICE_URL": "http://external:8082"},
            1,
            id="runtime-canonical-endpoint-cannot-hide-foreign-service-url",
        ),
        pytest.param(
            {},
            {"OCR__ENDPOINT": "http://ocr:8082", "OCR__ENABLED": "false"},
            1,
            id="runtime-canonical-endpoint-cannot-hide-disabled-ocr",
        ),
        pytest.param(
            {"endpoint": "http://ocr:8082", "manage_model_lifecycle": False},
            {"OCR__MANAGE_MODEL_LIFECYCLE": "true"},
            1,
            id="runtime-lifecycle-cannot-hide-saved-unmanaged-policy",
        ),
        pytest.param(
            {"endpoint": "http://ocr:8082", "allow_vision_fallback": True},
            {"OCR__ALLOW_VISION_FALLBACK": "false"},
            1,
            id="runtime-text-only-cannot-hide-saved-vision-fallback-policy",
        ),
        pytest.param(
            {"endpoint": "http://ocr:8082", "service_url": "http://external:8082"},
            {},
            1,
            id="saved-canonical-endpoint-cannot-hide-foreign-service-url",
        ),
        pytest.param(
            {"endpoint": "http://ocr:8082", "enabled": False},
            {},
            1,
            id="saved-canonical-endpoint-cannot-hide-disabled-ocr",
        ),
        pytest.param(
            {},
            {
                "OCR__ENDPOINT": "http://ocr:8082",
                "OCR__SERVICE_URL": "http://ocr:8082",
                "OCR__ENABLED": "1",
                "OCR__MANAGE_MODEL_LIFECYCLE": "on",
                "OCR__ALLOW_VISION_FALLBACK": "off",
            },
            0,
            id="aligned-runtime-aliases-preserve-normal-boolean-parsing",
        ),
        pytest.param(
            {
                "endpoint": "http://ocr:8082",
                "enabled": True,
                "service_url": "http://ocr:8082",
                "manage_model_lifecycle": True,
                "allow_vision_fallback": False,
            },
            {},
            0,
            id="aligned-restored-local-topology-with-explicit-policy",
        ),
    ],
)
def test_managed_runtime_guard_uses_effective_and_saved_topology(
    tmp_path, saved, runtime, expected
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "llm": {
                    "provider": "openai",
                    "host": "http://localhost:1234/v1",
                    "api_key": "secret-do-not-log",
                    "model_name": "explicit",
                    "temperature": 0,
                },
                "pg": {"host": "db", "user": "user", "password": "pw", "db": "db"},
                "auth": {"secret_key": "secret-do-not-log"},
                "ocr": saved,
            }
        )
    )
    config.chmod(0o600)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LLM__", "OCR__", "PG__", "AUTH__"))
    }
    env.update(runtime)
    env["VISION_BILL_CONFIG_PATH"] = str(config)
    before = config.read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "vision_bill.distribution.bootstrap", "--check-managed-ocr"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected
    assert config.read_bytes() == before
    assert "secret-do-not-log" not in result.stdout + result.stderr
    if expected:
        assert "Managed OCR bundle" in result.stderr


def test_managed_initialization_reuses_valid_restored_config_without_seeds(tmp_path):
    config_dir = tmp_path / "config"
    config_path = config_dir / "config.yaml"
    command = [
        sys.executable,
        "-m",
        "vision_bill.distribution.bootstrap",
        "--config-dir",
        str(config_dir),
        "--postgres-password-file",
        str(tmp_path / "pg-secret" / "password"),
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LLM__", "OCR__", "PG__", "AUTH__"))
    }
    env.update(
        {
            "VISION_BILL_CONFIG_PATH": str(config_path),
            "VISION_BILL_MANAGED_OCR": "true",
            "LLM__PROVIDER": "openai",
            "LLM__HOST": "http://localhost:1234/v1",
            "LLM__MODEL_NAME": "explicit",
            "OCR__ENDPOINT": "http://ocr:8082",
        }
    )
    first = subprocess.run(
        command, env=env, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert first.returncode == 0, first.stderr
    saved = yaml.safe_load(config_path.read_text())
    saved["ocr"] = {"required": True}
    config_path.write_text(yaml.safe_dump(saved))
    before = config_path.read_bytes()
    for name in ("LLM__PROVIDER", "LLM__HOST", "LLM__MODEL_NAME"):
        env.pop(name)
    second = subprocess.run(
        command, env=env, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert second.returncode == 0, second.stderr
    assert config_path.read_bytes() == before
