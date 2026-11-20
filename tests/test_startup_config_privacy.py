"""Startup diagnostics identify invalid settings without emitting config values."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("source", ["environment", "dotenv", "yaml"])
def test_broken_startup_config_does_not_log_credentials(tmp_path: Path, source: str) -> None:
    canary = "S3cRet"
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.yaml"
    env = {key: value for key, value in os.environ.items() if "__" not in key}
    env.update(
        PYTHONPATH=str(root / "src"),
        VISION_BILL_CONFIG_PATH=str(config),
        LLM__API_KEY=canary,
        LLM__HOST="http://model.local/v1",
        LLM__TEMPERATURE="0.4",
        PG__USER="test",
        PG__PASSWORD="test",
        AUTH__SECRET_KEY="test",
    )
    if source == "dotenv":
        (tmp_path / ".env.local").write_text(f"LLM='{{ {canary}\n'\n", encoding="utf-8")
        env.pop("LLM__API_KEY")
    elif source == "yaml":
        config.write_text(f"llm: {{api_key: {canary} : 'oops'}}\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", "import vision_bill.config"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert canary not in result.stderr
    expected = {
        "environment": "llm.model_name",
        "dotenv": 'field "llm"',
        "yaml": "Could not read application YAML config",
    }
    assert expected[source] in result.stderr
    assert "input_value" not in result.stderr
    assert "JSONDecodeError" not in result.stderr
    assert "yaml.parser.ParserError" not in result.stderr
