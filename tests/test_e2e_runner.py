"""The runner cannot adopt caller projects, dotenv, config or service ports."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "e2e"))


def test_owned_configuration_ignores_operator_compose_and_endpoint_environment(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location("owned_runner", ROOT / "e2e/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "operator")
    monkeypatch.setenv("VB_E2E_BASE_URL", "http://operator:8080")
    state = runner.create_state(tmp_path)
    assert state["project"].startswith("vb-e2e-")
    assert state["project"] != "operator"
    assert Path(state["env_file"]).read_text() == ""
    config = Path(state["directory"]) / "config/config.yaml"
    assert config.read_text() == (ROOT / "e2e/config/config.yaml").read_text()
    command, environment = runner.compose(state, "config")
    assert "--env-file" in command and state["env_file"] in command
    assert "--project-name" in command and state["project"] in command
    assert "COMPOSE_PROJECT_NAME" not in environment
    assert environment["VB_E2E_CONFIG_DIR"] == str(config.parent)
    assert "VB_E2E_BASE_URL" not in environment


def test_shipped_mandatory_fixture_images_fully_decode():
    import tomllib

    from PIL import Image

    data = ROOT / "tests/e2e/data"
    manifest = tomllib.loads((data / "fixtures.toml").read_text())
    for name in ("seed_receipt", "deposit_refund"):
        assert name in manifest["fixtures"]
        with Image.open(data / (name + ".png")) as image:
            image.load()
            assert image.width > 0 and image.height > 0


def test_fault_injection_accepts_only_exact_owned_submission_file(tmp_path):
    from uuid import uuid4

    import pytest
    from owned_files import remove_test_file

    image_id = str(uuid4())
    directory = tmp_path / "submissions" / image_id
    directory.mkdir(parents=True)
    original = directory / "0.png"
    original.write_bytes(b"owned original")
    thumbs = directory / "thumbnails"
    thumbs.mkdir()
    thumbnail = thumbs / "0.thumb.webp"
    thumbnail.write_bytes(b"owned thumbnail")
    remove_test_file(tmp_path, image_id, "thumbnail")
    assert original.read_bytes() == b"owned original" and not thumbnail.exists()
    with pytest.raises(ValueError):
        remove_test_file(tmp_path, "../not-owned", "original")
    remove_test_file(tmp_path, image_id, "original")
    assert not original.exists()


def test_owned_runner_rejects_malformed_saved_state_before_compose(tmp_path):
    import pytest

    spec = importlib.util.spec_from_file_location("typed_owned_runner", ROOT / "e2e/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    valid = runner.create_state(tmp_path)
    for state in ([], {**valid, "project": None}, {**valid, "directory": 12},
                  {**valid, "admin_can_see_all": "false"}, {**valid, "policy": "typo"}):
        with pytest.raises(ValueError, match="state"):
            runner.compose(state, "config")
