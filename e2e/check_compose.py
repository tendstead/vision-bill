"""Validate deployment overlays without reading operator dotenv or starting services."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def resolve_compose(command: list[str], directory: Path, env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        [*command, "config", "--format", "json"],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        # Never print expanded config/environment, including on failure.
        raise RuntimeError(f"Compose configuration failed (exit {result.returncode})")
    return cast(dict[str, Any], json.loads(result.stdout))


def main() -> None:
    # Compose config needs no service credentials. Use a fresh private dotenv for
    # interpolation AND replace the application's service-level env_file list.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"}
    }
    with tempfile.TemporaryDirectory(prefix="vb-compose-check-") as directory:
        private = Path(directory)
        dotenv = private / "test.env"
        dotenv.write_text(
            "PG__USER=compose_test\nPG__PASSWORD=unused_test_value\nPG__DB=compose_test\nAPI__PORT=58089\n"
        )
        override = private / "override.yaml"
        override.write_text(
            "services:\n  vision_bill:\n    env_file: !override\n      - "
            + json.dumps(str(dotenv))
            + "\n"
        )
        for accelerator in (None, "cuda", "rocm"):
            command = [
                "docker",
                "compose",
                "--env-file",
                str(dotenv),
                "--project-name",
                "vb-private-compose-check",
                "--project-directory",
                str(ROOT),
                *(["--profile", "ocr"] if accelerator else []),
                "-f",
                str(ROOT / "docker-compose.yml"),
                *(["-f", str(ROOT / f"docker-compose.{accelerator}.yml")] if accelerator else []),
                "-f",
                str(override),
            ]
            config = resolve_compose(command, private, environment)
            services = config["services"]
            assert set(services) == (
                {"vision_bill", "db", "paddleocr-service"} if accelerator else {"vision_bill", "db"}
            )
            assert services["db"]["image"] == "postgres:18"
            assert services["db"]["environment"]["POSTGRES_USER"] == "compose_test"
            assert services["vision_bill"]["environment"]["PG__PASSWORD"] == "unused_test_value"
            app_env = services["vision_bill"]["environment"]
            assert app_env["OCR__TIMEOUT_SECONDS"] == "600"
            assert str(app_env.get("OCR__ENABLED", "false")).lower() == "false"
            assert "OCR__ENDPOINT" not in app_env
            assert str(app_env.get("OCR__REQUIRED", "false")).lower() == "false"
            assert str(app_env.get("OCR__ALLOW_VISION_FALLBACK", "false")).lower() == "false"
            if accelerator is None:
                configured = resolve_compose(
                    command, private, {**environment, "OCR__TIMEOUT_SECONDS": "723"}
                )
                assert (
                    configured["services"]["vision_bill"]["environment"]["OCR__TIMEOUT_SECONDS"]
                    == "723"
                )
                assert "VISION_BILL_OCR_ACTIVATION" not in app_env
                assert set(config["volumes"]) == {"db_data"}
                print("main: OCR disabled (no services started)")
                continue
            ocr = services["paddleocr-service"]
            assert app_env["VISION_BILL_OCR_ACTIVATION"] == "true"
            assert app_env["OCR_ACTIVATION_ENDPOINT"] == "http://paddleocr-service:8082"
            assert (
                app_env["OCR_ACTIVATION_HANDOFF_LOCK_PATH"]
                == ocr["environment"]["OCR_HANDOFF_LOCK_PATH"]
            )
            assert ocr["command"] == ["python", "-m", "ocr_service.activation"]
            assert not ocr.get("env_file")
            assert not ocr.get("ports")
            assert set(ocr["networks"]) == {"default"}
            assert "default" in services["vision_bill"]["networks"]
            app_mounts = {mount["target"]: mount for mount in services["vision_bill"]["volumes"]}
            ocr_mounts = {mount["target"]: mount for mount in ocr["volumes"]}
            for target, app_read_only in [
                ("/activation/request", False),
                ("/activation/status", True),
                ("/activation/lock", False),
                ("/coord", False),
            ]:
                app_mount, ocr_mount = app_mounts[target], ocr_mounts[target]
                assert app_mount["type"] == ocr_mount["type"] == "volume"
                assert app_mount["source"] == ocr_mount["source"]
                assert config["volumes"][app_mount["source"]]["name"].startswith(
                    "vb-private-compose-check_"
                )
                assert app_mount.get("read_only", False) == app_read_only
                assert ocr_mount.get("read_only", False) == (target == "/activation/request")
            assert "/app/config" not in ocr_mounts
            assert ocr_mounts["/release"]["read_only"]
            assert not ocr_mounts["/var/lib/ocr"].get("read_only", False)
            assert ocr["build"]["context"] == str(ROOT / "paddleocr-service")
            assert ocr["build"]["dockerfile"] == f"Dockerfile.{accelerator}"
            assert ocr["profiles"] == ["ocr"]
            assert not ocr.get("privileged", False)
            if accelerator == "cuda":
                device = ocr["deploy"]["resources"]["reservations"]["devices"][0]
                assert device["driver"] == "nvidia" and device["device_ids"] == ["0"]
            else:
                assert {(device["source"], device["target"]) for device in ocr["devices"]} == {
                    ("/dev/kfd", "/dev/kfd"),
                    ("/dev/dri/renderD128", "/dev/dri/renderD128"),
                }
                selected = resolve_compose(
                    command, private, {**environment, "OCR_RENDER_NODE": "/dev/dri/renderD129"}
                )["services"]["paddleocr-service"]["devices"]
                assert [(device["source"], device["target"]) for device in selected] == [
                    ("/dev/kfd", "/dev/kfd"),
                    ("/dev/dri/renderD129", "/dev/dri/renderD129"),
                ]
            manual = resolve_compose(
                [*command, "-f", str(ROOT / "docker-compose.ocr-qualification.yml")],
                private,
                environment,
            )["services"]
            assert manual["vision_bill"]["environment"]["VISION_BILL_OCR_ACTIVATION"] == "false"
            qualifier = manual["paddleocr-service"]
            assert "build" not in qualifier
            assert qualifier["pull_policy"] == "never"
            assert qualifier["command"] == ["python", "-m", "ocr_service.entrypoint"]
            assert {mount["target"] for mount in qualifier["volumes"]} == {
                "/var/cache/paddlex",
                "/var/lib/ocr",
                "/coord",
            }
            print(f"main + {accelerator}: configuration valid (no services started)")


if __name__ == "__main__":
    main()
