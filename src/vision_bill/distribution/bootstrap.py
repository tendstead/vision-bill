"""Create persistent installation secrets before application settings are imported."""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import secrets
import stat
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from vision_bill.provider.llm import OPENAI_CREDENTIALLESS_API_KEY


class BootstrapError(ValueError):
    """Unsafe or incomplete persistent installation state."""


def private_directory(path: Path) -> None:
    """Create a private directory, refusing symlinks or another owner's state."""
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise BootstrapError("Secret directory must not traverse a symlink")
    if not path.parent.exists():
        private_directory(path.parent)
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise BootstrapError("Secret directory has an unexpected owner or type")
    path.chmod(0o700)


def private_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise BootstrapError("Secret file has an unexpected owner or type")
    path.chmod(0o600)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_private(path: Path, content: str, *, replace: bool = False) -> None:
    """Publish fully written 0600 files atomically; never clobber by default."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            if path.exists() or path.is_symlink():
                private_file(path)
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def config_lock(config_dir: Path) -> Iterator[None]:
    """Serialize bootstrap and application configuration publication."""
    private_directory(config_dir)
    lock_path = config_dir / ".bootstrap.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        private_file(lock_path)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class _UniqueKeyLoader(yaml.SafeLoader):
    def __init__(self, stream: str):
        super().__init__(stream)
        self._checked_mappings: set[yaml.MappingNode] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        # PyYAML flattens shared merge sources in place. Check each mapping
        # before its first flatten, while only explicitly declared keys are present.
        if node not in self._checked_mappings:
            self._checked_mappings.add(node)
            merge_key = object()
            keys: set[Any] = set()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    key = merge_key
                elif key_node.tag == "tag:yaml.org,2002:value":
                    key = self.construct_scalar(key_node)
                else:
                    key = self.construct_object(key_node)
                if key in keys:
                    raise BootstrapError("Persistent YAML has conflicting duplicate keys")
                keys.add(key)
        super().flatten_mapping(node)


def read_mapping(path: Path) -> dict[str, Any]:
    private_file(path)
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, TypeError) as exc:
        raise BootstrapError("Persistent YAML is malformed") from exc
    if not isinstance(value, dict):
        raise BootstrapError("Persistent YAML must be a mapping")
    return value


def read_config(path: Path) -> dict[str, Any]:
    """Read existing credentials without importing config's settings singleton."""
    data = read_mapping(path)
    validate_config_data(data)
    return data


def validate_config_data(data: dict[str, Any]) -> None:
    """Require the complete identity and credential fields of an installation."""
    for section, names in {
        "llm": ("host", "model_name"),
        "pg": ("user", "password"),
        "auth": ("secret_key",),
    }.items():
        values = data.get(section)
        if not isinstance(values, dict) or any(
            not isinstance(values.get(name), str) or not values[name] for name in names
        ):
            raise BootstrapError("Persistent configuration is incomplete")
    llm = data["llm"]
    if not all(llm[name].strip() for name in ("host", "model_name")):
        raise BootstrapError("Persistent LLM host and model_name must be nonblank")
    if llm.get("provider", "ollama") not in ("ollama", "openai"):
        raise BootstrapError("Unsupported LLM provider; select ollama or openai in llm.provider")
    if not isinstance(llm.get("api_key"), str) or not isinstance(
        llm.get("temperature"), (int, float)
    ):
        raise BootstrapError("Persistent LLM configuration is invalid")
    auth = data["auth"]
    username, password = auth.get("bootstrap_username"), auth.get("bootstrap_password")
    if bool(username) != bool(password) or (
        username and (not isinstance(username, str) or not isinstance(password, str))
    ):
        raise BootstrapError("Persistent bootstrap credentials are incomplete")


def _fresh_llm_settings(environment: Mapping[str, str]) -> dict[str, Any]:
    for name in ("LLM__PROVIDER", "LLM__HOST", "LLM__MODEL_NAME"):
        if not environment.get(name, "").strip():
            raise BootstrapError(f"Fresh installation requires an explicit nonblank {name}")
    provider = environment["LLM__PROVIDER"]
    if provider not in ("ollama", "openai"):
        raise BootstrapError("Unsupported LLM__PROVIDER; select ollama or openai")
    api_key = environment.get(
        "LLM__API_KEY", OPENAI_CREDENTIALLESS_API_KEY if provider == "openai" else ""
    )
    if provider == "openai" and not api_key.strip():
        raise BootstrapError(
            "LLM__API_KEY must be nonblank for OpenAI; use none for a credentialless endpoint"
        )
    try:
        temperature = float(environment.get("LLM__TEMPERATURE", "0"))
    except ValueError as exc:
        raise BootstrapError("LLM__TEMPERATURE must be a finite number between 0 and 2") from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise BootstrapError("LLM__TEMPERATURE must be a finite number between 0 and 2")
    return {
        "provider": provider,
        "host": environment["LLM__HOST"],
        "api_key": api_key,
        "model_name": environment["LLM__MODEL_NAME"],
        "temperature": temperature,
    }


def initialize(
    config_dir: Path,
    postgres_password_file: Path,
    *,
    pg_user: str = "vision_bill",
    pg_database: str = "vision_bill",
    pg_host: str = "db",
    runtime_environment: Mapping[str, str] | None = None,
) -> None:
    private_directory(config_dir)
    private_directory(postgres_password_file.parent)
    config_path = config_dir / "config.yaml"
    admin_path = config_dir / "initial-admin.yaml"
    pending = config_dir / ".bootstrap-pending"
    complete = config_dir / ".bootstrap-complete.yaml"
    with config_lock(config_dir):
        if pending.exists() or pending.is_symlink():
            raise BootstrapError("Interrupted installation requires explicit recovery")
        existing = config_path.exists() or config_path.is_symlink()
        if not existing and any(
            p.exists() or p.is_symlink() for p in (admin_path, postgres_password_file, complete)
        ):
            raise BootstrapError("Incomplete installation requires explicit recovery")
        if existing:
            data = read_config(config_path)
        else:
            llm = _fresh_llm_settings(
                os.environ if runtime_environment is None else runtime_environment
            )
            if not all((pg_user, pg_database, pg_host)):
                raise BootstrapError("Installation identities must not be empty")
            data = {
                "llm": llm,
                "pg": {
                    "user": pg_user,
                    "password": secrets.token_urlsafe(36),
                    "db": pg_database,
                    "host": pg_host,
                    "port": 5432,
                },
                "auth": {
                    "secret_key": secrets.token_urlsafe(48),
                    "bootstrap_username": "admin",
                    "bootstrap_password": secrets.token_urlsafe(36),
                    "allow_registration": True,
                    "admin_can_see_all": False,
                },
            }
        auth = data["auth"]
        admin = (
            {"username": auth["bootstrap_username"], "password": auth["bootstrap_password"]}
            if auth.get("bootstrap_username")
            else None
        )
        has_password = postgres_password_file.exists() or postgres_password_file.is_symlink()
        if has_password:
            private_file(postgres_password_file)
            if postgres_password_file.read_text(encoding="utf-8") != data["pg"]["password"] + "\n":
                raise BootstrapError("PostgreSQL password file conflicts with configuration")
        has_admin = admin_path.exists() or admin_path.is_symlink()
        if has_admin and read_mapping(admin_path) != admin:
            raise BootstrapError("Initial admin file conflicts with configuration")
        if complete.exists() or complete.is_symlink():
            state = read_mapping(complete)
            if state != {"version": 1, "admin_credentials": admin is not None} or (
                not has_password or (admin is not None and not has_admin)
            ):
                raise BootstrapError("Completed installation is incomplete or conflicting")
            return
        write_private(pending, "Installation incomplete; do not reset secrets automatically.\n")
        if not has_password:
            write_private(postgres_password_file, data["pg"]["password"] + "\n")
        if admin is not None and not has_admin:
            write_private(admin_path, yaml.safe_dump(admin))
        if not existing:
            write_private(config_path, yaml.safe_dump(data, sort_keys=False))
        write_private(
            complete, yaml.safe_dump({"version": 1, "admin_credentials": admin is not None})
        )
        pending.unlink()
        sync_directory(config_dir)


def validate_managed_ocr() -> None:
    """Reject conflicting prepared-bundle wiring using the application's real precedence."""
    try:
        from pydantic import TypeAdapter
        from pydantic_settings import EnvSettingsSource

        # Deliberately after initialization: config imports eagerly resolve settings.
        from vision_bill.config import (
            Settings,
            get_config_path,
            settings,
            validate_saved_ocr_config,
        )

        saved_ocr = read_config(get_config_path()).get("ocr", {})
        runtime_ocr = EnvSettingsSource(Settings)().get("ocr", {})
        for explicit in (saved_ocr, runtime_ocr):
            # Endpoint normalization overwrites aliases. Inspect each supplied
            # instruction independently so neither normalization nor an override
            # can conceal contradictory managed-bundle wiring or policy.
            if not isinstance(explicit, Mapping):
                raise BootstrapError("conflicting explicit OCR section")
            if "endpoint" in explicit and (
                not isinstance(explicit["endpoint"], str)
                or explicit["endpoint"].strip() != "http://ocr:8082"
            ):
                raise ValueError("conflicting explicit endpoint")
            if "service_url" in explicit and explicit["service_url"] != "http://ocr:8082":
                raise ValueError("conflicting explicit service URL")
            for name, expected in (
                ("enabled", True),
                ("manage_model_lifecycle", True),
                ("allow_vision_fallback", False),
            ):
                if (
                    name in explicit
                    and TypeAdapter(bool).validate_python(explicit[name]) != expected
                ):
                    raise ValueError("conflicting explicit managed policy")
        effective = settings.ocr
        if (
            not effective.enabled
            or effective.service_url != "http://ocr:8082"
            or not effective.manage_model_lifecycle
            or effective.allow_vision_fallback
        ):
            raise ValueError("conflicting effective topology")
        validate_saved_ocr_config(saved_ocr, effective_enabled=effective.enabled)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise BootstrapError(
            "Managed OCR bundle requires OCR__ENDPOINT=http://ocr:8082, "
            "managed model lifecycle and text-only formatting. Resolve conflicting saved "
            "OCR topology or runtime.env overrides explicitly, or use the app-only bundle."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("/app/config"))
    parser.add_argument("--postgres-password-file", type=Path)
    parser.add_argument("--check-managed-ocr", action="store_true")
    parser.add_argument("--pg-user", default="vision_bill")
    parser.add_argument("--pg-database", default="vision_bill")
    parser.add_argument("--pg-host", default="db")
    args = parser.parse_args()
    try:
        if args.check_managed_ocr:
            validate_managed_ocr()
        else:
            if args.postgres_password_file is None:
                parser.error("--postgres-password-file is required for initialization")
            options = vars(args).copy()
            options.pop("check_managed_ocr")
            initialize(**options)
            if os.environ.get("VISION_BILL_MANAGED_OCR") == "true":
                validate_managed_ocr()
    except BootstrapError as exc:
        print(f"Installation initialization failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, yaml.YAMLError):
        print(
            "Installation initialization failed: unsafe, conflicting, or incomplete state.",
            file=sys.stderr,
        )
        return 1
    print("Private installation configuration is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
