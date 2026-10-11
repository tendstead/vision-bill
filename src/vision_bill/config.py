from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PostgresDsn, computed_field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV = "VISION_BILL_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("/app/config/config.yaml")

EDITABLE_ENV_KEYS = {
    "llm.provider": "LLM__PROVIDER",
    "llm.host": "LLM__HOST",
    "llm.model_name": "LLM__MODEL_NAME",
    "llm.temperature": "LLM__TEMPERATURE",
    "auth.allow_registration": "AUTH__ALLOW_REGISTRATION",
}

_startup_llm_identity: tuple[LLMProviderEnum, str] | None = None


def get_config_path() -> Path:
    """Return the external YAML path used for persistent application settings."""
    return Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)).expanduser()


class LLMProviderEnum(str, Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class OcrProviderEnum(str, Enum):
    PADDLEOCR = "paddleocr"


class LLMSettings(BaseModel):
    provider: LLMProviderEnum = LLMProviderEnum.OLLAMA
    host: str  # Used for local LLMs
    api_key: str  # Used for external LLMs
    model_name: str
    temperature: float
    # Keep the complete extraction deadline below the 15-minute image claim lease.
    analysis_timeout_seconds: float = Field(default=600, gt=0, le=600)


class ApiSettings(BaseModel):
    port: int = 8080
    log_level: str = "INFO"


class ImageSettings(BaseModel):
    save_dir: str = "/app/uploads/"
    tmp_dir: str = "/app/uploads_tmp/"
    bypass_review_default: bool = False
    thumbnail_max_edge: int = Field(default=512, ge=64)
    thumbnail_quality: int = Field(default=80, ge=1, le=100)


class WorkerSettings(BaseModel):
    """Background analysis worker tuning."""

    check_interval_seconds: int = Field(default=300, ge=1)


class OcrSettings(BaseModel):
    """OCR service settings for the two-stage extraction pipeline."""

    provider: OcrProviderEnum = OcrProviderEnum.PADDLEOCR
    service_url: str = "http://paddleocr-service:8082"
    timeout_seconds: int = Field(default=300, ge=1)
    required: bool = False


class PGSettings(BaseModel):
    user: str
    password: str
    db: str = "vision_bill"
    host: str = "localhost"
    port: int = 5432

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            path=self.db,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pg_dsn(self) -> str:
        """Plain DSN suitable for asyncpg.create_pool."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class AuthSettings(BaseModel):
    """Multi-user authentication settings.

    ``secret_key`` is required and signs the stateless session cookie. The
    optional ``pepper`` salts passwords before hashing; when unset the
    ``secret_key`` is used as the pepper, so a pepper is always in play.
    """

    secret_key: str
    session_cookie_name: str = "vb_session"
    session_max_age_seconds: int = 1_209_600  # 14 days
    session_secure: bool = False
    bootstrap_username: str | None = None
    bootstrap_password: str | None = None
    allow_registration: bool = True
    admin_can_see_all: bool = False
    pepper: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_nested_delimiter="__",
        yaml_file=str(get_config_path()),
        extra="ignore",
    )

    api: ApiSettings = ApiSettings()
    images: ImageSettings = ImageSettings()
    llm: LLMSettings
    pg: PGSettings
    worker: WorkerSettings = WorkerSettings()
    auth: AuthSettings
    ocr: OcrSettings = OcrSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load YAML before dotenv/environment values.

        Pydantic Settings gives earlier sources precedence, so environment
        variables and dotenv files override the persistent YAML file.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def _read_yaml_config(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, returning an empty mapping when it is absent."""
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read YAML config %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _dotenv_names(path: Path) -> set[str]:
    """Return setting names present in a dotenv file without reading values."""
    if not path.is_file():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()

    names: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = stripped.removeprefix("export ").split("=", 1)[0].strip()
        if name in EDITABLE_ENV_KEYS.values():
            names.add(name)
    return names


def editable_setting_sources() -> dict[str, str]:
    """Return the source of each setting shown in the admin UI."""
    environment_names = set(os.environ)
    environment_names.update(_dotenv_names(Path(".env")))
    environment_names.update(_dotenv_names(Path(".env.local")))
    config_data = _read_yaml_config(get_config_path())
    sources: dict[str, str] = {}

    for field_name, environment_name in EDITABLE_ENV_KEYS.items():
        if environment_name in environment_names:
            sources[field_name] = "environment"
            continue
        section, key = field_name.split(".", 1)
        section_data = config_data.get(section)
        sources[field_name] = (
            "config" if isinstance(section_data, dict) and key in section_data else "default"
        )
    return sources


def write_config_file(config: Settings) -> None:
    """Atomically persist settings while keeping the external file private."""
    path = get_config_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude={"pg": {"database_url", "pg_dsn"}})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(data, config_file, sort_keys=False, default_flow_style=False)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def initialize_config_file(config: Settings) -> None:
    """Create the external config on first start from resolved env/default values."""
    path = get_config_path()
    if path.exists():
        return
    try:
        write_config_file(config)
        logger.info("Created initial YAML config at %s", path)
    except OSError as exc:
        # Unit tests and read-only deployments may intentionally omit the
        # config volume; the application can still run from environment values.
        logger.warning("Could not create YAML config at %s: %s", path, exc)


def mark_startup_settings(config: Settings) -> None:
    """Record which provider/host the running process actually initialized."""
    global _startup_llm_identity
    _startup_llm_identity = (config.llm.provider, config.llm.host)


def runtime_restart_required(config: Settings) -> bool:
    """Whether the editable LLM provider/host differs from the live provider."""
    return _startup_llm_identity is not None and _startup_llm_identity != (
        config.llm.provider,
        config.llm.host,
    )


# Required fields (llm, pg and auth) are populated from YAML, dotenv files, or
# environment variables. The first start writes the resolved values to YAML.
settings = Settings()  # type: ignore[call-arg]
initialize_config_file(settings)
