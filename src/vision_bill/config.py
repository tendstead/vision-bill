from __future__ import annotations

import logging
import os
import re
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote, urlsplit

import httpx
import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PostgresDsn,
    SecretStr,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
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

_startup_llm_identity: tuple[LLMProviderEnum, str, str | None] | None = None


def get_config_path() -> Path:
    """Return the external YAML path used for persistent application settings."""
    return Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)).expanduser()


class LLMProviderEnum(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class OcrProviderEnum(str, Enum):
    PADDLEOCR = "paddleocr"


def valid_bearer_key(value: str) -> bool:
    return re.fullmatch(r"[\x21-\x7e]{1,8192}", value) is not None


class LlmAuthentication(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    mode: Literal["none", "bearer"]
    api_key: SecretStr | None = None
    origin: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "none" and self.api_key is None and self.origin is None:
            return self
        if (
            self.mode == "bearer" and self.origin and self.api_key is not None
            and valid_bearer_key(self.api_key.get_secret_value())
        ):
            return self
        raise ValueError("Invalid LLM authentication settings")

    def private_record(self) -> dict[str, str]:
        if self.mode == "bearer" and self.api_key is not None and self.origin is not None:
            return {"mode": self.mode, "origin": self.origin, "api_key": self.api_key.get_secret_value()}
        return {"mode": self.mode}


class LLMSettings(BaseModel):
    provider: LLMProviderEnum = LLMProviderEnum.OLLAMA
    host: str  # Used for local LLMs
    api_key: str  # Used for external LLMs
    authentication: LlmAuthentication | None = Field(default=None, exclude=True, repr=False)
    model_name: str
    temperature: float
    # Keep the complete extraction deadline below the 15-minute image claim lease.
    analysis_timeout_seconds: float = Field(default=600, gt=0, le=600)
    max_output_tokens: int = Field(default=8192, gt=0)
    vision_model_ids: list[str] = Field(default_factory=list)
    # OpenAI-only option; other providers keep their own lifecycle after a switch.
    model_lifecycle: Literal["none", "llamacpp"] = "none"

    @field_validator("provider", mode="before")
    @classmethod
    def supported_provider(cls, value: Any) -> Any:
        if value not in ("ollama", "openai"):
            raise ValueError(
                "Unsupported LLM provider; select ollama or openai in LLM__PROVIDER or llm.provider"
            )
        return value

    @field_validator("host", "model_name")
    @classmethod
    def explicit_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LLM host and model_name must be explicitly selected and nonblank")
        return value

    def request_api_key(self) -> str | None:
        """Resolve the key for the selected endpoint, including an explicit omission."""
        auth = self.authentication
        if auth is None:
            return self.api_key
        if auth.api_key is None:
            return None
        try:
            if auth.origin == llm_origin(self.host):
                return auth.api_key.get_secret_value()
        except ValueError:
            pass
        return None


class ApiSettings(BaseModel):
    port: int = 8080
    log_level: str = "INFO"


class ImageSettings(BaseModel):
    save_dir: str = "/app/uploads/"
    tmp_dir: str = "/app/uploads_tmp/"
    bypass_review_default: bool = False
    thumbnail_max_edge: int = Field(default=512, ge=64)
    thumbnail_quality: int = Field(default=80, ge=1, le=100)


_DEFAULT_RETRY_BACKOFF_SECONDS = (30, 60)


class WorkerSettings(BaseModel):
    """Background analysis worker tuning."""

    check_interval_seconds: int = Field(default=300, ge=1)
    attempt_timeout_seconds: int = Field(default=600, gt=0, le=600)
    max_attempts: int = Field(default=3, ge=1, le=3)
    retry_backoff_seconds: tuple[Annotated[int, Field(gt=0)], ...] = _DEFAULT_RETRY_BACKOFF_SECONDS
    retry_backoff_cap_seconds: int = Field(default=300, gt=0)
    claim_lease_seconds: int = Field(default=900, gt=0)

    @model_validator(mode="after")
    def validate_budgets(self) -> Self:
        if self.claim_lease_seconds <= self.attempt_timeout_seconds:
            raise ValueError("claim lease must exceed the complete attempt deadline")
        if len(self.retry_backoff_seconds) < self.max_attempts - 1:
            raise ValueError("a backoff is required for each automatic retry")
        return self

    def retry_delay(self, attempt_number: int) -> int:
        """Resolve the current delay for an attempt from any accepted budget."""
        if attempt_number < 1:
            raise ValueError("attempt number must be positive")
        index = attempt_number - 1
        if index < len(self.retry_backoff_seconds):
            delay = self.retry_backoff_seconds[index]
        else:
            # A previously accepted job can outlive the schedule for new jobs.
            delay = _DEFAULT_RETRY_BACKOFF_SECONDS[min(index, len(_DEFAULT_RETRY_BACKOFF_SECONDS) - 1)]
        return min(delay, self.retry_backoff_cap_seconds)


class OcrSettings(BaseModel):
    """OCR service settings for the two-stage extraction pipeline."""

    provider: OcrProviderEnum = OcrProviderEnum.PADDLEOCR
    enabled: bool = False
    service_url: str = "http://paddleocr-service:8082"
    timeout_seconds: int = Field(default=300, ge=1)
    required: bool = False
    allow_vision_fallback: bool = False
    vision_fallback_model_id: str | None = None
    coordination_path: Path = Field(
        default_factory=lambda: get_config_path().parent / "gpu-handoff.lock"
    )
    manage_model_lifecycle: bool = True
    shared_gpu: bool = True

    @model_validator(mode="before")
    @classmethod
    def canonical_endpoint(cls, value: Any) -> Any:
        """An explicitly supplied endpoint selects OCR; legacy URLs do not."""
        if isinstance(value, dict) and "endpoint" in value:
            value = dict(value)
            endpoint = value.pop("endpoint")
            if not isinstance(endpoint, str):
                raise ValueError("OCR endpoint must be a string")
            value["service_url"] = endpoint.strip()
            if value["service_url"]:
                parsed = urlsplit(value["service_url"])
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    raise ValueError("OCR endpoint must be an absolute HTTP(S) URL")
                AnyHttpUrl(value["service_url"])
            value["enabled"] = bool(value["service_url"])
        return value

    @model_validator(mode="after")
    def required_needs_enabled(self) -> Self:
        if self.required and not self.enabled:
            raise ValueError("required OCR must also be enabled")
        return self


def validate_saved_ocr_config(saved: object, *, effective_enabled: bool) -> None:
    """Validate saved OCR fields with the effective enabled state for partial settings."""
    if (
        effective_enabled
        and isinstance(saved, dict)
        and "enabled" not in saved
        and "endpoint" not in saved
    ):
        saved = {**saved, "enabled": True}
    OcrSettings.model_validate(saved)


class PGSettings(BaseModel):
    user: str
    password: str
    db: str = "vision_bill"
    host: str = "localhost"
    port: int = 5432

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> PostgresDsn:
        return PostgresDsn(self.pg_dsn.replace("postgresql://", "postgresql+asyncpg://", 1))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pg_dsn(self) -> str:
        """Plain DSN suitable for asyncpg.create_pool."""
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"postgresql://{quote(self.user, safe='')}:{quote(self.password, safe='')}@{host}:{self.port}/{quote(self.db, safe='')}"


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


class AdminSettingsSource(YamlConfigSettingsSource):
    def __init__(
        self, settings_cls: type[BaseSettings],
        lower_sources: tuple[PydanticBaseSettingsSource, ...],
    ) -> None:
        super().__init__(settings_cls)
        self.lower_sources = lower_sources

    def __call__(self) -> dict[str, Any]:
        return _guarded_admin_overrides(super().__call__(), [source() for source in self.lower_sources])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_nested_delimiter="__",
        yaml_file=str(get_config_path()),
        extra="ignore",
        hide_input_in_errors=True,
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
        """Keep env defaults unless an admin explicitly adopted a safe YAML field.

        Earlier sources win. Credentials and infrastructure settings always keep
        their existing environment-over-YAML precedence.
        """
        return (
            init_settings,
            AdminSettingsSource(settings_cls, (init_settings, env_settings, dotenv_settings, file_secret_settings)),
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
    except (OSError, yaml.YAMLError):
        logger.warning("Could not read YAML config %s", path)
        return {}
    return data if isinstance(data, dict) else {}


LLM_ADOPTION_KEYS = frozenset(key for key in EDITABLE_ENV_KEYS if key.startswith("llm."))


def has_real_api_key(value: str) -> bool:
    return value.strip().lower() not in {"", "none"}


def ambient_openai_headers() -> dict[str, str]:
    """Read the custom header mapping used by the SDK's legacy connection."""
    return {
        name.strip(): value.strip()
        for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").splitlines()
        for name, separator, value in [line.partition(":")]
        if separator
    }


def has_ambient_openai_headers() -> bool:
    # A deployment-defined header can carry credentials under any name.
    return any(ambient_openai_headers().values())


def has_ambient_openai_credentials() -> bool:
    return any(
        name.lower() in {"authorization", "api-key", "x-api-key"} and bool(value)
        for name, value in ambient_openai_headers().items()
    )


def has_legacy_llm_credentials(api_key: str) -> bool:
    return has_real_api_key(api_key) or has_ambient_openai_headers()


def llm_origin(host: str) -> str:
    try:
        url = httpx.URL(host if "://" in host else "http://" + host)
    except httpx.InvalidURL:
        raise ValueError("LLM endpoint origin is invalid") from None
    if url.userinfo or not url.raw_host or url.scheme not in ("http", "https"):
        raise ValueError("LLM endpoint origin is invalid")
    port = url.port if url.port is not None else (443 if url.scheme == "https" else 80)
    return f"{url.scheme}://{url.raw_host.decode()}:{port}"


ADMIN_OVERRIDES_KEY = "admin_setting_overrides"
LLM_AUTHENTICATION_KEY = "llm_authentication"


def _admin_settings_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Only explicitly adopted safe fields may take precedence over environment."""
    result: dict[str, Any] = {}
    marked = data.get(ADMIN_OVERRIDES_KEY, [])
    if not isinstance(marked, list):
        return result
    for field in LLM_ADOPTION_KEYS:
        section, key = field.split(".", 1)
        if field in marked and isinstance(data.get(section), dict) and key in data[section]:
            result.setdefault(section, {})[key] = data[section][key]
    return result


def _guarded_admin_overrides(data: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the actual instance sources to keep a credential with its operator endpoint."""
    adopted = _admin_settings_overrides(data)
    authentication = data.get(LLM_AUTHENTICATION_KEY)
    if authentication is not None:
        adopted.setdefault("llm", {})["authentication"] = authentication
        return adopted
    pinned_origin = data.get("llm_credential_origin")
    if pinned_origin is None:
        return adopted

    def llm_values(source: dict[str, Any]) -> dict[str, Any]:
        value = source.get("llm", {})
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value if isinstance(value, dict) else {}

    # Source order is init, env, dotenv, secrets. YAML retains its ordinary
    # precedence above secrets and below dotenv; only the allowlist can outrank env.
    effective = dict(llm_values(sources[-1]))
    effective.update(llm_values(data))
    deployment: dict[str, Any] = {}
    for source in reversed(sources[:-1]):
        deployment.update(llm_values(source))
    effective.update(deployment)
    if not has_legacy_llm_credentials(effective.get("api_key", "")):
        return adopted
    deployment_host = deployment.get("host")
    if deployment_host is not None and llm_origin(deployment_host) != pinned_origin:
        # A rotated operator endpoint/key wins over a stale browser adoption.
        adopted.get("llm", {}).pop("host", None)
        return adopted
    effective.update(adopted.get("llm", {}))
    effective.update(llm_values(sources[0]))  # Explicit init values retain highest precedence.
    if llm_origin(effective["host"]) != pinned_origin:
        raise ValueError("LLM credential endpoint changed; operator approval in private configuration required")
    return adopted


def editable_setting_sources() -> dict[str, str]:
    """Return the source of each setting shown in the admin UI."""
    environment = EnvSettingsSource(Settings)()
    dotenv = DotEnvSettingsSource(Settings)()
    config_data = _read_yaml_config(get_config_path())
    sources: dict[str, str] = {}

    adopted = _guarded_admin_overrides(config_data, [{}, environment, dotenv, {}])
    for field_name in EDITABLE_ENV_KEYS:
        section, key = field_name.split(".", 1)
        if key in adopted.get(section, {}):
            sources[field_name] = "config"
            continue
        if any(
            isinstance(data, dict) and key in data
            for data in (environment.get(section), dotenv.get(section))
        ):
            sources[field_name] = "environment"
            continue
        section_data = config_data.get(section)
        sources[field_name] = (
            "config" if isinstance(section_data, dict) and key in section_data else "default"
        )
    return sources


def deployment_llm_credential_source() -> Literal["environment", "config"]:
    for data in (EnvSettingsSource(Settings)(), DotEnvSettingsSource(Settings)()):
        if isinstance(data.get("llm"), dict) and "api_key" in data["llm"]:
            return "environment"
    return "config"


class SavedLlmEndpointChanged(ValueError):
    """The shared settings document changed to a different LLM origin."""


def _saved_llm_host(data: dict[str, Any]) -> str | None:
    env = EnvSettingsSource(Settings)()
    dotenv = DotEnvSettingsSource(Settings)()
    adopted = _guarded_admin_overrides(data, [{}, env, dotenv, {}])
    for source in (adopted, env, dotenv, data):
        llm = source.get("llm")
        if isinstance(llm, dict) and isinstance(llm.get("host"), str):
            return str(llm["host"])
    return None


def saved_llm_settings() -> LLMSettings:
    """Read a fresh configuration under the same lock used to publish admin settings."""
    from .distribution.bootstrap import config_lock

    try:
        with config_lock(get_config_path().parent):
            return Settings().llm  # type: ignore[call-arg]
    except (OSError, ValueError, SettingsError, yaml.YAMLError):
        raise RuntimeError("Could not read saved LLM settings") from None


def write_config_file(
    config: Settings, *, changed_paths: frozenset[str] | None = None,
    credential_origin: str | None = None, llm_authentication_expected_origin: str | None = None,
) -> None:
    """Merge a save under the bootstrap lock without discarding private metadata."""
    _publish_config_file(
        get_config_path(), config, changed_paths=changed_paths, credential_origin=credential_origin,
        llm_authentication_expected_origin=llm_authentication_expected_origin,
    )


def _publish_config_file(
    path: Path,
    config: Settings,
    *,
    changed_paths: frozenset[str] | None = None,
    create_only: bool = False,
    credential_origin: str | None = None,
    llm_authentication_expected_origin: str | None = None,
) -> bool:
    from .distribution.bootstrap import (
        BootstrapError,
        config_lock,
        read_mapping,
        validate_config_data,
        write_private,
    )

    if changed_paths is not None and not changed_paths <= EDITABLE_ENV_KEYS.keys():
        raise BootstrapError("Only editable setting paths may be selectively saved")

    def merge(original: dict[str, Any], values: dict[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, dict):
                if key not in original:
                    original[key] = {}
                if not isinstance(original[key], dict):
                    raise BootstrapError("Persistent configuration has a conflicting section type")
                merge(original[key], value)
            else:
                original[key] = value

    with config_lock(path.parent):
        pending = path.parent / ".bootstrap-pending"
        if pending.exists() or pending.is_symlink():
            raise BootstrapError("Interrupted installation requires explicit recovery")
        existing = path.exists() or path.is_symlink()
        if create_only and existing:
            return False
        prepared = any(
            marker.exists() or marker.is_symlink()
            for marker in (
                path.parent / ".bootstrap-complete.yaml",
                path.parent / "initial-admin.yaml",
            )
        )
        if not existing and prepared:
            raise BootstrapError("Incomplete installation requires explicit recovery")
        resolved = config.model_dump(mode="json", exclude={"pg": {"database_url", "pg_dsn"}})
        data = read_mapping(path) if existing else (resolved if changed_paths is None else {})
        if llm_authentication_expected_origin is not None:
            saved_host = _saved_llm_host(data)
            if saved_host is not None and llm_origin(saved_host) != llm_authentication_expected_origin:
                raise SavedLlmEndpointChanged
        if existing:
            # Source deployments can keep credentials and other required fields in
            # the environment. Fill missing validation inputs only, without copying
            # them into the saved document or masking an invalid persisted value.
            validation = data if prepared else {
                section: (
                    {**values, **data[section]}
                    if isinstance(data.get(section), dict)
                    else data.get(section, values)
                )
                for section, values in resolved.items()
            }
            validate_config_data(validation)
            for section, model_field in Settings.model_fields.items():
                if section not in data:
                    continue
                if section == "ocr":
                    validate_saved_ocr_config(data[section], effective_enabled=config.ocr.enabled)
                else:
                    TypeAdapter(model_field.annotation).validate_python(validation[section])
        if changed_paths is None:
            if existing:
                merge(data, resolved)
        else:
            for field in sorted(changed_paths):
                section, key = field.split(".", 1)
                section_data = data.setdefault(section, {})
                if not isinstance(section_data, dict):
                    raise BootstrapError("Persistent configuration has a conflicting section type")
                section_data[key] = resolved[section][key]
        if llm_authentication_expected_origin is not None and config.llm.authentication is not None:
            data[LLM_AUTHENTICATION_KEY] = config.llm.authentication.private_record()
            if data.get("llm_credential_origin") is None:
                # Older loaders ignore the explicit mode. Keep the retained deployment key bound
                # to its original origin even when the same save changes the host.
                data["llm_credential_origin"] = llm_authentication_expected_origin
        if credential_origin is not None:
            if has_legacy_llm_credentials(config.llm.api_key):
                data["llm_credential_origin"] = credential_origin
            else:
                data.setdefault("llm_credential_origin", credential_origin)
        if changed_paths:
            sources = editable_setting_sources()
            adopted = set(data.get(ADMIN_OVERRIDES_KEY, []))
            adopted.update(field for field in changed_paths if field in LLM_ADOPTION_KEYS and sources[field] == "environment")
            if adopted:
                data[ADMIN_OVERRIDES_KEY] = sorted(adopted & EDITABLE_ENV_KEYS.keys())
        write_private(path, yaml.safe_dump(data, sort_keys=False), replace=existing)
        return True


def initialize_config_file(config: Settings) -> None:
    """Create the external config on first start from resolved env/default values."""
    path = get_config_path()
    if path.exists():
        return
    try:
        if _publish_config_file(path, config, create_only=True):
            logger.info("Created initial YAML config at %s", path)
    except OSError as exc:
        # Unit tests and read-only deployments may intentionally omit the
        # config volume; the application can still run from environment values.
        logger.warning("Could not create YAML config at %s: %s", path, exc)


def _llm_runtime_identity(config: Settings) -> tuple[LLMProviderEnum, str, str | None]:
    llm = config.llm
    key = llm.request_api_key() if llm.provider == LLMProviderEnum.OPENAI else None
    policy = "settings" if llm.authentication is not None else "deployment"
    fingerprint = sha256(f"{policy}\0{key}".encode()).hexdigest() if key is not None else None
    return llm.provider, llm.host, fingerprint


def mark_startup_settings(config: Settings) -> None:
    """Record the provider connection the running process initialized."""
    global _startup_llm_identity
    _startup_llm_identity = _llm_runtime_identity(config)


def runtime_restart_required(config: Settings) -> bool:
    """Whether the saved LLM connection differs from the live provider."""
    return _startup_llm_identity is not None and _startup_llm_identity != _llm_runtime_identity(config)


# Required fields (llm, pg and auth) are populated from YAML, dotenv files, or
# environment variables. The first start writes the resolved values to YAML.
try:
    settings = Settings()  # type: ignore[call-arg]
except SettingsError as exc:
    # The source diagnostic names the field and source. Its parser cause may
    # include the raw configuration, including sibling secrets.
    raise RuntimeError(str(exc)) from None
except (OSError, yaml.YAMLError):
    raise RuntimeError("Could not read application YAML config") from None
initialize_config_file(settings)
