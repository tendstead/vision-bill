import re
from collections.abc import Collection, Mapping
from typing import TypeGuard

OPENAI_CREDENTIALLESS_API_KEY = "none"
DEFAULT_FORMATTER_MAX_OUTPUT_TOKENS = 32_768
MAX_MODEL_ID_LENGTH = 1024


def valid_model_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= MAX_MODEL_ID_LENGTH and value.isprintable()


def header_credentials(headers: Mapping[str, str]) -> set[str]:
    credentials = {value for value in headers.values() if value}
    for name, value in headers.items():
        if name.lower() in {"authorization", "proxy-authorization"}:
            parts = value.split(maxsplit=1)
            if len(parts) == 2:
                credentials.add(parts[1])
    return credentials


def contains_catalog_credential(value: str, credentials: Collection[str]) -> bool:
    for secret in credentials:
        if not secret or secret not in value:
            continue
        # Tiny local keys can also be letters within ordinary names such as "llama".
        if len(secret) >= 4 or re.search(r"(?<![a-zA-Z0-9])" + re.escape(secret) + r"(?![a-zA-Z0-9])", value):
            return True
    return False
