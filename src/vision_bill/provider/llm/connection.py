"""Bounded model discovery for the saved formatter connection."""

import asyncio
import json
import os
from typing import Literal, NamedTuple

import httpx
from ollama._client import _parse_host

from ...config import LLMProviderEnum, LLMSettings, ambient_openai_headers, has_real_api_key
from . import (
    OPENAI_CREDENTIALLESS_API_KEY,
    contains_catalog_credential,
    header_credentials,
    valid_model_id,
)

ConnectionStatus = Literal[
    "connected", "authentication_failed", "unreachable", "invalid_response", "origin_mismatch", "unsupported_provider",
]

PROBE_TIMEOUT_SECONDS = 5
MAX_DISCOVERY_BYTES = 2 * 1024 * 1024
MAX_DISCOVERY_MODELS = 4096


class DiscoveryResult(NamedTuple):
    status: ConnectionStatus
    models: list[str]


def _model_ids(body: bytearray, native: bool, secrets: set[str]) -> DiscoveryResult:
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return DiscoveryResult("invalid_response", [])
    entries = payload.get("models" if native else "data") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or len(entries) > MAX_DISCOVERY_MODELS:
        return DiscoveryResult("invalid_response", [])
    ids: dict[str, None] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return DiscoveryResult("invalid_response", [])
        model_id = entry.get("model", entry.get("name")) if native else entry.get("id")
        if not valid_model_id(model_id) or contains_catalog_credential(model_id, secrets):
            return DiscoveryResult("invalid_response", [])
        ids[model_id] = None
    return DiscoveryResult("connected", list(ids))


async def check_saved_connection(settings: LLMSettings) -> DiscoveryResult:
    native = settings.provider == LLMProviderEnum.OLLAMA
    key = settings.request_api_key()
    if settings.authentication is not None and settings.authentication.mode == "bearer" and key is None:
        return DiscoveryResult("origin_mismatch", [])
    headers = {"Accept-Encoding": "identity"}
    secrets = {key} if key and (native or settings.authentication or has_real_api_key(key)) else set()
    if key is not None:
        headers["Authorization"] = f"Bearer {key or OPENAI_CREDENTIALLESS_API_KEY}"
    if not native and settings.authentication is None:
        custom = ambient_openai_headers()
        routing = {
            name: value for name, env in (
                ("OpenAI-Organization", "OPENAI_ORG_ID"), ("OpenAI-Project", "OPENAI_PROJECT_ID"),
            ) if (value := os.environ.get(env)) is not None
        }
        secrets.update(header_credentials(custom))
        secrets.update(header_credentials(routing))
        # SDK default_headers inserts custom names first, then the two routing
        # names, then reapplies custom values. Differently cased names coexist.
        inherited = {**custom, **routing, **custom}
        headers.update({name: value for name, value in inherited.items() if name.lower() != "accept-encoding"})
    try:
        host = settings.host
        # The native SDK accepts bare hosts (default port 11434) and preserves reverse-proxy paths.
        supplied = httpx.URL(f"http://{host}" if native and "://" not in host else host)
        if (
            supplied.scheme not in ("http", "https") or not supplied.host
            or supplied.userinfo or "?" in host or "#" in host
        ):
            return DiscoveryResult("unreachable", [])
        url = httpx.URL(_parse_host(host)) if native else supplied
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS), httpx.AsyncClient(
            base_url=url, headers=headers, trust_env=False, follow_redirects=False,
        ) as client, client.stream("GET", "api/tags" if native else "models") as response:
            if response.status_code in (401, 403):
                return DiscoveryResult("authentication_failed", [])
            if not response.is_success:
                return DiscoveryResult("unreachable", [])
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                return DiscoveryResult("invalid_response", [])
            body = bytearray()
            async for chunk in response.aiter_raw():
                if len(body) + len(chunk) > MAX_DISCOVERY_BYTES:
                    return DiscoveryResult("invalid_response", [])
                body.extend(chunk)
    except (httpx.HTTPError, httpx.InvalidURL, TimeoutError, ValueError):
        return DiscoveryResult("unreachable", [])
    return _model_ids(body, native, secrets)
