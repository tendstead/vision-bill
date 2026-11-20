"""Bounded model discovery for the saved OpenAI-compatible connection."""

import asyncio
import json
from typing import Literal

import httpx

from ...config import LLMProviderEnum, LLMSettings, ambient_openai_headers
from . import OPENAI_CREDENTIALLESS_API_KEY

ConnectionStatus = Literal[
    "connected", "authentication_failed", "unreachable", "invalid_response", "origin_mismatch", "unsupported_provider",
]

PROBE_TIMEOUT_SECONDS = 5
MAX_DISCOVERY_BYTES = 2 * 1024 * 1024


async def check_saved_connection(settings: LLMSettings) -> ConnectionStatus:
    if settings.provider != LLMProviderEnum.OPENAI:
        return "unsupported_provider"
    key = settings.request_api_key()
    if settings.authentication is not None and settings.authentication.mode == "bearer" and key is None:
        return "origin_mismatch"
    headers = {"Accept-Encoding": "identity"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key or OPENAI_CREDENTIALLESS_API_KEY}"
    if settings.authentication is None:
        headers.update({
            name: value for name, value in ambient_openai_headers().items() if name.lower() != "accept-encoding"
        })
    try:
        url = httpx.URL(settings.host)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo:
            return "unreachable"
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS), httpx.AsyncClient(
            base_url=url, headers=headers, trust_env=False, follow_redirects=False,
        ) as client, client.stream("GET", "models") as response:
            if response.status_code in (401, 403):
                return "authentication_failed"
            if not response.is_success:
                return "unreachable"
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                return "invalid_response"
            body = bytearray()
            async for chunk in response.aiter_raw():
                if len(body) + len(chunk) > MAX_DISCOVERY_BYTES:
                    return "invalid_response"
                body.extend(chunk)
    except (httpx.HTTPError, httpx.InvalidURL, TimeoutError, UnicodeEncodeError):
        return "unreachable"
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return "invalid_response"
    return "connected" if isinstance(payload, dict) and isinstance(payload.get("data"), list) else "invalid_response"
