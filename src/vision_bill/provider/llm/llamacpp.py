"""Confirmed release of exact model IDs on a dedicated llama.cpp router."""

import asyncio
import hashlib
from time import monotonic
from urllib.parse import urlsplit, urlunsplit

import httpx


class LlamaCppLifecycle:
    def __init__(self, host: str, api_key: str | None):
        parts = urlsplit(host)
        path = parts.path
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or "%" in path
            or "\\" in path
            or "//" in path
            or any(segment in (".", "..") for segment in path.split("/"))
        ):
            raise ValueError("llamacpp lifecycle requires an unambiguous local router URL")
        # Validate the port before any request or journal identity is created.
        _ = parts.port
        path = path.removesuffix("/").removesuffix("/v1")
        self._base_url = urlunsplit((parts.scheme, parts.netloc, path + "/", "", ""))
        self._api_key = api_key
        self.identity = "llamacpp:" + hashlib.sha256(self._base_url.encode()).hexdigest()

    async def release_model(self, model_id: str) -> None:
        from ...service.gpu_handoff import current_owner

        if not model_id:
            raise ValueError("model_release_unconfirmed: canonical model ID is required")
        owner = current_owner()
        if owner is not None:
            if owner.resources.llm != self.identity:
                raise ValueError("model_handoff_identity_changed")
            owner.require_release(model_id)
        deadline = min(monotonic() + 30, owner.deadline) if owner else monotonic() + 30
        await self._release(model_id, deadline)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key is not None else None,
            timeout=httpx.Timeout(30, connect=5), trust_env=False, follow_redirects=False,
        )

    async def resident_model_ids(self, *, deadline: float) -> frozenset[str]:
        async with asyncio.timeout(max(0, deadline - monotonic())):
            async with self._client() as client:
                states = await self._states(client)
                return frozenset(model for model, state in states.items() if state != "unloaded")

    async def _release(self, model_id: str, deadline: float) -> None:
        async with asyncio.timeout(max(0, deadline - monotonic())):
            async with self._client() as client:
                # A failed initial GET cannot have dispatched an unload.
                state = await self._status(client, model_id)
                if state == "unloaded":
                    return
                from ...service.gpu_handoff import current_owner
                owner = current_owner()
                call = self._unload(client, model_id, deadline)
                # Mutation and its affirmative confirmation share one boundary.
                if owner is not None:
                    await owner.model_call(call)
                else:
                    await call

    async def _unload(self, client: httpx.AsyncClient, model_id: str, deadline: float) -> None:
        response = await client.post("models/unload", json={"model": model_id})
        if response.status_code == 400 and await self._status(client, model_id) == "unloaded":
            return
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("success") is not True:
            raise RuntimeError("model_release_unconfirmed: router rejected unload")
        while await self._status(client, model_id) != "unloaded":
            await asyncio.sleep(min(0.1, max(0, deadline - monotonic())))

    @staticmethod
    async def _status(client: httpx.AsyncClient, model_id: str) -> str:
        states = await LlamaCppLifecycle._states(client)
        if model_id not in states:
            raise RuntimeError("model_release_unconfirmed: canonical model ID is unavailable")
        return states[model_id]

    @staticmethod
    async def _states(client: httpx.AsyncClient) -> dict[str, str]:
        response = await client.get("models")
        response.raise_for_status()
        body = response.json()
        entries = body.get("data") if isinstance(body, dict) else None
        if not isinstance(entries, list):
            # A malformed remote protocol response is a runtime failure.
            raise RuntimeError("model_release_unconfirmed: router status is unavailable")  # noqa: TRY004
        states: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError("model_release_unconfirmed: malformed router entry")  # noqa: TRY004
            model = entry.get("id")
            status = entry.get("status")
            value = status.get("value") if isinstance(status, dict) else None
            if (not isinstance(model, str) or not model.strip() or model != model.strip() or model in states
                    or value not in ("unloaded", "loaded", "loading", "sleeping")):
                raise RuntimeError("model_release_unconfirmed: unknown router status")
            states[model] = value
        return states
