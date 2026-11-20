"""Fixtures for the e2e suite (requires `make e2e-up`)."""

import os
from collections.abc import Callable, Generator
from typing import Any

import httpx
import pytest

from e2e.helpers import Stub, register_user


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ["VB_E2E_BASE_URL"]


@pytest.fixture
def http_factory(base_url: str) -> Generator[Callable[[], httpx.Client], None, None]:
    """Yields a factory creating a fresh httpx.Client (own cookie jar)."""
    clients: list[httpx.Client] = []

    def factory() -> httpx.Client:
        client = httpx.Client(base_url=base_url, timeout=30.0)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.close()


@pytest.fixture
def registered_user(http_factory) -> tuple[httpx.Client, dict[str, Any]]:
    return register_user(http_factory, "user")


@pytest.fixture
def stub() -> Generator[Stub, None, None]:
    s = Stub()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def reset_stub_mode(stub: Stub) -> None:
    """Every test starts with the stub in 'ok' mode (the control API is global).

    Tolerates a missing stack: the discovery tests in this directory are pure
    filesystem logic and run via `make test` without the e2e stack — there is
    nothing to reset in that case. Tests that truly need the stub will fail
    explicitly on their own `set_mode`/`get_requests` calls.
    """
    try:
        stub.set_mode("ok")
        stub.set_scenario("complete")
    except httpx.ConnectError:
        pass
