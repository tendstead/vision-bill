"""The built SPA is served by the app, incl. the client-route fallback."""

import httpx
import pytest

pytestmark = pytest.mark.e2e


def test_index_serves_html(base_url: str) -> None:
    response = httpx.get(f"{base_url}/", timeout=10.0)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "</html>" in response.text


def test_client_route_falls_back_to_index(base_url: str) -> None:
    response = httpx.get(f"{base_url}/statistics", timeout=10.0)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "</html>" in response.text
