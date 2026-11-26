"""The saved model-list probe and the running SDK must reach the same route."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from vision_bill.config import LlmAuthentication, LLMProviderEnum, llm_origin
from vision_bill.provider.llm.connection import check_saved_connection
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [
    "", "OpenAI-Project: custom-project", "openAI-ProJect: custom-project\nopenai-organization: custom-org",
])
@pytest.mark.parametrize("auth", ["inherit", "none", "bearer"])
async def test_probe_routing_matches_running_openai(settings, monkeypatch, custom, auth):
    monkeypatch.setenv("OPENAI_ORG_ID", "deployment-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "deployment-project")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", custom)
    calls = []

    class Stub(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get_all("OpenAI-Organization"), self.headers.get_all("OpenAI-Project")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[]}')

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Stub) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            llm = settings.llm
            llm.provider = LLMProviderEnum.OPENAI
            llm.host = f"http://127.0.0.1:{server.server_port}/prefix/v1"
            if auth == "none":
                llm.authentication = LlmAuthentication(mode="none")
            elif auth == "bearer":
                llm.authentication = LlmAuthentication.model_validate({
                    "mode": "bearer", "api_key": "selected-key", "origin": llm_origin(llm.host),
                })
            provider = OpenAIProvider(
                host=llm.host, api_key=llm.request_api_key(), explicit_authentication=auth != "inherit",
            )
            try:
                await provider.get_available_models(capability="text")
            finally:
                await provider._client.close()
            assert (await check_saved_connection(llm)).status == "connected"
        finally:
            server.shutdown()
            thread.join()
    assert len(calls) == 2
    assert calls[0][0] == "/prefix/v1/models"
    assert calls[1] == calls[0]
    if auth != "inherit":
        assert calls[1][1:] == (None, None)


@pytest.mark.asyncio
async def test_bare_ollama_host_and_proxy_path_reach_same_catalog(settings):
    paths = []

    class Stub(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"models":[]}')

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Stub) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            llm = settings.llm
            llm.host = f"127.0.0.1:{server.server_port}/prefix/"
            provider = OllamaProvider(host=llm.host)
            try:
                await provider.get_available_models(capability="text")
            finally:
                await provider._client._client.aclose()
            assert (await check_saved_connection(llm)).status == "connected"
        finally:
            server.shutdown()
            thread.join()
    assert paths == ["/prefix/api/tags", "/prefix/api/tags"]
