"""The independent source-bytes client is usable without the application."""

import hashlib
import importlib
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from uuid import uuid4

import httpx
import pytest
from test_ocr_worker_lifecycle import response


def module():
    return importlib.import_module("ocr_service.client")


def test_http_client_sends_original_bytes_and_identity_without_lifecycle(tmp_path):
    raw = b"original source bytes"
    request_id = uuid4()
    seen = []

    def endpoint(http_request):
        seen.append(http_request)
        body = http_request.read()
        assert raw in body
        assert str(request_id).encode() in body
        assert hashlib.sha256(raw).hexdigest().encode() in body
        assert b"owner_token" not in body
        result = response(
            type(
                "Request",
                (),
                {
                    "request_id": request_id,
                    "photo_index": 2,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                },
            )()
        )
        return httpx.Response(200, json=result.model_dump(mode="json"))

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as transport:
        result = module().recognize(
            raw,
            media_type="image/jpeg",
            url="http://ocr",
            request_id=request_id,
            photo_index=2,
            client=transport,
        )
    assert result.request_id == request_id
    assert len(seen) == 1
    assert seen[0].url.path == "/ocr"


def test_http_client_rejects_response_identity_mismatch():
    raw = b"source"
    result = response(
        type(
            "Request",
            (),
            {
                "request_id": uuid4(),
                "photo_index": 0,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            },
        )()
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=result.model_dump(mode="json"))
            )
        ) as transport,
        pytest.raises(ValueError, match="identity"),
    ):
        module().recognize(raw, media_type="image/png", url="http://ocr", client=transport)


def test_http_client_does_not_return_http_failure_as_ocr_success():
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as transport,
        pytest.raises(httpx.HTTPStatusError),
    ):
        module().recognize(b"source", media_type="image/png", url="http://ocr", client=transport)


@pytest.mark.parametrize("timeout", [0, -1, 601, float("nan")])
def test_http_client_rejects_invalid_deadlines_before_sending(timeout):
    with pytest.raises(ValueError, match="timeout"):
        module().recognize(
            b"source", media_type="image/png", url="http://ocr", timeout_seconds=timeout
        )


def test_cli_runs_in_isolated_python_without_app_imports(tmp_path):
    original = tmp_path / "receipt.png"
    original.write_bytes(b"private original")

    # Real HTTP makes the isolated subprocess exercise its normal client path.
    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            from email.parser import BytesParser
            from email.policy import default

            body = self.rfile.read(int(self.headers["Content-Length"]))
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body
            )
            parts = {
                p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                for p in message.iter_parts()
            }
            assert self.path == "/ocr"
            assert parts["image"] == original.read_bytes()
            result = response(
                type(
                    "Request",
                    (),
                    {
                        "request_id": parts["request_id"].decode(),
                        "photo_index": int(parts["photo_index"]),
                        "source_sha256": parts["source_sha256"].decode(),
                    },
                )()
            )
            payload = result.model_dump_json().encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    service = Path(__file__).resolve().parents[1] / "paddleocr-service"
    contract = Path(importlib.import_module("ocr_contract").__file__).resolve().parent.parent
    script = """
import importlib.abc, runpy, sys
class RejectApplication(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vision_bill" or fullname.startswith("vision_bill."):
            raise AssertionError("application import forbidden")
sys.meta_path.insert(0, RejectApplication())
sys.path[:0] = [sys.argv.pop(1), sys.argv.pop(1)]
sys.argv[0] = "ocr_service.client"
runpy.run_module("ocr_service.client", run_name="__main__")
"""
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    script,
                    str(service),
                    str(contract),
                    "--url",
                    f"http://127.0.0.1:{server.server_port}",
                    str(original),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        finally:
            server.shutdown()
    assert (
        json.loads(completed.stdout)["source_sha256"]
        == hashlib.sha256(original.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("declared_length", [True, False])
def test_standalone_ocr_client_stops_at_oversized_body(monkeypatch, declared_length):
    from ocr_contract import limits

    class OversizedBody(httpx.SyncByteStream):
        def __iter__(self):
            assert not declared_length, "oversized Content-Length must be rejected before reading"
            yield b"01234567"
            yield b"8"
            pytest.fail("client read another chunk after crossing the response cap")

    monkeypatch.setattr(limits, "MAX_OCR_RESPONSE_BYTES", 8)
    headers = {"Content-Length": "9"} if declared_length else {}
    with (
        httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers=headers, stream=OversizedBody())
        )) as transport,
        pytest.raises(ValueError, match="ocr_response_too_large"),
    ):
        module().recognize(b"source", media_type="image/png", url="http://ocr", client=transport)


def test_standalone_client_rejects_crc_correct_invalid_deflate(tmp_path):
    from test_ocr_worker_lifecycle import request, with_invalid_deflate

    raw = b"source"
    item = request(tmp_path)
    result = with_invalid_deflate(
        response(type("Request", (), {
            "request_id": item.request_id, "photo_index": 0,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        })())
    )
    with (
        httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=result.model_dump_json())
        )) as transport,
        pytest.raises(ValueError, match="invalid_working_artifact_png"),
    ):
        module().recognize(
            raw, media_type="image/png", url="http://ocr", request_id=item.request_id,
            client=transport,
        )
