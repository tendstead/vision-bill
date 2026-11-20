"""Exact v2 CPU wire fixture. No Paddle, model weights or production fault hooks."""

import base64
import hashlib
import json
import os
import re
import struct
import time
import zlib
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from e2e.owned_files import remove_test_file
    from e2e.stub_control import Scenario
else:
    from owned_files import remove_test_file
    from stub_control import Scenario

PROFILE = json.loads((Path(__file__).parent / "profile.json").read_text())
PROFILE_ID = hashlib.sha256(
    json.dumps(PROFILE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
RUNTIME = hashlib.sha256(b"cpu-test-wire-fixture-v2").hexdigest()
state = Scenario()


def png(width: int = 320, height: int = 480) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\xff\xff" * width) * height))
        + chunk(b"IEND", b"")
    )


def page(raw: bytes, request_id: str, photo_index: int, source_sha256: str, *, blank: bool) -> dict[str, Any]:
    UUID(request_id)
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise ValueError("source hash mismatch")
    if photo_index < 0:
        raise ValueError("invalid photo index")
    working = png()
    texts = (
        ["Printed merchant", "A 1.00", "B 2.00"]
        if photo_index == 0
        else ["Printed merchant", "B 2.00", "C 3.00"]
    )
    blocks: list[dict[str, Any]] = (
        []
        if blank
        else [
            {
                "id": f"photo-{photo_index}-{'merchant' if i == 0 else i}",
                "label": "text",
                "text": text,
                "bbox": [
                    [10, 20 + i * 60],
                    [210, 20 + i * 60],
                    [210, 50 + i * 60],
                    [10, 50 + i * 60],
                ],
            }
            for i, text in enumerate(texts)
        ]
    )
    return {
        "contract_version": 2,
        "request_id": request_id,
        "photo_index": photo_index,
        "source_sha256": source_sha256,
        "text": "\n".join(b["text"] for b in blocks),
        "blocks": blocks,
        "profile_id": PROFILE_ID,
        "runtime_fingerprint": RUNTIME,
        "working_image": {
            "view_id": str(uuid4()),
            "media_type": "image/png",
            "width": 320,
            "height": 480,
            "sha256": hashlib.sha256(working).hexdigest(),
            "png_base64": base64.b64encode(working).decode(),
            "transforms": [{"operation": "test_fixture_unwarp"}],
            "inverse_mapping": None,
        },
        "telemetry": {"test_cpu_fixture": True},
    }


def readiness() -> dict[str, Any]:
    return {
        "ready": True,
        "generation": 1,
        "profile_id": PROFILE_ID,
        "runtime_fingerprint": RUNTIME,
        "reason_code": None,
    }


class Handler(BaseHTTPRequestHandler):
    def send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        if self.path == "/ready":
            self.send(200, readiness())
        elif self.path == "/info":
            self.send(
                200,
                {
                    "contract_version": 2,
                    "profile": PROFILE,
                    "worker": readiness(),
                    "max_upload_bytes": 20971520,
                    "max_pending": 4,
                    "max_timeout_ms": 600000,
                },
            )
        elif self.path == "/__requests":
            self.send(200, {"requests": state.audit()})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if self.path == "/__files":
                body = json.loads(raw)
                root = os.environ.get("VB_E2E_OWNED_UPLOADS")
                if not root:
                    raise ValueError("owned uploads volume not configured")
                remove_test_file(
                    Path(root), body["image_id"], body["kind"], body.get("photo_index", 0)
                )
                self.send(200, {"removed": True})
                return
            if self.path == "/__scenario":
                state.configure(json.loads(raw))
                self.send(200, {"ok": True})
                return
            if self.path == "/__release":
                state.release(json.loads(raw)["barrier"])
                self.send(200, {"ok": True})
                return
            if self.path != "/ocr":
                self.send(404, {"error": "not found"})
                return
            message = BytesParser(policy=default).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw
            )
            fields: dict[str, bytes] = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                payload = part.get_payload(decode=True)
                if not isinstance(name, str) or not isinstance(payload, bytes) or name in fields:
                    raise ValueError("invalid multipart field")
                fields[name] = payload
            digest = fields["source_sha256"].decode()
            request_id = str(UUID(fields["request_id"].decode()))
            index = int(fields["photo_index"])
            timeout = int(fields["timeout_ms"])
            if not re.fullmatch("[0-9a-f]{64}", digest) or not 0 < timeout <= 600000 or index < 0:
                raise ValueError("invalid request")
            if hashlib.sha256(fields["image"]).hexdigest() != digest:
                raise ValueError("source hash mismatch")
            plan = state.plan(digest, index)
            entry = {
                "request_id": request_id,
                "source_sha256": digest,
                "photo_index": index,
                "result": plan["result"],
                "barrier": plan.get("barrier"),
            }
            state.record(entry)
            if plan.get("barrier") and not state.wait(plan["barrier"], timeout / 1000):
                self.send(503, {"error": "barrier timeout"})
                return
            if plan["result"] == "timeout":
                time.sleep(timeout / 1000 + 0.1)
                self.send(503, {"error": "request timeout"})
            elif plan["result"] == "503":
                self.send(503, {"error": "planned source unavailable"})
            else:
                self.send(
                    200,
                    page(
                        fields["image"], request_id, index, digest, blank=plan["result"] == "empty"
                    ),
                )
        except (ValueError, KeyError, TypeError) as exc:
            self.send(422, {"error": str(exc)})

    def log_message(self, *_args: object) -> None:
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 9124), Handler).serve_forever()
