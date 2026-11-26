"""Test-owned OpenAI wire service with explicit source-keyed complete/partial replies.

Aliases are registered before requests. Direct images use their actual normalized
bytes; OCR text uses the exact typed source packet. Unknown sources fail closed.
No scenario controls are sent to the application or inferred by production code.
Prepared-app checks may set VB_E2E_LLM_MODEL_ID and VB_E2E_LLM_BEARER_TOKEN.
"""

import base64
import hashlib
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NotRequired, TypedDict

FIXTURES_DIR = Path(os.environ.get("VB_E2E_FIXTURES_DIR", "/fixtures"))
REPLIES_DIR = Path(os.environ.get("VB_E2E_REPLIES_DIR", Path(__file__).parent / "replies"))
ALIASES = json.loads(
    Path(os.environ.get("VB_E2E_ALIASES", Path(__file__).parent / "aliases.json")).read_text()
)
MODES = ("ok", "down", "repair_first", "broken", "unreadable", "slow")
EXACT_MODEL_ID = os.environ.get("VB_E2E_LLM_MODEL_ID")
BEARER_TOKEN = os.environ.get("VB_E2E_LLM_BEARER_TOKEN")
MODELS = [
    {
        "id": "e2e-vision",
        "object": "model",
        "owned_by": "e2e",
        "architecture": {"input_modalities": ["image", "text"], "output_modalities": ["text"]},
    },
    {
        "id": "e2e-text",
        "object": "model",
        "owned_by": "e2e",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    },
]
if EXACT_MODEL_ID:
    MODELS = [{**MODELS[0], "id": EXACT_MODEL_ID}]
else:
    MODELS.append({**MODELS[0], "id": "e2e-alternate-vision"})

# `broken` tracks which image digests already got their one broken response in
# the current repair_first mode (cleared on every mode change) so the "first
# call is broken" rule is per-mode, not polluted by earlier tests' requests.
class LlmState(TypedDict):
    mode: str
    broken: set[tuple[str, tuple[str, ...]]]
    scenario: NotRequired[str]


state: LlmState = {"mode": "ok", "broken": set()}
requests_log: list[dict[str, Any]] = []


def _extract_image_hash(body: dict[str, Any]) -> str | None:
    for message in body.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    b64 = url.split(",", 1)[1]
                    return hashlib.sha256(base64.b64decode(b64)).hexdigest()
    return None


def configure_scenario(body: dict[str, Any]) -> None:
    scenario = body.get("scenario", "complete")
    if scenario not in ("complete", "partial", "policy_unknown", "overlap", "direct_merchant_mismatch"):
        raise ValueError("unknown scenario")
    state["scenario"] = scenario
    state["broken"] = set()
    requests_log.clear()


def _request_identity(body: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    texts, hashes = [], []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    texts.append(part["text"])
                if part.get("type") == "image_url":
                    hashes.append(
                        hashlib.sha256(
                            base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                        ).hexdigest()
                    )
    prompt = texts[0] if texts else ""
    if prompt.startswith("Classify these ordered receipt rows"):
        return "classification", [], [], [json.loads(prompt.split("; receipt: ", 1)[1])]
    refs, pages = [], []
    if "Source refs: " in prompt:
        refs = json.loads(prompt.split("Source refs: ", 1)[1].split("\nNullable schema:", 1)[0])
        observation = prompt.split("\nObservations:\n", 1)[1]
        if observation.startswith("PHOTO "):
            pages = [json.loads(part.split("\n", 1)[1]) for part in observation.split("PHOTO ")[1:]]
            refs = [ref for ref in refs if ref["coordinate_space"] == "working"]
    names = []
    aliases: dict[str, str] = {}
    for name, identity in ALIASES.items():
        for digest in identity.values():
            if digest in aliases and aliases[digest] != name:
                raise ValueError("ambiguous image alias")
            aliases[digest] = name
    identities = hashes if hashes else [source["source_sha256"] for source in (refs or pages)]
    for index, digest in enumerate(identities):
        if digest not in aliases:
            raise ValueError("unregistered input source")
        name = aliases[digest]
        if refs and (
            refs[index]["photo_index"] != index
            or refs[index]["source_sha256"] != ALIASES[name]["source_sha256"]
        ):
            raise ValueError("ordered source mismatch")
        names.append(name)
    if not names:
        raise ValueError("unregistered input source")
    if pages and [p["photo_index"] for p in pages] != list(range(len(pages))):
        raise ValueError("OCR photo order mismatch")
    if (
        pages
        and refs
        and [(p["source_sha256"], p["photo_index"]) for p in pages]
        != [(r["source_sha256"], r["photo_index"]) for r in refs]
    ):
        raise ValueError("OCR source mismatch")
    return "ocr" if pages else "direct", names, refs, pages


def _content_for(body: dict[str, Any]) -> str:
    kind, names, refs, pages = _request_identity(body)
    scenario = state.get("scenario", "complete")
    entry = {
        "kind": kind,
        "model": body.get("model"),
        "scenario": scenario,
        "fixture": names,
        "ordered_sources": [
            [r["source_sha256"], r["photo_index"]]
            for r in (refs or (pages if kind == "ocr" else []))
        ],
        "image_sha256": _extract_image_hash(body),
        "served": "fixture",
    }
    requests_log.append(entry)
    if state["mode"] == "unreadable":
        return json.dumps({"error": {"code": "unreadable", "message": "Photo is too blurry"}})
    if state["mode"] == "slow":
        time.sleep(3)
    key = (kind, tuple(names))
    if state["mode"] == "broken" or (
        state["mode"] == "repair_first" and key not in state["broken"]
    ):
        state["broken"].add(key)
        entry["served"] = "broken"
        return '{"merchant_name": "broken", oops'
    if kind == "classification":
        receipt = pages[0]
        return json.dumps(
            {
                "category": receipt.get("category", "grocery"),
                "line_items": [
                    {
                        "tags": ["food"] if row["description"] in ("Milk", "Bread") else [],
                        "spending_category": "unknown",
                    }
                    for row in receipt["line_items"]
                ],
            }
        )
    reply = "partial" if scenario == "partial" else "complete"
    payload = json.loads((REPLIES_DIR / f"{names[0]}.{reply}.json").read_text())
    if scenario == "direct_merchant_mismatch" and kind == "direct":
        payload["merchant_name"] = "Misread test merchant"
    if scenario == "policy_unknown":
        payload.pop("amount_policy")
    if scenario == "overlap":
        if len(names) != 2:
            raise ValueError("overlap scenario requires exactly two photos")
        payload = json.loads((REPLIES_DIR / "overlap.complete.json").read_text())
    elif len(names) > 1:
        items = payload["line_items"]
        payload["line_items"] = [
            {**item, "source_image": source} for source in range(len(names)) for item in items
        ]
        size = len(items)
        payload["duplicate_candidates"] = [
            {"first_index": source * size + row, "second_index": (source + 1) * size + row}
            for source in range(len(names) - 1)
            for row in range(size)
        ]
    sources = {}
    if refs:
        sources["merchant_name"] = [dict(refs[0])]
        for index, row in enumerate(payload["line_items"]):
            source = row.get("source_image", 0)
            ref = dict(refs[source])
            if pages:
                block = pages[source]["blocks"][
                    1
                    + index
                    % (
                        2
                        if scenario == "overlap"
                        else len(
                            json.loads((REPLIES_DIR / f"{names[0]}.{reply}.json").read_text())[
                                "line_items"
                            ]
                        )
                    )
                ]
                ref.update(block_id=block["id"], bbox=block["bbox"])
            sources[f"line_items.{index}.description"] = [ref]
        if pages:
            block = pages[0]["blocks"][0]
            sources["merchant_name"][0].update(block_id=block["id"], bbox=block["bbox"])
    payload["field_sources"] = sources
    return json.dumps(payload)


class Handler(BaseHTTPRequestHandler):
    def _authorized(self) -> bool:
        if not BEARER_TOKEN or self.headers.get_all("Authorization") == [f"Bearer {BEARER_TOKEN}"]:
            return True
        self._send(401, {"error": "unauthorized"})
        return False

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _raw(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_fixture(self, name: str) -> None:
        """Serve a committed fixture: /__fixtures/<n> -> png, /__fixtures/<n>.json -> truth."""
        if not re.fullmatch(r"[a-z0-9]+(\.json)?", name):
            self._send(400, {"error": "bad fixture name"})
            return
        is_json = name.endswith(".json")
        base = name.removesuffix(".json")
        image = FIXTURES_DIR / f"{base}.png"
        truth = FIXTURES_DIR / f"{base}.json"
        if is_json and truth.is_file():
            self._raw(200, "application/json", truth.read_bytes())
        elif not is_json and image.is_file():
            self._raw(200, "image/png", image.read_bytes())
        else:
            self._send(404, {"error": f"no fixture {base!r}"})

    def do_GET(self) -> None:
        if state["mode"] == "down":
            self._send(503, {"error": "stub is in down mode"})
            return
        if self.path == "/v1/models":
            if self._authorized():
                self._send(200, {"object": "list", "data": MODELS})
        elif self.path == "/__requests":
            self._send(200, {"requests": requests_log})
        elif self.path.startswith("/__fixtures/"):
            self._send_fixture(self.path[len("/__fixtures/") :])
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/__scenario":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            try:
                configure_scenario(body)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            else:
                self._send(200, {"scenario": state["scenario"]})
            return
        if self.path == "/__mode":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            mode = body.get("mode", "ok")
            if mode not in MODES:
                self._send(400, {"error": f"unknown mode {mode!r}"})
                return
            state["mode"] = mode
            state["broken"] = set()  # reset per-mode broken-once tracking
            self._send(200, {"mode": mode})
            return
        if state["mode"] == "down":
            self._send(503, {"error": "stub is in down mode"})
            return
        if self.path == "/v1/chat/completions":
            if not self._authorized():
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            if EXACT_MODEL_ID and body.get("model") != EXACT_MODEL_ID:
                self._send(422, {"error": "unexpected model"})
                return
            try:
                content = _content_for(body)
            except (ValueError, KeyError, IndexError) as exc:
                self._send(422, {"error": str(exc)})
                return
            self._send(
                200,
                {
                    "id": "e2e-stub",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": EXACT_MODEL_ID or "e2e-vision",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"llm-stub: {self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 9123), Handler).serve_forever()
