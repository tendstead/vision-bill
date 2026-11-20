"""HTTP-only client for source bytes; no application or native runtime imports."""

import argparse
import hashlib
import math
import mimetypes
import sys
from contextlib import nullcontext
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from ocr_contract import OcrResponse, limits
from ocr_contract.document_image import decode_working_png, validate_working_png


def recognize(
    source: bytes,
    *,
    media_type: str,
    url: str,
    timeout_seconds: float = 600,
    request_id: UUID | None = None,
    photo_index: int = 0,
    client: httpx.Client | None = None,
) -> OcrResponse:
    """Submit the exact original once and validate the returned request identity."""
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600:
        raise ValueError("OCR timeout must be finite, positive and at most 600 seconds")
    timeout_ms = int(timeout_seconds * 1000)
    if timeout_ms <= 0:
        raise ValueError("OCR timeout must be at least one millisecond")
    if photo_index < 0:
        raise ValueError("OCR photo index must be nonnegative")
    if len(source) > limits.NATIVE_SOURCE_MAX_BYTES:
        raise ValueError("ocr_source_too_large")
    identity = request_id or uuid4()
    digest = hashlib.sha256(source).hexdigest()
    # A small transport margin permits the service to confirm native cleanup.
    timeout = timeout_seconds + 10
    transport = nullcontext(client) if client is not None else httpx.Client(timeout=timeout)
    with transport as http, http.stream(
        "POST",
        url.rstrip("/") + "/ocr",
        headers={"Accept-Encoding": "identity"},
        files={"image": ("original", source, media_type)},
        data={
            "request_id": str(identity),
            "photo_index": str(photo_index),
            "source_sha256": digest,
            "timeout_ms": str(timeout_ms),
        },
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        limits.check_ocr_response_headers(
            response.headers.get("Content-Length"), response.headers.get("Content-Encoding", "")
        )
        body = bytearray()
        for chunk in response.iter_bytes():
            limits.append_ocr_response_chunk(body, chunk)
    result = OcrResponse.parse_wire(body)
    if (
        result.request_id != identity
        or result.photo_index != photo_index
        or result.source_sha256 != digest
    ):
        raise ValueError("OCR response identity differs from the submitted source")
    view = result.working_image
    validate_working_png(decode_working_png(view.png_base64), view)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize an original image using standalone OCR")
    parser.add_argument("source", type=Path)
    parser.add_argument("--url", default="http://localhost:8082")
    parser.add_argument("--media-type")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    media_type = args.media_type or mimetypes.guess_type(args.source.name)[0]
    if media_type is None:
        parser.error("cannot determine source media type; pass --media-type")
    try:
        with args.source.open("rb") as source:
            raw = source.read(limits.NATIVE_SOURCE_MAX_BYTES + 1)
        result = recognize(
            raw,
            media_type=media_type,
            url=args.url,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, ValueError, httpx.HTTPError):
        print("OCR request failed", file=sys.stderr)
        return 1
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
