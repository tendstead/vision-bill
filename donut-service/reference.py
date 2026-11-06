"""Inspect Donut and optionally Paddle outputs for the same local images.

Standard-library client; sends images only to the explicitly supplied endpoints.
Does not map CORD fields into Receipt or report a misleading accuracy score.
"""

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def build_multipart(filename, data, boundary=None):
    boundary = boundary or uuid.uuid4().hex
    filename = Path(filename).name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    body = header + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def read_response(response, limit=MAX_RESPONSE_BYTES):
    data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Response exceeds {limit} bytes; remove raw images/pixels server-side")
    return json.loads(data)


def query(url, filename, data, timeout):
    body, content_type = build_multipart(filename, data)
    request = urllib.request.Request(
        url.rstrip("/") + "/ocr", data=body,
        headers={"Content-Type": content_type}, method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = read_response(response)
        return {"elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "response": result}
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}: {detail}"}
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--donut-url", default="http://donut-service:8083")
    parser.add_argument("--paddle-url", help="Optional existing wrapper URL (must expose /ocr)")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    records = []
    failed = False
    for image in args.images:
        record = {"image": image.name}
        try:
            data = image.read_bytes()
        except OSError as exc:
            record["error"] = str(exc)
            records.append(record)
            failed = True
            continue
        endpoints = {"donut": args.donut_url}
        if args.paddle_url:
            endpoints["paddle"] = args.paddle_url
        for name, url in endpoints.items():
            print(f"Reading {image.name} with {name}...", file=sys.stderr, flush=True)
            record[name] = query(url, image.name, data, args.timeout)
            failed |= "error" in record[name]
        records.append(record)
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
