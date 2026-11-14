"""Engine adapters. Heavy imports live here, in separate CPU containers."""

import argparse
import csv
import importlib.metadata
import io
import ipaddress
import json
import mimetypes
import socket
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def plain(value):
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def block(text, points, confidence=None):
    return {
        "text": text,
        "bbox": plain(points),
        "confidence": float(confidence) if confidence is not None else None,
    }


def tesseract(options, image):
    variant = options.get("variant", "fast")
    directory = Path("/models/tesseract") / variant
    directory.mkdir(parents=True, exist_ok=True)
    if image is None:
        for lang in ("deu", "eng"):
            target = directory / f"{lang}.traineddata"
            if not target.exists():
                url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_{variant}/main/{lang}.traineddata"
                partial = target.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=120) as response:
                    partial.write_bytes(response.read())
                partial.replace(target)
        return None
    data = subprocess.check_output(
        [
            "tesseract",
            str(image),
            "stdout",
            "--tessdata-dir",
            str(directory),
            "-l",
            "deu+eng",
            "--psm",
            str(options.get("psm", 4)),
            "-c",
            "tessedit_create_tsv=1",
        ],
        text=True,
    )
    blocks = []
    for row in csv.DictReader(io.StringIO(data), delimiter="\t"):
        if row["text"].strip() and float(row["conf"]) >= 0:
            x, y, w, h = [int(row[k]) for k in ("left", "top", "width", "height")]
            blocks.append(block(row["text"], [[x, y], [x + w, y + h]], float(row["conf"]) / 100))
    return {
        "text": "\n".join(b["text"] for b in blocks),
        "blocks": blocks,
        "versions": {
            "tesseract": subprocess.check_output(
                ["tesseract", "--version"], text=True
            ).splitlines()[0]
        },
    }


def paddle(options, image):
    from paddleocr import PaddleOCR

    model = PaddleOCR(
        device="cpu",
        cpu_threads=4,
        # The pinned Paddle 3.3.1 oneDNN/PIR path fails on these static OCR
        # models. Native CPU inference works without the oneDNN optimization.
        enable_mkldnn=options.get("mkldnn", False),
        text_detection_model_name=options["det"],
        text_recognition_model_name=options["rec"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if image is None:
        return None
    result = model.predict(str(image))[0]
    blocks = [
        block(text, points, confidence)
        for text, points, confidence in zip(
            result["rec_texts"], result["rec_polys"], result["rec_scores"], strict=True
        )
    ]
    return {"text": "\n".join(b["text"] for b in blocks), "blocks": blocks}


def easyocr(options, image):
    import easyocr as library

    model = library.Reader(
        ["de", "en"],
        gpu=False,
        model_storage_directory="/models/easyocr",
        download_enabled=image is None,
    )
    if image is None:
        return None
    blocks = [
        block(text, points, score)
        for points, text, score in model.readtext(str(image), detail=1, paragraph=False)
    ]
    return {"text": "\n".join(b["text"] for b in blocks), "blocks": blocks}


def doctr(options, image):
    import torch
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    torch.set_num_threads(4)
    model = ocr_predictor(
        det_arch=options.get("det", "fast_base"),
        reco_arch=options.get("rec", "crnn_vgg16_bn"),
        pretrained=True,
    )
    if image is None:
        return None
    from PIL import Image

    width, height = Image.open(image).size
    result = model(DocumentFile.from_images(str(image))).export()
    blocks = []
    for page in result["pages"]:
        for group in page["blocks"]:
            for line in group["lines"]:
                for word in line["words"]:
                    points = [[float(x) * width, float(y) * height] for x, y in word["geometry"]]
                    blocks.append(block(word["value"], points, word["confidence"]))
    vocab = model.reco_predictor.model.cfg.get("vocab", "")
    missing = "".join(c for c in "äöüßÄÖÜ€" if c not in vocab)
    warnings = [f"Checkpoint vocabulary lacks German characters: {missing}"] if missing else []
    return {"text": "\n".join(b["text"] for b in blocks), "blocks": blocks, "warnings": warnings}


def require_local_endpoint(endpoint):
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
        raise ValueError("Expected a local HTTP OCR endpoint without embedded credentials")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 80, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_private for a in addresses):
        raise ValueError("OCR HTTP endpoint must resolve only to local/private addresses")


def http(options, image):
    endpoint = options["url"].rstrip("/")
    require_local_endpoint(endpoint)
    # Disable environment proxies: receipt bytes must go straight to the selected local endpoint.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if image is None:
        client.open(endpoint + "/health", timeout=10).close()
        return None
    boundary = "vision-bill-ocr-harness-boundary"
    content_type = mimetypes.guess_type(str(image))[0] or "application/octet-stream"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="receipt"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        + image.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        endpoint + "/ocr",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with client.open(request, timeout=options.get("timeout", 900)) as response:
        raw = response.read(8_000_001)
        if len(raw) > 8_000_000:
            raise ValueError("HTTP OCR response exceeds compact-result limit")
    return json.loads(raw)


ADAPTERS = {
    "paddle": paddle,
    "easyocr": easyocr,
    "tesseract": tesseract,
    "doctr": doctr,
    "http": http,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=ADAPTERS)
    parser.add_argument("--options", default="{}")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = ADAPTERS[args.family](json.loads(args.options), args.image)
    if result is not None:
        versions = {}
        for package in (
            "paddleocr",
            "paddlepaddle",
            "paddlex",
            "easyocr",
            "python-doctr",
            "torch",
            "torchvision",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        result.setdefault("versions", {}).update(versions)
        args.result.write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
