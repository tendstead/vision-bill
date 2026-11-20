"""Offline checks for maintainer model archive preparation."""

import hashlib
import io
import json
from http.client import HTTPResponse
from pathlib import Path
from runpy import run_path

import pytest
from ocr_service import paddle_snapshots
from ocr_service.paddle_snapshots import download as official_download
from ocr_service.provision import MODEL_NAMES, Archive, extract_archive, publish_directory

SCRIPT = Path(__file__).parents[1] / "paddleocr-service" / "build" / "prepare_paddle_models.py"
BASE = "https://artifacts.example.invalid/ocr"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def preparer(monkeypatch):
    prepare = run_path(str(SCRIPT))["prepare"]
    assets = {}
    models = {}
    for name in MODEL_NAMES:
        weights = name.encode()
        files = ("README.md", "weights")
        companion = name != "PaddleOCR-VL-1.6-0.9B"
        if not companion:
            files += ("LICENSE",)
        spec = paddle_snapshots.Model(name, "a" * 40, files, "weights", digest(weights))
        models[name] = spec
        assets[spec.resolve_url("weights")] = weights
        assets[spec.resolve_url("README.md")] = b"---\nlicense: apache-2.0\n---\n"
        assets[spec.resolve_url("LICENSE")] = paddle_snapshots.APACHE_LICENSE_HEADER
    assets[paddle_snapshots.COMPANION_LICENSE_URL] = paddle_snapshots.APACHE_LICENSE_HEADER
    monkeypatch.setattr(paddle_snapshots, "MODELS", models)
    monkeypatch.setattr(
        paddle_snapshots, "COMPANION_LICENSE_SHA256", digest(paddle_snapshots.APACHE_LICENSE_HEADER)
    )

    def fetch(url, destination, limit):
        destination.write_bytes(assets[url])

    monkeypatch.setattr(paddle_snapshots, "download", fetch)
    return prepare, assets


def test_preparer_writes_reproducible_provisionable_model_archives(tmp_path, preparer):
    prepare, _ = preparer
    first, second = tmp_path / "first", tmp_path / "second"
    prepare(first, BASE)
    prepare(second, BASE + "/")
    assert (first / "models.json").read_bytes() == (second / "models.json").read_bytes()
    mapping = json.loads((first / "models.json").read_bytes())
    assert set(mapping) == set(MODEL_NAMES)
    for name, raw_record in mapping.items():
        record = Archive.model_validate(raw_record)
        assert record.source == f"{BASE}/{name}.tar"
        assert record.provenance.revision == "a" * 40
        assert record.license.spdx == "Apache-2.0"
        packed = first / f"{name}.tar"
        assert digest(packed.read_bytes()) == record.sha256
        assert packed.read_bytes() == (second / packed.name).read_bytes()
        tree = tmp_path / name
        tree.mkdir()
        extract_archive(packed, tree, record)
        assert (tree / "weights").read_bytes() == name.encode()
        assert (tree / record.license.text_file).is_file()
        assert ("LICENSE-PROVENANCE.txt" in record.files) == (name != "PaddleOCR-VL-1.6-0.9B")


def test_mismatched_weight_leaves_no_output(tmp_path, preparer):
    prepare, assets = preparer
    weights = next(url for url in assets if url.endswith("/weights"))
    assets[weights] = b"corruption"
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Weight SHA-256 mismatch"):
        prepare(output, BASE)
    assert not output.exists()


def test_official_snapshot_preparation_writes_only_sealed_metadata(tmp_path, preparer):
    prepare, _ = preparer
    hosted = tmp_path / "hosted"
    official = tmp_path / "official"
    prepare(hosted, BASE)
    prepare(official, official_snapshots=True)

    assert [path.name for path in official.iterdir()] == ["models.json"]
    hosted_records = json.loads((hosted / "models.json").read_bytes())
    official_records = json.loads((official / "models.json").read_bytes())
    for name, record in official_records.items():
        assert (
            record.pop("source")
            == f"https://huggingface.co/PaddlePaddle/{name}/resolve/" + "a" * 40
        )
        hosted_records[name].pop("source")
        assert record == hosted_records[name]


@pytest.mark.parametrize(
    "base",
    ["http://artifacts.example.invalid", "https://user@example.invalid", BASE + "?query=1"],
)
def test_archive_base_requires_a_clean_https_url(tmp_path, preparer, base):
    prepare, _ = preparer
    with pytest.raises(ValueError, match="HTTPS archive base URL"):
        prepare(tmp_path / "output", base)


def test_mismatched_companion_license_leaves_no_output(tmp_path, preparer):
    prepare, assets = preparer
    assets[paddle_snapshots.COMPANION_LICENSE_URL] += b"corruption"
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="companion license SHA-256 mismatch"):
        prepare(output, BASE)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path, preparer):
    prepare, _ = preparer
    marker = tmp_path / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="Output directory already exists"):
        prepare(tmp_path, BASE)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize(
    ("status", "header", "error"),
    [
        ("200 OK", None, None),
        ("200 OK", "", None),
        ("206 Partial Content", None, "complete HTTP 200"),
        ("200 OK", "Content-Range: bytes 0-28/40", "complete HTTP 200"),
        ("200 OK", "Content-Length: 2048", "unexpected length"),
        ("200 OK", "Content-Length: +30", "invalid Content-Length"),
        ("200 OK", "Content-Length: 30\r\nContent-Length: 30", "invalid Content-Length"),
    ],
)
def test_download_response_framing(tmp_path, preparer, monkeypatch, status, header, error):
    prepare, assets = preparer
    target = next(url for url in assets if url.endswith("/README.md"))

    class Socket:
        def __init__(self, wire):
            self.wire = wire

        def makefile(self, mode):
            return io.BytesIO(self.wire)

    class Opener:
        def open(self, url, timeout):
            body = assets[url]
            response_status = status if url == target else "200 OK"
            headers = (
                header if url == target and header is not None else f"Content-Length: {len(body)}"
            )
            if headers:
                headers += "\r\n"
            wire = f"HTTP/1.1 {response_status}\r\n{headers}\r\n".encode() + body
            response = HTTPResponse(Socket(wire))
            response.begin()
            return response

    monkeypatch.setattr(paddle_snapshots, "build_opener", lambda *handlers: Opener())
    monkeypatch.setattr(paddle_snapshots, "download", official_download)
    output = tmp_path / "output"
    if error is None:
        prepare(output, BASE)
        assert (output / "models.json").is_file()
    else:
        with pytest.raises(ValueError, match=error):
            prepare(output, BASE)
        assert not output.exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_atomic_model_output(tmp_path, preparer, monkeypatch, interrupt):
    prepare, _ = preparer
    config = prepare.__globals__
    output = tmp_path / "output"

    def at_publication(staged, destination):
        assert destination == output
        assert not destination.exists()
        assert (staged / "models.json").is_file()
        if interrupt:
            raise KeyboardInterrupt
        destination.mkdir()
        publish_directory(staged, destination)

    monkeypatch.setitem(config, "publish_directory", at_publication)
    with pytest.raises(KeyboardInterrupt if interrupt else FileExistsError):
        prepare(output, BASE)
    if interrupt:
        assert not output.exists()
    else:
        assert output.is_dir()
        assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".paddle-models-*")) == []
