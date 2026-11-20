"""Official Paddle model revisions and their reproducible archive recipe."""

import hashlib
import re
import tarfile
from collections.abc import Mapping
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, NamedTuple
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MIB = 1024**2
APACHE_LICENSE_HEADER = b"Apache License\nVersion 2.0, January 2004"
COMPANION_LICENSE_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleX/"
    "ffb64904d23708863ff5b8da312a5cbd52a7f462/LICENSE"
)
COMPANION_LICENSE_SHA256 = "accc36c817ac5ede473d05732e14afc11cb6e55ef5919bc826b653f631165043"
STATIC_FILES = ("README.md", "inference.json", "inference.pdiparams", "inference.yml")


class Model(NamedTuple):
    repository: str
    revision: str
    files: tuple[str, ...]
    weights: str
    weight_sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/PaddlePaddle/{self.repository}"

    @property
    def source(self) -> str:
        return f"{self.url}/resolve/{self.revision}"

    @property
    def inventory(self) -> set[str]:
        if "LICENSE" in self.files:
            return set(self.files)
        return {*self.files, "LICENSE", "LICENSE-PROVENANCE.txt"}

    def resolve_url(self, filename: str) -> str:
        return f"{self.source}/{quote(filename)}"


MODELS = {
    "PP-DocLayoutV3": Model(
        "PP-DocLayoutV3",
        "241f8bdfc77a7c7bee915a5057aaee58c235a8d3",
        STATIC_FILES,
        "inference.pdiparams",
        "70bd316b0582769ec968829fd1feb1a6a58b7c941b938327e551b6b12b45c137",
    ),
    "PP-LCNet_x1_0_doc_ori": Model(
        "PP-LCNet_x1_0_doc_ori",
        "d3b95a6dff5fe8a94f2748e12b61cb26818a0df8",
        (*STATIC_FILES, "config.json"),
        "inference.pdiparams",
        "e8d6e7c5d264507e40e58a655779059d616b20d7441ea22047d829eb3931989c",
    ),
    "UVDoc": Model(
        "UVDoc",
        "16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3",
        (*STATIC_FILES, "config.json"),
        "inference.pdiparams",
        "810488899520e0da843b9bd9769ba4949f1c81e357f0eceb12d4a7da459c3eca",
    ),
    "PaddleOCR-VL-1.6-0.9B": Model(
        "PaddleOCR-VL-1.6",
        "132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd",
        (
            "LICENSE",
            "README.md",
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "inference.yml",
            "model.safetensors",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ),
        "model.safetensors",
        "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db",
    ),
}


class HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Official asset redirects must preserve HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, destination: Path, limit: int) -> None:
    opener = build_opener(HTTPSOnlyRedirect())
    with opener.open(url, timeout=60) as response:
        if response.status != 200 or "Content-Range" in response.headers:
            raise ValueError(f"Expected a complete HTTP 200 response for {destination.name}")
        content_lengths = response.headers.get_all("Content-Length", [])
        expected = None
        if content_lengths:
            if len(content_lengths) != 1 or not re.fullmatch(r"[0-9]+", content_lengths[0]):
                raise ValueError(
                    f"Official asset has invalid Content-Length for {destination.name}"
                )
            digits = content_lengths[0].lstrip("0") or "0"
            if len(digits) > len(str(limit)) or int(digits) > limit:
                raise ValueError(f"Official asset exceeds size bound for {destination.name}")
            expected = int(digits)

        remaining = limit
        with destination.open("xb") as output:
            while chunk := response.read(min(MIB, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError(f"Official asset exceeds size bound for {destination.name}")
                output.write(chunk)
        received = limit - remaining
        if expected is not None and received != expected:
            raise ValueError(f"Official asset has unexpected length for {destination.name}")
        if not received:
            raise ValueError(f"Official asset is empty for {destination.name}")


def official_snapshot_source(name: str) -> str:
    return MODELS[name].source


def is_official_snapshot(name: str, source: str) -> bool:
    return source == official_snapshot_source(name)


def is_snapshot_reference(source: str) -> bool:
    parsed = urlsplit(source)
    return parsed.hostname == "huggingface.co" and bool(
        re.fullmatch(r"/PaddlePaddle/[^/]+/resolve/[^/]+/?", parsed.path)
    )


def companion_license_provenance(model: Model) -> bytes:
    return (
        f"Model license declaration: {model.resolve_url('README.md')}\n"
        f"Apache-2.0 text: {COMPANION_LICENSE_URL}\n"
        f"Apache-2.0 text SHA-256: {COMPANION_LICENSE_SHA256}\n"
    ).encode()


def download_model(model: Model, tree: Path, sizes: Mapping[str, int] | None = None) -> None:
    for filename in model.files:
        limit = 3 * 1024 * MIB if filename == model.weights else 32 * MIB
        if sizes is not None:
            limit = min(limit, sizes[filename])
        download(model.resolve_url(filename), tree / filename, limit)
    with (tree / model.weights).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != model.weight_sha256:
            raise ValueError(f"Weight SHA-256 mismatch for {model.repository}")

    license_path = tree / "LICENSE"
    if "LICENSE" not in model.files:
        readme = (tree / "README.md").read_text(encoding="utf-8")
        frontmatter, delimiter, _ = readme.partition("\n---\n")
        if (
            not delimiter
            or not frontmatter.startswith("---\n")
            or not re.search(r"^license: apache-2\.0\s*$", frontmatter, re.MULTILINE)
        ):
            raise ValueError(
                f"Official Apache-2.0 model-card metadata missing for {model.repository}"
            )
        limit = MIB if sizes is None else min(MIB, sizes["LICENSE"])
        download(COMPANION_LICENSE_URL, license_path, limit)
        with license_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != COMPANION_LICENSE_SHA256:
                raise ValueError("PaddleX companion license SHA-256 mismatch")
        (tree / "LICENSE-PROVENANCE.txt").write_bytes(companion_license_provenance(model))
    license_lines = license_path.read_bytes().lstrip().splitlines()[:2]
    if b"\n".join(line.strip() for line in license_lines) != APACHE_LICENSE_HEADER:
        raise ValueError(f"Apache-2.0 license text missing for {model.repository}")


def write_model_archive(tree: Path, archive: Path) -> None:
    with tarfile.open(archive, mode="x", format=tarfile.USTAR_FORMAT) as tar:
        for file in sorted(tree.iterdir()):
            member = tarfile.TarInfo(file.name)
            member.size = file.stat().st_size
            member.mode = 0o644
            with file.open("rb") as stream:
                tar.addfile(member, stream)
