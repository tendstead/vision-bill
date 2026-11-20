#!/usr/bin/env python3
"""Resolve an official CUDA wheel before build using downloaded bytes and metadata.

Requires Python 3.12, uv and packaging (provided by the Make targets). Network
failures leave the existing inputs intact. Docker verifies the generated lock
and attestation together.
"""

import argparse
import email
import hashlib
import html.parser
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any, Never

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

INDEXES = {
    channel: f"https://www.paddlepaddle.org.cn/packages/stable/{channel}/paddlepaddle-gpu/"
    for channel in ("cu126", "cu129")
}
DEFAULT_CHANNEL = "cu129"
LEGACY_CHANNEL = "cu126"
HOSTS = {"www.paddlepaddle.org.cn", "paddle-whl.bj.bcebos.com", "paddle-whl.cdn.bcebos.com"}
WHEEL = re.compile(
    r"paddlepaddle_gpu-3\.3\.1-cp312-cp312-(?:linux_x86_64|manylinux[\w.]*x86_64)\.whl$",
    re.ASCII,
)


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Never:
        raise ValueError("Official CUDA input attempted an HTTP redirect")


class Links(html.parser.HTMLParser):
    def __init__(self, index: str) -> None:
        super().__init__()
        self.index = index
        self.urls: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value and re.fullmatch(r"[\x21-\x7e]+", value):
                    try:
                        self.urls.add(urllib.parse.urljoin(self.index, value.split("#", 1)[0]))
                    except ValueError:
                        continue


def official_index(channel: str) -> str:
    if channel not in INDEXES:
        raise ValueError(f"Unsupported CUDA channel {channel!r}")
    return INDEXES[channel]


def official_wheel(url: str, channel: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    directory, _, filename = parsed.path.rpartition("/")
    return (
        parsed.netloc in HOSTS
        and url == f"https://{parsed.netloc}{parsed.path}"
        and directory
        in {
            f"/stable/{channel}",
            f"/stable/{channel}/paddlepaddle-gpu",
            f"/packages/stable/{channel}/paddlepaddle-gpu",
        }
        and WHEEL.fullmatch(filename) is not None
    )


def select_wheel(document: str, channel: str = DEFAULT_CHANNEL) -> str:
    links = Links(official_index(channel))
    links.feed(document)
    candidates = {url for url in links.urls if official_wheel(url, channel)}
    if len(candidates) != 1:
        raise ValueError(f"Official index must identify exactly one {channel} cp312 x86_64 wheel")
    return candidates.pop()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def executable_requirements(text: str) -> dict[str, tuple[Requirement, frozenset[str]]]:
    """Accept only closed, unconditional, hash-pinned uv requirements output."""
    entries: dict[str, tuple[Requirement, frozenset[str]]] = {}
    pending = ""
    for raw in text.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line:
            continue
        continued = line.endswith("\\")
        pending += " " + (line[:-1].rstrip() if continued else line)
        if continued:
            continue
        statement, pending = pending.strip(), ""
        hash_tokens = re.findall(r"(?:^|\s)--hash=sha256:([0-9a-f]{64})(?=\s|$)", statement)
        requirement_text = re.sub(r"\s+--hash=sha256:[0-9a-f]{64}(?=\s|$)", "", statement)
        if not hash_tokens or " --" in requirement_text or requirement_text.startswith("-"):
            raise ValueError("CUDA lock must contain only executable hash-pinned requirements")
        requirement = Requirement(requirement_text)
        name = str(canonicalize_name(requirement.name))
        if name in entries or requirement.marker or requirement.extras:
            raise ValueError("Duplicate, conditional or extra-bearing CUDA lock requirement")
        if not requirement.url:
            pins = list(requirement.specifier)
            if len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
                raise ValueError("CUDA lock versions must be exact")
        entries[name] = (requirement, frozenset(hash_tokens))
    if pending or not entries:
        raise ValueError("CUDA executable lock is empty or incomplete")
    return entries


def validate_lock(
    lock: Path, runtime_lock: Path | None = None, *, channel: str | None = None
) -> dict[str, Any]:
    text = lock.read_text()
    attestation = lock.with_suffix(".attestation.json")
    if "UNRESOLVED" in text or not text.strip() or not attestation.is_file():
        raise ValueError("CUDA lock unresolved: run official-index resolver before building")
    data: dict[str, Any] = json.loads(attestation.read_text())
    fields = {
        "index",
        "index_sha256",
        "wheel_url",
        "wheel_sha256",
        "metadata_sha256",
        "metadata_name",
        "metadata_version",
        "requires_dist",
        "lock_sha256",
        "platform",
    }
    if not isinstance(data, dict) or set(data) not in (fields, fields | {"cuda_channel"}):
        raise ValueError("CUDA attestation schema is incomplete or unsupported")
    for key in ("index_sha256", "wheel_sha256", "metadata_sha256", "lock_sha256"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise ValueError("CUDA attestation digest is invalid")
    if (
        not isinstance(data["requires_dist"], list)
        or not data["requires_dist"]
        or not all(isinstance(value, str) for value in data["requires_dist"])
        or not isinstance(data["wheel_url"], str)
    ):
        raise ValueError("CUDA attestation metadata is invalid")
    attested_channel = data.get("cuda_channel", LEGACY_CHANNEL)
    if (
        not isinstance(attested_channel, str)
        or attested_channel not in INDEXES
        or (channel is not None and channel != attested_channel)
        or data["index"] != INDEXES[attested_channel]
        or not official_wheel(data["wheel_url"], attested_channel)
        or data["lock_sha256"] != sha256(lock)
        or data["metadata_name"] != "paddlepaddle-gpu"
        or data["metadata_version"] != "3.3.1"
        or data["platform"] != "linux-x86_64-cp312"
    ):
        raise ValueError("CUDA lock or official input attestation mismatch")
    entries = executable_requirements(text)
    natives = {name for name in entries if name.startswith("paddlepaddle")}
    if natives != {"paddlepaddle-gpu"}:
        raise ValueError("Exactly one intended native Paddle requirement is required")
    native, hashes = entries["paddlepaddle-gpu"]
    if native.url != data["wheel_url"] or hashes != {data["wheel_sha256"]}:
        raise ValueError("Executable native requirement differs from attested official wheel")
    common_path = runtime_lock or lock.with_name("requirements.runtime.lock")
    common = executable_requirements(common_path.read_text())
    for name, (requirement, hashes) in common.items():
        if (
            name not in entries
            or str(entries[name][0]) != str(requirement)
            or entries[name][1] != hashes
        ):
            raise ValueError("CUDA lock omits or changes a pinned runtime input")
    # Closed common runtime inputs are preserved verbatim. Extra wheel dependencies
    # must also be unconditional exact/hash pins; the resolver closes their graph,
    # and pip check verifies installed transitive metadata after installation.
    for name, (requirement, _) in entries.items():
        if requirement.url and name not in common and name != "paddlepaddle-gpu":
            raise ValueError("Unexpected direct URL in CUDA lock")
    environment = {
        "python_version": "3.12",
        "python_full_version": "3.12.0",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "extra": "",
    }
    for specification in data["requires_dist"]:
        required = Requirement(specification)
        if required.marker and not required.marker.evaluate(environment):
            continue
        name = str(canonicalize_name(required.name))
        if name not in entries:
            raise ValueError("CUDA lock omits an active native wheel dependency")
        actual = entries[name][0]
        if required.url:
            if required.url != actual.url:
                raise ValueError("Native dependency URL differs from wheel metadata")
        elif actual.url or not required.specifier.contains(
            next(iter(actual.specifier)).version, prereleases=True
        ):
            raise ValueError("Native dependency pin violates wheel metadata")
    return data


def wheel_cache() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache = root.expanduser() / "vision-bill" / "cuda-wheels"
    cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache.mkdir(mode=0o700, exist_ok=True)
    for directory in (cache.parent, cache):
        if (
            directory.is_symlink()
            or directory.stat().st_uid != os.getuid()
            or directory.stat().st_mode & 0o022
        ):
            raise ValueError("CUDA wheel cache must be an owned directory")
    return cache.resolve()


def download_wheel(url: str, target: Path) -> None:
    opener = urllib.request.build_opener(NoRedirects())
    with opener.open(url, timeout=600) as response, target.open("xb") as stream:
        shutil.copyfileobj(response, stream, 1024 * 1024)


def seed_from_previous_image(image: str, data: dict[str, Any], candidate: Path) -> bool:
    """Try the one HTTP body pip 26.3 might have retained in an earlier image."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("A previous CUDA image seed requires its exact Docker image ID")
    # Official URLs admitted above are already in CacheControl's normalized form.
    key = hashlib.sha224(data["wheel_url"].encode(), usedforsecurity=False).hexdigest()
    body = Path("/root/.cache/pip/http-v2", *key[:5], key + ".body")
    with candidate.open("xb") as output:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--log-driver=none",
                    "--entrypoint=/bin/cat",
                    image,
                    str(body),
                ],
                stdout=output,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
    return result.returncode == 0 and sha256(candidate) == data["wheel_sha256"]


def wheel_context(
    data: dict[str, Any], downloaded: Path | None = None, *, prior_image: str | None = None
) -> Path:
    """Expose only the verified wheel for this digest to the Docker builder."""
    base = wheel_cache()
    directory: Path = base / data["wheel_sha256"]
    directory.mkdir(mode=0o700, exist_ok=True)
    if (
        directory.is_symlink()
        or directory.stat().st_uid != os.getuid()
        or directory.stat().st_mode & 0o077
    ):
        raise ValueError("CUDA wheel context must be a private owned directory")
    target = directory / data["wheel_url"].rsplit("/", 1)[1]
    if any(path != target for path in directory.iterdir()) or target.is_symlink():
        raise ValueError("Unexpected files in CUDA wheel context")
    if target.exists():
        if not target.is_file():
            raise ValueError("CUDA wheel cache contains a nonregular entry")
        if sha256(target) == data["wheel_sha256"]:
            return directory
    if downloaded is not None:
        # The resolver hashed this newly downloaded private file on this filesystem.
        downloaded.replace(target)
    else:
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            candidate = Path(temporary) / target.name
            if prior_image is None or not seed_from_previous_image(prior_image, data, candidate):
                candidate.unlink(missing_ok=True)
                download_wheel(data["wheel_url"], candidate)
                if sha256(candidate) != data["wheel_sha256"]:
                    raise ValueError("Downloaded CUDA wheel hash mismatch")
            candidate.replace(target)
    return directory


def install_requirements(lock: Path, data: dict[str, Any], context: Path, output: Path) -> None:
    """Use the immutable bind for the attested wheel; retain the original release lock."""
    wheel = context / data["wheel_url"].rsplit("/", 1)[1]
    if not list(context.glob("*.whl")):
        # Direct Docker builds can still fetch the exact URL under --require-hashes.
        output.write_bytes(lock.read_bytes())
        return
    if wheel.is_symlink() or not wheel.is_file() or sha256(wheel) != data["wheel_sha256"]:
        raise ValueError("Build context CUDA wheel hash mismatch")
    entries = executable_requirements(lock.read_text())
    with output.open("w") as stream:
        for name, (requirement, hashes) in entries.items():
            source = (
                f"paddlepaddle-gpu @ {wheel.as_uri()}"
                if name == "paddlepaddle-gpu"
                else str(requirement)
            )
            pins = " ".join(f"--hash=sha256:{digest}" for digest in sorted(hashes))
            stream.write(f"{source} {pins}\n")


def resolve(service: Path, channel: str = DEFAULT_CHANNEL) -> None:
    index = official_index(channel)
    opener = urllib.request.build_opener(NoRedirects())
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PADDLEX"] = "3.7.2"
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PADDLEOCR"] = "3.7.0"
    with opener.open(index, timeout=60) as response:
        document = response.read().decode()
    url = select_wheel(document, channel)
    with tempfile.TemporaryDirectory(prefix="cuda-lock-", dir=wheel_cache()) as scratch:
        directory = Path(scratch)
        wheel = directory / urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
        download_wheel(url, wheel)
        digest = sha256(wheel)
        with zipfile.ZipFile(wheel) as archive:
            paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(paths) != 1:
                raise ValueError("Wheel must have one distribution metadata record")
            metadata_bytes = archive.read(paths[0])
        metadata = email.message_from_bytes(metadata_bytes)
        if (
            metadata["Name"].lower().replace("_", "-") != "paddlepaddle-gpu"
            or metadata["Version"] != "3.3.1"
        ):
            raise ValueError("Official wheel metadata does not match requested build")
        # Use actual wheel METADATA to seed all active Requires-Dist requirements.
        # Native wheel is installed no-deps only after this complete lock is installed.
        incoming = directory / "cuda.in"
        incoming.write_text(
            (service / "build/runtime.in").read_text()
            + "\n"
            + "\n".join(metadata.get_all("Requires-Dist", []))
            + "\n"
        )
        output = directory / "requirements.cuda.lock"
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(incoming),
                "--python-version",
                "3.12",
                "--python-platform",
                "x86_64-manylinux_2_39",
                "--generate-hashes",
                "--no-header",
                "--no-annotate",
                "--constraint",
                str(service / "requirements.runtime.lock"),
                "--output-file",
                str(output),
            ],
            check=True,
        )
        with output.open("a") as stream:
            stream.write(f"\npaddlepaddle-gpu @ {url} \\\n    --hash=sha256:{digest}\n")
        data = {
            "cuda_channel": channel,
            "index": index,
            "index_sha256": hashlib.sha256(document.encode()).hexdigest(),
            "wheel_url": url,
            "wheel_sha256": digest,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "metadata_name": "paddlepaddle-gpu",
            "metadata_version": "3.3.1",
            "requires_dist": metadata.get_all("Requires-Dist", []),
            "lock_sha256": sha256(output),
            "platform": "linux-x86_64-cp312",
        }
        attestation = output.with_suffix(".attestation.json")
        attestation.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
        validate_lock(output, service / "requirements.runtime.lock", channel=channel)
        wheel_context(data, downloaded=wheel)
        (service / output.name).write_bytes(output.read_bytes())
        (service / attestation.name).write_bytes(attestation.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--prior-image", help="optional exact previous image ID for --wheel-context")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--wheel-context", action="store_true", help="print the verified local Docker context"
    )
    mode.add_argument("--install-lock", nargs=2, type=Path, metavar=("WHEEL_CONTEXT", "OUTPUT"))
    parser.add_argument(
        "--channel",
        choices=INDEXES,
        help=f"CUDA channel (resolution defaults to {DEFAULT_CHANNEL})",
    )
    parser.add_argument("--service", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.prior_image is not None and not args.wheel_context:
        parser.error("--prior-image requires --wheel-context")
    if args.check:
        evidence = validate_lock(args.check, channel=args.channel)
        if args.wheel_context:
            print(wheel_context(evidence, prior_image=args.prior_image))
        if args.install_lock:
            install_requirements(args.check, evidence, *args.install_lock)
    elif args.wheel_context or args.install_lock:
        parser.error("wheel context and installation lock operations require --check")
    else:
        resolve(args.service, args.channel or DEFAULT_CHANNEL)
