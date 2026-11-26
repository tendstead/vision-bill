#!/usr/bin/env python3
"""Resolve an official CUDA wheel before build using downloaded bytes and metadata.

Requires Python 3.12, uv and packaging (provided by the Make targets). Network
failures leave the existing inputs intact. Docker verifies the generated lock
and attestation together.
"""

import argparse
import email
import fcntl
import hashlib
import html.parser
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPMessage, IncompleteRead
from importlib.metadata import version
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
    start = time.monotonic()
    last = start
    received = 0
    total = None
    etag = None
    print(f"Downloading CUDA wheel from {url}", file=sys.stderr, flush=True)
    with target.open("xb") as stream:
        for attempt in range(3):
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            if received and etag is not None:
                request.add_header("Range", f"bytes={received}-")
                request.add_header("If-Range", etag)
            try:
                with opener.open(request, timeout=600) as response:
                    if response.status == 200:
                        stream.seek(0)
                        stream.truncate()
                        received = 0
                        length = response.headers.get("Content-Length")
                        total = int(length) if length is not None else None
                        etag = response.headers.get("ETag")
                    elif not (
                        response.status == 206
                        and received
                        and total is not None
                        and response.headers.get("ETag") == etag
                        and response.headers.get("Content-Range")
                        == f"bytes {received}-{total - 1}/{total}"
                    ):
                        raise ValueError("Unexpected CUDA wheel range response")
                    elif response.headers.get("Content-Length") not in (
                        None,
                        str(total - received),
                    ):
                        raise ValueError("Unexpected CUDA wheel range response length")
                    while chunk := response.read(1024 * 1024):
                        if total is not None and received + len(chunk) > total:
                            raise ValueError("CUDA wheel exceeds the declared response length")
                        stream.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - last >= 10:
                            print(
                                f"CUDA wheel: {received} bytes, {now - start:.1f}s, "
                                f"{received / max(now - start, 0.001) / 1048576:.1f} MiB/s",
                                file=sys.stderr,
                                flush=True,
                            )
                            last = now
                    if total is not None and received != total:
                        raise EOFError("Incomplete CUDA wheel transfer")
                    break
            except (
                EOFError,
                IncompleteRead,
                ConnectionError,
                TimeoutError,
                urllib.error.URLError,
            ) as exc:
                if isinstance(exc, urllib.error.URLError) and not isinstance(
                    exc.reason, (ConnectionError, TimeoutError)
                ):
                    raise
                if attempt == 2:
                    raise
                if not (
                    received
                    and total
                    and received < total
                    and etag
                    and re.fullmatch(r'"[^"]+"', etag)
                ):
                    received = 0
                    total = None
                    etag = None
                    stream.seek(0)
                    stream.truncate()
                print(
                    f"Retrying CUDA wheel transfer from byte {received}",
                    file=sys.stderr,
                    flush=True,
                )
    elapsed = time.monotonic() - start
    print(
        f"CUDA wheel: {received} bytes, {elapsed:.1f}s, "
        f"{received / max(elapsed, 0.001) / 1048576:.1f} MiB/s",
        file=sys.stderr,
    )


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
    with cache_lock(base / (data["wheel_sha256"] + ".lock")):
        return populate_wheel_context(base, data, downloaded, prior_image)


def populate_wheel_context(
    base: Path, data: dict[str, Any], downloaded: Path | None, prior_image: str | None
) -> Path:
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
            print(f"CUDA wheel cache hit: verified {target.name}", file=sys.stderr, flush=True)
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


def atomic_bytes(path: Path, contents: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def cache_lock(path: Path) -> Iterator[None]:
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextmanager
def resolution_lock(service: Path, cache: Path) -> Iterator[None]:
    # Both files in the checkout can be replaced. Lock the stable directory inode
    # so every resolver writing here participates, even with a separate XDG cache.
    checkout_fd = os.open(service, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(checkout_fd, fcntl.LOCK_EX)
        with cache_lock(cache / ".resolution.lock"):
            yield
    finally:
        os.close(checkout_fd)


def resolution_key(service: Path, channel: str) -> str:
    inputs = {
        "channel": channel,
        "runtime": sha256(service / "requirements.runtime.lock"),
        "requirements": sha256(service / "build/runtime.in"),
        "resolver": sha256(Path(__file__)),
        "packaging": version("packaging"),
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "settings": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("UV_") and key not in {"UV_CACHE_DIR", "UV_OFFLINE"}
        },
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def resolve(service: Path, channel: str = DEFAULT_CHANNEL, *, ensure: bool = False) -> None:
    cache = wheel_cache()
    with resolution_lock(service, cache):
        resolve_locked(service, channel, ensure, cache)


def resolve_locked(service: Path, channel: str, ensure: bool, cache: Path) -> None:
    lock = service / "requirements.cuda.lock"
    attestation = lock.with_suffix(".attestation.json")
    entry = cache / ("resolution-" + resolution_key(service, channel) + ".json")
    if ensure:
        try:
            validate_lock(lock, channel=channel)
            markers = re.findall(
                r"^# vision-bill-resolution: (\S+)$", lock.read_text(), re.MULTILINE
            )
            if not markers or markers == [entry.stem]:
                return  # Legacy operator-supplied locks remain authoritative.
        except (ValueError, FileNotFoundError):
            pass
    if ensure and entry.exists():
        try:
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                restored = Path(temporary) / lock.name
                pair = json.loads(entry.read_bytes())
                if pair["key"] != entry.stem or re.findall(
                    r"^# vision-bill-resolution: (\S+)$", pair["lock"], re.MULTILINE
                ) != [entry.stem]:
                    raise ValueError("Cached CUDA resolution input mismatch")
                restored.write_text(pair["lock"])
                restored.with_suffix(".attestation.json").write_text(pair["attestation"])
                validate_lock(restored, service / "requirements.runtime.lock", channel=channel)
                atomic_bytes(lock, restored.read_bytes())
                atomic_bytes(attestation, restored.with_suffix(".attestation.json").read_bytes())
                return
        except (ValueError, KeyError, TypeError, UnicodeError):
            pass
    resolve_official(service, channel, entry)


def build_context(
    lock: Path, channel: str, prior_image: str | None = None, *, refresh: bool = False
) -> Path:
    cache = wheel_cache()
    with resolution_lock(lock.parent, cache):
        resolve_locked(lock.parent, channel, not refresh, cache)
        data = validate_lock(lock, channel=channel)
        lock_bytes = lock.read_bytes()
        attestation = lock.with_suffix(".attestation.json")
        attested_bytes = attestation.read_bytes()
    # Different channels can now resolve concurrently with this download/build.
    wheel = wheel_context(data, prior_image=prior_image) / data["wheel_url"].rsplit("/", 1)[1]
    snapshot = cache / ("build-" + hashlib.sha256(attested_bytes).hexdigest())
    with cache_lock(cache / (snapshot.name + ".lock")):
        if not snapshot.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staging = Path(temporary) / "build"
                staging.mkdir(mode=0o700)
                (staging / lock.name).write_bytes(lock_bytes)
                (staging / attestation.name).write_bytes(attested_bytes)
                os.link(wheel, staging / wheel.name)
                staging.rename(snapshot)
        if (
            snapshot.is_symlink()
            or set(snapshot.iterdir())
            != {snapshot / lock.name, snapshot / attestation.name, snapshot / wheel.name}
            or any(p.is_symlink() for p in snapshot.iterdir())
            or (snapshot / lock.name).read_bytes() != lock_bytes
            or (snapshot / attestation.name).read_bytes() != attested_bytes
        ):
            raise ValueError("CUDA build snapshot does not match verified resolution")
        if not (snapshot / wheel.name).samefile(wheel):
            # A corrupted wheel cache entry can have been replaced with verified bytes.
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                replacement = Path(temporary) / wheel.name
                os.link(wheel, replacement)
                replacement.replace(snapshot / wheel.name)
    return snapshot


def reuse_lock_digest(lock: Path, channel: str, image: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("CUDA reuse requires an exact local image ID")
    output = subprocess.check_output(
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
            "--entrypoint=sha256sum",
            image,
            "/opt/locks/requirements.cuda.lock",
        ],
        text=True,
    )
    match = re.fullmatch(r"([a-f0-9]{64})  /opt/locks/requirements.cuda.lock\n", output)
    if not match:
        raise ValueError("Cannot read the local CUDA image lock identity")
    digest = match[1]
    cache = wheel_cache()
    with resolution_lock(lock.parent, cache):
        for candidate in (lock, *cache.glob("build-*/requirements.cuda.lock")):
            if candidate.is_file() and sha256(candidate) == digest:
                validate_lock(
                    candidate, lock.with_name("requirements.runtime.lock"), channel=channel
                )
                return digest
    raise ValueError("Restore the validated CUDA lock and attestation for this local image")


def resolve_official(service: Path, channel: str, entry: Path) -> None:
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
                "--no-config",
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
            stdout=sys.stderr,
            check=True,
        )
        with output.open("a") as stream:
            stream.write(f"\n# vision-bill-resolution: {entry.stem}\n")
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
        atomic_bytes(
            entry,
            json.dumps(
                {
                    "key": entry.stem,
                    "lock": output.read_text(),
                    "attestation": attestation.read_text(),
                }
            ).encode(),
        )
        atomic_bytes(service / output.name, output.read_bytes())
        atomic_bytes(service / attestation.name, attestation.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="fresh official resolution for a build context"
    )
    parser.add_argument("--check", type=Path)
    parser.add_argument(
        "--prior-image", help="optional exact previous image ID for --wheel-context"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--wheel-context", action="store_true", help="print the verified local Docker context"
    )
    mode.add_argument(
        "--build-context", action="store_true", help="print immutable local build inputs"
    )
    mode.add_argument(
        "--reuse-image", help="verify and print the existing image lock digest offline"
    )
    mode.add_argument("--install-lock", nargs=2, type=Path, metavar=("WHEEL_CONTEXT", "OUTPUT"))
    parser.add_argument(
        "--channel",
        choices=INDEXES,
        help=f"CUDA channel (resolution defaults to {DEFAULT_CHANNEL})",
    )
    parser.add_argument("--service", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.prior_image is not None and not (args.wheel_context or args.build_context):
        parser.error("--prior-image requires --wheel-context")
    if args.refresh and not args.build_context:
        parser.error("--refresh requires --build-context")
    if args.check and args.build_context:
        print(
            build_context(
                args.check, args.channel or DEFAULT_CHANNEL, args.prior_image, refresh=args.refresh
            )
        )
    elif args.check and args.reuse_image:
        print(reuse_lock_digest(args.check, args.channel or DEFAULT_CHANNEL, args.reuse_image))
    elif args.check:
        evidence = validate_lock(args.check, channel=args.channel)
        if args.wheel_context:
            print(wheel_context(evidence, prior_image=args.prior_image))
        if args.install_lock:
            install_requirements(args.check, evidence, *args.install_lock)
    elif args.wheel_context or args.build_context or args.reuse_image or args.install_lock:
        parser.error("wheel context and installation lock operations require --check")
    else:
        resolve(args.service, args.channel or DEFAULT_CHANNEL)
