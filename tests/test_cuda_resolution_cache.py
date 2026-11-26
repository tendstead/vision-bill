import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_ocr_runtime_builds import cuda_lock_fixture


def test_cached_resolution_is_validated_and_keyed(tmp_path, monkeypatch):
    module, write, _, _ = cuda_lock_fixture(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "build").mkdir()
    incoming = tmp_path / "build/runtime.in"
    incoming.write_text("numpy")
    lock = write()
    attestation = lock.with_suffix(".attestation.json")
    original = lock.read_bytes()
    calls = []

    def official(service, channel, entry):
        calls.append(channel)
        write(original.decode() + f"\n# vision-bill-resolution: {entry.stem}\n")
        module.atomic_bytes(
            entry,
            json.dumps(
                {
                    "key": entry.stem,
                    "lock": lock.read_text(),
                    "attestation": attestation.read_text(),
                }
            ).encode(),
        )

    monkeypatch.setattr(module, "resolve_official", official)
    module.resolve(tmp_path, "cu126", ensure=True)
    assert not calls  # Existing legacy lock is authoritative.
    module.resolve(tmp_path, "cu126")
    assert calls == ["cu126"]  # Explicit refresh still consults the official source.
    lock.unlink()
    attestation.unlink()
    module.resolve(tmp_path, "cu126", ensure=True)
    assert original in lock.read_bytes() and calls == ["cu126"]
    lock.unlink()
    (entry,) = module.wheel_cache().glob("resolution-*.json")
    pair = json.loads(entry.read_bytes())
    pair["lock"] += "corrupt"
    entry.write_text(json.dumps(pair))
    module.resolve(tmp_path, "cu126", ensure=True)
    assert len(calls) == 2
    forged_lock = write(original.decode() + "\n# vision-bill-resolution: wrong-inputs\n")
    pair = json.loads(entry.read_bytes())
    pair.update(lock=forged_lock.read_text(), attestation=attestation.read_text())
    entry.write_text(json.dumps(pair))
    lock.unlink()
    module.resolve(tmp_path, "cu126", ensure=True)
    assert len(calls) == 3
    incoming.write_text("numpy>=1")
    module.resolve(tmp_path, "cu126", ensure=True)
    assert len(calls) == 4


def test_concurrent_consumers_share_one_wheel_download(tmp_path, monkeypatch):
    module, write, _, _ = cuda_lock_fixture(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    evidence = module.validate_lock(write())
    start = threading.Barrier(2)
    calls = []

    def download(url, output):
        calls.append(url)
        output.write_bytes(b"synthetic wheel test bytes")

    def consumer():
        start.wait(timeout=5)
        return module.wheel_context(evidence)

    monkeypatch.setattr(module, "download_wheel", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: consumer(), range(2)))
    assert first == second and len(calls) == 1


@pytest.mark.parametrize("valid_range", [True, False])
def test_resume_requires_matching_representation(tmp_path, monkeypatch, capsys, valid_range):
    module, _, _, _ = cuda_lock_fixture(tmp_path)
    requests = []

    class Reply(io.BytesIO):
        pass

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            if len(requests) == 1:
                reply = Reply(b"abc")
                reply.status = 200
                reply.headers = {"Content-Length": "6", "ETag": '"pinned"'}
            else:
                assert request.get_header("Range") == "bytes=3-"
                assert request.get_header("If-range") == '"pinned"'
                reply = Reply(b"def")
                reply.status = 206
                reply.headers = {
                    "ETag": '"pinned"',
                    "Content-Range": "bytes 3-5/6" if valid_range else "bytes 4-5/6",
                }
            return reply

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_: Opener())
    output = tmp_path / "wheel"
    if valid_range:
        module.download_wheel("https://example.org/wheel", output)
        assert output.read_bytes() == b"abcdef"
        captured = capsys.readouterr()
        assert not captured.out and "6 bytes" in captured.err and "MiB/s" in captured.err
    else:
        with pytest.raises(ValueError, match="range response"):
            module.download_wheel("https://example.org/wheel", output)


def test_parallel_channels_keep_inputs_through_build_and_offline_image_reuse(tmp_path, monkeypatch):
    module, write, text, _ = cuda_lock_fixture(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "build").mkdir()
    (tmp_path / "build/runtime.in").write_text("numpy")
    lock = write()
    initial = module.validate_lock(lock)
    start_download = threading.Barrier(2)
    first_snapshot = threading.Event()
    wheel_context = module.wheel_context

    def prepare_wheel(data, **kwargs):
        if data.get("cuda_channel", "cu126") == "cu126":
            first_snapshot.set()
        start_download.wait(timeout=10)
        return wheel_context(data, **kwargs)

    def official(service, channel, entry):
        assert channel == "cu129"
        modified = text.replace("cu126/", "cu129/")
        write(
            modified,
            fields=lambda data: {
                **data,
                "cuda_channel": channel,
                "index": module.INDEXES[channel],
                "wheel_url": data["wheel_url"].replace("cu126/", "cu129/"),
            },
        )

    monkeypatch.setattr(module, "resolve_official", official)
    monkeypatch.setattr(module, "wheel_context", prepare_wheel)
    monkeypatch.setattr(
        module, "download_wheel", lambda _, path: path.write_bytes(b"synthetic wheel test bytes")
    )
    monkeypatch.setattr(module, "resolution_key", lambda _, channel: channel)
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Ensure legacy cu126 is snapshotted before cu129 overwrites the checkout.
        first = pool.submit(module.build_context, lock, "cu126")
        assert first_snapshot.wait(timeout=5)
        second = pool.submit(module.build_context, lock, "cu129")
        contexts = {"cu126": first.result(), "cu129": second.result()}
    assert module.validate_lock(lock, channel="cu129")
    for channel, directory in contexts.items():
        resolved = directory / lock.name
        attested = module.validate_lock(
            resolved, tmp_path / "requirements.runtime.lock", channel=channel
        )
        wheel = next(directory.glob("*.whl"))
        assert module.sha256(wheel) == attested["wheel_sha256"]
        assert wheel.stat().st_nlink > 1  # One physical cached wheel.
    assert contexts["cu126"] != contexts["cu129"]
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **_: (
            (initial["lock_sha256"] + "  /opt/locks/requirements.cuda.lock\n")
            if "--network=none" in args
            else pytest.fail("unexpected resolver invocation")
        ),
    )
    lock.write_text("UNRESOLVED")
    assert module.reuse_lock_digest(lock, "cu126", "sha256:" + "b" * 64) == initial["lock_sha256"]
    with pytest.raises(ValueError, match="attestation mismatch"):
        module.reuse_lock_digest(lock, "cu129", "sha256:" + "b" * 64)


def test_release_context_refreshes_even_with_valid_checkout(tmp_path, monkeypatch):
    module, write, _, _ = cuda_lock_fixture(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    lock = write()
    official_calls = []
    monkeypatch.setattr(module, "resolution_key", lambda *_: "release-test")

    def official(service, channel, entry):
        official_calls.append(channel)
        write(fields=lambda data: {**data, "index_sha256": "a" * 64})

    monkeypatch.setattr(module, "resolve_official", official)
    monkeypatch.setattr(
        module, "download_wheel", lambda _, path: path.write_bytes(b"synthetic wheel test bytes")
    )
    directory = module.build_context(lock, "cu126", refresh=True)
    assert official_calls == ["cu126"]
    assert (
        json.loads((directory / "requirements.cuda.attestation.json").read_bytes())["index_sha256"]
        == "a" * 64
    )


def test_one_checkout_serializes_writers_with_separate_cache_roots(tmp_path, monkeypatch):
    import fcntl
    import os

    module, _, _, _ = cuda_lock_fixture(tmp_path)
    roots = {channel: tmp_path / channel for channel in ("cu126", "cu129")}
    for root in roots.values():
        root.mkdir()
    selected = threading.local()
    entered_first = threading.Event()
    second_contended = threading.Event()
    order = []

    def publish(service, channel, ensure, cache):
        assert service == tmp_path and cache == roots[channel]
        order.append(channel)
        if channel == "cu126":
            entered_first.set()
            assert second_contended.wait(5)

    def resolve(channel):
        selected.cache = roots[channel]
        if channel == "cu129":
            assert entered_first.wait(5)
            # Verify this worker really contends with the first on their only
            # shared resource before calling the public resolver.
            fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
                second_contended.set()
        module.resolve(tmp_path, channel)

    monkeypatch.setattr(module, "wheel_cache", lambda: selected.cache)
    monkeypatch.setattr(module, "resolve_locked", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(resolve, roots))
    assert order == ["cu126", "cu129"]


@pytest.mark.parametrize("etag", [None, 'W/"weak"'])
def test_unvalidated_interrupted_transfer_restarts_cleanly(tmp_path, monkeypatch, capsys, etag):
    module, _, _, _ = cuda_lock_fixture(tmp_path)
    attempts = []
    url = "https://paddle-whl.bj.bcebos.com/stable/cu129/wheel.whl"

    class Reply(io.BytesIO):
        status = 200

    class Opener:
        def open(self, request, timeout):
            attempts.append(request)
            assert not request.get_header("Range") and not request.get_header("If-range")
            body = b"bad" if len(attempts) == 1 else b"fresh!"
            reply = Reply(body)
            reply.headers = {"Content-Length": "6"}
            if etag:
                reply.headers["ETag"] = etag
            return reply

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_: Opener())
    target = tmp_path / "wheel"
    module.download_wheel(url, target)
    assert len(attempts) == 2 and target.read_bytes() == b"fresh!"
    output = capsys.readouterr()
    assert not output.out and output.err.count(url) == 1
    assert "from byte 0" in output.err


def test_verified_wheel_hit_is_reported_only_on_stderr(tmp_path, monkeypatch, capsys):
    module, write, _, _ = cuda_lock_fixture(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    wheel = tmp_path / "wheel"
    wheel.write_bytes(b"synthetic wheel test bytes")
    data = module.validate_lock(write())
    cached = module.wheel_context(data, downloaded=wheel)
    capsys.readouterr()
    assert module.wheel_context(data) == cached
    output = capsys.readouterr()
    assert not output.out and "CUDA wheel cache hit: verified" in output.err
