"""Tiny local artifacts exercise CPU integrity, never certify native OCR."""

import hashlib
import io
import json
import tarfile
from http.client import HTTPMessage
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE = Path(__file__).parents[1] / "paddleocr-service"
NAMES = ("PP-DocLayoutV3", "PP-LCNet_x1_0_doc_ori", "UVDoc", "PaddleOCR-VL-1.6-0.9B")
KINDS = ("upright", "rotated", "warped", "table", "formula")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def archive(tmp_path, name, files):
    path = tmp_path / (name + ".tar")
    with tarfile.open(path, "w") as tar:
        for filename, raw in files.items():
            member = tarfile.TarInfo(filename)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    inventory = {key: {"sha256": sha(raw), "size_bytes": len(raw)} for key, raw in files.items()}
    return {
        "source": str(path),
        "sha256": sha(path.read_bytes()),
        "size_bytes": path.stat().st_size,
        "unpacked_bytes": sum(len(raw) for raw in files.values()),
        "files": inventory,
        "tree_sha256": identity({key: row["sha256"] for key, row in inventory.items()}),
        "license": {"spdx": "CC0-1.0", "text_file": "LICENSE", "redistribution_allowed": True},
        "provenance": {"url": "https://example.invalid/test-only", "revision": "synthetic-test"},
    }


@pytest.fixture
def release(tmp_path):
    image_config = tmp_path / "image-config.json"
    image_config.write_text(
        '{"architecture":"amd64","os":"linux","rootfs":{"type":"layers","diff_ids":[]}}'
    )
    models = {
        name: archive(tmp_path, name, {"weights": name.encode(), "LICENSE": b"CC0 test"})
        for name in NAMES
    }
    files = {kind + ".png": ("tiny-private-test-" + kind).encode() for kind in KINDS}
    files["LICENSE"] = b"CC0 test fixtures"
    fixtures = archive(tmp_path, "fixtures", files)
    fixtures["cases"] = [
        {
            "kind": kind,
            "path": kind + ".png",
            "sha256": sha(files[kind + ".png"]),
            "required_labels": [kind] if kind in {"table", "formula"} else [],
            "license": fixtures["license"],
            "provenance": fixtures["provenance"],
        }
        for kind in KINDS
    ]
    source_files = {
        str(path.relative_to(SERVICE)): sha(path.read_bytes())
        for directory in ("ocr_service", "ocr_contract")
        for path in (SERVICE / directory).rglob("*.py")
    }
    source_files.update(
        {
            name: sha((SERVICE / name).read_bytes())
            for name in ("server.py", "profile.json", "runtime-lock.json")
        }
    )
    bundle = {
        "schema_version": 1,
        "accelerator": "cuda",
        "gpu_index": 0,
        "image_identity": "example.test/ocr@sha256:" + "a" * 64,
        "image_config_sha256": sha(image_config.read_bytes()),
        "profile_sha256": sha((SERVICE / "profile.json").read_bytes()),
        "source_files": source_files,
        "models": models,
        "fixtures": fixtures,
        "expected_paddle_commit": json.loads((SERVICE / "runtime-lock.json").read_bytes())[
            "sources"
        ]["paddle"],
        "expected_packages": {
            "paddlepaddle-gpu": "3.3.1",
            "paddleocr": "3.7.0",
            "paddlex": "3.7.2",
        },
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return path, bundle, image_config


def load(release):
    from ocr_service.provision import load_release_bundle

    path, data, _ = release
    path.write_text(json.dumps(data))
    return load_release_bundle(path)


def test_tiny_artifacts_provision_idempotently_without_selected_state(tmp_path, release):
    from ocr_service.provision import provision_bundle

    bundle = load(release)
    state = tmp_path / "state"
    first = provision_bundle(bundle, state)
    before = first.root.stat().st_mtime_ns
    second = provision_bundle(bundle, state)
    assert second.root == first.root and second.root.stat().st_mtime_ns == before
    assert (second.model_dirs[NAMES[0]] / "weights").read_bytes() == b"PP-DocLayoutV3"
    assert second.smoke_path.read_bytes() == b"tiny-private-test-upright"
    assert not (state / "selected").exists()


@pytest.fixture
def official_snapshot(tmp_path, release, monkeypatch):
    from ocr_service import paddle_snapshots

    name = NAMES[-1]
    files = {
        "LICENSE": b"Apache License\nVersion 2.0, January 2004\n",
        "config.json": b"{}",
        "model.safetensors": b"test weights",
    }
    model = paddle_snapshots.MODELS[name]._replace(
        files=tuple(files), weight_sha256=sha(files["model.safetensors"])
    )
    monkeypatch.setitem(paddle_snapshots.MODELS, name, model)
    record = archive(tmp_path, "canonical", files)
    packed = Path(record["source"])
    record.update(
        source=model.source,
        license={"spdx": "Apache-2.0", "text_file": "LICENSE", "redistribution_allowed": True},
        provenance={"url": model.url, "revision": model.revision},
    )
    release[1]["models"][name] = record
    requests = []

    def get(url, timeout):
        assert url.startswith(model.source + "/")
        requests.append(url)
        body = files[url.rsplit("/", 1)[1]]
        response = io.BytesIO(body)
        response.status = 200
        response.headers = HTTPMessage()
        response.headers["Content-Length"] = str(len(body))
        return response

    monkeypatch.setattr(paddle_snapshots, "build_opener", lambda *args: SimpleNamespace(open=get))
    return name, record, files, packed, requests


def test_pinned_snapshot_reproduces_archive_and_reuses_verified_tree(
    tmp_path, release, official_snapshot
):
    from ocr_service.provision import provision_bundle

    name, record, files, packed, requests = official_snapshot
    state = tmp_path / "state"
    bundle = load(release)
    runtime = provision_bundle(bundle, state)
    assert (state / "archives" / record["sha256"]).read_bytes() == packed.read_bytes()
    assert {path.name: path.read_bytes() for path in runtime.model_dirs[name].iterdir()} == files
    assert requests == [record["source"] + "/" + filename for filename in files]

    requests.clear()
    assert provision_bundle(bundle, state) == runtime
    assert requests == []


@pytest.mark.parametrize("tampered", ["file", "tar"])
def test_pinned_snapshot_rejects_tampered_bytes_before_publication(
    tmp_path, release, official_snapshot, tampered
):
    from ocr_service.provision import provision_bundle

    _, record, files, _, _ = official_snapshot
    if tampered == "file":
        files["config.json"] = b"[]"
        error = "Staged tree differs"
    else:
        record["sha256"] = "f" * 64
        error = "Snapshot archive hash/size mismatch"
    state = tmp_path / "state"
    with pytest.raises(ValueError, match=error):
        provision_bundle(load(release), state)
    assert list((state / "bundles").iterdir()) == []
    assert not (state / "archives" / record["sha256"]).exists()
    assert list((state / "archives").glob(".snapshot-*")) == []


@pytest.mark.parametrize("misbound", ["revision", "model", "fixture"])
def test_snapshot_reference_requires_its_exact_approved_model(release, official_snapshot, misbound):
    name, record, _, _, requests = official_snapshot
    if misbound == "revision":
        record["source"] = record["source"].rsplit("/", 1)[0] + "/main"
    elif misbound == "model":
        release[1]["models"][NAMES[0]] = record
    else:
        release[1]["fixtures"]["source"] = record["source"]
    with pytest.raises(ValueError, match="Official model snapshot"):
        load(release)
    assert release[1]["models"][name] is record
    assert requests == []


@pytest.mark.parametrize("native_name", ["paddlepaddle-dcu", "paddlepaddle", "paddlepaddle-gpu"])
def test_rocm_release_provisions_one_native_paddle_identity(tmp_path, release, native_name):
    from ocr_service.provision import provision_bundle

    _, data, _ = release
    data["accelerator"] = "rocm"
    packages = data["expected_packages"]
    packages[native_name] = packages.pop("paddlepaddle-gpu")
    bundle = load(release)
    assert bundle.expected_packages[native_name] == "3.3.1"
    assert provision_bundle(bundle, tmp_path / "state").smoke_path.is_file()


@pytest.mark.parametrize("conflicting", ["paddlepaddle", "paddlepaddle-gpu", "PaddlePaddle_DCU"])
def test_rocm_release_rejects_two_native_paddle_identities(release, conflicting):
    _, data, _ = release
    data["accelerator"] = "rocm"
    packages = data["expected_packages"]
    packages["paddlepaddle-dcu"] = packages.pop("paddlepaddle-gpu")
    packages[conflicting] = "3.3.1"
    with pytest.raises(ValueError, match="one.*Paddle"):
        load(release)


def test_cuda_release_rejects_rocm_wheel_identity(release):
    _, data, _ = release
    packages = data["expected_packages"]
    packages["paddlepaddle-dcu"] = packages.pop("paddlepaddle-gpu")
    with pytest.raises(ValueError, match="accelerator"):
        load(release)


@pytest.mark.parametrize(
    "breakage",
    ["model", "case", "license", "provenance", "tree", "case_hash", "label", "packages", "source"],
)
def test_incomplete_release_rejected(release, breakage):
    _, data, _ = release
    if breakage == "model":
        data["models"].pop(NAMES[0])
    elif breakage == "case":
        data["fixtures"]["cases"].pop()
    elif breakage in {"license", "provenance"}:
        data["models"][NAMES[0]].pop(breakage)
    elif breakage == "tree":
        data["models"][NAMES[0]]["tree_sha256"] = "0" * 64
    elif breakage == "case_hash":
        data["fixtures"]["cases"][0]["sha256"] = "0" * 64
    elif breakage == "label":
        data["fixtures"]["cases"][-1]["required_labels"] = []
    elif breakage == "packages":
        data["expected_packages"] = {}
    else:
        data["source_files"].pop("ocr_service/qualify.py")
    with pytest.raises(ValueError):
        load(release)


@pytest.mark.parametrize(
    "attack", ["hash", "traversal", "symlink", "duplicate", "oversized", "extra", "missing"]
)
def test_archive_integrity_attacks_never_publish(tmp_path, release, attack):
    from ocr_service.provision import provision_bundle

    _, data, _ = release
    asset = data["models"][NAMES[0]]
    path = Path(asset["source"])
    if attack == "hash":
        path.write_bytes(b"corrupt")
    else:
        with tarfile.open(path, "w") as tar:
            for filename, raw in {"weights": b"PP-DocLayoutV3", "LICENSE": b"CC0 test"}.items():
                if attack == "missing" and filename == "weights":
                    continue
                info = tarfile.TarInfo("../escape" if attack == "traversal" else filename)
                info.size = len(raw)
                if attack == "symlink":
                    info.type, info.linkname, info.size = tarfile.SYMTYPE, "/tmp/foreign", 0
                tar.addfile(info, io.BytesIO(raw))
                if attack == "duplicate":
                    tar.addfile(info, io.BytesIO(raw))
            if attack == "extra":
                tar.addfile(tarfile.TarInfo("unexpected"), io.BytesIO())
        asset["sha256"] = sha(path.read_bytes())
        asset["size_bytes"] = path.stat().st_size
        if attack == "oversized":
            asset["unpacked_bytes"] = 1
    state = tmp_path / "state"
    with pytest.raises(ValueError):
        provision_bundle(load(release), state)
    assert not (state / "selected").exists()
    assert not list((state / "bundles").glob("*"))


def test_interrupted_provision_reuses_verified_download_and_rejects_tampered_tree(
    tmp_path, release, monkeypatch
):
    from ocr_service import provision

    bundle = load(release)
    state = tmp_path / "state"
    original = provision.extract_archive
    attempts = []

    def interrupt(*args):
        attempts.append(args)
        raise KeyboardInterrupt

    monkeypatch.setattr(provision, "extract_archive", interrupt)
    with pytest.raises(KeyboardInterrupt):
        provision.provision_bundle(bundle, state)
    assert not (state / "selected").exists()
    # A completed verified archive is reusable without its original source.
    Path(release[1]["models"][NAMES[0]]["source"]).unlink()
    monkeypatch.setattr(provision, "extract_archive", original)
    result = provision.provision_bundle(bundle, state)
    (result.model_dirs[NAMES[0]] / "weights").write_bytes(b"tamper")
    with pytest.raises(ValueError):
        provision.provision_bundle(bundle, state)


def test_canonical_directory_entries_and_nested_files_are_supported(tmp_path, release):
    from ocr_service.provision import provision_bundle

    _, data, _ = release
    asset = archive(tmp_path, "nested", {"nested/weights": b"native bytes", "LICENSE": b"CC0"})
    path = Path(asset["source"])
    with tarfile.open(path, "w") as tar:
        directory = tarfile.TarInfo("nested/")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for name, raw in {"nested/weights": b"native bytes", "LICENSE": b"CC0"}.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    asset.update(sha256=sha(path.read_bytes()), size_bytes=path.stat().st_size)
    data["models"][NAMES[0]] = asset
    result = provision_bundle(load(release), tmp_path / "state")
    assert (result.model_dirs[NAMES[0]] / "nested/weights").read_bytes() == b"native bytes"


def test_existing_model_directory_symlink_is_rejected(tmp_path, release):
    from ocr_service.provision import provision_bundle

    bundle = load(release)
    state = tmp_path / "state"
    result = provision_bundle(bundle, state)
    model = result.model_dirs[NAMES[0]]
    moved = tmp_path / "foreign-model"
    model.rename(moved)
    model.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        provision_bundle(bundle, state)


def test_prepared_installation_reuses_legacy_model_trees_when_archives_were_removed(
    tmp_path, release
):
    from ocr_service.provision import provision_bundle, provision_models

    bundle = load(release)
    state = tmp_path / "state"
    legacy = provision_bundle(bundle, state)
    for asset in bundle.models.values():
        (state / "archives" / asset.sha256).unlink()
        Path(asset.source).unlink()

    models = provision_models(bundle, state)
    for name in NAMES:
        assert (models[name] / "weights").read_bytes() == (
            legacy.model_dirs[name] / "weights"
        ).read_bytes()
    assert models == provision_models(bundle, state)


def test_prepared_retry_resumes_from_already_published_models(tmp_path, release):
    from ocr_service.provision import model_paths, provision_models

    bundle = load(release)
    state = tmp_path / "state"
    last = Path(bundle.models[NAMES[-1]].source)
    data = last.read_bytes()
    last.unlink()
    with pytest.raises(FileNotFoundError):
        provision_models(bundle, state)
    paths = model_paths(bundle, state)
    first = paths[NAMES[0]] / "weights"
    inode = first.stat().st_ino
    assert not paths[NAMES[-1]].exists()
    Path(bundle.models[NAMES[0]].source).unlink()
    last.write_bytes(data)
    assert provision_models(bundle, state) == paths
    assert first.stat().st_ino == inode


@pytest.mark.parametrize("bad_download", [True, False])
def test_only_explicit_repair_replaces_corrupt_owned_model_and_archive(
    tmp_path, release, bad_download
):
    from ocr_service.provision import provision_models

    bundle = load(release)
    state = tmp_path / "state"
    models = provision_models(bundle, state)
    model, healthy = NAMES[:2]
    content = models[model] / "weights"
    content.write_bytes(b"bad model")
    archive = state / "archives" / bundle.models[model].sha256
    archive.write_bytes(b"bad archive")
    original = Path(bundle.models[model].source)
    if bad_download:
        original.write_bytes(b"bad download")
    source_bytes = original.read_bytes()
    healthy_inode = (models[healthy] / "weights").stat().st_ino
    with pytest.raises(ValueError):
        provision_models(bundle, state)
    if bad_download:
        with pytest.raises(ValueError):
            provision_models(bundle, state, repair=True)
        assert content.read_bytes() == b"bad model"
        assert archive.read_bytes() == b"bad archive"
    else:
        assert provision_models(bundle, state, repair=True) == models
        assert content.read_bytes() == model.encode()
        assert archive.read_bytes() == source_bytes
    assert original.read_bytes() == source_bytes
    assert (models[healthy] / "weights").stat().st_ino == healthy_inode


@pytest.mark.parametrize("unsafe", ["archive_symlink", "model_symlink", "extra_file", "mount"])
def test_explicit_repair_leaves_unsafe_cache_and_foreign_file_untouched(
    tmp_path, release, unsafe, monkeypatch
):
    from ocr_service.provision import provision_models

    bundle = load(release)
    state = tmp_path / "state"
    name = NAMES[0]
    model = provision_models(bundle, state)[name]
    owner_file = tmp_path / "outside"
    owner_file.write_bytes(b"private")
    (model / "weights").write_bytes(b"damaged")
    if unsafe == "archive_symlink":
        entry = state / "archives" / bundle.models[name].sha256
        entry.unlink()
        entry.symlink_to(owner_file)
    elif unsafe == "model_symlink":
        entry = model / "LICENSE"
        entry.unlink()
        entry.symlink_to(owner_file)
    elif unsafe == "mount":
        entry = model / "weights"
        entry.write_bytes(b"private")
        # Same-device bind mounts are not identified by Path.is_mount().
        read_bytes = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda self: (
                f"1 2 0:50 /external {entry} rw - ext4 /dev/test rw".encode()
                if self == Path("/proc/self/mountinfo")
                else read_bytes(self)
            ),
        )
    else:
        entry = model / "unexpected"
        entry.write_bytes(b"private")
    with pytest.raises(ValueError):
        provision_models(bundle, state, repair=True)
    assert entry.read_bytes() == owner_file.read_bytes() == b"private"
    if unsafe != "mount":
        assert (model / "weights").read_bytes() == b"damaged"
