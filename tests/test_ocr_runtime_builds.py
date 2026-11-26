"""Static recipe checks do not assert a working GPU binary."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "paddleocr-service"


def test_separate_digest_pinned_native_images_and_cuda_lock():
    for accelerator in ("cuda", "rocm"):
        path = SERVICE / f"Dockerfile.{accelerator}"
        assert path.is_file(), f"{accelerator} image missing"
        text = path.read_text()
        assert re.search(r"FROM \S+@sha256:[0-9a-f]{64}", text)
        assert "COPY ocr_service/" in text
        assert "COPY ocr_contract/" in text
        assert "vision_bill" not in text and "COPY paddleocr-service/" not in text
        assert "/health" in text
    cuda = (SERVICE / "requirements.cuda.lock").read_text()
    if "UNRESOLVED" in cuda:
        assert "paddlepaddle_gpu-3.3.1" in cuda and "cu129" in cuda
        with pytest.raises(ValueError, match="unresolved"):
            load_build_module("resolve_cuda").validate_lock(SERVICE / "requirements.cuda.lock")
    else:
        load_build_module("resolve_cuda").validate_lock(SERVICE / "requirements.cuda.lock")
    assert "requirements.cuda.lock" not in (SERVICE / "Dockerfile.rocm").read_text()


@pytest.mark.parametrize(("accelerator", "stage_count"), [("cuda", 1), ("rocm", 2)])
def test_native_apt_bootstraps_signed_https_snapshot(accelerator, stage_count):
    lock = json.loads((SERVICE / "runtime-lock.json").read_text())
    recipe = (SERVICE / f"Dockerfile.{accelerator}").read_text()
    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64} AS apt-ca$", recipe, re.MULTILINE)
    assert f"FROM {lock['apt_ca_image']} AS apt-ca\n" in recipe
    stages = re.split(r"^FROM ", recipe, flags=re.MULTILINE)
    apt_stages = [stage for stage in stages if "apt-get update" in stage]
    assert len(apt_stages) == stage_count
    ca_copy = "COPY --from=apt-ca /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/"
    snapshot = f"https://snapshot.ubuntu.com/ubuntu/{lock['ubuntu_snapshot']}"
    keyring = "/usr/share/keyrings/ubuntu-archive-keyring.gpg"
    for stage in apt_stages:
        before_update, update, install = stage.partition("apt-get update --error-on=any")
        assert update, "APT must fail on any package-index update error"
        assert ca_copy in before_update
        assert "http://snapshot.ubuntu.com" not in before_update
        for suite in ("noble", "noble-updates", "noble-security"):
            assert f"deb [signed-by={keyring}] {snapshot} {suite} " in before_update
        assert "ca-certificates" in install


def test_rocm_builder_checkpoints_llvm_before_copying_finish_inputs():
    recipe = (SERVICE / "Dockerfile.rocm").read_text().replace("\\\n", " ")
    builder, runtime = recipe.split(" AS native-builder\n", 1)[1].split("\nFROM ", 1)
    apt = next(line for line in builder.splitlines() if "apt-get install" in line)
    for package in ("python3", "tcl"):
        assert package in apt.split()
        assert package not in next(line for line in runtime.splitlines() if "apt-get install" in line).split()
    assert "test -x /usr/bin/python3" in apt and "command -v tclsh" in apt
    assert "libsqlite3-dev" not in builder
    phases = [line for line in builder.splitlines() if line.startswith("RUN ") and "build_therock.sh" in line]
    assert [line.split("build_therock.sh ")[1].split()[0] for line in phases] == ["prepare", "llvm", "finish"]
    assert "sharing=locked" in phases[0]
    assert "rm -rf /opt/native-src" in phases[2]
    before_llvm, after_llvm = builder.split(phases[1])
    for name in ("finish_therock.sh", "verify_rccl_targets.py", "checkpoint_therock.sh", "miopen-sqlite-transient.patch"):
        assert name not in before_llvm
    checkpoints = [
        "COPY build/checkpoint_therock.sh",
        "RUN bash /opt/ocr/build/checkpoint_therock.sh rccl",
        "RUN bash /opt/ocr/build/checkpoint_therock.sh miopen-configure",
        "COPY build/miopen-sqlite-transient.patch",
        "RUN bash /opt/ocr/build/checkpoint_therock.sh miopen",
        "COPY build/finish_therock.sh",
        phases[2],
    ]
    lines = after_llvm.splitlines()
    offsets = [
        next(i for i, line in enumerate(lines) if line == prefix or line.startswith(prefix + " "))
        for prefix in checkpoints
    ]
    assert offsets == sorted(offsets)


def test_rocm_exact_pins_full_sdk_targets_and_no_spoofing():
    path = SERVICE / "build/build_rocm.sh"
    assert path.is_file(), "native source build missing"
    script = path.read_text()
    lock = json.loads((SERVICE / "runtime-lock.json").read_text())
    assert lock["sources"]["paddle"] == "7688495538f4d6c1893f084dd238a402e8f68ab6"
    assert lock["sources"]["therock"] == "b324869fb3116353ac7eec5e6926aa3d452ac987"
    assert lock["rdna2_targets"] == ["gfx1030", "gfx1031", "gfx1032", "gfx1034"]
    sdk = (SERVICE / "build/build_therock.sh").read_text() + (SERVICE / "build/finish_therock.sh").read_text()
    assert "THEROCK_ENABLE_ALL=ON" in sdk and "--target dist-rocm" in sdk
    assert "THEROCK_ENABLE_WSL_ROCDXG=ON" in sdk and '-DTHEROCK_WSL_WIN_SDK=$WIN_SDK' in sdk
    assert lock["sources"]["win32metadata"] in sdk and lock["wkmi_dvc_md5"] in sdk
    assert 'sha256sum "$SDK/lib/librocdxg.so.1"' in sdk
    assert 'CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" cmake --build "$THEROCK_BUILD"' in sdk
    assert "HSA_OVERRIDE_GFX_VERSION" not in script + sdk
    assert "WITH_FLASHATTN=OFF" in script
    assert "git apply --check" in script
    patch = (SERVICE / "build/hip-rdna2-v1.patch").read_text()
    assert "+  list(APPEND HIP_HCC_FLAGS --offload-arch=${target})" in patch
    assert "+  list(APPEND HIP_CLANG_FLAGS --offload-arch=${target})" in patch
    assert "WITH_ROCM AND WITH_FLASHATTN" in patch


@pytest.mark.parametrize("existing", ["TheRock", "Paddle", "therock-build", "win32metadata"])
def test_rocm_rejects_existing_source_before_starting_build(tmp_path, existing):
    (tmp_path / existing).mkdir()
    result = subprocess.run(
        ["bash", SERVICE / "build/build_rocm.sh"],
        env={**os.environ, "SOURCE_ROOT": str(tmp_path), "AMDGPU_TARGETS": "invalid"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert f"Native build path already exists: {tmp_path / existing}" in result.stderr


def test_therock_preparation_inventories_gitlinks_omitted_by_upstream_fetch(tmp_path):
    env = {**os.environ, "GIT_ALLOW_PROTOCOL": "file"}

    def git(repo, *args):
        return subprocess.check_output(
            ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
            env=env,
            text=True,
        ).strip()

    def repo(name):
        path = tmp_path / name
        path.mkdir()
        git(path, "init", "--quiet")
        git(path, "config", "uploadpack.allowFilter", "true")
        git(path, "commit", "--quiet", "--allow-empty", "-m", f"Pin {name}")
        return path

    nested = repo("fin")
    libraries = repo("libraries")
    (libraries / "native-option").write_text("upstream\n")
    git(libraries, "add", "native-option")
    git(libraries, "commit", "--quiet", "-m", "Add upstream build input")
    docs = repo("docs")
    (docs / "docs-only-marker").write_text("This file should never be fetched.\n")
    git(docs, "add", "docs-only-marker")
    git(docs, "commit", "--quiet", "-m", "Seal docs marker")
    upstream = repo("upstream")
    git(libraries, "submodule", "add", "--quiet", nested.as_uri(), "projects/miopen/fin")
    git(libraries, "commit", "--quiet", "-am", "Pin nested source")
    git(upstream, "submodule", "add", "--quiet", libraries.as_uri(), "rocm-libraries")
    git(upstream, "submodule", "add", "--quiet", docs.as_uri(), "rocm-docs")

    # Model TheRock's pinned top-level fetch and patches without network or a compiler.
    fetch_sources = upstream / "build_tools/fetch_sources.py"
    fetch_sources.parent.mkdir()
    fetch_sources.write_text(
        "import subprocess\n"
        "from pathlib import Path\n"
        "subprocess.run(['git', 'submodule', 'update', '--init', '--', 'rocm-libraries'], check=True)\n"
        "subprocess.run(['git', '-C', 'rocm-libraries', '-c', 'user.name=Test', '-c', "
        "'user.email=test@example.com', 'commit', '--quiet', '--allow-empty', '-m', 'Patch'], check=True)\n"
        "Path('.rocm-libraries.smrev').write_text('patched checkout\\n')\n"
    )
    git(upstream, "add", "build_tools")
    git(upstream, "commit", "--quiet", "-m", "Pin TheRock inputs")
    pin = git(upstream, "rev-parse", "HEAD")
    old = tmp_path / "old-checkout"
    git(tmp_path, "clone", "--quiet", str(upstream), str(old))
    output = tmp_path / "sources.json"
    subprocess.run([sys.executable, "build_tools/fetch_sources.py"], cwd=old, env=env, check=True)
    original_inventory = subprocess.run(
        [sys.executable, SERVICE / "build/source_inventory.py", old, output],
        capture_output=True,
        text=True,
        check=False,
    )
    assert original_inventory.returncode != 0
    missing_gitlink = git(docs, "rev-parse", "HEAD") + " rocm-docs"
    assert f"Uninitialized recursive source gitlink: {missing_gitlink}" in original_inventory.stderr
    assert not output.exists()

    (upstream / "later.txt").write_text("Only present on the newer branch tip\n")
    git(upstream, "add", "later.txt")
    git(upstream, "commit", "--quiet", "-m", "Move the default branch after the pin")
    new = tmp_path / "new-checkout"
    git(tmp_path, "init", "--quiet", str(new))
    git(new, "remote", "add", "origin", upstream.as_uri())
    local_patch = tmp_path / "native.patch"
    local_patch.write_text(
        "diff --git a/native-option b/native-option\n--- a/native-option\n"
        "+++ b/native-option\n@@ -1 +1 @@\n-upstream\n+local\n"
    )
    subprocess.run(
        ["bash", SERVICE / "build/prepare_therock.sh", new, output, "4", pin,
         "rocm-libraries", local_patch], env=env, check=True,
    )

    assert git(new, "rev-parse", "HEAD") == pin
    assert git(new, "rev-list", "--count", "HEAD") == "1"
    assert not list((new / ".git/objects/pack").glob("*.promisor"))
    assert (new / "build_tools/fetch_sources.py").read_bytes() == fetch_sources.read_bytes()
    original_gitlinks = (tmp_path / "therock-original-gitlinks.txt").read_text().splitlines()
    assert f"160000 {git(libraries, 'rev-parse', 'HEAD')} 0\trocm-libraries" in original_gitlinks
    docs_commit = git(docs, "rev-parse", "HEAD")
    assert f"160000 {docs_commit} 0\trocm-docs" in original_gitlinks
    assert not (new / "rocm-docs/docs-only-marker").exists()
    assert not (new / ".git/modules/rocm-docs").exists()
    for path in ("rocm-libraries", "rocm-libraries/projects/miopen/fin"):
        assert git(new / path, "config", "--get", "remote.origin.partialclonefilter") == "blob:none"
    report = json.loads(output.read_text())
    assert "rocm-docs" not in report["checkouts"]
    assert report["omitted_root_gitlinks"] == {"rocm-docs": docs_commit}
    assert report["checkouts"]["rocm-libraries/projects/miopen/fin"]["head"] == git(nested, "rev-parse", "HEAD")
    assert report["checkouts"]["rocm-libraries"]["head"] != git(libraries, "rev-parse", "HEAD")
    assert report["checkouts"]["rocm-libraries"]["head"] == git(new / "rocm-libraries", "rev-parse", "HEAD")
    assert (new / "rocm-libraries/native-option").read_text() == "local\n"
    assert "+local" in report["checkouts"]["rocm-libraries"]["patch"]
    assert ".rocm-libraries.smrev" in report["patches_and_smrev"]

    inventory = [sys.executable, SERVICE / "build/source_inventory.py", new, tmp_path / "invalid.json",
                 "--omit-root-gitlink", "rocm-docs"]
    git(new, "update-index", "--cacheinfo", f"160000,{git(libraries, 'rev-parse', 'HEAD')},rocm-docs")
    changed = subprocess.run(inventory, capture_output=True, text=True, check=False)
    assert changed.returncode != 0
    assert "differs from the pinned root tree" in changed.stderr
    git(new, "reset", "HEAD", "--", "rocm-docs")
    git(new / "rocm-libraries", "submodule", "deinit", "-f", "projects/miopen/fin")
    incomplete = subprocess.run(inventory, capture_output=True, text=True, check=False)
    assert incomplete.returncode != 0
    assert f"Uninitialized recursive source gitlink: {git(nested, 'rev-parse', 'HEAD')} rocm-libraries/projects/miopen/fin" in incomplete.stderr
    git(new, "submodule", "deinit", "-f", "rocm-libraries")
    missing_root = subprocess.run(inventory, capture_output=True, text=True, check=False)
    assert missing_root.returncode != 0
    assert f"Uninitialized recursive source gitlink: {git(libraries, 'rev-parse', 'HEAD')} rocm-libraries" in missing_root.stderr
    assert not (tmp_path / "invalid.json").exists()



def test_sidecar_independent_context_cache_and_internal_readiness():
    base = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = base["services"]["paddleocr-service"]
    assert service["build"]["context"] == "./paddleocr-service"
    assert service["build"]["dockerfile"] == "Dockerfile"
    assert not service.get("ports")
    assert service["environment"]["PADDLE_PDX_CACHE_HOME"] == "/var/cache/paddlex"
    assert any("/var/cache/paddlex" in volume for volume in service["volumes"])
    assert "/health" in str(service["healthcheck"])
    ignore = (SERVICE / ".dockerignore").read_text().splitlines()
    assert "*" in ignore and "!ocr_contract/*.py" in ignore
    assert "!ocr_service/*.py" in ignore and "!**" not in ignore
    for accelerator in ("cuda", "rocm"):
        overlay = yaml.safe_load((ROOT / f"docker-compose.{accelerator}.yml").read_text())
        ocr = overlay["services"]["paddleocr-service"]
        assert ocr["build"]["context"] == "./paddleocr-service"
        assert ocr["build"]["dockerfile"] == f"Dockerfile.{accelerator}"
        assert not ocr.get("ports") and not ocr.get("privileged")
    rocm = yaml.safe_load((ROOT / "docker-compose.rocm.yml").read_text())["services"][
        "paddleocr-service"
    ]
    assert "/dev/kfd:/dev/kfd" in rocm["devices"]
    assert all("/dev/dri:" not in device for device in rocm["devices"])


def load_build_module(name):
    import importlib.util

    path = SERVICE / "build" / f"{name}.py"
    assert path.is_file(), f"{name} tool missing"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_resolver_selects_only_exact_official_cp312_artifact():
    module = load_build_module("resolve_cuda")
    official = "https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    html = f'<a href="{official}">wheel</a><a href="https://evil.invalid/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl">other</a>'
    assert module.select_wheel(html) == official
    import pytest

    assert module.select_wheel(html + f'<a href="{official}?other">untrusted</a>') == official
    with pytest.raises(ValueError):
        module.select_wheel(html.replace("3.3.1", "3.3.0"))


def test_cuda_resolver_selects_only_requested_channel():
    module = load_build_module("resolve_cuda")
    wheel = "paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    legacy = f"https://paddle-whl.bj.bcebos.com/stable/cu126/{wheel}"
    blackwell = f"https://paddle-whl.cdn.bcebos.com/stable/cu129/paddlepaddle-gpu/{wheel}"
    html = f'<a href="{legacy}">legacy</a><a href="{blackwell}#sha256=abc">candidate</a>'
    assert module.select_wheel(html) == blackwell
    assert module.select_wheel(html, "cu126") == legacy
    with pytest.raises(ValueError, match="cu129"):
        module.select_wheel(f'<a href="{legacy}">legacy</a>', "cu129")
    with pytest.raises(ValueError, match="channel"):
        module.select_wheel(html, "cu128")


@pytest.mark.parametrize(
    "untrusted",
    [
        "http://paddle-whl.cdn.bcebos.com/stable/cu129/",
        "https://user:pass@paddle-whl.cdn.bcebos.com/stable/cu129/",
        "https://paddle-whl.cdn.bcebos.com:443/stable/cu129/",
        "https://paddle-whl.cdn.bcebos.com/stable/cu129/../cu126/",
        "https://paddle-whl.cdn.bcebos.com/stable/cu129/%2e%2e/cu126/",
        "https://paddle-whl.cdn.bcebos.com/stable/cu126/cu129/",
        "https://paddle-whl.cdn.bcebos.com/experimental/cu129/",
        "https://paddle-whl.cdn.bcebos.com.evil.invalid/stable/cu129/",
    ],
)
def test_cuda_resolver_rejects_untrusted_wheel_links(untrusted):
    module = load_build_module("resolve_cuda")
    wheel = "paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    with pytest.raises(ValueError, match="cu129"):
        module.select_wheel(f'<a href="{untrusted}{wheel}">wheel</a>', "cu129")


def test_cuda_resolver_refuses_redirect_before_sending_followup_request():
    from http.client import HTTPMessage
    from io import BytesIO
    from urllib.request import Request

    module = load_build_module("resolve_cuda")
    index = "https://www.paddlepaddle.org.cn/packages/stable/cu129/paddlepaddle-gpu/"
    with pytest.raises(ValueError, match="redirect"):
        module.NoRedirects().redirect_request(
            Request(index), BytesIO(), 302, "Found", HTTPMessage(), "https://evil.invalid/index"
        )


def test_unresolved_cuda_lock_is_explicit_and_build_validation_rejects_it(tmp_path):
    module = load_build_module("resolve_cuda")
    import pytest

    lock = tmp_path / "requirements.cuda.lock"
    lock.write_text("# UNRESOLVED: official wheel access required\n")
    with pytest.raises(ValueError, match="unresolved"):
        module.validate_lock(lock)


def test_kernel_inspector_rejects_missing_required_targets():
    module = load_build_module("inspect_kernels")
    import pytest

    with pytest.raises(ValueError, match="missing"):
        module.require_targets({"gfx1030"}, {"gfx1030", "gfx1031"})


def test_opaque_kernel_payload_stays_unverified_not_missing_or_supported(tmp_path, monkeypatch):
    module = load_build_module("inspect_kernels")
    from types import SimpleNamespace

    lib = tmp_path / "librocblas.so"
    lib.write_bytes(b"\x7fELFopaque-packed-library")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    evidence = module.inventory([lib], tmp_path / "llvm-objdump")
    assert evidence["offline_status"] == "unverified"
    assert evidence["targets"] == [] and str(lib) in evidence["files"]
    import pytest

    with pytest.raises(ValueError, match="library"):
        module.inventory([], tmp_path / "llvm-objdump")


def test_native_compile_inventory_rejects_extra_or_missing_targets(tmp_path):
    module = load_build_module("inspect_kernels")
    import pytest

    commands = tmp_path / "compile_commands.json"
    commands.write_text(json.dumps([{"command": "hipcc -c x --offload-arch=gfx1030"}]))
    with pytest.raises(ValueError, match="target"):
        module.validate_compile_targets(commands)
    commands.write_text(
        json.dumps(
            [
                {
                    "command": "hipcc -c x "
                    + " ".join("--offload-arch=" + target for target in sorted(module.TARGETS))
                }
            ]
        )
    )
    module.validate_compile_targets(commands)


def test_build_bootstrap_tools_are_hashed_before_no_isolation_source_builds():
    bootstrap = SERVICE / "requirements.bootstrap.lock"
    assert bootstrap.is_file(), "locked build bootstrap missing"
    assert "setuptools==" in bootstrap.read_text() and "--hash=sha256:" in bootstrap.read_text()
    rocm = (SERVICE / "Dockerfile.rocm").read_text()
    assert "--only-binary=:all: -r /opt/locks/requirements.bootstrap.lock" in rocm
    assert "--no-build-isolation -r /opt/locks/requirements.build.lock" in rocm


def test_cuda_stages_only_install_the_locked_runtime_into_the_final_venv():
    recipe = (SERVICE / "Dockerfile.cuda").read_text()
    base, rest = recipe.split("FROM runtime-base AS cuda-builder")
    builder, runtime = rest.split("FROM runtime-base\n")
    assert "requirements.build.lock" not in recipe
    assert "build-essential" not in base
    assert "libgomp1" in base
    assert "--only-binary=:all: -r /opt/locks/requirements.cuda-build.lock" in builder
    assert "--root /runtime --ignore-installed --require-hashes --no-deps" in builder
    assert "COPY --from=cuda-builder /runtime/opt/venv/ /opt/venv/" in runtime
    assert "COPY --from=cuda-builder /opt/locks/ /opt/locks/" in runtime
    assert "COPY --from=cuda-builder /opt/ocr/build/resolve_cuda.py" in runtime
    assert "RUN --network=none python -m pip check" in runtime


def test_runtime_lock_hashes_match_committed_inputs():
    import hashlib

    lock = json.loads((SERVICE / "runtime-lock.json").read_text())
    assert {"ocr_contract/limits.py", "ocr_contract/wire_json.py"} <= lock["input_sha256"].keys()
    for name, digest in lock["input_sha256"].items():
        assert hashlib.sha256((SERVICE / name).read_bytes()).hexdigest() == digest, name


def test_sidecar_directory_negations_do_not_reinclude_private_descendants():
    lines = (SERVICE / ".dockerignore").read_text().splitlines()
    for directory in ("ocr_contract", "ocr_service", "build"):
        index = lines.index("!" + directory + "/")
        assert lines[index + 1] == directory + "/**", (
            "Docker negated directory otherwise re-includes all descendants"
        )


def test_kernel_targets_come_from_metadata_not_object_filenames(tmp_path, monkeypatch):
    module = load_build_module("inspect_kernels")
    from types import SimpleNamespace

    lib = tmp_path / "gfx1030.co"
    lib.write_bytes(b"\x7fELF")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=str(lib) + ": file format elf64-amdgpu\n"),
    )
    assert module.inventory([lib], tmp_path / "llvm-objdump")["offline_status"] == "unverified"
    output = "\n".join("arch            " + target for target in sorted(module.TARGETS))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
    assert module.inventory([lib], tmp_path / "llvm-objdump")["offline_status"] == "verified"


def cuda_lock_fixture(tmp_path):
    import hashlib

    module = load_build_module("resolve_cuda")
    lock = tmp_path / "requirements.cuda.lock"
    (tmp_path / "requirements.runtime.lock").write_bytes(
        (SERVICE / "requirements.runtime.lock").read_bytes()
    )
    url = "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    digest = hashlib.sha256(b"synthetic wheel test bytes").hexdigest()
    native = f"paddlepaddle-gpu @ {url} \\\n    --hash=sha256:{digest}\n"
    text = (SERVICE / "requirements.runtime.lock").read_text() + "\n" + native
    data = {
        "index": module.INDEXES["cu126"],
        "index_sha256": hashlib.sha256(b"test index").hexdigest(),
        "wheel_url": url,
        "wheel_sha256": digest,
        "metadata_sha256": hashlib.sha256(b"test metadata").hexdigest(),
        "metadata_name": "paddlepaddle-gpu",
        "metadata_version": "3.3.1",
        "requires_dist": ["numpy>=1.21", "opt_einsum==3.3.0"],
        "platform": "linux-x86_64-cp312",
    }

    def write(contents=text, fields=None):
        lock.write_text(contents)
        attested = {**data, "lock_sha256": hashlib.sha256(contents.encode()).hexdigest()}
        if fields is not None:
            attested = fields(attested)
        lock.with_suffix(".attestation.json").write_text(json.dumps(attested))
        return lock

    return module, write, text, native


def test_cuda_validator_accepts_complete_resolver_shaped_executable_lock(tmp_path):
    module, write, _, _ = cuda_lock_fixture(tmp_path)
    assert module.validate_lock(write())["metadata_version"] == "3.3.1"


def test_cuda_validator_binds_channel_to_index_and_lock(tmp_path):
    module, write, text, _ = cuda_lock_fixture(tmp_path)
    candidate = text.replace("cu126/", "cu129/")

    def attestation(data):
        return {
            **data,
            "cuda_channel": "cu129",
            "index": data["index"].replace("cu126/", "cu129/"),
            "wheel_url": data["wheel_url"].replace("cu126/", "cu129/"),
        }

    assert module.validate_lock(write(candidate, attestation))["cuda_channel"] == "cu129"
    with pytest.raises(ValueError, match="attestation mismatch"):
        module.validate_lock(write(candidate, attestation), channel="cu126")
    for field, value in [
        ("cuda_channel", "cu126"),
        ("cuda_channel", "cu128"),
        ("index", module.INDEXES["cu126"]),
    ]:
        with pytest.raises(ValueError, match="attestation mismatch"):
            module.validate_lock(
                write(
                    candidate,
                    lambda data, field=field, value=value: {**attestation(data), field: value},
                )
            )
    with pytest.raises(ValueError, match="attestation mismatch"):
        module.validate_lock(write(candidate, lambda data: {**data, "cuda_channel": "cu129"}))
    with pytest.raises(ValueError, match="attestation mismatch"):
        module.validate_lock(
            write(
                candidate,
                lambda data: {k: v for k, v in attestation(data).items() if k != "cuda_channel"},
            )
        )
    with pytest.raises(ValueError, match="differs"):
        module.validate_lock(write(text, attestation))


def test_cuda_validator_rejects_comment_only_and_shadowed_native_requirements(tmp_path):
    import pytest

    module, write, text, native = cuda_lock_fixture(tmp_path)
    commented = "\n".join("# " + line for line in native.splitlines()) + "\n"
    variants = [
        commented,
        text.replace(native, commented),
        text.replace(native, commented + "paddlepaddle==3.3.1 --hash=sha256:" + "b" * 64 + "\n"),
        text.replace(native, commented + native.replace("cu126/", "cu128/")),
        text + native.replace("paddlepaddle-gpu", "PaddlePaddle_GPU"),
        text.replace("paddlepaddle-gpu @", "paddlepaddle-gpu[other] @"),
        text.replace(native, native.replace(" \\\n", ' ; python_version < "1" \\\n')),
    ]
    for contents in variants:
        with pytest.raises(ValueError):
            module.validate_lock(write(contents))


def test_cuda_validator_rejects_incomplete_attestation_or_unclosed_lock(tmp_path):
    import pytest

    module, write, text, _ = cuda_lock_fixture(tmp_path)
    for field in ("index_sha256", "metadata_sha256", "requires_dist", "platform"):
        with pytest.raises(ValueError):
            module.validate_lock(
                write(
                    fields=lambda data, field=field: {k: v for k, v in data.items() if k != field}
                )
            )
    for field, value in [
        ("index_sha256", "not-a-hash"),
        ("requires_dist", "numpy"),
        ("platform", "windows"),
        ("requires_dist", ["absent-gpu-dependency==1"]),
    ]:
        with pytest.raises(ValueError):
            module.validate_lock(
                write(fields=lambda data, field=field, value=value: {**data, field: value})
            )
    variants = [
        text.replace("numpy==2.2.6", "numpy==1.0.0"),
        text.replace("pillow-heif==1.7.0", "unrelated==1.7.0"),
        text + "\n-r another.lock\n",
        text + "\n--index-url https://evil.invalid\n",
        text + "\nunpinned-package\n",
        text + "\nno-hash==1\n",
    ]
    for contents in variants:
        with pytest.raises(ValueError):
            module.validate_lock(write(contents))


def test_native_release_source_identity_covers_shared_ocr_limits():
    from hashlib import sha256

    from ocr_service.provision import source_identities

    source = source_identities()
    for name in ("ocr_contract/limits.py", "ocr_contract/wire_json.py"):
        assert source[name] == sha256((SERVICE / name).read_bytes()).hexdigest()


def test_verified_cuda_cache_reuse_and_pip_install_input(tmp_path, monkeypatch):
    module, write, _, _ = cuda_lock_fixture(tmp_path)
    lock = write()
    data = module.validate_lock(lock)
    original = lock.read_bytes()
    calls = []

    def download(url, target):
        calls.append(url)
        target.write_bytes(b"synthetic wheel test bytes")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache with spaces"))
    monkeypatch.setattr(module, "download_wheel", download)
    context = module.wheel_context(data)
    assert context.name == data["wheel_sha256"]
    assert module.wheel_context(data) == context
    assert calls == [data["wheel_url"]]

    (wheel,) = context.iterdir()
    output = tmp_path / "install.lock"
    module.install_requirements(lock, data, context, output)
    entries = module.executable_requirements(output.read_text())
    expected = module.executable_requirements(original.decode())
    local, hashes = entries.pop("paddlepaddle-gpu")
    _, original_hashes = expected.pop("paddlepaddle-gpu")
    assert entries == expected and local.url == wheel.as_uri() and hashes == original_hashes
    assert lock.read_bytes() == original

    wheel.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.install_requirements(lock, data, context, output)
    module.wheel_context(data)
    assert len(calls) == 2 and module.sha256(wheel) == data["wheel_sha256"]
    (context / "unrelated").write_text("extra file")
    with pytest.raises(ValueError, match="Unexpected files"):
        module.wheel_context(data)
    fallback = tmp_path / "empty"
    fallback.mkdir()
    module.install_requirements(lock, data, fallback, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    ("body", "code", "downloads"),
    [
        (b"synthetic wheel test bytes", 0, 0),
        (b"wrong", 0, 1),
        (b"", 1, 1),
    ],
)
def test_old_image_cache_is_a_verified_optional_seed(tmp_path, monkeypatch, body, code, downloads):
    from types import SimpleNamespace

    module, write, _, _ = cuda_lock_fixture(tmp_path)
    evidence = module.validate_lock(write())
    image = "sha256:" + "f" * 64
    seen = []

    def docker(arguments, **kwargs):
        assert arguments[-2] == image and arguments[-1].startswith("/root/.cache/pip/http-v2/")
        assert "--network=none" in arguments and "--read-only" in arguments
        assert "--pull=never" in arguments and "--log-driver=none" in arguments
        kwargs["stdout"].write(body)
        return SimpleNamespace(returncode=code)

    def download(url, candidate):
        seen.append(url)
        candidate.write_bytes(b"synthetic wheel test bytes")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(module.subprocess, "run", docker)
    monkeypatch.setattr(module, "download_wheel", download)
    context = module.wheel_context(evidence, prior_image=image)
    assert (
        len(seen) == downloads
        and module.sha256(next(context.iterdir())) == evidence["wheel_sha256"]
    )
    with pytest.raises(ValueError, match="exact Docker image ID"):
        module.seed_from_previous_image("previous:latest", evidence, tmp_path / "ignored")


def test_cuda_resolution_is_identical_across_scratch_paths(tmp_path, monkeypatch):
    import io
    import shutil
    import zipfile
    from types import SimpleNamespace

    module = load_build_module("resolve_cuda")
    wheelhouse = tmp_path / "test-wheels"
    wheelhouse.mkdir()
    dependency = wheelhouse / "locked_fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(dependency, "w") as archive:
        archive.writestr(
            "locked_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: locked-fixture\nVersion: 1.0\n",
        )
        archive.writestr(
            "locked_fixture-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n"
        )
        archive.writestr("locked_fixture-1.0.dist-info/RECORD", "")
    filename = "paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    official = "https://paddle-whl.bj.bcebos.com/stable/cu129/" + filename
    wheel = tmp_path / filename
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "paddlepaddle_gpu-3.3.1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: paddlepaddle-gpu\nVersion: 3.3.1\nRequires-Dist: locked-fixture==1.0\n",
        )
    monkeypatch.setattr(
        module.urllib.request,
        "build_opener",
        lambda *_: SimpleNamespace(
            open=lambda *_args, **_kw: io.BytesIO(f'<a href="{official}">wheel</a>'.encode())
        ),
    )
    monkeypatch.setattr(
        module, "download_wheel", lambda url, target: shutil.copyfile(wheel, target)
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_FIND_LINKS", str(wheelhouse))
    outputs = []
    for name in ["first", "second"]:
        service = tmp_path / name
        (service / "build").mkdir(parents=True)
        (service / "build/runtime.in").write_text("locked-fixture==1.0\n")
        (service / "requirements.runtime.lock").write_text(
            f"locked-fixture==1.0 --hash=sha256:{module.sha256(dependency)}\n"
        )
        module.resolve(service)
        lock = service / "requirements.cuda.lock"
        evidence = module.validate_lock(lock)
        assert (module.wheel_context(evidence) / filename).read_bytes() == wheel.read_bytes()
        outputs.append((lock.read_bytes(), lock.with_suffix(".attestation.json").read_bytes()))
    assert outputs[0] == outputs[1]
    assert str(tmp_path).encode() not in outputs[0][0]


@pytest.mark.parametrize(("transport", "phased"), [
    ("kfd", False), ("kfd", True), ("wsl", False), ("dual", False), ("unknown", False),
])
def test_therock_transport_controls_native_sources_before_build(tmp_path, transport, phased):
    import shutil

    recipe = tmp_path / "recipe"
    recipe.mkdir()
    for name in ("build_therock.sh", "finish_therock.sh", "checkpoint_therock.sh", "build_jobs.py", "miopen-sqlite-transient.patch"):
        shutil.copy(SERVICE / "build" / name, recipe)
    (recipe / "prepare_therock.sh").write_text("exit 0\n")
    for check in ("verify_therock_targets", "verify_rccl_targets"):
        (recipe / f"{check}.py").write_text(
            "import os, sys\n"
            f"with open(os.environ['LOG'], 'a') as log: print('{check}', *sys.argv[1:], file=log)\n"
            f"if os.getenv('FAIL_GUARD') == '{check}': sys.exit(46)\n"
        )
    (tmp_path / "ocr_service").mkdir()
    shutil.copy(SERVICE / "ocr_service/targets.py", tmp_path / "ocr_service")
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "commands.log"
    for name, body in {
        "git": """
if [[ $1 == init ]]; then
    mkdir -p "${@: -1}/dockerfiles"
    touch "${@: -1}/dockerfiles/install_pinned_patchelf.sh"
fi
if [[ "$*" == *win32metadata*"fetch --depth"* ]]; then exit 44; fi
""",
        "cmake": """
if [[ "$1" == -S ]]; then mkdir -p "$4"; touch "$4/CMakeCache.txt"; fi
if [[ "$1" == --build ]]; then
    echo "targets=$VISION_BILL_AMDGPU_TARGETS jobs=$CMAKE_BUILD_PARALLEL_LEVEL" >> "$LOG"
    if [[ "${@: -1}" == "$FAIL_TARGET" ]]; then exit 47; fi
    if [[ "${@: -1}" == dist-rocm && -z "$TEST_INSTALL" ]]; then exit 45; fi
fi
if [[ "$1" == --install ]]; then
    sdk="${@: -1}"
    mkdir -p "$sdk/bin" "$sdk/lib/cmake/hip/FindHIP"
    touch "$sdk/bin/"{hipcc,hipconfig,hipcc_cmake_linker_helper}
    chmod +x "$sdk/bin/hipcc_cmake_linker_helper"
    touch "$sdk/lib/cmake/hip/FindHIP.cmake" "$sdk/lib/cmake/hip/FindHIP/"{run_hipcc,run_make2cmake}.cmake
fi
""",
    }.items():
        executable = commands / name
        executable.write_text(f'#!/bin/bash\necho "{name} $*" >> "$LOG"\n{body}\n')
        executable.chmod(0o755)
    evidence = tmp_path / "evidence"

    def run(*phases, **overrides):
        return subprocess.run(
            ["bash", recipe / "build_therock.sh", *phases],
            env={
                **os.environ, "PATH": f"{commands}:{os.environ['PATH']}", "LOG": str(log),
                "SOURCE_ROOT": str(tmp_path / "source"), "EVIDENCE": str(evidence),
                "OCR_ROCM_TRANSPORT": transport, "AMDGPU_TARGETS": "gfx1100;gfx1032",
                "BUILD_JOBS": "1", "SDK": str(tmp_path / "sdk"), **overrides,
            },
            check=False, capture_output=True, text=True,
        )

    if phased:
        prepared = run("prepare")
        assert prepared.returncode == 0, prepared.stderr
        assert "cmake --build" not in log.read_text()
        for changed in ({"OCR_ROCM_TRANSPORT": "wsl"}, {"AMDGPU_TARGETS": "gfx1100"}):
            assert run("llvm", **changed).returncode != 0
        assert "cmake --build" not in log.read_text()
        llvm = run("llvm")
        assert llvm.returncode == 0, llvm.stderr
        assert "--target amd-llvm" in log.read_text()
        assert "--target rccl+configure" not in log.read_text()
        blocked_log = tmp_path / "rejected-rccl.log"
        assert run("finish", FAIL_GUARD="verify_rccl_targets", LOG=str(blocked_log)).returncode == 46
        assert "--target dist-rocm" not in blocked_log.read_text()
        interrupted = run("finish", FAIL_TARGET="MIOpen")
        assert interrupted.returncode == 47
        marker_root = tmp_path / "source/therock-build"
        assert (marker_root / ".vision-bill-rccl").is_file()
        assert (marker_root / ".vision-bill-miopen-configure").is_file()
        assert not (marker_root / ".vision-bill-miopen").exists()
        result = run("finish")
        assert (marker_root / ".vision-bill-miopen").is_file()
        output = prepared.stderr + llvm.stderr + interrupted.stderr + result.stderr
        installed = run("finish", TEST_INSTALL="1", LOG=str(tmp_path / "installed.log"))
        assert installed.returncode == 0, installed.stderr
        assert re.search(r"ocr_rocm_phase phase=therock_finish elapsed_s=[0-9.]+", installed.stderr)
        assert installed.stderr.count("phase=sdk_compile ") == 1
        resumed = (tmp_path / "installed.log").read_text()
        assert "--target dist-rocm" in resumed and "--target MIOpen" not in resumed
        assert "--target rccl" not in resumed and "--target amd-llvm" not in resumed
        patch = recipe / "miopen-sqlite-transient.patch"
        patch.write_bytes(patch.read_bytes() + b"\n")
        mismatch = run("finish", LOG=str(tmp_path / "mismatch.log"))
        assert mismatch.returncode != 0 and "Mismatched miopen checkpoint" in mismatch.stderr
        assert "phase=finish " in (evidence / "therock-checkpoints.txt").read_text()
    else:
        result = run()
        output = result.stderr
    if transport == "unknown":
        assert "Invalid OCR_ROCM_TRANSPORT" in result.stderr
        assert not log.exists()
        return
    assert (evidence / "rocm-transport.txt").read_text() == transport + "\n"
    calls = log.read_text()
    assert re.search(r"ocr_rocm_phase phase=therock_sources elapsed_s=[0-9.]+", output)
    if transport in {"wsl", "dual"}:
        assert result.returncode == 44 and "win32metadata" in calls
    else:
        assert result.returncode == 45 and "win32metadata" not in calls
        assert re.search(r"ocr_rocm_phase phase=sdk_compile elapsed_s=[0-9.]+", output)
        assert re.search(r"ocr_rocm_phase phase=llvm_compile elapsed_s=[0-9.]+", output)
        assert re.search(r"ocr_rocm_phase phase=rccl_configure elapsed_s=[0-9.]+", output)
        checkpoints = (evidence / "therock-checkpoints.txt").read_text()
        for phase in ("prepare", "llvm"):
            assert re.search(rf"phase={phase} path=\S+/therock-build allocated_bytes=\d+", checkpoints)
        for scope in ("", "DIST_", "TEST_"):
            assert f"-DTHEROCK_{scope}AMDGPU_FAMILIES= " in calls
            assert f"-DTHEROCK_{scope}AMDGPU_TARGETS=gfx1032;gfx1100 " in calls
        stages = ["verify_therock_targets", "--target amd-llvm", "--target rccl+configure",
                  "verify_rccl_targets", "--target rccl\n", "--target MIOpen+configure",
                  "miopen-sqlite-transient.patch", "--target MIOpen\n", "--target dist-rocm"]
        assert [calls.index(stage) for stage in stages] == sorted(calls.index(stage) for stage in stages)
        assert calls.count("targets=gfx1032;gfx1100 jobs=1") == (7 if phased else 6)
        for option in ("WSL", "WSL_ROCDXG", "MIRAGE", "HIPDNN_SAMPLES"):
            assert f"-DTHEROCK_ENABLE_{option}=OFF" in calls
        for option in ("LLVM_PARALLEL_LINK_JOBS", "FLANG_PARALLEL_COMPILE_JOBS"):
            assert f"-D{option}=1" in calls


@pytest.mark.parametrize("transport", ["kfd", "wsl", "dual"])
@pytest.mark.parametrize("script", ["build_therock.sh", "build_rocm.sh"])
def test_local_sdk_rejects_unspecified_target_before_source_fetch(tmp_path, transport, script):
    result = subprocess.run(
        ["bash", SERVICE / "build" / script, "--paddle-only"],
        env={
            **os.environ, "OCR_ROCM_TRANSPORT": transport, "AMDGPU_TARGETS": "",
            "BUILD_JOBS": "1", "SOURCE_ROOT": str(tmp_path / "sources"),
            "EVIDENCE": str(tmp_path / "evidence"), "GIT_ALLOW_PROTOCOL": "file",
        },
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "make ocr-build-rocm" in result.stderr
    assert not (tmp_path / "sources").exists()


def test_paddle_timing_preserves_wheel_failure_through_tee(tmp_path):
    script = (SERVICE / "build/build_rocm.sh").read_text()
    phase = script[script.index("TIMEFORMAT='ocr_rocm_phase phase=paddle_compile"):].split(
        '\ncp "$PADDLE_SRC"/dist/', 1
    )[0]
    (tmp_path / "setup.py").write_text("print('wheel failed')\nraise SystemExit(19)\n")
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + phase],
        env={**os.environ, "PADDLE_SRC": str(tmp_path), "EVIDENCE": str(tmp_path)},
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 19
    assert (tmp_path / "paddle-build.log").read_text() == "wheel failed\n"
    assert re.search(r"ocr_rocm_phase phase=paddle_compile elapsed_s=[0-9.]+", result.stderr)
