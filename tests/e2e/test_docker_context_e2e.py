"""Use actual BuildKit matching on synthetic files, never operator context data."""
import runpy
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile
import pytest

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]


def test_build_context_admits_sources_but_not_local_runtime_artifacts(tmp_path):
    context, output = tmp_path / 'context', tmp_path / 'output'
    context.mkdir()
    (context / '.dockerignore').write_bytes((ROOT / '.dockerignore').read_bytes())
    (context / 'Dockerfile').write_text('FROM scratch\nCOPY . /\n')
    admitted = ['frontend/src/app.ts', 'frontend/package.json', 'frontend/package-lock.json', 'frontend/.npmrc', 'frontend/patches/vendor.patch', 'src/vision_bill/main.py', 'alembic/versions/0001.py', 'paddleocr-service/ocr_contract/__init__.py', 'paddleocr-service/ocr_contract/document_image.py', 'paddleocr-service/ocr_contract/limits.py', 'paddleocr-service/ocr_contract/wire_json.py', 'pyproject.toml', 'uv.lock']
    excluded = ['frontend/test-results/private-trace.zip', 'frontend/playwright-report/private.html', 'frontend/.env', 'frontend/.env.local', 'src/vision_bill/__pycache__/main.cpython-312.pyc', 'alembic/__pycache__/env.pyc', 'frontend/node_modules/unneeded.js', 'paddleocr-service/qualification-private.json', 'paddleocr-service/ocr_service/targets.py', 'paddleocr-service/build/hip-rdna2-v1.patch', 'paddleocr-service/runtime-lock.json']
    for name in admitted + excluded:
        target = context / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text('synthetic marker')
    result = subprocess.run(['docker', 'buildx', 'build', '--network=none', '--progress=plain', '--output', f'type=local,dest={output}', str(context)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert all((output / name).is_file() for name in admitted)
    assert not [name for name in excluded if (output / name).exists()]


def test_native_context_admits_contract_and_build_inputs_without_app_or_private_files(tmp_path):
    context, output = tmp_path / "context", tmp_path / "output"
    context.mkdir()
    (context / ".dockerignore").write_bytes(
        (ROOT / "paddleocr-service/.dockerignore").read_bytes()
    )
    (context / "Dockerfile").write_text("FROM scratch\nCOPY . /\n")
    admitted = [
        "ocr_contract/__init__.py", "ocr_contract/document_image.py", "ocr_contract/limits.py",
        "ocr_contract/wire_json.py",
        "ocr_service/api.py", "ocr_service/worker.py", "build/build_rocm.sh",
        "build/prepare_therock.sh", "build/build_jobs.py", "build/hip-rdna2-v1.patch",
        "build/build_therock.sh", "build/finish_therock.sh", "build/verify_therock_targets.py",
        "build/checkpoint_therock.sh", "build/verify_rccl_targets.py", "build/therock-sqlite-headers.patch",
        "build/therock-rccl-translate.patch", "build/miopen-sqlite-transient.patch",
        "build/therock-hipblaslt-rdna2.patch", "build/rocm-libraries-hipblaslt-rdna2.patch",
        "profile.json", "runtime-lock.json", "requirements.cuda-build.lock",
        "requirements.runtime.lock", "requirements.cuda.attestation.json", "server.py",
    ]
    excluded = [
        "src/vision_bill/config.py", "vision_bill/model/ocr.py", "config.yaml",
        "models/private/model.json", "qualification-private.json", ".env",
        "ocr_contract/__pycache__/module.pyc", "ocr_contract/private.json",
        "ocr_service/private.json", "build/private.log", "uploads/private.png",
    ]
    for name in admitted + excluded:
        target = context / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic marker")
    result = subprocess.run(
        ["docker", "buildx", "build", "--network=none", "--progress=plain",
         "--output", f"type=local,dest={output}", str(context)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert all((output / name).is_file() for name in admitted)
    assert not [name for name in excluded if (output / name).exists()]


def test_operator_guide_build_skips_unreadable_runtime_data(tmp_path):
    context, output = tmp_path / "context", tmp_path / "output"
    context.mkdir()
    shutil.copy(ROOT / ".dockerignore", context)
    script = "frontend/scripts/operator_guides.py"
    public_files = runpy.run_path(str(ROOT / script))["PUBLIC_FILES"]
    for name in (*public_files, script):
        destination = context / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)

    recipe = (ROOT / "Dockerfile").read_text()
    (context / "Dockerfile").write_text(
        recipe + "\nFROM scratch AS guides-export\nCOPY --from=operator-guides /guides.zip /guides.zip\n"
    )
    private = context / "server_data/config"
    private.mkdir(parents=True)
    (private / "private-marker").write_text("synthetic runtime data")
    private.chmod(0)
    try:
        result = subprocess.run(
            ["docker", "buildx", "build", "--target", "guides-export", "--network=none",
             "--progress=plain", "--output", f"type=local,dest={output}", str(context)],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        private.chmod(0o700)

    assert result.returncode == 0, result.stderr
    with ZipFile(output / "guides.zip") as archive:
        assert set(archive.namelist()) == set(public_files)
        for name in public_files:
            assert archive.read(name) == (ROOT / name).read_bytes()
