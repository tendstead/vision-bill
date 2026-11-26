import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from runpy import run_path
from typing import Any

import pytest
from ocr_service.provision import (
    MODEL_NAMES,
    Fixtures,
    ReleaseBundle,
    content_identity,
    extract_archive,
    fetch_archive,
)
from PIL import Image

SERVICE = Path(__file__).resolve().parents[1] / "paddleocr-service"
KINDS = ("upright", "rotated", "warped", "table", "formula")


@pytest.fixture
def approved(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    license = {"spdx": "CC0-1.0", "text_file": "terms.txt", "redistribution_allowed": True}
    provenance = {"url": "https://example.org/fixture-source", "revision": "dataset-v1"}
    (tmp_path / "terms.txt").write_text("Test license text\n")
    cases = {}
    for number, kind in enumerate(KINDS):
        path = tmp_path / f"original-{kind}.png"
        Image.new("RGB", (2, 2), (number, 0, 0)).save(path)
        cases[kind] = path.name
    fixtures = tmp_path / "approved-fixtures.json"
    specification = {"license": license, "provenance": provenance, "cases": cases}
    fixtures.write_text(json.dumps(specification))

    terms = b"Test model license text\n"
    files = {"LICENSE": {"sha256": hashlib.sha256(terms).hexdigest(), "size_bytes": len(terms)}}
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps(
            {
                name: {
                    "source": f"https://example.org/models/{name}.tar",
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "size_bytes": 10240,
                    "unpacked_bytes": len(terms),
                    "files": files,
                    "tree_sha256": content_identity({"LICENSE": files["LICENSE"]["sha256"]}),
                    "license": {**license, "text_file": "LICENSE"},
                    "provenance": provenance,
                }
                for name in MODEL_NAMES
            }
        )
    )
    return models, fixtures, specification


def prepare(
    models: Path, fixtures: Path, output: Path, accelerator: str = "cuda"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SERVICE / "build/prepare_ocr_inputs.py"),
            accelerator,
            "--models",
            str(models),
            "--fixtures",
            str(fixtures),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_legacy_model_records_only_prepare_the_three_active_models(tmp_path, approved):
    models, fixtures, _ = approved
    records = json.loads(models.read_bytes())
    records["UVDoc"] = dict(records["PP-DocLayoutV3"], source="https://example.org/UVDoc.tar")
    models.write_text(json.dumps(records))
    output = tmp_path / "prepared"
    result = prepare(models, fixtures, output)
    assert result.returncode == 0, result.stderr
    assert set(json.loads((output / "bundle.json").read_bytes())["models"]) == {
        "PP-DocLayoutV3", "PP-LCNet_x1_0_doc_ori", "PaddleOCR-VL-1.6-0.9B"
    }
    assert json.loads(models.read_bytes()) == records


def test_prepares_deterministic_archive_and_round_trips_provisioning(
    tmp_path: Path, approved: tuple[Path, Path, dict[str, Any]]
) -> None:
    models, fixtures, specification = approved
    prepared, repeated, cache = (tmp_path / name for name in ("prepared", "repeated", "cache"))
    for output in (prepared, repeated):
        result = prepare(models, fixtures, output)
        assert result.returncode == 0, result.stderr
        assert {path.name for path in output.iterdir()} == {"fixtures.tar", "bundle.json"}
    for filename in ("fixtures.tar", "bundle.json"):
        assert (prepared / filename).read_bytes() == (repeated / filename).read_bytes()
    bundle_template = json.loads((prepared / "bundle.json").read_bytes())
    assert not {
        "image_identity",
        "image_config_sha256",
        "profile_sha256",
        "source_files",
    }.intersection(bundle_template)
    assert bundle_template["models"] == json.loads(models.read_bytes())
    bundle = ReleaseBundle.model_validate(
        {
            **bundle_template,
            "image_identity": "sha256:" + "a" * 64,
            "image_config_sha256": "a" * 64,
            "profile_sha256": "b" * 64,
            "source_files": {"profile.json": "b" * 64},
        }
    )
    assert (
        bundle.expected_paddle_commit
        == json.loads((SERVICE / "runtime-lock.json").read_bytes())["sources"]["paddle"]
    )
    assert bundle.expected_packages["paddlepaddle-gpu"] == "3.3.1"
    assert bundle.expected_packages["numpy"] == "2.2.6"
    assert bundle.fixtures.source == "fixtures.tar"
    extracted = tmp_path / "extracted"
    cache.mkdir()
    archive = fetch_archive(bundle.fixtures, prepared, cache)
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == sorted(["LICENSE.txt", *(f"{kind}.png" for kind in KINDS)])
    extract_archive(archive, extracted, bundle.fixtures)
    assert (extracted / bundle.fixtures.license.text_file).read_bytes() == (
        tmp_path / "terms.txt"
    ).read_bytes()
    for case in bundle.fixtures.cases:
        assert (extracted / case.path).read_bytes() == (
            tmp_path / specification["cases"][case.kind]
        ).read_bytes()
        assert case.provenance.model_dump() == specification["provenance"]
        assert case.required_labels == ([case.kind] if case.kind in {"table", "formula"} else [])


def test_packages_project_owned_first_run_fixtures(
    tmp_path: Path, approved: tuple[Path, Path, dict[str, Any]]
) -> None:
    models, _, _ = approved
    sources = SERVICE / "first-run-fixtures"
    output = tmp_path / "prepared"
    result = prepare(models, sources / "fixtures.json", output)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "bundle.json").read_bytes())
    fixtures = Fixtures.model_validate(manifest["fixtures"])
    extracted = tmp_path / "extracted"
    extract_archive(output / fixtures.source, extracted, fixtures)
    assert fixtures.license.spdx == "AGPL-3.0-or-later"
    assert (extracted / fixtures.license.text_file).read_bytes() == (
        SERVICE.parent / "LICENSE"
    ).read_bytes()
    assert {case.kind: case.required_labels for case in fixtures.cases} == {
        "upright": [],
        "rotated": [],
        "warped": [],
        "table": ["table"],
        "formula": ["formula"],
    }
    for case in fixtures.cases:
        with Image.open(extracted / case.path) as image:
            assert image.format == "PNG"
            assert min(image.size) >= 500
            assert image.getexif().get(274, 1) == 1


def test_rocm_inputs_pin_upstreams_dcu_distribution(
    tmp_path: Path, approved: tuple[Path, Path, dict[str, Any]]
) -> None:
    models, fixtures, _ = approved
    output = tmp_path / "prepared"
    result = prepare(models, fixtures, output, "rocm")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "bundle.json").read_bytes())
    assert manifest["accelerator"] == "rocm"
    assert manifest["expected_packages"]["paddlepaddle-dcu"] == "3.3.1"


def test_existing_output_is_preserved(
    tmp_path: Path, approved: tuple[Path, Path, dict[str, Any]]
) -> None:
    models, fixtures, _ = approved
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "bundle.json"
    marker.write_text("previous bundle")
    result = prepare(models, fixtures, output)
    assert result.returncode == 1
    assert "Output directory already exists" in result.stderr
    assert marker.read_text() == "previous bundle"
    assert not (output / "fixtures.tar").exists()


@pytest.mark.parametrize(
    "url",
    ["https://tester:placeholder@example.org/source", "https://example.org/source?token=dummy"],
)
def test_model_provenance_cannot_publish_credentials(
    tmp_path: Path, approved: tuple[Path, Path, dict[str, Any]], url: str
) -> None:
    models, fixtures, _ = approved
    mapping = json.loads(models.read_bytes())
    mapping[MODEL_NAMES[0]]["provenance"]["url"] = url
    models.write_text(json.dumps(mapping))
    output = tmp_path / "prepared"
    result = prepare(models, fixtures, output)
    assert result.returncode == 1
    assert "HTTPS URLs without credentials" in result.stderr
    assert url not in result.stderr
    assert not output.exists()


def test_concurrent_empty_output_is_preserved_at_publication(
    tmp_path: Path,
    approved: tuple[Path, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models, fixtures, _ = approved
    output = tmp_path / "prepared"
    run_prepare = run_path(str(SERVICE / "build/prepare_ocr_inputs.py"))["prepare"]
    chmod = Path.chmod

    def create_output_after_staging(path: Path, mode: int) -> None:
        chmod(path, mode)
        if path.name == "bundle.json":
            output.mkdir()

    monkeypatch.setattr(Path, "chmod", create_output_after_staging)
    with pytest.raises(FileExistsError):
        run_prepare("cuda", models, fixtures, output)
    assert sorted(tmp_path.glob(".ocr-inputs-*")) == []
    assert list(output.iterdir()) == []


def test_interruption_at_publication_leaves_output_absent(
    tmp_path: Path,
    approved: tuple[Path, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models, fixtures, _ = approved
    output = tmp_path / "prepared"
    run_prepare = run_path(str(SERVICE / "build/prepare_ocr_inputs.py"))["prepare"]

    def interrupt_at_publication(staged: Path, destination: Path) -> None:
        assert destination == output
        assert not destination.exists()
        assert {path.name for path in staged.iterdir()} == {"fixtures.tar", "bundle.json"}
        raise KeyboardInterrupt

    monkeypatch.setitem(run_prepare.__globals__, "publish_directory", interrupt_at_publication)
    with pytest.raises(KeyboardInterrupt):
        run_prepare("cuda", models, fixtures, output)
    assert not output.exists()
    assert sorted(tmp_path.glob(".ocr-inputs-*")) == []


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("unknown", "five fixture roles"),
        ("missing", "five fixture roles"),
        ("duplicate", "five distinct qualification kinds"),
        ("unsafe", "canonical safe relative paths"),
        ("symlink", "symlinks"),
        ("symlink-directory", "symlinks"),
        ("invalid-png", "PNG"),
        ("missing-provenance", "provenance"),
    ],
)
def test_rejects_bad_fixture_inputs(
    tmp_path: Path, approved: tuple[Path, Path, dict[str, Any]], change: str, error: str
) -> None:
    models, fixtures, specification = approved
    cases = specification["cases"]
    if change == "unknown":
        cases["other"] = cases["upright"]
    elif change == "missing":
        del cases["table"]
    elif change == "duplicate":
        (tmp_path / cases["warped"]).write_bytes((tmp_path / cases["upright"]).read_bytes())
    elif change == "unsafe":
        cases["upright"] = "../original-upright.png"
    elif change == "symlink":
        (tmp_path / "link.png").symlink_to(tmp_path / cases["upright"])
        cases["upright"] = "link.png"
    elif change == "symlink-directory":
        (tmp_path / "link-dir").symlink_to(tmp_path, target_is_directory=True)
        cases["upright"] = "link-dir/" + cases["upright"]
    elif change == "invalid-png":
        (tmp_path / cases["upright"]).write_bytes(b"not an image")
    elif change == "missing-provenance":
        del specification["provenance"]
    fixtures.write_text(json.dumps(specification))
    output = tmp_path / "failed"
    result = prepare(models, fixtures, output)
    assert result.returncode == 1
    assert error in result.stderr
    assert not output.exists()
