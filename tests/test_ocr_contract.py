"""The native source and app wheel each carry the same portable OCR contract."""

import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "paddleocr-service"


def test_native_source_imports_without_application(tmp_path):
    isolated = tmp_path / "native"
    shutil.copytree(SERVICE, isolated, ignore=shutil.ignore_patterns("__pycache__"))
    script = """
        import importlib.abc
        import sys
        from pathlib import Path
        class NoApplication(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'vision_bill' or fullname.startswith('vision_bill.'):
                    raise ImportError('native OCR must not import the app')
        sys.meta_path.insert(0, NoApplication())
        sys.path.insert(0, sys.argv[1])
        from ocr_contract import OcrResponse, SourceRef
        from ocr_contract.document_image import NORMALIZER_VERSION, document_rgb
        from ocr_service import api, images, pipeline, qualify, worker
        from ocr_service.profile import content_identity, effective_pipeline_config, load_profile
        from PIL import Image
        assert content_identity(OcrResponse.model_json_schema()) == '86aa8a10686ac01547bba6ef80d85bc0a39d193e6911b4df99d7145ff178701b'
        assert content_identity(SourceRef.model_json_schema()) == '6473c0469cb3b365dca0629f7d2f1202f12a4d6ebaf5cd0c1fa1c5c417210b00'
        config = effective_pipeline_config(load_profile(Path(sys.argv[1]) / 'profile.json'))
        assert content_identity(config) == '490492272e1a7d40ade25b976392bebedceb4b039701f6e57ab087b2b415c212'
        assert NORMALIZER_VERSION == 'white-document-rgb-v2'
        assert document_rgb(Image.new('RGBA', (1, 1), (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
        assert not any(name == 'vision_bill' or name.startswith('vision_bill.') for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script), str(isolated)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_app_imports_reexport_canonical_contract_and_normalizer():
    import ocr_contract
    from ocr_contract import document_image
    from vision_bill.helper import document_image as app_image
    from vision_bill.model import ocr

    for name in ocr_contract.__all__:
        assert getattr(ocr, name) is getattr(ocr_contract, name)
    assert app_image.document_rgb is document_image.document_rgb
    assert app_image.NORMALIZER_VERSION == document_image.NORMALIZER_VERSION


def test_app_wheel_installs_neutral_contract(tmp_path):
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "ocr_contract/__init__.py" in archive.namelist()
        assert "ocr_contract/document_image.py" in archive.namelist()
        assert "ocr_contract/limits.py" in archive.namelist()
        assert "ocr_contract/wire_json.py" in archive.namelist()
        archive.extractall(tmp_path / "installed")
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent("""
            import json, sys
            sys.path.insert(0, sys.argv[1])
            from ocr_contract import OcrResponse
            from vision_bill.model.ocr import OcrResponse as AppResponse
            assert OcrResponse is AppResponse
            print(json.dumps({'module': OcrResponse.__module__}))
        """), str(tmp_path / "installed")],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["module"].startswith("ocr_contract")


@pytest.mark.parametrize(("raw_size", "wire_size"), [(0, 0), (1, 4), (3, 4), (4, 8), (6, 8), (7, 12)])
def test_base64_wire_bound_includes_padding(raw_size, wire_size):
    from ocr_contract.limits import base64_bound

    assert base64_bound(raw_size) == wire_size


def test_working_png_bound_accounts_for_one_filter_per_row_at_maximum_height():
    from ocr_contract.limits import MAX_IMAGE_PIXELS, MAX_WORKING_PNG_BYTES, working_png_bound

    assert MAX_IMAGE_PIXELS == 40_000_000
    # RGB plus one per-row filter byte: a 1 × 40,000,000 image needs 160,000,000
    # scanline bytes, 182,500,011 bounded zlib bytes and 2,785 IDAT chunks.
    assert working_png_bound(40_000_000) == MAX_WORKING_PNG_BYTES == 182_533_476


@pytest.mark.parametrize(
    ("encoded", "valid"),
    [("YWJj", True), ("YWJjZA==", True), ("YWJjZGU=", False), ("YWJjZGVmZw==", False)],
)
def test_working_png_decode_checks_encoded_and_decoded_bounds(monkeypatch, encoded, valid):
    from ocr_contract import limits
    from ocr_contract.document_image import decode_working_png

    monkeypatch.setattr(limits, "MAX_WORKING_PNG_BYTES", 4)
    if valid:
        assert decode_working_png(encoded) in (b"abc", b"abcd")
    else:
        with pytest.raises(ValueError, match="ocr_working_png_too_large"):
            decode_working_png(encoded)


def test_wire_metadata_budget_preserves_exact_utf8_and_reordered_escaped_wire(
    tmp_path, monkeypatch
):
    from ocr_contract import OcrBlock, OcrResponse, limits
    from test_ocr_worker_lifecycle import request, response

    result = response(request(tmp_path))
    result.telemetry = {"note": ["テ𐍈\x00\n\\/\"", {"ratio": 1e-7, "flag": True}]}
    result.working_image.transforms = [{"name": "source\tテ", "angle": 0}]
    result.blocks = [
        OcrBlock(id="0", label="text", text="SHOP", bbox=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    ]
    canonical = result.model_dump_json().encode("utf-8")
    view = result.working_image
    exact_metadata_bytes = len(canonical) - len(view.png_base64)
    monkeypatch.setattr(limits, "MAX_RESPONSE_METADATA_BYTES", exact_metadata_bytes)

    # Present the PNG first, reorder the whole page, and escape the keys,
    # base64, Unicode and whitespace without changing the canonical metadata.
    reordered = dict(reversed(list(json.loads(canonical).items())))
    reordered["working_image"] = dict(reversed(list(reordered["working_image"].items())))
    result.check_wire_bounds()
    assert result.dump_checked_wire() == canonical
    assert OcrResponse.parse_wire(canonical) == result
    alternate = json.dumps(reordered, ensure_ascii=True, indent=2)
    alternate = alternate.replace('"working_image"', r'"working_\u0069mage"')
    alternate = alternate.replace('"png_base64"', r'"png_\u0062ase64"')
    alternate = alternate.replace(view.png_base64, r"\u0069" + view.png_base64[1:])
    assert OcrResponse.parse_wire(bytearray(alternate.encode("utf-8"))) == result

    monkeypatch.setattr(limits, "MAX_RESPONSE_METADATA_BYTES", exact_metadata_bytes - 1)
    with pytest.raises(ValueError, match="ocr_response_metadata_too_large"):
        OcrResponse.parse_wire(canonical)
    with pytest.raises(ValueError, match="ocr_response_metadata_too_large"):
        result.check_wire_bounds()
