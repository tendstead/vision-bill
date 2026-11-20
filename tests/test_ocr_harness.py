import json
import sys
import time
from decimal import Decimal

import pytest

from scripts.ocr_harness.common import normalize_result, row_text
from scripts.ocr_harness.runner import run_matrix


def engine(tmp_path, name="fake", body=None):
    worker = tmp_path / f"{name}.py"
    worker.write_text(
        body
        or (
            "import json,sys\nfrom pathlib import Path\n"
            "Path(sys.argv[2]).write_text(json.dumps({'text':Path(sys.argv[1]).read_text(),'blocks':[]}))"
        )
    )
    return {"id": name, "command": [sys.executable, str(worker), "{image}", "{result}"]}


def test_cache_tracks_image_and_engine_configuration(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    e = engine(tmp_path)
    out = tmp_path / "out"
    first = run_matrix([e], [image], out, timeout=5)
    assert first[0]["status"] == "ok"
    assert not first[0]["cached"]
    assert run_matrix([e], [image], out, timeout=5)[0]["cached"]
    image.write_text("TOTAL 42,00")
    assert not run_matrix([e], [image], out, timeout=5)[0]["cached"]
    e["options"] = {"variant": "other"}
    assert not run_matrix([e], [image], out, timeout=5)[0]["cached"]


def test_failure_does_not_stop_other_engines_and_is_not_cached(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    bad = engine(tmp_path, "bad", "raise RuntimeError('missing model')")
    rows = run_matrix([bad, engine(tmp_path)], [image], tmp_path / "out", timeout=5)
    assert [r["status"] for r in rows] == ["error", "ok"]
    assert rows[0]["error"]
    assert not run_matrix([bad], [image], tmp_path / "out", timeout=5)[0]["cached"]


def test_real_subprocess_timeout_is_reported(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("receipt")
    slow = engine(tmp_path, "slow", "import time; time.sleep(30)")
    start = time.monotonic()
    rows = run_matrix([slow], [image], tmp_path / "out", timeout=0.1)
    assert rows[0]["status"] == "timeout"
    assert time.monotonic() - start < 3


@pytest.mark.parametrize(
    "data",
    [
        {"text": " "},
        {"text": 7},
        {"text": "ok", "blocks": [{"text": "x", "bbox": [[float("nan"), 0]]}]},
    ],
)
def test_invalid_recognition_is_not_success(data):
    with pytest.raises(ValueError):
        normalize_result(data)


def test_row_reconstruction_preserves_price_pairing():
    blocks = [
        {"text": "1,29", "bbox": [[100, 11], [130, 21]]},
        {"text": "MILCH", "bbox": [[0, 10], [50, 20]]},
        {"text": "PFAND", "bbox": [[0, 35], [50, 45]]},
        {"text": "0,25", "bbox": [[100, 35], [130, 45]]},
    ]
    assert row_text(blocks) == "MILCH  1,29\nPFAND  0,25"


def test_native_nested_coordinates_are_json_serializable():
    from scripts.ocr_harness.worker import block

    class NativeNumber:
        def tolist(self):
            return 3

    result = block("MILCH", [[NativeNumber(), NativeNumber()], [10, 20]], 0.9)
    assert json.loads(json.dumps(result))["bbox"] == [[3, 3], [10, 20]]


def test_report_includes_failures_and_no_unearned_accuracy(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    out = tmp_path / "out"
    run_matrix([engine(tmp_path)], [image], out, timeout=5)
    report = json.loads((out / "report.json").read_text())
    assert report["results"][0]["accuracy"] is None
    assert (out / "report.csv").is_file()
    assert (out / "report.md").is_file()


def test_model_weights_invalidate_cache(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_text("TOTAL 12,34")
    out = tmp_path / "results"
    model = tmp_path / "models" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"weights-v1")
    e = engine(tmp_path)
    run_matrix([e], [image], out, timeout=5)
    assert run_matrix([e], [image], out, timeout=5)[0]["cached"]
    model.write_bytes(b"weights-v2")
    assert not run_matrix([e], [image], out, timeout=5)[0]["cached"]


def test_receipt_scoring_does_not_forgive_one_cent(tmp_path):
    from scripts.ocr_harness.extraction import score_expected
    from vision_bill.model.receipt import Receipt

    expected = Receipt(
        confidence=90,
        merchant_name="SHOP",
        date="2026-11-07",
        currency="EUR",
        line_items=[],
        subtotal="12.34",
        total="12.34",
    )
    path = tmp_path / "expected.json"
    path.write_text(expected.model_dump_json())
    actual = expected.model_copy(update={"total": Decimal("12.35")})
    score = score_expected(path, actual)
    assert any(d["path"] == "total" and not d["equivalent"] for d in score["differences"])


def test_custom_adapter_needs_no_docker_preparation(tmp_path):
    from scripts.ocr_harness.__main__ import prepare

    records = prepare([engine(tmp_path)], tmp_path, timeout=1)
    assert records[0]["status"] == "ok"


def test_local_llm_client_ignores_environment_proxy(monkeypatch):
    from scripts.ocr_harness.extraction import make_provider

    monkeypatch.setenv("ALL_PROXY", "http://untrusted.invalid:1234")
    provider = make_provider("http://127.0.0.1:11434")
    assert provider._client._client.trust_env is False
    assert provider._client._client.follow_redirects is False


def test_scores_for_different_expected_receipts_do_not_collide(tmp_path):
    from scripts.ocr_harness.extraction import score_path

    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text('{"total":"1.00"}')
    second.write_text('{"total":"2.00"}')
    assert score_path(tmp_path, first) != score_path(tmp_path, second)


def test_changed_raw_or_normalized_artifact_is_not_reused(tmp_path):
    image = tmp_path / 'source.png'
    image.write_text('TOTAL 1.00')
    out, e = tmp_path / 'out', engine(tmp_path)
    row = run_matrix([e], [image], out, timeout=5)[0]
    from pathlib import Path
    artifact = Path(row['artifact'])
    artifact.write_text(json.dumps({'text': 'changed', 'blocks': []}))
    assert not run_matrix([e], [image], out, timeout=5)[0]['cached']
    raw = artifact.with_name('raw.json')
    raw.write_text(raw.read_text() + '\n')
    assert not run_matrix([e], [image], out, timeout=5)[0]['cached']


def test_http_native_adapter_uses_v2_identity_and_keeps_typed_raw_page(monkeypatch,tmp_path):
    import hashlib
    from test_ocr_provider import wire_page
    from scripts.ocr_harness import worker
    from PIL import Image
    image = tmp_path/'source.png'; Image.new('RGB',(3,4)).save(image)
    requests=[]
    class Response:
        def __init__(self,data): self.data=json.dumps(data).encode()
        def read(self,*args): return self.data
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def close(self): pass
    class Client:
        def open(self,request,**kwargs):
            requests.append(request)
            if isinstance(request,str):
                return Response({'ready':True,'profile_id':'fixed-profile','runtime_fingerprint':'runtime'})
            body=request.data.decode('latin1')
            import re
            rid = re.search('name="request_id"\r\n\r\n([^\r]+)',body).group(1)
            return Response(wire_page(0,hashlib.sha256(image.read_bytes()).hexdigest(),rid))
    monkeypatch.setattr(worker.urllib.request,'build_opener',lambda *args:Client())
    page=worker.http({'url':'http://127.0.0.1:8082','timeout':600},image)
    assert page['contract_version']==2
    body=requests[-1].data
    assert b'name="photo_index"' in body and b'name="timeout_ms"' in body
    assert b'name="source_sha256"' in body and image.read_bytes() in body


def test_native_http_cache_identity_tracks_actual_ready_profile_and_runtime(monkeypatch):
    from scripts.ocr_harness import runner, worker
    state={'ready':True,'profile_id':'p1','runtime_fingerprint':'r1'}
    class Response:
        def read(self,*args): return json.dumps(state).encode()
        def __enter__(self): return self
        def __exit__(self,*args): pass
    class Client:
        def open(self,*args,**kwargs): return Response()
    monkeypatch.setattr(worker.urllib.request,'build_opener',lambda *args:Client())
    engine={'family':'http','options':{'url':'http://127.0.0.1:8082'}}
    first=runner.image_identity(engine)
    state['profile_id']='p2'
    assert runner.image_identity(engine)!=first
    state['profile_id']='p1'; state['runtime_fingerprint']='r2'
    assert runner.image_identity(engine)!=first


def test_native_normalization_preserves_reading_order_block_identity_without_png():
    from test_ocr_provider import wire_page
    page=wire_page(text='Bottom\nTop')
    page['blocks']=[{'id':'bottom','label':'text','text':'Bottom','bbox':[[0,10],[3,10],[3,12]]},
                    {'id':'top','label':'text','text':'Top','bbox':[[0,0],[3,0],[3,2]]}]
    result=normalize_result(page)
    assert result['row_text']=='Bottom\nTop'
    assert result['blocks'][0]['id']=='bottom'
    assert result['page_metadata']['working_image']['view_id']==page['working_image']['view_id']
    assert 'png_base64' not in json.dumps(result)


async def test_optional_extraction_stage_uses_typed_replay_not_legacy_text(tmp_path):
    from test_extraction_pipeline import formatter
    from test_ocr_provider import wire_page
    from scripts.ocr_harness.extraction import service_extract
    from PIL import Image
    from scripts.ocr_harness.runner import digest
    image=tmp_path/'source.png'; Image.new('RGB',(3,4)).save(image)
    raw=tmp_path/'raw.json';raw.write_text(json.dumps(wire_page(digest=digest(image.read_bytes()),text='ONLY OBSERVED')))
    row={'input':str(image),'engine':'native','status':'ok','artifact':str(raw),'identity':{'input_sha256':digest(image.read_bytes())}}
    llm=formatter(False)
    receipt,metadata=await service_extract(llm,'fixture',row,'REFERENCE_SENTINEL',tmp_path)
    prompt=llm._client.chat.call_args.kwargs['messages'][0]['content']
    assert receipt is not None and metadata['ocr_replayed']
    assert 'ONLY OBSERVED' in prompt and 'REFERENCE_SENTINEL' not in prompt


def test_paddle_adapter_preserves_mapping_results_and_lazy_optional_import(monkeypatch, tmp_path):
    from collections import UserDict
    from types import SimpleNamespace

    import numpy as np

    from scripts.ocr_harness import worker

    calls = []
    class Model:
        def predict(self, image):
            assert image == str(tmp_path / "image.png")
            return [UserDict(rec_texts=["MILCH"], rec_polys=[np.array([[0, 0], [1, 1]])], rec_scores=[np.float32(0.5)])]

    def construct(**options):
        calls.append(options)
        return Model()

    def native_import(name):
        assert name == "paddleocr"
        return SimpleNamespace(PaddleOCR=construct)

    monkeypatch.setattr(worker.importlib, "import_module", native_import)
    result = worker.paddle({"det": "test-det", "rec": "test-rec"}, tmp_path / "image.png")
    assert result["text"] == "MILCH"
    assert result["blocks"] == [{"text": "MILCH", "bbox": [[0, 0], [1, 1]], "confidence": 0.5}]
    assert calls[0]["device"] == "cpu" and calls[0]["enable_mkldnn"] is False


@pytest.mark.parametrize("value", [None, {}, [42], [{"id": None}]])
def test_engine_manifest_rejects_non_objects_and_non_string_ids(tmp_path, value):
    from scripts.ocr_harness.__main__ import load_engines

    manifest = tmp_path / "engines.json"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="objects with string IDs"):
        load_engines(manifest)


def test_historical_paddle_build_never_adopts_or_builds_production_sidecar(tmp_path,monkeypatch):
    from scripts.ocr_harness import __main__ as cli
    calls=[]
    monkeypatch.setattr(cli,'bounded_run',lambda argv,*args: calls.append(argv))
    def no_inspect(*args,**kwargs):
        raise AssertionError('historical comparator must not inspect stale production tag')
    monkeypatch.setattr(cli.subprocess,'run',no_inspect)
    assert cli.build([{'target':'paddle','image':'vb-historical-cpu-test'}],tmp_path,30)==[]
    assert len(calls)==1
    assert calls[0] == ['docker','build','--target','paddle','-t','vb-historical-cpu-test',str(cli.WORKER.parent)]
    recipe=(cli.WORKER.parent/'Dockerfile').read_text()
    # Build-recipe boundary: historical CPU stage must be independently constructible.
    paddle_stage=recipe.split(' AS paddle\n')[1]
    assert 'requirements.paddle-cpu.lock' in paddle_stage
    assert 'vision-bill-paddleocr-service' not in recipe
