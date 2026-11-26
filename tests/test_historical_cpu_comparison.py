"""Real CPU artifact, formatter, merger and orchestration seams; no CPU inference."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ollama import ChatResponse
from test_experiment_scoring import receipt
from test_extraction_pipeline import formatter
from test_paired_benchmark import case_files, typed_rows

from scripts.ocr_harness import comparison, extraction
from scripts.ocr_harness.common import normalize_result
from scripts.ocr_harness.runner import digest


def cpu_rows(case, tmp_path, monkeypatch=None):
    rows = []
    for i, path in enumerate(case['images']):
        raw = tmp_path / f'cpu-{i}-raw.json'
        artifact = tmp_path / f'cpu-{i}-ocr.json'
        if monkeypatch is None:
            data = {'text': f'CPU OBSERVED {i}', 'blocks': []}
        else:
            from scripts.ocr_harness import worker
            def construct(**options):
                assert options['device'] == 'cpu'
                assert options['text_detection_model_name'] == 'PP-OCRv6_medium_det'
                assert options['text_recognition_model_name'] == 'PP-OCRv6_medium_rec'
                return SimpleNamespace(predict=lambda image: [{'rec_texts': [f'CPU OBSERVED {i}'], 'rec_polys': [[[0, 0], [1, 1]]], 'rec_scores': [0.9]}])
            with monkeypatch.context() as patch:
                patch.setattr(worker.importlib, 'import_module', lambda name: SimpleNamespace(PaddleOCR=construct))
                data = worker.paddle({'det': 'PP-OCRv6_medium_det', 'rec': 'PP-OCRv6_medium_rec'}, path)
        data['versions'] = {'paddleocr': 'synthetic-adapter-seam'}
        raw.write_text(json.dumps(data))
        artifact.write_text(json.dumps(normalize_result(data)))
        rows.append({'engine': 'pp-medium', 'input': str(path), 'status': 'ok', 'cached': False,
                     'raw_artifact': str(raw), 'artifact': str(artifact), 'elapsed_ms': 12,
                     'raw_sha256': digest(raw.read_bytes()), 'artifact_sha256': digest(artifact.read_bytes()),
                     'identity': {'input_sha256': digest(path.read_bytes()),
                                  'engine': {'id': 'pp-medium', 'family': 'paddle', 'options': {'variant': 'medium'}},
                                  'runtime': {'image': 'sha256:cpu-image'}, 'models': 'weights-hash',
                                  'worker_sha256': 'adapter-hash', 'normalizer_sha256': 'normalizer-hash',
                                  'limits': {'memory': '2g'}, 'timeout': 30}})
    return rows


def multi_payload():
    payload = receipt().model_dump(mode='json')
    item = payload['line_items'][0]
    payload['line_items'] = [{**item, 'source_image': i} for i in (0, 0, 1, 1)]
    payload['duplicate_candidates'] = [{'first_index': 1, 'second_index': 2}]
    return payload


async def test_optional_cpu_extraction_uses_real_formatter_and_honest_metadata(tmp_path, monkeypatch):
    case = case_files(tmp_path)
    rows = cpu_rows(case, tmp_path, monkeypatch)[:1]
    provider = formatter(False)
    provider._client.list = AsyncMock(return_value=SimpleNamespace(models=[SimpleNamespace(model='fixture', digest='model-hash')]))
    monkeypatch.setattr(extraction, 'make_provider', lambda host: provider)
    monkeypatch.setattr(extraction.logging, 'disable', lambda level: None)
    sidecar = case['images'][0].with_suffix('.json')
    sidecar.write_text(receipt().model_copy(update={'merchant_name': 'EXPECTED SECRET'}).model_dump_json())
    output = tmp_path / 'optional'
    await extraction.extract_results(rows, output, 'http://127.0.0.1:11434', 'fixture')
    assert rows[0]['llm_status'] == 'ok', rows[0].get('llm_error')
    assert rows[0]['execution_path'] == 'historical_cpu_transcript'
    prompt = provider._client.chat.call_args.kwargs['messages'][0]
    assert 'CPU OBSERVED 0' in prompt['content'] and 'EXPECTED SECRET' not in prompt['content']
    assert not prompt.get('images')
    assert rows[0]['score_artifact'] and rows[0]['trace_artifact']
    count = provider._client.chat.call_count
    await extraction.extract_results(rows, output, 'http://127.0.0.1:11434', 'fixture')
    assert rows[0]['llm_cached'] and provider._client.chat.call_count == count


async def test_paired_default_cpu_all_photos_shared_merger_cache_and_reference_isolation(tmp_path, monkeypatch):
    case = case_files(tmp_path)
    rows = cpu_rows(case, tmp_path, monkeypatch)
    provider = formatter(True, multi_payload())
    provider._client.list = AsyncMock(return_value=SimpleNamespace(models=[SimpleNamespace(model='fixture', digest='model-hash')]))
    monkeypatch.setattr(comparison, 'make_target_provider', lambda *args: provider)
    target = {'id': 'local', 'provider': 'ollama', 'host': 'http://127.0.0.1:11434', 'model': 'fixture'}
    async def run():
        return await comparison.run_comparison([case], [target], rows, 'pp-medium', tmp_path/'paired', 10, 1234)
    result = await run()
    assert [r['llm_status'] for r in result] == ['ok', 'ok'], result
    cpu = result[1]
    assert cpu['execution_path'] == 'historical_cpu_transcript'
    assert cpu['metadata']['used_pipeline'] == 'historical_cpu_ocr'
    assert cpu['metadata']['recorded_ocr_ms'] == 24 and cpu['metadata']['source_image_count'] == 2
    from pathlib import Path
    actual = json.loads(Path(cpu['receipt_artifact']).read_text())
    assert len(actual['line_items']) == 3  # repeated purchases retained; one boundary copy removed
    messages = provider._client.chat.call_args.kwargs['messages']
    prompt = messages[0]['content']
    assert prompt.index('CPU OBSERVED 0') < prompt.index('CPU OBSERVED 1')
    assert 'PHOTO 1' in prompt and digest(case['images'][1].read_bytes()) in prompt
    assert not messages[0].get('images')
    assert provider._client.chat.call_args.kwargs['options']['num_predict'] == 1234
    before = provider._client.chat.call_count
    assert all(r['llm_cached'] for r in await run())
    assert provider._client.chat.call_count == before
    case['expected'].write_text(receipt().model_copy(update={'merchant_name': 'EXPECTED SECRET'}).model_dump_json())
    result = await run()
    assert not result[1]['llm_cached']
    assert 'EXPECTED SECRET' not in provider._client.chat.call_args.kwargs['messages'][0]['content']
    rows[1]['status'] = 'error'
    result = await run()
    assert result[1]['llm_status'] == 'error'
    summary = comparison.summary(result)[1]
    assert summary['cases'] == 1 and summary['valid'] == 0


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'stale', 'raw', 'normalized', 'mixed'])
def test_cpu_loader_rejects_late_photo_integrity_and_mixed_contract(tmp_path, fault):
    from scripts.ocr_harness.historical import load_replay
    case = case_files(tmp_path)
    rows = cpu_rows(case, tmp_path)
    if fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows.append(copy.deepcopy(rows[1]))
    elif fault == 'stale': case['images'][1].write_bytes(b'changed')
    elif fault in {'raw', 'normalized'}:
        from pathlib import Path
        Path(rows[1]['raw_artifact' if fault == 'raw' else 'artifact']).write_text('{}')
    else: rows[1]['identity']['engine']['family'] = 'http'
    with pytest.raises(ValueError): load_replay(case, rows, 'pp-medium')


def test_native_never_falls_back_and_cpu_identity_tracks_late_adapter(tmp_path):
    from scripts.ocr_harness.historical import load_replay
    case = case_files(tmp_path)
    rows = cpu_rows(case, tmp_path)
    pages, _ = load_replay(case, rows, 'pp-medium')
    key = comparison.job_identity({}, case, 'ocr', pages, 10, 100)
    rows[1]['identity']['runtime']['image'] = 'sha256:new-image'
    changed, _ = load_replay(case, rows, 'pp-medium')
    assert comparison.job_identity({}, case, 'ocr', changed, 10, 100) != key
    rows[0]['identity']['engine']['family'] = 'http'
    rows[1]['identity']['engine']['family'] = 'http'
    with pytest.raises(ValueError, match='v2'): load_replay(case, rows, 'pp-medium')
    native = typed_rows(case, tmp_path)
    assert load_replay(case, native, 'native')[0][0]['contract_version'] == 2


@pytest.mark.parametrize('field', ['currency', 'tax_total'])
async def test_missing_financial_observations_never_inherit_defaults(tmp_path, field):
    from scripts.ocr_harness.historical import load_replay
    case = case_files(tmp_path)
    pages, _ = load_replay(case, cpu_rows(case, tmp_path), 'pp-medium')
    payload = multi_payload()
    payload.pop(field)
    provider = formatter(False, payload)
    with pytest.raises(ValueError):
        await comparison.compare_case(provider, 'fixture', case, 'ocr', pages, tmp_path, 5)
    assert provider._client.chat.call_count == 3


async def test_historical_replay_keeps_an_unprinted_discount_null(tmp_path):
    from scripts.ocr_harness.historical import load_replay

    case = case_files(tmp_path)
    pages, _ = load_replay(case, cpu_rows(case, tmp_path), 'pp-medium')
    payload = multi_payload()
    payload.pop('discount_total')
    provider = formatter(False, payload)

    result = await comparison.compare_case(provider, 'fixture', case, 'ocr', pages, tmp_path, 5)

    assert result.receipt.discount_total is None
    assert provider._client.chat.call_count == 1


async def test_cpu_uses_shared_repair_and_rejects_nonboundary_duplicate(tmp_path):
    from scripts.ocr_harness.historical import load_replay
    case = case_files(tmp_path)
    pages, _ = load_replay(case, cpu_rows(case, tmp_path), 'pp-medium')
    invalid = multi_payload()
    invalid['duplicate_candidates'] = [{'first_index': 0, 'second_index': 2}]
    provider = formatter(False, invalid)
    provider._client.chat.side_effect = [ChatResponse(message={'role':'assistant', 'content': json.dumps(p)}) for p in [invalid, multi_payload()]]
    result = await comparison.compare_case(provider, 'fixture', case, 'ocr', pages, tmp_path, 5)
    assert result.attempts == 2 and len(result.receipt.line_items) == 3
    assert provider.experiment_options is None  # live provider settings unchanged
