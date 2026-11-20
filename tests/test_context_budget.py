"""Diagnostic request/arithmetic contract with fake HTTP; no model allocation."""
import json
import sys
from unittest.mock import Mock

import httpx
import pytest
from scripts import context_budget as cb


def test_prompt_only_provider_conforms_and_never_runs_inference():
    provider = cb._provider()
    assert 'merchant_name' in provider.build_prompt(['food'])


def test_missing_tags_context_falls_back_to_show_and_honors_cap(monkeypatch):
    b = cb.Budget('http://fake', 'model', 4096, 10)
    monkeypatch.setattr(b, '_get', lambda _: {'models': [{'name': 'model', 'details': {}}]})
    payloads = []
    def post(path, payload):
        payloads.append((path, payload))
        return {'model_info': {'llama.context_length': 8192}} if path == '/api/show' else {'prompt_eval_count': 120}
    monkeypatch.setattr(b, '_post', post)
    assert b.context_length() == 8192
    assert b.count('x' * 30000, image_b64='synthetic') == 120
    assert payloads[-1][1]['options'] == {'num_ctx': 4096, 'num_predict': 1}
    assert 'num_predict' not in payloads[-1][1]
    b.ctx = 2048
    b.count('short')
    assert payloads[-1][1]['options']['num_ctx'] == 2048


@pytest.mark.parametrize('count', [None, 0, 1023, 1024])
def test_unknown_or_context_edge_counts_are_not_reported_as_complete(count, monkeypatch):
    b = cb.Budget('http://fake', 'model', 1024, 10); b.ctx = 1024
    monkeypatch.setattr(b, '_post', lambda *a: {} if count is None else {'prompt_eval_count': count})
    with pytest.raises(ValueError, match='count|truncat|complete'):
        b.count('prompt')


def test_unknown_context_cannot_be_an_exact_complete_budget(monkeypatch):
    b = cb.Budget('http://fake', 'model', 1024, 10)
    monkeypatch.setattr(b, '_post', lambda *a: {'prompt_eval_count': 100})
    with pytest.raises(ValueError, match='context'):
        b.count('prompt')


def test_cli_counts_sample_image_and_reserved_output_once(tmp_path, monkeypatch, capsys):
    tags = tmp_path / 'tags.json'; tags.write_text('[]')
    image = tmp_path / 'image.png'; image.write_bytes(b'fake-image')
    monkeypatch.setattr(sys, 'argv', ['context_budget', '--model', 'fixture', '--tags-file', str(tags), '--image', str(image), '--output-budget', '200', '--tag-probe', '0', '--num-ctx-cap', '1000'])
    monkeypatch.setattr(httpx, 'Client', lambda **k: Mock(close=lambda: None))
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: httpx.Response(200, json={'models': [{'name': 'fixture', 'details': {'context_length': 1000}}]}, request=httpx.Request('GET', 'http://fake')))
    calls = []
    def post(url, *, json, **kwargs):
        calls.append(json)
        return httpx.Response(200, json={'prompt_eval_count': 700 if json.get('images') else 100}, request=httpx.Request('POST', 'http://fake'))
    monkeypatch.setattr(httpx, 'post', post)
    assert cb.main() == 0
    output = capsys.readouterr().out
    assert 'fits' in output and 'DOES NOT FIT' not in output
    assert '100 tokens' in output and 'remaining after prompt, sample image and reserved output' in output
    assert all(call['options'] == {'num_ctx': 1000, 'num_predict': 1} for call in calls)
