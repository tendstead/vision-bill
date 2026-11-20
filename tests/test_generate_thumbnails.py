import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_thumbnails.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_thumbnails", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_finds_file_under_extra_dir(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "receipt_1.png"
    target.write_bytes(b"x")
    # stored path is container-absolute; the file actually lives under tmp_path
    assert mod._resolve("/app/uploads/receipt_1.png", [str(tmp_path)]) == target


def test_resolve_uses_stored_path_when_it_exists(tmp_path: Path) -> None:
    mod = _load()
    direct = tmp_path / "direct.png"
    direct.write_bytes(b"x")
    assert mod._resolve(str(direct), []) == direct


def test_resolve_returns_none_when_missing(tmp_path: Path) -> None:
    mod = _load()
    assert mod._resolve("/app/uploads/nope.png", [str(tmp_path)]) is None


def test_resolve_empty_returns_none() -> None:
    mod = _load()
    assert mod._resolve("", []) is None


async def test_dry_run_counts_resolvable_candidates_without_decode_files_or_db_writes(tmp_path, monkeypatch, capsys, settings):
    from argparse import Namespace
    from unittest.mock import AsyncMock
    import asyncpg
    mod = _load()
    original = tmp_path / 'original.png'; original.write_bytes(b'synthetic-not-decoded')
    settings.images.save_dir = str(tmp_path / 'must-not-create')
    settings.images.tmp_dir = str(tmp_path / 'also-not-create')
    monkeypatch.setattr(mod, 'Settings', lambda: settings)
    conn = AsyncMock()
    conn.fetch.return_value = [{'id': 1, 'image_path': str(original)}, {'id': 2, 'image_path': str(tmp_path / 'missing.png')}]
    monkeypatch.setattr(asyncpg, 'connect', AsyncMock(return_value=conn))
    assert await mod._run(Namespace(limit=None, dry_run=True)) == 0
    output = capsys.readouterr().out
    assert 'would process 1 image(s)' in output
    assert '1 had no resolvable file' in output
    assert 'decoding not attempted' in output
    assert 'failed to decode' not in output
    conn.execute.assert_not_called()
    assert not Path(settings.images.save_dir).exists() and not Path(settings.images.tmp_dir).exists()
    assert original.read_bytes() == b'synthetic-not-decoded'
