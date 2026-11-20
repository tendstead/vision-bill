"""Synthetic literal credentials only; no connection or secrets are involved."""
import inspect
import logging
import runpy
from contextlib import nullcontext
from unittest.mock import AsyncMock

import pytest
from asyncpg.connect_utils import _parse_connect_dsn_and_args
from sqlalchemy.engine import make_url
from vision_bill.config import PGSettings


def parsed(dsn):
    kwargs = dict.fromkeys(inspect.signature(_parse_connect_dsn_and_args).parameters)
    kwargs.update(dsn=dsn, ssl=False, direct_tls=False)
    return _parse_connect_dsn_and_args(**kwargs)


@pytest.mark.parametrize('host', ['localhost', '::1', '[::1]'])
@pytest.mark.parametrize('literal', ['ordinary', 'a@b/c#?d%2F%25: x'])
def test_runtime_and_migration_urls_preserve_literal_components(host, literal):
    pg = PGSettings(user=literal, password=literal, db=literal, host=host)
    addresses, params = parsed(pg.pg_dsn)
    assert params.user == params.password == params.database == literal
    assert addresses == [(host.strip('[]'), 5432)]
    _, compatibility = parsed(str(pg.database_url).replace('postgresql+asyncpg://', 'postgresql://'))
    assert compatibility.user == compatibility.password == compatibility.database == literal


def test_actual_alembic_environment_preserves_literal_driver_arguments(monkeypatch):
    import alembic.context
    from alembic.config import Config
    from vision_bill.config import settings
    pg = PGSettings(user='user@literal', password='secret%2F/#@', db='db%25/name', host='::1')
    monkeypatch.setattr(settings, 'pg', pg)
    monkeypatch.setattr(alembic.context, 'config', Config(), raising=False)
    monkeypatch.setattr(alembic.context, 'is_offline_mode', lambda: True)
    captured = {}
    monkeypatch.setattr(alembic.context, 'configure', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(alembic.context, 'begin_transaction', nullcontext)
    monkeypatch.setattr(alembic.context, 'run_migrations', lambda: None)
    runpy.run_path('alembic/env.py')
    url = make_url(captured['url'])
    assert (url.username, url.password, url.database, url.host) == (pg.user, pg.password, pg.db, '::1')
    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(url)
    _, kwargs = engine.dialect.create_connect_args(url)
    assert (kwargs['user'], kwargs['password'], kwargs['database'], kwargs['host']) == (pg.user, pg.password, pg.db, '::1')


@pytest.mark.parametrize('module_name,class_name', [('receipt_db', 'ReceiptDB'), ('image_db', 'ImageDB'), ('user_db', 'UserDB')])
async def test_pool_logs_are_content_free_at_info(module_name, class_name, monkeypatch, caplog):
    import importlib
    module = importlib.import_module('vision_bill.provider.db.' + module_name)
    monkeypatch.setattr(module.asyncpg, 'create_pool', AsyncMock(side_effect=RuntimeError('synthetic failure')))
    caplog.set_level(logging.INFO)
    with pytest.raises(RuntimeError):
        await getattr(module, class_name)(PGSettings(user='u', password='uniqueSyntheticSecret')).init_db()
    assert caplog.records
    assert 'uniqueSyntheticSecret' not in caplog.text and 'postgresql://' not in caplog.text


async def test_legacy_persistence_info_log_contains_no_receipt_observations(caplog):
    from test_receipt_db import _make_receipt
    from vision_bill.provider.db.receipt_db import ReceiptDB
    caplog.set_level(logging.INFO)
    receipt = _make_receipt('SyntheticPrivateMerchant')
    with pytest.raises(RuntimeError):
        await ReceiptDB(PGSettings(user='u', password='p')).persist_receipt(receipt)
    assert 'SyntheticPrivateMerchant' not in caplog.text
    assert str(receipt.date) not in caplog.text
