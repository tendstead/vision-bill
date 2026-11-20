"""Frozen Gold consent, current source identity, isolation and deletion on private PG18."""
import json
from uuid import uuid4

import asyncpg
import pytest

from tests.e2e import test_receipt_review_e2e as reviews
from vision_bill.model.workflow import AccessScope, ReviewRequest
from vision_bill.provider.db.benchmark_db import BenchmarkDB

postgres, db, workflow_db = reviews.postgres, reviews.db, reviews.workflow_db
pytestmark = pytest.mark.e2e


def bench(w):
    result = BenchmarkDB(w.pool)
    result._receipts = w._receipts
    return result


async def human(w, scope, tmp_path):
    row, body, path = await reviews.seed(w, scope, tmp_path)
    await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    return row, body, path


async def test_gold_requires_separate_current_human_confirmation(workflow_db, tmp_path):
    w, scope = workflow_db
    row, body, path = await human(w, scope, tmp_path)
    b = bench(w)
    assert hasattr(b, 'reference_preview'), 'scoped current confirmation preview missing'
    preview = await b.reference_preview(row.id, scope=scope)
    assert preview.review_revision == 1
    assert preview.sources[0]['part_id'] is None
    assert preview.sources[0]['photo_index'] == 0
    with pytest.raises(ValueError, match='qualified'):
        await b.current_reference(row.id, scope=scope)
    with pytest.raises(ValueError, match='confirmation'):
        await b.qualify_reference(row.id, 1, '0' * 64, scope=scope)
    frozen = await b.qualify_reference(row.id, 1, preview.document_sha256, scope=scope)
    assert frozen.qualified_by == scope.user_id and frozen.owner_id == scope.user_id
    assert frozen.expected['total'] == '10.00' or frozen.expected['total'] == '10'
    assert 'confidence' not in frozen.scored_fields and 'category' not in frozen.scored_fields
    body.expected_operation_revision = 2
    body.merchant_name = 'Clean edit'
    edited = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert not edited.verified  # legacy identity still requires ordinary review after editing
    await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    with pytest.raises(ValueError, match='qualified'):
        await b.current_reference(row.id, scope=scope)
    fresh = await b.reference_preview(row.id, scope=scope)
    with pytest.raises(ValueError, match='confirmation'):
        await b.qualify_reference(row.id, 1, preview.document_sha256, scope=scope)
    second = await b.qualify_reference(row.id, 2, fresh.document_sha256, scope=scope)
    assert second.id != frozen.id and frozen.expected['merchant_name'] == 'Shop'
    path.write_bytes(b'replaced legacy original')
    with pytest.raises(ValueError, match='confirmation'):
        await b.qualify_reference(row.id, 2, fresh.document_sha256, scope=scope)
    with pytest.raises(ValueError, match='qualified'):
        await b.current_reference(row.id, scope=scope)


async def test_gold_preserves_unprinted_line_factors_and_adjustments(workflow_db, tmp_path):
    from decimal import Decimal

    from vision_bill.model.receipt import Receipt
    from vision_bill.service.experiment_scoring import detailed_score

    w, scope = workflow_db
    row, body, _ = await reviews.seed(w, scope, tmp_path)
    body.line_items[0].quantity = None
    body.line_items[0].unit_price = None
    body.discount_total = None
    await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    b = bench(w)
    preview = await b.reference_preview(row.id, scope=scope)
    frozen = await b.qualify_reference(
        row.id, preview.review_revision, preview.document_sha256, scope=scope,
    )

    assert frozen.expected['discount_total'] is None and frozen.expected['tip'] is None
    assert frozen.expected['line_items'][0]['quantity'] is None
    assert frozen.expected['line_items'][0]['unit_price'] is None
    assert 'discount_total' not in frozen.scored_fields
    actual = Receipt.model_validate(frozen.expected).model_copy(update={'discount_total': Decimal(0)})
    assert detailed_score(frozen.expected, actual, scored_fields=frozen.scored_fields)['full_receipt_correct']


@pytest.mark.parametrize('source', ['legacy', 'automatic_bypass', None])
async def test_verified_flag_never_qualifies_gold(workflow_db, tmp_path, source):
    w, scope = workflow_db
    row, _, _ = await reviews.seed(w, scope, tmp_path, verified=True)
    async with w.pool.acquire() as conn:
        await conn.execute('UPDATE receipts SET verification_source=$2 WHERE id=$1', row.id, source)
    b = bench(w)
    assert hasattr(b, 'reference_preview'), 'human provenance gate missing'
    with pytest.raises(ValueError, match='human'):
        await b.reference_preview(row.id, scope=scope)


async def test_gold_owner_scope_immutable_history_survives_receipt_and_user_delete(workflow_db, tmp_path):
    w, scope = workflow_db
    row, _, _ = await human(w, scope, tmp_path)
    b = bench(w)
    assert hasattr(b, 'reference_preview'), 'reference storage missing'
    other = uuid4()
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO users(id,username,hashed_password,is_admin) VALUES($1,$2,'test',true)", other, str(other))
    with pytest.raises(ValueError, match='not found'):
        await b.reference_preview(row.id, scope=AccessScope(user_id=other))
    admin_scope = AccessScope(user_id=other, can_see_all=True)
    preview = await b.reference_preview(row.id, scope=admin_scope)
    frozen = await b.qualify_reference(row.id, 1, preview.document_sha256, scope=admin_scope)
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.RaiseError, match='immutable'):
            await conn.execute("UPDATE benchmark_references SET expected='{}' WHERE id=$1", frozen.id)
        with pytest.raises(asyncpg.RaiseError, match='immutable'):
            await conn.execute('DELETE FROM benchmark_references WHERE id=$1', frozen.id)
        await conn.execute('DELETE FROM receipts WHERE id=$1', row.id)
        await conn.execute('DELETE FROM users WHERE id=ANY($1::uuid[])', [other, scope.user_id])
        saved = await conn.fetchrow('SELECT * FROM benchmark_references WHERE id=$1', frozen.id)
    assert saved['receipt_id'] is None and saved['owner_id'] is None and saved['qualified_by'] is None
    assert json.loads(saved['expected']) == frozen.expected
    assert saved['receipt_identity'] == row.id and saved['owner_identity'] == scope.user_id
    assert saved['actor_identity'] == other


async def test_clean_durable_edit_can_attest_current_gold_without_second_verify(workflow_db, tmp_path):
    import hashlib
    w, scope = workflow_db
    row, body, path = await human(w, scope, tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO image_source_parts(part_id,image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,$2,0,$3,$4,'{}')",
                           uuid4(), row.image_id, hashlib.sha256(path.read_bytes()).hexdigest(), str(path))
    body.expected_operation_revision = 2
    body.merchant_name = 'Current human attestation'
    edited = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert edited.verified and edited.verification_source == 'human'
    b = bench(w)
    preview = await b.reference_preview(row.id, scope=scope)
    gold = await b.qualify_reference(row.id, 2, preview.document_sha256, scope=scope)
    assert gold.review_revision == 2 and gold.expected['merchant_name'] == body.merchant_name


async def test_new_runs_scope_owner_and_preserve_unknown_historical_ownership(workflow_db, tmp_path):
    from vision_bill.model.benchmark import CouncilPolicy
    w, scope = workflow_db
    row, _, _ = await human(w, scope, tmp_path)
    b = bench(w)
    kwargs = dict(model_ids=['m'], receipt_ids=[row.id], fingerprint='f', model_digests={}, timeout=600,
                  policy=CouncilPolicy.ALL, absolute=None, relative=None, apply_flags=False,
                  prompt_version='4', scoring_version='4')
    run = await b.create_run(**kwargs, owner_id=scope.user_id)
    stranger = AccessScope(user_id=uuid4())
    assert await b.run(run.id, scope=stranger) is None
    assert await b.runs(scope=stranger) == []
    assert (await b.run(run.id, scope=scope)).id == run.id
    async with w.pool.acquire() as conn:
        await conn.execute('UPDATE benchmark_runs SET owner_id=NULL WHERE id=$1', run.id)
    assert await b.run(run.id, scope=scope) is None
    assert (await b.run(run.id, scope=AccessScope(user_id=scope.user_id,can_see_all=True))).id == run.id
    # Active tasks retain their existing temporary deletion refusal.
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await conn.execute('DELETE FROM receipts WHERE id=$1', row.id)


async def test_two_tax_roundtrip_keeps_printed_positions_and_source_paths(workflow_db, tmp_path):
    from uuid import UUID
    from vision_bill.model.receipt import TaxLine
    from vision_bill.model.ocr import SourceRef
    import hashlib
    w, scope = workflow_db
    row, body, path = await reviews.seed(w, scope, tmp_path)
    part_id = uuid4()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO image_source_parts(part_id,image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,$2,0,$3,$4,'{}')", part_id, row.image_id, source_hash, str(path))
    body.taxes = [TaxLine(name='VAT seven', rate=0.07, amount='1'), TaxLine(name='VAT nineteen', rate=0.19, amount='2')]
    from decimal import Decimal
    body.tax_total = Decimal(3)
    body.amount_policy = reviews.POLICY.model_copy(update={'tax_information': 'present'})
    body.field_sources = {f'taxes.{i}.amount': [SourceRef(part_id=part_id,photo_index=0,source_sha256=source_hash,block_id=f'printed-tax-{i}')] for i in range(2)}
    saved = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE taxes SET id=$2 WHERE receipt_id=$1 AND name='VAT seven'",row.id,UUID('ffffffff-ffff-ffff-ffff-ffffffffffff'))
        await conn.execute("UPDATE taxes SET id=$2 WHERE receipt_id=$1 AND name='VAT nineteen'",row.id,UUID('00000000-0000-0000-0000-000000000001'))
    details = await w._receipts.get_receipt_with_details(row.id)
    assert [tax.name for tax in details.taxes] == ['VAT seven','VAT nineteen']
    assert [tax.id for tax in details.taxes] == [UUID('ffffffff-ffff-ffff-ffff-ffffffffffff'), UUID('00000000-0000-0000-0000-000000000001')]
    for index in range(2):
        source = details.receipt.field_sources[f'taxes.{index}.amount'][0]
        assert source.block_id == f'printed-tax-{index}'
        assert source.part_id == part_id and source.source_sha256 == source_hash
    async with w.pool.acquire() as conn:
        reread = await w._receipts._review_document(conn, await conn.fetchrow('SELECT * FROM receipts WHERE id=$1',row.id))
    from vision_bill.service.receipt_review import material_document
    assert material_document(reread) == material_document(body)
    assert saved.findings == details.receipt.findings


async def test_run_uses_explicit_frozen_reference_and_never_feeds_expected_to_model(workflow_db,tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from ollama import ChatResponse
    from vision_bill.config import ImageSettings, OcrSettings
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.model.ocr import OcrCapabilities, OcrResponse
    from tests.test_ocr_provider import wire_page
    w, scope = workflow_db
    row, body, path = await human(w,scope,tmp_path)
    # Two ordered originals exercise the same service path as live uploads.
    second = tmp_path / 'second'
    second.write_bytes(b'second original')
    async with w.pool.acquire() as conn:
        await conn.execute('UPDATE images SET additional_images=$2 WHERE id=$1',row.image_id,json.dumps([{'image_path':str(second)}]))
    provider = OllamaProvider('http://local')
    model = ModelInfo('fixture'); model.capabilities=['completion','vision']
    provider.get_available_models = AsyncMock(return_value=[model])
    from vision_bill.model.receipt import Receipt
    payload = Receipt.model_validate(body.model_dump()).model_dump(mode='json')
    payload['merchant_name'] = 'Observed from provider'
    payload['line_items'][0]['source_image'] = 0
    payload['duplicate_candidates'] = []
    provider._client.chat = AsyncMock(return_value=ChatResponse(message={'role':'assistant','content':json.dumps(payload)}))
    async def page(path,*,photo_index,source_sha256,**kwargs):
        return OcrResponse.model_validate(wire_page(photo_index,source_sha256,text=f'Only observed photo {photo_index}'))
    ocr = SimpleNamespace(get_capabilities=AsyncMock(return_value=OcrCapabilities(enabled=True,configured=True,reachable=True,ready=True,profile_id='fixed-profile',runtime_fingerprint='runtime')),extract_page=page)
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path),tmp_dir=str(tmp_path)),w._settings,provider,ocr_provider=ocr,ocr_settings=OcrSettings(enabled=True, manage_model_lifecycle=False))
    receipts._db = w._receipts
    service = BenchmarkService(provider,receipts)
    request = BenchmarkCreate(model_ids=['fixture'],receipt_ids=[row.id],experiment=ExperimentOptions(pipeline='ocr',include_classification=False))
    with pytest.raises(ValueError,match='qualified'):
        await service.create_run(request,scope=scope)
    preview = await service.reference_preview(row.id,scope=scope)
    gold = await service.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
    run = await service.create_run(request,scope=scope)
    manifest = await service._db.manifest(run.id)
    assert manifest['inputs'][0]['reference']['id'] == str(gold.id)
    assert len(manifest['inputs'][0]['images']) == 2
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE receipts SET merchant_name='REFERENCE_SENTINEL' WHERE id=$1",row.id)
    task = await service._db.lease_task()
    await service._execute(dict(task))
    result = (await service.results(run.id,limit=10,offset=0,scope=scope))[0]
    assert result['success'] and result['expected']['merchant_name']=='Shop', result
    assert result['metadata']['source_image_count']==2
    assert len(result['metadata']['ocr_pages'])==2
    prompt = provider._client.chat.call_args.kwargs['messages'][0]['content']
    assert 'REFERENCE_SENTINEL' not in prompt and 'Shop' not in prompt
    before = await service._db.results(run.id)
    reevaluated = await service.reevaluate(run.id,row.id,'fixture',scope=scope)
    assert reevaluated['expected'] == gold.expected
    assert await service._db.results(run.id)==before
    original_execute = receipts.analyse_submission_with_metadata
    async def changed_sources(*args, **kwargs):
        result = await original_execute(*args, **kwargs)
        result.metadata['ordered_source_sha256'] = list(reversed(result.metadata['ordered_source_sha256']))
        return result
    receipts.analyse_submission_with_metadata = changed_sources
    with pytest.raises(ValueError, match='source identity'):
        await service.reevaluate(run.id,row.id,'fixture',scope=scope)
    assert await service._db.results(run.id)==before


async def test_runtime_role_enforces_reference_immutability_and_current_scope(workflow_db,tmp_path):
    w,scope=workflow_db
    row,_,_=await human(w,scope,tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute('CREATE ROLE benchmark_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE')
        await conn.execute('GRANT USAGE ON SCHEMA public TO benchmark_runtime')
        await conn.execute('GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA public TO benchmark_runtime')
    async def init(conn): await conn.execute('SET ROLE benchmark_runtime')
    pool=await asyncpg.create_pool(w._settings.pg_dsn,init=init,min_size=1,max_size=2)
    try:
        b=BenchmarkDB(pool,w._receipts)
        preview=await b.reference_preview(row.id,scope=scope)
        gold=await b.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
        async with pool.acquire() as conn:
            assert not await conn.fetchval("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
            with pytest.raises(asyncpg.RaiseError,match='immutable'):
                await conn.execute('UPDATE benchmark_references SET owner_id=NULL WHERE id=$1',gold.id)
        with pytest.raises(ValueError,match='not found'):
            await b.reference_preview(row.id,scope=AccessScope(user_id=uuid4()))
    finally: await pool.close()


async def test_qualification_race_before_run_creation_is_rejected(workflow_db,tmp_path):
    from vision_bill.model.benchmark import CouncilPolicy
    w,scope=workflow_db
    row,_,path=await human(w,scope,tmp_path)
    b=bench(w)
    preview=await b.reference_preview(row.id,scope=scope)
    gold=await b.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
    manifest={'inputs':[{'receipt_id':str(row.id),'reference':gold.model_dump(mode='json')} ]}
    path.write_bytes(b'changed after selection before insertion')
    with pytest.raises(ValueError,match='reference'):
        await b.create_run(model_ids=['m'],receipt_ids=[row.id],fingerprint='f',model_digests={},timeout=600,
            policy=CouncilPolicy.ALL,absolute=None,relative=None,apply_flags=False,prompt_version='4',scoring_version='4',
            manifest=manifest,owner_id=scope.user_id,scope=scope)


async def test_upgrade_keeps_legacy_tax_order_and_unknown_run_owner(db):
    from tests.e2e.test_workflow_db_e2e import migrate
    from uuid import UUID
    migrate(db,'0015')
    conn=await asyncpg.connect(db.pg_dsn)
    try:
        row=await conn.fetchval("INSERT INTO receipts(merchant_name,date,confidence,subtotal,total) VALUES('legacy','2026-01-01',90,10,10) RETURNING id")
        ids=[UUID('ffffffff-ffff-ffff-ffff-ffffffffffff'),UUID('00000000-0000-0000-0000-000000000001')]
        await conn.execute("INSERT INTO taxes(id,receipt_id,name,rate,amount) VALUES($1,$3,'first inserted',19,1.234567),($2,$3,'second inserted',7,2.345678)",*ids,row)
        before=await conn.fetch('SELECT id,rate,amount FROM taxes ORDER BY id')
        migrate(db, '0016')
        after=await conn.fetch('SELECT id,rate,amount FROM taxes ORDER BY position,id')
        assert before==after
        assert await conn.fetchval('SELECT version_num FROM alembic_version')=='0016'
        assert migrate(db,'0015','downgrade',check=False).returncode==0
    finally: await conn.close()


async def test_new_summary_counts_extraction_failures_without_redefining_success_average(workflow_db,tmp_path):
    from vision_bill.model.benchmark import CouncilPolicy
    w,scope=workflow_db
    first,_,_=await human(w,scope,tmp_path)
    second,_,_=await human(w,scope,tmp_path)
    b=bench(w)
    run=await b.create_run(model_ids=['m'],receipt_ids=[first.id,second.id],fingerprint='f',model_digests={},timeout=600,
        policy=CouncilPolicy.ALL,absolute=None,relative=None,apply_flags=False,prompt_version='4',scoring_version='4',owner_id=scope.user_id)
    tasks = {task['receipt_id']: task for task in [await b.lease_task(), await b.lease_task()]}
    await b.finish_task(await b.begin_attempt(tasks[first.id]),success=True,score=1,result={'version':'4','full_receipt_correct':True})
    await b.finish_task(await b.begin_attempt(tasks[second.id]),success=False,result={'version':'4','full_receipt_correct':False})
    summary=(await b.summaries(run.id))[0]
    assert summary.average_score==1
    assert summary.all_case_average_score==.5
    assert summary.full_receipt_correct_rate==.5
    assert summary.completed==2 and summary.failed==1


@pytest.mark.parametrize('evidence', ['sources', 'raw_draft', 'tax_policy'])
async def test_observed_defaults_are_frozen_scored_and_stale_masks_rejected(workflow_db, tmp_path, evidence):
    import hashlib
    from vision_bill.model.receipt import Receipt
    from vision_bill.service.experiment_scoring import detailed_score
    w, scope = workflow_db
    row, _, path = await human(w, scope, tmp_path)
    b = bench(w)
    async def acknowledge_current_findings():
        from vision_bill.service.receipt_review import evaluate_receipt, persistent_findings
        async with w.pool.acquire() as conn:
            record = await conn.fetchrow('SELECT * FROM receipts WHERE id=$1', row.id)
            current = w._receipts._receipt_row_from_record(record)
            document = await w._receipts._review_document(conn, record)
            originals = await w._receipts._review_originals(conn, row.image_id)
            evaluation = evaluate_receipt(document, current.amount_policy, current.field_sources,
                current.overlap_decisions, originals, retained_findings=persistent_findings(current.findings))
            acknowledgements = [{'finding_id':f.id,'comment':'Human checked the original'} for f in evaluation.findings if f.severity == 'review']
            await conn.execute('UPDATE receipts SET acknowledgements=$2::jsonb WHERE id=$1', row.id,json.dumps(acknowledgements))
    # Unknown legacy defaults are not declarations that these values were printed.
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE receipts SET amount_policy=jsonb_set(amount_policy,'{tax_information}','\"missing\"') WHERE id=$1", row.id)
        if evidence in {'sources', 'raw_draft'}:
            part = uuid4()
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            await conn.execute("INSERT INTO image_source_parts(part_id,image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,$2,0,$3,$4,'{}')",part,row.image_id,sha,str(path))
    await acknowledge_current_findings()
    before = await b.reference_preview(row.id, scope=scope)
    assert not {'currency', 'discount_total', 'tax_total'} & set(before.scored_fields)
    legacy = await b.qualify_reference(row.id, before.review_revision, before.document_sha256, scope=scope)
    async with w.pool.acquire() as conn:
        if evidence == 'sources':
            refs = {key: [{'part_id':str(part),'photo_index':0,'source_sha256':sha,'block_id':key}] for key in ('currency','discount_total','tax_total')}
            await conn.execute('UPDATE receipts SET field_sources=$2::jsonb WHERE id=$1',row.id,json.dumps(refs))
        elif evidence == 'raw_draft':
            extraction = await conn.fetchval("INSERT INTO receipt_extractions(image_id,generation,attempt_number,outcome_kind,envelope) VALUES($1,1,1,'complete',$2::jsonb) RETURNING id", row.image_id,json.dumps({'draft':{'currency':'EUR','discount_total':'0','tax_total':'0'}}))
            await conn.execute('UPDATE receipts SET extraction_id=$2 WHERE id=$1',row.id,extraction)
        else:
            await conn.execute("UPDATE receipts SET amount_policy=jsonb_set(amount_policy,'{tax_information}','\"explicit_zero\"') WHERE id=$1", row.id)
    await acknowledge_current_findings()
    preview = await b.reference_preview(row.id, scope=scope)
    observed = {'tax_total'} if evidence == 'tax_policy' else {'currency','discount_total','tax_total'}
    assert observed <= set(preview.scored_fields)
    assert preview.sources == before.sources and preview.expected == before.expected
    assert preview.document_sha256 != before.document_sha256
    with pytest.raises(ValueError, match='confirmation'):
        await b.qualify_reference(row.id,before.review_revision,before.document_sha256,scope=scope)
    with pytest.raises(ValueError, match='qualified'):
        await b.current_reference(row.id, scope=scope)
    gold = await b.qualify_reference(row.id,preview.review_revision,preview.document_sha256,scope=scope)
    assert (await b.current_reference(row.id,scope=scope)).model_dump() == gold.model_dump()
    assert legacy.scored_fields == before.scored_fields
    expected = Receipt.model_validate(gold.expected)
    assert detailed_score(gold.expected,expected,scored_fields=gold.scored_fields)['full_receipt_correct']
    for field in observed:
        wrong = expected.model_copy(update={field: 'USD' if field == 'currency' else __import__('decimal').Decimal('1')})
        assert not detailed_score(gold.expected,wrong,scored_fields=gold.scored_fields)['full_receipt_correct']
        assert detailed_score(legacy.expected,wrong,scored_fields=legacy.scored_fields)['full_receipt_correct']


async def test_filtered_creation_qualifies_before_limit_and_explicit_ids_are_strict(workflow_db, tmp_path):
    from datetime import date
    from unittest.mock import AsyncMock
    from vision_bill.config import ImageSettings, OcrSettings
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.benchmark_service import BenchmarkService
    w, scope = workflow_db
    foreign_scope = AccessScope(user_id=uuid4())
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO users(id,username,hashed_password,is_admin) VALUES($1,$2,'test',true)",foreign_scope.user_id,str(foreign_scope.user_id))
    b = bench(w)
    rows = {}
    for index, name in enumerate(('valid', 'missing_source', 'stale', 'unqualified', 'foreign', 'wrong_category', 'high_confidence', 'unverified'), 1):
        owner = foreign_scope if name == 'foreign' else scope
        row, _, path = await human(w,owner,tmp_path)
        async with w.pool.acquire() as conn:
            await conn.execute('UPDATE receipts SET date=$2,category=$3,confidence=$4 WHERE id=$1', row.id,date(2026,1,index),'grocery' if name != 'wrong_category' else 'other',99 if name == 'high_confidence' else 50)
        if name not in {'unqualified','unverified'}:
            preview = await b.reference_preview(row.id,scope=owner)
            await b.qualify_reference(row.id,preview.review_revision,preview.document_sha256,scope=owner)
        rows[name] = row.id
        if name == 'missing_source': path.unlink()
        if name in {'stale','unverified'}:
            async with w.pool.acquire() as conn:
                if name == 'stale': await conn.execute("UPDATE receipts SET merchant_name='edited' WHERE id=$1",row.id)
                else: await conn.execute("UPDATE receipts SET verified=false,status='unverified' WHERE id=$1",row.id)
    provider = OllamaProvider('http://local')
    model = ModelInfo('fixture'); model.capabilities=['completion','vision']
    provider.get_available_models = AsyncMock(return_value=[model])
    provider._client.chat = AsyncMock(side_effect=AssertionError('selection must not infer'))
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path),tmp_dir=str(tmp_path)),w._settings,provider,ocr_settings=OcrSettings(enabled=False))
    receipts._db = w._receipts
    service = BenchmarkService(provider,receipts)
    request = BenchmarkCreate(model_ids=['fixture'],category='grocery',max_source_confidence=75,limit=1)
    run = await service.create_run(request,scope=scope)
    assert run.receipt_ids == [rows['valid']]
    manifest = await b.manifest(run.id)
    assert manifest['inputs'][0]['receipt_id'] == str(rows['valid'])
    # A restricted admin cannot include a foreign reference; configured see-all can.
    privileged = AccessScope(user_id=scope.user_id, can_see_all=True)
    admin_run = await service.create_run(request, scope=privileged)
    assert admin_run.receipt_ids == [rows['foreign']]
    mixed = await service.create_run(
        BenchmarkCreate(model_ids=["fixture"], receipt_ids=[rows["valid"], rows["foreign"]]),
        scope=privileged,
    )
    async with w.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO benchmark_results(run_id,model_id,receipt_id,success,payload) "
            "VALUES($1,'fixture',$2,TRUE,$3::jsonb)",
            mixed.id, rows["foreign"], json.dumps({"expected": {"merchant_name": "Foreign data"}}),
        )
        before = await conn.fetchval('SELECT count(*) FROM benchmark_runs')

    assert {listed.id for listed in await service.list_runs(scope=privileged)} == {
        run.id, admin_run.id, mixed.id,
    }
    assert {listed.id for listed in await service.list_runs(scope=scope)} == {run.id}
    assert await b.run(mixed.id) is not None  # Workers continue to own old jobs.
    assert await service.get_status(run.id, scope=scope) is not None
    assert await service.get_status(mixed.id, scope=privileged) is not None
    assert await service.get_status(mixed.id, scope=scope) is None
    assert (await service.results(mixed.id, limit=5, offset=0, scope=privileged))[0]["expected"] == {
        "merchant_name": "Foreign data",
    }
    with pytest.raises(ValueError, match="Benchmark run not found"):
        await service.results(mixed.id, limit=5, offset=0, scope=scope)
    with pytest.raises(ValueError, match="not part of this benchmark run"):
        await service.reevaluate(mixed.id, rows["foreign"], "fixture", scope=scope)

    for invalid in [uuid4(), *[rows[name] for name in ('foreign','unqualified','stale','missing_source','unverified')]]:
        with pytest.raises((ValueError,FileNotFoundError)):
            await service.create_run(BenchmarkCreate(model_ids=['fixture'],receipt_ids=[rows['valid'],invalid],limit=1),scope=scope)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM benchmark_runs') == before
    provider._client.chat.assert_not_awaited()


@pytest.mark.parametrize('time,canonical', [('14:30:00.000001','14:30:00.000001'), ('14:30:15.1','14:30:15.100000')])
async def test_public_human_null_completion_remains_observed_in_gold(workflow_db, tmp_path, time, canonical):
    from decimal import Decimal
    from uuid import UUID
    import httpx
    from fastapi import FastAPI
    from vision_bill.api.drafts import router as drafts_router
    from vision_bill.api.receipts import router as receipts_router
    from vision_bill.api.benchmarks import router as benchmarks_router
    from vision_bill.api.helper.helper import get_receipt_service, get_benchmark_service
    from vision_bill.config import ImageSettings
    from vision_bill.model.receipt import Receipt
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.security.dependencies import get_current_user
    from vision_bill.security.models import User
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.submission_service import SubmissionService
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.service.experiment_scoring import detailed_score
    from tests.e2e import test_workflow_db_e2e as workflow
    w, scope = workflow_db
    w._receipts._pool = w.pool
    image = await workflow.partial_draft(w, scope, tmp_path/'human-original')
    async with w.pool.acquire() as conn:
        raw_before = await conn.fetchval('SELECT envelope FROM receipt_extractions WHERE image_id=$1',image.id)
    raw_draft = json.loads(raw_before)['draft']
    assert raw_draft['currency'] is None and raw_draft['discount_total'] is None
    provider = OllamaProvider('http://local')
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path),tmp_dir=str(tmp_path)),w._settings,provider)
    receipts._db = w._receipts
    submissions = object.__new__(SubmissionService)
    submissions.db = w
    benchmarks = BenchmarkService(provider, receipts)
    app = FastAPI()
    app.include_router(drafts_router,prefix='/images')
    app.include_router(receipts_router,prefix='/receipts')
    app.include_router(benchmarks_router,prefix='/benchmarks')
    app.state.submission_service = submissions
    app.dependency_overrides[get_current_user] = lambda: User(id=scope.user_id,username='owner',is_admin=True,can_see_all=False)
    app.dependency_overrides[get_receipt_service] = lambda: receipts
    app.dependency_overrides[get_benchmark_service] = lambda: benchmarks
    changes = workflow.complete(total='10').envelope.receipt.model_dump(mode='json')
    changes.update(currency='EUR',discount_total='0',taxes=[],tax_total='0',amount_policy=reviews.POLICY.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        initial = (await client.get(f'/images/{image.id}/draft')).json()
        assert initial['document']['time'] is None
        partial = await client.patch(f'/images/{image.id}/draft',json={'expected_revision':1,'changes':{'merchant_name':'Human'}})
        assert partial.status_code == 200, partial.text
        assert partial.json()['document']['time'] is None and partial.json()['document']['total'] is None
        invalid = await client.patch(f'/images/{image.id}/draft',json={'expected_revision':2,'changes':{'time':'14:30:00.1234567'}})
        assert invalid.status_code == 422
        assert (await client.get(f'/images/{image.id}/draft')).json() == partial.json()
        changes['time'] = time
        patched = await client.patch(f'/images/{image.id}/draft',json={'expected_revision':2,'changes':changes})
        assert patched.status_code == 200, patched.text
        assert patched.json()['field_sources'] == {}
        assert patched.json()['document']['time'] == canonical
        assert (await client.get(f'/images/{image.id}/draft')).json()['document']['time'] == canonical
        finalized = await client.post(f'/images/{image.id}/draft/finalize',json={'expected_revision':3,'request_key':'human-final'})
        assert finalized.status_code == 200, finalized.text
        receipt_id = UUID(finalized.json()['id'])
        assert (await client.post(f'/images/{image.id}/draft/finalize',json={'expected_revision':3,'request_key':'human-final'})).json()['id'] == str(receipt_id)
        assert (await client.get(f'/receipts/{receipt_id}')).json()['receipt']['time'] == canonical
        assert finalized.json()['findings'] == []
        verified = await client.post(f'/receipts/{receipt_id}/verify')
        assert verified.status_code == 200, verified.text
        preview = await client.get(f'/benchmarks/references/{receipt_id}')
        assert preview.status_code == 200, preview.text
        current = preview.json()
        assert {'currency','discount_total'} <= set(current['scored_fields'])
        assert (await client.get(f'/benchmarks/references/{receipt_id}')).json()['document_sha256'] == current['document_sha256']
        qualified = await client.post(f'/benchmarks/references/{receipt_id}',json={'expected_review_revision':current['review_revision'],'expected_document_sha256':current['document_sha256']})
        assert qualified.status_code == 200, qualified.text
        assert (await client.get(f'/benchmarks/references/{receipt_id}')).json() == current
    gold = await benchmarks._db.current_reference(receipt_id,scope=scope)
    assert str(gold.id) == qualified.json()['id'] and gold.scored_fields == current['scored_fields']
    expected = Receipt.model_validate(gold.expected)
    assert gold.expected['time'] == canonical
    assert detailed_score(gold.expected, expected, scored_fields=gold.scored_fields)['full_receipt_correct']
    assert not detailed_score(gold.expected, expected.model_copy(update={'time':'14:30:00.000002'}), scored_fields=gold.scored_fields)['full_receipt_correct']
    async with w.pool.acquire() as conn:
        frozen = await conn.fetchrow('SELECT * FROM benchmark_references WHERE id=$1',gold.id)
    body = reviews.ReceiptUpdate.model_validate(expected.model_dump(mode='json') | {'time':'14:30:00.000002','expected_operation_revision':2})
    edited = await w._receipts.update_receipt(receipt_id,body,user_id=scope.user_id)
    assert edited.verified and edited.review_revision == 2  # Existing clean-edit policy.
    fresh = await benchmarks.reference_preview(receipt_id,scope=scope)
    assert fresh.document_sha256 != current['document_sha256']
    await benchmarks.qualify_reference(receipt_id,fresh.review_revision,fresh.document_sha256,scope=scope)
    async with w.pool.acquire() as conn:
        assert await conn.fetchrow('SELECT * FROM benchmark_references WHERE id=$1',gold.id) == frozen
    for field, wrong in [('currency','USD'),('discount_total',Decimal('1'))]:
        assert not detailed_score(gold.expected,expected.model_copy(update={field:wrong}),scored_fields=gold.scored_fields)['full_receipt_correct']
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT envelope FROM receipt_extractions WHERE image_id=$1',image.id) == raw_before
    legacy,_,_ = await human(w,scope,tmp_path)
    legacy_preview = await benchmarks.reference_preview(legacy.id,scope=scope)
    assert not {'currency','discount_total'} & set(legacy_preview.scored_fields)


async def test_reference_pool_failure_aborts_before_partial_run(workflow_db,tmp_path):
    from unittest.mock import AsyncMock
    from vision_bill.config import ImageSettings, OcrSettings
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.benchmark_service import BenchmarkService
    w,scope = workflow_db
    b = bench(w)
    for _ in range(2):
        row,_,_ = await human(w,scope,tmp_path)
        preview = await b.reference_preview(row.id,scope=scope)
        await b.qualify_reference(row.id,preview.review_revision,preview.document_sha256,scope=scope)
    provider = OllamaProvider('http://local')
    model = ModelInfo('fixture'); model.capabilities=['vision','completion']
    provider.get_available_models = AsyncMock(return_value=[model])
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path),tmp_dir=str(tmp_path)),w._settings,provider,ocr_settings=OcrSettings(enabled=False))
    receipts._db = w._receipts
    service = BenchmarkService(provider,receipts)
    class ReconnectingPool:
        calls = 0
        def acquire(self):
            self.calls += 1
            if self.calls == 2:  # Candidate enumeration succeeds; first reference acquisition fails.
                raise ConnectionRefusedError('private pool reconnect failed')
            return w.pool.acquire()
    service._db._pool = ReconnectingPool()
    with pytest.raises(ConnectionRefusedError,match='private pool reconnect failed'):
        await service.create_run(BenchmarkCreate(model_ids=['fixture'],limit=1),scope=scope)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval('SELECT count(*) FROM benchmark_runs') == 0
        assert await conn.fetchval('SELECT count(*) FROM benchmark_tasks') == 0


async def test_reference_hash_read_unavailability_is_a_domain_failure(workflow_db,tmp_path,monkeypatch):
    from vision_bill.service import experiment_execution
    w,scope = workflow_db
    row,_,path = await human(w,scope,tmp_path)
    assert path.is_file()  # Initial integrity check succeeds; the later hash open fails.
    def vanished(_path):
        raise FileNotFoundError('original vanished during hash read')
    monkeypatch.setattr(experiment_execution,'image_digest',vanished)
    with pytest.raises(ValueError,match='original_integrity_failed'):
        await bench(w).reference_preview(row.id,scope=scope)


async def test_readonly_gold_dataset_freezes_current_owner_references(workflow_db,tmp_path):
    from scripts.standalone_benchmark import freeze_references, fetch_receipt_details
    from datetime import time
    w,scope=workflow_db
    row,body,path=await human(w,scope,tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE receipts SET time=$2 WHERE id=$1",row.id,time(14,30,15,1))
    assert (await fetch_receipt_details(w.pool,row.id)).time=="14:30:15.000001"
    b=bench(w)
    # Exercise both source-reference validation and original validation without FOR SHARE.
    from hashlib import sha256
    part=uuid4();digest=sha256(path.read_bytes()).hexdigest()
    source={"part_id":str(part),"photo_index":0,"source_sha256":digest,"coordinate_space":"original"}
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO image_source_parts(part_id,image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,$2,0,$3,$4,'{}')",part,row.image_id,digest,str(path))
        await conn.execute("UPDATE receipts SET field_sources=$2::jsonb WHERE id=$1",row.id,json.dumps({"total":[source]}))
    preview=await b.reference_preview(row.id,scope=scope)
    gold=await b.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
    # Genuine SELECT-only role; read-only transaction must use no row locks.
    async with w.pool.acquire() as conn:
        await conn.execute('CREATE ROLE standalone_reader')
        await conn.execute('GRANT USAGE ON SCHEMA public TO standalone_reader')
        await conn.execute('GRANT SELECT ON ALL TABLES IN SCHEMA public TO standalone_reader')
    async def reader(conn): await conn.execute('SET ROLE standalone_reader')
    pool=await asyncpg.create_pool(dsn=w._settings.pg_dsn,setup=reader)
    try:
        frozen=await freeze_references(pool,w._settings,scope,limit=1,category=None)
        assert len(frozen)==1 and frozen[0].id==gold.id
        assert frozen[0].sources[0]['sha256']==gold.sources[0]['sha256']
        with pytest.raises(ValueError,match='application user not found'):
            await freeze_references(pool,w._settings,AccessScope(user_id=uuid4()),limit=1,category=None)
    finally:
        await pool.close()
        async with w.pool.acquire() as conn:
            await conn.execute('DROP OWNED BY standalone_reader')
            await conn.execute('DROP ROLE standalone_reader')


async def test_standalone_gold_selection_is_one_snapshot_and_excludes_bypass_legacy_stale(workflow_db,tmp_path,monkeypatch):
    from scripts.standalone_benchmark import freeze_references
    w,scope=workflow_db
    b=bench(w)
    rows=[]
    for _ in range(5):
        row,_,_=await human(w,scope,tmp_path)
        preview=await b.reference_preview(row.id,scope=scope)
        await b.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
        rows.append(row)
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE receipts SET verification_source='bypass' WHERE id=$1",rows[2].id)
        await conn.execute("UPDATE receipts SET verification_source='legacy' WHERE id=$1",rows[3].id)
        await conn.execute("UPDATE receipts SET merchant_name='stale' WHERE id=$1",rows[4].id)
    original=BenchmarkDB._reference_preview
    seen=[]
    changed=[]
    async def barrier(self,conn,receipt_id,scope,**kwargs):
        result=await original(self,conn,receipt_id,scope,**kwargs)
        seen.append(receipt_id)
        if len(seen)==1:
            other=next(row.id for row in rows[:2] if row.id!=receipt_id)
            changed.append(other)
            async with w.pool.acquire() as writer:
                await writer.execute("UPDATE receipts SET merchant_name='concurrent edit' WHERE id=$1",other)
        return result
    monkeypatch.setattr(BenchmarkDB,'_reference_preview',barrier)
    frozen=await freeze_references(w.pool,w._settings,scope,limit=None,category=None)
    assert {r.receipt_identity for r in frozen}=={r.id for r in rows[:2]}
    assert all(r.expected['merchant_name']=='Shop' for r in frozen)
    monkeypatch.setattr(BenchmarkDB,'_reference_preview',original)
    current=await freeze_references(w.pool,w._settings,scope,limit=1,category=None)
    assert len(current)==1 and current[0].receipt_identity in {row.id for row in rows[:2]} - set(changed)


async def test_gold2_requires_fresh_label_confirmation_and_preserves_old_run(workflow_db, tmp_path, monkeypatch):
    from hashlib import sha256

    from vision_bill.model.benchmark import CouncilPolicy
    from vision_bill.service.receipt_review import material_document

    w, scope = workflow_db
    row, body, _ = await human(w, scope, tmp_path)
    b = bench(w)
    load_preview = b._reference_preview

    async def gold1_preview(conn, receipt_id, scope, *, lock=True):
        preview = await load_preview(conn, receipt_id, scope, lock=lock)
        record = await conn.fetchrow("SELECT * FROM receipts WHERE id=$1", receipt_id)
        material = material_document(await w._receipts._review_document(conn, record))
        material["line_items"] = [
            {key: item[key] for key in ("description", "quantity", "unit_price", "total_price")}
            for item in preview.expected["line_items"]
        ]
        material["taxes"] = preview.expected["taxes"]
        payload = {
            "version": "gold-1",
            "material": material,
            "sources": [
                {key: source[key] for key in ("part_id", "photo_index", "sha256")}
                for source in preview.sources
            ],
            "schema_version": preview.schema_version,
            "scored_fields": preview.scored_fields,
        }
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return preview.model_copy(update={"document_sha256": digest})

    async def frozen_bytes(reference, run):
        async with w.pool.acquire() as conn:
            return (
                await conn.fetchval(
                    "SELECT to_jsonb(r)::text FROM benchmark_references r WHERE id=$1", reference.id
                ),
                await conn.fetchval(
                    "SELECT to_jsonb(r)::text FROM benchmark_runs r WHERE id=$1", run.id
                ),
            )

    with monkeypatch.context() as context:
        context.setattr(b, "_reference_preview", gold1_preview)
        preview = await b.reference_preview(row.id, scope=scope)
        gold1 = await b.qualify_reference(
            row.id, preview.review_revision, preview.document_sha256, scope=scope
        )
        run = await b.create_run(
            model_ids=["fixture"],
            receipt_ids=[row.id],
            fingerprint="frozen",
            model_digests={},
            timeout=600,
            policy=CouncilPolicy.ALL,
            absolute=None,
            relative=None,
            apply_flags=False,
            prompt_version="4",
            scoring_version=gold1.scoring_version,
            owner_id=scope.user_id,
            scope=scope,
            manifest={
                "inputs": [{"receipt_id": str(row.id), "reference": gold1.model_dump(mode="json")}]
            },
        )

    old_bytes = await frozen_bytes(gold1, run)
    with pytest.raises(ValueError, match="qualified"):
        await b.current_reference(row.id, scope=scope)
    with pytest.raises(ValueError, match="confirmation_conflict"):
        await b.qualify_reference(row.id, gold1.review_revision, gold1.document_sha256, scope=scope)
    preview = await b.reference_preview(row.id, scope=scope)
    assert preview.document_sha256 != gold1.document_sha256
    assert preview.expected_sha256 == gold1.expected_sha256
    current = await b.qualify_reference(
        row.id, preview.review_revision, preview.document_sha256, scope=scope
    )
    assert current.id != gold1.id

    updates = [
        (body, "category", "grocery"),
        (body.line_items[0], "spending_category", "milk"),
        (body.line_items[0], "tags", ["Corrected tag"]),
    ]
    for revision, (target, field, value) in enumerate(updates, start=2):
        setattr(target, field, value)
        body.expected_operation_revision = revision
        edited = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
        assert edited.verified and edited.operation_revision == revision + 1
        assert edited.review_revision == current.review_revision == gold1.review_revision == 1
        with pytest.raises(ValueError, match="qualified"):
            await b.current_reference(row.id, scope=scope)
        with pytest.raises(ValueError, match="confirmation_conflict"):
            await b.qualify_reference(
                row.id, current.review_revision, current.document_sha256, scope=scope
            )
        new = await b.reference_preview(row.id, scope=scope)
        assert new.document_sha256 != current.document_sha256
        assert new.expected_sha256 != current.expected_sha256
        assert new.scored_fields == current.scored_fields
        updated = await b.qualify_reference(
            row.id, new.review_revision, new.document_sha256, scope=scope
        )
        assert updated.id != current.id
        current = await b.current_reference(row.id, scope=scope)
        assert current == updated

    assert current.expected["category"] == "grocery"
    assert current.expected["line_items"][0]["spending_category"] == "milk"
    assert current.expected["line_items"][0]["tags"] == ["Corrected tag"]
    assert await frozen_bytes(gold1, run) == old_bytes
