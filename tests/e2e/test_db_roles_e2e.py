"""Real PG18 login/default ACL proof; every resource and connection is test-owned."""

import base64
import hashlib
import os
import secrets
import subprocess
import sys
import time
from datetime import time as Time
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from e2e.test_workflow_db_e2e import complete
from vision_bill.config import PGSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.extraction import DerivedInputView
from vision_bill.model.ocr import OcrResponse, SourceRef, WorkingImageMetadata
from vision_bill.model.workflow import (
    AccessScope,
    OcrCheckpoint,
    PersistedWorkingView,
    SubmissionCreate,
)
from vision_bill.provider.db.workflow_db import WorkflowDB
from vision_bill.service.image_service import ImageService

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]


def docker(*args, **kwargs):
    # Honor the selected Docker endpoint; callers choose the managed socket in cloud.
    result = subprocess.run(["docker", *args], capture_output=True, check=False, **kwargs)
    assert result.returncode == 0, "private PostgreSQL Docker operation failed"
    return result


@pytest.fixture(scope="module")
def role_cluster():
    name = "vb-roles-" + secrets.token_hex(8)
    password = secrets.token_hex(24)
    docker(
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "-e",
        "POSTGRES_USER=role_admin",
        "-e",
        f"POSTGRES_PASSWORD={password}",
        "-e",
        "POSTGRES_DB=role_admin",
        "-p",
        "127.0.0.1::5432",
        "postgres:18",
    )
    try:
        port = int(docker("port", name, "5432/tcp", text=True).stdout.strip().rsplit(":", 1)[1])
        for _ in range(100):
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "role_admin"],
                capture_output=True,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.1)
        else:
            pytest.fail("private PostgreSQL did not become ready")
        yield (
            name,
            PGSettings(
                user="role_admin", password=password, host="127.0.0.1", port=port, db="role_admin"
            ),
        )
    finally:
        docker("rm", "--force", "--volumes", name)


def migrate(owner, directory, revision):
    directory.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(ROOT / "src"),
        "PG__USER": owner.user,
        "PG__PASSWORD": owner.password,
        "PG__HOST": owner.host,
        "PG__PORT": str(owner.port),
        "PG__DB": owner.db,
        "AUTH__SECRET_KEY": secrets.token_hex(32),
        "LLM__PROVIDER": "ollama",
        "LLM__HOST": "http://127.0.0.1:1",
        "LLM__API_KEY": "unused",
        "LLM__MODEL_NAME": "private-migration-unused",
        "LLM__TEMPERATURE": "0",
        "VISION_BILL_CONFIG_PATH": str(directory / "settings.yaml"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ROOT / "alembic.ini"), "upgrade", revision],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
async def identities(role_cluster):
    container, admin_settings = role_cluster
    admin = await asyncpg.connect(admin_settings.pg_dsn)
    suffix = secrets.token_hex(8)
    database = "role_case_" + suffix
    names = {kind: "vb_" + kind + "_" + suffix for kind in ("migration", "runtime", "readonly")}
    settings = {}
    try:
        for group in ("vision_bill_runtime", "vision_bill_readonly"):
            if not await admin.fetchval("SELECT 1 FROM pg_roles WHERE rolname=$1", group):
                await admin.execute(
                    f"CREATE ROLE {group} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE"
                )
        for kind, name in names.items():
            password = secrets.token_hex(24)
            await admin.execute(
                f"CREATE ROLE {name} LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD '{password}'"
            )
            settings[kind] = admin_settings.model_copy(
                update={"user": name, "password": password, "db": database}
            )
            if kind != "migration":
                await admin.execute(f"GRANT vision_bill_{kind} TO {name}")
        await admin.execute(f"CREATE DATABASE {database} OWNER {names['migration']}")
        owner = await asyncpg.connect(settings["migration"].pg_dsn)
        try:
            await owner.execute(f"REVOKE ALL ON DATABASE {database} FROM PUBLIC")
            await owner.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
        finally:
            await owner.close()
        yield container, settings
    finally:
        await admin.execute(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
        for name in names.values():
            await admin.execute(f"DROP ROLE IF EXISTS {name}")
        await admin.close()


async def assert_login(conn, settings, migration):
    identity = await conn.fetchrow("""SELECT session_user AS session, current_user AS current,
        rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls
        FROM pg_roles WHERE rolname=current_user""")
    assert identity["session"] == identity["current"] == settings.user
    assert not any(
        identity[key]
        for key in ("rolsuper", "rolcreaterole", "rolcreatedb", "rolreplication", "rolbypassrls")
    )
    if settings.user != migration.user:
        assert not await conn.fetchval(
            "SELECT pg_has_role(current_user,$1,'MEMBER')", migration.user
        )


async def seed_legacy(conn, directory):
    original = directory / "legacy.png"
    original.write_bytes(b"preserved legacy original")
    owner = await conn.fetchval(
        "INSERT INTO users(username,hashed_password) VALUES('legacy','test') RETURNING id"
    )
    image = await conn.fetchval(
        "INSERT INTO images(user_id,image_path,status,error) VALUES($1,$2,'failed','old attempt') RETURNING id",
        owner,
        str(original),
    )
    receipt = await conn.fetchval(
        """INSERT INTO receipts(user_id,image_id,merchant_name,date,time,subtotal,total,tax_total,currency,verified,status)
        VALUES($1,$2,'Legacy','2020-01-01',$3,12.34,12.34,1.23,'EUR',TRUE,'verified') RETURNING id""",
        owner,
        image,
        Time(14, 30, 15, 123456),
    )
    await conn.execute("UPDATE images SET receipt_id=$2 WHERE id=$1", image, receipt)
    await conn.execute(
        """INSERT INTO line_items(receipt_id,description,original_description,quantity,unit_price,total_price,position)
        VALUES($1,'preserved','preserved',1.2345,12.34,12.34,0)""",
        receipt,
    )
    await conn.execute(
        "INSERT INTO taxes(receipt_id,name,rate,amount) VALUES($1,'VAT',0.1234,1.23)", receipt
    )
    run = await conn.fetchval("""INSERT INTO benchmark_runs(status,model_ids,receipt_ids,dataset_fingerprint,prompt_version,scoring_version,experiment_manifest)
        VALUES('completed','["legacy"]','[]','frozen','old','3','{"time":"14:30:15.123456"}') RETURNING id""")
    await conn.execute(
        "INSERT INTO benchmark_results(run_id,model_id,receipt_id,success,payload) VALUES($1,'legacy',$2,TRUE,'{\"score\":0.875}')",
        run,
        receipt,
    )
    return image, original


LEGACY_QUERIES = [
    "SELECT id,user_id,image_id,time,subtotal,total,tax_total,currency,verified,status FROM receipts ORDER BY id",
    "SELECT id,receipt_id,description,quantity,unit_price,total_price,position FROM line_items ORDER BY id",
    "SELECT id,receipt_id,name,rate,amount FROM taxes ORDER BY id",
    "SELECT id,user_id,image_path,receipt_id FROM images ORDER BY id",
    "SELECT id,experiment_manifest FROM benchmark_runs ORDER BY id",
    "SELECT run_id,payload FROM benchmark_results ORDER BY run_id,model_id,receipt_id",
]


async def assert_probe_permissions(runtime, readonly, table):
    # Identity INSERT alone does not prove nextval/last_value sequence permissions.
    sequence = await runtime.fetchval("SELECT pg_get_serial_sequence($1,'id')", table)
    for privilege in ("USAGE", "SELECT"):
        assert await runtime.fetchval(
            "SELECT has_sequence_privilege(current_user,$1,$2)", sequence, privilege
        )
        assert not await readonly.fetchval(
            "SELECT has_sequence_privilege(current_user,$1,$2)", sequence, privilege
        )
    row_id = await runtime.fetchval(f"INSERT INTO {table}(value) VALUES('inserted') RETURNING id")
    assert await readonly.fetchval(f"SELECT value FROM {table} WHERE id=$1", row_id) == "inserted"
    assert await runtime.fetchval("SELECT nextval($1::regclass)", sequence) > row_id
    assert await runtime.fetchval(f"SELECT last_value FROM {sequence}") > row_id
    assert (
        await runtime.execute(f"UPDATE {table} SET value='updated' WHERE id=$1", row_id)
        == "UPDATE 1"
    )
    assert await runtime.fetchval(f"SELECT value FROM {table} WHERE id=$1", row_id) == "updated"
    for sql in (
        f"INSERT INTO {table}(value) VALUES('denied')",
        f"UPDATE {table} SET value='denied' WHERE false",
        f"DELETE FROM {table} WHERE false",
    ):
        with pytest.raises(asyncpg.InsufficientPrivilegeError) as denied:
            await readonly.execute(sql)
        assert denied.value.sqlstate == "42501"
    assert await runtime.execute(f"DELETE FROM {table} WHERE id=$1", row_id) == "DELETE 1"


async def exercise_workflow(settings, readonly, directory):
    provider = WorkflowDB(settings, original_checker=ImageService.check_originals)
    await provider.init_db()
    try:
        scope = AccessScope(user_id=uuid4())
        async with provider.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users(id,username,hashed_password) VALUES($1,$2,'test')",
                scope.user_id,
                str(scope.user_id),
            )
        original = directory / "original.png"
        from PIL import Image

        Image.new("RGB", (2, 3), "white").save(original)
        raw = original.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        request = SubmissionCreate(
            id=uuid4(),
            parts=[ImagePart(image_path=str(original))],
            source_manifest=[{"photo_index": 0, "source_sha256": source_hash}],
            requested_options={},
            effective_options={},
            request_fingerprint="role-smoke",
        )
        image = (
            await provider.enqueue_submission(request, scope=scope, idempotency_key=None)
        ).submission
        claim = await provider.claim_next()
        assert claim is not None and claim.image.id == image.id
        working = directory / "working.png"
        working.write_bytes(raw)
        page = OcrResponse(
            request_id=uuid4(),
            photo_index=0,
            source_sha256=source_hash,
            text="Merchant",
            blocks=[],
            profile_id="role-smoke-profile",
            runtime_fingerprint="role-smoke-runtime",
            working_image={
                "view_id": uuid4(),
                "width": 2,
                "height": 3,
                "sha256": source_hash,
                "png_base64": base64.b64encode(raw).decode(),
            },
        )
        parts = await provider.get_source_parts(image.id, scope=scope)
        view = PersistedWorkingView(
            part_id=parts[0].part_id,
            photo_index=0,
            source_sha256=source_hash,
            artifact_path=str(working),
            metadata=WorkingImageMetadata.model_validate(page.working_image.model_dump()),
        )
        checkpoint = OcrCheckpoint(
            ordered_source_sha256=[source_hash],
            profile_id=page.profile_id,
            runtime_fingerprint=page.runtime_fingerprint,
            pages=[page],
            persisted_views=[view],
        )
        assert await provider.save_ocr_checkpoint(claim, checkpoint)
        assert await provider.get_ocr_checkpoint(claim) == checkpoint
        outcome = complete(total="12.34")
        outcome.envelope.provenance.ordered_source_sha256 = [source_hash]
        outcome.envelope.ocr_pages = [page]
        outcome.envelope.input_views = [
            DerivedInputView(
                photo_index=0, source_sha256=source_hash, working_image=page.working_image
            )
        ]
        outcome.envelope.field_sources = {
            "total": [
                SourceRef(
                    part_id=view.part_id,
                    photo_index=0,
                    source_sha256=source_hash,
                    view_id=view.metadata.view_id,
                    coordinate_space="working",
                )
            ]
        }
        outcome.persisted_views = [view]
        result = await provider.finish_attempt(claim, outcome)
        assert not result.stale and result.submission.receipt_id is not None
        extraction_id = await readonly.fetchval(
            "SELECT id FROM receipt_extractions WHERE image_id=$1", image.id
        )
        extraction = await provider.get_extraction(image.id, extraction_id, scope=scope)
        assert extraction is not None
        saved = await provider.get_working_view(image.id, view.metadata.view_id, scope=scope)
        assert saved is not None and Path(saved.artifact_path).read_bytes() == raw
        assert original.read_bytes() == raw
        for table in (
            "images",
            "image_source_parts",
            "extraction_working_views",
            "receipt_extractions",
            "receipt_review_revisions",
            "receipts",
            "line_items",
            "taxes",
        ):
            assert await readonly.fetchval(f"SELECT count(*) FROM {table}") > 0
        # Mutating an immutable artifact as runtime still hits its integrity trigger.
        async with provider.pool.acquire() as conn:
            with pytest.raises(asyncpg.RaiseError, match="immutable"):
                await conn.execute(
                    "UPDATE receipt_extractions SET result_code='changed' WHERE image_id=$1",
                    image.id,
                )
    finally:
        await provider.destroy_db()


@pytest.mark.parametrize("start", ["fresh", "0010"])
async def test_real_logins_existing_and_future_grants(identities, tmp_path, start):
    container, settings = identities
    owner_settings = settings["migration"]
    owner = await asyncpg.connect(owner_settings.pg_dsn)
    runtime = readonly = None
    try:
        await assert_login(owner, owner_settings, owner_settings)
        if start == "0010":
            migrate(owner_settings, tmp_path / "migrations", "0010")
            legacy_image, original = await seed_legacy(owner, tmp_path)
            before = [await owner.fetch(query) for query in LEGACY_QUERIES]
        await owner.execute(
            "CREATE TABLE role_existing(id bigint GENERATED ALWAYS AS IDENTITY, value text)"
        )
        # Existing objects inaccessible before running the actual role script.
        assert not await owner.fetchval(
            "SELECT has_table_privilege($1,'role_existing','SELECT')", settings["runtime"].user
        )
        docker(
            "exec",
            "--interactive",
            "--env",
            f"PGPASSWORD={owner_settings.password}",
            container,
            "psql",
            "-X",
            "-h",
            "127.0.0.1",
            "-U",
            owner_settings.user,
            "-d",
            owner_settings.db,
            "-v",
            "ON_ERROR_STOP=1",
            "-v",
            f"database_name={owner_settings.db}",
            "-f",
            "-",
            input=(ROOT / "scripts/configure_db_roles.sql").read_text(),
            text=True,
        )
        # Never rerun grants: migrations and role_future must inherit owner defaults.
        migrate(owner_settings, tmp_path / "migrations", "head")
        expected = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_current_head()
        assert await owner.fetchval("SELECT version_num FROM alembic_version") == expected
        await owner.execute(
            "CREATE TABLE role_future(id bigint GENERATED ALWAYS AS IDENTITY, value text)"
        )
        if start == "0010":
            assert [await owner.fetch(query) for query in LEGACY_QUERIES] == before
            assert original.read_bytes() == b"preserved legacy original"
            assert (
                await owner.fetchval(
                    "SELECT legacy_events->0->>'error' FROM images WHERE id=$1", legacy_image
                )
                == "old attempt"
            )
            assert await owner.fetchval("SELECT verification_source FROM receipts") == "legacy"
        runtime = await asyncpg.connect(settings["runtime"].pg_dsn)
        readonly = await asyncpg.connect(settings["readonly"].pg_dsn)
        for conn, kind in ((runtime, "runtime"), (readonly, "readonly")):
            await assert_login(conn, settings[kind], owner_settings)
            assert await conn.fetchval("SELECT has_schema_privilege(current_user,'public','USAGE')")
            with pytest.raises(asyncpg.InsufficientPrivilegeError) as denied:
                await conn.execute("CREATE TABLE forbidden(id int)")
            assert denied.value.sqlstate == "42501"
        for table in ("role_existing", "role_future"):
            await assert_probe_permissions(runtime, readonly, table)
        for table in (
            "image_retry_requests",
            "image_source_parts",
            "extraction_working_views",
            "receipt_extractions",
            "receipt_drafts",
            "receipt_review_revisions",
            "benchmark_references",
            "benchmark_tasks",
        ):
            for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                assert await runtime.fetchval(
                    "SELECT has_table_privilege(current_user,$1,$2)", table, privilege
                )
            assert await readonly.fetchval(
                "SELECT has_table_privilege(current_user,$1,'SELECT')", table
            )
            for privilege in ("INSERT", "UPDATE", "DELETE"):
                assert not await readonly.fetchval(
                    "SELECT has_table_privilege(current_user,$1,$2)", table, privilege
                )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await readonly.execute(f"DELETE FROM {table} WHERE false")
        await exercise_workflow(settings["runtime"], readonly, tmp_path)
        print(
            f"role proof {start}->{expected}: migration={owner_settings.user} "
            f"runtime={settings['runtime'].user} readonly={settings['readonly'].user}"
        )
    finally:
        for conn in (runtime, readonly, owner):
            if conn is not None:
                await conn.close()
