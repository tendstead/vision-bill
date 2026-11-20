import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from vision_bill import config
from vision_bill.distribution import handoff
from vision_bill.provider import factory
from vision_bill.provider.db import workflow_db

IMAGE_ID = UUID("7d29ec10-4814-4c99-9df7-a6ed76731af0")
CHECKPOINT_ID = UUID("da9f531a-4e79-421f-b4e1-fc2c8e317db5")
INFERENCE_IDENTITY = "openai:" + "f" * 64
PRIVATE = "synthetic-private-credential-or-receipt-data"
COMMAND = ["reconcile-classification", "--image-id", str(IMAGE_ID)]
ATTESTATION = ["--request-settled", "--expected-generation", "3", "--checkpoint-id", str(CHECKPOINT_ID)]


class WorkflowDouble:
    def __init__(self):
        self.events = []
        self.state = "recovery_required"
        self.reconcile = True
        self.fail_on = None

    def record(self, operation, *values):
        self.events.append((operation, *values))
        if self.fail_on == operation:
            raise RuntimeError(PRIVATE)

    async def init_db(self):
        self.record("init_db")

    async def destroy_db(self):
        self.record("destroy_db")

    async def unmanaged_classification_status(self, image_id, *, inference_identity):
        self.record("status", image_id, inference_identity)
        return None if self.state is None else {
            "state": self.state,
            "image_id": image_id,
            "generation": 3,
            "checkpoint_id": CHECKPOINT_ID,
            "receipt": PRIVATE,
        }

    async def reconcile_unmanaged_classification(
        self, image_id, checkpoint_id, *, expected_generation, inference_identity, request_settled,
    ):
        self.record(
            "reconcile", image_id, checkpoint_id, expected_generation, inference_identity, request_settled,
        )
        return self.reconcile


@pytest.fixture
def workflow(monkeypatch, settings):
    db = WorkflowDouble()

    def provider(llm_settings):
        assert llm_settings is settings.llm
        db.record("provider")

        async def close():
            db.record("close_client")

        return SimpleNamespace(inference_identity=INFERENCE_IDENTITY, _client=SimpleNamespace(close=close))

    def database(pg_settings):
        assert pg_settings is settings.pg
        db.record("database")
        return db

    monkeypatch.setattr(config, "settings", settings)
    monkeypatch.setattr(factory, "get_llm_provider", provider)
    monkeypatch.setattr(workflow_db, "WorkflowDB", database)
    return db


@pytest.mark.parametrize("state", ["recovery_required", "busy", "not_required", None])
def test_reconciliation_probe_reads_status_without_mutating(workflow, capsys, state):
    workflow.state = state

    assert handoff.main(COMMAND) == (1 if state is None else 0)

    output = capsys.readouterr()
    assert output.err == ""
    if state is None:
        assert json.loads(output.out) == {
            "state": "configuration_required", "reason_code": "unmanaged_classification_checkpoint_unavailable",
        }
    else:
        assert json.loads(output.out) == {
            "state": state, "image_id": str(IMAGE_ID), "generation": 3, "checkpoint_id": str(CHECKPOINT_ID),
        }
    assert workflow.events == [
        ("provider",), ("database",), ("init_db",), ("status", IMAGE_ID, INFERENCE_IDENTITY),
        ("destroy_db",), ("close_client",),
    ]


@pytest.mark.parametrize("accepted", [True, False])
def test_reconciliation_mutation_uses_exact_attested_checkpoint(workflow, capsys, accepted):
    workflow.reconcile = accepted

    assert handoff.main([*COMMAND, *ATTESTATION]) == (0 if accepted else 1)

    output = capsys.readouterr()
    assert output.err == ""
    if accepted:
        assert json.loads(output.out) == {
            "state": "reconciled", "image_id": str(IMAGE_ID), "generation": 3,
            "checkpoint_id": str(CHECKPOINT_ID),
        }
    else:
        assert json.loads(output.out) == {
            "state": "recovery_required", "reason_code": "unmanaged_classification_reconciliation_refused",
        }
    assert workflow.events == [
        ("provider",), ("database",), ("init_db",),
        ("reconcile", IMAGE_ID, CHECKPOINT_ID, 3, INFERENCE_IDENTITY, True),
        ("destroy_db",), ("close_client",),
    ]


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--request-settled"], "must be supplied together"),
        (ATTESTATION[1:], "must be supplied together"),
        (ATTESTATION[:3], "must be supplied together"),
        (["--request-settled", "--checkpoint-id", str(CHECKPOINT_ID)], "must be supplied together"),
        (["--request-settled", "--expected-generation", "0", *ATTESTATION[3:]], "positive integer"),
        (["--request-settled", "--expected-generation", "2147483648", *ATTESTATION[3:]], "positive integer"),
        (["--request-settled", "--expected-generation", "invalid", *ATTESTATION[3:]], "invalid int value"),
        ([*ATTESTATION[:3], "--checkpoint-id", "invalid"], "invalid UUID value"),
        (["--request-set", *ATTESTATION[1:]], "unrecognized arguments"),
    ],
)
def test_invalid_reconciliation_flags_never_run_command(monkeypatch, capsys, flags, message):
    run = Mock(side_effect=AssertionError("invalid arguments reached execution"))
    monkeypatch.setattr(handoff, "_run", run)

    with pytest.raises(SystemExit) as caught:
        handoff.main([*COMMAND, *flags])

    assert caught.value.code == 2
    run.assert_not_called()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("operation", ["provider", "init_db", "status", "reconcile", "destroy_db", "close_client"])
def test_reconciliation_sanitizes_failures_and_closes_resources(workflow, capsys, operation):
    workflow.fail_on = operation
    argv = [*COMMAND, *ATTESTATION] if operation == "reconcile" else COMMAND

    assert handoff.main(argv) == 1

    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "state": "recovery_required", "reason_code": "unmanaged_classification_setup_failed",
    }
    assert PRIVATE not in output.out
    if operation != "provider":
        assert workflow.events[-2:] == [("destroy_db",), ("close_client",)]
