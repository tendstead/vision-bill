"""Operator commands for external inference ownership and request recovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from uuid import UUID

from ..model.handoff import HandoffBinding, HandoffStatus, HandoffWait
from ..service.gpu_handoff import ModelHandoff, current_owner
from .bootstrap import write_private

if TYPE_CHECKING:
    from ..provider.db.workflow_db import WorkflowDB


async def adopt_binding(
    handoff: ModelHandoff,
    model_ids: Sequence[str],
    *,
    dedicated_coordinated_requests: bool,
    external_eager_loading_disabled: bool,
    deadline: float,
) -> HandoffBinding:
    """Record explicit operator attestations only while all owned state is idle."""
    if dedicated_coordinated_requests is not True or external_eager_loading_disabled is not True:
        raise ValueError("model_handoff_explicit_attestations_required")
    binding = HandoffBinding(
        coordination_path=str(handoff.path.resolve()), resources=handoff.resources,
        model_ids=list(model_ids), dedicated_coordinated_requests=True,
        external_eager_loading_disabled=True,
    )
    if not handoff.provider.supports_model_release:
        raise ValueError("model_release_unsupported")
    if current_owner() is not None:
        raise HandoffWait(HandoffStatus(state="busy", reason_code="model_handoff_busy"))
    async with asyncio.timeout(max(0, deadline - monotonic())):
        async with handoff.exclusion(deadline, wait=False):
            if binding.resources != handoff.resources or binding.coordination_path != str(handoff.path.resolve()):
                raise ValueError("model_handoff_identity_changed")
            saved = handoff.read_journal()
            if saved is not None and (saved.unsettled or saved.ocr_pending or saved.models):
                raise HandoffWait(HandoffStatus(
                    state="recovery_required", reason_code="model_handoff_journal_not_clean",
                ))
            # Existing bindings also fence endpoint/path changes and malformed state.
            if handoff.binding_path.exists():
                handoff.read_binding()
            await handoff.require_residency([], deadline)
            write_private(handoff.binding_path, binding.model_dump_json() + "\n", replace=True)
    return binding


async def _close_provider(provider: object) -> None:
    client = getattr(provider, "_client", None)
    if client is not None:
        if hasattr(client, "aclose"):
            await client.aclose()
        elif hasattr(client, "close"):
            await client.close()
        elif hasattr(client, "_client"):
            await client._client.aclose()


async def _classification_result(
    args: argparse.Namespace, workflow: WorkflowDB, inference_identity: str,
) -> tuple[int, dict[str, object]]:
    if args.request_settled:
        reconciled = await workflow.reconcile_unmanaged_classification(
            args.image_id, args.checkpoint_id, expected_generation=args.expected_generation,
            inference_identity=inference_identity, request_settled=True,
        )
        if not reconciled:
            return 1, {
                "state": "recovery_required", "reason_code": "unmanaged_classification_reconciliation_refused",
            }
        return 0, {
            "state": "reconciled", "image_id": str(args.image_id),
            "generation": args.expected_generation, "checkpoint_id": str(args.checkpoint_id),
        }

    status = await workflow.unmanaged_classification_status(args.image_id, inference_identity=inference_identity)
    if status is None:
        return 1, {
            "state": "configuration_required", "reason_code": "unmanaged_classification_checkpoint_unavailable",
        }
    state, generation = status["state"], status["generation"]
    if (
        state not in ("recovery_required", "busy", "not_required")
        or type(generation) is not int or generation < 1
        or UUID(str(status["image_id"])) != args.image_id
    ):
        raise ValueError("unmanaged_classification_invalid_status")
    return 0, {
        "state": state, "image_id": str(args.image_id), "generation": generation,
        "checkpoint_id": str(UUID(str(status["checkpoint_id"]))),
    }


async def _run_reconciliation(args: argparse.Namespace) -> int:
    try:
        from ..config import settings
        from ..provider.db.workflow_db import WorkflowDB
        from ..provider.factory import get_llm_provider

        async with AsyncExitStack() as resources:
            provider = get_llm_provider(settings.llm)
            resources.push_async_callback(_close_provider, provider)
            if provider.inference_identity in ("", "unsupported"):
                raise ValueError("unmanaged_classification_unsupported_identity")
            workflow = WorkflowDB(settings.pg)
            resources.push_async_callback(workflow.destroy_db)
            await workflow.init_db()
            code, result = await _classification_result(args, workflow, provider.inference_identity)
    except Exception:  # noqa: BLE001 - config, SDK, and database errors may contain private data.
        code = 1
        result = {"state": "recovery_required", "reason_code": "unmanaged_classification_setup_failed"}
    print(json.dumps(result))
    return code


async def _run(args: argparse.Namespace) -> int:
    if args.command == "reconcile-classification":
        return await _run_reconciliation(args)
    provider = None
    try:
        from ..config import settings
        from ..provider.factory import get_llm_provider, get_ocr_provider

        provider = get_llm_provider(settings.llm)
        ocr = get_ocr_provider(settings.ocr)
        if ocr is None or not ocr.managed_lifecycle:
            status = HandoffStatus(state="configuration_required", reason_code="model_handoff_managed_ocr_required")
        else:
            handoff = ModelHandoff(args.handoff_lock or settings.ocr.coordination_path, provider, ocr)
            deadline = monotonic() + 30
            if args.command == "adopt":
                await adopt_binding(
                    handoff, args.model_id,
                    dedicated_coordinated_requests=args.dedicated_coordinated_requests,
                    external_eager_loading_disabled=args.external_eager_loading_disabled,
                    deadline=deadline,
                )
                status = HandoffStatus(state="ready")
            else:
                status = await handoff.inspect(model_id=args.model_id or settings.llm.model_name, deadline=deadline)
        print(status.model_dump_json())
        return 0 if status.state == "ready" else 1
    except HandoffWait as exc:
        print(exc.status.model_dump_json())
        if exc.status.reason_code == "model_handoff_foreign_residency":
            print("Quiesce the external endpoint using its own controls, then retry with zero resident models.", file=sys.stderr)
        return 1
    except (ValueError, OSError, TimeoutError):
        # Endpoint URLs and SDK exceptions can contain configured credentials.
        print(json.dumps({"state": "recovery_required", "reason_code": "model_handoff_setup_failed"}))
        return 1
    finally:
        await _close_provider(provider)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "adopt"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--handoff-lock", type=Path)
        sub.add_argument("--model-id", action="append" if command == "adopt" else "store", required=command == "adopt")
        if command == "adopt":
            sub.add_argument("--dedicated-coordinated-requests", action="store_true", required=True)
            sub.add_argument("--external-eager-loading-disabled", action="store_true", required=True)
    reconcile = subparsers.add_parser(
        "reconcile-classification", allow_abbrev=False,
        help="Inspect or attest settlement of an unmanaged classification request",
    )
    reconcile.add_argument("--image-id", type=UUID, required=True)
    reconcile.add_argument("--expected-generation", type=int)
    reconcile.add_argument("--checkpoint-id", type=UUID)
    reconcile.add_argument(
        "--request-settled", action="store_true",
        help="Attest that the previous request has settled at the configured inference endpoint",
    )
    args = parser.parse_args(argv)
    if args.command == "reconcile-classification":
        mutation_flags = (args.request_settled, args.expected_generation is not None, args.checkpoint_id is not None)
        if any(mutation_flags) and not all(mutation_flags):
            reconcile.error("--request-settled, --expected-generation, and --checkpoint-id must be supplied together")
        if args.expected_generation is not None and not 1 <= args.expected_generation <= 2_147_483_647:
            reconcile.error("--expected-generation must be a positive integer in the database generation range")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
