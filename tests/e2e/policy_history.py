"""Seed/inspect one real owned-stack historical row for the browser adoption test."""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import UUID

import asyncpg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "e2e"))
from run import compose, endpoints, validate_state


async def main():
    state = validate_state(json.loads(Path(os.environ["VB_E2E_STATE"]).read_text()))
    compose(state, "ps")  # Enforce runner ownership before any database access.
    assert state["policy"] != "off", "adoption requires configured current OCR"
    conn = await asyncpg.connect(endpoints(state)["VB_E2E_PG_URL"])
    try:
        image_id = UUID(sys.argv[2])
        owner_id = UUID(sys.argv[3])
        if sys.argv[1] == "seed":
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT * FROM images WHERE id=$1 AND user_id=$2 FOR UPDATE", image_id, owner_id
                )
                if row is None:
                    raise ValueError("owned submission not found")
                assert row["status"] == "pending" and row["attempt_count"] == 0
                assert row["receipt_id"] is None
                snapshot = json.loads(row["effective_options"])
                snapshot.pop("execution_policy_version")
                snapshot.pop("managed_binding_sha256")
                snapshot.pop("inference_identity")
                snapshot["ocr"]["enabled"] = False
                snapshot["ocr"]["required"] = False
                snapshot["allow_ocr_fallback"] = True
                await conn.execute(
                    "UPDATE images SET effective_options=$3::jsonb WHERE id=$1 AND user_id=$2",
                    image_id,
                    owner_id,
                    json.dumps(snapshot),
                )
        row = await conn.fetchrow(
            """SELECT user_id,source_manifest,requested_options,request_fingerprint,idempotency_key,
            effective_options,legacy_events,generation,
            observation_checkpoint->>'checkpoint_id' AS checkpoint_id,
            observation_checkpoint#>>'{execution_identity,inference_identity}' AS inference_identity
            FROM images WHERE id=$1 AND user_id=$2""",
            image_id,
            owner_id,
        )
        if row is None:
            raise ValueError("owned submission not found")
        payload = dict(row)
        assert payload["user_id"] == owner_id
        payload["user_id"] = str(payload["user_id"])
        for key in ("source_manifest", "requested_options", "effective_options", "legacy_events"):
            payload[key] = json.loads(payload[key])
        print(json.dumps(payload))
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
