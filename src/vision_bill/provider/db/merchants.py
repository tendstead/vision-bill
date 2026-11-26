"""Transaction-local merchant suggestions and explicit account-directory writes."""

from hashlib import sha256
from uuid import UUID

import asyncpg

from ...model.db.receipt import MerchantOptions
from ...model.item_category import normalize_history_key
from ...model.merchant_catalog import (
    is_generic_merchant,
    merchant_candidates,
    merchant_choices,
    new_merchant_name,
    suggest_merchant,
)


def needs_address(printed: str, address: str | None) -> bool:
    return (
        is_generic_merchant(suggest_merchant(printed, address) or printed)
        or len(merchant_candidates(printed, address)) > 1
    )


def _alias_key(printed: str, address: str | None) -> tuple[str, str] | None:
    key = normalize_history_key(new_merchant_name(printed))
    generic = needs_address(printed, address)
    address_key = normalize_history_key(address or "") if generic else ""
    if not key or (generic and not address_key):
        return None
    # OCR headers and addresses have no size guarantee. Fixed-size digests fit
    # PostgreSQL's B-tree index even when the evidence contains a full OCR page.
    return sha256(key.encode()).hexdigest(), sha256(address_key.encode()).hexdigest()


async def suggested_name(
    conn: asyncpg.Connection, owner: UUID | None, printed: str, address: str | None, *,
    candidate: str | None = None, printed_fallback: bool = True,
) -> str | None:
    alias = _alias_key(printed, address)
    if owner and alias:
        saved = await conn.fetchval(
            """SELECT m.display_name FROM merchant_aliases a JOIN merchants m
               USING (user_id, name_key)
               WHERE a.user_id=$1 AND a.printed_key=$2 AND a.address_key=$3""",
            owner, *alias,
        )
        if isinstance(saved, str):
            return saved
    default = suggest_merchant(printed, address)
    if default is None and candidate and candidate in merchant_candidates(printed, address):
        default = candidate
    if default is None and printed_fallback:
        default = new_merchant_name(printed)
    return default if default and len(default) <= 200 else None


async def learn_name(
    conn: asyncpg.Connection, owner: UUID | None, actor: UUID | None,
    printed: str, address: str | None, display: str | None,
) -> None:
    if owner is None or owner != actor or not display:
        return
    name_key = normalize_history_key(display)
    alias = _alias_key(printed, address)
    if not name_key or (not alias and is_generic_merchant(display)):
        return
    await conn.execute(
        """INSERT INTO merchants (user_id, name_key, display_name) VALUES ($1,$2,$3)
           ON CONFLICT (user_id, name_key) DO NOTHING""",
        owner, name_key, display,
    )
    if alias:
        await conn.execute(
            """INSERT INTO merchant_aliases (user_id, printed_key, address_key, name_key)
               VALUES ($1,$2,$3,$4) ON CONFLICT (user_id, printed_key, address_key)
               DO UPDATE SET name_key=EXCLUDED.name_key""",
            owner, *alias, name_key,
        )


async def options(
    conn: asyncpg.Connection, owner: UUID | None, printed: str, address: str | None, query: str,
) -> MerchantOptions:
    names = {normalize_history_key(name): name for name in merchant_choices(query)}
    if owner:
        rows = await conn.fetch(
            """SELECT name_key, display_name FROM merchants
               WHERE user_id=$1 AND strpos(name_key,$2)>0 ORDER BY name_key LIMIT 100""",
            owner, normalize_history_key(query),
        )
        names.update({row["name_key"]: row["display_name"] for row in rows})
    return MerchantOptions(
        suggestion=await suggested_name(conn, owner, printed, address),
        merchants=[names[key] for key in sorted(names)],
    )
