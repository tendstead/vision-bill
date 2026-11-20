"""Decode receipt JSON numbers without rounding printed observations before validation."""

import json
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request, Response
from fastapi.routing import APIRoute

_OBSERVATION_PATHS = {
    ("subtotal",),
    ("discount_total",),
    ("tax_total",),
    ("tip",),
    ("total",),
    ("line_items", "quantity"),
    ("line_items", "unit_price"),
    ("line_items", "total_price"),
    ("taxes", "rate"),
    ("taxes", "amount"),
}


class _Fraction(str):
    """A JSON numeric token, distinct from an actual JSON string."""


def _decode(value: Any, path: tuple[str, ...]) -> Any:
    if isinstance(value, _Fraction):
        return str(value) if path in _OBSERVATION_PATHS else float(value)
    if isinstance(value, dict):
        return {key: _decode(child, (*path, key)) for key, child in value.items()}
    if isinstance(value, list):
        return [_decode(child, path) for child in value]
    return value


class _ReceiptRequest(Request):
    async def json(self) -> Any:
        payload = json.loads(await self.body(), parse_float=_Fraction)
        if (
            self.method == "PATCH"
            and isinstance(payload, dict)
            and isinstance(payload.get("changes"), dict)
        ):
            payload["changes"] = _decode(payload["changes"], ())
            return payload
        return _decode(payload, ())


class ReceiptWriteRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        route = super().get_route_handler()

        async def handle(request: Request) -> Response:
            if request.method in {"PUT", "PATCH"}:
                request = _ReceiptRequest(request.scope, request.receive)
            return await route(request)

        return handle
