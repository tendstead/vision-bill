"""Cancellable waits for isolated synchronous work under an absolute attempt budget."""

import asyncio
from collections.abc import Callable
from time import monotonic
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def check_deadline(deadline: float) -> None:
    if monotonic() >= deadline:
        raise TimeoutError("extraction_deadline")


async def run_sync_before(
    deadline: float, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Only accept a completed private result within budget; cancellation never owns helper files."""
    check_deadline(deadline)
    async with asyncio.timeout(max(0, deadline - monotonic())):
        result = await asyncio.to_thread(operation, *args, **kwargs)
    check_deadline(deadline)
    return result
