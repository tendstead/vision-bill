"""Identify the native Paddle wheel before importing its shared module."""

from collections.abc import Iterable
from importlib import metadata

from packaging.utils import canonicalize_name

NATIVE_PADDLE_DISTRIBUTIONS = {"paddlepaddle", "paddlepaddle-gpu", "paddlepaddle-dcu"}


def native_paddle_distribution(names: Iterable[str], *, accelerator: str | None = None) -> str:
    candidates = [
        name for raw in names if (name := canonicalize_name(raw)).startswith("paddlepaddle")
    ]
    if len(candidates) != 1 or candidates[0] not in NATIVE_PADDLE_DISTRIBUTIONS:
        raise ValueError("Exactly one recognized native Paddle distribution is required")
    name = candidates[0]
    if name == "paddlepaddle-dcu" and accelerator not in (None, "rocm"):
        raise ValueError("The paddlepaddle-dcu distribution requires the ROCm accelerator")
    return name


def installed_native_paddle(*, accelerator: str | None = None) -> str:
    # Count installations before deduplication. The wheels share the `paddle` module.
    return native_paddle_distribution(
        (str(dist.metadata.get("Name", "")) for dist in metadata.distributions()),
        accelerator=accelerator,
    )
