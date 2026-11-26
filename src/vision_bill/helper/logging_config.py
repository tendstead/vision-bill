"""Operator logs with per-submission correlation across async service calls."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

from ocr_contract.diagnostic_logging import configure_namespaces, diagnostic_level

_submission_id: ContextVar[UUID | None] = ContextVar("log_submission_id", default=None)


@contextmanager
def submission_logs(submission_id: UUID) -> Iterator[None]:
    token = _submission_id.set(submission_id)
    try:
        yield
    finally:
        _submission_id.reset(token)


class _SubmissionFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        submission_id = _submission_id.get()
        return f"{message} submission_id={submission_id}" if submission_id is not None else message


def setup_logging(level: str = "INFO") -> None:
    # The source and installed entry points import the application under these names.
    configure_namespaces(
        ("vision_bill", "src.vision_bill"), diagnostic_level(level),
        _SubmissionFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )
