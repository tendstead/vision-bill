"""Process-local operator logging shared by the API and standalone native image."""

import logging
from collections.abc import Sequence

_POLL_PATHS = {"/health", "/ready", "/info", "/api/v1/system/ui-config"}


def diagnostic_level(value: str) -> int:
    """Older installations may have an arbitrary value in their API settings."""
    return {
        "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
        "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
    }.get(value.strip().upper(), logging.INFO)


class SafeAccessFilter(logging.Filter):
    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn access log arguments are (client, method, path, version, status).
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return False
        client, method, path, version, status = args
        if not isinstance(path, str):
            return False
        path = path.partition("?")[0]
        record.args = (client, method, path, version, status)
        return self.level == logging.DEBUG or not (method == "GET" and status == 200 and path in _POLL_PATHS)


def configure_namespaces(names: Sequence[str], level: int, formatter: logging.Formatter) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    for name in names:
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level)
        logger.propagate = False
    access = logging.getLogger("uvicorn.access")
    for previous in tuple(access.filters):
        if isinstance(previous, SafeAccessFilter):
            access.removeFilter(previous)
    access.addFilter(SafeAccessFilter(level))
