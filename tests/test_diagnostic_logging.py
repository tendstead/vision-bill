"""The operator can trace OCR without turning on request bodies or private query values."""

import logging
from copy import copy

from ocr_contract.diagnostic_logging import SafeAccessFilter, diagnostic_level
from ocr_service.logging_config import configure_native_logging

from vision_bill.helper.logging_config import setup_logging


def test_diagnostic_levels_apply_to_real_namespaces_only(monkeypatch):
    namespaces = ["vision_bill", "src.vision_bill", "ocr_service", "uvicorn.access"]
    loggers = [logging.getLogger(name) for name in namespaces]
    saved = [(item.level, copy(item.handlers), copy(item.filters), item.propagate) for item in loggers]
    http_level = logging.getLogger("httpx").getEffectiveLevel()
    try:
        setup_logging("debug")
        monkeypatch.setenv("OCR_LOG_LEVEL", "DEBUG")
        configure_native_logging()
        configure_native_logging()
        for name in namespaces[:-1]:
            assert logging.getLogger(f"{name}.worker").isEnabledFor(logging.DEBUG)
            assert len(logging.getLogger(name).handlers) == 1
        assert logging.getLogger("httpx").getEffectiveLevel() == http_level
        assert diagnostic_level("unknown") == logging.INFO
    finally:
        for item, (level, handlers, filters, propagate) in zip(loggers, saved, strict=True):
            item.setLevel(level)
            item.handlers, item.filters, item.propagate = handlers, filters, propagate


def test_access_log_hides_query_and_preserves_failed_health_checks():
    entry = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                              ("127.0.0.1", "GET", "/api/v1/search?q=private", "1.1", 200), None)
    access = SafeAccessFilter(logging.INFO)
    assert access.filter(entry)
    assert "private" not in entry.getMessage()
    assert "/api/v1/search HTTP/1.1" in entry.getMessage()
    entry.args = ("127.0.0.1", "GET", "/health", "1.1", 200)
    assert not access.filter(entry)
    assert SafeAccessFilter(logging.DEBUG).filter(entry)
    entry.args = ("127.0.0.1", "GET", "/ready", "1.1", 503)
    assert access.filter(entry)
