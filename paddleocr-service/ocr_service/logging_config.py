"""Configure only native-service diagnostics, including spawned receipt workers."""

import logging
import os

from ocr_contract.diagnostic_logging import configure_namespaces, diagnostic_level


def configure_native_logging() -> None:
    configure_namespaces(
        ("ocr_service",), diagnostic_level(os.environ.get("OCR_LOG_LEVEL", "INFO")),
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )
