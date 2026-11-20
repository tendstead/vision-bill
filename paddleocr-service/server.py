"""Standalone OCR service entrypoint (native state lives in child processes)."""

import os

import uvicorn
from ocr_service.api import create_app
from ocr_service.images import process_ocrvl_result

__all__ = ["app", "process_ocrvl_result"]

app = create_app(mode=os.environ.get("OCR_SERVICE_MODE", "managed"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("OCR_PORT", "8082")))
