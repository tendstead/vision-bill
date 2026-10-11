import logging
from pathlib import Path
from typing import Any

import httpx

from .base import OcrProvider

logger = logging.getLogger(__name__)


class PaddleOcrProvider(OcrProvider):
    """PaddleOCR-VL-1.6 provider communicating with an external HTTP service."""

    def __init__(self, service_url: str, timeout: int = 300):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout

    async def extract_text(self, image: Path) -> dict[str, Any]:
        """Send image to PaddleOCR service and return structured OCR result."""
        logger.debug("Sending image %s to PaddleOCR service at %s", image.name, self.service_url)

        # Read image bytes before async HTTP call (blocking I/O on local file)
        image_bytes = image.read_bytes()
        logger.debug("Image size: %d bytes", len(image_bytes))

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            files = {"image": (image.name, image_bytes, "image/jpeg")}
            logger.debug("POST %s/ocr", self.service_url)
            response = await client.post(f"{self.service_url}/ocr", files=files)
            logger.debug("PaddleOCR response status: %d", response.status_code)
            if response.status_code != 200:
                logger.debug("PaddleOCR response body: %s", response.text)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            logger.debug("PaddleOCR result keys: %s", list(result.keys()))

        logger.debug("PaddleOCR extraction completed for %s", image.name)
        return result

    async def check_connection(self) -> bool:
        """Check if PaddleOCR service is reachable."""
        logger.debug("Checking PaddleOCR service connection at %s", self.service_url)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.service_url}/health")
            logger.debug("PaddleOCR health check status: %d", response.status_code)
            return response.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            logger.debug("PaddleOCR service connection check failed: %s", exc)
            return False
