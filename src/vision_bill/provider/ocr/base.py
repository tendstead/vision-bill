from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class OcrProvider(ABC):
    """Abstract base class for OCR providers."""

    @abstractmethod
    async def extract_text(self, image: Path) -> dict[str, Any]:
        """Extract structured text from image.

        Returns a dict with OCR results including text content,
        bounding boxes, confidence scores, and layout information.
        """

    @abstractmethod
    async def check_connection(self) -> bool:
        """Check if OCR service is reachable."""
