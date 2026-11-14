from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def compact_ocr_text(result: dict[str, Any]) -> str:
    """Extract transcription only; never send layout/image arrays to an LLM."""
    candidates: list[str] = []
    text = result.get("text")
    if isinstance(text, str) and text.strip():
        candidates.append(text)
    if not candidates:
        markdown = result.get("markdown")
        if isinstance(markdown, dict):
            markdown = markdown.get("text") or markdown.get("markdown_texts")
        if isinstance(markdown, str) and markdown.strip():
            candidates.append(markdown)
    if not candidates:
        blocks = result.get("blocks") or result.get("parsing_res_list") or []
        for block in blocks:
            if isinstance(block, dict):
                content = block.get("text") or block.get("block_content")
                if isinstance(content, str) and content.strip():
                    candidates.append(content)
    text = "\n".join(candidates).strip()
    if not text:
        raise ValueError("OCR returned no recognized text; layout boxes are not transcription")
    if len(text) > 200_000:
        raise ValueError("Recognized text exceeds the input limit")
    return text


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
