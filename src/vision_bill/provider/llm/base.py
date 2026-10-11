import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from ...model.item_category import CATEGORY_CODES, CATEGORY_DEFINITIONS
from ...model.receipt import Receipt
from ...model.receipt_extraction import MultiImageReceipt
from .errors import UnreadableReceiptError


@dataclass
class ModelInfo:
    def __init__(self, id: str, parameter_size: str | None = None):
        self.id = id
        self.parameter_size = parameter_size

    id: str
    parameter_size: str | None = None
    capabilities: list[str] | None = None
    digest: str | None = None


@dataclass
class AnalysisResult:
    receipt: Receipt
    attempts: int
    elapsed_ms: float
    model_digest: str | None = None


class LLMProvider(ABC):
    """Interface every LLM backend must implement."""

    analysis_timeout_seconds: float = 600

    def update_runtime_settings(self, *, temperature: float) -> None:
        """Apply settings which are safe to change without rebuilding a provider."""

    @abstractmethod
    async def get_available_models(self) -> list[ModelInfo]:
        """Return the models this provider currently exposes."""

    @abstractmethod
    async def analyse_receipt_from_model(
        self, model_id: str, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> Receipt:
        """Send an image + prompt to the given model and return the result.

        ``tags`` is the allowed line-item tag vocabulary; when provided it is
        embedded in the prompt so the model only emits known tags.
        """

    @abstractmethod
    async def analyse_receipt_from_ocr_text(
        self, model_id: str, ocr_result: dict[str, Any], tags: Sequence[str] | None = None
    ) -> Receipt:
        """Send OCR text + prompt to the given model and return the result.

        Used in the two-stage OCR → LLM pipeline where OCR has already
        extracted text from the image and the LLM formats it into a Receipt.
        """

    @abstractmethod
    async def send_message(self, model_id: str, messages: Sequence[Mapping[str, Any]]) -> str:
        """Send a message to the given model and return the response."""

    @abstractmethod
    async def check_connection(self) -> bool:
        """Return True if the backend is reachable, False otherwise."""

    async def analyse_receipt_with_metadata(
        self, model_id: str, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> AnalysisResult:
        """Compatibility wrapper for providers which do not expose attempt telemetry."""
        started = perf_counter()
        receipt = await self.analyse_receipt_from_model(model_id, image, tags=tags)
        return AnalysisResult(
            receipt=receipt, attempts=1, elapsed_ms=(perf_counter() - started) * 1000
        )

    def build_prompt(self, tags: Sequence[str] | None = None, *, image_count: int = 1) -> str:
        """Build a prompt for the LLM to analyze an image.

        When ``tags`` is provided, the prompt gives the model the full tag
        vocabulary and lets it suggest one new tag when nothing fits; otherwise
        the model may invent short free-form tags.
        """
        if tags:
            tag_instruction = (
                "Add short tags to each line_item when useful. Prefer tags from this list: "
                f"{', '.join(tags)}. If no tag in the list fits, you may suggest a single new "
                "short tag (lowercase snake_case, at most three words); suggested tags are "
                "reviewed by a human before they become standard tags."
            )
        else:
            tag_instruction = "Add short free-form tags to each line_item when useful."
        categories = "; ".join(f"{code}: {CATEGORY_DEFINITIONS[code]}" for code in CATEGORY_CODES)
        schema = MultiImageReceipt if image_count > 1 else Receipt
        overlap_instruction = (
            "Extract EVERY purchased row separately from EACH photo, including overlap rows. "
            "Do NOT remove duplicates yourself. Set source_image to its zero-based photo index. "
            "Group line_items by photo, then top-to-bottom within each photo. "
            "In duplicate_candidates identify only the SAME printed rows visible at the end "
            "of one photo and the start of the next. Use zero-based line_items array indices. "
            "Include the entire ordered overlap, not isolated matching products. "
            "Use an empty candidate list when there is no clear visual overlap. "
            "Receipt-level totals must come from the printed receipt, not the duplicated rows."
            if image_count > 1
            else "List line_items in the exact top-to-bottom order on the receipt."
        )
        return f"""You are a receipt analyser. You will be given one or more images of ONE receipt.
    Images are ordered from top to bottom and may overlap. Return ONE complete receipt.
    There are {image_count} input photos. {overlap_instruction}
    Preserve genuinely repeated purchases, even when descriptions and prices are identical.
    If the receipt is too blurry, cut off, or unreadable to extract reliably, return ONLY
    {{"error": {{"code": "unreadable", "message": "Explain which part needs a clearer photo"}}}}.
    This error response is an alternative to the receipt schema. Do not invent missing values.
    You should analyze the receipt and provide a machine-readable JSON response based on the receipt.
    Determine merchant_name from text that is actually printed on the receipt (store name in the
    header or footer, the address line, a website, or any other written hint).
    Do not guess or infer the company name from a logo, photo, or design; if no printed name exists,
    use the best available written hint.
    {tag_instruction}
    Set exactly one spending_category on each line_item using only these codes:
    {categories}.
    Use unknown when the text does not support a category; use deposit for Pfand lines.
    Set the top-level category to the single best category for the whole purchase.
    Use positive unit_price and total_price for deposit/Pfand charges; use negative prices
    for refunds, credits, or deposits returned to the customer.
    Tag every deposit charge or return with `deposit`, and set spending_category to deposit.
    Do not add any additional text or commentary. Only provide the JSON response matching this schema:

    {schema.model_json_schema()}
    """

    def parse_llm_response(self, response: str, *, image_count: int = 1) -> Receipt:
        """Parse the LLM response into a Receipt object."""
        raw_text = response.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            if error.get("code") == "unreadable" and isinstance(error.get("message"), str):
                raise UnreadableReceiptError(error["message"][:1000])
        try:
            if image_count > 1:
                return MultiImageReceipt.model_validate_json(raw_text).reconciled(image_count)
            return Receipt.model_validate_json(raw_text)
        except ValidationError as e:
            # retry logic: feed error back to the LLM, or raise
            raise ValueError(f"LLM output failed validation: {e}")
