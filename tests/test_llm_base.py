from pathlib import Path

import pytest

from vision_bill.model.receipt import Receipt
from vision_bill.provider.llm.base import LLMProvider


def test_check_connection_is_abstract():
    assert "check_connection" in LLMProvider.__abstractmethods__


def test_cannot_instantiate_llm_provider_directly():
    with pytest.raises(TypeError):
        LLMProvider()


class _IncompleteProvider(LLMProvider):
    async def get_available_models(self):
        return []

    async def analyse_receipt_from_model(self, model_id, image):
        raise NotImplementedError

    async def send_message(self, model_id, messages):
        raise NotImplementedError


def test_subclass_missing_check_connection_cannot_instantiate():
    with pytest.raises(TypeError):
        _IncompleteProvider()


class _ConcreteProvider(LLMProvider):
    """Minimal concrete provider so the shared prompt builder can be exercised."""

    async def get_available_models(self):
        return []

    async def analyse_receipt_from_model(self, model_id, image, tags=None):
        raise NotImplementedError

    async def analyse_receipt_from_ocr_text(self, model_id, ocr_result, tags=None):
        raise NotImplementedError

    async def send_message(self, model_id, messages):
        raise NotImplementedError

    async def check_connection(self) -> bool:
        return True


def test_build_prompt_tells_not_to_use_logo_for_merchant_name():
    """The prompt must steer the model away from guessing names from a logo."""
    prompt = _ConcreteProvider().build_prompt()

    assert "Do not guess or infer the company name from a logo" in prompt
    assert "merchant_name" in prompt


def test_build_prompt_without_tags_allows_free_form_tags():
    prompt = _ConcreteProvider().build_prompt()

    assert "free-form tags" in prompt
    assert "Prefer tags from this list" not in prompt


def test_build_prompt_distinguishes_deposit_charge_and_refund():
    """Deposit charges cost money; returns credit money back."""
    prompt = _ConcreteProvider().build_prompt()

    assert "positive unit_price and total_price for deposit/Pfand charges" in prompt
    assert "negative prices" in prompt
    assert "Pfand" in prompt


def test_build_prompt_tells_to_use_deposit_tag_for_deposit_pfand_lines():
    """The prompt must tell the model to mark deposit/Pfand line items with the `deposit` tag."""
    prompt = _ConcreteProvider().build_prompt()

    assert "Tag every deposit charge or return with `deposit`" in prompt
    assert "spending_category to deposit" in prompt


@pytest.mark.parametrize(
    "tags, expected_snippet",
    [
        (["coffee"], "Prefer tags from this list: coffee"),
        (["coffee", "food", "fresh"], "Prefer tags from this list: coffee, food, fresh"),
    ],
)
def test_build_prompt_with_tags_includes_vocabulary_and_suggestion(
    tags: list[str], expected_snippet: str
) -> None:
    """When a vocabulary is supplied the model must be told about it and that
    a single new tag may be suggested."""
    prompt = _ConcreteProvider().build_prompt(tags)

    assert expected_snippet in prompt
    assert "you may suggest a single new" in prompt
    assert "free-form tags" not in prompt
    # The tag list is embedded, so the prompt is part of the reproducible unit
    # tracked by the benchmark's prompt version.
    assert "lowercase snake_case" in prompt


@pytest.mark.asyncio
async def test_analyse_receipt_with_metadata_forwards_tags() -> None:
    """The compatibility wrapper must forward the tag vocabulary to the
    provider's extraction method."""
    from datetime import date as Date
    from decimal import Decimal

    calls: list[list[str] | None] = []

    class _RecordingProvider(_ConcreteProvider):
        async def analyse_receipt_from_model(self, model_id, image, tags=None):
            calls.append(tags)
            return Receipt(
                confidence=90,
                merchant_name="Shop",
                date=Date(2024, 1, 1),
                line_items=[],
                subtotal=Decimal("1.00"),
                total=Decimal("1.00"),
            )

        async def analyse_receipt_from_ocr_text(self, model_id, ocr_result, tags=None):
            raise NotImplementedError

    provider = _RecordingProvider()
    await provider.analyse_receipt_with_metadata("m", Path("x.png"), tags=["coffee"])

    assert calls == [["coffee"]]
