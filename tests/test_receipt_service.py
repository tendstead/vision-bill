import io
from collections.abc import Generator
from datetime import date as Date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import UploadFile

from vision_bill.config import Settings
from vision_bill.model.collection import Collection
from vision_bill.model.db.image import ImageRow
from vision_bill.model.db.receipt import ReceiptRow
from vision_bill.model.receipt import LineItem, Receipt, TaxLine
from vision_bill.model.search import ProductPurchase
from vision_bill.provider.llm.base import LLMProvider, ModelInfo
from vision_bill.service.receipt_service import ReceiptService

ServiceContext = tuple[ReceiptService, MagicMock, MagicMock, MagicMock]
DelegationContext = tuple[ReceiptService, MagicMock, MagicMock]
RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000001")
OTHER_RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000002")
IMAGE_ID = UUID("00000000-0000-4000-8000-000000000003")
OTHER_IMAGE_ID = UUID("00000000-0000-4000-8000-000000000004")


@pytest.mark.parametrize("entry", ["path", "paths", "bytes", "all_models"])
async def test_legacy_helpers_cannot_bypass_configured_ocr_on_failure(settings, tmp_path, entry):
    from PIL import Image
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.config import OcrSettings

    paths = [tmp_path / f"{i}.png" for i in range(2)]
    for path in paths:
        Image.new("RGB", (4, 5)).save(path)
    llm = formatter()
    ocr = sidecar(1 if entry == "paths" else 0)
    service = ReceiptService(
        settings.images, settings.pg, llm, ocr_provider=ocr,
        ocr_settings=OcrSettings(enabled=True, allow_vision_fallback=True),
    )
    if entry == "all_models":
        results = await service.extract_receipt_all_models(
            UploadFile(file=io.BytesIO(paths[0].read_bytes()))
        )
        assert len(results) == 1
        assert results[0].receipt is None and results[0].error == "ocr_page_failed"
        assert llm.get_available_models.call_args_list[0].kwargs == {"capability": "text"}
    else:
        with pytest.raises(ValueError, match="ocr_page_failed"):
            if entry == "bytes":
                await service.analyse_receipt_from_model("fixture", paths[0].read_bytes())
            else:
                await service.analyse_receipt_from_path("fixture", paths if entry == "paths" else paths[0])
    llm._client.chat.assert_not_awaited()
    assert ocr.extract_page.await_count == (2 if entry == "paths" else 1)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Fixture for a mocked LLM provider."""
    provider = MagicMock(spec=LLMProvider)
    provider.analysis_timeout_seconds = 600
    provider.analyse_receipt_from_model = AsyncMock()
    provider.get_available_models = AsyncMock(return_value=[])
    return provider


@pytest.fixture
def receipt_service_context(
    settings: Settings, mock_provider: MagicMock
) -> Generator[ServiceContext, None, None]:
    """
    Context fixture that provides a ReceiptService with mocked dependencies.
    Returns a tuple: (service, mock_provider, mock_image_service, mock_path)
    """
    with patch("vision_bill.service.receipt_service.ImageService") as mock_image_service_class:
        mock_image_service = mock_image_service_class.return_value

        # Setup mock path
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_image_service.store_tmp_image.return_value = mock_path

        # Inject mock_provider directly into ReceiptService
        service = ReceiptService(settings.images, settings.pg, mock_provider)
        yield service, mock_provider, mock_image_service, mock_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "merchant_name, subtotal, total",
    [
        ("Test Vendor", "10.50", "10.50"),
        ("Grocery Store", "100.00", "105.50"),
    ],
)
async def test_analyse_receipt_from_model_success(
    receipt_service_context: ServiceContext,
    merchant_name: str,
    subtotal: str,
    total: str,
) -> None:
    # Arrange
    service, mock_provider, _, mock_path = receipt_service_context
    mock_receipt = Receipt(
        confidence=95,
        merchant_name=merchant_name,
        date=Date(2023, 1, 1),
        currency="USD",
        line_items=[],
        subtotal=Decimal(subtotal),
        total=Decimal(total),
    )
    mock_provider.analyse_receipt_from_model.return_value = mock_receipt

    image_bytes = b"fake-image-content"
    model_id = "test-model"

    # Act
    result = await service.analyse_receipt_from_model(model_id, image_bytes)

    # Assert
    assert result == mock_receipt
    # No DB in this context -> the provider gets an empty tag vocabulary.
    mock_provider.analyse_receipt_from_model.assert_called_once_with(model_id, mock_path, tags=[])
    mock_path.unlink.assert_called_once()


@pytest.mark.asyncio
async def test_analyse_receipt_from_model_failure(
    receipt_service_context: ServiceContext,
) -> None:
    # Arrange
    service, mock_provider, _, mock_path = receipt_service_context
    error_msg = "LLM Error"
    mock_provider.analyse_receipt_from_model.side_effect = Exception(error_msg)

    image_bytes = b"fake-image-content"
    model_id = "test-model"

    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await service.analyse_receipt_from_model(model_id, image_bytes)

    assert error_msg in str(exc_info.value)
    mock_provider.analyse_receipt_from_model.assert_called_once_with(model_id, mock_path, tags=[])
    mock_path.unlink.assert_called_once()


@pytest.mark.asyncio
async def test_extract_receipt_all_models_success(
    receipt_service_context: ServiceContext,
) -> None:
    # Arrange
    service, mock_provider, _, _ = receipt_service_context
    model1 = ModelInfo(id="model1")
    model2 = ModelInfo(id="model2")
    mock_provider.get_available_models.return_value = [model1, model2]

    receipt1 = Receipt(
        confidence=95,
        merchant_name="Merchant 1",
        date=Date(2023, 1, 1),
        currency="USD",
        line_items=[],
        subtotal=Decimal("10.00"),
        total=Decimal("10.00"),
    )
    receipt2 = Receipt(
        confidence=95,
        merchant_name="Merchant 2",
        date=Date(2023, 1, 1),
        currency="USD",
        line_items=[],
        subtotal=Decimal("20.00"),
        total=Decimal("20.00"),
    )

    # Use side_effect to return different values for different calls
    async def side_effect(model_id: str, path: Path, tags: list[str] | None = None) -> Receipt:
        if model_id == "model1":
            return receipt1
        elif model_id == "model2":
            return receipt2
        raise ValueError("Unexpected model id")

    mock_provider.analyse_receipt_from_model.side_effect = side_effect

    # Mock UploadFile
    image_content = b"fake-image-content"
    upload_file = UploadFile(file=io.BytesIO(image_content))

    # Act
    results = await service.extract_receipt_all_models(upload_file)

    # Assert
    assert len(results) == 2
    # Test Case 1: Success
    res1 = next(r for r in results if r.model_id == "model1")
    res2 = next(r for r in results if r.model_id == "model2")

    assert res1.receipt == receipt1
    assert res1.error is None
    assert res2.receipt == receipt2
    assert res2.error is None


@pytest.mark.asyncio
async def test_extract_receipt_all_models_value_error(
    receipt_service_context: ServiceContext,
) -> None:
    # Arrange
    service, mock_provider, _, _ = receipt_service_context
    model1 = ModelInfo(id="model1")
    model2 = ModelInfo(id="model2")
    mock_provider.get_available_models.return_value = [model1, model2]

    receipt2 = Receipt(
        confidence=95,
        merchant_name="Merchant 2",
        date=Date(2023, 1, 1),
        currency="USD",
        line_items=[],
        subtotal=Decimal("20.00"),
        total=Decimal("20.00"),
    )

    async def side_effect(model_id: str, path: Path, tags: list[str] | None = None) -> Receipt:
        if model_id == "model1":
            raise ValueError("Invalid receipt format")
        elif model_id == "model2":
            return receipt2
        raise ValueError("Unexpected model id")

    mock_provider.analyse_receipt_from_model.side_effect = side_effect

    image_content = b"fake-image-content"
    upload_file = UploadFile(file=io.BytesIO(image_content))

    # Act
    results = await service.extract_receipt_all_models(upload_file)

    # Assert
    assert len(results) == 2
    # Test Case 2: ValueError Handling
    res1 = next(r for r in results if r.model_id == "model1")
    res2 = next(r for r in results if r.model_id == "model2")

    assert res1.receipt is None
    assert res1.error is not None
    assert "Invalid receipt format" in res1.error
    assert res2.receipt == receipt2
    assert res2.error is None


@pytest.mark.asyncio
async def test_extract_receipt_all_models_exception(
    receipt_service_context: ServiceContext,
) -> None:
    # Arrange
    service, mock_provider, _, _ = receipt_service_context
    model1 = ModelInfo(id="model1")
    model2 = ModelInfo(id="model2")
    mock_provider.get_available_models.return_value = [model1, model2]

    receipt2 = Receipt(
        confidence=95,
        merchant_name="Merchant 2",
        date=Date(2023, 1, 1),
        currency="USD",
        line_items=[],
        subtotal=Decimal("20.00"),
        total=Decimal("20.00"),
    )

    async def side_effect(model_id: str, path: Path, tags: list[str] | None = None) -> Receipt:
        if model_id == "model1":
            raise RuntimeError("Generic failure")
        elif model_id == "model2":
            return receipt2
        raise ValueError("Unexpected model id")

    mock_provider.analyse_receipt_from_model.side_effect = side_effect

    image_content = b"fake-image-content"
    upload_file = UploadFile(file=io.BytesIO(image_content))

    # Act
    results = await service.extract_receipt_all_models(upload_file)

    # Assert
    assert len(results) == 2
    # Test Case 3: Exception Handling
    res1 = next(r for r in results if r.model_id == "model1")
    res2 = next(r for r in results if r.model_id == "model2")

    assert res1.receipt is None
    assert res1.error is not None
    assert "Generic failure" in res1.error
    assert res2.receipt == receipt2
    assert res2.error is None


# ── Database delegation tests ────────────────────────────────────────


def _make_receipt() -> Receipt:
    return Receipt(
        confidence=95,
        merchant_name="Test Store",
        merchant_address="123 Main St",
        receipt_number="R-001",
        date=Date(2024, 1, 15),
        time="14:30",
        currency="USD",
        category="grocery",
        line_items=[
            LineItem(
                description="Item A",
                quantity=2,
                unit_price=Decimal("10.00"),
                total_price=Decimal("20.00"),
                tags=["test"],
            )
        ],
        taxes=[TaxLine(name="VAT", rate=0.19, amount=Decimal("4.50"))],
        subtotal=Decimal("50.00"),
        discount_total=Decimal(0),
        tax_total=Decimal("4.50"),
        total=Decimal("54.50"),
        payment_method="credit_card",
    )


def _make_row(**overrides: object) -> ReceiptRow:
    base: dict[str, Any] = {
        "id": RECEIPT_ID,
        "confidence": 95,
        "merchant_name": "Test Store",
        "merchant_address": "123 Main St",
        "receipt_number": "R-001",
        "date": Date(2024, 1, 15),
        "time": "14:30:00",
        "currency": "USD",
        "category": "other",
        "subtotal": Decimal("50.00"),
        "discount_total": Decimal(0),
        "tax_total": Decimal("4.50"),
        "tip": None,
        "total": Decimal("54.50"),
        "payment_method": "credit_card",
        "created_at": Date(2024, 1, 15),
        "status": "unverified",
        "image_id": None,
        "verified": False,
    }
    base.update(overrides)
    return ReceiptRow(**base)


@pytest.fixture
def delegation_context(settings: Settings) -> Generator[DelegationContext, None, None]:
    """ReceiptService with the ReceiptDB and ImageDB providers patched out."""
    with (
        patch("vision_bill.service.receipt_service.ReceiptDB") as mock_db_class,
        patch("vision_bill.service.receipt_service.ImageDB") as mock_image_db_class,
        patch("vision_bill.service.receipt_service.UserDB") as mock_user_db_class,
        patch("vision_bill.service.receipt_service.WorkflowDB") as mock_workflow_db_class,
    ):
        mock_db = mock_db_class.return_value
        mock_db.init_db = AsyncMock()
        mock_db.destroy_db = AsyncMock()
        mock_db.persist_receipt = AsyncMock()
        mock_db.get_receipt_by_id = AsyncMock()
        mock_db.list_receipts = AsyncMock()
        mock_db.search_products = AsyncMock()
        mock_db.get_receipt_with_details = AsyncMock()
        mock_db.update_receipt = AsyncMock()
        mock_db.verify_receipt = AsyncMock()
        mock_db.list_tags = AsyncMock(return_value=[])
        mock_db.create_tag = AsyncMock(return_value=True)
        mock_db.pool = MagicMock()
        mock_db.is_ready = True

        mock_image_db = mock_image_db_class.return_value
        mock_image_db.init_db = AsyncMock()
        mock_image_db.destroy_db = AsyncMock()
        mock_image_db.is_ready = True
        mock_image_db.store_image = AsyncMock()
        mock_image_db.get_image_by_id = AsyncMock()
        mock_image_db.list_pending_images = AsyncMock()
        mock_image_db.mark_analyzed = AsyncMock()
        mock_image_db.mark_failed = AsyncMock()
        mock_image_db.update_image_path = AsyncMock()
        mock_image_db.delete_image = AsyncMock()

        mock_user_db = mock_user_db_class.return_value
        mock_user_db.init_db = AsyncMock()
        mock_user_db.destroy_db = AsyncMock()
        mock_user_db.is_ready = True

        mock_workflow_db_class.return_value.init_db = AsyncMock()
        mock_workflow_db_class.return_value.destroy_db = AsyncMock()
        service = ReceiptService(settings.images, settings.pg, MagicMock(spec=LLMProvider))
        yield service, mock_db, mock_image_db


@pytest.mark.asyncio
async def test_init_db_delegates(delegation_context: DelegationContext) -> None:
    """init_db should delegate to both the ReceiptDB and ImageDB providers."""
    service, mock_db, mock_image_db = delegation_context

    await service.init_db()

    mock_db.init_db.assert_awaited_once_with()
    mock_image_db.init_db.assert_awaited_once_with()
    service.user_db.init_db.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_destroy_db_delegates(delegation_context: DelegationContext) -> None:
    """destroy_db should delegate to both the ReceiptDB and ImageDB providers."""
    service, mock_db, mock_image_db = delegation_context

    await service.destroy_db()

    mock_db.destroy_db.assert_awaited_once_with()
    mock_image_db.destroy_db.assert_awaited_once_with()
    service.user_db.destroy_db.assert_awaited_once_with()


def test_pool_property_delegates(delegation_context: DelegationContext) -> None:
    """pool should be the ReceiptDB pool."""
    service, mock_db, _ = delegation_context

    assert service.pool is mock_db.pool


def test_db_ready_reflects_db_is_ready(delegation_context: DelegationContext) -> None:
    """db_ready should mirror ReceiptDB.is_ready."""
    service, mock_db, _ = delegation_context

    mock_db.is_ready = True
    assert service.db_ready is True

    mock_db.is_ready = False
    assert service.db_ready is False


@pytest.mark.asyncio
async def test_persist_receipt_delegates(delegation_context: DelegationContext) -> None:
    """persist_receipt should delegate with image_id and status."""
    service, mock_db, _ = delegation_context
    receipt = _make_receipt()
    row = _make_row()
    mock_db.persist_receipt.return_value = row

    result = await service.persist_receipt(
        receipt, image_id=IMAGE_ID, status="verified", verified=True
    )

    mock_db.persist_receipt.assert_awaited_once_with(
        receipt, image_id=IMAGE_ID, status="verified", verified=True, user_id=None
    )
    assert result is row


@pytest.mark.asyncio
async def test_get_receipt_by_id_delegates(delegation_context: DelegationContext) -> None:
    """get_receipt_by_id should delegate with the receipt id."""
    service, mock_db, _ = delegation_context
    row = _make_row(id=RECEIPT_ID)
    mock_db.get_receipt_by_id.return_value = row

    result = await service.get_receipt_by_id(RECEIPT_ID)

    mock_db.get_receipt_by_id.assert_awaited_once_with(RECEIPT_ID, user_id=None, can_see_all=False)
    assert result is row


@pytest.mark.asyncio
async def test_list_receipts_delegates(delegation_context: DelegationContext) -> None:
    """list_receipts should delegate with limit and offset."""
    service, mock_db, _ = delegation_context
    rows: list[ReceiptRow] = [
        _make_row(id=RECEIPT_ID),
        _make_row(id=OTHER_RECEIPT_ID),
    ]
    mock_db.list_receipts.return_value = rows

    result = await service.list_receipts(limit=10, offset=5)

    mock_db.list_receipts.assert_awaited_once_with(
        limit=10,
        offset=5,
        status=None,
        date_from=None,
        date_to=None,
        search=None,
        collection_id=None,
        user_id=None,
        can_see_all=False,
    )
    assert result is rows


@pytest.mark.asyncio
async def test_search_keeps_purchases_with_missing_prices_without_counting_them_in_summary(
    delegation_context: DelegationContext,
) -> None:
    service, mock_db, _ = delegation_context
    missing = ProductPurchase(
        receipt_id=RECEIPT_ID, description="Eggs", merchant_name="Store A",
        date=Date(2026, 3, 1), quantity=None, unit_price=None, currency="EUR",
    )
    known = missing.model_copy(update={
        "receipt_id": OTHER_RECEIPT_ID, "quantity": 1, "unit_price": Decimal("2.49"),
    })
    mock_db.search_products.return_value = [missing, known]

    result = await service.search_products("Eggs")
    assert result.purchases == [missing, known]
    assert result.latest_price == result.cheapest_price == result.average_price == Decimal("2.49")

    mock_db.search_products.return_value = [missing]
    result = await service.search_products("Eggs")
    assert result.purchases == [missing]
    assert result.latest_price is result.cheapest_price is result.average_price is None


@pytest.mark.asyncio
async def test_search_products_calculates_unit_price_summary(
    delegation_context: DelegationContext,
) -> None:
    """search_products delegates to the DB and summarizes returned prices."""
    service, mock_db, _ = delegation_context
    purchases = [
        ProductPurchase(
            receipt_id=RECEIPT_ID,
            description="Gouda Mittelalt",
            merchant_name="Store A",
            date=Date(2024, 3, 1),
            quantity=1,
            unit_price=Decimal("3.20"),
            currency="EUR",
        ),
        ProductPurchase(
            receipt_id=OTHER_RECEIPT_ID,
            description="Gouda",
            merchant_name="Store B",
            date=Date(2024, 2, 1),
            quantity=1,
            unit_price=Decimal("2.80"),
            currency="EUR",
        ),
    ]
    mock_db.search_products.return_value = purchases

    result = await service.search_products("Gouda")

    mock_db.search_products.assert_awaited_once_with("Gouda", user_id=None, can_see_all=False)
    assert result.query == "Gouda"
    assert result.purchases == purchases
    assert result.latest_price == Decimal("3.20")
    assert result.cheapest_price == Decimal("2.80")
    assert result.average_price == Decimal("3.00")
    assert result.currency == "EUR"


@pytest.mark.asyncio
async def test_search_products_cheapest_price_ignores_negative_refund_lines(
    delegation_context: DelegationContext,
) -> None:
    """cheapest_price is the lowest non-negative unit price.

    A Pfand refund line item is negative and must not be reported as the
    cheapest price.
    """
    service, mock_db, _ = delegation_context
    mock_db.search_products.return_value = [
        ProductPurchase(
            receipt_id=RECEIPT_ID,
            description="Gouda",
            merchant_name="Store A",
            date=Date(2024, 3, 1),
            quantity=1,
            unit_price=Decimal("3.20"),
            currency="EUR",
        ),
        ProductPurchase(
            receipt_id=OTHER_RECEIPT_ID,
            description="Gouda",
            merchant_name="Store B",
            date=Date(2024, 2, 1),
            quantity=1,
            unit_price=Decimal("2.80"),
            currency="EUR",
        ),
        ProductPurchase(
            receipt_id=RECEIPT_ID,
            description="Pfand (Flaschen)",
            merchant_name="Store A",
            date=Date(2024, 3, 1),
            quantity=1,
            unit_price=Decimal("-0.25"),
            currency="EUR",
        ),
    ]

    result = await service.search_products("Gouda")

    assert result.cheapest_price == Decimal("2.80")


@pytest.mark.asyncio
async def test_search_products_cheapest_price_none_when_only_refund_lines(
    delegation_context: DelegationContext,
) -> None:
    """With no non-negative unit price to report, cheapest_price is None."""
    service, mock_db, _ = delegation_context
    mock_db.search_products.return_value = [
        ProductPurchase(
            receipt_id=RECEIPT_ID,
            description="Pfand (Flaschen)",
            merchant_name="Store A",
            date=Date(2024, 3, 1),
            quantity=1,
            unit_price=Decimal("-0.25"),
            currency="EUR",
        ),
        ProductPurchase(
            receipt_id=OTHER_RECEIPT_ID,
            description="Pfand (Flaschen)",
            merchant_name="Store B",
            date=Date(2024, 2, 1),
            quantity=1,
            unit_price=Decimal("-0.75"),
            currency="EUR",
        ),
    ]

    result = await service.search_products("Pfand")

    assert result.cheapest_price is None


@pytest.mark.asyncio
async def test_search_products_cheapest_price_includes_free_lines(
    delegation_context: DelegationContext,
) -> None:
    """A free (0.00) line item is non-negative, so it is the cheapest price."""
    service, mock_db, _ = delegation_context
    mock_db.search_products.return_value = [
        ProductPurchase(
            receipt_id=RECEIPT_ID,
            description="Free Sample",
            merchant_name="Store A",
            date=Date(2024, 3, 1),
            quantity=1,
            unit_price=Decimal("0.00"),
            currency="EUR",
        ),
        ProductPurchase(
            receipt_id=OTHER_RECEIPT_ID,
            description="Gouda",
            merchant_name="Store B",
            date=Date(2024, 2, 1),
            quantity=1,
            unit_price=Decimal("2.80"),
            currency="EUR",
        ),
    ]

    result = await service.search_products("Gouda")

    assert result.cheapest_price == Decimal("0.00")


@pytest.mark.asyncio
async def test_get_receipt_with_details_delegates(
    delegation_context: DelegationContext,
) -> None:
    """get_receipt_with_details should delegate with the receipt id."""
    service, mock_db, _ = delegation_context
    mock_db.get_receipt_with_details.return_value = None

    result = await service.get_receipt_with_details(RECEIPT_ID)

    mock_db.get_receipt_with_details.assert_awaited_once_with(
        RECEIPT_ID, user_id=None, can_see_all=False
    )
    assert result is None


@pytest.mark.asyncio
async def test_update_receipt_delegates(delegation_context: DelegationContext) -> None:
    """update_receipt should delegate with id and receipt."""
    service, mock_db, _ = delegation_context
    receipt = _make_receipt()
    row = _make_row()
    mock_db.update_receipt.return_value = row

    result = await service.update_receipt(RECEIPT_ID, receipt)

    mock_db.update_receipt.assert_awaited_once_with(
        RECEIPT_ID, receipt, user_id=None, can_see_all=False
    )
    assert result is row


@pytest.mark.asyncio
async def test_verify_receipt_delegates(delegation_context: DelegationContext) -> None:
    """verify_receipt should delegate with the receipt id only (no image path)."""
    service, mock_db, _ = delegation_context
    row = _make_row(status="verified")
    mock_db.verify_receipt.return_value = row

    result = await service.verify_receipt(RECEIPT_ID)

    mock_db.verify_receipt.assert_awaited_once_with(RECEIPT_ID, None, scope=None, user_id=None, can_see_all=False)
    assert result is row


# ── Tag vocabulary delegation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tags_delegates(delegation_context: DelegationContext) -> None:
    """list_tags should return the DB vocabulary as-is."""
    service, mock_db, _ = delegation_context
    mock_db.list_tags.return_value = ["coffee", "food"]

    result = await service.list_tags()

    mock_db.list_tags.assert_awaited_once_with()
    assert result == ["coffee", "food"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw, normalized",
    [
        ("coffee", "coffee"),
        ("  Coffee  ", "coffee"),
        ("  Hot   Drink ", "hot drink"),
    ],
)
async def test_create_tag_normalizes_and_reports_created(
    delegation_context: DelegationContext, raw: str, normalized: str
) -> None:
    """create_tag trims/collapses whitespace and lower-cases before insert."""
    service, mock_db, _ = delegation_context
    mock_db.create_tag.return_value = True

    name, created = await service.create_tag(raw)

    mock_db.create_tag.assert_awaited_once_with(normalized)
    assert name == normalized
    assert created is True


@pytest.mark.asyncio
async def test_create_tag_existing_is_not_created(delegation_context: DelegationContext) -> None:
    """When the DB reports the tag already exists, created is False (no error)."""
    service, mock_db, _ = delegation_context
    mock_db.create_tag.return_value = False

    name, created = await service.create_tag("coffee")

    assert name == "coffee"
    assert created is False


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
async def test_create_tag_blank_raises(delegation_context: DelegationContext, raw: str) -> None:
    """Blank names never reach the DB."""
    service, mock_db, _ = delegation_context

    with pytest.raises(ValueError, match="must not be blank"):
        await service.create_tag(raw)

    mock_db.create_tag.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_tag_too_long_raises(delegation_context: DelegationContext) -> None:
    """Names longer than the column width are rejected before the DB."""
    service, mock_db, _ = delegation_context

    with pytest.raises(ValueError, match="at most 100"):
        await service.create_tag("x" * 101)

    mock_db.create_tag.assert_not_awaited()


# ── Image (images table) delegation tests ────────────────────────────


def _make_image_row(**overrides: object) -> ImageRow:
    base: dict[str, Any] = {
        "id": IMAGE_ID,
        "original_filename": "a.png",
        "media_type": "image/png",
        "size_bytes": 123,
        "image_path": "/tmp/a.png",
        "status": "pending",
        "error": None,
        "receipt_id": None,
        "bypass_review": False,
        "created_at": Date(2024, 1, 15),
        "analyzed_at": None,
    }
    base.update(overrides)
    return ImageRow(**base)


@pytest.mark.asyncio
async def test_store_image_delegates(delegation_context: DelegationContext) -> None:
    """store_image should delegate to the ImageDB provider with all kwargs."""
    service, _, mock_image_db = delegation_context
    row = _make_image_row()
    mock_image_db.store_image.return_value = row

    result = await service.store_image(
        image_path="/tmp/a.png",
        original_filename="a.png",
        media_type="image/png",
        size_bytes=123,
        status="pending",
        bypass_review=True,
    )

    mock_image_db.store_image.assert_awaited_once_with(
        image_path="/tmp/a.png",
        original_filename="a.png",
        media_type="image/png",
        size_bytes=123,
        status="pending",
        user_id=None,
        bypass_review=True,
        thumbnail_path=None,
        additional_images=None,
        model_id=None,
    )
    assert result is row


@pytest.mark.asyncio
async def test_get_image_by_id_delegates(delegation_context: DelegationContext) -> None:
    """get_image_by_id should delegate with the image id."""
    service, _, mock_image_db = delegation_context
    row = _make_image_row(id=IMAGE_ID)
    mock_image_db.get_image_by_id.return_value = row

    result = await service.get_image_by_id(IMAGE_ID)

    mock_image_db.get_image_by_id.assert_awaited_once_with(
        IMAGE_ID, user_id=None, can_see_all=False
    )
    assert result is row


@pytest.mark.asyncio
async def test_list_pending_images_delegates(delegation_context: DelegationContext) -> None:
    """list_pending_images should delegate to the ImageDB provider."""
    service, _, mock_image_db = delegation_context
    rows: list[ImageRow] = [
        _make_image_row(id=IMAGE_ID),
        _make_image_row(id=OTHER_IMAGE_ID, status="failed"),
    ]
    mock_image_db.list_pending_images.return_value = rows

    result = await service.list_pending_images()

    mock_image_db.list_pending_images.assert_awaited_once_with()
    assert result is rows


@pytest.mark.asyncio
async def test_mark_image_analyzed_delegates(delegation_context: DelegationContext) -> None:
    """mark_image_analyzed should delegate to ImageDB.mark_analyzed."""
    service, _, mock_image_db = delegation_context

    await service.mark_image_analyzed(IMAGE_ID, RECEIPT_ID)

    mock_image_db.mark_analyzed.assert_awaited_once_with(IMAGE_ID, RECEIPT_ID)


@pytest.mark.asyncio
async def test_mark_image_failed_delegates(delegation_context: DelegationContext) -> None:
    """mark_image_failed should delegate to ImageDB.mark_failed."""
    service, _, mock_image_db = delegation_context

    await service.mark_image_failed(IMAGE_ID, "boom")

    mock_image_db.mark_failed.assert_awaited_once_with(IMAGE_ID, "boom")


@pytest.mark.asyncio
async def test_update_image_path_delegates(delegation_context: DelegationContext) -> None:
    """update_image_path should delegate to the ImageDB provider."""
    service, _, mock_image_db = delegation_context

    await service.update_image_path(IMAGE_ID, "/save/b.png")

    mock_image_db.update_image_path.assert_awaited_once_with(IMAGE_ID, "/save/b.png")


@pytest.mark.asyncio
async def test_delete_image_row_delegates(delegation_context: DelegationContext) -> None:
    """delete_image_row should delegate to ImageDB.delete_image."""
    service, _, mock_image_db = delegation_context

    await service.delete_image_row(IMAGE_ID)

    mock_image_db.delete_image.assert_awaited_once_with(IMAGE_ID)


# ── Auto-capture into the active collection ────────────────────────────

OWNER_ID = UUID("00000000-0000-4000-8000-00000000000a")
ACTIVE_COLLECTION_ID = UUID("00000000-0000-4000-8000-0000000000c1")


def _active_collection() -> "Collection":
    return Collection(id=ACTIVE_COLLECTION_ID, name="Berlin", active=True)


@pytest.mark.asyncio
async def test_persist_receipt_captures_to_active_collection(delegation_context) -> None:
    service, mock_db, _ = delegation_context
    row = _make_row()
    mock_db.persist_receipt = AsyncMock(return_value=row)
    coll_db = AsyncMock()
    coll_db.active_collection = AsyncMock(return_value=_active_collection())
    coll_db.assign = AsyncMock()
    service.set_collection_db(coll_db)

    result = await service.persist_receipt(_make_receipt(), user_id=OWNER_ID)

    assert result is not None
    coll_db.active_collection.assert_awaited_once_with(OWNER_ID, can_see_all=False)
    coll_db.assign.assert_awaited_once_with(ACTIVE_COLLECTION_ID, RECEIPT_ID, user_id=OWNER_ID)


@pytest.mark.asyncio
async def test_persist_receipt_without_owner_not_captured(delegation_context) -> None:
    service, mock_db, _ = delegation_context
    mock_db.persist_receipt = AsyncMock(return_value=_make_row())
    coll_db = AsyncMock()
    service.set_collection_db(coll_db)

    await service.persist_receipt(_make_receipt(), user_id=None)

    coll_db.active_collection.assert_not_called()
    coll_db.assign.assert_not_called()


@pytest.mark.asyncio
async def test_persist_receipt_capture_failure_is_fail_safe(delegation_context) -> None:
    service, mock_db, _ = delegation_context
    mock_db.persist_receipt = AsyncMock(return_value=_make_row())
    coll_db = AsyncMock()
    coll_db.active_collection = AsyncMock(side_effect=RuntimeError("boom"))
    service.set_collection_db(coll_db)

    row = await service.persist_receipt(_make_receipt(), user_id=OWNER_ID)  # must not raise

    assert row is not None


async def test_workflow_provider_lifecycle_and_readiness(delegation_context):
    service, _, _ = delegation_context
    database = service.workflow_db
    database.is_ready = False
    assert service.workflow_ready is False
    await service.init_db()
    database.init_db.assert_awaited_once_with()
    database.is_ready = True
    assert service.workflow_ready is True
    await service.destroy_db()
    database.destroy_db.assert_awaited_once_with()
