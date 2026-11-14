"""Tests for the OCR provider interface and PaddleOCR implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vision_bill.config import OcrSettings
from vision_bill.provider.factory import get_ocr_provider
from vision_bill.provider.ocr.base import OcrProvider
from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider


@pytest.mark.parametrize(
    "provider, expected_class",
    [
        ("paddleocr", PaddleOcrProvider),
    ],
)
def test_get_ocr_provider_success(provider, expected_class):
    """Test that the correct OCR provider is returned."""
    settings = OcrSettings(provider=provider)
    ocr_provider = get_ocr_provider(settings)
    assert isinstance(ocr_provider, expected_class)


def test_get_ocr_provider_failure():
    """Test that a ValueError is raised for unsupported OCR providers."""
    settings = OcrSettings()
    settings.provider = "non_existent_provider"

    with pytest.raises(ValueError, match="Unsupported OCR provider"):
        get_ocr_provider(settings)


@pytest.mark.asyncio
async def test_paddleocr_extract_text_success(tmp_path):
    """Test successful OCR text extraction."""
    provider = PaddleOcrProvider(service_url="http://localhost:8082")

    # Create a temporary image file
    image_file = tmp_path / "test_image.jpg"
    image_file.write_bytes(b"fake image data")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "Test receipt text", "blocks": []}

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = mock_client.return_value.__aenter__.return_value
        mock_instance.post.return_value = mock_response

        result = await provider.extract_text(image_file)

        assert result["text"] == "Test receipt text"
        mock_instance.post.assert_called_once()


@pytest.mark.asyncio
async def test_paddleocr_extract_text_failure():
    """Test OCR extraction failure."""
    provider = PaddleOcrProvider(service_url="http://localhost:8082")

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = mock_client.return_value.__aenter__.return_value
        mock_instance.post.return_value = mock_response

        with pytest.raises(Exception):
            await provider.extract_text("/path/to/image.jpg")


@pytest.mark.asyncio
async def test_paddleocr_check_connection_success():
    """Test successful connection check."""
    provider = PaddleOcrProvider(service_url="http://localhost:8082")

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = mock_client.return_value.__aenter__.return_value
        mock_instance.get.return_value = mock_response

        result = await provider.check_connection()
        assert result is True


@pytest.mark.asyncio
async def test_paddleocr_check_connection_failure():
    """Test connection check failure."""
    import httpx as real_httpx

    provider = PaddleOcrProvider(service_url="http://localhost:8082")

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = mock_client.return_value.__aenter__.return_value
        mock_instance.get.side_effect = real_httpx.ConnectError("Connection failed")

        result = await provider.check_connection()
        assert result is False
