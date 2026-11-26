"""Resource bounds shared by the native OCR writer and its readers."""

APP_SOURCE_MAX_BYTES = 20 * 1024 * 1024
NATIVE_SOURCE_MAX_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_OCR_TEXT_CHARS = 200_000
MAX_RESPONSE_METADATA_BYTES = 8 * 1024 * 1024


def working_png_bound(max_pixels: int) -> int:
    """Bound the metadata-free RGB8 encoder in document_image.py."""
    if max_pixels < 1:
        raise ValueError("A PNG requires at least one pixel")

    # PNG filtering emits one selector per row: (3W + 1)H <= 4WH.
    scanlines = 4 * max_pixels

    # Pillow 12.3 ZipEncode.c uses a 15-bit window, memory level 9, no
    # dictionary, Z_NO_FLUSH and Z_FINISH. Use zlib-ng's conservative
    # deflateBound, which also dominates zlib's bound for this configuration.
    # The 11 bytes include block overhead (5) and the zlib header/trailer (6).
    deflated = scanlines + (scanlines + 7) // 8 + (scanlines + 63) // 64 + 11

    # ImageFile._save uses buffers of max(65536, 4W) bytes. ZipEncode.c
    # fills every buffer before yielding except its final (possibly empty)
    # buffer. PngImagePlugin._idat writes each buffer as a 12-byte-framed chunk.
    idat_chunks = deflated // 65_536 + 1
    # PNG signature (8), IHDR (25), IEND (12).
    return 45 + deflated + 12 * idat_chunks


def base64_bound(raw_bytes: int) -> int:
    """Size of RFC 4648 base64 with padding and no line wrapping."""
    if raw_bytes < 0:
        raise ValueError("A byte count cannot be negative")
    return 4 * ((raw_bytes + 2) // 3)


MAX_WORKING_PNG_BYTES = working_png_bound(MAX_IMAGE_PIXELS)
# Canonical UTF-8 metadata already includes every delimiter, including the PNG
# string's quotes. Pydantic's compact JSON emits base64 ASCII without escaping.
MAX_OCR_RESPONSE_BYTES = MAX_RESPONSE_METADATA_BYTES + base64_bound(MAX_WORKING_PNG_BYTES)


def check_ocr_response_size(size: int) -> None:
    if size > MAX_OCR_RESPONSE_BYTES:
        raise ValueError("ocr_response_too_large")


def check_ocr_response_headers(content_length: str | None, content_encoding: str) -> None:
    # Identity avoids decompressing an arbitrarily large HTTP chunk before the
    # byte limit can be checked. Clients advertise it on their OCR requests.
    if content_encoding.strip().lower() not in ("", "identity"):
        raise ValueError("ocr_response_unsupported_encoding")
    if content_length is not None:
        size = int(content_length)
        if size < 0:
            raise ValueError("ocr_response_invalid_length")
        check_ocr_response_size(size)


def append_ocr_response_chunk(body: bytearray, chunk: bytes) -> None:
    check_ocr_response_size(len(body) + len(chunk))
    body.extend(chunk)


def check_working_image_dimensions(width: int, height: int) -> None:
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("ocr_invalid_working_image_dimensions")


def check_working_png_base64(encoded: str) -> None:
    length = len(encoded)
    if length > base64_bound(MAX_WORKING_PNG_BYTES):
        raise ValueError("ocr_working_png_too_large")
    # A string can fit the encoded cap and still decode to too many bytes if
    # it has less padding than the maximum raw length requires.
    if length % 4:
        raise ValueError("ocr_invalid_working_png_base64")
    padding = 2 if encoded.endswith("==") else int(encoded.endswith("="))
    if length // 4 * 3 - padding > MAX_WORKING_PNG_BYTES:
        raise ValueError("ocr_working_png_too_large")
