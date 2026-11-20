"""Bound the OCR wire metadata before constructing or serializing a full page."""

import re
from uuid import UUID

from pydantic import BaseModel
from pydantic_core import from_json, to_json

from . import limits

# Match Pydantic's JSON parser nesting ceiling. The wire model itself has
# shallow envelopes, but telemetry and transforms may contain nested JSON.
_MAX_DEPTH = 200
_TEXT_CHUNK = 4096
_WHITESPACE = re.compile(rb"[ \t\r\n]*")
_SCALAR = re.compile(rb"null|true|false|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_BASE64_BREAK = re.compile(rb"[^A-Za-z0-9+/=]")
_STRING_BREAK = re.compile(rb'["\\\x00-\x1f]')
_ESCAPE = re.compile(rb'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})')


class _MetadataBudget:
    def __init__(self) -> None:
        self.remaining = limits.MAX_RESPONSE_METADATA_BYTES

    def use(self, size: int) -> None:
        self.remaining -= size
        if self.remaining < 0:
            raise ValueError("ocr_response_metadata_too_large")


def check_encoded_metadata(page_without_png: BaseModel) -> None:
    """Measure compact Pydantic JSON using bounded string slices and scalars."""
    budget = _MetadataBudget()

    def walk(value: object, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise ValueError("ocr_invalid_response_metadata")
        if isinstance(value, str):
            # Each character costs at least one UTF-8 JSON byte. Slicing never
            # acquires the entire escaped form of an over-budget string.
            if len(value) + 2 > budget.remaining:
                raise ValueError("ocr_response_metadata_too_large")
            budget.use(2)
            for offset in range(0, len(value), _TEXT_CHUNK):
                encoded = to_json(value[offset : offset + _TEXT_CHUNK])
                budget.use(len(encoded) - 2)
        elif isinstance(value, (BaseModel, dict)):
            budget.use(2)
            entries = value.items() if isinstance(value, dict) else value
            for index, (key, item) in enumerate(entries):
                if not isinstance(key, str):
                    raise TypeError("ocr_invalid_response_metadata")
                budget.use(1 + bool(index))
                walk(key, depth + 1)
                walk(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            budget.use(2)
            for index, item in enumerate(value):
                budget.use(bool(index))
                walk(item, depth + 1)
        elif value is None or isinstance(value, (int, float, UUID)):
            if isinstance(value, int) and value.bit_length() > 4 * budget.remaining:
                raise ValueError("ocr_response_metadata_too_large")
            budget.use(len(to_json(value, inf_nan_mode="null")))
        else:
            raise ValueError("ocr_invalid_response_metadata")

    walk(page_without_png, 0)


class _WirePreflight:
    def __init__(self, payload: bytes | bytearray) -> None:
        self.payload = payload
        self.offset = 0
        self.budget = _MetadataBudget()
        self.working_image_seen = False
        self.png_seen = False

    def _peek(self) -> int:
        whitespace = _WHITESPACE.match(self.payload, self.offset)
        assert whitespace is not None
        self.offset = whitespace.end()
        return self.payload[self.offset] if self.offset < len(self.payload) else -1

    def _take(self, expected: int) -> None:
        if self._peek() != expected:
            raise ValueError("ocr_invalid_response_json")
        self.offset += 1
        self.budget.use(1)

    def _string(self, *, png: bool = False) -> None:
        self._take(34)  # opening quote
        allowance = (
            limits.base64_bound(limits.MAX_WORKING_PNG_BYTES)
            if png else self.budget.remaining
        )
        delimiter = _BASE64_BREAK if png else _STRING_BREAK
        length = 0
        while True:
            token = delimiter.search(self.payload, self.offset)
            if token is None:
                raise ValueError("ocr_invalid_response_json")
            length += token.start() - self.offset
            self.offset = token.start()
            if length > allowance:
                raise ValueError(
                    "ocr_working_png_too_large" if png else "ocr_response_metadata_too_large"
                )
            marker = self.payload[self.offset]
            if marker == 34:
                break
            if marker != 92:
                raise ValueError(
                    "ocr_invalid_working_png_base64" if png else "ocr_invalid_response_json"
                )
            length += self._escape_size(png)
        if not png:
            self.budget.use(length)
        self._take(34)  # closing quote

    def _escape_size(self, png: bool) -> int:
        escape = _ESCAPE.match(self.payload, self.offset)
        if escape is None:
            raise ValueError("ocr_invalid_response_json")
        start, self.offset = self.offset, escape.end()
        marker = self.payload[start + 1]
        if marker != 117:  # u
            if png and marker != 47:  # /
                raise ValueError("ocr_invalid_working_png_base64")
            return 1 if marker == 47 else 2
        codepoint = int(self.payload[start + 2 : self.offset], 16)
        if 0xD800 <= codepoint <= 0xDBFF:
            # Let the bounded scalar parser check the second surrogate.
            self.offset += 6
        character = from_json(b'"' + self.payload[start : self.offset] + b'"')
        if not png:
            return len(to_json(character)) - 2
        if _BASE64_BREAK.search(character.encode("utf-8")):
            raise ValueError("ocr_invalid_working_png_base64")
        return 1

    def _object(self, depth: int, member: str = "") -> None:
        self._take(123)  # {
        if self._peek() != 125:
            while True:
                start = self.offset
                self._string()
                # Even fully escaped, one of the two wire path keys occupies
                # at most six bytes per character plus its two quotes.
                special = (
                    member
                    and self.offset - start <= 6 * len(member) + 2
                    and from_json(self.payload[start : self.offset]) == member
                )
                self._take(58)  # :
                if special and member == "working_image":
                    if self.working_image_seen:
                        raise ValueError("ocr_invalid_response_json")
                    self.working_image_seen = True
                    self._object(depth + 1, "png_base64")
                elif special:
                    if self.png_seen:
                        raise ValueError("ocr_invalid_response_json")
                    self.png_seen = True
                    self._string(png=True)
                else:
                    self._value(depth + 1)
                if self._peek() != 44:
                    break
                self._take(44)  # ,
                self._peek()
        self._take(125)  # }

    def _value(self, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise ValueError("ocr_invalid_response_json")
        marker = self._peek()
        if marker == 34:
            self._string()
        elif marker == 123:
            self._object(depth)
        elif marker == 91:  # [
            self._take(91)
            if self._peek() != 93:
                while True:
                    self._value(depth + 1)
                    if self._peek() != 44:
                        break
                    self._take(44)
            self._take(93)  # ]
        else:
            token = _SCALAR.match(self.payload, self.offset)
            if token is None:
                raise ValueError("ocr_invalid_response_json")
            size = token.end() - self.offset
            # Normalize ordinary scalar spellings (for example 1e-07)
            # without allocating a large token from an untrusted input.
            if size <= _TEXT_CHUNK and marker not in b"ntf":
                size = len(to_json(from_json(token.group()), inf_nan_mode="null"))
            self.budget.use(size)
            self.offset = token.end()

    def check(self) -> None:
        self._object(0, "working_image")
        if self._peek() != -1 or not self.png_seen:
            raise ValueError("ocr_invalid_response_json")


def preflight_wire_metadata(payload: bytes | bytearray) -> None:
    """Walk JSON structure and budget metadata without allocating the decoded graph."""
    _WirePreflight(payload).check()
