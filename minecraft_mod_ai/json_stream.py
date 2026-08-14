from __future__ import annotations

import hashlib
import codecs
import io
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Iterable


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be represented as finite canonical JSON."""


class StreamingJsonDecodeError(ValueError):
    """Raised when chunked UTF-8 input is not one complete JSON value."""


def iter_canonical_json(value: Any) -> Iterable[str]:
    """Yield canonical JSON text without building one monolithic string."""

    active: set[int] = set()
    yield from _iter_value(value, active)


def canonical_json_sha256(value: Any) -> str:
    """Hash canonical JSON with bounded fragment coalescing."""

    digest = hashlib.sha256()
    buffer: list[str] = []
    buffered_characters = 0
    for text in iter_canonical_json(value):
        buffer.append(text)
        buffered_characters += len(text)
        if buffered_characters >= 16 * 1024:
            digest.update("".join(buffer).encode("utf-8"))
            buffer.clear()
            buffered_characters = 0
    if buffer:
        digest.update("".join(buffer).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def validate_canonical_json(value: Any) -> None:
    for _ in iter_canonical_json(value):
        pass


def parse_json_byte_chunks(chunks: Iterable[bytes]) -> Any:
    """Parse UTF-8 JSON incrementally without joining all source chunks."""

    try:
        return _StreamingJsonParser(_Utf8ChunkReader(chunks)).parse()
    except UnicodeDecodeError as exc:
        raise StreamingJsonDecodeError("JSON chunks are not valid UTF-8.") from exc


def _iter_value(value: Any, active: set[int]) -> Iterable[str]:
    if isinstance(value, Enum):
        yield from _iter_value(value.value, active)
        return
    if value is None:
        yield "null"
        return
    if value is True:
        yield "true"
        return
    if value is False:
        yield "false"
        return
    if isinstance(value, str):
        yield from _iter_json_string(value)
        return
    if isinstance(value, int):
        yield str(value)
        return
    if isinstance(value, float):
        try:
            yield json.dumps(value, allow_nan=False, separators=(",", ":"))
        except ValueError as exc:
            raise CanonicalJsonError(
                "JSON contains a non-finite number."
            ) from exc
        return
    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in active:
            raise CanonicalJsonError("JSON contains a circular dataclass.")
        active.add(identity)
        try:
            yield "{"
            ordered = sorted(fields(value), key=lambda item: item.name)
            for index, item in enumerate(ordered):
                if index:
                    yield ","
                yield from _iter_json_string(item.name)
                yield ":"
                yield from _iter_value(getattr(value, item.name), active)
            yield "}"
        finally:
            active.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise CanonicalJsonError("JSON contains a circular array.")
        active.add(identity)
        try:
            yield "["
            for index, item in enumerate(value):
                if index:
                    yield ","
                yield from _iter_value(item, active)
            yield "]"
        finally:
            active.remove(identity)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CanonicalJsonError("JSON object keys must be strings.")
        identity = id(value)
        if identity in active:
            raise CanonicalJsonError("JSON contains a circular object.")
        active.add(identity)
        try:
            yield "{"
            for index, key in enumerate(sorted(value)):
                if index:
                    yield ","
                yield from _iter_json_string(key)
                yield ":"
                yield from _iter_value(value[key], active)
            yield "}"
        finally:
            active.remove(identity)
        return
    raise CanonicalJsonError(
        f"JSON contains an unsupported value: {type(value).__name__}"
    )


def _iter_json_string(value: str) -> Iterable[str]:
    yield '"'
    buffer: list[str] = []
    buffered_characters = 0
    escapes = {
        '"': '\\"',
        "\\": "\\\\",
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    index = 0
    while index < len(value):
        character = value[index]
        index += 1
        escaped = escapes.get(character)
        codepoint = ord(character)
        if escaped is None:
            if 0xD800 <= codepoint <= 0xDBFF:
                if index >= len(value):
                    raise CanonicalJsonError(
                        "JSON string contains a lone high surrogate."
                    )
                low = ord(value[index])
                if not 0xDC00 <= low <= 0xDFFF:
                    raise CanonicalJsonError(
                        "JSON string high surrogate is not paired."
                    )
                index += 1
                escaped = chr(
                    0x10000
                    + ((codepoint - 0xD800) << 10)
                    + (low - 0xDC00)
                )
            elif 0xDC00 <= codepoint <= 0xDFFF:
                raise CanonicalJsonError(
                    "JSON string contains a lone low surrogate."
                )
            elif codepoint < 0x20:
                escaped = f"\\u{codepoint:04x}"
            else:
                escaped = character
        buffer.append(escaped)
        buffered_characters += len(escaped)
        if buffered_characters >= 16 * 1024:
            yield "".join(buffer)
            buffer.clear()
            buffered_characters = 0
    if buffer:
        yield "".join(buffer)
    yield '"'


class _Utf8ChunkReader:
    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = iter(chunks)
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._position = 0
        self._finished = False

    def peek(self) -> str | None:
        self._fill()
        if self._position >= len(self._buffer):
            return None
        return self._buffer[self._position]

    def take(self) -> str | None:
        value = self.peek()
        if value is not None:
            self._position += 1
        return value

    def _fill(self) -> None:
        if self._position < len(self._buffer) or self._finished:
            return
        self._buffer = ""
        self._position = 0
        while not self._finished and not self._buffer:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                self._buffer = self._decoder.decode(b"", final=True)
                self._finished = True
                return
            if not isinstance(chunk, bytes):
                raise StreamingJsonDecodeError(
                    "JSON chunk iterator yielded a non-bytes value."
                )
            self._buffer = self._decoder.decode(chunk, final=False)


class _StreamingJsonParser:
    def __init__(self, reader: _Utf8ChunkReader) -> None:
        self.reader = reader

    def parse(self) -> Any:
        self._skip_whitespace()
        value = self._value()
        self._skip_whitespace()
        if self.reader.peek() is not None:
            raise StreamingJsonDecodeError(
                "JSON contains trailing content."
            )
        return value

    def _value(self) -> Any:
        character = self.reader.peek()
        if character is None:
            raise StreamingJsonDecodeError("JSON ended before a value.")
        if character == '"':
            return self._string()
        if character == "{":
            return self._object()
        if character == "[":
            return self._array()
        if character == "t":
            self._literal("true")
            return True
        if character == "f":
            self._literal("false")
            return False
        if character == "n":
            self._literal("null")
            return None
        if character == "-" or character.isdigit():
            return self._number()
        raise StreamingJsonDecodeError(
            f"Unexpected JSON character: {character!r}"
        )

    def _object(self) -> dict[str, Any]:
        self._expect("{")
        result: dict[str, Any] = {}
        self._skip_whitespace()
        if self.reader.peek() == "}":
            self.reader.take()
            return result
        while True:
            if self.reader.peek() != '"':
                raise StreamingJsonDecodeError(
                    "JSON object key must be a string."
                )
            key = self._string()
            self._skip_whitespace()
            self._expect(":")
            self._skip_whitespace()
            result[key] = self._value()
            self._skip_whitespace()
            delimiter = self.reader.take()
            if delimiter == "}":
                return result
            if delimiter != ",":
                raise StreamingJsonDecodeError(
                    "JSON object is missing a comma or closing brace."
                )
            self._skip_whitespace()

    def _array(self) -> list[Any]:
        self._expect("[")
        result: list[Any] = []
        self._skip_whitespace()
        if self.reader.peek() == "]":
            self.reader.take()
            return result
        while True:
            result.append(self._value())
            self._skip_whitespace()
            delimiter = self.reader.take()
            if delimiter == "]":
                return result
            if delimiter != ",":
                raise StreamingJsonDecodeError(
                    "JSON array is missing a comma or closing bracket."
                )
            self._skip_whitespace()

    def _string(self) -> str:
        self._expect('"')
        output = io.StringIO()
        while True:
            character = self.reader.take()
            if character is None:
                raise StreamingJsonDecodeError(
                    "JSON string ended before its closing quote."
                )
            if character == '"':
                return output.getvalue()
            if character == "\\":
                escape = self.reader.take()
                if escape is None:
                    raise StreamingJsonDecodeError(
                        "JSON string ended inside an escape."
                    )
                mapped = {
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                    "b": "\b",
                    "f": "\f",
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                }.get(escape)
                if mapped is not None:
                    output.write(mapped)
                    continue
                if escape != "u":
                    raise StreamingJsonDecodeError(
                        f"Invalid JSON string escape: \\{escape}"
                    )
                digits = "".join(
                    self.reader.take() or ""
                    for _ in range(4)
                )
                if len(digits) != 4 or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in digits
                ):
                    raise StreamingJsonDecodeError(
                        "Invalid JSON unicode escape."
                    )
                codepoint = int(digits, 16)
                if 0xD800 <= codepoint <= 0xDBFF:
                    if self.reader.take() != "\\" or self.reader.take() != "u":
                        raise StreamingJsonDecodeError(
                            "JSON high surrogate is not followed by a low surrogate."
                        )
                    low_digits = "".join(
                        self.reader.take() or ""
                        for _ in range(4)
                    )
                    if len(low_digits) != 4 or any(
                        character not in "0123456789abcdefABCDEF"
                        for character in low_digits
                    ):
                        raise StreamingJsonDecodeError(
                            "Invalid JSON low-surrogate escape."
                        )
                    low = int(low_digits, 16)
                    if not 0xDC00 <= low <= 0xDFFF:
                        raise StreamingJsonDecodeError(
                            "JSON high surrogate is not paired with a low surrogate."
                        )
                    output.write(
                        chr(
                            0x10000
                            + ((codepoint - 0xD800) << 10)
                            + (low - 0xDC00)
                        )
                    )
                    continue
                if 0xDC00 <= codepoint <= 0xDFFF:
                    raise StreamingJsonDecodeError(
                        "JSON contains a lone low surrogate."
                    )
                output.write(chr(codepoint))
                continue
            if ord(character) < 0x20:
                raise StreamingJsonDecodeError(
                    "JSON string contains an unescaped control character."
                )
            output.write(character)

    def _number(self) -> int | float:
        characters: list[str] = []
        while True:
            character = self.reader.peek()
            if character is None or character not in "-+0123456789.eE":
                break
            characters.append(str(self.reader.take()))
        token = "".join(characters)
        try:
            value = json.loads(token)
        except json.JSONDecodeError as exc:
            raise StreamingJsonDecodeError(
                f"Invalid JSON number: {token!r}"
            ) from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StreamingJsonDecodeError(
                f"Invalid JSON number: {token!r}"
            )
        return value

    def _literal(self, expected: str) -> None:
        actual = "".join(self.reader.take() or "" for _ in expected)
        if actual != expected:
            raise StreamingJsonDecodeError(
                f"Invalid JSON literal: {actual!r}"
            )

    def _expect(self, expected: str) -> None:
        actual = self.reader.take()
        if actual != expected:
            raise StreamingJsonDecodeError(
                f"Expected {expected!r}, found {actual!r}."
            )

    def _skip_whitespace(self) -> None:
        while self.reader.peek() in {" ", "\t", "\r", "\n"}:
            self.reader.take()
