"""Typed JSON literals, escaped and rendered entirely by the Java host."""
from __future__ import annotations

import math
import struct

from .incremental_java_schema import obj
from .incremental_java_values_model import JavaValue, request

_INTEGER_RANGES = {"byte": (-128, 127), "short": (-32768, 32767),
                   "int": (-(2**31), 2**31 - 1), "long": (-(2**63), 2**63 - 1)}
_LITERAL_ALIASES = {"String": "java.lang.String", "Boolean": "boolean", "Byte": "byte",
                    "Short": "short", "Character": "char", "Integer": "int", "Long": "long",
                    "Float": "float", "Double": "double"}
_ESCAPES = {"\\": "\\\\", '"': '\\"', "'": "\\'", "\b": "\\b", "\t": "\\t",
            "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _escape(text, quote):
    answer = []
    for char in text:
        if char in _ESCAPES and (char not in {"'", '"'} or char == quote):
            answer.append(_ESCAPES[char])
        elif ord(char) < 32 or ord(char) == 127:
            answer.append("\\" + format(ord(char), "03o"))
        elif 0xD800 <= ord(char) <= 0xDFFF:
            answer.append("\\u" + format(ord(char), "04x"))
        else:
            answer.append(char)
    return quote + "".join(answer) + quote


def _schema(java_type):
    if java_type in _INTEGER_RANGES:
        lower, upper = _INTEGER_RANGES[java_type]
        return {"type": "integer", "minimum": lower, "maximum": upper}
    if java_type == "boolean":
        return {"type": "boolean"}
    if java_type in {"float", "double"}:
        limit = 3.4028234663852886e38 if java_type == "float" else 1.7976931348623157e308
        return {"type": "number", "minimum": -limit, "maximum": limit}
    if java_type == "char":
        return {"type": "string", "minLength": 1, "maxLength": 1}
    if java_type == "java.lang.String":
        return {"type": "string"}
    raise ValueError("JAVA_STRUCTURED_LITERAL_TYPE_REQUIRED: " + java_type)


def _utf8_width(char):
    code = ord(char)
    if code == 0:
        return 2
    if code < 128:
        return 1
    if code < 2048:
        return 2
    return 3 if code <= 0xFFFF else 6


def _string_literal(text):
    # CONSTANT_Utf8 has an unsigned-short byte length. Split representation,
    # never the model's string data, when one constant cannot carry the value.
    chunks, current, length = [], [], 0
    for char in text:
        if 0xD800 <= ord(char) <= 0xDFFF:
            if current:
                chunks.append(_escape("".join(current), '"'))
            chunks.append("java.lang.String.valueOf((char)" + str(ord(char)) + ")")
            current, length = [], 0
            continue
        width = _utf8_width(char)
        if length + width >= 65535:
            chunks.append(_escape("".join(current), '"'))
            current, length = [], 0
        current.append(char)
        length += width
    chunks.append(_escape("".join(current), '"'))
    if len(chunks) == 1:
        return chunks[0]
    return 'java.lang.String.join("", new java.lang.String[] {' + ", ".join(chunks) + "})"


def _render(value, java_type):
    if java_type == "java.lang.String":
        return _string_literal(value)
    if java_type == "char":
        if ord(value) > 0xFFFF:
            raise ValueError("JAVA_CHAR_LITERAL_REQUIRES_ONE_UTF16_UNIT")
        if 0xD800 <= ord(value) <= 0xDFFF:
            return "(char)" + str(ord(value))
        return _escape(value, "'")
    if java_type == "boolean":
        return "true" if value else "false"
    if java_type in {"float", "double"}:
        if not math.isfinite(value):
            raise ValueError("JAVA_FLOAT_LITERAL_REQUIRES_FINITE_NUMBER")
        if java_type == "float":
            return struct.unpack("!f", struct.pack("!f", value))[0].hex() + "f"
        text = repr(value)
        return text + "d"
    if java_type in {"byte", "short"}:
        return "(" + java_type + ")" + str(int(value))
    return str(int(value)) + ("L" if java_type == "long" else "")


def author_literal(builder, expected):
    selected = builder.select_java_type("literal_type", expected=expected, scalar_only=True)
    simple = selected.removeprefix("java.lang.")
    java_type = _LITERAL_ALIASES.get(simple, selected)
    value = request(builder, obj({"value": _schema(java_type)}), "literal", {"literal_type": java_type})["value"]
    return JavaValue(_render(value, java_type), java_type)
