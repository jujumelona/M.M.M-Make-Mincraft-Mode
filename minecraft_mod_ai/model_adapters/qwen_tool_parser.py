"""Qwen tool-markup transport parser.

This module owns syntax recovery only. It does not know or enforce host tool schemas,
required fields, enums, defaults, visibility, or execution policy. Every parsed candidate
is handed to the adapter's single admission boundary before it can execute.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .base import ToolCall

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "</parameter>"
MALFORMED_TOOL_CALL_NAME = "__mmm_malformed_tool_call__"
_ARGUMENT_CONTAINER_KEYS = frozenset({"apply", "arguments", "args", "parameters", "params", "input", "payload"})


def _call_id(index: int, name: str, raw: str) -> str:
    digest = hashlib.sha256(f"{index}\0{name}\0{raw}".encode()).hexdigest()[:16]
    return f"call_{digest}"


def _decode_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    lowered = text.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    return text


def _malformed(index: int, raw: str, error: str, *, original_tool: str = "") -> ToolCall:
    payload = {
        "original_tool": original_tool,
        "failure_code": "TOOL_MARKUP_MALFORMED",
        "error": " ".join(str(error).split())[:320],
        "raw_arguments": raw[:4000],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ToolCall(
        id=_call_id(index, MALFORMED_TOOL_CALL_NAME, serialized),
        name=MALFORMED_TOOL_CALL_NAME,
        arguments=payload,
        raw_arguments=serialized,
    )


def _next_call_start(text: str, cursor: int) -> int:
    starts = [
        pos for pos in (text.find(TOOL_CALL_OPEN, cursor), text.find(FUNCTION_OPEN, cursor))
        if pos >= 0
    ]
    return min(starts) if starts else -1


def _bounded_end(text: str, start: int) -> int:
    """Bound one malformed candidate without leaking wrapper syntax into prose."""

    wrapped = text.startswith(TOOL_CALL_OPEN, start)
    inner_start = start + len(TOOL_CALL_OPEN) if wrapped else start
    while inner_start < len(text) and text[inner_start].isspace():
        inner_start += 1
    # A wrapper and its immediate function form one candidate, even when decoding
    # ended before either closing tag. Never count that owned function as a sibling.
    search_start = inner_start
    if text.startswith(FUNCTION_OPEN, inner_start):
        search_start += len(FUNCTION_OPEN)
    next_start = _next_call_start(text, max(search_start, start + 1))
    limit = next_start if next_start >= 0 else len(text)

    if wrapped:
        wrapped_end = text.find(TOOL_CALL_CLOSE, inner_start, limit)
        if wrapped_end >= 0:
            return wrapped_end + len(TOOL_CALL_CLOSE)

    function_end = text.find(FUNCTION_CLOSE, inner_start, limit)
    if function_end >= 0:
        return function_end + len(FUNCTION_CLOSE)

    return limit


def _parse_function(text: str, function_start: int, index: int) -> tuple[ToolCall, int]:
    name_start = function_start + len(FUNCTION_OPEN)
    name_end = text.find(">", name_start)
    if name_end < 0:
        raise ValueError("function tag is missing '>'")
    name = text[name_start:name_end].strip()
    if not name:
        raise ValueError("function tag has an empty tool name")

    function_close = text.find(FUNCTION_CLOSE, name_end + 1)
    if function_close < 0:
        raise ValueError("function block is missing </function>")

    arguments: dict[str, Any] = {}
    cursor = name_end + 1
    while cursor < function_close:
        parameter_start = text.find(PARAMETER_OPEN, cursor, function_close)
        if parameter_start < 0:
            if text[cursor:function_close].strip():
                raise ValueError("unexpected text exists inside function block")
            break
        if text[cursor:parameter_start].strip():
            raise ValueError("unexpected text exists before parameter tag")
        key_start = parameter_start + len(PARAMETER_OPEN)
        key_end = text.find(">", key_start, function_close)
        if key_end < 0:
            raise ValueError("parameter tag is missing '>'")
        key = text[key_start:key_end].strip()
        if not key:
            raise ValueError("parameter tag has an empty name")
        value_end = text.find(PARAMETER_CLOSE, key_end + 1, function_close)
        if value_end < 0:
            raise ValueError(f"parameter {key!r} is missing </parameter>")
        arguments[key] = _decode_value(text[key_end + 1:value_end])
        cursor = value_end + len(PARAMETER_CLOSE)

    for _depth in range(3):
        if len(arguments) != 1:
            break
        only_key, only_value = next(iter(arguments.items()))
        if only_key.casefold() not in _ARGUMENT_CONTAINER_KEYS or not isinstance(only_value, Mapping):
            break
        arguments = dict(only_value)

    raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), default=str)
    return (
        ToolCall(
            id=_call_id(index, name, raw_arguments),
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
        ),
        function_close + len(FUNCTION_CLOSE),
    )



def _malformed_original_tool(text: str, function_start: int, end: int) -> str:
    if not text.startswith(FUNCTION_OPEN, function_start):
        return ""
    name_start = function_start + len(FUNCTION_OPEN)
    name_end = text.find(">", name_start, end)
    if name_end < 0:
        return ""
    return text[name_start:name_end].strip()


def parse_qwen_tool_markup(
    text: str,
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, tuple[ToolCall, ...]]:
    """Recover Qwen tool candidates without making admission or schema decisions.

    ``schemas`` is accepted only for call-site compatibility. The transport parser never
    reads it. Malformed markup becomes a non-executable candidate that the adapter rejects
    through the same admission path as malformed native tool calls.
    """
    del schemas
    if not text:
        return "", ()

    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        start = _next_call_start(text, cursor)
        if start < 0:
            break
        wrapped = text.startswith(TOOL_CALL_OPEN, start)
        function_start = start + len(TOOL_CALL_OPEN) if wrapped else start
        while function_start < len(text) and text[function_start].isspace():
            function_start += 1
        end = _bounded_end(text, start)
        try:
            if not text.startswith(FUNCTION_OPEN, function_start):
                raise ValueError("tool_call block does not begin with a function")
            # Closing tags from a later sibling cannot complete this candidate.
            call, function_end = _parse_function(text[:end], function_start, len(calls))
            end = function_end
            if wrapped:
                close_at = function_end
                while close_at < len(text) and text[close_at].isspace():
                    close_at += 1
                if not text.startswith(TOOL_CALL_CLOSE, close_at):
                    raise ValueError("tool_call block is missing </tool_call>")
                end = close_at + len(TOOL_CALL_CLOSE)
            calls.append(call)
        except (TypeError, ValueError) as exc:
            raw = text[start:end]
            original_tool = _malformed_original_tool(text, function_start, end)
            calls.append(_malformed(len(calls), raw, str(exc), original_tool=original_tool))
        spans.append((start, max(end, start + 1)))
        cursor = max(end, start + 1)

    if not spans:
        return text, ()
    visible: list[str] = []
    previous = 0
    for begin, end in spans:
        if begin > previous:
            visible.append(text[previous:begin])
        previous = max(previous, end)
    visible.append(text[previous:])
    return "".join(visible).strip(), tuple(calls)
