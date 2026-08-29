from __future__ import annotations

"""Hardening for tool-enabled coder turns.

Tool calls used by the production coder intentionally bypass the JSON structured-output
repair path. That also meant they bypassed the token/reasoning budget policy entirely,
so a large accumulated planning/research transcript could leave only a few hundred
tokens for ``apply_source_edit``. This module keeps the tool schema and newest task
context intact while compacting stale transcript text before payload construction.
"""

import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .pipeline_hardening import _replace_bound_references

_INSTALLED = False
_MARKER = "_mmm_source_edit_budget_v7"
_SOURCE_EDIT_TOOL = "apply_source_edit"


def _tool_names(request: Any) -> tuple[str, ...]:
    names: list[str] = []
    for tool in getattr(request, "tools", ()) or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = str(function.get("name", "")).strip()
        else:
            name = str(tool.get("name", "")).strip()
        if name:
            names.append(name)
    return tuple(dict.fromkeys(names))


def _is_source_edit_request(request: Any) -> bool:
    return _SOURCE_EDIT_TOOL in _tool_names(request)


def _message_content(message: Any) -> str:
    if not isinstance(message, Mapping):
        return ""
    value = message.get("content", "")
    if isinstance(value, str):
        return value
    return str(value or "")


def _clip_text(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    marker = "\n...[host compacted stale coder context]...\n"
    if budget <= len(marker) + 32:
        return text[-budget:]
    remaining = budget - len(marker)
    head = max(16, remaining // 3)
    tail = remaining - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def _copy_message_with_content(message: Mapping[str, Any], content: str) -> dict[str, Any]:
    copied = dict(message)
    copied["content"] = content
    return copied


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value)


def _compact_source_edit_request(request: Any) -> tuple[Any, int, int]:
    """Compact stale text while preserving tool metadata and newest task messages."""

    messages = tuple(getattr(request, "messages", ()) or ())
    original_chars = sum(len(_message_content(message)) for message in messages)

    trigger = _env_int("MMM_CODER_TOOL_COMPACT_TRIGGER_CHARS", 36000, 8000)
    if original_chars <= trigger:
        return request, original_chars, original_chars

    target = min(
        trigger,
        _env_int("MMM_CODER_TOOL_CONTEXT_CHARS", 26000, 12000),
    )

    first_system_index: int | None = None
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and str(message.get("role", "")) == "system":
            first_system_index = index
            break

    system_budget = (
        min(8000, max(2000, target // 4))
        if first_system_index is not None
        else 0
    )
    remaining = target - system_budget
    allocations = [0] * len(messages)

    if first_system_index is not None:
        allocations[first_system_index] = system_budget

    old_floor = 256
    for index in range(len(messages) - 1, -1, -1):
        if index == first_system_index:
            continue
        length = len(_message_content(messages[index]))
        if not length:
            continue
        if remaining <= 0:
            allocations[index] = min(old_floor, length)
            continue
        if index >= max(0, len(messages) - 5):
            grant = min(length, remaining)
        else:
            grant = min(length, max(old_floor, remaining // max(1, index + 1)))
        allocations[index] = grant
        remaining -= min(remaining, grant)

    compacted: list[Any] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            compacted.append(message)
            continue
        content = _message_content(message)
        if not content:
            compacted.append(dict(message))
            continue
        compacted.append(
            _copy_message_with_content(
                message,
                _clip_text(content, allocations[index]),
            )
        )

    compact_chars = sum(len(_message_content(message)) for message in compacted)
    try:
        compact_request = replace(request, messages=tuple(compacted))
    except TypeError:
        return request, original_chars, original_chars
    return compact_request, original_chars, compact_chars


def _configured_max_tokens(adapter: Any) -> int:
    config = getattr(adapter, "config", None)
    return max(1, int(getattr(config, "max_new_tokens", 1) or 1))


def _source_edit_min_tokens(adapter: Any) -> int:
    configured = _configured_max_tokens(adapter)
    requested = _env_int("MMM_CODER_TOOL_MIN_OUTPUT_TOKENS", 2048, 512)
    return min(configured, requested)


def _install_source_edit_payload_hardening() -> None:
    from . import llama_server_hardware_policy as hardware

    original = hardware._server_payload
    if getattr(original, _MARKER, False):
        return

    def hardened(adapter: Any, request: Any) -> dict[str, Any]:
        if not _is_source_edit_request(request):
            return original(adapter, request)

        compact_request, original_chars, compact_chars = _compact_source_edit_request(request)
        result = dict(original(adapter, compact_request))

        # Raise the output floor only after compaction actually reclaimed context.
        # This prevents overriding a legitimate context clamp if nothing was freed.
        if compact_chars < original_chars:
            current_max = max(1, int(result.get("max_tokens", 1) or 1))
            result["max_tokens"] = max(
                current_max,
                _source_edit_min_tokens(adapter),
            )

        # Source-edit tool calls need deterministic tool/body capacity rather than
        # hidden reasoning tokens.
        result["reasoning_effort"] = "none"
        template_kwargs = dict(result.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = False
        result["chat_template_kwargs"] = template_kwargs
        result.pop("thinking_budget_tokens", None)

        print(
            "coder source-edit hardening:"
            f" input_chars={original_chars}->{compact_chars}"
            f" max_tokens={result.get('max_tokens')}"
            " thinking=off",
            file=sys.stderr,
            flush=True,
        )
        return result

    setattr(hardened, _MARKER, True)
    hardware._server_payload = hardened
    _replace_bound_references(original, hardened)


def install_pipeline_hardening_v7() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_source_edit_payload_hardening()
    _INSTALLED = True


__all__ = [
    "_compact_source_edit_request",
    "_is_source_edit_request",
    "install_pipeline_hardening_v7",
]
