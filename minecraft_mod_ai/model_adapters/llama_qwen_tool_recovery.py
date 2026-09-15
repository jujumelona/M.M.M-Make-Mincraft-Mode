"""Fail-closed recovery for Qwen native tool markup returned as assistant content.

llama-server can occasionally leave Qwen's native ``<tool_call>`` markup in
``message.content`` while returning an empty ``message.tool_calls`` array.  This module
bridges only that transport mismatch: native structured calls stay authoritative, and
recovered calls still pass the existing request-visible schema validation surface.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .base import GenerationRequest, GenerationResponse
from .qwen_tool_parser import ToolCallValidationError, parse_qwen_tool_markup
from . import llama_cpp_adapter as _llama

_ORIGINAL_DECODER = _llama._native_tool_generation_response


def _contains_qwen_tool_markup(text: str) -> bool:
    stripped = text.lstrip()
    return "<tool_call>" in text or stripped.startswith("<function=")


def _recover_qwen_content_tool_calls(
    message: Mapping[str, Any],
    request: GenerationRequest,
    *,
    original_decoder: Callable[[Mapping[str, Any], GenerationRequest], GenerationResponse] = _ORIGINAL_DECODER,
) -> GenerationResponse:
    """Recover strict Qwen markup only when llama-server omitted native tool_calls."""

    raw_calls = _llama._raw_native_tool_calls(message)
    if raw_calls:
        return original_decoder(message, request)

    content = message.get("content")
    content_text = content if isinstance(content, str) else ""
    if not _contains_qwen_tool_markup(content_text):
        return original_decoder(message, request)

    normalized = _llama._normalized_tool_request(request)
    schemas = _llama._request_tool_schema_map(normalized)
    visible_content, parsed_calls = parse_qwen_tool_markup(content_text, schemas)
    if not parsed_calls:
        return original_decoder(message, normalized)
    if not normalized.parallel_tool_calls and len(parsed_calls) > 1:
        raise ToolCallValidationError(
            "model emitted parallel tool calls when they are disabled"
        )

    valid_calls, schema_rejections = _llama._partition_tool_calls_against_host_schema(
        parsed_calls, schemas
    )
    if not valid_calls and schema_rejections:
        error = str(
            schema_rejections[0].arguments.get("error", "invalid recovered tool call")
        )
        raise ToolCallValidationError(error)

    reasoning = message.get("reasoning_content", message.get("reasoning"))
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    return GenerationResponse(
        content=visible_content.strip(),
        tool_calls=(*valid_calls, *schema_rejections),
        reasoning_content=reasoning_text.strip(),
    )


def install_llama_qwen_tool_recovery() -> None:
    """Install one idempotent adapter-boundary decoder wrapper."""

    current = _llama._native_tool_generation_response
    if getattr(current, "_mmm_qwen_content_tool_recovery", False):
        return

    def _wrapped(
        message: Mapping[str, Any], request: GenerationRequest
    ) -> GenerationResponse:
        return _recover_qwen_content_tool_calls(
            message,
            request,
            original_decoder=current,
        )

    _wrapped._mmm_qwen_content_tool_recovery = True  # type: ignore[attr-defined]
    _llama._native_tool_generation_response = _wrapped


__all__ = [
    "install_llama_qwen_tool_recovery",
    "_recover_qwen_content_tool_calls",
]
