from __future__ import annotations

"""Final output-budget guard for native llama.cpp tool turns.

Tool-capable turns only need enough output to emit a semantic action or a compact
host-parsed tool payload. Letting them inherit ``max_tokens=-1`` can consume the
remaining context window before the action closes, especially after large RAG/tool
observations. Top-level text generation remains governed by the existing model policy.
"""

import os
from functools import wraps
from typing import Any, Mapping

_MARKER = "_mmm_llama_tool_output_budget_v1"
_DEFAULT_TOOL_MAX_TOKENS = 4096
_MAX_TOOL_MAX_TOKENS = 16384


def _tool_max_tokens() -> int:
    raw = os.environ.get("MMM_LLAMA_TOOL_MAX_TOKENS", "").strip()
    if not raw:
        return _DEFAULT_TOOL_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MMM_LLAMA_TOOL_MAX_TOKENS must be a positive integer") from exc
    if value <= 0:
        raise ValueError("MMM_LLAMA_TOOL_MAX_TOKENS must be a positive integer")
    return min(value, _MAX_TOOL_MAX_TOKENS)


def _bounded_tool_limit(
    adapter: Any,
    payload: Mapping[str, Any],
) -> int:
    candidates = [_tool_max_tokens()]
    for value in (
        getattr(getattr(adapter, "config", None), "max_new_tokens", None),
        payload.get("max_tokens"),
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            candidates.append(parsed)
    return max(1, min(candidates))


def install(hardware_module: Any) -> None:
    """Keep tool turns bounded after every earlier output-policy wrapper."""

    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def bounded_tool_payload(adapter: Any, request: Any) -> dict[str, Any]:
        payload = dict(current(adapter, request))
        if getattr(request, "tools", ()) or ():
            payload["max_tokens"] = _bounded_tool_limit(adapter, payload)
        return payload

    setattr(bounded_tool_payload, _MARKER, True)
    hardware_module._server_payload = bounded_tool_payload


__all__ = ["_bounded_tool_limit", "_tool_max_tokens", "install"]
