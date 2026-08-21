from __future__ import annotations

"""Bounded output policy for native llama.cpp tool turns.

Tool calls must remain bounded, but source-edit actions can legitimately need several
thousand tokens of structured payload.  The bound therefore follows the configured
model output budget (up to a hard host cap) instead of forcing every tool action into
an arbitrary 4K decode.  Input fitting reserves this same budget so increasing the
tool allowance cannot steal space from the runtime context window.
"""

import os
from functools import wraps
from typing import Any, Mapping

_MARKER = "_mmm_llama_tool_output_budget_v1"
_DEFAULT_TOOL_MAX_TOKENS = 8192
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


def tool_output_budget(config: Any) -> int:
    """Return the authoritative positive output reserve for one tool turn."""

    limit = _tool_max_tokens()
    try:
        configured = int(getattr(config, "max_new_tokens", 0) or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        limit = min(limit, configured)
    return max(1, limit)


def _bounded_tool_limit(
    adapter: Any,
    payload: Mapping[str, Any],
) -> int:
    candidates = [tool_output_budget(getattr(adapter, "config", None))]
    try:
        payload_limit = int(payload.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        payload_limit = 0
    if payload_limit > 0:
        candidates.append(payload_limit)
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


__all__ = [
    "_bounded_tool_limit",
    "_tool_max_tokens",
    "install",
    "tool_output_budget",
]
