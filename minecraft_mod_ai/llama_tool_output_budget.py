from __future__ import annotations

"""Bounded output policy for native llama.cpp tool turns.

The model-facing source ACI is scalar: one reviewed action is emitted per turn and
large multi-edit arrays are host-owned.  An 8K default therefore wastes decode budget
on every retrieval/edit decision.  Keep 4K as the conservative default so create-file
operations can still carry a substantial UTF-8 source body, while explicit operators
may raise the bound through ``MMM_LLAMA_TOOL_MAX_TOKENS`` when a deployment genuinely
needs larger single actions.  Final text synthesis keeps the model's normal output
budget because this policy applies only while tool schemas are exposed.
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
