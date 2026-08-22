from __future__ import annotations

"""Output policy for native llama.cpp tool turns.

Compact retrieval/navigation actions benefit from a bounded scalar decode budget, but
source mutation actions carry reviewed source text inside the tool arguments.  Those
payload-heavy actions must retain the model/runtime ``max_new_tokens`` budget selected
by the normal server payload policy.  Applying the compact 4K cap to ``apply_source_*``
turns creates a second, hidden output-budget owner and can truncate a valid action
before its JSON/tool envelope closes.
"""

import os
from functools import wraps
from typing import Any, Mapping

_MARKER = "_mmm_llama_tool_output_budget_v2"
_DEFAULT_TOOL_MAX_TOKENS = 4096
_MAX_TOOL_MAX_TOKENS = 16384
_PAYLOAD_HEAVY_TOOL_NAMES = frozenset({
    "apply_source_edit",
    "apply_source_patch",
})


def _tool_max_tokens() -> int:
    """Return the operator-configurable cap for compact scalar tool turns."""

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
    """Return the positive compact-tool decode reserve, bounded by model config."""

    limit = _tool_max_tokens()
    try:
        configured = int(getattr(config, "max_new_tokens", 0) or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        limit = min(limit, configured)
    return max(1, limit)


def _tool_name(tool: Any) -> str:
    """Extract a tool name from OpenAI-style mappings or typed tool objects."""

    if isinstance(tool, Mapping):
        name = tool.get("name")
        if isinstance(name, str) and name:
            return name
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name:
                return name

    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    function = getattr(tool, "function", None)
    name = getattr(function, "name", None)
    if isinstance(name, str) and name:
        return name
    return ""


def _tool_names(request: Any) -> frozenset[str]:
    tools = getattr(request, "tools", ()) or ()
    return frozenset(name for name in (_tool_name(tool) for tool in tools) if name)


def _requires_model_output_budget(request: Any) -> bool:
    """Return True when a tool can legitimately carry a large source payload."""

    names = _tool_names(request)
    return bool(names & _PAYLOAD_HEAVY_TOOL_NAMES)


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
    """Cap compact tool turns without overriding payload-heavy mutation budgets."""

    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def bounded_tool_payload(adapter: Any, request: Any) -> dict[str, Any]:
        payload = dict(current(adapter, request))
        tools = getattr(request, "tools", ()) or ()
        if tools and not _requires_model_output_budget(request):
            payload["max_tokens"] = _bounded_tool_limit(adapter, payload)
        return payload

    setattr(bounded_tool_payload, _MARKER, True)
    hardware_module._server_payload = bounded_tool_payload


__all__ = [
    "_bounded_tool_limit",
    "_requires_model_output_budget",
    "_tool_max_tokens",
    "_tool_names",
    "install",
    "tool_output_budget",
]
