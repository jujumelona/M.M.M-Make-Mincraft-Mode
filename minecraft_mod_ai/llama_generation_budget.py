from __future__ import annotations

"""Finite native llama.cpp decode-page budgeting.

One model action is always bounded by the live server context. Long tasks continue
through the normal assistant/tool continuation machinery instead of requesting an
unbounded decode from llama-server.
"""

import os
from functools import wraps
from typing import Any

from .model_context_budget import effective_context_tokens, tool_action_token_budget

_MARKER = "_mmm_finite_generation_budget_v1"


def _positive_override(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def plain_action_token_budget(config: Any) -> int:
    """Return one finite text/reasoning page allowance for the active runtime slot."""

    override = _positive_override("MMM_LLAMA_TEXT_MAX_TOKENS")
    if override is not None:
        return override
    configured = int(getattr(config, "max_new_tokens", 0) or 0)
    if configured > 0:
        return configured
    context = effective_context_tokens(config)
    if context <= 0:
        return 8192
    return max(4096, min(16384, context // 2))


def install(hardware_module: Any) -> None:
    """Make the canonical server payload finite for both tool and plain pages."""

    current = hardware_module._server_payload
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def bounded_server_payload(adapter: Any, request: Any) -> dict[str, Any]:
        payload = current(adapter, request)
        if payload.get("tools"):
            budget = tool_action_token_budget(adapter.config)
        else:
            budget = plain_action_token_budget(adapter.config)
        try:
            current_budget = int(payload.get("max_tokens", 0) or 0)
        except (TypeError, ValueError):
            current_budget = 0
        if current_budget <= 0 or current_budget > budget:
            payload["max_tokens"] = budget
        return payload

    setattr(bounded_server_payload, _MARKER, True)
    hardware_module._server_payload = bounded_server_payload


__all__ = ["install", "plain_action_token_budget"]
