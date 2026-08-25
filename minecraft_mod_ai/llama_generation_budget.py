from __future__ import annotations

"""Finite native llama.cpp action-page budgeting.

Every model action is bounded by the live server context. Structured-output semantics
remain host-owned: this layer only decides a finite decode budget and never injects a
second JSON-schema/grammar contract into llama-server.
"""

import os
from functools import wraps
from typing import Any, Mapping

from .model_context_budget import effective_context_tokens, tool_action_token_budget

_MARKER = "_mmm_finite_generation_budget"


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


def action_token_budget(config: Any, *, constrained_action: bool) -> int:
    return (
        tool_action_token_budget(config)
        if constrained_action
        else plain_action_token_budget(config)
    )


def apply_generation_budget(
    payload: Mapping[str, Any],
    *,
    config: Any,
) -> dict[str, Any]:
    """Clamp one payload at construction time; ``-1`` is never a transport value."""

    bounded = dict(payload)
    constrained_action = bool(bounded.get("tools"))
    budget = action_token_budget(config, constrained_action=constrained_action)
    try:
        requested = int(bounded.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0 or requested > budget:
        bounded["max_tokens"] = budget
    return bounded


def install(hardware_module: Any) -> None:
    """Own the finite llama-server generation budget in one idempotent boundary."""

    current = hardware_module._server_payload
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def bounded_server_payload(adapter: Any, request: Any) -> dict[str, Any]:
        return apply_generation_budget(current(adapter, request), config=adapter.config)

    setattr(bounded_server_payload, _MARKER, True)
    hardware_module._server_payload = bounded_server_payload


__all__ = [
    "action_token_budget",
    "apply_generation_budget",
    "install",
    "plain_action_token_budget",
]
