from __future__ import annotations

"""Finite native llama.cpp generation budgeting.

The backend-specific wrapper delegates budget ownership to the transport-neutral
``generation_output_budget`` policy. This module remains the idempotent llama-server
installation boundary for compatibility with runtime bootstrap.
"""

from collections.abc import Mapping
from functools import wraps
from typing import Any

from .generation_output_budget import (
    apply_payload_generation_budget,
    generation_output_token_budget,
)
from .model_context_budget import tool_action_token_budget

_MARKER = "_mmm_finite_generation_budget"


def plain_action_token_budget(config: Any) -> int:
    """Return the finite plain-text budget for the active runtime slot."""

    return generation_output_token_budget(config, input_tokens=0, tools=())


def action_token_budget(config: Any, *, constrained_action: bool) -> int:
    """Compatibility helper for callers that do not expose the concrete tool schemas."""

    if constrained_action:
        return tool_action_token_budget(config)
    return plain_action_token_budget(config)


def apply_generation_budget(
    payload: Mapping[str, Any],
    *,
    config: Any,
) -> dict[str, Any]:
    """Apply the common finite output policy; ``-1`` is never a transport value."""

    return apply_payload_generation_budget(payload, config=config)


def install(hardware_module: Any) -> None:
    """Install the common output budget at the llama-server payload boundary."""

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
