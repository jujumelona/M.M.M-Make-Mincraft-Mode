from __future__ import annotations

"""Finite default safety budget for the host-owned agent tool loop.

No-progress convergence already stops repeated or duplicate work, but genuinely novel
observations can otherwise keep the loop alive forever when ``MMM_AGENT_TOOL_ROUNDS`` is
unset. Preserve explicit operator limits and the ability to exceed old eight-round
limits; only the implicit-unbounded case receives a generous finite safety ceiling.
"""

import os
from functools import wraps
from typing import Any

_MARKER = "_mmm_finite_default_tool_rounds_v1"
_DEFAULT_TOOL_ROUNDS = 128
_MIN_DEFAULT_TOOL_ROUNDS = 16
_MAX_DEFAULT_TOOL_ROUNDS = 512


def _default_round_limit() -> int:
    raw = os.environ.get("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "").strip()
    if not raw:
        return _DEFAULT_TOOL_ROUNDS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_TOOL_ROUNDS
    return max(_MIN_DEFAULT_TOOL_ROUNDS, min(_MAX_DEFAULT_TOOL_ROUNDS, value))


def install(model_router_module: Any) -> None:
    current = model_router_module._agent_tool_round_limit
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def finite_tool_round_limit() -> int:
        explicit = current()
        if explicit is not None:
            return int(explicit)
        return _default_round_limit()

    setattr(finite_tool_round_limit, _MARKER, True)
    finite_tool_round_limit.__wrapped__ = current  # type: ignore[attr-defined]
    model_router_module._agent_tool_round_limit = finite_tool_round_limit


__all__ = ["_default_round_limit", "install"]
