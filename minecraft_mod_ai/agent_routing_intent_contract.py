from __future__ import annotations

"""One terminal-intent projection shared by small-model and causal tool routing."""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False


def install(*, small_model_module: Any) -> None:
    """Route tool retrieval from the same structured user intent as causal planning.

    ``small_model_max_agent_contract`` historically built its routing query from the
    last 12 KiB of user/system/tool prose. Large custom-module JSON places phase/task
    before large research/grounding blobs, so tail truncation can erase the actual
    coding goal. The live causal adapter already owns a structured, user-only intent
    projection; reuse that exact projection rather than maintaining two drifting
    parsers.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    current = small_model_module._request_query
    if getattr(current, "_mmm_structured_terminal_intent", False):
        _INSTALLED = True
        return

    from .agent_intent import structured_user_intent

    @wraps(current)
    def request_query(messages: Sequence[Mapping[str, Any]]) -> str:
        return structured_user_intent(messages)

    request_query._mmm_structured_terminal_intent = True  # type: ignore[attr-defined]
    request_query.__wrapped__ = current  # type: ignore[attr-defined]
    small_model_module._request_query = request_query
    _INSTALLED = True


__all__ = ["install"]
