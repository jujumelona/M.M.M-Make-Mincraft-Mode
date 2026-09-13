from __future__ import annotations

"""Canonical mandatory message envelope for forced model tool decisions."""

from collections.abc import Mapping, Sequence
from typing import Any

from .model_router import (
    _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT,
    _inject_system_context,
)


def mandatory_tool_decision_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_name: str,
) -> tuple[dict[str, Any], ...]:
    """Return the router-owned mandatory messages that precede context fitting.

    This mirrors the immutable envelope added by ``ModelRouter.generate_tool_decision``.
    Callers that admit bounded source payloads use it before inference so their capacity
    check includes repository policy and the required-function protocol rather than only
    the caller-authored semantic messages.
    """

    name = str(tool_name or "").strip()
    if not name:
        raise ValueError("tool_name must not be empty")
    request_messages = _inject_system_context(
        messages,
        _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT,
    )
    return (
        *request_messages,
        {
            "role": "system",
            "content": (
                f"Call the required function {name} exactly once. "
                "Do not answer in prose."
            ),
        },
    )


__all__ = ["mandatory_tool_decision_messages"]
