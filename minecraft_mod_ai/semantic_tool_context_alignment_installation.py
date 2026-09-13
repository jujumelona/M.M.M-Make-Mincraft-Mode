from __future__ import annotations

"""Align semantic-research admission with the real forced-tool request envelope.

Semantic source windows are sized before inference. The model router adds mandatory
repository policy and forced-function protocol messages after those windows are built.
This installation makes admission account for those exact messages with the same
canonical byte counter used by the context safety contract, so an admitted window cannot
later fail only because router-owned mandatory context was appended.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_MARKER = "_mmm_semantic_tool_context_alignment_v1"


def tool_decision_request_messages(
    model_router_module: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_name: str,
) -> tuple[dict[str, Any], ...]:
    """Build the mandatory message envelope used by ModelRouter.generate_tool_decision."""

    name = str(tool_name or "").strip()
    if not name:
        raise ValueError("tool_name must not be empty")
    request_messages = model_router_module._inject_system_context(
        messages,
        model_router_module._REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT,
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


def install(
    *,
    semantic_module: Any,
    model_router_module: Any,
    context_module: Any,
) -> None:
    """Make semantic source-window admission use the final mandatory request envelope."""

    current = semantic_module._semantic_window_fits
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def aligned_semantic_window_fits(
        window: str,
        *,
        requirement_statement: Any,
        obligation: str,
        source_id: Any,
        assessment_budget: int,
        verification_budget: int,
    ) -> bool:
        if not window:
            return False
        units = semantic_module._source_units(window)
        if not units:
            return False

        max_index = len(units) - 1
        assessment_messages = semantic_module._assessment_messages(
            requirement_statement,
            obligation,
            source_id,
            units,
        )
        verification_messages = semantic_module._verification_messages(
            requirement_statement,
            obligation,
            source_id,
            units,
            max_index,
            max_index,
        )

        assessment_request = tool_decision_request_messages(
            model_router_module,
            assessment_messages,
            tool_name=semantic_module._ASSESSMENT_TOOL_NAME,
        )
        verification_request = tool_decision_request_messages(
            model_router_module,
            verification_messages,
            tool_name=semantic_module._VERIFICATION_TOOL_NAME,
        )

        return (
            context_module._canonical_size(assessment_request)
            <= int(assessment_budget)
            and context_module._canonical_size(verification_request)
            <= int(verification_budget)
        )

    setattr(aligned_semantic_window_fits, _MARKER, True)
    aligned_semantic_window_fits.__wrapped__ = current  # type: ignore[attr-defined]
    semantic_module._semantic_window_fits = aligned_semantic_window_fits


__all__ = ["install", "tool_decision_request_messages"]
