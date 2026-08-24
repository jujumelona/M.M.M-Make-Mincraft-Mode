from __future__ import annotations

"""Canonical host action classes for causal agent execution.

The model may generate arguments inside the selected action, but it never owns the
transition from one action class to another. The ordered causal frontier determines the
class first. A mutation class deterministically collapses to exactly one host-selected
mutation tool so a small model is never asked to choose a writable tool name.
"""

from enum import Enum
from typing import Iterable, Sequence

from .source_mutation_contract import SOURCE_MUTATION_NAMES


class CausalAction(str, Enum):
    RETRIEVE = "RETRIEVE"
    INSPECT = "INSPECT"
    MUTATE = "MUTATE"
    VERIFY = "VERIFY"
    FINISH = "FINISH"


_RETRIEVAL_TOOLS = frozenset(
    {
        "search_code_rag",
        "search_project_rag",
        "external_mcp_call",
        "external_mcp_capabilities",
        "external_mcp_schema",
    }
)
_VERIFY_TOOLS = frozenset(
    {
        "java_diagnostics",
        "jdt_diagnostics",
        "run_gradle_build",
        "gradle_build",
        "run_gametest",
        "verify_project",
    }
)


def action_for_tool(name: str) -> CausalAction:
    normalized = str(name).strip()
    if not normalized:
        return CausalAction.FINISH
    if normalized in SOURCE_MUTATION_NAMES:
        return CausalAction.MUTATE
    if normalized in _VERIFY_TOOLS:
        return CausalAction.VERIFY
    if normalized in _RETRIEVAL_TOOLS:
        return CausalAction.RETRIEVE
    return CausalAction.INSPECT


def select_action_frontier(tool_names: Sequence[str]) -> tuple[CausalAction, tuple[str, ...]]:
    """Choose one action class from an ordered executable frontier.

    ``executable_frontier`` already returns host-ranked transitions. Its first entry is
    therefore the deterministic class decision. Other entries survive only when they
    belong to that same class. MUTATE is stricter: only the first writable transition
    survives, making the tool name host-owned before model generation begins.
    """

    ordered = tuple(str(name).strip() for name in tool_names if str(name).strip())
    if not ordered:
        return CausalAction.FINISH, ()
    action = action_for_tool(ordered[0])
    same_class = tuple(name for name in ordered if action_for_tool(name) is action)
    if action is CausalAction.MUTATE:
        return action, same_class[:1]
    return action, same_class


def classify_action(tool_names: Iterable[str]) -> CausalAction:
    """Classify an already-published frontier without consulting model prose."""

    ordered = tuple(str(name).strip() for name in tool_names if str(name).strip())
    action, _ = select_action_frontier(ordered)
    return action


__all__ = [
    "CausalAction",
    "action_for_tool",
    "classify_action",
    "select_action_frontier",
]
