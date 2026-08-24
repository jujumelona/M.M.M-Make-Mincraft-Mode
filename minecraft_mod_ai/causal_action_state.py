from __future__ import annotations

"""Canonical host action classes for causal agent execution.

The model may propose arguments for an action, but it does not own the transition from
one action class to another.  Frontier/ledger state determines that class first.
"""

from enum import Enum
from typing import Iterable

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


def classify_action(tool_names: Iterable[str]) -> CausalAction:
    """Classify one published executable frontier without consulting model prose."""

    names = frozenset(str(name).strip() for name in tool_names if str(name).strip())
    if not names:
        return CausalAction.FINISH
    if names and names <= frozenset(SOURCE_MUTATION_NAMES):
        return CausalAction.MUTATE
    if names & _VERIFY_TOOLS:
        return CausalAction.VERIFY
    if names & _RETRIEVAL_TOOLS:
        return CausalAction.RETRIEVE
    return CausalAction.INSPECT


__all__ = ["CausalAction", "classify_action"]
