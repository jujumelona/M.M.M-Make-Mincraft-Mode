from __future__ import annotations

"""Host-owned causal tool planning for small-model tool exposure.

The graph is deliberately deterministic: tools declare preconditions/effects,
current verified state is inferred from the active request, and breadth-first
search selects only actions that can advance the state toward the goal.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ToolTransition:
    name: str
    preconditions: frozenset[str]
    effects: frozenset[str]
    cost: int = 1


_DEFAULT_TRANSITIONS: dict[str, ToolTransition] = {
    "inspect_existing_mod": ToolTransition(
        "inspect_existing_mod", frozenset({"workspace_bound"}), frozenset({"project_observed"})
    ),
    "search_project_rag": ToolTransition(
        "search_project_rag", frozenset({"workspace_bound"}), frozenset({"project_evidence"})
    ),
    "search_code_rag": ToolTransition(
        "search_code_rag", frozenset({"workspace_bound"}), frozenset({"code_evidence"})
    ),
    "external_mcp_capabilities": ToolTransition(
        "external_mcp_capabilities", frozenset({"workspace_bound"}), frozenset({"external_capabilities"})
    ),
    "external_mcp_schema": ToolTransition(
        "external_mcp_schema", frozenset({"external_capabilities"}), frozenset({"external_schema"})
    ),
    "external_mcp_call": ToolTransition(
        "external_mcp_call", frozenset({"external_schema"}), frozenset({"external_observation"})
    ),
}


_GOAL_REQUIREMENTS: dict[str, frozenset[str]] = {
    "observe": frozenset({"project_observed"}),
    "evidence": frozenset({"project_evidence", "code_evidence"}),
    "verify": frozenset({"code_evidence"}),
    "act": frozenset({"project_observed", "code_evidence"}),
    "runtime": frozenset({"project_observed"}),
    "external": frozenset({"external_schema"}),
}


def _tool_name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def infer_verified_state(
    *,
    query: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    require_fresh_evidence: bool,
) -> frozenset[str]:
    state = {"workspace_bound"}
    value = query.casefold()
    if not require_fresh_evidence and any(marker in value for marker in ("existing", "current", "기존", "현재")):
        state.add("project_observed")
    names = {_tool_name(schema) for schema in tool_schemas}
    if "external_mcp_schema" not in names:
        state.add("external_schema")
    return frozenset(state)


def transition_for_schema(schema: Mapping[str, Any]) -> ToolTransition:
    name = _tool_name(schema)
    if name in _DEFAULT_TRANSITIONS:
        return _DEFAULT_TRANSITIONS[name]
    fn = schema.get("function")
    description = str(fn.get("description", "")) if isinstance(fn, Mapping) else ""
    text = (name + " " + description).casefold()
    if any(token in text for token in ("search", "discover", "evidence", "lookup")):
        return ToolTransition(name, frozenset({"workspace_bound"}), frozenset({"generic_evidence"}), 1)
    if any(token in text for token in ("inspect", "read", "status", "logs", "symbols")):
        return ToolTransition(name, frozenset({"workspace_bound"}), frozenset({"generic_observation"}), 1)
    if any(token in text for token in ("diagnostic", "validate", "test", "verify", "smoke")):
        return ToolTransition(name, frozenset({"code_evidence"}), frozenset({"verified"}), 2)
    if any(token in text for token in ("patch", "write", "apply", "generate", "execute", "command")):
        return ToolTransition(name, frozenset({"project_observed"}), frozenset({"project_changed"}), 2)
    return ToolTransition(name, frozenset({"workspace_bound"}), frozenset({"generic_observation"}), 2)


def _goal_facts(goals: Iterable[str]) -> frozenset[str]:
    facts: set[str] = set()
    for goal in goals:
        facts.update(_GOAL_REQUIREMENTS.get(goal, ()))
    return frozenset(facts or {"project_observed"})


def shortest_causal_frontier(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    protected: Iterable[str] = (),
    max_depth: int = 4,
) -> tuple[str, ...]:
    transitions = {transition_for_schema(schema).name: transition_for_schema(schema) for schema in schemas}
    target = _goal_facts(goals)
    protected_names = tuple(name for name in protected if name in transitions)
    if target.issubset(state):
        return protected_names

    queue: deque[tuple[frozenset[str], tuple[str, ...]]] = deque([(state, ())])
    visited = {state}
    solutions: list[tuple[int, tuple[str, ...]]] = []
    while queue:
        current_state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for name, transition in transitions.items():
            if name in path or not transition.preconditions.issubset(current_state):
                continue
            next_state = frozenset(set(current_state) | set(transition.effects))
            next_path = path + (name,)
            if target.issubset(next_state):
                solutions.append((sum(transitions[item].cost for item in next_path), next_path))
                continue
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, next_path))
    if not solutions:
        return protected_names
    solutions.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    winner = list(protected_names)
    for name in solutions[0][1]:
        if name not in winner:
            winner.append(name)
    return tuple(winner)


__all__ = [
    "ToolTransition",
    "infer_verified_state",
    "shortest_causal_frontier",
    "transition_for_schema",
]
