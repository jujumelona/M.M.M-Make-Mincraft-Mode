from __future__ import annotations

"""Host-owned state/precondition/effect causal planning for model tool exposure."""

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .tool_transition_registry import opaque_transition, reviewed_transition


@dataclass(frozen=True)
class ToolTransition:
    name: str
    preconditions: frozenset[str]
    effects: frozenset[str]
    cost: int = 1
    reviewed: bool = True


_GOAL_REQUIREMENTS: dict[str, frozenset[str]] = {
    "observe": frozenset({"project_observed"}),
    "evidence": frozenset({"evidence_ready"}),
    "verify": frozenset({"verified"}),
    "act": frozenset({"project_changed"}),
    "runtime": frozenset({"runtime_observed"}),
    "runtime_verify": frozenset({"runtime_verified"}),
    "external": frozenset({"external_observation"}),
    "plan": frozenset({"plan_ready"}),
    "release": frozenset({"packaged"}),
}
_EXTERNAL_NAMES = frozenset({
    "external_mcp_capabilities",
    "external_mcp_schema",
    "external_mcp_call",
})


def _tool_name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def transition_for_schema(schema: Mapping[str, Any]) -> ToolTransition:
    name = _tool_name(schema)
    spec = reviewed_transition(name)
    reviewed = spec is not None
    if spec is None:
        spec = opaque_transition(name)
    return ToolTransition(
        name=name,
        preconditions=spec.preconditions,
        effects=spec.effects,
        cost=spec.cost,
        reviewed=reviewed,
    )


def _payload_ok(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, Mapping):
        return content.get("ok") is True
    if not isinstance(content, str) or not content.strip():
        return False
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(value, Mapping) and value.get("ok") is True


def verified_state_from_messages(
    messages: Sequence[Mapping[str, Any]],
    schemas: Sequence[Mapping[str, Any]],
    *,
    require_fresh_evidence: bool = False,
) -> frozenset[str]:
    transitions = {
        transition.name: transition
        for schema in schemas
        if (transition := transition_for_schema(schema)).name
    }
    state: set[str] = {"workspace_bound"}
    for message in messages:
        if str(message.get("role", "")).casefold() != "tool" or not _payload_ok(message):
            continue
        name = str(message.get("name", "")).strip()
        transition = transitions.get(name)
        if transition is None or not transition.reviewed:
            continue
        if transition.preconditions.issubset(state):
            state.update(transition.effects)
    if require_fresh_evidence and not any(
        fact in state for fact in ("code_evidence", "project_evidence", "ecosystem_evidence", "external_observation")
    ):
        state.discard("evidence_ready")
    return frozenset(state)


def infer_verified_state(
    *,
    query: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    require_fresh_evidence: bool,
) -> frozenset[str]:
    del query, tool_schemas, require_fresh_evidence
    return frozenset({"workspace_bound"})


def _goal_facts(goals: Iterable[str]) -> frozenset[str]:
    facts: set[str] = set()
    for goal in goals:
        facts.update(_GOAL_REQUIREMENTS.get(goal, ()))
    return frozenset(facts or {"project_observed"})


def shortest_causal_path(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    max_depth: int = 6,
) -> tuple[str, ...]:
    transitions = {
        transition.name: transition
        for schema in schemas
        if (transition := transition_for_schema(schema)).name and transition.reviewed
    }
    target = _goal_facts(goals)
    if target.issubset(state):
        return ()
    queue: deque[tuple[frozenset[str], tuple[str, ...], int]] = deque([(state, (), 0)])
    best_cost: dict[frozenset[str], int] = {state: 0}
    solutions: list[tuple[int, int, tuple[str, ...]]] = []
    while queue:
        current_state, path, cost = queue.popleft()
        if len(path) >= max_depth:
            continue
        for name in sorted(transitions):
            transition = transitions[name]
            if name in path or not transition.preconditions.issubset(current_state):
                continue
            if not (set(transition.effects) - set(current_state)):
                continue
            next_state = frozenset(set(current_state) | set(transition.effects))
            next_path = path + (name,)
            next_cost = cost + transition.cost
            if target.issubset(next_state):
                solutions.append((next_cost, len(next_path), next_path))
                continue
            if next_cost < best_cost.get(next_state, 1 << 30):
                best_cost[next_state] = next_cost
                queue.append((next_state, next_path, next_cost))
    if not solutions:
        return ()
    solutions.sort(key=lambda item: (item[0], item[1], item[2]))
    return solutions[0][2]


def executable_frontier(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    limit: int = 3,
    max_depth: int = 6,
) -> tuple[str, ...]:
    limit = max(1, min(int(limit), 3))
    transitions = {
        transition_for_schema(schema).name: transition_for_schema(schema)
        for schema in schemas
        if transition_for_schema(schema).name
    }
    path = shortest_causal_path(schemas, state=state, goals=goals, max_depth=max_depth)
    # If the authorized surface is external-MCP-only, a generic "inspect" request
    # means inspect through that external surface rather than exposing nothing.
    if not path and transitions and set(transitions) <= _EXTERNAL_NAMES:
        path = shortest_causal_path(schemas, state=state, goals=("external",), max_depth=max_depth)
    if not path:
        return ()
    first = transitions[path[0]]
    frontier: list[str] = [first.name]

    if first.effects & {"evidence_ready", "project_evidence", "code_evidence", "ecosystem_evidence"}:
        alternatives: list[tuple[int, str]] = []
        for name, transition in transitions.items():
            if (
                name != first.name
                and transition.reviewed
                and transition.preconditions.issubset(state)
                and transition.effects & {"evidence_ready", "project_evidence", "code_evidence", "ecosystem_evidence"}
            ):
                alternatives.append((transition.cost, name))
        alternatives.sort()
        frontier.extend(name for _cost, name in alternatives[: limit - 1])

    # External capabilities/schema/call are three read-only alternatives, not a fake
    # mandatory chain. The bridge itself says schema is needed only when arguments are
    # not already known. Keep all executable choices available, still bounded to 3.
    if first.name in _EXTERNAL_NAMES:
        external_alternatives = sorted(
            name
            for name, transition in transitions.items()
            if name in _EXTERNAL_NAMES
            and name not in frontier
            and transition.reviewed
            and transition.preconditions.issubset(state)
        )
        frontier.extend(external_alternatives[: limit - len(frontier)])
    return tuple(frontier[:limit])


def shortest_causal_frontier(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    protected: Iterable[str] = (),
    max_depth: int = 4,
) -> tuple[str, ...]:
    del protected
    return executable_frontier(schemas, state=state, goals=goals, limit=3, max_depth=max_depth)


__all__ = [
    "ToolTransition",
    "executable_frontier",
    "infer_verified_state",
    "shortest_causal_frontier",
    "shortest_causal_path",
    "transition_for_schema",
    "verified_state_from_messages",
]
