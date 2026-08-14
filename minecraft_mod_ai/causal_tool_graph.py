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
    "repair": frozenset({"repaired"}),
    "generate": frozenset({"generated"}),
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
        fact in state
        for fact in (
            "code_evidence",
            "project_evidence",
            "ecosystem_evidence",
            "external_observation",
        )
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


def _transitions(schemas: Sequence[Mapping[str, Any]]) -> dict[str, ToolTransition]:
    result: dict[str, ToolTransition] = {}
    for schema in schemas:
        transition = transition_for_schema(schema)
        if transition.name and transition.reviewed:
            result[transition.name] = transition
    return result


def shortest_causal_path(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    max_depth: int = 6,
) -> tuple[str, ...]:
    transitions = _transitions(schemas)
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


def _path_cost(path: Sequence[str], transitions: Mapping[str, ToolTransition]) -> int:
    return sum(transitions[name].cost for name in path if name in transitions)


def executable_frontier(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    limit: int = 3,
    max_depth: int = 6,
    preference: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Return up to three first edges on globally minimum-cost causal paths.

    Causal legality and total path cost are authoritative. ``preference`` is used
    only after all first transitions with greater total causal cost have been
    discarded, so semantic/tool relevance can never shortcut a precondition.
    """

    limit = max(1, min(int(limit), 3))
    transitions = _transitions(schemas)
    if not transitions:
        return ()
    target_goals = tuple(goals)
    target = _goal_facts(target_goals)
    if target.issubset(state):
        return ()

    candidates: list[tuple[int, int, str]] = []
    for name, transition in transitions.items():
        if not transition.preconditions.issubset(state):
            continue
        new_facts = set(transition.effects) - set(state)
        if not new_facts:
            continue
        next_state = frozenset(set(state) | set(transition.effects))
        if target.issubset(next_state):
            total_cost = transition.cost
            total_steps = 1
        else:
            tail = shortest_causal_path(
                schemas,
                state=next_state,
                goals=target_goals,
                max_depth=max(0, max_depth - 1),
            )
            if not tail:
                continue
            total_cost = transition.cost + _path_cost(tail, transitions)
            total_steps = 1 + len(tail)
        candidates.append((total_cost, total_steps, name))

    # If the entire authorized surface is external-MCP-only, a generic inspect
    # request means inspect through that external surface rather than expose nothing.
    if not candidates and set(transitions) <= _EXTERNAL_NAMES:
        return executable_frontier(
            schemas,
            state=state,
            goals=("external",),
            limit=limit,
            max_depth=max_depth,
            preference=preference,
        )
    if not candidates:
        return ()

    min_cost = min(item[0] for item in candidates)
    min_steps = min(item[1] for item in candidates if item[0] == min_cost)
    equally_minimal = [
        name
        for cost, steps, name in candidates
        if cost == min_cost and steps == min_steps
    ]
    rank = preference or {}
    fallback_rank = len(rank) + len(equally_minimal) + 1
    equally_minimal.sort(
        key=lambda name: (
            int(rank.get(name, fallback_rank)),
            transitions[name].cost,
            name,
        )
    )
    return tuple(equally_minimal[:limit])


def shortest_causal_frontier(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    protected: Iterable[str] = (),
    max_depth: int = 4,
) -> tuple[str, ...]:
    del protected
    return executable_frontier(
        schemas,
        state=state,
        goals=goals,
        limit=3,
        max_depth=max_depth,
    )


__all__ = [
    "ToolTransition",
    "executable_frontier",
    "infer_verified_state",
    "shortest_causal_frontier",
    "shortest_causal_path",
    "transition_for_schema",
    "verified_state_from_messages",
]
