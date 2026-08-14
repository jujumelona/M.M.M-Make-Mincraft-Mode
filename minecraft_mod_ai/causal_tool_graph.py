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


def _tool_name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def transition_for_schema(schema: Mapping[str, Any]) -> ToolTransition:
    """Return reviewed causal semantics; never infer them from descriptions."""

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
    """Replay successful tool observations into an externally verified state.

    State advances only from host tool observations with ``ok=true``.  Model text,
    intent words and assistant self-reports never satisfy a precondition.
    """

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
        # A successful tool call can only certify its effects if the host-known
        # preconditions were already certified at that point in the trace.
        if transition.preconditions.issubset(state):
            state.update(transition.effects)

    # Fresh-evidence requests deliberately never inherit evidence merely because an
    # earlier user/system sentence said it existed.  Only successful tool receipts
    # above can add evidence_ready/code_evidence/project_evidence.
    if require_fresh_evidence and not any(
        fact in state for fact in ("code_evidence", "project_evidence", "ecosystem_evidence")
    ):
        state.discard("evidence_ready")
    return frozenset(state)


def infer_verified_state(
    *,
    query: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    require_fresh_evidence: bool,
) -> frozenset[str]:
    """Compatibility helper for pre-loop selection.

    Query text never certifies state.  The live tool loop uses
    :func:`verified_state_from_messages` and therefore advances after each host
    observation.
    """

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
    """Find the minimum-cost reviewed transition path from state to goal facts."""

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
            new_effects = set(transition.effects) - set(current_state)
            if not new_effects:
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
    """Expose only the first 1-3 executable edges that can advance a goal.

    Multiple first edges are returned only when they are equally minimal alternatives
    or jointly useful evidence reads.  Downstream edges remain hidden until their
    preconditions become verified by actual tool observations.
    """

    limit = max(1, min(int(limit), 3))
    path = shortest_causal_path(schemas, state=state, goals=goals, max_depth=max_depth)
    if not path:
        return ()
    transitions = {transition_for_schema(schema).name: transition_for_schema(schema) for schema in schemas}
    first = transitions[path[0]]
    frontier: list[str] = [first.name]

    # Evidence acquisition is a safe parallel-read opportunity. If the goal requires
    # a later mutation/verification, expose up to two other executable reviewed
    # evidence transitions at the same frontier so the model can gather independent
    # evidence in one wave without exposing downstream write tools.
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
    return tuple(frontier[:limit])


# Backward-compatible name used by older focused tests/contracts.
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
