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
_GOAL_SUPPORT: dict[str, frozenset[str]] = {
    "observe": frozenset({"project_observed", "work_observed", "quality_observed"}),
    "evidence": frozenset({"evidence_ready", "code_evidence", "project_evidence", "ecosystem_evidence", "external_observation"}),
    "verify": frozenset({"project_observed", "evidence_ready", "static_verified", "build_verified", "verified"}),
    "repair": frozenset({"project_observed", "evidence_ready", "code_evidence", "project_evidence", "repaired"}),
    "generate": frozenset({"project_observed", "evidence_ready", "code_evidence", "project_evidence", "generated"}),
    "act": frozenset({"project_observed", "evidence_ready", "code_evidence", "project_evidence", "project_changed"}),
    "runtime": frozenset({"project_observed", "build_verified", "runtime_prepared", "server_started", "client_started", "mineflayer_connected", "runtime_observed"}),
    "runtime_verify": frozenset({"project_observed", "build_verified", "runtime_prepared", "server_started", "client_started", "runtime_observed", "runtime_verified"}),
    "external": frozenset({"external_capabilities", "external_schema", "external_observation", "evidence_ready"}),
    "plan": frozenset({"plan_ready", "plan_observed", "plan_approved"}),
    "release": frozenset({"project_observed", "build_verified", "artifact_observed", "packaged"}),
}
_EXTERNAL_NAMES = frozenset({
    "external_mcp_capabilities",
    "external_mcp_schema",
    "external_mcp_call",
})
_RAG_NAMES = frozenset({"search_code_rag", "search_project_rag"})
_GRADLE_VERIFY_NAMES = frozenset({"run_gradle_build", "gradle_build", "run_gametest"})
_DIAGNOSTIC_VERIFY_NAMES = frozenset({"java_diagnostics", "jdt_diagnostics"})
_SEMANTIC_FAILURE_STATUSES = frozenset({
    "FAIL",
    "FAILED",
    "ERROR",
    "UNAVAILABLE",
    "PARTIAL",
    "BLOCKED",
    "INVALID",
    "REJECTED",
    "CANCELLED",
    "CANCELED",
    "TIMEOUT",
})
_CERTIFICATION_EFFECTS = frozenset({
    "project_changed",
    "repaired",
    "generated",
    "source_generated",
    "assets_generated",
    "static_verified",
    "build_verified",
    "gametest_verified",
    "test_verified",
    "geometry_verified",
    "benchmark_verified",
    "verified",
    "quality_verified",
    "runtime_verified",
    "packaged",
    "external_schema",
    "external_observation",
    "evidence_ready",
    "runtime_prepared",
    "server_started",
    "client_started",
    "mineflayer_connected",
    "model_ready",
})
_SAFE_PROGRESS_EFFECTS = frozenset({
    "project_observed",
    "work_observed",
    "quality_observed",
    "capabilities_observed",
    "code_evidence",
    "project_evidence",
    "ecosystem_evidence",
    "evidence_ready",
    "rag_index_ready",
    "plan_observed",
    "quality_contract",
    "static_verified",
    "build_verified",
    "verified",
    "geometry_verified",
    "benchmark_verified",
    "runtime_prepared",
    "server_started",
    "client_started",
    "mineflayer_connected",
    "runtime_observed",
    "runtime_verified",
    "external_capabilities",
    "external_schema",
    "external_observation",
    "artifact_observed",
    "model_observed",
    "model_ready",
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


def _payload(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    content = message.get("content")
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _result_mappings(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Walk normalized tool results without assuming one transport envelope shape."""

    root: Any = payload.get("result", payload)
    pending: list[Any] = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            yield current
            pending.extend(current.values())
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            pending.extend(current)


def _primary_result_mappings(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return only transport-envelope result roots used for semantic status checks.

    Do not recursively inspect arbitrary evidence rows here: an inner provider attempt
    may legitimately be ERROR while the enclosing corroborated bundle is PASS.
    """

    root = payload.get("result", payload)
    if not isinstance(root, Mapping):
        return ()
    values: list[Mapping[str, Any]] = [root]
    for key in ("structured_content", "structured", "parsed_text"):
        nested = root.get(key)
        if isinstance(nested, Mapping):
            values.append(nested)
    return tuple(values)


def _has_explicit_semantic_failure(payload: Mapping[str, Any]) -> bool:
    for result in _primary_result_mappings(payload):
        status = str(result.get("status", "")).strip().upper()
        if status in _SEMANTIC_FAILURE_STATUSES:
            return True
    return False


def _rag_receipt_ready(payload: Mapping[str, Any]) -> bool:
    """Certify terminal RAG evidence only from objective retrieval-quality fields."""

    for result in _result_mappings(payload):
        receipt = result.get("receipt")
        if not isinstance(receipt, Mapping):
            continue
        try:
            count = int(receipt.get("result_count", 0) or 0)
            coverage = float(receipt.get("coverage_score", 0.0) or 0.0)
            relevance = float(receipt.get("relevance_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if count > 0 and coverage >= 0.50 and relevance > 0.0:
            return True
    return False


def _external_evidence_ready(payload: Mapping[str, Any]) -> bool:
    """Accept external evidence only from a satisfied federation bundle."""

    for result in _result_mappings(payload):
        if str(result.get("schema_version", "")) != "mmm/external-mcp-evidence-bundle-v1":
            continue
        if str(result.get("status", "")).strip().upper() != "PASS":
            return False
        evidence = result.get("evidence")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes, bytearray)):
            return False
        try:
            required = max(1, int(result.get("required_corroboration", 1) or 1))
        except (TypeError, ValueError):
            return False
        return len(evidence) >= required
    return False


def _gradle_build_passed(payload: Mapping[str, Any]) -> bool:
    """Separate a successful MCP transport from a successful Gradle process."""

    for result in _result_mappings(payload):
        if "status" not in result or not ({"command", "returncode"} & set(result)):
            continue
        if str(result.get("status", "")).strip().upper() != "PASS":
            return False
        if "returncode" in result:
            try:
                if int(result.get("returncode", -1)) != 0:
                    return False
            except (TypeError, ValueError):
                return False
        return True
    return False


def _diagnostics_clean(payload: Mapping[str, Any]) -> bool:
    """Treat a diagnostics transport as verification only when it reports zero errors."""

    for result in _result_mappings(payload):
        schema_version = str(result.get("schema_version", ""))
        if not schema_version.startswith("mmm/java-diagnostics-"):
            continue
        try:
            return int(result.get("error_count", -1)) == 0
        except (TypeError, ValueError):
            return False
    return False


def _semantic_effects(
    name: str,
    payload: Mapping[str, Any],
    effects: set[str],
) -> set[str]:
    """Gate state-changing facts on tool-specific semantic success.

    ``payload['ok']`` certifies only that the host call returned without raising. It
    must never by itself turn a FAIL/UNAVAILABLE result into verified evidence.
    """

    if _has_explicit_semantic_failure(payload):
        effects.difference_update(_CERTIFICATION_EFFECTS)

    if name in _RAG_NAMES and "evidence_ready" in effects:
        if not _rag_receipt_ready(payload):
            effects.discard("evidence_ready")
    if name == "external_mcp_call":
        if not _external_evidence_ready(payload):
            effects.discard("external_observation")
            effects.discard("evidence_ready")
    if name in _GRADLE_VERIFY_NAMES:
        if not _gradle_build_passed(payload):
            effects.discard("build_verified")
            effects.discard("gametest_verified")
            effects.discard("verified")
    if name in _DIAGNOSTIC_VERIFY_NAMES:
        if not _diagnostics_clean(payload):
            effects.discard("static_verified")
            effects.discard("verified")
    return effects


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
        if str(message.get("role", "")).casefold() != "tool":
            continue
        payload = _payload(message)
        if payload is None or payload.get("ok") is not True:
            continue
        name = str(message.get("name", "")).strip()
        transition = transitions.get(name)
        if transition is None or not transition.reviewed:
            continue
        if not transition.preconditions.issubset(state):
            continue
        effects = _semantic_effects(name, payload, set(transition.effects))
        state.update(effects)
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


def _support_facts(goals: Iterable[str]) -> frozenset[str]:
    facts: set[str] = set()
    for goal in goals:
        facts.update(_GOAL_SUPPORT.get(goal, ()))
    return frozenset(facts)


def _transitions(schemas: Sequence[Mapping[str, Any]]) -> dict[str, ToolTransition]:
    result: dict[str, ToolTransition] = {}
    for schema in schemas:
        transition = transition_for_schema(schema)
        if transition.name and transition.reviewed:
            result[transition.name] = transition
    return result


def _shortest_causal_path_from_transitions(
    transitions: Mapping[str, ToolTransition],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    max_depth: int,
) -> tuple[str, ...]:
    target = _goal_facts(goals)
    if target.issubset(state):
        return ()
    ordered_transitions = tuple(sorted(transitions.items()))
    queue: deque[tuple[frozenset[str], tuple[str, ...], int]] = deque([(state, (), 0)])
    best_cost: dict[frozenset[str], int] = {state: 0}
    solutions: list[tuple[int, int, tuple[str, ...]]] = []
    while queue:
        current_state, path, cost = queue.popleft()
        if len(path) >= max_depth:
            continue
        for name, transition in ordered_transitions:
            if name in path or not transition.preconditions.issubset(current_state):
                continue
            if not transition.effects.difference(current_state):
                continue
            next_state = current_state | transition.effects
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


def shortest_causal_path(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    max_depth: int = 6,
) -> tuple[str, ...]:
    return _shortest_causal_path_from_transitions(
        _transitions(schemas),
        state=state,
        goals=goals,
        max_depth=max_depth,
    )


def _minimal_causal_first_steps_from_transitions(
    transitions: Mapping[str, ToolTransition],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    max_depth: int,
) -> tuple[str, ...]:
    """Return all first tools on globally minimal causal paths in one traversal.

    The previous frontier implementation launched a fresh shortest-path search for
    every executable first tool. This search carries the equally optimal first-step
    set through each best state label, so one state-space traversal is sufficient.
    """

    target = _goal_facts(goals)
    if target.issubset(state):
        return ()

    depth_limit = max(1, int(max_depth))
    ordered_transitions = tuple(sorted(transitions.items()))
    queue: deque[tuple[frozenset[str], int, int]] = deque([(state, 0, 0)])
    best_label: dict[frozenset[str], tuple[int, int]] = {state: (0, 0)}
    first_steps: dict[frozenset[str], frozenset[str]] = {state: frozenset()}
    best_goal_label: tuple[int, int] | None = None
    best_goal_first_steps: set[str] = set()

    while queue:
        current_state, cost, steps = queue.popleft()
        current_label = (cost, steps)
        if best_label.get(current_state) != current_label:
            continue
        if steps >= depth_limit:
            continue

        current_first_steps = first_steps[current_state]
        for name, transition in ordered_transitions:
            if not transition.preconditions.issubset(current_state):
                continue
            if not transition.effects.difference(current_state):
                continue

            next_state = current_state | transition.effects
            next_cost = cost + transition.cost
            next_steps = steps + 1
            next_label = (next_cost, next_steps)
            candidate_first_steps = (
                frozenset({name}) if steps == 0 else current_first_steps
            )

            if target.issubset(next_state):
                if best_goal_label is None or next_label < best_goal_label:
                    best_goal_label = next_label
                    best_goal_first_steps = set(candidate_first_steps)
                elif next_label == best_goal_label:
                    best_goal_first_steps.update(candidate_first_steps)
                continue

            previous_label = best_label.get(next_state)
            if previous_label is None or next_label < previous_label:
                best_label[next_state] = next_label
                first_steps[next_state] = candidate_first_steps
                queue.append((next_state, next_cost, next_steps))
                continue
            if next_label != previous_label:
                continue

            merged_first_steps = first_steps[next_state] | candidate_first_steps
            if merged_first_steps != first_steps[next_state]:
                first_steps[next_state] = merged_first_steps
                queue.append((next_state, next_cost, next_steps))

    return tuple(sorted(best_goal_first_steps))


def _progress_frontier(
    transitions: Mapping[str, ToolTransition],
    *,
    state: frozenset[str],
    goals: Sequence[str],
    preference: Mapping[str, int] | None,
    limit: int,
) -> tuple[str, ...]:
    support = _support_facts(goals).difference(state)
    rank = preference or {}
    fallback_rank = len(rank) + len(transitions) + 1
    scored: list[tuple[int, int, int, str]] = []
    for name, transition in transitions.items():
        if not transition.preconditions.issubset(state):
            continue
        new_effects = transition.effects.difference(state)
        safe_effects = new_effects.intersection(_SAFE_PROGRESS_EFFECTS)
        if not safe_effects:
            continue
        direct_support = len(safe_effects.intersection(support))
        scored.append(
            (
                -direct_support,
                transition.cost,
                int(rank.get(name, fallback_rank)),
                name,
            )
        )
    scored.sort()
    if not scored:
        return ()
    best_support = scored[0][0]
    best_cost = scored[0][1]
    selected = [
        name
        for support_score, cost, _rank, name in scored
        if support_score == best_support and cost == best_cost
    ]
    return tuple(selected[:limit])


def executable_frontier(
    schemas: Sequence[Mapping[str, Any]],
    *,
    state: frozenset[str],
    goals: Iterable[str],
    limit: int = 3,
    max_depth: int = 6,
    preference: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    limit = max(1, min(int(limit), 3))
    transitions = _transitions(schemas)
    if not transitions:
        return ()
    target_goals = tuple(goals)
    target = _goal_facts(target_goals)
    if target.issubset(state):
        return ()

    equally_minimal = list(
        _minimal_causal_first_steps_from_transitions(
            transitions,
            state=state,
            goals=target_goals,
            max_depth=max_depth,
        )
    )

    if (
        not equally_minimal
        and set(transitions) <= _EXTERNAL_NAMES
        and target_goals != ("external",)
    ):
        return executable_frontier(
            schemas,
            state=state,
            goals=("external",),
            limit=limit,
            max_depth=max_depth,
            preference=preference,
        )
    if not equally_minimal:
        return _progress_frontier(
            transitions,
            state=state,
            goals=target_goals,
            preference=preference,
            limit=limit,
        )

    support = _support_facts(target_goals).difference(state)
    rank = preference or {}
    fallback_rank = len(rank) + len(equally_minimal) + 1
    equally_minimal.sort(
        key=lambda name: (
            -len(transitions[name].effects.difference(state).intersection(support)),
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
