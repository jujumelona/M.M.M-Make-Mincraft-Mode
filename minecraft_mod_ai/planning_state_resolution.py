from __future__ import annotations

"""Compile player-visible requirements without model-authored identity contracts.

The planner model describes behavior. Host code owns bookkeeping. Model output is never
required to reproduce prompt IDs, evidence IDs, hashes, receipts, or any other internal
identifier, so a harmless metadata mismatch cannot abort planning.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .minecraft_template_catalog import (
    CUSTOM_CAPABILITY_SENTINEL,
    capability_catalog_for_model,
    semantic_capability_choices,
)
from .planner_operation import planner_operation
from .planning_state_contract import ROUTE_SOURCES
from .root_cause_trace import emit_root_cause

_REQUIREMENT_TOOL = "submit_researched_requirements"
_SEMANTIC_CAPABILITY_CHOICES = semantic_capability_choices()
_SEMANTIC_CAPABILITY_CHOICE_SET = frozenset(_SEMANTIC_CAPABILITY_CHOICES)

# Deliberately permissive. The model is not an authority for host identity/provenance and
# malformed optional fields are normalized below instead of becoming planner exceptions.
_REQUIREMENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "semantic_capability": {"type": "string"},
                    "acceptance": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
        }
    },
    "additionalProperties": True,
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(dict.fromkeys(text for item in value if (text := _text(item))))


def _resolved_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Give the model semantic facts, never host receipts or identity bookkeeping."""
    known = [
        {"statement": _text(item.get("statement"))}
        for item in state.get("known", [])
        if isinstance(item, Mapping) and _text(item.get("statement"))
    ] if isinstance(state.get("known"), list) else []
    goal = state.get("goal")
    goal_statement = _text(goal.get("statement")) if isinstance(goal, Mapping) else ""
    resolved = [
        {"resolution": _text(item.get("resolution"))}
        for item in state.get("resolved", [])
        if isinstance(item, Mapping) and _text(item.get("resolution"))
    ] if isinstance(state.get("resolved"), list) else []
    evidence_claims: list[str] = []
    for item in state.get("evidence", []) if isinstance(state.get("evidence"), list) else []:
        if not isinstance(item, Mapping):
            continue
        claims = item.get("claims")
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, Mapping):
                    text = _text(claim.get("claim") or claim.get("statement") or claim.get("text"))
                else:
                    text = _text(claim)
                if text:
                    evidence_claims.append(text)
    return {
        "goal": goal_statement,
        "known": known,
        "resolved": resolved,
        "research_claims": list(dict.fromkeys(evidence_claims)),
    }


def _next_id(items: Sequence[Mapping[str, Any]], key: str, prefix: str) -> str:
    maximum = 0
    for item in items:
        raw = str(item.get(key) or "")
        if not raw.startswith(prefix):
            continue
        try:
            maximum = max(maximum, int(raw.removeprefix(prefix)))
        except ValueError:
            continue
    return f"{prefix}{maximum + 1:03d}"


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _fallback_requirement_rows(state: Mapping[str, Any], prompt: str) -> list[dict[str, Any]]:
    """Deterministically preserve authored behavior if model output is absent or unusable."""
    statements: list[str] = []
    rows = state.get("known")
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, Mapping):
                statement = _text(item.get("statement"))
                if statement:
                    statements.append(statement)
    if not statements:
        goal = state.get("goal")
        if isinstance(goal, Mapping):
            statement = _text(goal.get("statement"))
            if statement:
                statements.append(statement)
    if not statements and _text(prompt):
        statements.append(_text(prompt))

    return [
        {
            "statement": statement,
            "semantic_capability": CUSTOM_CAPABILITY_SENTINEL,
            "acceptance": [f"Observe the requested behavior: {statement}"],
        }
        for statement in dict.fromkeys(statements)
    ]


def _normalize_requirement_rows(
    raw: Any,
    state: Mapping[str, Any],
    prompt: str,
) -> list[dict[str, Any]]:
    raw_rows = raw.get("requirements") if isinstance(raw, Mapping) else None
    normalized: list[dict[str, Any]] = []
    if isinstance(raw_rows, list):
        for item in raw_rows:
            if not isinstance(item, Mapping):
                continue
            statement = _text(item.get("statement"))
            if not statement:
                continue
            semantic_capability = _text(item.get("semantic_capability")).casefold()
            if semantic_capability not in _SEMANTIC_CAPABILITY_CHOICE_SET:
                semantic_capability = CUSTOM_CAPABILITY_SENTINEL
            acceptance = _strings(item.get("acceptance"))
            if not acceptance:
                acceptance = [f"Observe the requested behavior: {statement}"]
            normalized.append(
                {
                    "statement": statement,
                    "semantic_capability": semantic_capability,
                    "acceptance": acceptance,
                }
            )

    if not normalized:
        normalized = _fallback_requirement_rows(state, prompt)

    # Duplicate prose is harmless; collapse it instead of raising another planner error.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in normalized:
        key = _text(row.get("statement")).casefold()
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def compile_researched_requirements(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Add behavior requirements; semantic model mistakes never become planner invariants."""
    context = _resolved_context(state)
    messages = [
        {
            "role": "system",
            "content": (
                "Compile independently testable, player-visible requirements from the supplied "
                "task semantics. Do not output or reason about host IDs, evidence IDs, hashes, "
                "receipts, provenance keys, files, classes, registrations, or invented APIs. "
                "Choose a semantic capability from the supplied catalog when it clearly fits; "
                f"otherwise use '{CUSTOM_CAPABILITY_SENTINEL}'. Missing balance values or detailed "
                "mechanics are later design work. Return behavior statements and observable "
                "acceptance conditions only."
            ),
        },
        {
            "role": "user",
            "content": str(
                {
                    "task": context,
                    "semantic_capability_catalog": capability_catalog_for_model(),
                    "custom_capability": CUSTOM_CAPABILITY_SENTINEL,
                }
            ),
        },
    ]

    raw: Any = None
    generation_error: BaseException | None = None
    try:
        with planner_operation("researched_requirement_compile", output_tokens=2048):
            raw = router.generate_tool_decision(
                "planner",
                messages,
                tool_name=_REQUIREMENT_TOOL,
                parameters=_REQUIREMENT_PARAMETERS,
                description="Submit player-visible requirements.",
            )
    except BaseException as exc:
        generation_error = exc
        emit_root_cause(
            "planner_model_decision_failure",
            stage="planning_state",
            operation="researched_requirement_compile",
            result="FALLBACK",
            reason=f"{type(exc).__name__}: {exc}",
            details={
                "messages": messages,
                "tool_name": _REQUIREMENT_TOOL,
                "parameters": _REQUIREMENT_PARAMETERS,
            },
            exc=exc,
        )
    else:
        emit_root_cause(
            "planner_model_decision",
            stage="planning_state",
            operation="researched_requirement_compile",
            result="PASS",
            details={
                "messages": messages,
                "tool_name": _REQUIREMENT_TOOL,
                "parameters": _REQUIREMENT_PARAMETERS,
                "raw_output": raw,
            },
        )

    requirement_rows = _normalize_requirement_rows(raw, state, prompt)
    emit_root_cause(
        "planner_requirement_normalization",
        stage="planning_state",
        operation="researched_requirement_compile",
        result="FALLBACK" if generation_error is not None or raw is None else "PASS",
        details={
            "raw_output": raw,
            "normalized_requirements": requirement_rows,
            "generation_error": (
                f"{type(generation_error).__name__}: {generation_error}"
                if generation_error is not None
                else None
            ),
        },
    )

    value = deepcopy(dict(state))
    value.setdefault("decisions", [])
    value.setdefault("unresolved", [])
    value.setdefault("research_queue", [])
    value.setdefault("blockers", [])
    value["blockers"] = [
        item
        for item in value["blockers"]
        if not (isinstance(item, Mapping) and item.get("stage") == "requirement_selection")
    ]
    value["decisions"] = [
        item
        for item in value["decisions"]
        if not (isinstance(item, Mapping) and item.get("decision_type") == "requirement")
    ]

    implementation_sources = list(ROUTE_SOURCES["implementation_research"])
    for index, row in enumerate(requirement_rows, start=1):
        requirement_id = f"req_{index:03d}"
        requirement = {
            "requirement_id": requirement_id,
            "statement": row["statement"],
            "semantic_capability": row["semantic_capability"],
            "acceptance": row["acceptance"],
            "status": "implementation_research_pending",
        }
        value["decisions"].append(
            {
                "decision_id": f"d_{len(value['decisions']) + 1:03d}",
                "decision_type": "requirement",
                **deepcopy(requirement),
            }
        )
        unresolved_id = _next_id(value["unresolved"], "unresolved_id", "u_")
        research_id = _next_id(value["research_queue"], "research_id", "r_")
        value["unresolved"].append(
            {
                "unresolved_id": unresolved_id,
                "question": f"How can this requirement be implemented correctly: {row['statement']}",
                "reason": "implementation_method",
                "blocks": [],
                "information_needed": (
                    "Reusable implementation patterns, Minecraft API/source behavior, support "
                    f"artifacts, dependencies, and verification guidance for: {row['statement']}"
                ),
                "resolution_route": "implementation_research",
                "source_kinds": implementation_sources,
                "status": "open",
                "research_ref": research_id,
                "requirement_ref": requirement_id,
            }
        )
        value["research_queue"].append(
            {
                "research_id": research_id,
                "resolves": [unresolved_id],
                "requirement_ref": requirement_id,
                "objective": f"Find useful implementation/reuse options for {row['statement']}",
                "information_needed": (
                    "Concrete implementation patterns and support artifacts without inventing APIs."
                ),
                "source_kinds": implementation_sources,
                "queries": [],
                "status": "pending",
            }
        )

    value["plan_ready"] = False
    return _rehash(value)


__all__ = ["compile_researched_requirements"]
