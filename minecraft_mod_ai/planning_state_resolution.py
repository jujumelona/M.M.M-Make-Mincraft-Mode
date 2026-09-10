from __future__ import annotations

"""Compile player-visible requirements without model-authored identity contracts.

The planner model describes behavior. Host code owns bookkeeping. Model output is never
required to reproduce prompt IDs, evidence IDs, hashes, receipts, or any other internal
identifier, so a harmless metadata mismatch cannot abort planning.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

CUSTOM_CAPABILITY_SENTINEL = "custom"

from .planner_operation import planner_operation
from .planning_contract_ssot import SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA
from .planning_state_contract import ROUTE_SOURCES, validate_planning_state
from .root_cause_trace import emit_root_cause

_REQUIREMENT_TOOL = "submit_researched_requirements"
_REQUIREMENT_PARAMETERS: dict[str, Any] = SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(dict.fromkeys(text for item in value if (text := _text(item))))


def _resolution_prose(value: Any) -> str:
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        texts: list[str] = []
        for elem in value:
            if isinstance(elem, Mapping):
                elem_text = _text(
                    elem.get("claim")
                    or elem.get("statement")
                    or elem.get("fact")
                    or elem.get("text")
                )
            else:
                elem_text = _text(elem)
            if elem_text:
                texts.append(elem_text)
        return " ".join(dict.fromkeys(texts))
    return _text(value)


def _planner_context_budget(router: Any) -> int:
    registry = getattr(router, "registry", None)
    resolve = getattr(registry, "role", None)
    profile = str(getattr(router, "profile", "") or "").strip()
    if callable(resolve) and profile:
        try:
            config = resolve(profile, "planner")
            from .model_context_budget import request_message_budget

            return int(request_message_budget(config, (_REQUIREMENT_PARAMETERS,)))
        except Exception:
            pass
    from .model_context_budget import _default_context_bytes

    return int(_default_context_bytes())


def _fit_context_to_budget(
    context: dict[str, Any],
    *,
    byte_budget: int,
    catalog_and_system_bytes: int,
) -> dict[str, Any]:
    """Ensure serialized user task payload strictly respects the active runtime context budget."""
    target_budget = max(4096, byte_budget - catalog_and_system_bytes - 1024)
    fitted = deepcopy(context)

    def _size() -> int:
        return len(
            json.dumps(
                fitted,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    if _size() <= target_budget:
        return fitted

    # Phase 1: Trim research claims
    claims = fitted.get("research_claims", [])
    while claims and _size() > target_budget:
        claims.pop()
    if not claims:
        fitted.pop("research_claims", None)

    if _size() <= target_budget:
        return fitted

    # Phase 2: Trim resolved items
    resolved = fitted.get("resolved", [])
    while len(resolved) > 1 and _size() > target_budget:
        resolved.pop()

    if _size() <= target_budget:
        return fitted

    # Phase 3: Shorten resolution prose in remaining resolved item
    if resolved:
        res = str(resolved[0].get("resolution") or "")
        if len(res) > 200:
            resolved[0]["resolution"] = res[:200] + "..."

    if _size() <= target_budget:
        return fitted

    # Phase 4: Bounded prompt preview if context is still constrained
    if "original_prompt" in fitted and _size() > target_budget:
        orig = str(fitted["original_prompt"])
        if len(orig) > 1000:
            fitted["original_prompt"] = orig[:1000] + "..."

    return fitted


def _resolved_context(state: Mapping[str, Any], prompt: str = "") -> dict[str, Any]:
    """Give the model semantic facts, never host receipts or identity bookkeeping."""
    known = [
        {"statement": _text(item.get("statement"))}
        for item in state.get("known", [])
        if isinstance(item, Mapping) and _text(item.get("statement"))
    ] if isinstance(state.get("known"), list) else []
    goal = state.get("goal")
    goal_statement = _text(goal.get("statement")) if isinstance(goal, Mapping) else ""
    original_prompt = _text(prompt or state.get("original_prompt"))

    unresolved_questions: dict[str, str] = {}
    if isinstance(state.get("unresolved"), list):
        for item in state.get("unresolved", []):
            if isinstance(item, Mapping):
                uid = str(item.get("unresolved_id") or "")
                q = _text(item.get("question"))
                if uid and q:
                    unresolved_questions[uid] = q

    resolved: list[dict[str, Any]] = []
    seen_resolutions: set[str] = set()
    if isinstance(state.get("resolved"), list):
        for item in state.get("resolved", []):
            if not isinstance(item, Mapping):
                continue
            prose = _resolution_prose(item.get("resolution"))
            if not prose or prose in seen_resolutions:
                continue
            seen_resolutions.add(prose)
            question = unresolved_questions.get(str(item.get("unresolved_id") or ""))
            entry: dict[str, Any] = {}
            if question:
                entry["question"] = question
            entry["resolution"] = prose
            resolved.append(entry)

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
                if text and text not in seen_resolutions:
                    evidence_claims.append(text)
                    seen_resolutions.add(text)

    res: dict[str, Any] = {}
    if original_prompt:
        res["original_prompt"] = original_prompt
    res["goal"] = goal_statement
    res["known"] = known
    res["resolved"] = resolved
    res["research_claims"] = list(dict.fromkeys(evidence_claims))[:8]
    return res


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
            if not semantic_capability:
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

    # Collapse only exact semantic duplicates. Equal player-facing prose can still
    # represent distinct capabilities or independently testable acceptance contracts.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for row in normalized:
        key = (
            _text(row.get("statement")).casefold(),
            _text(row.get("semantic_capability")).casefold(),
            tuple(text.casefold() for text in _strings(row.get("acceptance"))),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _blocking_unknowns(
    state: Mapping[str, Any],
    *,
    stage: str = "requirement_selection",
) -> list[Mapping[str, Any]]:
    """Return only unresolved rows that explicitly block the requested host stage."""
    rows = state.get("unresolved", [])
    if not isinstance(rows, list):
        return []
    return [
        item
        for item in rows
        if isinstance(item, Mapping)
        and item.get("status") != "resolved"
        and stage in _strings(item.get("blocks"))
    ]


def _preserve_blocked_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Represent incomplete requirement knowledge in state instead of throwing it away."""
    value = deepcopy(dict(state))
    blocking = _blocking_unknowns(value, stage="requirement_selection")
    existing = [
        item
        for item in value.get("blockers", [])
        if isinstance(item, Mapping) and item.get("stage") != "requirement_selection"
    ]
    if blocking:
        existing.append(
            {
                "blocker_id": "blocker_requirement_selection",
                "stage": "requirement_selection",
                "statement": "Requirement selection is waiting for unresolved task-state knowledge.",
                "caused_by": [str(item.get("unresolved_id") or "") for item in blocking],
            }
        )
    value["blockers"] = existing
    value["plan_ready"] = False
    return _rehash(value)


def compile_researched_requirements(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Add behavior requirements; semantic model mistakes never become planner invariants."""
    validate_planning_state(state, prompt=prompt)
    if _blocking_unknowns(state, stage="requirement_selection"):
        return _preserve_blocked_state(state)

    raw_context = _resolved_context(state, prompt=prompt)
    budget = _planner_context_budget(router)
    system_content = (
        "Compile independently testable, player-visible requirements from the supplied "
        "task semantics. Do not output or reason about host IDs, evidence IDs, hashes, "
        "receipts, provenance keys, files, classes, registrations, or invented APIs. "
        "Use a short descriptive semantic capability label for bookkeeping only; "
        "it must not choose Minecraft artifacts or architecture. Missing balance values or detailed "
        "mechanics are later design work. Return behavior statements and observable "
        "acceptance conditions only."
    )
    catalog = {}
    overhead_bytes = len(system_content.encode("utf-8")) + len(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    context = _fit_context_to_budget(
        raw_context,
        byte_budget=budget,
        catalog_and_system_bytes=overhead_bytes,
    )
    user_payload = {
        "task": context,
        "custom_capability": CUSTOM_CAPABILITY_SENTINEL,
    }
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": json.dumps(
                user_payload,
                ensure_ascii=False,
                separators=(",", ":"),
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
        from .model_adapters import ModelConfigurationError

        if isinstance(exc, ModelConfigurationError):
            raise
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
    raw_requirement_count = (
        len(raw.get("requirements", []))
        if isinstance(raw, Mapping) and isinstance(raw.get("requirements"), list)
        else 0
    )
    emit_root_cause(
        "planner_requirement_normalization",
        stage="planning_state",
        operation="researched_requirement_compile",
        result="FALLBACK" if generation_error is not None or raw is None else "PASS",
        details={
            "raw_output": raw,
            "raw_requirement_count": raw_requirement_count,
            "normalized_requirement_count": len(requirement_rows),
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
