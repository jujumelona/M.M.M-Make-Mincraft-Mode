from __future__ import annotations

"""Compile the smallest complete set of player-visible implementation capabilities.

The model describes semantic capabilities and observable acceptance. Host code owns IDs,
bookkeeping and convergence. Variants that share one implementation subsystem/state
owner are acceptance cases of one requirement, not separate requirements.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

CUSTOM_CAPABILITY_SENTINEL = "custom"

from .planner_operation import planner_operation
from .planning_contract_ssot import SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA
from .planning_state_contract import validate_planning_state
from .root_cause_trace import emit_root_cause

_REQUIREMENT_TOOL = "submit_researched_requirements"
_REQUIREMENT_PARAMETERS: dict[str, Any] = SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        text = _text(value)
        return [text] if text else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
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
    target_budget = max(4096, byte_budget - catalog_and_system_bytes - 1024)
    fitted = deepcopy(context)

    def _size() -> int:
        return len(
            json.dumps(fitted, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )

    if _size() <= target_budget:
        return fitted

    claims = fitted.get("research_claims", [])
    while claims and _size() > target_budget:
        claims.pop()
    if not claims:
        fitted.pop("research_claims", None)

    resolved = fitted.get("resolved", [])
    while len(resolved) > 1 and _size() > target_budget:
        resolved.pop()
    if resolved and _size() > target_budget:
        prose = str(resolved[0].get("resolution") or "")
        if len(prose) > 200:
            resolved[0]["resolution"] = prose[:200] + "..."

    if "original_prompt" in fitted and _size() > target_budget:
        original = str(fitted["original_prompt"])
        if len(original) > 1000:
            fitted["original_prompt"] = original[:1000] + "..."
    return fitted


def _resolved_context(state: Mapping[str, Any], prompt: str = "") -> dict[str, Any]:
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
                question = _text(item.get("question"))
                if uid and question:
                    unresolved_questions[uid] = question

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
            entry: dict[str, Any] = {"resolution": prose}
            question = unresolved_questions.get(str(item.get("unresolved_id") or ""))
            if question:
                entry["question"] = question
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

    result: dict[str, Any] = {}
    if original_prompt:
        result["original_prompt"] = original_prompt
    result["goal"] = goal_statement
    result["known"] = known
    result["resolved"] = resolved
    result["research_claims"] = list(dict.fromkeys(evidence_claims))[:8]
    return result


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _fallback_requirement_rows(state: Mapping[str, Any], prompt: str) -> list[dict[str, Any]]:
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


def _capability_key(row: Mapping[str, Any]) -> str:
    capability = _text(row.get("semantic_capability")).casefold()
    if capability and capability != CUSTOM_CAPABILITY_SENTINEL:
        return capability
    return _text(row.get("statement")).casefold()


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

    # Same capability means one implementation obligation. Merge observable cases instead
    # of multiplying detailed plans for acquisition/stat/location/specialization variants.
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in normalized:
        key = _capability_key(row)
        if key not in merged:
            merged[key] = deepcopy(row)
            order.append(key)
            continue
        current = merged[key]
        current["acceptance"] = list(
            dict.fromkeys(_strings(current.get("acceptance")) + _strings(row.get("acceptance")))
        )
    return [merged[key] for key in order]


def _blocking_unknowns(
    state: Mapping[str, Any],
    *,
    stage: str = "requirement_selection",
) -> list[Mapping[str, Any]]:
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


def _page_identity(row: Mapping[str, Any]) -> str:
    return _capability_key(row)


def _generate_requirement_pages(router: Any, messages: list[dict[str, str]], budget: int) -> Any:
    """Page only genuinely new capabilities until the semantic frontier is exhausted."""
    from .model_adapters import ModelConfigurationError

    page_size = int(_REQUIREMENT_PARAMETERS["properties"]["requirements"]["maxItems"])
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    payload = json.loads(messages[1]["content"])
    page_index = 0
    while True:
        current_messages = deepcopy(messages)
        if collected:
            current_messages[1]["content"] = json.dumps(
                {**payload, "already_compiled_requirements": collected},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if sum(len(row["content"].encode("utf-8")) for row in current_messages) > budget:
                raise ModelConfigurationError(
                    "REQUIREMENT_PAGINATION_CONTEXT_EXHAUSTED: continuation cannot fit "
                    "without losing authored task or prior capability coverage"
                )
        try:
            with planner_operation("researched_requirement_compile", output_tokens=2048):
                raw = router.generate_tool_decision(
                    "planner",
                    current_messages,
                    tool_name=_REQUIREMENT_TOOL,
                    parameters=_REQUIREMENT_PARAMETERS,
                    description="Submit the next page of minimal distinct player-visible capabilities.",
                )
        except Exception as exc:
            if not collected:
                raise
            raise ModelConfigurationError(
                "REQUIREMENT_PAGINATION_FAILED: continuation failed; partial requirements are not complete"
            ) from exc

        rows = raw.get("requirements") if isinstance(raw, Mapping) else None
        if not isinstance(rows, list):
            if not collected:
                return raw
            raise ModelConfigurationError("REQUIREMENT_PAGINATION_FAILED: invalid continuation page")

        emit_root_cause(
            "planner_requirement_page_received",
            stage="planning_state",
            operation="researched_requirement_compile",
            result="INFO",
            details={"page_index": page_index + 1, "requirements": rows},
        )

        page_rows: list[dict[str, Any]] = []
        page_seen: set[str] = set()
        repeated_prior: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping) or not _text(row.get("statement")):
                raise ModelConfigurationError("REQUIREMENT_PAGINATION_FAILED: invalid requirement row")
            identity = _page_identity(row)
            if identity in seen or identity in page_seen:
                repeated_prior.append(dict(row))
                continue
            page_seen.add(identity)
            page_rows.append(dict(row))

        page_index += 1
        if not page_rows:
            emit_root_cause(
                "planner_requirement_page",
                stage="planning_state",
                operation="researched_requirement_compile",
                result="COMPLETE",
                reason="REQUIREMENT_SEMANTIC_FRONTIER_EXHAUSTED",
                details={
                    "page_index": page_index,
                    "page_requirement_count": len(rows),
                    "accepted_requirement_count": 0,
                    "discarded_repeat_count": len(repeated_prior),
                    "repeated_prior_requirements": repeated_prior,
                    "total_requirement_count": len(collected),
                    "requirements": rows,
                },
            )
            return {"requirements": collected}

        seen.update(page_seen)
        collected.extend(page_rows)
        page_full = len(rows) >= page_size
        emit_root_cause(
            "planner_requirement_page",
            stage="planning_state",
            operation="researched_requirement_compile",
            result="CONTINUE" if page_full else "COMPLETE",
            reason="REQUIREMENT_PAGE_FULL_CONTINUE" if page_full else "REQUIREMENT_PAGE_PARTIAL_COMPLETE",
            details={
                "page_index": page_index,
                "page_requirement_count": len(rows),
                "accepted_requirement_count": len(page_rows),
                "discarded_repeat_count": len(repeated_prior),
                "total_requirement_count": len(collected),
                "requirements": rows,
            },
        )
        if not page_full:
            return {"requirements": collected}


def compile_researched_requirements(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the minimum distinct capabilities needed to cover the authored request."""
    validate_planning_state(state, prompt=prompt)
    if _blocking_unknowns(state, stage="requirement_selection"):
        return _preserve_blocked_state(state)

    raw_context = _resolved_context(state, prompt=prompt)
    budget = _planner_context_budget(router)
    system_content = (
        "Compile the smallest complete set of player-visible implementation capabilities "
        "from the supplied task semantics. Group behaviors that share the same subsystem, "
        "state owner, lifecycle, or implementation responsibility into ONE requirement and "
        "put the observable variants in that requirement's acceptance conditions. Do not "
        "split a capability merely because acquisition method, stat type, location, actor, "
        "specialization, or wording differs. Examples: planetary exploration and interplanetary "
        "travel belong to one travel capability when they use one travel system; mineral discovery "
        "and extraction belong to one planetary-resource capability; generic and specialized crew "
        "recruitment belong to one crew capability; combat variants belong to one combat capability "
        "when they share one combat system. Split only when a genuinely distinct implementation "
        "capability/state owner is required. Do not output or reason about host IDs, evidence IDs, "
        "hashes, receipts, provenance keys, files, classes, registrations, or invented APIs. Use a "
        "short stable subsystem-level semantic capability label. Missing balance values or detailed "
        "mechanics are later design work. Return at most four NEW capabilities on this page; four is "
        "a ceiling, never a target. Exclude anything already covered by already_compiled_requirements. "
        "If no genuinely new capability remains, return an empty requirements array immediately. "
        "Return fewer than four whenever that is the complete remaining set. Preserve user qualifiers "
        "as acceptance conditions rather than multiplying requirements."
    )
    catalog: dict[str, Any] = {}
    overhead_bytes = len(system_content.encode("utf-8")) + len(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    context = _fit_context_to_budget(
        raw_context,
        byte_budget=budget,
        catalog_and_system_bytes=overhead_bytes,
    )
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": json.dumps(
                {"task": context, "custom_capability": CUSTOM_CAPABILITY_SENTINEL},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]

    raw: Any = None
    generation_error: BaseException | None = None
    try:
        raw = _generate_requirement_pages(router, messages, budget)
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
            details={"messages": messages, "tool_name": _REQUIREMENT_TOOL, "parameters": _REQUIREMENT_PARAMETERS},
            exc=exc,
        )
    else:
        emit_root_cause(
            "planner_model_decision",
            stage="planning_state",
            operation="researched_requirement_compile",
            result="PASS",
            details={"messages": messages, "tool_name": _REQUIREMENT_TOOL, "parameters": _REQUIREMENT_PARAMETERS, "raw_output": raw},
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
            "generation_error": f"{type(generation_error).__name__}: {generation_error}" if generation_error is not None else None,
        },
    )

    value = deepcopy(dict(state))
    value.setdefault("decisions", [])
    value.setdefault("blockers", [])
    value["blockers"] = [
        item for item in value["blockers"]
        if not (isinstance(item, Mapping) and item.get("stage") == "requirement_selection")
    ]
    value["decisions"] = [
        item for item in value["decisions"]
        if not (isinstance(item, Mapping) and item.get("decision_type") == "requirement")
    ]

    for index, row in enumerate(requirement_rows, start=1):
        value["decisions"].append(
            {
                "decision_id": f"d_{len(value['decisions']) + 1:03d}",
                "decision_type": "requirement",
                "requirement_id": f"req_{index:03d}",
                "statement": row["statement"],
                "semantic_capability": row["semantic_capability"],
                "acceptance": row["acceptance"],
            }
        )

    value["plan_ready"] = False
    return _rehash(value)


__all__ = ["compile_researched_requirements"]
