from __future__ import annotations

"""Canonical task-state SSOT for prompt understanding and grounded planning.

The model performs exactly one bounded semantic extraction at the request boundary. It
may report authored facts, named references, scope status, and genuine unknowns. Host
code owns IDs, reason->route policy, allowed sources, query compilation, evidence,
state transitions, decisions, coverage, blockers, and readiness.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .planner_operation import planner_operation

SCHEMA = "mmm/planning-state-v1"
MODEL_TOOL = "submit_prompt_state"

UNRESOLVED_REASONS = (
    "reference_semantics",
    "external_fact",
    "scope",
    "repository_fact",
    "minecraft_api",
    "implementation_method",
    "compatibility",
    "contradiction",
    "user_preference",
    "insufficient_evidence",
)
_MODEL_UNRESOLVED_REASONS = (
    "external_fact",
    "repository_fact",
    "minecraft_api",
    "implementation_method",
    "compatibility",
    "contradiction",
    "user_preference",
)
RESOLUTION_ROUTES = (
    "reference_research",
    "external_research",
    "repository_rag",
    "minecraft_research",
    "implementation_research",
    "compatibility_research",
    "default_policy",
    "user_only",
)
SOURCE_KINDS = (
    "reference_sources",
    "web_sources",
    "repository",
    "existing_mods",
    "minecraft_docs",
    "minecraft_source",
    "project_rag",
)

_ROUTE_BY_REASON: dict[str, str] = {
    "reference_semantics": "reference_research",
    "external_fact": "external_research",
    "scope": "default_policy",
    "repository_fact": "repository_rag",
    "minecraft_api": "minecraft_research",
    "implementation_method": "implementation_research",
    "compatibility": "compatibility_research",
    "contradiction": "user_only",
    "user_preference": "user_only",
}
_RESEARCH_ROUTES = frozenset(
    {
        "reference_research",
        "external_research",
        "repository_rag",
        "minecraft_research",
        "implementation_research",
        "compatibility_research",
    }
)
ROUTE_SOURCES: dict[str, tuple[str, ...]] = {
    "reference_research": ("reference_sources", "web_sources"),
    "external_research": ("web_sources",),
    "repository_rag": ("repository", "project_rag"),
    "minecraft_research": ("minecraft_docs", "minecraft_source"),
    "implementation_research": (
        "repository",
        "existing_mods",
        "minecraft_docs",
        "minecraft_source",
        "project_rag",
    ),
    "compatibility_research": (
        "repository",
        "existing_mods",
        "minecraft_docs",
        "minecraft_source",
        "project_rag",
    ),
    "default_policy": (),
    "user_only": (),
}

MODEL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "source_quote": {"type": "string"},
            },
            "required": ["statement"],
            "additionalProperties": False,
        },
        "known": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "source_quote": {"type": "string"},
                },
                "required": ["statement"],
                "additionalProperties": False,
            },
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "source_quote": {"type": "string"},
                    "what_must_be_learned": {"type": "string"},
                },
                "required": ["name", "what_must_be_learned"],
                "additionalProperties": False,
            },
        },
        "scope_status": {
            "type": "string",
            "enum": ["explicit", "partial", "unspecified"],
        },
        "unresolved": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reason": {
                        "type": "string",
                        "enum": list(_MODEL_UNRESOLVED_REASONS),
                    },
                    "blocks": {"type": "array", "items": {"type": "string"}},
                    "information_needed": {"type": "string"},
                },
                "required": ["question", "reason", "blocks", "information_needed"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal", "known", "references", "scope_status", "unresolved"],
    "additionalProperties": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = deepcopy(dict(value))
    payload[field] = ""
    return _sha(payload)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(dict.fromkeys(text for item in value if (text := _text(item))))


def _source_receipt(prompt: str, quote: Any) -> dict[str, Any]:
    """Store an optional prompt provenance hint; literal formatting is never a fail gate."""
    text = str(quote or "")
    start = prompt.find(text) if text else -1
    exact = start >= 0
    return {
        "source_id": "requested_prompt",
        "char_start": start if exact else -1,
        "char_end": start + len(text) if exact else -1,
        "text": text,
        "text_sha256": _sha(text) if text else "",
        "verification": "exact_span" if exact else "prompt_authored_hint",
    }


def _route_for_reason(reason: str) -> str:
    try:
        return _ROUTE_BY_REASON[reason]
    except KeyError as exc:
        raise ValueError(
            f"PROMPT_STATE_ROUTE: reason {reason!r} has no initial host route"
        ) from exc


def _validated_route(reason: str, route: str) -> str:
    """Single route-policy validator used by state construction and invariants."""
    if reason == "insufficient_evidence":
        if route not in _RESEARCH_ROUTES:
            raise ValueError(
                "PROMPT_STATE_ROUTE: insufficient_evidence must retain a research route"
            )
        return route
    expected = _route_for_reason(reason)
    if route != expected:
        raise ValueError(
            f"PROMPT_STATE_ROUTE: reason {reason!r} requires {expected!r}, got {route!r}"
        )
    return route


def _unknown(
    *,
    unresolved_id: str,
    question: str,
    reason: str,
    blocks: Sequence[str],
    information_needed: str,
    research_id: str,
) -> dict[str, Any]:
    route = _route_for_reason(reason)
    return {
        "unresolved_id": unresolved_id,
        "question": _text(question),
        "reason": reason,
        "blocks": list(dict.fromkeys(_text(item) for item in blocks if _text(item))),
        "information_needed": _text(information_needed),
        "resolution_route": route,
        "source_kinds": list(ROUTE_SOURCES[route]),
        "status": "open",
        "research_ref": research_id if ROUTE_SOURCES[route] else "",
    }


def _model_unknown(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    question = _text(raw.get("question"))
    information_needed = _text(raw.get("information_needed"))
    reason = _text(raw.get("reason"))
    if not question or not information_needed:
        raise ValueError(
            "PROMPT_STATE_UNRESOLVED: question/information_needed must not be empty"
        )
    if reason not in _MODEL_UNRESOLVED_REASONS:
        raise ValueError(
            f"PROMPT_STATE_UNRESOLVED: model cannot author reason {reason!r}"
        )
    route = _route_for_reason(reason)
    return _unknown(
        unresolved_id=f"u_{index + 1:03d}",
        question=question,
        reason=reason,
        blocks=_strings(raw.get("blocks")),
        information_needed=information_needed,
        research_id=f"r_{index + 1:03d}" if route in _RESEARCH_ROUTES else "",
    )


def _next_numeric_id(items: Sequence[Mapping[str, Any]], field: str, prefix: str) -> str:
    maximum = 0
    for item in items:
        raw = str(item.get(field) or "")
        if not raw.startswith(prefix):
            continue
        try:
            maximum = max(maximum, int(raw.removeprefix(prefix)))
        except ValueError:
            continue
    return f"{prefix}{maximum + 1:03d}"


def _append_host_unknown(
    unresolved: list[dict[str, Any]],
    *,
    question: str,
    reason: str,
    blocks: Sequence[str],
    information_needed: str,
) -> None:
    route = _route_for_reason(reason)
    unresolved.append(
        _unknown(
            unresolved_id=_next_numeric_id(unresolved, "unresolved_id", "u_"),
            question=question,
            reason=reason,
            blocks=blocks,
            information_needed=information_needed,
            research_id=(
                _next_numeric_id(unresolved, "research_ref", "r_")
                if route in _RESEARCH_ROUTES
                else ""
            ),
        )
    )


def _host_reference(prompt: str, raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    name = _text(raw.get("name"))
    needed = _text(raw.get("what_must_be_learned"))
    if not name or not needed:
        raise ValueError(
            "PROMPT_STATE_REFERENCE: reference name/research need must not be empty"
        )
    return {
        "reference_id": f"ref_{index + 1:03d}",
        "name": name,
        "source": _source_receipt(prompt, raw.get("source_quote")),
        "what_must_be_learned": needed,
    }


def _host_known(prompt: str, raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    statement = _text(raw.get("statement"))
    if not statement:
        raise ValueError("PROMPT_STATE_KNOWN: statement must not be empty")
    return {
        "known_id": f"known_{index + 1:03d}",
        "statement": statement,
        "source": _source_receipt(prompt, raw.get("source_quote")),
    }


def _ensure_mechanical_unknowns(
    unresolved: list[dict[str, Any]],
    references: Sequence[Mapping[str, Any]],
    scope_status: str,
) -> None:
    for reference in references:
        name = _text(reference.get("name"))
        _append_host_unknown(
            unresolved,
            question=f"What documented systems, rules, and behavior define {name}?",
            reason="reference_semantics",
            blocks=("requirement_selection", "implementation_plan"),
            information_needed=(
                _text(reference.get("what_must_be_learned"))
                or f"Documented behavior and structure of {name}"
            ),
        )
    if scope_status in {"partial", "unspecified"}:
        _append_host_unknown(
            unresolved,
            question=(
                "What bounded implementation scope may be selected without inventing "
                "unauthored requirements?"
            ),
            reason="scope",
            blocks=("requirement_selection",),
            information_needed=(
                "Apply the canonical default scope policy after externally grounded facts "
                "are available."
            ),
        )


def _build_host_state(prompt: str, model_value: Mapping[str, Any]) -> dict[str, Any]:
    goal_raw = model_value.get("goal")
    known_raw = model_value.get("known")
    references_raw = model_value.get("references")
    unresolved_raw = model_value.get("unresolved")
    scope_status = _text(model_value.get("scope_status"))
    if not isinstance(goal_raw, Mapping) or not _text(goal_raw.get("statement")):
        raise ValueError("PROMPT_STATE_GOAL: goal must contain a statement")
    if not isinstance(known_raw, list) or not isinstance(references_raw, list) or not isinstance(unresolved_raw, list):
        raise ValueError("PROMPT_STATE_SHAPE: known/references/unresolved must be arrays")
    if scope_status not in {"explicit", "partial", "unspecified"}:
        raise ValueError("PROMPT_STATE_SCOPE: scope_status is invalid")

    references = [
        _host_reference(prompt, item, index=index)
        for index, item in enumerate(references_raw)
        if isinstance(item, Mapping)
    ]
    unresolved = [
        _model_unknown(item, index=index)
        for index, item in enumerate(unresolved_raw)
        if isinstance(item, Mapping)
    ]
    _ensure_mechanical_unknowns(unresolved, references, scope_status)

    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "original_prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "goal": {
            "statement": _text(goal_raw.get("statement")),
            "source": _source_receipt(prompt, goal_raw.get("source_quote")),
        },
        "known": [
            _host_known(prompt, item, index=index)
            for index, item in enumerate(known_raw)
            if isinstance(item, Mapping)
        ],
        "references": references,
        "scope_status": scope_status,
        "unresolved": unresolved,
        "research_queue": [
            {
                "research_id": item["research_ref"],
                "resolves": [item["unresolved_id"]],
                "objective": item["question"],
                "information_needed": item["information_needed"],
                "source_kinds": list(item["source_kinds"]),
                "queries": [],
                "status": "pending",
            }
            for item in unresolved
            if item["research_ref"]
        ],
        "evidence": [],
        "resolved": [],
        "decisions": [],
        "implementation_candidates": [],
        "coverage": [],
        "blockers": [],
        "plan_ready": False,
        "state_sha256": "",
    }
    state["state_sha256"] = _hash_without(state, "state_sha256")
    validate_planning_state(state, prompt=prompt)
    _validate_initial_state(state)
    return state


def validate_planning_state(
    state: Mapping[str, Any],
    *,
    prompt: str | None = None,
) -> None:
    """Validate state topology and host policies without re-interpreting prompt text."""
    if state.get("schema_version") != SCHEMA:
        raise ValueError("PROMPT_STATE_SCHEMA: unsupported planning-state schema")
    original = str(state.get("original_prompt") or "")
    if prompt is not None and original != prompt:
        raise ValueError("PROMPT_STATE_PROMPT: planning state belongs to another prompt")
    if state.get("prompt_sha256") != _sha(original):
        raise ValueError("PROMPT_STATE_PROMPT: prompt hash mismatch")
    if state.get("state_sha256") != _hash_without(state, "state_sha256"):
        raise ValueError("PROMPT_STATE_HASH: state hash mismatch")

    unresolved = state.get("unresolved")
    queue = state.get("research_queue")
    if not isinstance(unresolved, list) or not isinstance(queue, list):
        raise ValueError("PROMPT_STATE_SHAPE: unresolved/research_queue must be arrays")

    unresolved_ids: set[str] = set()
    for item in unresolved:
        if not isinstance(item, Mapping):
            raise ValueError("PROMPT_STATE_UNRESOLVED: unresolved item must be an object")
        unresolved_id = str(item.get("unresolved_id") or "")
        if not unresolved_id or unresolved_id in unresolved_ids:
            raise ValueError("PROMPT_STATE_UNRESOLVED: IDs must be non-empty and unique")
        unresolved_ids.add(unresolved_id)
        reason = str(item.get("reason") or "")
        if reason not in UNRESOLVED_REASONS:
            raise ValueError(f"PROMPT_STATE_UNRESOLVED: unsupported reason {reason!r}")
        route = _validated_route(reason, str(item.get("resolution_route") or ""))
        if list(item.get("source_kinds") or []) != list(ROUTE_SOURCES[route]):
            raise ValueError(
                "PROMPT_STATE_ROUTE: unresolved source kinds differ from host route policy"
            )

    research_ids: set[str] = set()
    for item in queue:
        if not isinstance(item, Mapping):
            raise ValueError("PROMPT_STATE_RESEARCH: research item must be an object")
        research_id = str(item.get("research_id") or "")
        if not research_id or research_id in research_ids:
            raise ValueError("PROMPT_STATE_RESEARCH: research IDs must be non-empty and unique")
        research_ids.add(research_id)
        resolves = item.get("resolves")
        if not isinstance(resolves, list) or not resolves or any(str(ref) not in unresolved_ids for ref in resolves):
            raise ValueError(
                "PROMPT_STATE_RESEARCH: every research item must resolve a real unresolved item"
            )
        if item.get("queries") and not item.get("information_needed"):
            raise ValueError(
                "PROMPT_STATE_RESEARCH: queries require an information_needed objective"
            )

    evidence_rows = state.get("evidence", [])
    if not isinstance(evidence_rows, list):
        raise ValueError("PROMPT_STATE_EVIDENCE: evidence must be an array")
    for evidence in evidence_rows:
        if not isinstance(evidence, Mapping):
            raise ValueError("PROMPT_STATE_EVIDENCE: evidence item must be an object")
        if str(evidence.get("research_ref") or "") not in research_ids:
            raise ValueError("PROMPT_STATE_EVIDENCE: evidence must belong to queued research")
        if str(evidence.get("source") or "") != "grounded_materialized_pages":
            raise ValueError("PROMPT_STATE_EVIDENCE: model output is not an evidence source")

    if type(state.get("plan_ready")) is not bool:
        raise ValueError("PROMPT_STATE_READY: plan_ready must be boolean")
    if state.get("plan_ready"):
        if any(
            isinstance(item, Mapping) and item.get("status") != "resolved"
            for item in unresolved
        ):
            raise ValueError(
                "PROMPT_STATE_READY: plan cannot be ready while blocking unknowns remain"
            )
        coverage = state.get("coverage")
        if not isinstance(coverage, list) or not coverage:
            raise ValueError("PROMPT_STATE_READY: ready plan requires coverage records")

    from .planning_state_invariants import validate_state_links

    validate_state_links(state)


def _validate_initial_state(state: Mapping[str, Any]) -> None:
    if any(
        state.get(field)
        for field in (
            "evidence",
            "resolved",
            "decisions",
            "implementation_candidates",
            "coverage",
            "blockers",
        )
    ):
        raise ValueError(
            "PROMPT_STATE_INITIAL: semantic extraction cannot contain derived artifacts"
        )
    if state.get("plan_ready") is not False:
        raise ValueError("PROMPT_STATE_INITIAL: initial state cannot be plan-ready")
    if any(
        item.get("queries")
        for item in state.get("research_queue", [])
        if isinstance(item, Mapping)
    ):
        raise ValueError(
            "PROMPT_STATE_INITIAL: information_needed must exist before retrieval queries"
        )


def build_initial_planning_state(router: Any, prompt: str) -> dict[str, Any]:
    """Perform the sole model-owned semantic extraction for an authored request."""
    authored = str(prompt or "")
    if not authored.strip():
        raise ValueError("PROMPT_STATE_PROMPT: prompt must not be empty")
    messages = [
        {
            "role": "system",
            "content": (
                "Fill the canonical prompt-understanding template only. Preserve the user's "
                "goal. known contains only facts explicitly authored by the user. references "
                "contains named games, mods, products, styles, works, or external concepts that "
                "must be understood; do not duplicate them as reference_semantics unknowns, "
                "because the host creates those. Set scope_status from the request; do not create "
                "scope unknowns, because the host creates those. For every other genuine unknown, "
                "provide question, semantic reason, information_needed, and what it blocks. Do not "
                "choose routes, sources, IDs, queries, APIs, files, architecture, mechanics, or "
                "implementation details. Do not use model memory as evidence. Contradictions and "
                "user preferences remain unknown rather than invented answers. source_quote is "
                "optional metadata and need not preserve literal whitespace. There is no target "
                "number of rows: represent the request faithfully without count-driven splitting."
            ),
        },
        {"role": "user", "content": "USER REQUEST:\n" + authored},
    ]
    with planner_operation("prompt_state", output_tokens=1536):
        raw = router.generate_tool_decision(
            "planner",
            messages,
            tool_name=MODEL_TOOL,
            parameters=MODEL_PARAMETERS,
            description=(
                "Submit authored facts, references, scope status, and semantic unknowns only."
            ),
        )
    if not isinstance(raw, Mapping):
        raise ValueError("PROMPT_STATE_MODEL: planner did not return an object")
    return _build_host_state(authored, raw)


__all__ = [
    "MODEL_PARAMETERS",
    "MODEL_TOOL",
    "RESOLUTION_ROUTES",
    "ROUTE_SOURCES",
    "SCHEMA",
    "SOURCE_KINDS",
    "UNRESOLVED_REASONS",
    "build_initial_planning_state",
    "validate_planning_state",
]
