from __future__ import annotations

"""Prompt-first SSOT for planning, research, and implementation readiness.

The first planner call fills only authored facts and unknowns. Host code owns IDs,
provenance, research queues, route legality, state transitions, evidence, coverage, and
readiness. The model may describe an unknown; it may not choose a route that bypasses the
kind of evidence the unknown requires.
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
    "reference_semantics", "external_fact", "scope", "repository_fact", "minecraft_api",
    "implementation_method", "compatibility", "contradiction", "user_preference",
    "insufficient_evidence",
)
RESOLUTION_ROUTES = (
    "reference_research", "external_research", "repository_rag", "minecraft_research",
    "implementation_research", "compatibility_research", "default_policy", "user_only",
)
SOURCE_KINDS = (
    "reference_sources", "web_sources", "repository", "existing_mods", "minecraft_docs",
    "minecraft_source", "project_rag",
)
_ALLOWED_ROUTES_BY_REASON: dict[str, frozenset[str]] = {
    "reference_semantics": frozenset({"reference_research"}),
    "external_fact": frozenset({"external_research"}),
    "scope": frozenset({"default_policy"}),
    "repository_fact": frozenset({"repository_rag"}),
    "minecraft_api": frozenset({"minecraft_research"}),
    "implementation_method": frozenset({"implementation_research"}),
    "compatibility": frozenset({"compatibility_research"}),
    "contradiction": frozenset({"external_research", "repository_rag", "minecraft_research", "default_policy"}),
    "user_preference": frozenset({"user_only"}),
    "insufficient_evidence": frozenset({"reference_research", "external_research", "repository_rag", "minecraft_research", "implementation_research", "compatibility_research"}),
}

MODEL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "object",
            "properties": {"statement": {"type": "string"}, "source_quote": {"type": "string"}},
            "required": ["statement", "source_quote"],
            "additionalProperties": False,
        },
        "known": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"statement": {"type": "string"}, "source_quote": {"type": "string"}},
                "required": ["statement", "source_quote"],
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
                "required": ["name", "source_quote", "what_must_be_learned"],
                "additionalProperties": False,
            },
        },
        "scope_status": {"type": "string", "enum": ["explicit", "partial", "unspecified"]},
        "unresolved": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reason": {"type": "string", "enum": list(UNRESOLVED_REASONS)},
                    "blocks": {"type": "array", "items": {"type": "string"}},
                    "information_needed": {"type": "string"},
                    "resolution_route": {"type": "string", "enum": list(RESOLUTION_ROUTES)},
                    "source_kinds": {"type": "array", "items": {"type": "string", "enum": list(SOURCE_KINDS)}},
                },
                "required": ["question", "reason", "blocks", "information_needed", "resolution_route", "source_kinds"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal", "known", "references", "scope_status", "unresolved"],
    "additionalProperties": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"), default=str)


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
    return list(dict.fromkeys(_text(item) for item in value if _text(item)))


def _quote_receipt(prompt: str, quote: Any) -> dict[str, Any]:
    text = str(quote or "").strip()
    if not text:
        raise ValueError("PROMPT_STATE_SOURCE: source_quote must not be empty")
    start = prompt.find(text)
    if start < 0:
        raise ValueError(f"PROMPT_STATE_SOURCE: source_quote is not an exact authored span: {text!r}")
    return {
        "source_id": "requested_prompt",
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
        "text_sha256": _sha(text),
    }


def _host_item(prompt: str, raw: Mapping[str, Any], *, prefix: str, index: int) -> dict[str, Any]:
    statement = _text(raw.get("statement"))
    if not statement:
        raise ValueError(f"PROMPT_STATE_{prefix.upper()}: statement must not be empty")
    return {
        f"{prefix}_id": f"{prefix}_{index + 1:03d}",
        "statement": statement,
        "source": _quote_receipt(prompt, raw.get("source_quote")),
    }


def _validated_route(reason: str, route: str) -> str:
    allowed = _ALLOWED_ROUTES_BY_REASON.get(reason, frozenset())
    if route not in allowed:
        raise ValueError(
            f"PROMPT_STATE_ROUTE: reason {reason!r} cannot use resolution route {route!r}; "
            f"allowed={sorted(allowed)}"
        )
    return route


def _research_item(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    question = _text(raw.get("question"))
    information_needed = _text(raw.get("information_needed"))
    reason = _text(raw.get("reason"))
    route = _text(raw.get("resolution_route"))
    if not question or not information_needed:
        raise ValueError("PROMPT_STATE_UNRESOLVED: question/information_needed must not be empty")
    if reason not in UNRESOLVED_REASONS or route not in RESOLUTION_ROUTES:
        raise ValueError("PROMPT_STATE_UNRESOLVED: reason/resolution route is unsupported")
    route = _validated_route(reason, route)
    sources = [item for item in _strings(raw.get("source_kinds")) if item in SOURCE_KINDS]
    unresolved_id = f"u_{index + 1:03d}"
    return {
        "unresolved_id": unresolved_id,
        "question": question,
        "reason": reason,
        "blocks": _strings(raw.get("blocks")),
        "information_needed": information_needed,
        "resolution_route": route,
        "source_kinds": sources,
        "status": "open",
        "research_ref": f"r_{index + 1:03d}" if route != "user_only" else "",
    }


def _reference_item(prompt: str, raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    name = _text(raw.get("name"))
    needed = _text(raw.get("what_must_be_learned"))
    if not name or not needed:
        raise ValueError("PROMPT_STATE_REFERENCE: reference name/research need must not be empty")
    return {
        "reference_id": f"ref_{index + 1:03d}",
        "name": name,
        "source": _quote_receipt(prompt, raw.get("source_quote")),
        "what_must_be_learned": needed,
    }


def _next_unresolved_id(unresolved: Sequence[Mapping[str, Any]]) -> str:
    maximum = 0
    for item in unresolved:
        raw = str(item.get("unresolved_id") or "")
        if raw.startswith("u_"):
            try:
                maximum = max(maximum, int(raw.removeprefix("u_")))
            except ValueError:
                pass
    return f"u_{maximum + 1:03d}"


def _next_research_id(unresolved: Sequence[Mapping[str, Any]]) -> str:
    maximum = 0
    for item in unresolved:
        raw = str(item.get("research_ref") or "")
        if raw.startswith("r_"):
            try:
                maximum = max(maximum, int(raw.removeprefix("r_")))
            except ValueError:
                pass
    return f"r_{maximum + 1:03d}"


def _ensure_host_unknowns(
    unresolved: list[dict[str, Any]],
    references: Sequence[Mapping[str, Any]],
    scope_status: str,
) -> None:
    """Close omissions that are mechanically implied by the model's own extracted state."""
    for reference in references:
        name = _text(reference.get("name"))
        already = any(
            item.get("reason") == "reference_semantics"
            and name.casefold() in _text(item.get("question")).casefold()
            for item in unresolved
        )
        if already:
            continue
        unresolved_id = _next_unresolved_id(unresolved)
        research_id = _next_research_id(unresolved)
        unresolved.append(
            {
                "unresolved_id": unresolved_id,
                "question": f"What documented systems, rules, and distinctive behavior define the referenced subject {name}?",
                "reason": "reference_semantics",
                "blocks": ["requirement_selection", "implementation_plan"],
                "information_needed": _text(reference.get("what_must_be_learned")) or f"Documented behavior and structure of {name}",
                "resolution_route": "reference_research",
                "source_kinds": ["reference_sources", "web_sources"],
                "status": "open",
                "research_ref": research_id,
            }
        )
    if scope_status in {"partial", "unspecified"} and not any(
        item.get("reason") == "scope" for item in unresolved
    ):
        unresolved_id = _next_unresolved_id(unresolved)
        unresolved.append(
            {
                "unresolved_id": unresolved_id,
                "question": "What bounded implementation scope may be selected without inventing unauthored requirements?",
                "reason": "scope",
                "blocks": ["requirement_selection"],
                "information_needed": "A deterministic default scope policy applied only after reference/external facts are grounded.",
                "resolution_route": "default_policy",
                "source_kinds": [],
                "status": "open",
                "research_ref": "",
            }
        )


def _build_host_state(prompt: str, model_value: Mapping[str, Any]) -> dict[str, Any]:
    goal_raw = model_value.get("goal")
    if not isinstance(goal_raw, Mapping) or not _text(goal_raw.get("statement")):
        raise ValueError("PROMPT_STATE_GOAL: goal must contain a statement")
    known_raw = model_value.get("known")
    references_raw = model_value.get("references")
    unresolved_raw = model_value.get("unresolved")
    if not isinstance(known_raw, list) or not isinstance(references_raw, list) or not isinstance(unresolved_raw, list):
        raise ValueError("PROMPT_STATE_SHAPE: known/references/unresolved must be arrays")
    references = [
        _reference_item(prompt, item, index=i)
        for i, item in enumerate(references_raw)
        if isinstance(item, Mapping)
    ]
    unresolved = [
        _research_item(item, index=index)
        for index, item in enumerate(unresolved_raw)
        if isinstance(item, Mapping)
    ]
    scope_status = _text(model_value.get("scope_status"))
    if scope_status not in {"explicit", "partial", "unspecified"}:
        raise ValueError("PROMPT_STATE_SCOPE: scope_status is invalid")
    _ensure_host_unknowns(unresolved, references, scope_status)
    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "original_prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "goal": {
            "statement": _text(goal_raw.get("statement")),
            "source": _quote_receipt(prompt, goal_raw.get("source_quote")),
        },
        "known": [_host_item(prompt, item, prefix="known", index=i) for i, item in enumerate(known_raw) if isinstance(item, Mapping)],
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
            for item in unresolved if item["research_ref"]
        ],
        "evidence": [],
        "resolved": [],
        "decisions": [],
        "coverage": [],
        "blockers": [],
        "plan_ready": False,
        "state_sha256": "",
    }
    state["state_sha256"] = _hash_without(state, "state_sha256")
    validate_planning_state(state, prompt=prompt)
    _validate_initial_state(state)
    return state


def validate_planning_state(state: Mapping[str, Any], *, prompt: str | None = None) -> None:
    """Validate both initial and evolved host-owned planning states."""
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
    unresolved_ids = {
        str(item.get("unresolved_id") or "") for item in unresolved if isinstance(item, Mapping)
    }
    if "" in unresolved_ids:
        raise ValueError("PROMPT_STATE_UNRESOLVED: every unresolved item needs an ID")
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
            raise ValueError("PROMPT_STATE_RESEARCH: every research item must resolve a real unresolved item")
    for evidence in state.get("evidence", []) if isinstance(state.get("evidence"), list) else []:
        if not isinstance(evidence, Mapping):
            raise ValueError("PROMPT_STATE_EVIDENCE: evidence item must be an object")
        if str(evidence.get("research_ref") or "") not in research_ids:
            raise ValueError("PROMPT_STATE_EVIDENCE: evidence must belong to a queued research item")
        if str(evidence.get("source") or "") != "grounded_materialized_pages":
            raise ValueError("PROMPT_STATE_EVIDENCE: model output is not an evidence source")
    if type(state.get("plan_ready")) is not bool:
        raise ValueError("PROMPT_STATE_READY: plan_ready must be boolean")
    if state.get("plan_ready"):
        blocking = [
            item for item in unresolved
            if isinstance(item, Mapping)
            and item.get("status") != "resolved"
            and item.get("resolution_route") != "user_only"
        ]
        if blocking:
            raise ValueError("PROMPT_STATE_READY: plan cannot be ready while blocking unknowns remain")
        coverage = state.get("coverage")
        if not isinstance(coverage, list) or not coverage:
            raise ValueError("PROMPT_STATE_READY: ready plan requires coverage records")


def _validate_initial_state(state: Mapping[str, Any]) -> None:
    if state.get("evidence") or state.get("resolved") or state.get("decisions") or state.get("coverage"):
        raise ValueError("PROMPT_STATE_INITIAL: model-authored initial state cannot contain derived artifacts")
    if state.get("plan_ready") is not False:
        raise ValueError("PROMPT_STATE_INITIAL: initial state cannot be plan-ready")
    if any(item.get("queries") for item in state.get("research_queue", []) if isinstance(item, Mapping)):
        raise ValueError("PROMPT_STATE_INITIAL: retrieval queries are compiled only after information needs exist")


def build_initial_planning_state(router: Any, prompt: str) -> dict[str, Any]:
    authored = str(prompt or "")
    if not authored.strip():
        raise ValueError("PROMPT_STATE_PROMPT: prompt must not be empty")
    messages = [
        {
            "role": "system",
            "content": (
                "Fill only the prompt-understanding template. Do not design a Minecraft implementation. "
                "Do not invent APIs, files, systems, features, mechanics, or facts absent from the prompt. "
                "KNOWN entries require exact source_quote support. Named games/products/styles/works/concepts "
                "whose meaning must be learned belong in references and unresolved(reference_semantics). "
                "If scope is not authored, mark it partial or unspecified instead of guessing. Route each "
                "unknown according to its reason; the host rejects research-bypass routes."
            ),
        },
        {"role": "user", "content": "USER REQUEST (verbatim):\n" + authored},
    ]
    with planner_operation("prompt_state", output_tokens=1536):
        raw = router.generate_tool_decision(
            "planner",
            messages,
            tool_name=MODEL_TOOL,
            parameters=MODEL_PARAMETERS,
            description="Submit the bounded authored-fact and unresolved-research state for this request.",
        )
    if not isinstance(raw, Mapping):
        raise ValueError("PROMPT_STATE_MODEL: planner did not return an object")
    return _build_host_state(authored, raw)


__all__ = [
    "MODEL_PARAMETERS", "MODEL_TOOL", "RESOLUTION_ROUTES", "SCHEMA", "SOURCE_KINDS",
    "UNRESOLVED_REASONS", "build_initial_planning_state", "validate_planning_state",
]
