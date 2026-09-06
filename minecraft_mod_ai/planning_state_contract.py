from __future__ import annotations

"""Prompt-first SSOT for planning, research, and implementation readiness.

The first planner model call is intentionally *not* an implementation plan.  It fills a
small epistemic-state template containing only authored facts and questions that must be
resolved before design.  Host code owns IDs, provenance, research queue construction,
state transitions, and readiness.  Model output is never evidence.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .planner_operation import planner_operation

SCHEMA = "mmm/planning-state-v1"
MODEL_TOOL = "submit_prompt_state"
_UNRESOLVED_REASONS = (
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
_RESOLUTION_ROUTES = (
    "reference_research",
    "external_research",
    "repository_rag",
    "minecraft_research",
    "implementation_research",
    "compatibility_research",
    "default_policy",
    "user_only",
)
_SOURCE_KINDS = (
    "reference_sources",
    "web_sources",
    "repository",
    "existing_mods",
    "minecraft_docs",
    "minecraft_source",
    "project_rag",
)

MODEL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "source_quote": {"type": "string"},
            },
            "required": ["statement", "source_quote"],
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
                    "reason": {"type": "string", "enum": list(_UNRESOLVED_REASONS)},
                    "blocks": {"type": "array", "items": {"type": "string"}},
                    "information_needed": {"type": "string"},
                    "resolution_route": {"type": "string", "enum": list(_RESOLUTION_ROUTES)},
                    "source_kinds": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(_SOURCE_KINDS)},
                    },
                },
                "required": [
                    "question",
                    "reason",
                    "blocks",
                    "information_needed",
                    "resolution_route",
                    "source_kinds",
                ],
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


def _quote_receipt(prompt: str, quote: Any) -> dict[str, Any]:
    text = str(quote or "").strip()
    if not text:
        raise ValueError("PROMPT_STATE_SOURCE: source_quote must not be empty")
    start = prompt.find(text)
    if start < 0:
        raise ValueError(
            f"PROMPT_STATE_SOURCE: source_quote is not an exact authored span: {text!r}"
        )
    end = start + len(text)
    return {
        "source_id": "requested_prompt",
        "char_start": start,
        "char_end": end,
        "text": text,
        "text_sha256": _sha(text),
    }


def _host_item(prompt: str, raw: Mapping[str, Any], *, prefix: str, index: int) -> dict[str, Any]:
    statement = _text(raw.get("statement"))
    if not statement:
        raise ValueError(f"PROMPT_STATE_{prefix.upper()}: statement must not be empty")
    source = _quote_receipt(prompt, raw.get("source_quote"))
    return {
        f"{prefix}_id": f"{prefix}_{index + 1:03d}",
        "statement": statement,
        "source": source,
    }


def _research_item(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    question = _text(raw.get("question"))
    information_needed = _text(raw.get("information_needed"))
    reason = _text(raw.get("reason"))
    route = _text(raw.get("resolution_route"))
    if not question or not information_needed:
        raise ValueError("PROMPT_STATE_UNRESOLVED: question/information_needed must not be empty")
    if reason not in _UNRESOLVED_REASONS:
        raise ValueError(f"PROMPT_STATE_UNRESOLVED: unsupported reason {reason!r}")
    if route not in _RESOLUTION_ROUTES:
        raise ValueError(f"PROMPT_STATE_UNRESOLVED: unsupported route {route!r}")
    blocks = [
        _text(item)
        for item in raw.get("blocks", [])
        if _text(item)
    ] if isinstance(raw.get("blocks"), Sequence) and not isinstance(raw.get("blocks"), (str, bytes, bytearray)) else []
    sources = [
        _text(item)
        for item in raw.get("source_kinds", [])
        if _text(item) in _SOURCE_KINDS
    ] if isinstance(raw.get("source_kinds"), Sequence) and not isinstance(raw.get("source_kinds"), (str, bytes, bytearray)) else []
    unresolved_id = f"u_{index + 1:03d}"
    return {
        "unresolved_id": unresolved_id,
        "question": question,
        "reason": reason,
        "blocks": list(dict.fromkeys(blocks)),
        "information_needed": information_needed,
        "resolution_route": route,
        "source_kinds": list(dict.fromkeys(sources)),
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


def _build_host_state(prompt: str, model_value: Mapping[str, Any]) -> dict[str, Any]:
    goal_raw = model_value.get("goal")
    if not isinstance(goal_raw, Mapping):
        raise ValueError("PROMPT_STATE_GOAL: goal must be an object")
    goal_statement = _text(goal_raw.get("statement"))
    if not goal_statement:
        raise ValueError("PROMPT_STATE_GOAL: goal statement must not be empty")
    goal = {
        "statement": goal_statement,
        "source": _quote_receipt(prompt, goal_raw.get("source_quote")),
    }

    known_raw = model_value.get("known")
    references_raw = model_value.get("references")
    unresolved_raw = model_value.get("unresolved")
    if not isinstance(known_raw, list) or not isinstance(references_raw, list) or not isinstance(unresolved_raw, list):
        raise ValueError("PROMPT_STATE_SHAPE: known/references/unresolved must be arrays")

    known = [
        _host_item(prompt, item, prefix="known", index=index)
        for index, item in enumerate(known_raw)
        if isinstance(item, Mapping)
    ]
    references = [
        _reference_item(prompt, item, index=index)
        for index, item in enumerate(references_raw)
        if isinstance(item, Mapping)
    ]
    unresolved = [
        _research_item(item, index=index)
        for index, item in enumerate(unresolved_raw)
        if isinstance(item, Mapping)
    ]

    research_queue = [
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
    ]
    scope_status = _text(model_value.get("scope_status"))
    if scope_status not in {"explicit", "partial", "unspecified"}:
        raise ValueError(f"PROMPT_STATE_SCOPE: unsupported scope_status {scope_status!r}")

    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "original_prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "goal": goal,
        "known": known,
        "references": references,
        "scope_status": scope_status,
        "unresolved": unresolved,
        "research_queue": research_queue,
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
    return state


def validate_planning_state(state: Mapping[str, Any], *, prompt: str | None = None) -> None:
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
        str(item.get("unresolved_id") or "")
        for item in unresolved
        if isinstance(item, Mapping)
    }
    for item in queue:
        if not isinstance(item, Mapping):
            raise ValueError("PROMPT_STATE_RESEARCH: research item must be an object")
        resolves = item.get("resolves")
        if not isinstance(resolves, list) or not resolves or any(str(ref) not in unresolved_ids for ref in resolves):
            raise ValueError("PROMPT_STATE_RESEARCH: every research item must resolve a real unresolved item")
        if item.get("queries"):
            raise ValueError(
                "PROMPT_STATE_RESEARCH: initial state cannot invent retrieval queries before research routing"
            )
    if state.get("evidence"):
        raise ValueError("PROMPT_STATE_EVIDENCE: model-authored initial state cannot contain evidence")
    if state.get("plan_ready") is not False:
        raise ValueError("PROMPT_STATE_READY: initial planning state cannot be plan-ready")


def build_initial_planning_state(router: Any, prompt: str) -> dict[str, Any]:
    """Ask the small planner for one bounded prompt-understanding decision."""

    authored = str(prompt or "")
    if not authored.strip():
        raise ValueError("PROMPT_STATE_PROMPT: prompt must not be empty")
    messages = [
        {
            "role": "system",
            "content": (
                "Fill only the prompt-understanding template. Do not design a Minecraft implementation. "
                "Do not invent APIs, files, systems, features, mechanics, or facts absent from the prompt. "
                "KNOWN entries must be directly supported by exact source_quote text. External names, games, "
                "products, styles, works, or concepts that must be understood before deciding scope belong in "
                "references and unresolved(reference_semantics). If the prompt leaves scope open, mark it partial "
                "or unspecified rather than guessing. Unresolved questions describe what information is needed; "
                "they are not implementation decisions."
            ),
        },
        {
            "role": "user",
            "content": "USER REQUEST (verbatim):\n" + authored,
        },
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
    "MODEL_PARAMETERS",
    "MODEL_TOOL",
    "SCHEMA",
    "build_initial_planning_state",
    "validate_planning_state",
]
