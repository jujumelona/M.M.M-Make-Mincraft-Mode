from __future__ import annotations

"""Turn resolved prompt/reference knowledge into research-backed implementation requirements.

This is the second bounded planner template. It runs only after initial unknowns were
resolved. The model selects user-visible requirements from authored text and grounded
reference evidence; host code validates every provenance reference and automatically
creates implementation-research obligations before any code plan may become ready.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .planner_operation import planner_operation
from .planning_state_contract import validate_planning_state

_REQUIREMENT_TOOL = "submit_researched_requirements"
_REQUIREMENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "prompt_quote": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "acceptance": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["statement", "prompt_quote", "evidence_refs", "acceptance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["requirements"],
    "additionalProperties": False,
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _known_evidence_refs(state: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for evidence in state.get("evidence", []) if isinstance(state.get("evidence"), list) else []:
        if not isinstance(evidence, Mapping) or evidence.get("sufficient") is not True:
            continue
        refs.update(
            _text(item)
            for item in evidence.get("evidence_refs", [])
            if _text(item)
        )
    return refs


def _resolved_context(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "goal": deepcopy(state.get("goal")),
        "known": deepcopy(state.get("known", [])),
        "references": deepcopy(state.get("references", [])),
        "scope_status": state.get("scope_status"),
        "resolved": deepcopy(state.get("resolved", [])),
        "evidence": [
            deepcopy(item)
            for item in state.get("evidence", [])
            if isinstance(item, Mapping) and item.get("sufficient") is True
        ],
    }


def _next_id(items: Sequence[Mapping[str, Any]], key: str, prefix: str) -> str:
    maximum = 0
    for item in items:
        raw = str(item.get(key) or "")
        if raw.startswith(prefix):
            try:
                maximum = max(maximum, int(raw.removeprefix(prefix)))
            except ValueError:
                pass
    return f"{prefix}{maximum + 1:03d}"


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def compile_researched_requirements(router: Any, prompt: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Add requirement decisions plus implementation-research work to the SSOT state."""

    validate_planning_state(state, prompt=prompt)
    open_blocking = [
        item for item in state.get("unresolved", [])
        if isinstance(item, Mapping)
        and item.get("status") == "open"
        and item.get("resolution_route") != "user_only"
    ]
    if open_blocking:
        raise ValueError(
            "PLANNING_REQUIREMENTS_BLOCKED: prompt/reference research is still unresolved: "
            + ", ".join(str(item.get("unresolved_id") or "") for item in open_blocking)
        )

    messages = [
        {
            "role": "system",
            "content": (
                "Compile the user-visible requirements that the implementation must satisfy. "
                "Use only the authored prompt and supplied grounded evidence. Do not invent Minecraft APIs, "
                "files, classes, registrations, or implementation architecture. For a reference-driven request, "
                "use only externally evidenced reference systems within the resolved host scope policy. "
                "Each requirement must be independently testable at the player-facing behavior level. "
                "prompt_quote must be an exact substring when the requirement is directly authored; otherwise "
                "use an empty string and cite one or more evidence_refs. Every non-authored requirement must cite "
                "evidence. Do not output implementation steps."
            ),
        },
        {"role": "user", "content": str(_resolved_context(state))},
    ]
    with planner_operation("researched_requirement_compile", output_tokens=2048):
        raw = router.generate_tool_decision(
            "planner",
            messages,
            tool_name=_REQUIREMENT_TOOL,
            parameters=_REQUIREMENT_PARAMETERS,
            description="Submit only evidence-backed, player-visible requirements for the resolved request scope.",
        )
    raw_requirements = raw.get("requirements") if isinstance(raw, Mapping) else None
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise ValueError("PLANNING_REQUIREMENTS_EMPTY: planner returned no researched requirements")

    allowed_refs = _known_evidence_refs(state)
    value = deepcopy(dict(state))
    requirement_decisions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_requirements, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("PLANNING_REQUIREMENT_SHAPE: requirement must be an object")
        statement = _text(item.get("statement"))
        quote = str(item.get("prompt_quote") or "").strip()
        refs = list(dict.fromkeys(_text(ref) for ref in item.get("evidence_refs", []) if _text(ref)))
        acceptance = list(dict.fromkeys(_text(check) for check in item.get("acceptance", []) if _text(check)))
        if not statement or not acceptance:
            raise ValueError("PLANNING_REQUIREMENT_CONTENT: statement and acceptance are required")
        if quote and quote not in prompt:
            raise ValueError(f"PLANNING_REQUIREMENT_SOURCE: prompt_quote is not authored text: {quote!r}")
        unknown_refs = [ref for ref in refs if ref not in allowed_refs]
        if unknown_refs:
            raise ValueError(
                "PLANNING_REQUIREMENT_EVIDENCE: requirement cited unknown evidence refs: "
                + ", ".join(unknown_refs)
            )
        if not quote and not refs:
            raise ValueError(
                "PLANNING_REQUIREMENT_PROVENANCE: every derived requirement needs grounded evidence"
            )
        requirement_decisions.append(
            {
                "requirement_id": f"req_{index:03d}",
                "statement": statement,
                "prompt_quote": quote,
                "evidence_refs": refs,
                "acceptance": acceptance,
                "status": "implementation_research_pending",
            }
        )

    value["decisions"] = [
        item for item in value.get("decisions", [])
        if not (isinstance(item, Mapping) and item.get("decision_type") == "requirement")
    ]
    for requirement in requirement_decisions:
        value["decisions"].append(
            {
                "decision_id": f"d_{len(value['decisions']) + 1:03d}",
                "decision_type": "requirement",
                **deepcopy(requirement),
            }
        )
        unresolved_id = _next_id(value.get("unresolved", []), "unresolved_id", "u_")
        research_id = _next_id(value.get("research_queue", []), "research_id", "r_")
        value["unresolved"].append(
            {
                "unresolved_id": unresolved_id,
                "question": f"How can this requirement be implemented correctly for the resolved Minecraft target: {requirement['statement']}",
                "reason": "implementation_method",
                "blocks": [requirement["requirement_id"], "implementation_plan"],
                "information_needed": (
                    "Verified reusable mod/code patterns, Minecraft API/source behavior, required artifacts, "
                    "dependencies, and verification obligations for: " + requirement["statement"]
                ),
                "resolution_route": "implementation_research",
                "source_kinds": [
                    "repository", "existing_mods", "minecraft_docs", "minecraft_source", "project_rag"
                ],
                "status": "open",
                "research_ref": research_id,
                "requirement_ref": requirement["requirement_id"],
            }
        )
        value["research_queue"].append(
            {
                "research_id": research_id,
                "resolves": [unresolved_id],
                "requirement_ref": requirement["requirement_id"],
                "objective": f"Find evidence-backed implementation/reuse options for {requirement['statement']}",
                "information_needed": (
                    "Concrete implementation patterns and all required support artifacts/dependencies, "
                    "without inventing APIs or assuming prompt vocabulary matches code vocabulary."
                ),
                "source_kinds": [
                    "repository", "existing_mods", "minecraft_docs", "minecraft_source", "project_rag"
                ],
                "queries": [],
                "status": "pending",
            }
        )

    value["plan_ready"] = False
    return _rehash(value)


__all__ = ["compile_researched_requirements"]
