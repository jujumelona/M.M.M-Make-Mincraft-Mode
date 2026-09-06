from __future__ import annotations

"""Resolve planning-state unknowns through the existing grounded RAG evidence path.

Research objectives come from the planning-state SSOT.  A small model may compile a few
retrieval queries for exactly one objective, but it cannot add requirements or treat its
own prose as evidence.  Claims are accepted only through the existing materialized-page
research validator.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .central_research import normalize_research_brief
from .planner_operation import planner_operation
from .planning_state_contract import SCHEMA as PLANNING_STATE_SCHEMA
from .planning_state_contract import validate_planning_state
from .pre_design_domain_research import research_document_domain

_QUERY_TOOL = "submit_research_queries"
_QUERY_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {"type": "string"},
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

_DEFAULT_SCOPE_POLICY = (
    "When authored scope is unspecified, select only an externally evidenced, coherent "
    "end-to-end gameplay slice that preserves the reference's distinctive loop. Never "
    "claim a full clone and never invent unevidenced reference features."
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _query_context(state: Mapping[str, Any], research: Mapping[str, Any]) -> str:
    references = state.get("references")
    reference_names = [
        _text(item.get("name"))
        for item in references
        if isinstance(item, Mapping) and _text(item.get("name"))
    ] if isinstance(references, list) else []
    return json.dumps(
        {
            "objective": research.get("objective"),
            "information_needed": research.get("information_needed"),
            "source_kinds": research.get("source_kinds"),
            "reference_names": reference_names,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _compile_queries(router: Any, state: Mapping[str, Any], research: Mapping[str, Any]) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "Compile retrieval queries for exactly the supplied research objective. "
                "Do not design the mod and do not add features. Preserve named references exactly. "
                "For reference_research, search the referenced subject itself and its documented "
                "systems/behavior; do NOT turn it into a '<name> Minecraft mod' search. For "
                "repository/API implementation research, queries may mention Minecraft only when "
                "the objective itself is already about implementation. Return 1-4 concise queries."
            ),
        },
        {"role": "user", "content": _query_context(state, research)},
    ]
    with planner_operation("research_query_compile", output_tokens=384):
        raw = router.generate_tool_decision(
            "planner",
            messages,
            tool_name=_QUERY_TOOL,
            parameters=_QUERY_PARAMETERS,
            description="Submit bounded retrieval queries for one already-defined information need.",
        )
    values = raw.get("queries") if isinstance(raw, Mapping) else None
    if not isinstance(values, list):
        raise ValueError("PLANNING_RESEARCH_QUERY: query compiler returned no query list")
    queries = list(dict.fromkeys(_text(item) for item in values if _text(item)))
    if not queries:
        raise ValueError("PLANNING_RESEARCH_QUERY: query compiler returned only empty queries")
    return queries[:4]


def _providers_for(source_kinds: Sequence[str]) -> list[str]:
    kinds = set(source_kinds)
    providers: list[str] = []
    if kinds & {"repository", "existing_mods", "web_sources", "reference_sources"}:
        providers.extend(["github", "modrinth", "curseforge"])
    if kinds & {"minecraft_docs", "minecraft_source"}:
        providers.extend(["official_docs", "github"])
    if "project_rag" in kinds:
        providers.append("project_rag")
    return list(dict.fromkeys(providers)) or ["github", "project_rag"]


def _research_brief(router: Any, prompt: str, state: Mapping[str, Any]) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    for raw in state.get("research_queue", []) if isinstance(state.get("research_queue"), list) else []:
        if not isinstance(raw, Mapping) or str(raw.get("status") or "") != "pending":
            continue
        research = dict(raw)
        queries = _compile_queries(router, state, research)
        domains.append(
            {
                "domain_id": str(research.get("research_id") or ""),
                "objective": _text(research.get("objective")),
                "requirements": [_text(research.get("information_needed"))],
                "evidence_kinds": list(research.get("source_kinds") or []),
                "queries": queries,
                "providers": _providers_for(list(research.get("source_kinds") or [])),
                "depends_on": [],
            }
        )
    candidate = {
        "summary": "Resolve only the open information needs in the prompt-first planning state.",
        "domains": domains,
        "unresolved_questions": [
            str(item.get("question") or "")
            for item in state.get("unresolved", [])
            if isinstance(item, Mapping) and item.get("status") == "open"
        ],
    }
    return normalize_research_brief(prompt, {"title": "prompt-state research"}, candidate)


def _domain_note_by_id(notes: Sequence[Mapping[str, Any]], domain_id: str) -> Mapping[str, Any] | None:
    return next(
        (
            note for note in notes
            if str(note.get("domain_id") or "") == domain_id
        ),
        None,
    )


def _evidence_refs(note: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    claims = note.get("claims")
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, Mapping):
            continue
        raw = claim.get("evidence_refs") or claim.get("citations") or claim.get("source_refs")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            refs.extend(_text(item) for item in raw if _text(item))
    return list(dict.fromkeys(refs))


def _apply_scope_policy(state: dict[str, Any], unresolved: dict[str, Any]) -> None:
    unresolved["status"] = "resolved"
    state["resolved"].append(
        {
            "unresolved_id": unresolved["unresolved_id"],
            "resolution": _DEFAULT_SCOPE_POLICY,
            "basis": "host_default_policy",
            "evidence_refs": [],
        }
    )


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def collect_planning_state_research(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Research every non-user-only unresolved item and return an updated state."""

    from . import agentic_research_game_design as agentic
    from . import pre_design_grounded_rag as project_rag
    from .agent_capability_context import target_neutral_research_scope
    from .pre_design_research_pipeline import _grounded_domain_evidence, _validate_document_grounding

    validate_planning_state(state, prompt=prompt)
    value = deepcopy(dict(state))

    # Host policy decisions are explicit state transitions, not pretend research.
    unresolved_by_id = {
        str(item.get("unresolved_id") or ""): item
        for item in value.get("unresolved", [])
        if isinstance(item, dict)
    }
    for unresolved in unresolved_by_id.values():
        if unresolved.get("status") == "open" and unresolved.get("resolution_route") == "default_policy":
            _apply_scope_policy(value, unresolved)

    brief = _research_brief(router, prompt, value)
    if not brief.get("domains"):
        return _rehash(value)

    bundle = project_rag._forced_rag_bundle(router, brief)
    notes: list[dict[str, Any]] = []
    for domain in brief.get("domains", []):
        if not isinstance(domain, Mapping):
            continue
        domain_id = str(domain.get("domain_id") or "")
        grounded = _grounded_domain_evidence(domain_id, bundle)
        document = project_rag._materialize_domain_evidence_document(
            domain_id,
            {"grounded_rag": grounded},
        )
        with target_neutral_research_scope():
            note = research_document_domain(
                agentic,
                project_rag,
                router,
                prompt=prompt,
                domain=domain,
                document=document,
                trace_metadata=trace_metadata,
            )
        _validate_document_grounding(
            agentic,
            project_rag,
            note,
            document,
            domain_id=domain_id,
        )
        notes.append(dict(note))

    for research in value.get("research_queue", []):
        if not isinstance(research, dict) or research.get("status") != "pending":
            continue
        research_id = str(research.get("research_id") or "")
        note = _domain_note_by_id(notes, research_id)
        if note is None:
            research["status"] = "blocked"
            continue
        sufficient = note.get("sufficient") is True and bool(note.get("claims"))
        research["status"] = "complete" if sufficient else "blocked"
        evidence_id = f"e_{len(value['evidence']) + 1:03d}"
        value["evidence"].append(
            {
                "evidence_id": evidence_id,
                "research_ref": research_id,
                "claims": deepcopy(note.get("claims") or []),
                "evidence_refs": _evidence_refs(note),
                "sufficient": sufficient,
                "source": "grounded_materialized_pages",
            }
        )
        for unresolved_id in research.get("resolves", []):
            unresolved = unresolved_by_id.get(str(unresolved_id))
            if unresolved is None:
                continue
            if sufficient:
                unresolved["status"] = "resolved"
                value["resolved"].append(
                    {
                        "unresolved_id": unresolved["unresolved_id"],
                        "resolution": deepcopy(note.get("claims") or []),
                        "basis": "grounded_research",
                        "evidence_refs": _evidence_refs(note),
                    }
                )
            else:
                value["blockers"].append(
                    {
                        "blocker_id": f"b_{len(value['blockers']) + 1:03d}",
                        "unresolved_id": unresolved["unresolved_id"],
                        "statement": "Grounded research did not produce sufficient evidence for this required decision.",
                    }
                )

    return _rehash(value)


__all__ = ["collect_planning_state_research"]
