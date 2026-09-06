from __future__ import annotations

"""Resolve planning-state unknowns with route-appropriate grounded evidence.

The planning-state SSOT owns why information is needed before any retrieval query is
compiled. Reference/world knowledge is retrieved without Minecraft filtering; Minecraft
implementation research uses the existing mod/code/API RAG path. Model prose is never
promoted to evidence without materialized-page grounding and validation.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .central_research import normalize_research_brief
from .planner_operation import planner_operation
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
_REFERENCE_SOURCE_KINDS = frozenset({"reference_sources", "web_sources"})


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
                "Compile retrieval queries for exactly the supplied information need. Do not design the mod, "
                "add features, or guess implementation. Preserve named references exactly. When source_kinds "
                "contains reference_sources or web_sources, search the referenced subject itself and documented "
                "systems/behavior; never turn it into a '<name> Minecraft mod' query. When the need is already "
                "Minecraft implementation research, queries may target Minecraft APIs, source, existing mods, "
                "or code. Return 1-4 concise queries only."
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
    if kinds & _REFERENCE_SOURCE_KINDS:
        return ["wikipedia", "github"]
    providers: list[str] = []
    if kinds & {"repository", "existing_mods"}:
        providers.extend(["github", "modrinth", "curseforge"])
    if kinds & {"minecraft_docs", "minecraft_source"}:
        providers.extend(["official_docs", "github"])
    if "project_rag" in kinds:
        providers.append("project_rag")
    return list(dict.fromkeys(providers)) or ["github", "project_rag"]


def _evidence_kinds_for(source_kinds: Sequence[str]) -> list[str]:
    kinds = set(source_kinds)
    output: list[str] = []
    if kinds & _REFERENCE_SOURCE_KINDS:
        output.append("gameplay_reference")
    if kinds & {"repository", "existing_mods"}:
        output.append("source_code")
    if kinds & {"minecraft_docs", "minecraft_source"}:
        output.append("minecraft_api")
    if "project_rag" in kinds:
        output.append("local_project")
    return list(dict.fromkeys(output)) or ["scholarly_reference"]


def _compile_pending_queries(router: Any, state: dict[str, Any]) -> None:
    """Persist every generated query in the SSOT before retrieval starts."""
    for research in state.get("research_queue", []):
        if not isinstance(research, dict) or str(research.get("status") or "") != "pending":
            continue
        existing = research.get("queries")
        if isinstance(existing, list) and any(_text(item) for item in existing):
            research["queries"] = list(dict.fromkeys(_text(item) for item in existing if _text(item)))[:4]
            continue
        research["queries"] = _compile_queries(router, state, research)


def _research_brief(prompt: str, state: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    domains: list[dict[str, Any]] = []
    reference_domain_ids: set[str] = set()
    for raw in state.get("research_queue", []) if isinstance(state.get("research_queue"), list) else []:
        if not isinstance(raw, Mapping) or str(raw.get("status") or "") != "pending":
            continue
        queries = [
            _text(item) for item in raw.get("queries", []) if _text(item)
        ] if isinstance(raw.get("queries"), list) else []
        if not queries:
            raise ValueError(
                f"PLANNING_RESEARCH_QUERY: pending research {raw.get('research_id')!r} has no compiled queries"
            )
        source_kinds = [str(item) for item in raw.get("source_kinds", [])]
        domain_id = str(raw.get("research_id") or "")
        if set(source_kinds) & _REFERENCE_SOURCE_KINDS:
            reference_domain_ids.add(domain_id)
        domains.append(
            {
                "domain_id": domain_id,
                "objective": _text(raw.get("objective")),
                "requirements": [_text(raw.get("information_needed"))],
                "evidence_kinds": _evidence_kinds_for(source_kinds),
                "queries": queries,
                "providers": _providers_for(source_kinds),
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
    return (
        normalize_research_brief(prompt, {"title": "prompt-state research"}, candidate),
        reference_domain_ids,
    )


def _domain_note_by_id(notes: Sequence[Mapping[str, Any]], domain_id: str) -> Mapping[str, Any] | None:
    return next((note for note in notes if str(note.get("domain_id") or "") == domain_id), None)


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


def _grounded_reference_domain(domain: Mapping[str, Any]) -> dict[str, Any]:
    from .reference_source_research import retrieve_reference_grounded_evidence
    queries = [
        _text(item) for item in domain.get("queries", []) if _text(item)
    ] if isinstance(domain.get("queries"), list) else []
    return retrieve_reference_grounded_evidence(queries)


def collect_planning_state_research(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve every pending host research item using its declared source route."""
    from . import agentic_research_game_design as agentic
    from . import pre_design_grounded_rag as project_rag
    from .agent_capability_context import target_neutral_research_scope
    from .pre_design_research_pipeline import _grounded_domain_evidence, _validate_document_grounding

    validate_planning_state(state, prompt=prompt)
    value = deepcopy(dict(state))
    unresolved_by_id = {
        str(item.get("unresolved_id") or ""): item
        for item in value.get("unresolved", [])
        if isinstance(item, dict)
    }
    for unresolved in unresolved_by_id.values():
        if unresolved.get("status") == "open" and unresolved.get("resolution_route") == "default_policy":
            _apply_scope_policy(value, unresolved)

    _compile_pending_queries(router, value)
    brief, reference_domain_ids = _research_brief(prompt, value)
    if not brief.get("domains"):
        return _rehash(value)

    minecraft_domains = [
        dict(domain)
        for domain in brief.get("domains", [])
        if isinstance(domain, Mapping) and str(domain.get("domain_id") or "") not in reference_domain_ids
    ]
    minecraft_bundle: dict[str, Any] | None = None
    if minecraft_domains:
        minecraft_bundle = project_rag._forced_rag_bundle(
            router,
            {**brief, "domains": minecraft_domains},
        )

    notes: list[dict[str, Any]] = []
    for domain in brief.get("domains", []):
        if not isinstance(domain, Mapping):
            continue
        domain_id = str(domain.get("domain_id") or "")
        if domain_id in reference_domain_ids:
            grounded = _grounded_reference_domain(domain)
        else:
            if minecraft_bundle is None:
                raise ValueError("PLANNING_RESEARCH_ROUTE: Minecraft RAG bundle is unexpectedly absent")
            grounded = _grounded_domain_evidence(domain_id, minecraft_bundle)
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
        refs = _evidence_refs(note)
        sufficient = note.get("sufficient") is True and bool(note.get("claims")) and bool(refs)
        research["status"] = "complete" if sufficient else "blocked"
        value["evidence"].append(
            {
                "evidence_id": f"e_{len(value['evidence']) + 1:03d}",
                "research_ref": research_id,
                "claims": deepcopy(note.get("claims") or []),
                "evidence_refs": refs,
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
                        "evidence_refs": refs,
                    }
                )
            else:
                value["blockers"].append(
                    {
                        "blocker_id": f"b_{len(value['blockers']) + 1:03d}",
                        "unresolved_id": unresolved["unresolved_id"],
                        "statement": "Grounded research did not produce sufficient cited evidence for this required decision.",
                    }
                )

    return _rehash(value)


__all__ = ["collect_planning_state_research"]
