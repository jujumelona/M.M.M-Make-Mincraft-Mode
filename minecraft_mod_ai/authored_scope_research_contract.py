from __future__ import annotations

"""Bind the approved authored requirement graph to research without reinterpreting raw text.

The authoritative semantic catalog owns user-intent decomposition.  This module enriches
that frozen catalog with two things the retrieval layer actually needs:

* authored/logically-required dependency edges between approved requirements;
* English multi-query retrieval plans per requirement.

The raw user prompt remains provenance.  It is never used as an external search query.
"""

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

_INSTALLED = False
_MARKER = "_mmm_approved_scope_downstream_authority_v2"
_RETRIEVAL_MARKER = "_mmm_requirement_retrieval_plan_v1"
_SPACE = re.compile(r"\s+")
_ASCII_WORD = re.compile(r"[A-Za-z]")
_QUERY_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")


def _active_catalog(prompt: str) -> dict[str, Any] | None:
    from . import evidence_request_guard as guard

    active = guard._ACTIVE_REQUEST_CATALOG.get()
    if active is None or active[0] != prompt:
        return None
    catalog = active[1]
    return deepcopy(catalog) if isinstance(catalog, Mapping) else None


def _query_text(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _is_english_retrieval_query(value: str) -> bool:
    """Accept concise ASCII/English search queries, never raw non-English prompt text."""

    query = _query_text(value)
    if not query or not _ASCII_WORD.search(query):
        return False
    try:
        query.encode("ascii")
    except UnicodeEncodeError:
        return False
    words = _QUERY_WORD.findall(query)
    return 2 <= len(words) <= 24


def _semantic_interpretation_messages(
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Legacy helper retained for compatibility; semantic authority owns live prompts."""

    system = (
        "Interpret the supplied authored Minecraft-mod request semantically, even when it "
        "contains typos, missing punctuation, repeated spaces, shorthand, or a language "
        "different from English. The host owns the original UTF-8 text and exact offsets: "
        "never rewrite provenance, but normalize the MEANING in semantic_statement and "
        "capability_id. Return every independently observable player-facing behavior. "
        "Split long run-on clauses by actions, state changes, resource flows, construction, "
        "purchases, upgrades, travel stages, encounters, exploration, and progression. "
        "Do not add an umbrella requirement merely because several concrete leaves share a "
        "theme; add a mode/integration requirement only when that mode itself has distinct "
        "observable behavior. Preserve authored causal and temporal structure inside "
        "Given/When/Then: if the request says one outcome enables another, make the latter "
        "behavior's Given condition state that prerequisite clearly. This semantic pass does "
        "not emit graph edges; a later host-bound planner derives edges only between these "
        "approved requirements. capability_id must be a concise lower-case dotted English "
        "gameplay semantic identifier, not a translation of a typo and not an implementation "
        "primitive. source_anchor must remain a short locator supported by the ORIGINAL host "
        "clause, including its original language/typo when necessary for grounding. Do not "
        "choose APIs, classes, networking, persistence, UI architecture, libraries, donor "
        "mods, or other implementation alternatives unless explicitly authored. Return one "
        "concrete Given/When/Then contract per independent behavior. Do not omit behavior "
        "just because the request has no punctuation."
    )
    payload = {
        "host_owned_clauses": [
            {
                "source_clause_index": int(clause["clause_index"]),
                "text": str(clause["text"]),
            }
            for clause in clauses
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _retrieval_plan_schema(requirement_ids: Sequence[str]) -> dict[str, Any]:
    ids = list(dict.fromkeys(str(item) for item in requirement_ids if str(item)))
    return {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement_id": {"type": "string", "enum": ids},
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string", "enum": ids},
                            "uniqueItems": True,
                            "maxItems": max(0, len(ids) - 1),
                        },
                        "search_queries": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 5,
                            "items": {"type": "string", "minLength": 4, "maxLength": 180},
                            "uniqueItems": True,
                        },
                    },
                    "required": ["requirement_id", "depends_on", "search_queries"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["requirements"],
        "additionalProperties": False,
    }


def _retrieval_plan_messages(
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    frozen = []
    for raw in requirements:
        span = raw.get("source_span")
        frozen.append(
            {
                "requirement_id": str(raw.get("requirement_id") or ""),
                "capability": str(raw.get("capability") or ""),
                "semantic_statement": str(raw.get("semantic_statement") or ""),
                "observable_behavior": dict(raw.get("observable_behavior") or {})
                if isinstance(raw.get("observable_behavior"), Mapping)
                else {},
                "source_text": str(span.get("text") or "")
                if isinstance(span, Mapping)
                else "",
            }
        )
    system = (
        "Plan retrieval for an already-approved Minecraft-mod requirement graph. "
        "Do not add, delete, merge, rename, or reinterpret requirements. "
        "For depends_on, reference another supplied requirement only when the authored "
        "behavior or its observable Given/When/Then makes that requirement a genuine "
        "player-visible prerequisite. Preserve explicit authored ordering such as build "
        "before launch, launch before planet exploration, or earning currency before a "
        "purchase when the supplied requirements actually state that relationship. "
        "Do not invent implementation/API dependencies. "
        "For search_queries, rewrite each semantic requirement into 2-5 concise ENGLISH "
        "information-retrieval queries suitable for public Minecraft mod ecosystems, "
        "GitHub source search, and implementation/reference discovery. Translate the "
        "meaning; do not copy the user's raw sentence, typo, spacing, or language. "
        "Use concrete mechanic nouns and useful ecosystem terms such as minecraft, mod, "
        "source, implementation, vehicle, economy, planet, colony, NPC, crafting, etc. "
        "Do not fabricate a specific mod/project name. Queries must describe what evidence "
        "to find, not what answer to generate."
    )
    payload = {
        "authoritative_request_for_context_only": prompt,
        "approved_requirements": frozen,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _call_retrieval_planner(
    router: Any,
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
) -> Any:
    """Build query structure deterministically; the small model owns no JSON protocol."""
    del router, prompt
    from .canonical_capability_ontology import search_queries_for_capability

    rows: list[dict[str, Any]] = []
    for item in requirements:
        rid = str(item.get("requirement_id") or "").strip()
        capability = str(item.get("capability") or "").strip()
        if not rid:
            continue
        raw_deps = item.get("depends_on")
        deps = (
            [str(dep).strip() for dep in raw_deps if str(dep).strip() and str(dep).strip() != rid]
            if isinstance(raw_deps, list)
            else []
        )
        queries = list(search_queries_for_capability(capability)) if capability else []
        concept = re.sub(r"[^A-Za-z0-9]+", " ", capability.replace("_", " ")).strip()
        if not concept:
            semantic = str(item.get("semantic_statement") or "")
            concept = " ".join(_QUERY_WORD.findall(semantic))[:120].strip()
        if not concept:
            concept = "requested minecraft mechanic"
        queries.extend((
            f"minecraft mod {concept} implementation",
            f"minecraft fabric {concept} source",
        ))
        cleaned: list[str] = []
        for query in queries:
            value = _query_text(query)
            if value and "minecraft" not in value.casefold():
                value = f"minecraft {value}"
            if _is_english_retrieval_query(value) and value.casefold() not in {q.casefold() for q in cleaned}:
                cleaned.append(value)
            if len(cleaned) >= 5:
                break
        if len(cleaned) < 2:
            raise ValueError(f"host retrieval planner could not build two queries for {rid}")
        rows.append({
            "requirement_id": rid,
            "depends_on": list(dict.fromkeys(deps)),
            "search_queries": cleaned,
        })
    return {"requirements": rows}


def _validate_dependency_dag(
    dependencies: Mapping[str, Sequence[str]],
    known: set[str],
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError("requirement retrieval dependency graph contains a cycle")
        visiting.add(node)
        for dep in dependencies.get(node, ()):
            if dep not in known:
                raise ValueError(f"unknown dependency requirement id: {dep}")
            if dep == node:
                raise ValueError(f"requirement depends on itself: {node}")
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for node in known:
        visit(node)


def _normalize_retrieval_plan(
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
    payload: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("retrieval query planner returned a non-object payload")
    raw_items = payload.get("requirements")
    if not isinstance(raw_items, list):
        raise ValueError("retrieval query planner omitted requirements")

    known = {
        str(item.get("requirement_id") or "")
        for item in requirements
        if str(item.get("requirement_id") or "")
    }
    source_text_by_id = {}
    for item in requirements:
        rid = str(item.get("requirement_id") or "")
        span = item.get("source_span")
        source_text_by_id[rid] = (
            _query_text(span.get("text")) if isinstance(span, Mapping) else ""
        )

    result: dict[str, dict[str, Any]] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    prompt_key = _query_text(prompt).casefold()

    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ValueError("retrieval query planner emitted a non-object requirement")
        rid = str(raw.get("requirement_id") or "")
        if rid not in known or rid in result:
            raise ValueError(f"invalid/duplicate retrieval requirement id: {rid!r}")

        raw_deps = raw.get("depends_on")
        if not isinstance(raw_deps, list):
            raise ValueError(f"depends_on must be a list for {rid}")
        deps = tuple(
            dict.fromkeys(str(dep) for dep in raw_deps if str(dep) and str(dep) != rid)
        )
        if any(dep not in known for dep in deps):
            raise ValueError(f"unknown dependency in retrieval plan for {rid}")

        raw_queries = raw.get("search_queries")
        if not isinstance(raw_queries, list):
            raise ValueError(f"search_queries must be a list for {rid}")
        queries: list[str] = []
        source_key = source_text_by_id.get(rid, "").casefold()
        for raw_query in raw_queries:
            query = _query_text(raw_query)
            key = query.casefold()
            if not query or key in {prompt_key, source_key}:
                continue
            if not _is_english_retrieval_query(query):
                continue
            if key not in {item.casefold() for item in queries}:
                queries.append(query)
        if len(queries) < 2:
            raise ValueError(
                f"retrieval query planner produced fewer than two English queries for {rid}"
            )
        dependencies[rid] = deps
        result[rid] = {
            "depends_on": list(deps),
            "search_queries": queries,
        }

    if set(result) != known:
        missing = sorted(known - set(result))
        raise ValueError(f"retrieval query planner omitted requirements: {missing}")
    _validate_dependency_dag(dependencies, known)
    return result


def _enrich_catalog_with_retrieval_plan(
    prompt: str,
    catalog: Mapping[str, Any],
    router: Any,
) -> dict[str, Any]:
    from . import evidence_first_planning as evidence

    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return dict(catalog)

    plan_payload = _call_retrieval_planner(router, prompt, requirements)
    plan = _normalize_retrieval_plan(prompt, requirements, plan_payload)

    enriched = deepcopy(dict(catalog))
    enriched_requirements = []
    edges: list[list[str]] = []
    for raw in enriched.get("requirements", []):
        item = dict(raw)
        rid = str(item.get("requirement_id") or "")
        planned = plan[rid]
        item["depends_on"] = list(planned["depends_on"])
        item["search_queries"] = list(planned["search_queries"])
        for dep in item["depends_on"]:
            edges.append([dep, rid])
        enriched_requirements.append(item)
    enriched["requirements"] = enriched_requirements
    enriched["requirement_graph"] = {
        "node_ids": [str(item.get("requirement_id") or "") for item in enriched_requirements],
        "edges": edges,
    }
    audit = dict(enriched.get("semantic_audit") or {})
    audit["normal_model_turns"] = int(audit.get("normal_model_turns") or 1) + 1
    audit["retrieval_query_planning"] = "english_multi_query_rewrite"
    audit["dependency_edge_count"] = len(edges)
    enriched["semantic_audit"] = audit
    enriched["catalog_sha256"] = ""
    enriched["catalog_sha256"] = evidence._hash_without(enriched, "catalog_sha256")
    return enriched


def _rewrite_pre_design_candidate(prompt: str, candidate: Any) -> Any:
    """Route approved English queries to explicit public evidence retrievers."""

    if not isinstance(candidate, Mapping):
        return candidate
    catalog = _active_catalog(prompt)
    if catalog is None:
        return dict(candidate)
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list):
        return dict(candidate)

    queries: list[str] = []
    seen: set[str] = set()
    for raw in requirements:
        if not isinstance(raw, Mapping):
            continue
        planned = raw.get("search_queries")
        if not isinstance(planned, list):
            continue
        for value in planned:
            query = _query_text(value)
            key = query.casefold()
            if _is_english_retrieval_query(query) and key not in seen:
                seen.add(key)
                queries.append(query)
    if not queries:
        return dict(candidate)

    rewritten = deepcopy(dict(candidate))
    raw_domains = rewritten.get("domains")
    if not isinstance(raw_domains, list):
        return rewritten
    domains: list[Any] = []
    for raw in raw_domains:
        if not isinstance(raw, Mapping):
            domains.append(raw)
            continue
        domain = dict(raw)
        if str(domain.get("domain_id") or "") == "request":
            domain["queries"] = list(queries)
            raw_providers = domain.get("providers")
            providers = (
                [str(item).strip() for item in raw_providers if str(item).strip()]
                if isinstance(raw_providers, Sequence)
                and not isinstance(raw_providers, (str, bytes, bytearray))
                else []
            )
            domain["providers"] = list(
                dict.fromkeys([*providers, "github", "modrinth"])
            )
        domains.append(domain)
    rewritten["domains"] = domains
    return rewritten


def _approved_research_normalize(
    obligation_module: Any,
    previous_normalize: Any,
    prompt: str,
    game_design: dict[str, Any],
    candidate: Any | None = None,
) -> dict[str, Any]:
    """Preserve pre-design query planning; expand obligations only after design."""

    if (
        candidate is not None
        and isinstance(game_design, Mapping)
        and set(game_design) == {"title"}
        and game_design.get("title") == "pre-design research"
    ):
        return previous_normalize(
            prompt,
            game_design,
            _rewrite_pre_design_candidate(prompt, candidate),
        )

    catalog = obligation_module._catalog_for(prompt)
    if catalog is None:
        catalog = _active_catalog(prompt)
    if catalog is not None:
        return obligation_module.build_evidence_obligation_brief(
            prompt,
            catalog,
            game_design,
        )
    return previous_normalize(prompt, game_design, candidate)


def _compile_knowledge_plan_with_active_catalog(
    knowledge_module: Any,
    previous_compile: Any,
    prompt: str,
    game_design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile technical hints around the frozen catalog instead of reconstructing scope."""

    catalog = _active_catalog(prompt)
    if catalog is None:
        return previous_compile(prompt, game_design)

    design = dict(game_design or {})
    plan = dict(knowledge_module._base_compile_minecraft_knowledge_plan(prompt, design))
    plan["authored_request_catalog"] = deepcopy(catalog)
    plan["authored_requirements"] = knowledge_module._authored_requirement_lifecycle(catalog)

    requirements = catalog.get("requirements")
    routes: list[dict[str, Any]] = []
    if isinstance(requirements, list):
        from .canonical_capability_ontology import search_queries_for_capability

        for raw in requirements:
            if not isinstance(raw, Mapping):
                continue
            capability = str(raw.get("capability") or "").strip()
            source = raw.get("source_span")
            source_text = (
                str(source.get("text") or "").strip()
                if isinstance(source, Mapping)
                else ""
            )
            semantic = str(
                raw.get("semantic_statement") or raw.get("statement") or source_text
            ).strip()
            planned = raw.get("search_queries")
            rewritten = (
                [_query_text(item) for item in planned if _query_text(item)]
                if isinstance(planned, list)
                else []
            )
            ontology_queries = list(search_queries_for_capability(capability)) if capability else []
            selected_queries = rewritten if rewritten else ontology_queries
            queries = list(
                dict.fromkeys(
                    query
                    for query in selected_queries
                    if query and _is_english_retrieval_query(query)
                )
            )
            routes.append(
                {
                    "requirement_id": str(raw.get("requirement_id") or ""),
                    "capability": capability,
                    "source_text": source_text,
                    "semantic_statement": semantic,
                    "depends_on": list(raw.get("depends_on") or []),
                    "research_queries": queries,
                }
            )
    plan["authored_capability_routes"] = routes

    policy = dict(plan.get("policy", {}))
    policy.update(
        {
            "request_completeness_owner": "evidence_request_catalog",
            "feature_detection_role": "routing_hint_only",
            "authored_requirements_may_be_dropped": False,
            "unknown_authored_requirements": "preserve_for_research",
            "authored_requirement_routing_owner": "approved_requirement_graph",
            "catalog_rebuild_after_freeze": False,
            "pre_design_query_owner": "approved_requirement_retrieval_plan",
            "raw_prompt_is_search_query": False,
        }
    )
    plan["policy"] = policy
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = knowledge_module._nodes._sha({**plan, "plan_sha256": ""})
    knowledge_module.validate_plan(plan)
    return plan


def _approved_pre_design_brief(prompt: str) -> dict[str, Any]:
    """Legacy helper retained for compatibility; the pipeline owns the live phase."""

    from . import minecraft_knowledge_contract as knowledge
    from . import pre_design_research_pipeline as pipeline

    plan = knowledge.compile_minecraft_knowledge_plan(prompt)
    routes = plan.get("authored_capability_routes")
    if not isinstance(routes, list) or not routes:
        return pipeline._pre_design_brief(prompt)

    candidate = pipeline._pre_design_brief(prompt)
    return _rewrite_pre_design_candidate(prompt, candidate)


def install() -> None:
    """Bind the approved graph into semantic planning, research routing, and knowledge plans."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import agentic_research_game_design as agentic
    from . import central_research as central
    from . import evidence_obligation_contract as obligations
    from . import evidence_request_guard as guard
    from . import minecraft_knowledge_contract as knowledge
    from . import pre_design_research_pipeline as pipeline
    from . import semantic_requirement_authority as semantic

    current_builder = guard.build_authoritative_request_catalog
    if not getattr(current_builder, _RETRIEVAL_MARKER, False):
        def build_catalog(prompt: str, router: Any | None = None) -> dict[str, Any]:
            catalog = current_builder(prompt, router=router)
            if (
                router is None
                or not isinstance(catalog, Mapping)
                or catalog.get("schema_version") != "mmm/approved-requirement-graph-v1"
            ):
                return dict(catalog)
            enriched = _enrich_catalog_with_retrieval_plan(prompt, catalog, router)
            semantic.validate_approved_requirement_catalog(enriched, prompt=prompt)
            return enriched

        setattr(build_catalog, _RETRIEVAL_MARKER, True)
        build_catalog.__wrapped__ = current_builder  # type: ignore[attr-defined]
        guard.build_authoritative_request_catalog = build_catalog

    current_normalize = central.normalize_research_brief
    if not getattr(current_normalize, _MARKER, False):
        def normalize(
            prompt: str,
            game_design: dict[str, Any],
            candidate: Any | None = None,
        ) -> dict[str, Any]:
            return _approved_research_normalize(
                obligations,
                current_normalize,
                prompt,
                game_design,
                candidate,
            )

        setattr(normalize, _MARKER, True)
        normalize.__wrapped__ = current_normalize  # type: ignore[attr-defined]
        central.normalize_research_brief = normalize
        agentic.normalize_research_brief = normalize
        pipeline.normalize_research_brief = normalize

    current_compile = knowledge.compile_minecraft_knowledge_plan
    if not getattr(current_compile, _MARKER, False):
        def compile_plan(
            prompt: str,
            game_design: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            return _compile_knowledge_plan_with_active_catalog(
                knowledge,
                current_compile,
                prompt,
                game_design,
            )

        setattr(compile_plan, _MARKER, True)
        compile_plan.__wrapped__ = current_compile  # type: ignore[attr-defined]
        knowledge.compile_minecraft_knowledge_plan = compile_plan
        pipeline.compile_minecraft_knowledge_plan = compile_plan

    _INSTALLED = True


__all__ = [
    "_active_catalog",
    "_approved_pre_design_brief",
    "_approved_research_normalize",
    "_compile_knowledge_plan_with_active_catalog",
    "_enrich_catalog_with_retrieval_plan",
    "_rewrite_pre_design_candidate",
    "install",
]