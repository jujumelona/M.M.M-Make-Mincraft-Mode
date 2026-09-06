from __future__ import annotations

"""Expose the frozen host request catalog to research and knowledge planning.

Request decomposition, capability IDs, dependency edges, and search queries are owned by
``planning_authority``. This module does not rebuild or refine that catalog. It only makes
the active immutable catalog visible to downstream research/knowledge projections.
"""

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

_INSTALLED = False
_MARKER = "_mmm_approved_scope_downstream_authority_v3"
_MAX_QUERIES_PER_REQUIREMENT = 2
_SPACE = re.compile(r"\s+")
_ASCII_WORD = re.compile(r"[A-Za-z]")
_QUERY_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")


def _active_catalog(prompt: str) -> dict[str, Any] | None:
    from . import planning_authority

    active = planning_authority._ACTIVE_REQUEST_CATALOG.get()
    if active is None or active[0] != prompt:
        return None
    catalog = active[1]
    return deepcopy(catalog) if isinstance(catalog, Mapping) else None


def _query_text(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _is_english_retrieval_query(value: str) -> bool:
    query = _query_text(value)
    if not query or not _ASCII_WORD.search(query):
        return False
    try:
        query.encode("ascii")
    except UnicodeEncodeError:
        return False
    words = _QUERY_WORD.findall(query)
    return 2 <= len(words) <= 24


def _catalog_queries(catalog: Mapping[str, Any]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    requirements = catalog.get("requirements")
    for raw in requirements if isinstance(requirements, list) else []:
        if not isinstance(raw, Mapping):
            continue
        planned = raw.get("search_queries")
        for value in planned if isinstance(planned, list) else []:
            query = _query_text(value)
            key = query.casefold()
            if _is_english_retrieval_query(query) and key not in seen:
                seen.add(key)
                queries.append(query)
    return queries


def _rewrite_pre_design_candidate(prompt: str, candidate: Any) -> Any:
    """Replace raw-request search strings with host-approved catalog queries."""

    if not isinstance(candidate, Mapping):
        return candidate
    catalog = _active_catalog(prompt)
    if catalog is None:
        return dict(candidate)
    queries = _catalog_queries(catalog)
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
            domain["providers"] = list(dict.fromkeys([*providers, "github", "modrinth"]))
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
    """Project the active catalog into research without creating another authority."""

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
    for raw in requirements if isinstance(requirements, list) else []:
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
        queries = list(
            dict.fromkeys(
                query
                for value in planned if isinstance(planned, list)
                if (query := _query_text(value)) and _is_english_retrieval_query(query)
            )
        )[:_MAX_QUERIES_PER_REQUIREMENT]
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
            "request_completeness_owner": "planning_authority",
            "feature_detection_role": "routing_hint_only",
            "authored_requirements_may_be_dropped": False,
            "unknown_authored_requirements": "preserve_for_research",
            "authored_requirement_routing_owner": "planning_authority",
            "catalog_rebuild_after_freeze": False,
            "pre_design_query_owner": "planning_authority",
            "raw_prompt_is_search_query": False,
            "retrieval_queries_per_requirement": _MAX_QUERIES_PER_REQUIREMENT,
        }
    )
    plan["policy"] = policy
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = knowledge_module._nodes._sha({**plan, "plan_sha256": ""})
    knowledge_module.validate_plan(plan)
    return plan


def install() -> None:
    """Bind research/knowledge consumers to the already-frozen planning authority."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import agentic_research_game_design as agentic
    from . import central_research as central
    from . import evidence_obligation_contract as obligations
    from . import minecraft_knowledge_contract as knowledge
    from . import pre_design_research_pipeline as pipeline

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
    "_approved_research_normalize",
    "_compile_knowledge_plan_with_active_catalog",
    "_rewrite_pre_design_candidate",
    "install",
]
