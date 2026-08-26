from __future__ import annotations

"""Adaptive pre-design RAG routing for the frozen small-model runtime.

The research brief already classifies each domain by provider and evidence kind. Use
that host-owned routing graph instead of querying every retriever for every query.
Code RAG is reserved for explicit local/source-code evidence and is used as a
corrective fallback when the authoritative catalog has no usable source.
"""

import copy
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any

_MARKER = "_mmm_adaptive_pre_design_rag_v1"

_CATALOG_EVIDENCE_KINDS = frozenset(
    {
        "minecraft_api",
        "dependency",
        "gameplay_reference",
        "compatibility",
        "runtime_behavior",
        "performance",
        "accessibility",
        "testing",
        "release",
        "ai_inference",
        "agent_tool_use",
        "model_runtime",
        "model_license",
        "latency_budget",
    }
)
_CODE_EVIDENCE_KINDS = frozenset({"source_code", "local_project"})


def _strings(value: Any) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(item).strip() for item in value if str(item).strip())


def _route(domain: Mapping[str, Any], *, code_index_available: bool) -> dict[str, Any]:
    """Select retrievers from validated domain metadata, with a compatibility fallback."""

    providers_present = isinstance(domain.get("providers"), list)
    kinds_present = isinstance(domain.get("evidence_kinds"), list)
    providers = _strings(domain.get("providers"))
    kinds = _strings(domain.get("evidence_kinds"))

    # Hand-written/test briefs from the pre-routing API do not carry route metadata.
    # Preserve the historical behavior for them rather than silently reducing coverage.
    if not (providers_present and kinds_present):
        return {
            "catalog": True,
            "code": bool(code_index_available),
            "providers": providers,
            "evidence_kinds": kinds,
            "reason": "legacy_brief_compatibility",
        }

    if "project_rag" not in providers:
        return {
            "catalog": False,
            "code": False,
            "providers": providers,
            "evidence_kinds": kinds,
            "reason": "project_rag_not_routed",
        }

    catalog = bool(kinds & _CATALOG_EVIDENCE_KINDS)
    code = bool(code_index_available and kinds & _CODE_EVIDENCE_KINDS)
    return {
        "catalog": catalog,
        "code": code,
        "providers": providers,
        "evidence_kinds": kinds,
        "reason": "evidence_kind_route" if (catalog or code) else "no_matching_retriever",
    }


def _skipped_project(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "mmm/forced-project-rag-query-v1",
        "status": "skipped",
        "sources": [],
        "errors": [],
        "skip_reason": reason,
    }


def _skipped_code(index_path: Path | None, reason: str) -> dict[str, Any]:
    if index_path is None:
        return {
            "schema_version": "mmm/forced-code-rag-query-v1",
            "status": "not_indexed",
            "hits": [],
        }
    return {
        "schema_version": "mmm/forced-code-rag-query-v1",
        "status": "skipped",
        "hits": [],
        "skip_reason": reason,
    }


def _has_project_sources(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value.get("sources"))


def _has_code_hits(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value.get("hits"))


def _job_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity for one semantically equivalent retrieval execution."""

    route = job["route"]
    return (
        str(job["query"]),
        bool(route["catalog"]),
        bool(route["code"]),
        tuple(sorted(route["providers"])),
        tuple(sorted(route["evidence_kinds"])),
        str(route["reason"]),
    )


def harden(pre_design_module: Any, small_model_module: Any) -> None:
    """Replace blanket project/code fan-out with routed retrieval and correction."""

    current = pre_design_module._forced_rag_bundle
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def adaptive_forced_rag(router: Any, research_brief: Mapping[str, Any]) -> dict[str, Any]:
        raw_domains = research_brief.get("domains")
        domains = [item for item in raw_domains or [] if isinstance(item, Mapping)]
        versions = pre_design_module._research_versions(router)
        code_index = pre_design_module._existing_code_index()

        jobs: list[dict[str, Any]] = []
        unique_jobs: dict[tuple[Any, ...], dict[str, Any]] = {}
        for domain in domains:
            domain_id = str(domain.get("domain_id", "")).strip()
            queries = domain.get("queries")
            if not domain_id or not isinstance(queries, list):
                continue
            route = _route(domain, code_index_available=code_index is not None)
            for query in queries:
                query_text = str(query).strip()
                if not query_text:
                    continue
                job = {
                    "index": len(jobs),
                    "domain_id": domain_id,
                    "query": query_text,
                    "route": route,
                }
                jobs.append(job)
                unique_jobs.setdefault(_job_key(job), job)

        def search_code(query: str) -> dict[str, Any]:
            context = getattr(small_model_module, "_RAG_ROUTER", None)
            if context is None or not hasattr(context, "set") or not hasattr(context, "reset"):
                return pre_design_module._search_code_index(code_index, query)
            token = context.set(router)
            try:
                return pre_design_module._search_code_index(code_index, query)
            finally:
                context.reset(token)

        def run(job: Mapping[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            query = str(job["query"])
            route = job["route"]
            planned_catalog = bool(route["catalog"])
            planned_code = bool(route["code"])

            project_result = _skipped_project(str(route["reason"]))
            code_result = _skipped_code(code_index, str(route["reason"]))
            executed: list[str] = []
            expansion = ""

            if planned_catalog:
                project_result = pre_design_module._search_authoritative_catalog(
                    query, versions
                )
                executed.append("project_catalog")
            if planned_code:
                code_result = search_code(query)
                executed.append("code_rag")

            # CRAG-style correction: widen only after the selected lane fails to
            # produce evidence. API/reference domains may consult local code on an
            # authoritative-catalog miss; mixed code/API domains already run both.
            if (
                planned_catalog
                and not planned_code
                and code_index is not None
                and not _has_project_sources(project_result)
            ):
                code_result = search_code(query)
                executed.append("code_rag")
                expansion = "code_on_catalog_miss"
            elif (
                planned_code
                and not planned_catalog
                and bool(route["evidence_kinds"] & _CATALOG_EVIDENCE_KINDS)
                and not _has_code_hits(code_result)
            ):
                project_result = pre_design_module._search_authoritative_catalog(
                    query, versions
                )
                executed.append("project_catalog")
                expansion = "catalog_on_code_miss"

            result = {
                "query": query,
                "query_sha256": pre_design_module._sha256_text(query),
                "project_rag": project_result,
                "code_rag": code_result,
                "retrieval_route": {
                    "schema_version": "mmm/adaptive-pre-design-rag-route-v1",
                    "policy": "provider_and_evidence_kind_then_correct_on_empty",
                    "providers": sorted(route["providers"]),
                    "evidence_kinds": sorted(route["evidence_kinds"]),
                    "planned": [
                        name
                        for enabled, name in (
                            (planned_catalog, "project_catalog"),
                            (planned_code, "code_rag"),
                        )
                        if enabled
                    ],
                    "executed": executed,
                    "expansion": expansion,
                    "reason": str(route["reason"]),
                },
            }
            return _job_key(job), result

        executed_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        unique_values = list(unique_jobs.values())
        search_jobs = [
            job
            for job in unique_values
            if bool(job["route"]["catalog"] or job["route"]["code"])
        ]
        skipped_jobs = [
            job
            for job in unique_values
            if not bool(job["route"]["catalog"] or job["route"]["code"])
        ]

        for job in skipped_jobs:
            key, result = run(job)
            executed_by_key[key] = result

        if len(search_jobs) == 1:
            key, result = run(search_jobs[0])
            executed_by_key[key] = result
        elif search_jobs:
            worker_count = min(8, len(search_jobs))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="mmm_adaptive_pre_design_rag",
            ) as pool:
                for key, result in pool.map(run, search_jobs):
                    executed_by_key[key] = result

        by_domain: dict[str, list[dict[str, Any]]] = {
            str(item.get("domain_id", "")): [] for item in domains
        }
        for job in jobs:
            result = executed_by_key[_job_key(job)]
            by_domain.setdefault(str(job["domain_id"]), []).append(copy.deepcopy(result))

        query_results = [result for values in by_domain.values() for result in values]
        payload = {
            "schema_version": "mmm/forced-pre-design-rag-v2",
            "versions": list(versions),
            "domain_count": len(domains),
            "query_count": len(jobs),
            "unique_route_query_count": len(unique_jobs),
            "project_source_count": sum(
                len(item.get("project_rag", {}).get("sources", []))
                for item in query_results
            ),
            "code_index_status": "available" if code_index is not None else "not_indexed",
            "adaptive_routing": {
                "schema_version": "mmm/adaptive-pre-design-rag-summary-v1",
                "policy": "provider_and_evidence_kind_then_correct_on_empty",
                "catalog_query_count": sum(
                    "project_catalog" in item["retrieval_route"]["executed"]
                    for item in query_results
                ),
                "code_query_count": sum(
                    "code_rag" in item["retrieval_route"]["executed"]
                    for item in query_results
                ),
                "fully_skipped_query_count": sum(
                    not item["retrieval_route"]["executed"] for item in query_results
                ),
                "expanded_query_count": sum(
                    bool(item["retrieval_route"]["expansion"])
                    for item in query_results
                ),
            },
            "domains": [
                {
                    "domain_id": str(domain.get("domain_id", "")),
                    "queries": by_domain.get(str(domain.get("domain_id", "")), []),
                }
                for domain in domains
            ],
        }
        payload["research_sha256"] = pre_design_module._sha256(payload)
        return payload

    setattr(adaptive_forced_rag, _MARKER, True)
    adaptive_forced_rag.__wrapped__ = current
    pre_design_module._forced_rag_bundle = adaptive_forced_rag


__all__ = ["harden"]
