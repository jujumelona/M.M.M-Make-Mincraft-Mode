from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .knowledge import AuthoritativeEvidenceRetriever, evidence_catalog_for_version
from .rag_index import ProjectRAGIndex


_MARKER = "_mmm_forced_pre_design_rag_v1"


def harden_pre_design_research(agentic_module: Any) -> None:
    """Force deterministic RAG over every research query before design generation.

    The research agent can still issue further MCP retrieval/actions adaptively, but it
    never starts from an empty evidence slate. Every research-brief query is searched
    against the code-owned authoritative project catalog. If a durable source-code RAG
    index exists, the same queries are searched there too. Full receipts stay in the
    pre-design research bundle; final section workers receive only compact receipts and
    the research agent's claims/gaps so prompt size cannot grow with raw retrieval data.
    """

    if getattr(agentic_module.collect_pre_design_research, _MARKER, False):
        return

    original_collect = agentic_module.collect_pre_design_research
    original_domain_slice = agentic_module._domain_evidence_slice

    def collect(
        router: Any,
        prompt: str,
        *,
        trace_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Run the established official/technology/ecosystem + ReAct research flow.
        # Its domain agents call _domain_evidence_slice dynamically, so expose the
        # forced receipts to that helper through a short-lived module context.
        brief = agentic_module.normalize_research_brief(
            prompt,
            {"title": "pre-design research"},
        )
        forced = _forced_rag_bundle(router, brief)
        previous = getattr(agentic_module, "_MMM_FORCED_RAG_CONTEXT", None)
        agentic_module._MMM_FORCED_RAG_CONTEXT = forced
        try:
            result = original_collect(
                router,
                prompt,
                trace_metadata=trace_metadata,
            )
        finally:
            if previous is None:
                try:
                    delattr(agentic_module, "_MMM_FORCED_RAG_CONTEXT")
                except AttributeError:
                    pass
            else:
                agentic_module._MMM_FORCED_RAG_CONTEXT = previous

        deterministic = result.get("deterministic")
        if not isinstance(deterministic, dict):
            deterministic = {}
        result = {
            **result,
            "deterministic": {
                **deterministic,
                "forced_project_rag": forced,
            },
        }
        result["research_sha256"] = _sha256(result)
        return result

    def domain_slice(domain_id: str, deterministic: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(original_domain_slice(domain_id, deterministic))
        forced = getattr(agentic_module, "_MMM_FORCED_RAG_CONTEXT", None)
        if not isinstance(forced, Mapping):
            forced = deterministic.get("forced_project_rag")
        if isinstance(forced, Mapping):
            domains = forced.get("domains")
            if isinstance(domains, list):
                selected = next(
                    (
                        item
                        for item in domains
                        if isinstance(item, Mapping)
                        and item.get("domain_id") == domain_id
                    ),
                    None,
                )
                if selected is not None:
                    value["forced_project_rag"] = selected
        return value

    def compact_receipt(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        keep = (
            "schema_version",
            "research_sha256",
            "evidence_sha256",
            "radar_sha256",
            "route_sha256",
            "query_sha256",
            "status",
            "unresolved_official_domains",
            "candidate_count",
            "domain_count",
            "query_count",
            "project_source_count",
            "code_index_status",
            "code_index_path",
        )
        return {key: value[key] for key in keep if key in value}

    setattr(collect, _MARKER, True)
    agentic_module.collect_pre_design_research = collect
    agentic_module._domain_evidence_slice = domain_slice
    # Section workers must consume claims/gaps + receipts, not raw retrieval pages.
    agentic_module._research_receipt = compact_receipt


def _forced_rag_bundle(router: Any, research_brief: Mapping[str, Any]) -> dict[str, Any]:
    raw_domains = research_brief.get("domains")
    domains = [item for item in raw_domains or [] if isinstance(item, Mapping)]
    jobs: list[tuple[str, str]] = []
    for domain in domains:
        domain_id = str(domain.get("domain_id", "")).strip()
        queries = domain.get("queries")
        if not domain_id or not isinstance(queries, list):
            continue
        for query in queries:
            query_text = str(query).strip()
            if query_text:
                jobs.append((domain_id, query_text))

    versions = _research_versions(router)
    code_index = _existing_code_index()
    worker_count = max(1, min(8, len(jobs)))

    def run(job: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        domain_id, query = job
        project_results = _search_authoritative_catalog(query, versions)
        code_result = _search_code_index(code_index, query)
        return domain_id, {
            "query": query,
            "query_sha256": _sha256_text(query),
            "project_rag": project_results,
            "code_rag": code_result,
        }

    by_domain: dict[str, list[dict[str, Any]]] = {
        str(item.get("domain_id", "")): [] for item in domains
    }
    if jobs:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="mmm_pre_design_rag",
        ) as pool:
            # map preserves the authoritative query order while searches overlap.
            for domain_id, result in pool.map(run, jobs):
                by_domain.setdefault(domain_id, []).append(result)

    payload = {
        "schema_version": "mmm/forced-pre-design-rag-v1",
        "versions": list(versions),
        "domain_count": len(domains),
        "query_count": len(jobs),
        "project_source_count": sum(
            len(item.get("project_rag", {}).get("sources", []))
            for values in by_domain.values()
            for item in values
        ),
        "code_index_status": "available" if code_index is not None else "not_indexed",
        "code_index_path": str(code_index) if code_index is not None else "",
        "domains": [
            {
                "domain_id": str(domain.get("domain_id", "")),
                "queries": by_domain.get(str(domain.get("domain_id", "")), []),
            }
            for domain in domains
        ],
    }
    payload["research_sha256"] = _sha256(payload)
    return payload


def _research_versions(router: Any) -> tuple[str, ...]:
    requested = str(
        getattr(router, "_mmm_requested_minecraft_version", "") or ""
    ).strip()
    existing = str(
        getattr(router, "_mmm_existing_minecraft_version", "") or ""
    ).strip()
    if requested:
        return (requested,)
    if existing:
        return (existing,)
    # Before live target resolution, search both offline seed scopes. Generic official
    # records apply to both; exact Javadocs remain version-scoped. No network lookup is
    # introduced just to choose a pre-design RAG scope.
    return ("1.20.1", "1.21.1")


def _search_authoritative_catalog(
    query: str,
    versions: tuple[str, ...],
) -> dict[str, Any]:
    retriever = AuthoritativeEvidenceRetriever()
    sources: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for version in versions:
        try:
            catalog = evidence_catalog_for_version(version)
            limit = min(6, len(catalog))
            for source in retriever.search(
                query,
                minecraft_version=version,
                limit=limit,
            ):
                payload = asdict(source)
                payload["matched_version"] = version
                sources.setdefault(source.source_id, payload)
        except Exception as exc:
            errors.append(
                {
                    "minecraft_version": version,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema_version": "mmm/forced-project-rag-query-v1",
        "sources": [sources[key] for key in sorted(sources)],
        "errors": errors,
    }


def _existing_code_index() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path("rag/project-index.json"))
    workspace = os.environ.get("MMM_WORKSPACE", "").strip()
    if workspace:
        candidates.append(Path(workspace).expanduser() / "rag/project-index.json")
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _search_code_index(index_path: Path | None, query: str) -> dict[str, Any]:
    if index_path is None:
        return {
            "schema_version": "mmm/forced-code-rag-query-v1",
            "status": "not_indexed",
            "hits": [],
        }
    try:
        result = ProjectRAGIndex(index_path).search_with_receipt(
            query,
            limit=8,
            semantic=False,
            rerank=False,
        )
        return {
            "schema_version": "mmm/forced-code-rag-query-v1",
            "status": "searched",
            "hits": [asdict(hit) for hit in result.hits],
            "receipt": asdict(result.receipt),
        }
    except Exception as exc:
        return {
            "schema_version": "mmm/forced-code-rag-query-v1",
            "status": "error",
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["harden_pre_design_research"]
