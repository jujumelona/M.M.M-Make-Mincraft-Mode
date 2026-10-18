from __future__ import annotations

"""Official API and local-project evidence policy for Minecraft research.

External mod catalogues, donor repositories and source transplant providers are
not consulted. Keep the existing module entrypoints for internal callers while
only admitting official target APIs and local project source evidence.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .deadline_executor import iter_completed_with_deadlines


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _providers(domain: Mapping[str, Any]) -> tuple[str, ...]:
    raw = domain.get("providers")
    values = (
        [_text(item).casefold() for item in raw if _text(item)]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))
        else []
    )
    return tuple(dict.fromkeys(v for v in values if v in {"official_docs", "project_rag", "runtime"}))


def _domain_spec_occurrences(domain: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    """Only authored queries against official APIs and the current project."""
    requested = _providers(domain)
    allowed = tuple(p for p in requested if p in {"official_docs", "project_rag"})
    if not allowed:
        allowed = ("official_docs", "project_rag")
    return [(_text(query), allowed) for query in domain.get("queries", []) if _text(query)]


def _domain_specs(domain: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    """Return unique execution specs while preserving first-authored order."""

    return list(dict.fromkeys(_domain_spec_occurrences(domain)))


def _query_bundle(backend: Any, router: Any, query: str,
                  providers: tuple[str, ...]) -> dict[str, Any]:
    """Consult target-appropriate official API and local project only."""
    allowed = set(providers) & {"official_docs", "project_rag"}
    if not allowed:
        allowed = {"official_docs", "project_rag"}
    versions = backend._versions(router)
    project_rag = backend._search_authoritative_catalog(query, versions)
    code_rag = (backend._search_code_index(backend._existing_code_index(), query)
                if "project_rag" in allowed else
                {"schema_version": "mmm/code-rag-query-v3",
                 "status": "not_requested", "hits": []})
    return {
        "query": query,
        "query_sha256": backend._sha256_text(query),
        "project_rag": project_rag,
        "code_rag": code_rag,
        "external_rag": {
            "schema_version": "mmm/external-pre-design-discovery-v4",
            "sources": [], "errors": [], "providers": {},
            "provider_policy": {"catalog_first": False, "catalog_order": []},
        },
    }


def _query_failure(backend: Any, query: str, exc: Exception) -> dict[str, Any]:
    return {
        "query": query,
        "query_sha256": backend._sha256_text(query),
        "project_rag": {"sources": [], "errors": []},
        "code_rag": {"status": "error", "hits": []},
        "external_rag": {
            "schema_version": "mmm/external-pre-design-discovery-v4",
            "sources": [],
            "errors": [backend._error("query_worker", exc)],
            "providers": {},
            "provider_policy": {},

        },
    }


def forced_rag_bundle(
    backend: Any,
    router: Any,
    research_brief: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the grounded-RAG wire shape under official/project provider policy.

    Query orchestration is intentionally serial. Provider implementations own the only
    executor layer for source/README expansion, preventing nested executor fan-out while
    still using hardware parallelism where the expensive I/O actually occurs.
    """

    domains = (
        [
            dict(item)
            for item in research_brief.get("domains", [])
            if isinstance(item, Mapping)
        ]
        if isinstance(research_brief.get("domains"), list)
        else []
    )
    specs: list[tuple[str, tuple[str, ...]]] = []
    for domain in domains:
        for key in _domain_specs(domain):
            if key not in specs:
                specs.append(key)

    by_spec: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    if specs:
        query_workers = backend._query_worker_count(len(specs))

        def run_query_spec(
            spec: tuple[str, tuple[str, ...]],
        ) -> tuple[tuple[str, tuple[str, ...]], dict[str, Any]]:
            query, providers = spec
            try:
                row = _query_bundle(
                    backend,
                    router,
                    query,
                    providers,
                )
            except Exception as exc:
                row = _query_failure(backend, query, exc)
            return spec, row

        for _spec, result in iter_completed_with_deadlines(
            tuple(specs),
            run_query_spec,
            max_workers=max(1, query_workers),
            stage="predesign-query-bundles",
            sort_key=lambda item: (item[0], item[1]),
        ):
            spec, row = result
            by_spec[spec] = row

    out_domains: list[dict[str, Any]] = []
    external_count = 0
    query_count = 0
    for domain in domains:
        providers = _providers(domain)
        rows: list[dict[str, Any]] = []
        for query, query_providers in _domain_spec_occurrences(domain):
            query_count += 1
            row = dict(by_spec[(query, query_providers)])
            rows.append(row)
            external = row.get("external_rag")
            sources = external.get("sources") if isinstance(external, Mapping) else []
            external_count += len(sources) if isinstance(sources, list) else 0
        out_domains.append(
            {
                "domain_id": _text(domain.get("domain_id")),
                "providers": list(providers),
                "queries": rows,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "mmm/pre-design-grounded-rag-v6",
        "versions": list(backend._versions(router)),
        "domain_count": len(domains),
        "query_count": query_count,
        "unique_query_count": len(specs),
        "provider_policy": "official_api_and_local_project_only",
        "external_source_count": external_count,
        "domains": out_domains,
    }
    payload["research_sha256"] = backend._sha256(payload)
    return payload


__all__ = ["forced_rag_bundle"]
