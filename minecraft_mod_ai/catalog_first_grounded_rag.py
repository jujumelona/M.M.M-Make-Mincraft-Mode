from __future__ import annotations

"""Catalog-first provider policy for Minecraft research.

Provider transport lives in ``pre_design_grounded_rag``. This module owns retrieval
policy: discover actual Minecraft mods from CurseForge/Modrinth first, prefer their
explicit source repository links, and fall back to GitHub repository discovery when a
catalog candidate has no linked source or the catalog stage is empty. GitHub remains an
internal source-discovery mechanism, not a peer Minecraft catalog provider.
Official/project sources remain separate from ecosystem discovery.
"""

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _providers(domain: Mapping[str, Any]) -> tuple[str, ...]:
    raw = domain.get("providers")
    values = (
        [_text(item).casefold() for item in raw if _text(item)]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))
        else []
    )
    return tuple(dict.fromkeys(values))


def _run_catalogs(
    backend: Any,
    query: str,
    allowed: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run mod catalogs concurrently but merge in deterministic authority order."""
    calls: dict[str, Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]]] = {}
    receipts: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    if "curseforge" in allowed:
        if os.environ.get("CURSEFORGE_API_KEY", "").strip():
            calls["curseforge"] = lambda: backend._search_curseforge(query)
        else:
            receipts["curseforge"] = {
                "provider": "curseforge",
                "status": "not_configured",
                "result_count": 0,
            }
    if "modrinth" in allowed:
        calls["modrinth"] = lambda: backend._search_modrinth(query)

    results: dict[str, list[dict[str, Any]]] = {}
    if calls:
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {pool.submit(call): provider for provider, call in calls.items()}
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    found, receipt = future.result()
                    results[provider] = list(found)
                    receipts[provider] = receipt
                except Exception as exc:
                    receipt = backend._error(provider, exc)
                    receipts[provider] = receipt
                    errors.append(receipt)
                    results[provider] = []

    records = results.get("curseforge", []) + results.get("modrinth", [])
    return records, receipts, errors


def _dedupe(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        record = dict(raw)
        key = _text(record.get("source_id") or record.get("url"))
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def _query_bundle(
    backend: Any,
    router: Any,
    query: str,
    providers: tuple[str, ...],
    *,
    github_disabled: Callable[[], bool],
    disable_github: Callable[[], None],
) -> dict[str, Any]:
    allowed = set(providers)
    records: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    catalog_allowed = bool(allowed & {"curseforge", "modrinth"})
    catalog_records: list[dict[str, Any]] = []
    if catalog_allowed:
        catalog_records, catalog_receipts, catalog_errors = _run_catalogs(
            backend, query, allowed
        )
        records.extend(catalog_records)
        receipts.update(catalog_receipts)
        errors.extend(catalog_errors)

    # Catalog discovery is only the first stage. A catalog hit without source code must
    # not terminate reuse discovery: exact project source links are preferred, then a
    # bounded GitHub repository search is allowed to recover a source donor candidate.
    github_policy_active = catalog_allowed or "github" in allowed
    if github_policy_active:
        if catalog_records:
            linked, linked_receipt = backend._linked_github_sources(
                catalog_records,
                disabled=github_disabled,
                disable=disable_github,
            )
            records.extend(linked)
            if linked:
                receipts["github"] = {
                    **dict(linked_receipt),
                    "policy": "exact_catalog_link_first",
                }
            elif github_disabled():
                receipts["github"] = {
                    **dict(linked_receipt),
                    "policy": "catalog_source_discovery_disabled",
                }
            else:
                try:
                    found, fallback_receipt = backend._search_github(
                        query,
                        disabled=github_disabled,
                        disable=disable_github,
                    )
                    records.extend(found)
                    receipts["github"] = {
                        **dict(fallback_receipt),
                        "linked_source_status": _text(linked_receipt.get("status")),
                        "policy": "catalog_candidate_source_discovery_fallback",
                    }
                except Exception as exc:
                    receipt = backend._error("github", exc)
                    receipts["github"] = {
                        **dict(receipt),
                        "linked_source_status": _text(linked_receipt.get("status")),
                        "policy": "catalog_candidate_source_discovery_fallback",
                    }
                    errors.append(receipt)
        elif catalog_allowed:
            try:
                found, receipt = backend._search_github(
                    query,
                    disabled=github_disabled,
                    disable=disable_github,
                )
                records.extend(found)
                receipts["github"] = {
                    **dict(receipt),
                    "policy": "catalog_empty_fallback",
                }
            except Exception as exc:
                receipt = backend._error("github", exc)
                receipts["github"] = receipt
                errors.append(receipt)
        else:
            # Explicit repository-only domains may still request direct GitHub lookup.
            try:
                found, receipt = backend._search_github(
                    query,
                    disabled=github_disabled,
                    disable=disable_github,
                )
                records.extend(found)
                receipts["github"] = {
                    **dict(receipt),
                    "policy": "repository_domain_direct",
                }
            except Exception as exc:
                receipt = backend._error("github", exc)
                receipts["github"] = receipt
                errors.append(receipt)

    versions = backend._versions(router)
    project_rag = (
        backend._search_authoritative_catalog(query, versions)
        if allowed & {"official_docs", "project_rag"}
        else {
            "schema_version": "mmm/project-rag-query-v3",
            "sources": [],
            "errors": [],
        }
    )
    code_rag = (
        backend._search_code_index(backend._existing_code_index(), query)
        if "project_rag" in allowed
        else {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "not_requested",
            "hits": [],
        }
    )

    gh = receipts.get("github", {}) if isinstance(receipts.get("github"), Mapping) else {}
    unique = _dedupe(records)
    return {
        "query": query,
        "query_sha256": backend._sha256_text(query),
        "project_rag": project_rag,
        "code_rag": code_rag,
        "external_rag": {
            "schema_version": "mmm/external-pre-design-discovery-v4",
            "sources": unique,
            "errors": errors,
            "providers": receipts,
            "provider_policy": {
                "catalog_first": catalog_allowed,
                "catalog_order": [
                    provider
                    for provider in ("curseforge", "modrinth")
                    if provider in allowed
                ],
                "github_role": (
                    "internal_exact_source_then_source_discovery_fallback"
                    if catalog_allowed
                    else (
                        "direct_repository_domain" if "github" in allowed else "not_requested"
                    )
                ),
                "github_broad_search": (
                    "fallback_after_missing_linked_source_or_empty_catalog"
                    if catalog_allowed
                    else (
                        "direct_repository_domain" if "github" in allowed else "not_requested"
                    )
                ),
                "github_linked_source": (
                    "exact_catalog_source_url_preferred" if catalog_allowed else "not_applicable"
                ),
            },
            "github_retrieval": {
                "provider_status": _text(gh.get("status") or "not_requested"),
                "saturation_reason": (
                    "rate_or_auth_failure"
                    if _text(gh.get("status"))
                    in {"error", "disabled_after_rate_or_auth_failure"}
                    else ""
                ),
                "search_requests": int(gh.get("search_requests") or 0),
                "source_requests": int(gh.get("source_requests") or 0),
            },
        },
    }


def forced_rag_bundle(
    backend: Any,
    router: Any,
    research_brief: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the grounded-RAG wire shape under catalog-first provider policy."""
    domains = (
        [
            dict(item)
            for item in research_brief.get("domains", [])
            if isinstance(item, Mapping)
        ]
        if isinstance(research_brief.get("domains"), list)
        else []
    )
    github_blocked = False
    github_lock = threading.Lock()

    def github_disabled() -> bool:
        with github_lock:
            return github_blocked

    def disable_github() -> None:
        nonlocal github_blocked
        with github_lock:
            github_blocked = True

    specs: list[tuple[str, tuple[str, ...]]] = []
    for domain in domains:
        providers = _providers(domain)
        raw_queries = domain.get("queries")
        for raw in raw_queries if isinstance(raw_queries, list) else []:
            query = _text(raw)
            key = (query, providers)
            if query and key not in specs:
                specs.append(key)

    by_spec: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    if specs:
        max_workers = max(
            1,
            min(int(getattr(backend, "_MAX_QUERY_WORKERS", 4)), len(specs)),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _query_bundle,
                    backend,
                    router,
                    query,
                    providers,
                    github_disabled=github_disabled,
                    disable_github=disable_github,
                ): (query, providers)
                for query, providers in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    by_spec[spec] = future.result()
                except Exception as exc:
                    query, _providers_for_query = spec
                    by_spec[spec] = {
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
                            "github_retrieval": {
                                "provider_status": "not_requested",
                                "saturation_reason": "",
                                "search_requests": 0,
                                "source_requests": 0,
                            },
                        },
                    }

    out_domains: list[dict[str, Any]] = []
    external_count = 0
    query_count = 0
    for domain in domains:
        providers = _providers(domain)
        rows: list[dict[str, Any]] = []
        raw_queries = domain.get("queries")
        for raw in raw_queries if isinstance(raw_queries, list) else []:
            query = _text(raw)
            if not query:
                continue
            query_count += 1
            row = dict(by_spec[(query, providers)])
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
        "provider_policy": "catalog_first_exact_source_then_fallback",
        "external_source_count": external_count,
        "domains": out_domains,
    }
    payload["research_sha256"] = backend._sha256(payload)
    return payload


__all__ = ["forced_rag_bundle"]
