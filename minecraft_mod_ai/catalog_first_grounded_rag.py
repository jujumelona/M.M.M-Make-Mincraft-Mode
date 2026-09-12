from __future__ import annotations

"""Catalog-first provider policy for Minecraft research.

Provider transport lives in ``pre_design_grounded_rag``. This module owns retrieval
policy: discover actual Minecraft mods from CurseForge/Modrinth first, prefer their
explicit source repository links, and fall back to GitHub repository discovery when a
catalog candidate has no linked source or the catalog stage is empty. GitHub remains an
internal source-discovery mechanism, not a peer Minecraft catalog provider.
Official/project sources remain separate from ecosystem discovery.

Query-level orchestration uses the backend-owned dynamic worker authority and the
shared deadline executor. Provider/source retrieval remains independently bounded by its
transport timeouts, while deterministic output order is reconstructed from authored specs.
"""

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .deadline_executor import iter_completed_with_deadlines
from .planning_mod_discovery import CATALOG_PROVIDERS


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


def _domain_spec_occurrences(
    domain: Mapping[str, Any],
) -> list[tuple[str, tuple[str, ...]]]:
    """Return authored query occurrences without collapsing duplicate user intent."""

    providers = _providers(domain)
    catalog = domain.get("catalog_queries")
    groups = [(domain.get("queries", []), providers)]
    if isinstance(catalog, list):
        groups = [
            (catalog, tuple(p for p in providers if p in CATALOG_PROVIDERS)),
            (
                domain.get("queries", []),
                tuple(p for p in providers if p not in CATALOG_PROVIDERS),
            ),
        ]
    return [
        (_text(query), allowed)
        for queries, allowed in groups
        if allowed
        for query in (queries if isinstance(queries, list) else [])
        if _text(query)
    ]


def _domain_specs(domain: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    """Return unique execution specs while preserving first-authored order."""

    return list(dict.fromkeys(_domain_spec_occurrences(domain)))


def _run_catalogs(
    backend: Any,
    query: str,
    allowed: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run catalogs in authority order; leaf source retrieval owns parallelism."""

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
    for provider, call in calls.items():
        try:
            found, receipt = call()
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


def _combined_github_receipt(
    linked_receipt: Mapping[str, Any],
    fallback_receipt: Mapping[str, Any],
    *,
    policy: str,
) -> dict[str, Any]:
    """Preserve request accounting across exact-link and fallback discovery stages."""

    receipt = {**dict(linked_receipt), **dict(fallback_receipt), "policy": policy}
    receipt["search_requests"] = int(linked_receipt.get("search_requests") or 0) + int(
        fallback_receipt.get("search_requests") or 0
    )
    receipt["source_requests"] = int(linked_receipt.get("source_requests") or 0) + int(
        fallback_receipt.get("source_requests") or 0
    )
    return receipt


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
    # progress-driven GitHub repository search may recover a source donor candidate.
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
                    "policy": "exact_catalog_link_only",
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
                    receipts["github"] = _combined_github_receipt(
                        linked_receipt,
                        fallback_receipt,
                        policy="catalog_candidate_source_discovery_fallback",
                    )
                except Exception as exc:
                    receipt = backend._error("github", exc)
                    receipts["github"] = _combined_github_receipt(
                        linked_receipt,
                        receipt,
                        policy="catalog_candidate_source_discovery_fallback",
                    )
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
            "github_retrieval": {
                "provider_status": "not_requested",
                "saturation_reason": "",
                "search_requests": 0,
                "source_requests": 0,
            },
        },
    }


def forced_rag_bundle(
    backend: Any,
    router: Any,
    research_brief: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the grounded-RAG wire shape under catalog-first provider policy.

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
                    github_disabled=github_disabled,
                    disable_github=disable_github,
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
        "provider_policy": "catalog_first_exact_source_then_fallback",
        "external_source_count": external_count,
        "domains": out_domains,
    }
    payload["research_sha256"] = backend._sha256(payload)
    return payload


__all__ = ["forced_rag_bundle"]