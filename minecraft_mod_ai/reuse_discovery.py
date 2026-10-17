from __future__ import annotations

"""CurseForge-only Minecraft mod catalogue inspection.

No repository discovery, public source hosting, README retrieval, or external
code fetching occurs here. Results provide catalogue evidence, not source
code reuse permission.
"""

import atexit
import os
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

import httpx

from .curseforge_catalog import (
    curseforge_api_key as _curseforge_api_key,
    curseforge_mod_metadata,
    curseforge_mods_class_id,
    curseforge_next_index,
    curseforge_target_filters,
)
from .ecosystem_discovery import provider_request_failure
from .curseforge_query import curseforge_search_queries, curseforge_catalogue_score, curseforge_query_family

_MINECRAFT_GAME_ID = 432
_HTTP_CLIENT_LOCK = Lock()
_HTTP_CLIENT: httpx.Client | None = None

def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _workers() -> int:
    return _env_int("MMM_REUSE_PARALLELISM", 12, minimum=1, maximum=64)


def _query_limit() -> int:
    return _env_int("MMM_REUSE_DISCOVERY_LIMIT", 6, minimum=1, maximum=12)


def _representative_limit() -> int:
    return _env_int("MMM_REUSE_REPRESENTATIVE_LIMIT", 3, minimum=1, maximum=5)


def _query_variant_limit() -> int:
    return _env_int("MMM_REUSE_QUERY_VARIANTS", 3, minimum=1, maximum=4)


def _minimum_candidates() -> int:
    return _env_int("MMM_REUSE_MIN_CANDIDATES_PER_CAPABILITY", 2, minimum=1, maximum=5)


def _discovery_pages() -> int:
    return _env_int("MMM_REUSE_DISCOVERY_PAGES", 1, minimum=1, maximum=2)


def _pooled_http_client() -> httpx.Client:
    """Return one thread-safe connection pool for auxiliary discovery APIs."""

    global _HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
            workers = _workers()
            _HTTP_CLIENT = httpx.Client(
                timeout=10.0,
                follow_redirects=False,
                headers={"User-Agent": "MMM-reuse-discovery/1"},
                limits=httpx.Limits(
                    max_connections=max(8, workers * 2),
                    max_keepalive_connections=max(4, workers),
                    keepalive_expiry=30.0,
                ),
            )
        return _HTTP_CLIENT


def _close_pooled_http_client() -> None:
    global _HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        client = _HTTP_CLIENT
        _HTTP_CLIENT = None
    if client is not None and not client.is_closed:
        client.close()


atexit.register(_close_pooled_http_client)


def _discovery_provider_error(provider: str, exc: BaseException, *, stage: str) -> dict[str, Any]:
    """Retain typed transport facts without exception prose or response bodies."""

    failure = provider_request_failure(exc, provider=provider)
    status = failure.status_code if failure is not None else getattr(exc, "http_status", None)
    status = status if isinstance(status, int) else None
    unavailable = bool(failure is not None and failure.backend_unavailable)
    unavailable = unavailable or bool(getattr(exc, "backend_unavailable", False))
    unavailable = unavailable or provider == "curseforge" and status in {401, 403, 429, 500, 502, 503, 504}
    return {
        "provider": provider,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "http_status": status,
        "backend_unavailable": unavailable,
        "request_failure": failure.to_dict() if failure is not None else None,
        "error": (
            f"{provider} {stage} failed ({type(exc).__name__}"
            + (f", HTTP {status}" if status is not None else "") + ")."
        ),
    }


class _CurseForgeSearchError(RuntimeError):
    """Provider failure containing no credential or response-body text."""

    def __init__(
        self, message: str, *, http_status: int | None = None, request_failure: Any = None
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.request_failure = request_failure


def _search_curseforge(
    query: str,
    *,
    limit: int,
    with_metadata: bool = False,
    minecraft_version: str | None = None,
    loader: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    category_context: dict[str, Any] | None = None,
) -> list:
    """Search provider-side by a nonempty feature keyword, then rerank locally.

    No popularity browse or source-repository lookup is permitted. The request
    remains restricted to the bound version, loader and Minecraft Mods class.
    """
    query = " ".join(str(query or "").split())
    if not query:
        raise ValueError("CURSEFORGE_EMPTY_SEARCH_FILTER_FORBIDDEN")
    if limit < 1:
        raise ValueError("CURSEFORGE_SEARCH_LIMIT_INVALID")
    api_key = _curseforge_api_key()
    if not api_key:
        return []
    try:
        target_filters = curseforge_target_filters(minecraft_version, loader)
    except ValueError as exc:
        raise _CurseForgeSearchError(f"Invalid CurseForge target: {exc}") from None
    client = _pooled_http_client()
    headers = {"Accept": "application/json", "x-api-key": api_key}
    ctx = category_context if category_context is not None else {}
    mods_class_id = ctx.get("mods_class_id")
    if mods_class_id is None:
        try:
            category_response = client.get(
                "https://api.curseforge.com/v1/categories",
                params={"gameId": str(_MINECRAFT_GAME_ID)},
                headers=headers, timeout=10.0,
            )
            category_response.raise_for_status()
            category_payload = category_response.json()
            category_rows = (category_payload.get("data")
                             if isinstance(category_payload, Mapping) else None)
            mods_class_id = curseforge_mods_class_id(category_rows)
        except (httpx.HTTPError, ValueError) as exc:
            raise _CurseForgeSearchError(
                "CurseForge Mods class lookup failed.",
                request_failure=provider_request_failure(exc, provider="curseforge"),
            ) from None
        if mods_class_id is None:
            raise _CurseForgeSearchError(
                "CurseForge Mods class could not be verified; refusing unscoped search."
            )
        ctx["mods_class_id"] = mods_class_id

    result: list[tuple[str, float, dict[str, Any]]] = []
    seen_ids: set[int] = set()
    page_size = min(12, limit)
    index = 0
    for _ in range(_discovery_pages()):
        params = {
            "gameId": str(_MINECRAFT_GAME_ID), "classId": str(mods_class_id),
            "gameVersion": str(minecraft_version or "").strip(),
            "searchFilter": query, "sortField": "13", "sortOrder": "desc",
            "pageSize": str(page_size), "index": str(index),
        }
        params.update(target_filters)
        if not params["gameVersion"]:
            params.pop("gameVersion")
        try:
            response = client.get(
                "https://api.curseforge.com/v1/mods/search",
                params=params, headers=headers, timeout=10.0,
            )
            response.raise_for_status()
            raw = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            failure = _CurseForgeSearchError(
                f"CurseForge search failed (HTTP {status}).", http_status=status,
                request_failure=provider_request_failure(exc, provider="curseforge"),
            )
        except httpx.HTTPError as exc:
            failure = _CurseForgeSearchError(
                f"CurseForge search transport failed ({type(exc).__name__}).",
                request_failure=provider_request_failure(exc, provider="curseforge"),
            )
        except ValueError:
            failure = _CurseForgeSearchError("CurseForge search returned invalid JSON.")
        else:
            failure = None
        if failure is not None:
            if result and errors is not None:
                errors.append(_discovery_provider_error(
                    "curseforge", failure, stage="later_page"))
                break
            raise failure

        rows = raw.get("data") if isinstance(raw, Mapping) else None
        if not isinstance(rows, list):
            failure = _CurseForgeSearchError("CurseForge search returned no valid data list.")
            if result and errors is not None:
                errors.append(_discovery_provider_error(
                    "curseforge", failure, stage="later_page"))
                break
            raise failure
        for item in rows:
            if not isinstance(item, Mapping) or type(item.get("id")) is not int:
                continue
            mod_id = item["id"]
            if mod_id in seen_ids or item.get("classId") != mods_class_id:
                continue
            seen_ids.add(mod_id)
            # The server's searchFilter may match author only. Reject unrelated
            # projects rather than promoting every provider hit as an answer.
            score = curseforge_catalogue_score(query, item)
            if score <= 0:
                continue
            info = {
                "title": str(item.get("name") or ""),
                "summary": str(item.get("summary") or ""),
                "repository": "",
                "query": query,
                "catalog_class_verified": True,
                "target_filter_applied": bool(target_filters),
                **curseforge_mod_metadata(
                    item, minecraft_version=minecraft_version, loader=loader
                ),
            }
            result.append(("", float(score), info))
        if len(result) >= limit:
            break
        try:
            next_index = curseforge_next_index(
                raw.get("pagination") if isinstance(raw, Mapping) else None,
                index=index, page_size=page_size, row_count=len(rows),
            )
        except ValueError as exc:
            failure = _CurseForgeSearchError(str(exc))
            if result and errors is not None:
                errors.append(_discovery_provider_error(
                    "curseforge", failure, stage="later_page"))
                break
            raise failure from None
        if next_index is None:
            break
        index = next_index
    result.sort(key=lambda row: (-row[1], str(row[2].get("title", "")).casefold()))
    result = result[:limit]
    return result if with_metadata else [(repo, rank) for repo, rank, _ in result]


def discover_repositories_for_graph(
    capabilities: Sequence[str],
    client: Any,
    *,
    capability_graph: Mapping[str, Any] | None = None,
    minecraft_version: str | None = None,
    loader: str | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    providers: Sequence[str] | None = None,
    query_cache: dict[tuple[str, str, str, str], Any] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Collect per-capability CurseForge evidence without discovering repos.

    Return shape stays empty by design: a catalog listing cannot authorize a
    source transplant. Diagnostic receipts carry ranked, target-filtered facts.
    """
    del client
    selected = tuple(dict.fromkeys(str(c).strip() for c in capabilities if str(c).strip()))
    result: dict[str, tuple[str, ...]] = {capability: () for capability in selected}
    requested = tuple(providers) if providers is not None else ("curseforge",)
    if requested != ("curseforge",):
        raise ValueError("Only the CurseForge catalogue provider is supported.")
    if not selected:
        return result
    version = str(minecraft_version or "").strip()
    selected_loader = str(loader or "").strip().casefold()
    # Validate target even when the provider is unavailable.
    curseforge_target_filters(version, selected_loader)
    terms_by_cap: dict[str, tuple[str, ...]] = {}
    if isinstance(capability_graph, Mapping):
        for row in capability_graph.get("search_terms", ()):
            if isinstance(row, Mapping):
                name = str(row.get("capability") or "")
                terms = row.get("terms")
                if isinstance(terms, (list, tuple)):
                    terms_by_cap[name] = tuple(str(v) for v in terms if str(v).strip())
    category_context: dict[str, Any] = {}
    # Deduplicate identical provider requests even if callers omit a cache.
    shared_cache = query_cache if query_cache is not None else {}
    for capability in selected:
        queries = curseforge_search_queries(
            capability, search_terms=terms_by_cap.get(capability, ()),
            max_variants=_query_variant_limit(),
        )
        if not queries:
            if diagnostics is not None:
                diagnostics.append({
                    "provider": "curseforge", "status": "query_unresolved",
                    "capability": capability, "queries": [],
                    "catalog_result_count": 0, "ranked_catalogue_hits": [],
                    "minecraft_version": version, "loader": selected_loader,
                })
            continue
        if not _curseforge_api_key():
            if diagnostics is not None:
                diagnostics.append({
                    "provider": "curseforge", "status": "not_configured",
                    "capability": capability, "queries": list(queries),
                    "catalog_result_count": 0, "ranked_catalogue_hits": [],
                    "minecraft_version": version, "loader": selected_loader,
                })
            continue
        combined: dict[str, dict[str, Any]] = {}
        family_sources: dict[str, set[str]] = {}
        query_receipts: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for query in queries:
            family = curseforge_query_family(query)
            # Equivalent spellings are fallback only; different mechanics retain
            # independent evidence shortlists even within a long requirement.
            if len(family_sources.get(family, ())) >= _minimum_candidates():
                continue
            key = ("curseforge", query.casefold(), version, selected_loader)
            cached = shared_cache.get(key)
            if isinstance(cached, Mapping):
                rows = list(cached.get("rows", ()))
                errors = list(cached.get("errors", ()))
                cache_hit = True
            else:
                errors = []
                try:
                    rows = _search_curseforge(
                        query, limit=_query_limit(), with_metadata=True,
                        minecraft_version=version, loader=selected_loader,
                        errors=errors, category_context=category_context,
                    )
                except Exception as exc:
                    rows = []
                    errors.append(_discovery_provider_error(
                        "curseforge", exc, stage="search"))
                cache_hit = False
                shared_cache[key] = {"rows": list(rows), "errors": list(errors)}
            failures.extend(errors)
            query_receipts.append({
                "query": query, "hit_count": len(rows),
                "cache_hit": cache_hit, "errors": list(errors),
            })
            for _repository, provider_score, meta in rows:
                if not isinstance(meta, Mapping):
                    continue
                ident = str(meta.get("source_id") or "")
                if not ident:
                    continue
                # Re-evaluate full original requirement (not only the short
                # provider keyword), preferring exact capability relevance.
                score = curseforge_catalogue_score(
                    capability,
                    {"name": meta.get("title"), "summary": meta.get("summary")},
                    search_terms=terms_by_cap.get(capability, ()),
                )
                if score <= 0:
                    continue
                family_sources.setdefault(family, set()).add(ident)
                previous = combined.get(ident)
                if previous is None:
                    combined[ident] = {
                        "source_id": ident, "catalog_id": meta.get("catalog_id"),
                        "title": str(meta.get("title") or ""),
                        "summary": str(meta.get("summary") or "")[:400],
                        "catalog_url": str(meta.get("catalog_url") or ""),
                        "minecraft_version": version, "loader": selected_loader,
                        "score": score, "matched_queries": [query],
                        "source_reuse_authority": "catalogue_only",
                    }
                else:
                    previous["score"] = max(int(previous["score"]), score)
                    if query not in previous["matched_queries"]:
                        previous["matched_queries"].append(query)
        ranked = sorted(
            combined.values(),
            key=lambda row: (-int(row["score"]), -len(row["matched_queries"]),
                             str(row["title"]).casefold(), str(row["source_id"])),
        )
        # Keep a few per distinct topic rather than letting spaceship hits
        # displace unrelated crew, trading or combat evidence.
        shortlist: list[dict[str, Any]] = []
        seen_shortlist: set[str] = set()
        for family in dict.fromkeys(curseforge_query_family(query) for query in queries):
            count = 0
            for hit in ranked:
                ident = str(hit["source_id"])
                if ident not in family_sources.get(family, ()) or ident in seen_shortlist:
                    continue
                shortlist.append(hit)
                seen_shortlist.add(ident)
                count += 1
                if count >= _representative_limit():
                    break
        if diagnostics is not None:
            diagnostics.append({
                "provider": "curseforge", "capability": capability,
                "queries": [receipt["query"] for receipt in query_receipts],
                "query_receipts": query_receipts,
                "status": ("provider_error" if failures
                           else "catalog_only" if ranked else "search_empty"),
                "catalog_result_count": len(shortlist),
                "catalog_scanned_count": len(ranked),
                "ranked_catalogue_hits": shortlist,
                "repository_count": 0, "repository_evidence": [],
                "errors": failures,
                "minecraft_version": version, "loader": selected_loader,
                "relevance_policy": "exact_requirement_ranked_metadata",
                "search_exhaustiveness": "bounded_targeted_name_queries",
            })
    return result


__all__ = ["discover_repositories_for_graph"]
