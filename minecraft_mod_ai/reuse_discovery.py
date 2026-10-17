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
    curseforge_browse_windows,
    curseforge_mod_metadata,
    curseforge_mods_class_id,
    curseforge_next_index,
    curseforge_target_filters,
)
from .ecosystem_discovery import provider_request_failure

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
    return _env_int("MMM_REUSE_DISCOVERY_LIMIT", 16, minimum=4, maximum=50)


def _representative_limit() -> int:
    return _env_int("MMM_REUSE_REPRESENTATIVE_LIMIT", 16, minimum=4, maximum=48)


def _query_variant_limit() -> int:
    return _env_int("MMM_REUSE_QUERY_VARIANTS", 3, minimum=1, maximum=6)


def _minimum_candidates() -> int:
    return _env_int("MMM_REUSE_MIN_CANDIDATES_PER_CAPABILITY", 3, minimum=1, maximum=12)


def _discovery_pages() -> int:
    return _env_int("MMM_REUSE_DISCOVERY_PAGES", 2, minimum=1, maximum=4)


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
    with_metadata=False,
    minecraft_version=None,
    loader=None,
    errors: list[dict[str, Any]] | None = None,
) -> list:
    """Browse real Minecraft mod listings, not invented feature-name queries.

    query/limit remain for the existing caller API, but do not filter or truncate
    this shared catalogue sample: relevance is calculated per capability later.
    The selected target version/loader is always passed to CurseForge.
    Catalogue metadata is never treated as code reuse proof.
    """
    api_key = _curseforge_api_key()
    if not api_key:
        return []
    try:
        target_filters = curseforge_target_filters(minecraft_version, loader)
    except ValueError as exc:
        raise _CurseForgeSearchError(f"Invalid CurseForge target: {exc}") from None
    client = _pooled_http_client()
    headers = {"Accept": "application/json", "x-api-key": api_key}
    try:
        category_response = client.get(
            "https://api.curseforge.com/v1/categories",
            params={"gameId": str(_MINECRAFT_GAME_ID)},
            headers=headers, timeout=10.0,
        )
        category_response.raise_for_status()
        category_payload = category_response.json()
        category_rows = category_payload.get("data") if isinstance(category_payload, Mapping) else None
        mods_class_id = curseforge_mods_class_id(category_rows)
    except (httpx.HTTPError, ValueError):
        # The search endpoint may remain functional even if categories do not.
        # Results without a known Mods class must not masquerade as exhaustive.
        mods_class_id = None
        category_rows = None
    if mods_class_id is None:
        raise _CurseForgeSearchError(
            "CurseForge Mods class could not be verified; refusing unscoped search."
        )

    result = []
    seen_ids: set[int] = set()
    page_size = 50  # Each browse page can expose candidates beyond a single shortlist.
    # Categories reveal less popular projects that broad popularity/recency
    # windows miss. Keep names out of searchFilter for every browse window.
    windows = curseforge_browse_windows(
        category_rows, mods_class_id, broad_pages=_discovery_pages()
    )
    try:
        for sort_field, category_id, max_pages in windows:
            index = 0
            for _ in range(max_pages):
                params = {
                    "gameId": str(_MINECRAFT_GAME_ID),
                    "sortField": str(sort_field),
                    "sortOrder": "desc",
                    "pageSize": str(page_size),
                    "index": str(index),
                    "classId": str(mods_class_id),
                }
                params.update(target_filters)
                if category_id is not None:
                    params["categoryId"] = str(category_id)
                try:
                    response = client.get(
                        "https://api.curseforge.com/v1/mods/search",
                        params=params, headers=headers, timeout=10.0,
                    )
                    response.raise_for_status()
                    raw = response.json()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    raise _CurseForgeSearchError(
                        f"CurseForge search failed (HTTP {status}).", http_status=status,
                        request_failure=provider_request_failure(exc, provider="curseforge"),
                    ) from None
                except httpx.HTTPError as exc:
                    raise _CurseForgeSearchError(
                        f"CurseForge search transport failed ({type(exc).__name__}).",
                        request_failure=provider_request_failure(exc, provider="curseforge"),
                    ) from None
                except ValueError:
                    raise _CurseForgeSearchError("CurseForge search returned invalid JSON.") from None

                rows = raw.get("data") if isinstance(raw, Mapping) else None
                if not isinstance(rows, list):
                    raise _CurseForgeSearchError("CurseForge search returned no valid data list.")
                for item in rows:
                    if not isinstance(item, Mapping) or type(item.get("id")) is not int:
                        continue
                    mod_id = item["id"]
                    if mod_id in seen_ids:
                        continue
                    if item.get("classId") != mods_class_id:
                        continue
                    seen_ids.add(mod_id)
                    repository = ""  # Catalogue entry only, not a source repository.
                    result.append((repository, 1.0 / (1 + len(result)), {
                        "title": item.get("name", ""), "summary": item.get("summary", ""),
                        "repository": repository, "sort_field": sort_field,
                        "catalog_class_verified": mods_class_id is not None,
                        **curseforge_mod_metadata(
                            item, minecraft_version=minecraft_version, loader=loader
                        ),
                    }))
                try:
                    next_index = curseforge_next_index(
                        raw.get("pagination") if isinstance(raw, Mapping) else None,
                        index=index, page_size=page_size, row_count=len(rows),
                    )
                except ValueError as exc:
                    raise _CurseForgeSearchError(str(exc)) from None
                if next_index is None:
                    break
                index = next_index
    except _CurseForgeSearchError as exc:
        if not result:
            raise
        if errors is not None:
            errors.append(_discovery_provider_error("curseforge", exc, stage="later_page"))
        else:
            raise
    return result if with_metadata else [(repo, rank) for repo, rank, _meta in result if repo]


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
    """Inspect CurseForge once for selected target; never discover source repos.

    The return shape is kept for the host's fresh-authoring contract.
    Inspecting catalogue metadata alone must not be reported as reusable Java.
    """
    selected = tuple(dict.fromkeys(str(c).strip() for c in capabilities if str(c).strip()))
    result = {capability: () for capability in selected}
    requested = tuple(providers) if providers is not None else ("curseforge",)
    if requested != ("curseforge",):
        raise ValueError("Only the CurseForge catalogue provider is supported.")
    if not selected:
        return result
    version = str(minecraft_version or "").strip()
    selected_loader = str(loader or "").strip().casefold()
    if not _curseforge_api_key():
        if diagnostics is not None:
            diagnostics.append({
                "provider": "curseforge", "status": "not_configured",
                "capabilities": list(selected), "query": "",
                "minecraft_version": version, "loader": selected_loader,
                "repository_count": 0, "catalog_result_count": 0,
            })
        return result
    key = ("curseforge", "", version, selected_loader)
    cached = query_cache.get(key) if query_cache is not None else None
    if isinstance(cached, Mapping):
        rows = list(cached.get("rows", ()))
        errors = list(cached.get("errors", ()))
        cache_hit = True
    else:
        errors = []
        try:
            rows = _search_curseforge(
                "", limit=_query_limit(), with_metadata=True,
                minecraft_version=version, loader=selected_loader,
                errors=errors,
            )
        except Exception as exc:
            rows = []
            errors.append(_discovery_provider_error("curseforge", exc, stage="search"))
        cache_hit = False
        if query_cache is not None:
            query_cache[key] = {"rows": list(rows), "errors": list(errors)}
    if diagnostics is not None:
        diagnostics.append({
            "provider": "curseforge", "query": "",
            "capabilities": list(selected),
            "status": "provider_error" if errors else ("catalog_only" if rows else "search_empty"),
            "catalog_result_count": len(rows),
            "repository_count": 0,
            "repositories": [],
            "repository_evidence": [],
            "source_repository_missing_count": 0,
            "usable_source_repository_count": 0,
            "cache_hit": cache_hit,
            "minecraft_version": version, "loader": selected_loader,
            "errors": list(errors),
            "error": "; ".join(str(error.get("error", "")) for error in errors),
            "http_status": next((error.get("http_status") for error in errors if error.get("http_status")), None),
            "search_exhaustiveness": "bounded_category_and_multi_sort_sample",
            "relevance_policy": "catalogue_metadata_only_fresh_code",
        })
    return result


__all__ = ["discover_repositories_for_graph"]
