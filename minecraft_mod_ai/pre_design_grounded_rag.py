from __future__ import annotations

"""Host-owned, requirement-complete pre-design source discovery."""

import hashlib
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .knowledge import (
    AuthoritativeEvidenceRetriever,
    evidence_catalog_for_version,
    target_neutral_evidence_catalog,
)
from .rag_index import ProjectRAGIndex

_TIMEOUT = 8.0
_MAX_QUERY_WORKERS = max(1, min(8, int(os.environ.get("MMM_PREDESIGN_QUERY_WORKERS", "4") or 4)))
_MAX_SOURCE_WORKERS = max(
    1, min(16, int(os.environ.get("MMM_PREDESIGN_SOURCE_WORKERS", "8") or 8))
)
# Provider search endpoints are relevance-ranked. Exhaustively paging their catalogs
# multiplies detail fetches and the later small-model read without improving scope
# coverage. Keep authored query breadth, but bound expensive fan-out per query.
_MAX_PROVIDER_RESULTS_PER_QUERY = max(
    1,
    min(
        24,
        int(os.environ.get("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY", "6") or 6),
    ),
)
_MAX_PROVIDER_SEARCH_PAGES = max(
    1,
    min(
        4,
        int(os.environ.get("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES", "2") or 2),
    ),
)
_UA = "MMM-PreDesignResearch/2.0 (+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _query_terms(value: str) -> str:
    stop = {
        "minecraft",
        "fabric",
        "mod",
        "mods",
        "game",
        "system",
        "feature",
        "implementation",
        "source",
    }
    words: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_+.#/-]+|[가-힣]{2,}", value):
        key = token.casefold()
        if len(key) < 3 or key in stop or key in words:
            continue
        words.append(key)
    return " ".join(words) or "minecraft fabric"


def _json(url: str, headers: Mapping[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": _UA,
            **dict(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _text(url: str, headers: Mapping[str, str] | None = None) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": _UA,
            **dict(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def _error(provider: str, exc: BaseException) -> dict[str, Any]:
    code = getattr(exc, "code", None)
    return {
        "provider": provider,
        "status": "error",
        "http_status": code if isinstance(code, int) else None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _search_modrinth(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    offset = 0
    search_requests = 0
    source_requests = 0
    provider_total = 0
    while (
        search_requests < _MAX_PROVIDER_SEARCH_PAGES
        and len(records) < _MAX_PROVIDER_RESULTS_PER_QUERY
    ):
        page_size = min(100, _MAX_PROVIDER_RESULTS_PER_QUERY - len(records))
        params = urllib.parse.urlencode(
            {
                "query": _query_terms(query),
                "limit": page_size,
                "offset": offset,
                "index": "relevance",
                "facets": json.dumps([["project_type:mod"]]),
            }
        )
        payload = _json(f"https://api.modrinth.com/v2/search?{params}")
        search_requests += 1
        raw_hits = payload.get("hits", []) if isinstance(payload, Mapping) else []
        hits = [hit for hit in raw_hits if isinstance(hit, Mapping)] if isinstance(raw_hits, list) else []
        if not hits:
            break
        try:
            provider_total = max(provider_total, int(payload.get("total_hits", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            provider_total = max(provider_total, offset + len(hits))

        project_ids = list(
            dict.fromkeys(
                str(hit.get("project_id") or "").strip()
                for hit in hits
                if str(hit.get("project_id") or "").strip()
            )
        )
        details: dict[str, Mapping[str, Any]] = {}
        if project_ids:
            source_requests += 1
            try:
                detail_params = urllib.parse.urlencode(
                    {"ids": json.dumps(project_ids, separators=(",", ":"))}
                )
                fetched = _json(
                    f"https://api.modrinth.com/v2/projects?{detail_params}"
                )
                if isinstance(fetched, list):
                    details = {
                        str(item.get("id") or "").strip(): item
                        for item in fetched
                        if isinstance(item, Mapping)
                        and str(item.get("id") or "").strip()
                    }
            except Exception as exc:
                errors.append(f"bulk-projects:{type(exc).__name__}:{exc}")

        for hit in hits:
            project_id = str(hit.get("project_id") or "").strip()
            slug = str(hit.get("slug") or project_id).strip()
            detail = details.get(project_id, hit)
            body = str(detail.get("body") or hit.get("description") or "").strip()
            source_key = project_id or slug
            if not body or not source_key or source_key in seen:
                continue
            seen.add(source_key)
            game_versions = detail.get("game_versions")
            if not isinstance(game_versions, list):
                game_versions = hit.get("versions")
            records.append(
                {
                    "source_id": f"modrinth:{source_key}",
                    "source_type": "modrinth_project_body",
                    "source_locator": f"modrinth:{source_key}",
                    "url": f"https://modrinth.com/mod/{slug}",
                    "title": str(detail.get("title") or hit.get("title") or slug),
                    "content": body,
                    "content_sha256": _sha256_text(body),
                    "body_retrieved": True,
                    "evidence_origin": "modrinth_project_body",
                    "metadata": {
                        "project_id": project_id,
                        "slug": slug,
                        "versions": list(game_versions or []),
                        "source_url": str(detail.get("source_url") or ""),
                    },
                }
            )

        try:
            server_offset = int(payload.get("offset", offset) or offset)
        except (TypeError, ValueError, OverflowError):
            server_offset = offset
        next_offset = server_offset + len(hits)
        if next_offset <= offset:
            errors.append("nonadvancing_search_offset")
            break
        if provider_total and next_offset >= provider_total:
            break
        if len(hits) < page_size and not provider_total:
            break
        offset = next_offset

    return records, {
        "provider": "modrinth",
        "status": "available",
        "result_count": len(records),
        "provider_total": provider_total,
        "search_requests": search_requests,
        "source_requests": source_requests,
        "detail_errors": errors,
    }


def _search_curseforge(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = os.environ.get("CURSEFORGE_API_KEY", "").strip()
    if not key:
        return [], {
            "provider": "curseforge",
            "status": "not_configured",
            "result_count": 0,
        }
    headers = {"x-api-key": key}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[int] = set()
    index = 0
    search_requests = 0
    source_requests = 0
    provider_total = 0

    def description(mod_id: int) -> tuple[int, str, str]:
        try:
            desc = _json(
                f"https://api.curseforge.com/v1/mods/{mod_id}/description",
                headers,
            )
            if isinstance(desc, Mapping) and str(desc.get("data") or "").strip():
                body = " ".join(
                    re.sub(r"<[^>]+>", " ", str(desc["data"])).split()
                )
                return mod_id, body, ""
        except Exception as exc:
            return mod_id, "", f"{mod_id}:{type(exc).__name__}:{exc}"
        return mod_id, "", ""

    while (
        search_requests < _MAX_PROVIDER_SEARCH_PAGES
        and len(records) < _MAX_PROVIDER_RESULTS_PER_QUERY
    ):
        page_size = min(50, _MAX_PROVIDER_RESULTS_PER_QUERY - len(records))
        params = urllib.parse.urlencode(
            {
                "gameId": 432,
                "searchFilter": _query_terms(query),
                "index": index,
                "pageSize": page_size,
                "sortField": 2,
                "sortOrder": "desc",
            }
        )
        payload = _json(
            f"https://api.curseforge.com/v1/mods/search?{params}", headers
        )
        search_requests += 1
        raw_rows = payload.get("data", []) if isinstance(payload, Mapping) else []
        rows = [row for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
        if not rows:
            break

        mod_ids = [
            int(row["id"])
            for row in rows
            if isinstance(row.get("id"), int) and int(row["id"]) not in seen
        ]
        source_requests += len(mod_ids)
        descriptions: dict[int, str] = {}
        if mod_ids:
            workers = min(_MAX_SOURCE_WORKERS, len(mod_ids))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(description, mod_id) for mod_id in mod_ids]
                for future in as_completed(futures):
                    mod_id, body, error = future.result()
                    if body:
                        descriptions[mod_id] = body
                    if error:
                        errors.append(error)

        for row in rows:
            if not isinstance(row.get("id"), int):
                continue
            mod_id = int(row["id"])
            if mod_id in seen:
                continue
            seen.add(mod_id)
            body = descriptions.get(mod_id) or str(row.get("summary") or "").strip()
            if not body:
                continue
            links = row.get("links") if isinstance(row.get("links"), Mapping) else {}
            records.append(
                {
                    "source_id": f"curseforge:{mod_id}",
                    "source_type": "curseforge_mod_body",
                    "source_locator": f"curseforge:{mod_id}",
                    "url": str(links.get("websiteUrl") or ""),
                    "title": str(row.get("name") or mod_id),
                    "content": body,
                    "content_sha256": _sha256_text(body),
                    "body_retrieved": True,
                    "evidence_origin": "curseforge_mod_body",
                    "metadata": {
                        "mod_id": mod_id,
                        "slug": str(row.get("slug") or ""),
                        "source_url": str(links.get("sourceUrl") or ""),
                    },
                }
            )

        pagination = payload.get("pagination") if isinstance(payload, Mapping) else None
        if not isinstance(pagination, Mapping):
            if len(rows) < page_size:
                break
            next_index = index + len(rows)
        else:
            try:
                server_index = int(pagination.get("index", index) or index)
                result_count = int(pagination.get("resultCount", len(rows)) or len(rows))
                provider_total = max(
                    provider_total,
                    int(pagination.get("totalCount", 0) or 0),
                )
            except (TypeError, ValueError, OverflowError):
                server_index = index
                result_count = len(rows)
            next_index = server_index + result_count
            if provider_total and next_index >= provider_total:
                break
        if next_index <= index:
            errors.append("nonadvancing_search_index")
            break
        index = next_index

    return records, {
        "provider": "curseforge",
        "status": "available",
        "result_count": len(records),
        "provider_total": provider_total,
        "search_requests": search_requests,
        "source_requests": source_requests,
        "detail_errors": errors,
    }


def _search_github(
    query: str,
    *,
    disabled: Callable[[], bool] | None = None,
    disable: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if disabled is not None and disabled():
        return [], {
            "provider": "github",
            "status": "disabled_after_rate_or_auth_failure",
            "result_count": 0,
        }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    records: list[dict[str, Any]] = []
    readme_errors: list[str] = []
    seen: set[str] = set()
    page = 1
    consumed = 0
    search_requests = 0
    source_requests = 0
    provider_total = 0

    def repository_body(row: Mapping[str, Any]) -> tuple[str, str, str]:
        full_name = str(row.get("full_name") or "").strip()
        if not full_name:
            return "", "", ""
        body = str(row.get("description") or "").strip()
        try:
            fetched = _text(
                f"https://api.github.com/repos/{full_name}/readme", headers
            ).strip()
            if fetched:
                body = fetched
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 429} and disable is not None:
                disable()
            return full_name, body, f"{full_name}:{type(exc).__name__}:{exc}"
        except Exception as exc:
            return full_name, body, f"{full_name}:{type(exc).__name__}:{exc}"
        return full_name, body, ""

    while (
        search_requests < _MAX_PROVIDER_SEARCH_PAGES
        and len(records) < _MAX_PROVIDER_RESULTS_PER_QUERY
    ):
        if disabled is not None and disabled():
            break
        page_size = min(100, _MAX_PROVIDER_RESULTS_PER_QUERY - len(records))
        params = urllib.parse.urlencode(
            {
                "q": _query_terms(query) + " minecraft fabric mod",
                "per_page": page_size,
                "page": page,
            }
        )
        try:
            payload = _json(
                f"https://api.github.com/search/repositories?{params}", headers
            )
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 429} and disable is not None:
                disable()
            if exc.code == 422 and records:
                readme_errors.append(
                    f"search_page_{page}:{type(exc).__name__}:{exc}"
                )
                break
            raise
        search_requests += 1
        raw_rows = payload.get("items", []) if isinstance(payload, Mapping) else []
        rows = [row for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
        if not rows:
            break
        try:
            provider_total = max(
                provider_total, int(payload.get("total_count", 0) or 0)
            )
        except (TypeError, ValueError, OverflowError):
            provider_total = max(provider_total, consumed + len(rows))

        candidates = [
            row
            for row in rows
            if str(row.get("full_name") or "").strip() not in seen
        ]
        source_requests += len(candidates)
        bodies: dict[str, str] = {}
        if candidates:
            workers = min(_MAX_SOURCE_WORKERS, len(candidates))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(repository_body, row) for row in candidates]
                for future in as_completed(futures):
                    full_name, body, error = future.result()
                    if full_name and body:
                        bodies[full_name] = body
                    if error:
                        readme_errors.append(error)

        for row in rows:
            full_name = str(row.get("full_name") or "").strip()
            if not full_name or full_name in seen:
                continue
            seen.add(full_name)
            body = bodies.get(full_name) or str(row.get("description") or "").strip()
            if not body:
                continue
            records.append(
                {
                    "source_id": f"github:{full_name}",
                    "source_type": "github_repository_body",
                    "source_locator": f"github:{full_name}",
                    "url": str(row.get("html_url") or f"https://github.com/{full_name}"),
                    "title": str(row.get("name") or full_name),
                    "content": body,
                    "content_sha256": _sha256_text(body),
                    "body_retrieved": True,
                    "evidence_origin": "github_readme_body",
                    "metadata": {
                        "repository": full_name,
                        "default_branch": str(row.get("default_branch") or ""),
                    },
                }
            )
        consumed += len(rows)
        if provider_total and consumed >= provider_total:
            break
        page += 1

    status = (
        "disabled_after_rate_or_auth_failure"
        if disabled is not None and disabled()
        else "available"
    )
    return records, {
        "provider": "github",
        "status": status,
        "result_count": len(records),
        "provider_total": provider_total,
        "readme_errors": readme_errors,
        "search_requests": search_requests,
        "source_requests": source_requests,
    }


def _versions(router: Any) -> tuple[str, ...]:
    requested = str(
        getattr(router, "_mmm_requested_minecraft_version", "") or ""
    ).strip()
    existing = str(getattr(router, "_mmm_existing_minecraft_version", "") or "").strip()
    return (requested,) if requested else ((existing,) if existing else ())


def _search_authoritative_catalog(
    query: str, versions: tuple[str, ...]
) -> dict[str, Any]:
    retriever = AuthoritativeEvidenceRetriever()
    records: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    targets = versions or ("",)
    for version in targets:
        try:
            catalog = (
                evidence_catalog_for_version(version)
                if version
                else target_neutral_evidence_catalog()
            )
            kwargs = {"minecraft_version": version} if version else {}
            for source in retriever.search(query, limit=len(catalog), **kwargs):
                item = asdict(source)
                item["matched_version"] = version
                records.setdefault(source.source_id, item)
        except Exception as exc:
            errors.append(
                {"minecraft_version": version, "error": f"{type(exc).__name__}: {exc}"}
            )
    return {
        "schema_version": "mmm/project-rag-query-v3",
        "sources": list(records.values()),
        "errors": errors,
    }


def _existing_code_index() -> Path | None:
    values = [os.environ.get("MMM_PROJECT_RAG_INDEX", ""), "rag/project-index.json"]
    if os.environ.get("MMM_WORKSPACE"):
        values.append(str(Path(os.environ["MMM_WORKSPACE"]) / "rag/project-index.json"))
    for raw in values:
        if not str(raw).strip():
            continue
        path = Path(str(raw)).expanduser().resolve()
        if path.is_file():
            return path
    return None


def _search_code_index(index: Path | None, query: str) -> dict[str, Any]:
    if index is None:
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "not_indexed",
            "hits": [],
        }
    try:
        result = ProjectRAGIndex(index).search_with_receipt(
            query, limit=8, semantic=True, rerank=True
        )
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "searched",
            "hits": [asdict(hit) for hit in result.hits],
            "receipt": asdict(result.receipt),
        }
    except Exception as exc:
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "error",
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _github_repo_from_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlparse(str(value or "").strip())
    except ValueError:
        return ""
    if parsed.netloc.casefold() not in {"github.com", "www.github.com"}:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    owner, repo = parts[0], parts[1]
    repo = repo.removesuffix(".git")
    return f"{owner}/{repo}" if owner and repo else ""


def _linked_github_sources(
    records: list[dict[str, Any]],
    *,
    disabled: Callable[[], bool],
    disable: Callable[[], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repositories: list[str] = []
    for record in records:
        metadata = record.get("metadata")
        source_url = (
            str(metadata.get("source_url") or "")
            if isinstance(metadata, Mapping)
            else ""
        )
        repo = _github_repo_from_url(source_url)
        if repo and repo not in repositories:
            if len(repositories) >= _MAX_PROVIDER_RESULTS_PER_QUERY:
                continue
            repositories.append(repo)
    if not repositories:
        return [], {
            "provider": "github",
            "status": "skipped_no_linked_source",
            "result_count": 0,
            "search_requests": 0,
            "source_requests": 0,
        }
    if disabled():
        return [], {
            "provider": "github",
            "status": "disabled_after_rate_or_auth_failure",
            "result_count": 0,
            "search_requests": 0,
            "source_requests": 0,
        }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch_readme(full_name: str) -> tuple[str, str, str]:
        try:
            body = _text(
                f"https://api.github.com/repos/{full_name}/readme", headers
            ).strip()
            return full_name, body, ""
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 429}:
                disable()
            return full_name, "", f"{full_name}:{type(exc).__name__}:{exc}"
        except Exception as exc:
            return full_name, "", f"{full_name}:{type(exc).__name__}:{exc}"

    workers = min(_MAX_SOURCE_WORKERS, len(repositories))
    if workers <= 1:
        fetched = [fetch_readme(full_name) for full_name in repositories]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = list(pool.map(fetch_readme, repositories))

    found: list[dict[str, Any]] = []
    errors: list[str] = []
    for full_name, body, error in fetched:
        if error:
            errors.append(error)
        if not body:
            continue
        found.append(
            {
                "source_id": f"github:{full_name}",
                "source_type": "github_repository_body",
                "source_locator": f"github:{full_name}",
                "url": f"https://github.com/{full_name}",
                "title": full_name.rsplit("/", 1)[-1],
                "content": body,
                "content_sha256": _sha256_text(body),
                "body_retrieved": True,
                "evidence_origin": "github_linked_source_readme",
                "metadata": {"repository": full_name},
            }
        )
    return found, {
        "provider": "github",
        "status": "available" if found else "linked_source_unavailable",
        "result_count": len(found),
        "search_requests": 0,
        "source_requests": len(repositories),
        "readme_errors": errors[:3],
    }


def _forced_rag_bundle(
    router: Any, research_brief: Mapping[str, Any]
) -> dict[str, Any]:
    domains = (
        [x for x in research_brief.get("domains", []) if isinstance(x, Mapping)]
        if isinstance(research_brief.get("domains"), list)
        else []
    )
    versions = _versions(router)
    code_index = _existing_code_index()
    github_blocked = False
    github_state_lock = threading.Lock()
    github_fallback_lock = threading.Lock()

    def disabled() -> bool:
        with github_state_lock:
            return github_blocked

    def disable() -> None:
        nonlocal github_blocked
        with github_state_lock:
            github_blocked = True

    def run_query(query: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        receipts: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []

        primary_calls: dict[str, Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]]] = {
            "modrinth": lambda: _search_modrinth(query),
        }
        if os.environ.get("CURSEFORGE_API_KEY", "").strip():
            primary_calls["curseforge"] = lambda: _search_curseforge(query)
        else:
            receipts["curseforge"] = {
                "provider": "curseforge",
                "status": "not_configured",
                "result_count": 0,
            }

        with ThreadPoolExecutor(max_workers=len(primary_calls)) as pool:
            futures = {pool.submit(call): provider for provider, call in primary_calls.items()}
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    found, receipt = future.result()
                    records.extend(found)
                    receipts[provider] = receipt
                except Exception as exc:
                    receipt = _error(provider, exc)
                    receipts[provider] = receipt
                    errors.append(receipt)

        linked, linked_receipt = _linked_github_sources(
            records, disabled=disabled, disable=disable
        )
        records.extend(linked)
        receipts["github"] = linked_receipt

        # Ecosystem descriptions are useful design evidence, but they are not source
        # repositories. Source-reuse mode must still search GitHub when Modrinth or
        # CurseForge returned metadata without a linked repository; otherwise the
        # downstream immutable donor pipeline is guaranteed to receive zero candidates.
        if not linked:
            with github_fallback_lock:
                try:
                    found, receipt = _search_github(
                        query, disabled=disabled, disable=disable
                    )
                    records.extend(found)
                    receipts["github"] = receipt
                except Exception as exc:
                    receipt = _error("github", exc)
                    receipts["github"] = receipt
                    errors.append(receipt)

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            key = str(record.get("source_id") or record.get("url") or "")
            if key and key not in seen:
                seen.add(key)
                unique.append(record)
        gh = receipts.get("github", {}) if isinstance(receipts.get("github"), Mapping) else {}
        return {
            "query": query,
            "query_sha256": _sha256_text(query),
            "project_rag": _search_authoritative_catalog(query, versions),
            "code_rag": _search_code_index(code_index, query),
            "external_rag": {
                "schema_version": "mmm/external-pre-design-discovery-v3",
                "sources": unique,
                "errors": errors,
                "providers": receipts,
                "github_retrieval": {
                    "provider_status": str(gh.get("status") or "not_requested"),
                    "saturation_reason": (
                        "rate_or_auth_failure"
                        if str(gh.get("status"))
                        in {"error", "disabled_after_rate_or_auth_failure"}
                        else ""
                    ),
                    "search_requests": int(gh.get("search_requests") or 0),
                    "source_requests": int(gh.get("source_requests") or 0),
                },
            },
        }

    unique_queries: list[str] = []
    for domain in domains:
        raw_queries = domain.get("queries", [])
        for raw in raw_queries if isinstance(raw_queries, list) else []:
            query = str(raw or "").strip()
            if query and query not in unique_queries:
                unique_queries.append(query)

    by_query: dict[str, dict[str, Any]] = {}
    if unique_queries:
        workers = min(_MAX_QUERY_WORKERS, len(unique_queries))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_query, query): query for query in unique_queries}
            for future in as_completed(futures):
                query = futures[future]
                try:
                    by_query[query] = future.result()
                except Exception as exc:
                    by_query[query] = {
                        "query": query,
                        "query_sha256": _sha256_text(query),
                        "project_rag": {"sources": [], "errors": []},
                        "code_rag": {"status": "error", "hits": []},
                        "external_rag": {
                            "schema_version": "mmm/external-pre-design-discovery-v3",
                            "sources": [],
                            "errors": [_error("query_worker", exc)],
                            "providers": {},
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
        rows: list[dict[str, Any]] = []
        raw_queries = domain.get("queries", [])
        for raw in raw_queries if isinstance(raw_queries, list) else []:
            query = str(raw or "").strip()
            if not query:
                continue
            query_count += 1
            row = dict(by_query[query])
            rows.append(row)
            external = row.get("external_rag")
            if isinstance(external, Mapping):
                sources = external.get("sources")
                external_count += len(sources) if isinstance(sources, list) else 0
        out_domains.append(
            {"domain_id": str(domain.get("domain_id") or ""), "queries": rows}
        )
    payload: dict[str, Any] = {
        "schema_version": "mmm/pre-design-grounded-rag-v5",
        "versions": list(versions),
        "domain_count": len(domains),
        "query_count": query_count,
        "unique_query_count": len(unique_queries),
        "query_workers": min(_MAX_QUERY_WORKERS, len(unique_queries)) if unique_queries else 0,
        "provider_result_budget": _MAX_PROVIDER_RESULTS_PER_QUERY,
        "provider_search_page_budget": _MAX_PROVIDER_SEARCH_PAGES,
        "external_source_count": external_count,
        "domains": out_domains,
    }
    payload["research_sha256"] = _sha256(payload)
    return payload


def _body(record: Mapping[str, Any]) -> str:
    return str(
        record.get("content") or record.get("body") or record.get("text") or ""
    ).strip()


def _units(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    grounded = (
        evidence.get("grounded_rag")
        if isinstance(evidence.get("grounded_rag"), Mapping)
        else {}
    )
    result: list[dict[str, Any]] = []
    seen_bodies: set[str] = set()
    for row in (
        grounded.get("queries", []) if isinstance(grounded.get("queries"), list) else []
    ):
        if not isinstance(row, Mapping):
            continue
        for record in (
            row.get("evidence_records", [])
            if isinstance(row.get("evidence_records"), list)
            else []
        ):
            if not isinstance(record, Mapping):
                continue
            body = _body(record)
            if not body:
                continue
            body_key = str(record.get("content_sha256") or "").strip() or _sha256_text(body)
            if body_key in seen_bodies:
                continue
            seen_bodies.add(body_key)
            result.append(
                {
                    "query": str(row.get("query") or ""),
                    "source_id": str(record.get("source_id") or ""),
                    "source_type": str(record.get("source_type") or ""),
                    "url": str(record.get("url") or ""),
                    "title": str(record.get("title") or ""),
                    "content_sha256": str(
                        record.get("content_sha256") or _sha256_text(body)
                    ),
                    "content": body,
                }
            )
    return result


def _root() -> Path:
    configured = os.environ.get("MMM_RESEARCH_DOCUMENT_DIR", "").strip()
    workspace = os.environ.get("MMM_WORKSPACE", "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else (Path(workspace).expanduser() if workspace else Path.cwd())
        / "mmm-output"
        / "research-evidence"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    try:
        if path.is_file() and path.read_bytes() == encoded:
            return
    except OSError:
        # Preserve the existing atomic rewrite as the repair path for unreadable files.
        pass
    name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            name = handle.name
        os.replace(name, path)
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def _materialize_domain_evidence_document(
    domain_id: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    raw = json.dumps(dict(evidence), ensure_ascii=False, sort_keys=True, default=str)
    digest = _sha256_text(raw)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain_id).strip("_") or "unknown"
    root = _root()
    raw_path = root / f"{safe}-{digest[7:19]}.json"
    pages_path = root / f"{safe}-{digest[7:19]}.pages.jsonl"
    units = _units(evidence)
    pages: list[dict[str, Any]] = []
    for ui, unit in enumerate(units):
        rendered = json.dumps(
            unit, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        pages.append(
            {
                "schema_version": "mmm/research-evidence-page-v2",
                "domain_id": domain_id,
                "unit_id": f"source:{ui}",
                "part_index": 0,
                "part_count": 1,
                "content": rendered,
            }
        )
    for index, page in enumerate(pages):
        page.update(
            {
                "page_index": index,
                "page_count": len(pages),
                "page_ref": f"{digest}#page={index + 1}/{len(pages)}",
            }
        )
    _write(raw_path, raw)
    _write(
        pages_path,
        "".join(
            json.dumps(page, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for page in pages
        ),
    )
    return {
        "schema_version": "mmm/research-evidence-document-v2",
        "domain_id": domain_id,
        "document_sha256": digest,
        "raw_path": str(raw_path),
        "pages_path": str(pages_path),
        "page_count": len(pages),
        "page_partition": "claim_bearing_source_unit",
        "source_keys": sorted(str(key) for key in evidence),
        "model_unit_count": len(units),
        "model_projection": "claim_bearing_source_bodies_only",
    }


def _read_evidence_pages(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = int(document.get("page_count") or 0)
    if expected == 0:
        return []
    path = Path(str(document.get("pages_path") or "")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    pages = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(pages) != expected:
        raise ValueError(
            f"Research evidence page count mismatch: expected {expected}, got {len(pages)}"
        )
    return pages


def _prompt_document_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "domain_id",
        "document_sha256",
        "page_count",
        "page_partition",
        "source_keys",
        "model_unit_count",
        "model_projection",
    )
    return {k: document[k] for k in keep if k in document}


__all__ = [
    "_forced_rag_bundle",
    "_materialize_domain_evidence_document",
    "_prompt_document_receipt",
    "_read_evidence_pages",
]
