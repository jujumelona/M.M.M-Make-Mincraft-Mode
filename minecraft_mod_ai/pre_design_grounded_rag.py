from __future__ import annotations

"""Transport and evidence storage for host-owned Minecraft research.

Retrieval policy intentionally lives in ``catalog_first_grounded_rag``. This module owns
provider I/O and exact evidence materialization only. CurseForge browsing
is intentionally bounded to multiple catalogue windows; receipts identify these
samples as incomplete. Other providers retain their own retrieval budgets.
"""

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from threading import Lock
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .curseforge_catalog import (
    curseforge_api_key,
    curseforge_mod_metadata,
    curseforge_mods_class_id,
    curseforge_next_index,
    curseforge_target_filters,
)
from .deadline_executor import iter_completed_with_deadlines
from .knowledge import (
    AuthoritativeEvidenceRetriever,
    evidence_catalog_for_version,
    target_neutral_evidence_catalog,
)
from .rag_index import ProjectRAGIndex
from .source_intent_aliases import source_intent_terms, source_query_variants
from .curseforge_query import curseforge_search_queries, curseforge_catalogue_score

_TIMEOUT = 8.0
_UA = "MMM-PreDesignResearch/3.0 (+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"


def _positive_env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _worker_count(task_count: int, env_name: str) -> int:
    """Use all independent tasks unless the operator explicitly constrains fan-out."""
    count = max(0, int(task_count))
    if count <= 1:
        return count
    configured = _positive_env_int(env_name)
    return min(count, configured) if configured is not None else count


def _query_worker_count(task_count: int) -> int:
    return _worker_count(task_count, "MMM_PREDESIGN_QUERY_WORKERS")


def _source_worker_count(task_count: int) -> int:
    return _worker_count(task_count, "MMM_PREDESIGN_SOURCE_WORKERS")


def _provider_result_limit() -> int | None:
    return min(8, _positive_env_int("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY") or 4)


def _provider_page_limit() -> int | None:
    return min(2, _positive_env_int("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES") or 1)


def _limit_reached(value: int, limit: int | None) -> bool:
    return limit is not None and value >= limit


def _search_terms(value: str) -> tuple[str, ...]:
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
    return tuple(words)


def _query_terms(value: str) -> str:
    words = _search_terms(value)
    return " ".join(words) or "minecraft fabric"


def _semantic_page_size(
    query: str,
    provider_max: int,
    result_limit: int | None,
    current: int,
) -> int:
    """Use provider pagination capacity; query word count is not a recall budget."""
    remaining = None if result_limit is None else max(0, result_limit - current)
    if remaining == 0:
        return 0
    page_size = provider_max
    return page_size if remaining is None else min(page_size, remaining)


def _coverage_gain(
    wanted: set[str],
    covered: set[str],
    values: Sequence[Any],
) -> int:
    if not wanted:
        return 1
    observed: set[str] = set()
    for value in values:
        observed.update(_search_terms(str(value or "")))
    before = len(covered)
    covered.update(wanted & observed)
    return len(covered) - before


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
    seen_records: set[str] = set()
    seen_search_keys: set[str] = set()
    covered: set[str] = set()
    wanted = set(_search_terms(query)) | set(source_intent_terms(query))
    search_variants = source_query_variants(query) or (query,)
    search_requests = 0
    source_requests = 0
    provider_total = 0
    result_limit = _provider_result_limit()
    page_limit = _provider_page_limit()

    # Search each semantic variant independently. A Korean request must not
    # become one long, untranslated Modrinth search phrase; the host performs
    # these queries, not the small model.
    for search_phrase in search_variants:
        offset = 0
        variant_requests = 0
        variant_total = 0
        while not _limit_reached(variant_requests, page_limit) and not _limit_reached(
            len(records), result_limit
        ):
            page_size = _semantic_page_size(query, 100, result_limit, len(records))
            if page_size <= 0:
                break
            params = urllib.parse.urlencode(
                {
                    "query": _query_terms(search_phrase),
                    "limit": page_size,
                    "offset": offset,
                    "index": "relevance",
                    "facets": json.dumps([["project_type:mod"]]),
                }
            )
            try:
                payload = _json(f"https://api.modrinth.com/v2/search?{params}")
            except (OSError, ValueError) as exc:
                if not records:
                    raise
                errors.append(f"partial_search_page:{type(exc).__name__}:{exc}")
                break
            search_requests += 1
            variant_requests += 1
            raw_hits = payload.get("hits", []) if isinstance(payload, Mapping) else []
            hits = (
                [hit for hit in raw_hits if isinstance(hit, Mapping)]
                if isinstance(raw_hits, list)
                else []
            )
            if not hits:
                break

            try:
                variant_total = max(variant_total, int(payload.get("total_hits", 0) or 0))
            except (TypeError, ValueError, OverflowError):
                variant_total = max(variant_total, offset + len(hits))
            provider_total = max(provider_total, variant_total)

            search_keys = {
                str(hit.get("project_id") or hit.get("slug") or "").strip()
                for hit in hits
                if str(hit.get("project_id") or hit.get("slug") or "").strip()
            }
            if search_keys and not (search_keys - seen_search_keys):
                if variant_requests > 1:
                    errors.append("duplicate_only_search_page")
                break  # Another semantic variant may find new donors.
            seen_search_keys.update(search_keys)

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
                    fetched = _json(f"https://api.modrinth.com/v2/projects?{detail_params}")
                    if isinstance(fetched, list):
                        details = {
                            str(item.get("id") or "").strip(): item
                            for item in fetched
                            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
                        }
                except Exception as exc:
                    errors.append(f"bulk-projects:{type(exc).__name__}:{exc}")

            coverage_values: list[str] = []
            for hit in hits:
                project_id = str(hit.get("project_id") or "").strip()
                slug = str(hit.get("slug") or project_id).strip()
                detail = details.get(project_id, hit)
                body = str(detail.get("body") or hit.get("description") or "").strip()
                coverage_values.extend(
                    [
                        str(detail.get("title") or hit.get("title") or slug),
                        str(hit.get("description") or ""),
                        body,
                    ]
                )
                source_key = project_id or slug
                if not body or not source_key or source_key in seen_records:
                    continue
                seen_records.add(source_key)
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
                            "loaders": list(detail.get("loaders") or []),
                            "license": detail.get("license"),
                            "source_url": str(detail.get("source_url") or ""),
                        },
                    }
                )
                if _limit_reached(len(records), result_limit):
                    break

            _coverage_gain(wanted, covered, coverage_values)
            try:
                server_offset = int(payload.get("offset", offset) or offset)
            except (TypeError, ValueError, OverflowError):
                server_offset = offset
            next_offset = server_offset + len(hits)
            if next_offset <= offset:
                errors.append("nonadvancing_search_offset")
                break
            if variant_total and next_offset >= variant_total:
                break
            if len(hits) < page_size and not variant_total:
                break
            offset = next_offset

    if len(search_variants) > 1:
        # Real title/body evidence outranks search result position. Run this
        # only after collecting all independent search variants.
        requested = set(source_intent_terms(query))
        records.sort(key=lambda record: (
            len(requested & set(source_intent_terms(
                str(record.get("title") or "") + " " + str(record.get("content") or "")
            ))),
            bool((record.get("metadata") or {}).get("source_url")),
        ), reverse=True)

    return records, {
        "provider": "modrinth",
        "status": "available",
        "result_count": len(records),
        "provider_total": provider_total,
        "search_variants": list(search_variants),
        "search_requests": search_requests,
        "source_requests": source_requests,
        "detail_errors": errors,
        "retrieval_complete": (len(search_variants) == 1 and not errors
                               and (not provider_total or len(seen_search_keys) >= provider_total)),
        "retrieval_stop": "provider_exhaustion_or_explicit_budget",
    }


_CURSEFORGE_BROWSE_LOCK = Lock()
_CURSEFORGE_BROWSE_CACHE: dict[tuple[Callable[..., Any], str, str], tuple[float, Any]] = {}


def _curseforge_browse_json(url: str, headers: Mapping[str, str]) -> Any:
    """Deduplicate one-run catalogue requests across independent feature queries.

    This cache contains recently fetched API metadata, not previously verified
    source code. A test-supplied transport is isolated by its function identity.
    """
    credential_hash = hashlib.sha256(
        str(headers.get("x-api-key") or "").encode("utf-8")
    ).hexdigest()
    cache_key = (_json, credential_hash, url)
    with _CURSEFORGE_BROWSE_LOCK:
        now = time.monotonic()
        cached = _CURSEFORGE_BROWSE_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < 300:
            return cached[1]
        result = _json(url, headers)
        if len(_CURSEFORGE_BROWSE_CACHE) >= 128:
            _CURSEFORGE_BROWSE_CACHE.clear()
        _CURSEFORGE_BROWSE_CACHE[cache_key] = (time.monotonic(), result)
        return result


def _search_curseforge(
    query: str,
    *,
    minecraft_version: str | None = None,
    loader: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find actual matching mods using targeted provider-side keyword searches.

    CurseForge searches mod names/authors, not prose descriptions. Produce short
    query variants from the real requirement, then rerank by full intent and
    fetch description bodies for only the best target-filtered results.
    """
    variants = curseforge_search_queries(query, max_variants=3)
    if not variants:
        return [], {
            "provider": "curseforge", "status": "query_unresolved",
            "result_count": 0, "search_variants": [],
            "retrieval_complete": False,
        }
    key = curseforge_api_key()
    if not key:
        return [], {
            "provider": "curseforge", "status": "not_configured",
            "result_count": 0, "search_variants": list(variants),
        }
    try:
        target_filters = curseforge_target_filters(minecraft_version, loader)
    except ValueError as exc:
        return [], {
            "provider": "curseforge", "status": "provider_error", "result_count": 0,
            "detail_errors": [f"invalid_target:{exc}"], "retrieval_complete": False,
        }

    headers = {"x-api-key": key}
    errors: list[str] = []
    try:
        categories = _curseforge_browse_json(
            "https://api.curseforge.com/v1/categories?gameId=432", headers
        )
        mods_class = curseforge_mods_class_id(
            categories.get("data") if isinstance(categories, Mapping) else None
        )
    except (OSError, ValueError) as exc:
        mods_class = None
        errors.append("categories_unavailable:" + type(exc).__name__)
    if mods_class is None:
        return [], {
            "provider": "curseforge", "status": "provider_error", "result_count": 0,
            "detail_errors": [*errors, "mods_class_unavailable"],
            "retrieval_complete": False,
        }

    seen_ids: set[int] = set()
    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    search_requests = 0
    source_requests = 0
    provider_total = 0
    page_limit = _provider_page_limit() or 1
    output_limit = _provider_result_limit() or 4
    # Fetch only a small shortlist, never a full 50-mod provider page.
    page_size = min(12, output_limit * 2)
    query_receipts: list[dict[str, Any]] = []
    for variant in variants:
        index, variant_matches, variant_requests = 0, 0, 0
        for _ in range(page_limit):
            params = {
                "gameId": 432, "index": index, "pageSize": page_size,
                "sortField": 13, "sortOrder": "desc", "classId": mods_class,
                "searchFilter": variant,
            }
            params.update(target_filters)
            url = "https://api.curseforge.com/v1/mods/search?" + urllib.parse.urlencode(params)
            try:
                payload = _curseforge_browse_json(url, headers)
            except (OSError, ValueError) as exc:
                errors.append(f"catalogue_page_failed:{variant}:{type(exc).__name__}")
                break
            search_requests += 1
            variant_requests += 1
            raw_rows = payload.get("data") if isinstance(payload, Mapping) else None
            if not isinstance(raw_rows, list):
                errors.append(f"invalid_catalogue_page:{variant}")
                break
            for row in raw_rows:
                if not isinstance(row, Mapping) or type(row.get("id")) is not int:
                    continue
                mod_id = row["id"]
                if mod_id in seen_ids or row.get("classId") != mods_class:
                    continue
                seen_ids.add(mod_id)
                score = curseforge_catalogue_score(query, row)
                if score == 0:
                    continue
                variant_matches += 1
                # Existing source links are not proof of source availability.
                links = row.get("links") if isinstance(row.get("links"), Mapping) else {}
                candidates.append((score, int(bool(links.get("sourceUrl"))), row))
            pagination = payload.get("pagination") if isinstance(payload, Mapping) else None
            if isinstance(pagination, Mapping):
                try:
                    provider_total = max(provider_total, int(pagination.get("totalCount") or 0))
                except (TypeError, ValueError, OverflowError):
                    errors.append("invalid_total_count")
            try:
                next_index = curseforge_next_index(
                    pagination, index=index, page_size=page_size, row_count=len(raw_rows)
                )
            except ValueError:
                errors.append(f"invalid_pagination:{variant}")
                break
            if next_index is None:
                break
            index = next_index
            if variant_matches >= output_limit:
                break
        query_receipts.append({
            "query": variant, "search_requests": variant_requests,
            "relevant_new_mods": variant_matches,
        })
        # Move to implementation once this topic has enough plausible options.
        # Remaining spellings are fallback searches, not an exhaustive sweep.
        if len(candidates) >= min(3, output_limit):
            break

    candidates.sort(
        key=lambda row: (-row[0], -row[1],
                         str(row[2].get("name") or "").casefold(), row[2]["id"])
    )
    shortlisted = candidates[:output_limit]
    detail_count = min(3, len(shortlisted))
    details: dict[int, str] = {}
    if detail_count:
        jobs = tuple(enumerate(row[2]["id"] for row in shortlisted[:detail_count]))

        def fetch_description(job: tuple[int, int]) -> tuple[int, str, str]:
            position, mod_id = job
            try:
                payload = _json(
                    f"https://api.curseforge.com/v1/mods/{mod_id}/description", headers
                )
                if isinstance(payload, Mapping):
                    body = " ".join(
                        re.sub(r"<[^>]+>", " ", str(payload.get("data") or "")).split()
                    )
                    return position, body, ""
            except (OSError, ValueError) as exc:
                return position, "", type(exc).__name__
            return position, "", ""

        results: dict[int, tuple[str, str]] = {}
        for _job, found in iter_completed_with_deadlines(
            jobs, fetch_description,
            max_workers=max(1, _source_worker_count(len(jobs))),
            stage="predesign-curseforge-descriptions",
            sort_key=lambda item: item[0],
        ):
            position, body, error = found
            results[position] = body, error
        source_requests = len(jobs)
        for position, (_, _, mod) in enumerate(shortlisted[:detail_count]):
            body, error = results.get(position, ("", "missing_detail_result"))
            if body:
                details[mod["id"]] = body
            if error:
                errors.append(f"{mod['id']}:{error}")

    records: list[dict[str, Any]] = []
    for score, _bonus, row in shortlisted:
        mod_id = row["id"]
        body = details.get(mod_id)
        summary = str(row.get("summary") or "").strip()
        text = body or summary
        if not text:
            continue
        links = row.get("links") if isinstance(row.get("links"), Mapping) else {}
        metadata = curseforge_mod_metadata(
            row, minecraft_version=minecraft_version, loader=loader
        )
        metadata["retrieval_relevance_score"] = score
        metadata["source_reuse_authority"] = "catalogue_only"
        records.append({
            "source_id": f"curseforge:{mod_id}",
            "source_type": "curseforge_mod_body" if body else "curseforge_mod_summary",
            "source_locator": f"curseforge:{mod_id}",
            "url": str(links.get("websiteUrl") or ""),
            "title": str(row.get("name") or mod_id),
            "content": text,
            "content_sha256": _sha256_text(text),
            "body_retrieved": bool(body),
            "evidence_origin": "curseforge_mod_body" if body else "curseforge_mod_summary",
            "metadata": metadata,
        })

    return records, {
        "provider": "curseforge",
        "status": "provider_error" if errors and not search_requests else "available",
        "result_count": len(records), "provider_total": provider_total,
        "search_requests": search_requests, "source_requests": source_requests,
        "detail_errors": errors, "authenticated": True,
        "target_filters": dict(target_filters),
        "search_filter_used": True,
        "search_variants": [row["query"] for row in query_receipts],
        "query_receipts": query_receipts,
        "catalog_class_verified": True,
        "retrieval_complete": False,
        "retrieval_stop": "bounded_targeted_search_not_exhaustive",
    }


def _versions(router: Any) -> tuple[str, ...]:
    requested = str(
        getattr(router, "_mmm_requested_minecraft_version", "") or ""
    ).strip()
    existing = str(
        getattr(router, "_mmm_existing_minecraft_version", "") or ""
    ).strip()
    return (requested,) if requested else ((existing,) if existing else ())


def _search_authoritative_catalog(
    query: str, versions: tuple[str, ...]
) -> dict[str, Any]:
    retriever = AuthoritativeEvidenceRetriever()
    records: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for version in versions or ("",):
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
                {
                    "minecraft_version": version,
                    "error": f"{type(exc).__name__}: {exc}",
                }
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
            query,
            limit=max(1, len(_search_terms(query))),
            semantic=False,
            rerank=False,
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
        grounded.get("queries", [])
        if isinstance(grounded.get("queries"), list)
        else []
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
        pass

    name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            name = handle.name

        for attempt in range(4):
            try:
                os.replace(name, path)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def _build_evidence_pages(
    domain_id: str,
    evidence: Mapping[str, Any],
    digest: str,
) -> tuple[list[dict[str, Any]], str]:
    units = _units(evidence)
    pages: list[dict[str, Any]] = []

    for unit_index, unit in enumerate(units):
        rendered = json.dumps(
            unit, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        pages.append(
            {
                "schema_version": "mmm/research-evidence-page-v2",
                "domain_id": domain_id,
                "unit_id": f"source:{unit_index}",
                "part_index": 0,
                "part_count": 1,
                "content": rendered,
            }
        )

    total_pages = len(pages)
    for index, page in enumerate(pages):
        page.update(
            {
                "page_index": index,
                "page_count": total_pages,
                "page_ref": f"{digest}#page={index + 1}/{total_pages}",
            }
        )

    page_lines: list[str] = []
    for page in pages:
        rendered_page = json.dumps(
            page,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        clean_page = (
            rendered_page.replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085")
        )
        page_lines.append(clean_page)

    serialized = "".join(f"{line}\n" for line in page_lines)
    return pages, serialized


def _materialize_domain_evidence_document(
    domain_id: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    raw = json.dumps(dict(evidence), ensure_ascii=False, sort_keys=True, default=str)
    digest = _sha256_text(raw)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain_id).strip("_") or "unknown"
    root = _root()
    raw_path = root / f"{safe}-{digest[7:19]}.json"
    pages_path = root / f"{safe}-{digest[7:19]}.pages.jsonl"

    pages, pages_text = _build_evidence_pages(domain_id, evidence, digest)
    _write(raw_path, raw)
    _write(pages_path, pages_text)

    return {
        "schema_version": "mmm/research-evidence-document-v2",
        "domain_id": domain_id,
        "document_sha256": digest,
        "raw_path": str(raw_path),
        "pages_path": str(pages_path),
        "page_count": len(pages),
        "page_partition": "claim_bearing_source_unit",
        "source_keys": sorted(str(key) for key in evidence),
        "model_unit_count": len(pages),
        "model_projection": "claim_bearing_source_bodies_only",
        "_pages": pages,
    }


def _read_evidence_pages(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = int(document.get("page_count") or 0)
    if expected == 0:
        return []

    cached = document.get("_pages")
    if isinstance(cached, list) and len(cached) == expected:
        return [dict(page) for page in cached if isinstance(page, Mapping)]

    path = Path(str(document.get("pages_path") or "")).expanduser()
    raw_path = Path(str(document.get("raw_path") or "")).expanduser()
    pages: list[dict[str, Any]] = []
    read_ok = False

    if path.is_file():
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.strip("\r\n")
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    if isinstance(record, Mapping):
                        pages.append(dict(record))
            if len(pages) == expected:
                read_ok = True
        except (OSError, json.JSONDecodeError, ValueError):
            pages = []
            read_ok = False

    if not read_ok and raw_path.is_file():
        try:
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
            evidence_dict = json.loads(raw_text)
            if isinstance(evidence_dict, Mapping):
                domain_id = str(document.get("domain_id") or "")
                digest = str(document.get("document_sha256") or "") or _sha256_text(raw_text)
                reconstructed, pages_text = _build_evidence_pages(
                    domain_id, evidence_dict, digest
                )
                _write(path, pages_text)
                pages = reconstructed
                read_ok = True
        except Exception:
            pass

    if not read_ok and not pages:
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
        except OSError:
            pass
        return []

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
    return {key: document[key] for key in keep if key in document}


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


__all__ = [
    "_error",
    "_existing_code_index",
    "_materialize_domain_evidence_document",
    "_prompt_document_receipt",
    "_query_worker_count",
    "_read_evidence_pages",
    "_search_authoritative_catalog",
    "_search_code_index",
    "_search_curseforge",
    "_search_modrinth",
    "_sha256",
    "_sha256_text",
    "_source_worker_count",
    "_versions",
]
