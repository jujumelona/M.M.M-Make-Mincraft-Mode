from __future__ import annotations

"""Host-owned pre-design retrieval with provider-level fault isolation.

External discovery is advisory implementation evidence. A failed provider must never
abort the authored requirement or the remaining requirements. Modrinth is queried first,
CurseForge is used only when an API key exists, and GitHub is a best-effort fallback.
"""

import hashlib
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .knowledge import (
    AuthoritativeEvidenceRetriever,
    evidence_catalog_for_version,
    target_neutral_evidence_catalog,
)
from .rag_index import ProjectRAGIndex

_EVIDENCE_PAGE_BYTES = 1_800
_HTTP_TIMEOUT_SECONDS = 8.0
_USER_AGENT = "MMM-PreDesignResearch/1.0 (+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return _sha256_text(rendered)


def _query_terms(value: str, *, limit: int = 10) -> str:
    stop = {
        "minecraft", "fabric", "mod", "mods", "game", "system", "feature",
        "player", "players", "requested", "implementation", "existing",
    }
    result: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_+.#/-]+|[가-힣]{2,}", value):
        folded = token.casefold()
        if len(folded) < 3 or folded in stop or folded in result:
            continue
        result.append(folded)
        if len(result) >= limit:
            break
    return " ".join(result) or "minecraft fabric"


def _request_json(url: str, *, headers: Mapping[str, str] | None = None) -> Any:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
        **dict(headers or {}),
    }
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _request_text(url: str, *, headers: Mapping[str, str] | None = None) -> str:
    request_headers = {
        "Accept": "text/plain, application/vnd.github.raw+json, */*",
        "User-Agent": _USER_AGENT,
        **dict(headers or {}),
    }
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def _provider_error(provider: str, exc: BaseException) -> dict[str, Any]:
    status = getattr(exc, "code", None)
    return {
        "provider": provider,
        "status": "error",
        "http_status": int(status) if isinstance(status, int) else None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _search_modrinth(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    search_query = _query_terms(query)
    facets = json.dumps([["project_type:mod"]], separators=(",", ":"))
    params = urllib.parse.urlencode(
        {"query": search_query, "limit": 3, "index": "relevance", "facets": facets}
    )
    payload = _request_json(f"https://api.modrinth.com/v2/search?{params}")
    hits = payload.get("hits", []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    detail_errors: list[str] = []
    for hit in hits[:3] if isinstance(hits, list) else []:
        if not isinstance(hit, Mapping):
            continue
        project_id = str(hit.get("project_id") or "").strip()
        slug = str(hit.get("slug") or project_id).strip()
        detail: Mapping[str, Any] = hit
        if project_id:
            try:
                candidate = _request_json(
                    "https://api.modrinth.com/v2/project/"
                    + urllib.parse.quote(project_id, safe="")
                )
                if isinstance(candidate, Mapping):
                    detail = candidate
            except Exception as exc:
                detail_errors.append(f"{project_id}:{type(exc).__name__}:{exc}")
        body = str(detail.get("body") or hit.get("description") or "").strip()
        if not body:
            continue
        records.append(
            {
                "source_id": f"modrinth:{project_id or slug}",
                "source_type": "modrinth_project",
                "source_locator": f"modrinth:{project_id or slug}",
                "url": f"https://modrinth.com/mod/{slug}",
                "title": str(detail.get("title") or hit.get("title") or slug),
                "content": body,
                "content_sha256": _sha256_text(body),
                "evidence_origin": "modrinth_discovery",
                "metadata": {
                    "project_id": project_id,
                    "slug": slug,
                    "project_type": str(hit.get("project_type") or ""),
                    "versions": list(hit.get("versions") or [])[:20],
                    "categories": list(hit.get("categories") or [])[:20],
                },
            }
        )
    return records, {
        "provider": "modrinth",
        "status": "available",
        "query": search_query,
        "result_count": len(records),
        "detail_errors": detail_errors[:3],
    }


def _search_curseforge(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_key = os.environ.get("CURSEFORGE_API_KEY", "").strip()
    if not api_key:
        return [], {"provider": "curseforge", "status": "not_configured", "result_count": 0}
    params = urllib.parse.urlencode(
        {
            "gameId": 432,
            "searchFilter": _query_terms(query),
            "pageSize": 3,
            "sortField": 2,
            "sortOrder": "desc",
        }
    )
    headers = {"x-api-key": api_key}
    payload = _request_json(
        f"https://api.curseforge.com/v1/mods/search?{params}", headers=headers
    )
    rows = payload.get("data", []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    detail_errors: list[str] = []
    for row in rows[:3] if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        mod_id = row.get("id")
        if not isinstance(mod_id, int):
            continue
        body = str(row.get("summary") or "").strip()
        try:
            description = _request_json(
                f"https://api.curseforge.com/v1/mods/{mod_id}/description",
                headers=headers,
            )
            if isinstance(description, Mapping):
                candidate = str(description.get("data") or "").strip()
                if candidate:
                    body = re.sub(r"<[^>]+>", " ", candidate)
                    body = " ".join(body.split())
        except Exception as exc:
            detail_errors.append(f"{mod_id}:{type(exc).__name__}:{exc}")
        if not body:
            continue
        links = row.get("links") if isinstance(row.get("links"), Mapping) else {}
        url = str(links.get("websiteUrl") or "")
        records.append(
            {
                "source_id": f"curseforge:{mod_id}",
                "source_type": "curseforge_mod",
                "source_locator": f"curseforge:{mod_id}",
                "url": url,
                "title": str(row.get("name") or f"CurseForge mod {mod_id}"),
                "content": body,
                "content_sha256": _sha256_text(body),
                "evidence_origin": "curseforge_discovery",
                "metadata": {"mod_id": mod_id, "slug": str(row.get("slug") or "")},
            }
        )
    return records, {
        "provider": "curseforge",
        "status": "available",
        "result_count": len(records),
        "detail_errors": detail_errors[:3],
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
    search_query = _query_terms(query, limit=8) + " minecraft fabric mod"
    params = urllib.parse.urlencode({"q": search_query, "per_page": 3})
    try:
        payload = _request_json(
            f"https://api.github.com/search/repositories?{params}", headers=headers
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 429} and disable is not None:
            disable()
        raise
    items = payload.get("items", []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    readme_errors: list[str] = []
    for item in items[:3] if isinstance(items, list) else []:
        if not isinstance(item, Mapping):
            continue
        full_name = str(item.get("full_name") or "").strip()
        if not full_name:
            continue
        body = str(item.get("description") or "").strip()
        try:
            readme = _request_text(
                f"https://api.github.com/repos/{full_name}/readme", headers=headers
            ).strip()
            if readme:
                body = readme
        except Exception as exc:
            readme_errors.append(f"{full_name}:{type(exc).__name__}:{exc}")
        if not body:
            continue
        records.append(
            {
                "source_id": f"github:{full_name}",
                "source_type": "github_repository",
                "source_locator": f"github:{full_name}",
                "url": str(item.get("html_url") or f"https://github.com/{full_name}"),
                "title": str(item.get("name") or full_name),
                "content": body,
                "content_sha256": _sha256_text(body),
                "evidence_origin": "github_fallback_discovery",
                "metadata": {
                    "repository": full_name,
                    "default_branch": str(item.get("default_branch") or ""),
                },
            }
        )
    return records, {
        "provider": "github",
        "status": "available",
        "query": search_query,
        "result_count": len(records),
        "readme_errors": readme_errors[:3],
        "search_requests": 1,
        "source_requests": len(items[:3]) if isinstance(items, list) else 0,
    }


def _research_versions(router: Any) -> tuple[str, ...]:
    requested = str(getattr(router, "_mmm_requested_minecraft_version", "") or "").strip()
    existing = str(getattr(router, "_mmm_existing_minecraft_version", "") or "").strip()
    if requested:
        return (requested,)
    if existing:
        return (existing,)
    return ()


def _search_authoritative_catalog(query: str, versions: tuple[str, ...]) -> dict[str, Any]:
    retriever = AuthoritativeEvidenceRetriever()
    sources: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    if not versions:
        try:
            catalog = target_neutral_evidence_catalog()
            for source in retriever.search(query, limit=min(6, len(catalog))):
                item = asdict(source)
                item["matched_version"] = ""
                sources.setdefault(source.source_id, item)
        except Exception as exc:
            errors.append({"minecraft_version": "", "error": f"{type(exc).__name__}: {exc}"})
    for version in versions:
        try:
            catalog = evidence_catalog_for_version(version)
            for source in retriever.search(query, minecraft_version=version, limit=min(6, len(catalog))):
                item = asdict(source)
                item["matched_version"] = version
                sources.setdefault(source.source_id, item)
        except Exception as exc:
            errors.append({"minecraft_version": version, "error": f"{type(exc).__name__}: {exc}"})
    return {"schema_version": "mmm/project-rag-query-v2", "sources": list(sources.values()), "errors": errors}


def _existing_code_index() -> Path | None:
    candidates = [Path("rag/project-index.json")]
    configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
    workspace = os.environ.get("MMM_WORKSPACE", "").strip()
    if configured:
        candidates.insert(0, Path(configured).expanduser())
    if workspace:
        candidates.append(Path(workspace).expanduser() / "rag/project-index.json")
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except OSError:
            continue
        if path.is_file():
            return path
    return None


def _search_code_index(index_path: Path | None, query: str) -> dict[str, Any]:
    if index_path is None:
        return {"schema_version": "mmm/code-rag-query-v2", "status": "not_indexed", "hits": []}
    try:
        result = ProjectRAGIndex(index_path).search_with_receipt(
            query, limit=8, semantic=True, rerank=True
        )
        return {
            "schema_version": "mmm/code-rag-query-v2",
            "status": "searched",
            "hits": [asdict(hit) for hit in result.hits],
            "receipt": asdict(result.receipt),
        }
    except Exception as exc:
        return {
            "schema_version": "mmm/code-rag-query-v2",
            "status": "error",
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _forced_rag_bundle(router: Any, research_brief: Mapping[str, Any]) -> dict[str, Any]:
    raw_domains = research_brief.get("domains")
    domains = [item for item in raw_domains or [] if isinstance(item, Mapping)]
    versions = _research_versions(router)
    code_index = _existing_code_index()
    github_blocked = False
    github_lock = threading.Lock()

    def github_disabled() -> bool:
        with github_lock:
            return github_blocked

    def disable_github() -> None:
        nonlocal github_blocked
        with github_lock:
            github_blocked = True

    query_count = 0
    external_source_count = 0
    domain_payloads: list[dict[str, Any]] = []
    for domain in domains:
        domain_id = str(domain.get("domain_id") or "").strip()
        raw_queries = domain.get("queries")
        query_payloads: list[dict[str, Any]] = []
        for raw_query in raw_queries if isinstance(raw_queries, list) else []:
            query = str(raw_query or "").strip()
            if not query:
                continue
            query_count += 1
            provider_receipts: dict[str, Any] = {}
            provider_errors: list[dict[str, Any]] = []
            records: list[dict[str, Any]] = []
            providers = (
                ("modrinth", lambda: _search_modrinth(query)),
                ("curseforge", lambda: _search_curseforge(query)),
                (
                    "github",
                    lambda: _search_github(
                        query, disabled=github_disabled, disable=disable_github
                    ),
                ),
            )
            for provider, call in providers:
                try:
                    found, receipt = call()
                    provider_receipts[provider] = receipt
                    records.extend(found)
                except Exception as exc:
                    error = _provider_error(provider, exc)
                    provider_receipts[provider] = error
                    provider_errors.append(error)
            deduped: list[dict[str, Any]] = []
            seen: set[str] = set()
            for record in records:
                key = str(record.get("source_id") or record.get("url") or "")
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(record)
            external_source_count += len(deduped)
            github_receipt = provider_receipts.get("github", {})
            github_status = str(
                github_receipt.get("status") if isinstance(github_receipt, Mapping) else "error"
            )
            query_payloads.append(
                {
                    "query": query,
                    "query_sha256": _sha256_text(query),
                    "project_rag": _search_authoritative_catalog(query, versions),
                    "code_rag": _search_code_index(code_index, query),
                    "external_rag": {
                        "schema_version": "mmm/external-pre-design-discovery-v1",
                        "sources": deduped,
                        "errors": provider_errors,
                        "providers": provider_receipts,
                        "github_retrieval": {
                            "provider_status": github_status,
                            "saturation_reason": (
                                "rate_or_auth_failure"
                                if github_status in {"error", "disabled_after_rate_or_auth_failure"}
                                else ""
                            ),
                            "search_requests": int(
                                github_receipt.get("search_requests") or 0
                                if isinstance(github_receipt, Mapping)
                                else 0
                            ),
                            "source_requests": int(
                                github_receipt.get("source_requests") or 0
                                if isinstance(github_receipt, Mapping)
                                else 0
                            ),
                        },
                    },
                }
            )
        domain_payloads.append({"domain_id": domain_id, "queries": query_payloads})
    payload: dict[str, Any] = {
        "schema_version": "mmm/pre-design-grounded-rag-v3",
        "versions": list(versions),
        "domain_count": len(domains),
        "query_count": query_count,
        "external_source_count": external_source_count,
        "code_index_status": "available" if code_index is not None else "not_indexed",
        "domains": domain_payloads,
    }
    payload["research_sha256"] = _sha256(payload)
    return payload


def _record_body(record: Mapping[str, Any]) -> str:
    return str(record.get("content") or record.get("body") or record.get("text") or "").strip()


def _model_units(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    grounded = evidence.get("grounded_rag")
    queries = grounded.get("queries") if isinstance(grounded, Mapping) else None
    for query in queries if isinstance(queries, list) else []:
        if not isinstance(query, Mapping):
            continue
        records = query.get("evidence_records")
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, Mapping):
                continue
            body = _record_body(record)
            if not body:
                continue
            units.append(
                {
                    "query": str(query.get("query") or ""),
                    "source_id": str(record.get("source_id") or ""),
                    "source_type": str(record.get("source_type") or ""),
                    "url": str(record.get("url") or ""),
                    "title": str(record.get("title") or ""),
                    "content_sha256": str(record.get("content_sha256") or _sha256_text(body)),
                    "content": body,
                }
            )
    return units


def _evidence_root() -> Path:
    configured = os.environ.get("MMM_RESEARCH_DOCUMENT_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        workspace = os.environ.get("MMM_WORKSPACE", "").strip()
        root = (Path(workspace).expanduser() if workspace else Path.cwd()) / "mmm-output" / "research-evidence"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def _split_utf8(value: str, max_bytes: int) -> list[str]:
    if not value:
        return []
    result: list[str] = []
    start = 0
    while start < len(value):
        end = min(len(value), start + max_bytes)
        while end > start and len(value[start:end].encode("utf-8")) > max_bytes:
            end -= 1
        if end <= start:
            end = start + 1
        result.append(value[start:end])
        start = end
    return result


def _materialize_domain_evidence_document(
    domain_id: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    raw_payload = dict(evidence)
    raw_text = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str)
    document_sha256 = _sha256_text(raw_text)
    safe_domain = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain_id).strip("_") or "unknown"
    root = _evidence_root()
    raw_path = root / f"{safe_domain}-{document_sha256[7:19]}.json"
    pages_path = root / f"{safe_domain}-{document_sha256[7:19]}.pages.jsonl"
    units = _model_units(evidence)
    pages: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(units):
        rendered = json.dumps(unit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        parts = _split_utf8(rendered, _EVIDENCE_PAGE_BYTES)
        for part_index, content in enumerate(parts):
            pages.append(
                {
                    "schema_version": "mmm/research-evidence-page-v2",
                    "domain_id": domain_id,
                    "unit_id": f"source:{unit_index}",
                    "part_index": part_index,
                    "part_count": len(parts),
                    "content": content,
                }
            )
    page_count = len(pages)
    for index, page in enumerate(pages):
        page["page_index"] = index
        page["page_count"] = page_count
        page["page_ref"] = f"{document_sha256}#page={index + 1}/{page_count}"
    _atomic_write_text(raw_path, raw_text)
    pages_text = "\n".join(
        json.dumps(page, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for page in pages
    )
    if pages_text:
        pages_text += "\n"
    _atomic_write_text(pages_path, pages_text)
    return {
        "schema_version": "mmm/research-evidence-document-v2",
        "domain_id": domain_id,
        "document_sha256": document_sha256,
        "raw_path": str(raw_path),
        "pages_path": str(pages_path),
        "page_count": page_count,
        "page_chars": _EVIDENCE_PAGE_BYTES,
        "page_bytes": _EVIDENCE_PAGE_BYTES,
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
        raise FileNotFoundError(f"Research evidence pages are missing: {path}")
    pages: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid research evidence page in {path}")
        if len(str(value.get("content") or "").encode("utf-8")) > _EVIDENCE_PAGE_BYTES:
            raise ValueError("Research evidence page exceeds byte budget")
        pages.append(value)
    if len(pages) != expected:
        raise ValueError(f"Research evidence page count mismatch: expected {expected}, got {len(pages)}")
    return pages


def _prompt_document_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version", "domain_id", "document_sha256", "page_count", "page_chars",
        "page_bytes", "source_keys", "model_unit_count", "model_projection",
    )
    return {key: document[key] for key in keep if key in document}


__all__ = [
    "_forced_rag_bundle",
    "_materialize_domain_evidence_document",
    "_prompt_document_receipt",
    "_read_evidence_pages",
]
