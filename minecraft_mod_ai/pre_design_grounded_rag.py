from __future__ import annotations

"""Host-owned, requirement-complete pre-design source discovery."""

import hashlib
import json
import os
import re
import tempfile
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

_PAGE_BYTES = 1800
_TIMEOUT = 8.0
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


def _query_terms(value: str, limit: int = 12) -> str:
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
        if len(words) >= limit:
            break
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
    params = urllib.parse.urlencode(
        {
            "query": _query_terms(query),
            "limit": 3,
            "index": "relevance",
            "facets": json.dumps([["project_type:mod"]]),
        }
    )
    payload = _json(f"https://api.modrinth.com/v2/search?{params}")
    hits = payload.get("hits", []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for hit in hits[:3] if isinstance(hits, list) else []:
        if not isinstance(hit, Mapping):
            continue
        project_id = str(hit.get("project_id") or "").strip()
        slug = str(hit.get("slug") or project_id).strip()
        detail: Mapping[str, Any] = hit
        if project_id:
            try:
                fetched = _json(
                    "https://api.modrinth.com/v2/project/"
                    + urllib.parse.quote(project_id, safe="")
                )
                if isinstance(fetched, Mapping):
                    detail = fetched
            except Exception as exc:
                errors.append(f"{project_id}:{type(exc).__name__}:{exc}")
        body = str(detail.get("body") or hit.get("description") or "").strip()
        if not body:
            continue
        records.append(
            {
                "source_id": f"modrinth:{project_id or slug}",
                "source_type": "modrinth_project_body",
                "source_locator": f"modrinth:{project_id or slug}",
                "url": f"https://modrinth.com/mod/{slug}",
                "title": str(detail.get("title") or hit.get("title") or slug),
                "content": body,
                "content_sha256": _sha256_text(body),
                "body_retrieved": True,
                "evidence_origin": "modrinth_project_body",
                "metadata": {
                    "project_id": project_id,
                    "slug": slug,
                    "versions": list(hit.get("versions") or [])[:20],
                },
            }
        )
    return records, {
        "provider": "modrinth",
        "status": "available",
        "result_count": len(records),
        "detail_errors": errors[:3],
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
    params = urllib.parse.urlencode(
        {
            "gameId": 432,
            "searchFilter": _query_terms(query),
            "pageSize": 3,
            "sortField": 2,
            "sortOrder": "desc",
        }
    )
    payload = _json(f"https://api.curseforge.com/v1/mods/search?{params}", headers)
    rows = payload.get("data", []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows[:3] if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), int):
            continue
        mod_id = int(row["id"])
        body = str(row.get("summary") or "").strip()
        try:
            desc = _json(
                f"https://api.curseforge.com/v1/mods/{mod_id}/description", headers
            )
            if isinstance(desc, Mapping) and str(desc.get("data") or "").strip():
                body = " ".join(re.sub(r"<[^>]+>", " ", str(desc["data"])).split())
        except Exception as exc:
            errors.append(f"{mod_id}:{type(exc).__name__}:{exc}")
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
                "metadata": {"mod_id": mod_id, "slug": str(row.get("slug") or "")},
            }
        )
    return records, {
        "provider": "curseforge",
        "status": "available",
        "result_count": len(records),
        "detail_errors": errors[:3],
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
    params = urllib.parse.urlencode(
        {"q": _query_terms(query, 10) + " minecraft fabric mod", "per_page": 3}
    )
    try:
        payload = _json(f"https://api.github.com/search/repositories?{params}", headers)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 429} and disable is not None:
            disable()
        raise
    rows = payload.get("items", []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    readme_errors: list[str] = []
    for row in rows[:3] if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        full_name = str(row.get("full_name") or "").strip()
        if not full_name:
            continue
        body = str(row.get("description") or "").strip()
        try:
            fetched = _text(
                f"https://api.github.com/repos/{full_name}/readme", headers
            ).strip()
            if fetched:
                body = fetched
        except Exception as exc:
            readme_errors.append(f"{full_name}:{type(exc).__name__}:{exc}")
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
    return records, {
        "provider": "github",
        "status": "available",
        "result_count": len(records),
        "readme_errors": readme_errors[:3],
        "search_requests": 1,
        "source_requests": len(rows[:3]) if isinstance(rows, list) else 0,
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
            for source in retriever.search(query, limit=min(6, len(catalog)), **kwargs):
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

    def disabled() -> bool:
        return github_blocked

    def disable() -> None:
        nonlocal github_blocked
        github_blocked = True

    out_domains: list[dict[str, Any]] = []
    query_count = 0
    external_count = 0
    for domain in domains:
        rows: list[dict[str, Any]] = []
        for raw in (
            domain.get("queries", []) if isinstance(domain.get("queries"), list) else []
        ):
            query = str(raw or "").strip()
            if not query:
                continue
            query_count += 1
            records: list[dict[str, Any]] = []
            receipts: dict[str, Any] = {}
            errors: list[dict[str, Any]] = []
            calls = (
                ("modrinth", lambda: _search_modrinth(query)),
                ("curseforge", lambda: _search_curseforge(query)),
                (
                    "github",
                    lambda: _search_github(query, disabled=disabled, disable=disable),
                ),
            )
            for provider, call in calls:
                try:
                    found, receipt = call()
                    records.extend(found)
                    receipts[provider] = receipt
                except Exception as exc:
                    receipt = _error(provider, exc)
                    receipts[provider] = receipt
                    errors.append(receipt)
            unique: list[dict[str, Any]] = []
            seen: set[str] = set()
            for record in records:
                key = str(record.get("source_id") or record.get("url") or "")
                if key and key not in seen:
                    seen.add(key)
                    unique.append(record)
            external_count += len(unique)
            gh = (
                receipts.get("github", {})
                if isinstance(receipts.get("github"), Mapping)
                else {}
            )
            rows.append(
                {
                    "query": query,
                    "query_sha256": _sha256_text(query),
                    "project_rag": _search_authoritative_catalog(query, versions),
                    "code_rag": _search_code_index(code_index, query),
                    "external_rag": {
                        "schema_version": "mmm/external-pre-design-discovery-v2",
                        "sources": unique,
                        "errors": errors,
                        "providers": receipts,
                        "github_retrieval": {
                            "provider_status": str(gh.get("status") or "not_requested"),
                            "saturation_reason": "rate_or_auth_failure"
                            if str(gh.get("status"))
                            in {"error", "disabled_after_rate_or_auth_failure"}
                            else "",
                            "search_requests": int(gh.get("search_requests") or 0),
                            "source_requests": int(gh.get("source_requests") or 0),
                        },
                    },
                }
            )
        out_domains.append(
            {"domain_id": str(domain.get("domain_id") or ""), "queries": rows}
        )
    payload: dict[str, Any] = {
        "schema_version": "mmm/pre-design-grounded-rag-v4",
        "versions": list(versions),
        "domain_count": len(domains),
        "query_count": query_count,
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


def _split(value: str) -> list[str]:
    result: list[str] = []
    start = 0
    while start < len(value):
        end = min(len(value), start + _PAGE_BYTES)
        while end > start and len(value[start:end].encode("utf-8")) > _PAGE_BYTES:
            end -= 1
        if end <= start:
            end = start + 1
        result.append(value[start:end])
        start = end
    return result


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
        parts = _split(rendered)
        for pi, content in enumerate(parts):
            pages.append(
                {
                    "schema_version": "mmm/research-evidence-page-v2",
                    "domain_id": domain_id,
                    "unit_id": f"source:{ui}",
                    "part_index": pi,
                    "part_count": len(parts),
                    "content": content,
                }
            )
    for i, page in enumerate(pages):
        page.update(
            {
                "page_index": i,
                "page_count": len(pages),
                "page_ref": f"{digest}#page={i + 1}/{len(pages)}",
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
        "page_chars": _PAGE_BYTES,
        "page_bytes": _PAGE_BYTES,
        "source_keys": sorted(str(k) for k in evidence),
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
        "page_chars",
        "page_bytes",
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
