from __future__ import annotations

"""Research-grounded retrieval for pre-design planning.

Planning discovery and source-code reuse are deliberately different retrieval jobs.
Initial planning uses a small, representative set of public project/reference searches.
Deep GitHub source inspection is reserved for a concrete evidence gap (or later reuse
work), matching an agentic reason -> search -> observe -> refine loop instead of an
exhaustive batch crawl.
"""

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from .github_adaptive_retrieval import (
    adaptive_github_evidence,
    discover_repositories,
    retrieve_repository_documents,
)
from .rag_index import ProjectRAGIndex

_HTTP_TIMEOUT_SECONDS = 12.0
_MAX_HTTP_BYTES = 2 * 1024 * 1024
_MAX_EXTERNAL_PROJECTS = 4
_MAX_HTTP_CACHE_ITEMS = 256
_PLANNING_DISCOVERY = "planning_discovery"
_PLANNING_GAP_SOURCE = "planning_gap_source"

_TOKEN = re.compile(r"[^\W_]+(?:[.$:/_-][^\W_]+)*", flags=re.UNICODE)
_HTTP_CACHE_LOCK = threading.RLock()
_HTTP_CACHE: dict[tuple[str, bool], bytes] = {}
_INSTALLED = False


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)).strip())
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _planning_query_budget() -> int:
    return _env_int("MMM_PREDESIGN_DISCOVERY_QUERY_BUDGET", 6, minimum=3, maximum=12)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token.casefold()
        for token in _TOKEN.findall(str(value))
        if len(token.strip()) >= 2
    )


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _spread(values: Sequence[str], limit: int) -> list[str]:
    """Select representative queries across the authored lifecycle, not the first N."""

    unique = _dedupe(values)
    if len(unique) <= limit:
        return unique
    if limit <= 1:
        return unique[:1]
    indices = [round(i * (len(unique) - 1) / (limit - 1)) for i in range(limit)]
    return [unique[index] for index in dict.fromkeys(indices)]


def _query_variants(query: str) -> tuple[str, ...]:
    original = str(query).strip()
    normalized = re.sub(r"[._:/\\-]+", " ", original)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return tuple(_dedupe((original, normalized))[:2])


def _domain_queries(raw_domain: Mapping[str, Any]) -> list[str]:
    raw_queries = raw_domain.get("queries", ())
    if not isinstance(raw_queries, Sequence) or isinstance(
        raw_queries, (str, bytes, bytearray)
    ):
        return []
    return _dedupe(
        [
            str(raw.get("query") or "").strip()
            if isinstance(raw, Mapping)
            else str(raw).strip()
            for raw in raw_queries
        ]
    )


def _external_brief_queries(value: Any) -> tuple[str, ...]:
    """Return only the public searches that this phase should actually execute."""

    if not isinstance(value, Mapping):
        return ()
    raw_domains = value.get("domains", ())
    if not isinstance(raw_domains, Sequence) or isinstance(
        raw_domains, (str, bytes, bytearray)
    ):
        return ()

    found: list[str] = []
    corrective = (
        str(value.get("schema_version") or "")
        == "mmm/corrective-retrieval-request-v1"
    )
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, Mapping):
            continue
        routes = {
            str(item).strip().casefold()
            for key in ("providers", "required_providers")
            for item in (
                raw_domain.get(key, ())
                if isinstance(raw_domain.get(key, ()), Sequence)
                and not isinstance(
                    raw_domain.get(key, ()), (str, bytes, bytearray)
                )
                else ()
            )
            if str(item).strip()
        }
        if not routes.intersection({"github", "modrinth"}):
            continue
        queries = _domain_queries(raw_domain)
        domain_id = str(raw_domain.get("domain_id") or "")
        mode = str(raw_domain.get("retrieval_mode") or "")
        if (
            domain_id == "request"
            and not corrective
            and mode in {"", _PLANNING_DISCOVERY}
        ):
            queries = _spread(queries, _planning_query_budget())
        found.extend(queries)
    return tuple(_dedupe(found))


def _workspace_root(router: Any | None = None) -> Path | None:
    attached = str(getattr(router, "_mmm_workspace_root", "") or "").strip()
    if attached:
        return Path(attached).expanduser().resolve()
    configured = os.environ.get("MMM_WORKSPACE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return None


def _index_path(workspace: Path, router: Any | None = None) -> Path:
    attached = str(getattr(router, "_mmm_project_rag_index", "") or "").strip()
    if attached:
        return Path(attached).expanduser().resolve()
    configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
    if configured and not str(
        getattr(router, "_mmm_workspace_root", "") or ""
    ).strip():
        return Path(configured).expanduser().resolve()
    return (workspace / ".minecraft_ai" / "project-index.db").resolve()


def _index_metadata() -> dict[str, Any]:
    return {
        "minecraft_version": os.environ.get(
            "MMM_MINECRAFT_VERSION", "target-neutral"
        ).strip()
        or "target-neutral",
        "loader": os.environ.get("MMM_LOADER", "fabric").strip() or "fabric",
        "mapping_namespace": os.environ.get(
            "MMM_MAPPING_NAMESPACE", "yarn"
        ).strip()
        or "yarn",
        "java_version": os.environ.get("MMM_JAVA_VERSION", "21").strip() or "21",
        "license": "project-local",
        "source_commit": os.environ.get("GITHUB_SHA", "").strip() or "WORKTREE",
    }


def _ensure_local_index(agentic_module: Any, router: Any) -> dict[str, Any]:
    workspace = _workspace_root(router)
    if workspace is None:
        return {
            "status": "workspace_unconfigured",
            "built": False,
            "index_path": "",
            "reason": "local project root was not explicitly attached or configured",
        }
    if not workspace.exists() or not workspace.is_dir():
        return {
            "status": "workspace_missing",
            "built": False,
            "index_path": "",
            "workspace": str(workspace),
        }

    existing = agentic_module._existing_code_index()
    if existing is not None:
        existing_path = Path(existing).expanduser().resolve()
        legacy_configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
        attached_scope = bool(
            str(getattr(router, "_mmm_workspace_root", "") or "").strip()
        )
        try:
            existing_path.relative_to(workspace)
        except ValueError:
            if attached_scope or not legacy_configured:
                existing_path = None
        if existing_path is not None:
            return {
                "status": "available",
                "built": False,
                "index_path": str(existing_path),
                "workspace": str(workspace),
            }

    index_path = _index_path(workspace, router)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = ProjectRAGIndex(index_path)
    max_files_raw = os.environ.get("MMM_RAG_MAX_FILES", "").strip()
    max_files = int(max_files_raw) if max_files_raw else None
    try:
        receipt = index.build(
            [workspace],
            metadata=_index_metadata(),
            router=None,
            semantic=False,
            max_files=max_files,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "build_failed",
            "built": False,
            "index_path": str(index_path),
            "workspace": str(workspace),
            "error": f"{type(exc).__name__}: {exc}",
        }

    os.environ["MMM_PROJECT_RAG_INDEX"] = str(index_path)
    if router is not None:
        router._mmm_project_rag_index = str(index_path)
    return {
        "status": "available",
        "built": True,
        "index_path": str(index_path),
        "workspace": str(workspace),
        "receipt": receipt,
    }


def _request_headers(*, github: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "MMM-Minecraft-Mod-AI/grounded-rag",
    }
    if github:
        token = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get(
            "GH_TOKEN", ""
        ).strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    return headers


def _http_bytes(url: str, *, github: bool = False) -> bytes:
    cache_key = (url, github)
    with _HTTP_CACHE_LOCK:
        cached = _HTTP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    request = Request(url, headers=_request_headers(github=github))
    with urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > _MAX_HTTP_BYTES:
            raise ValueError(f"remote document too large: {length} bytes")
        payload = response.read(_MAX_HTTP_BYTES + 1)
    if len(payload) > _MAX_HTTP_BYTES:
        raise ValueError("remote document exceeded the retrieval byte limit")
    with _HTTP_CACHE_LOCK:
        if len(_HTTP_CACHE) >= _MAX_HTTP_CACHE_ITEMS:
            _HTTP_CACHE.pop(next(iter(_HTTP_CACHE)))
        _HTTP_CACHE[cache_key] = payload
    return payload


def _http_json(url: str, *, github: bool = False) -> Any:
    return json.loads(_http_bytes(url, github=github).decode("utf-8"))


def _http_text(url: str, *, github: bool = False) -> str:
    return _http_bytes(url, github=github).decode("utf-8", errors="replace")


def _github_repo_from_url(url: str) -> tuple[str, str] | None:
    parsed = urlparse(str(url).strip())
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    if not owner or not repo:
        return None
    return owner, repo


def _content_sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_document(
    *,
    source_id: str,
    title: str,
    url: str,
    content: str,
    source_type: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize a complete body already bounded by the HTTP/retrieval layer."""

    text = str(content)
    source_metadata = dict(metadata or {})
    source_metadata.update(
        {
            "body_retrieved": True,
            "content_truncated": False,
            "original_content_chars": len(text),
        }
    )
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "source_type": source_type,
        "content": text,
        "content_sha256": _content_sha256(text),
        "metadata": source_metadata,
    }


def _modrinth_search(
    query: str,
    versions: Sequence[str],
    *,
    include_details: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Search project candidates; fetch detail bodies only when source evidence is needed."""

    errors: list[str] = []
    facets: list[list[str]] = [["project_type:mod"]]
    concrete_versions = [
        str(version).strip()
        for version in versions
        if str(version).strip()
        and str(version).strip() not in {"*", "target-neutral", "unknown"}
    ]
    if concrete_versions:
        facets.append([f"versions:{version}" for version in concrete_versions[:3]])
    params = urlencode(
        {
            "query": query,
            "limit": _MAX_EXTERNAL_PROJECTS,
            "index": "relevance",
            "facets": json.dumps(facets, separators=(",", ":")),
        }
    )
    try:
        payload = _http_json(f"https://api.modrinth.com/v2/search?{params}")
    except Exception as exc:  # noqa: BLE001
        return [], [f"modrinth_search:{type(exc).__name__}: {exc}"]

    projects: list[dict[str, Any]] = []
    for hit in payload.get("hits", []) if isinstance(payload, Mapping) else []:
        if not isinstance(hit, Mapping):
            continue
        project_id = str(hit.get("project_id", "")).strip()
        if not project_id:
            continue
        detail: Mapping[str, Any] = {}
        if include_details:
            try:
                value = _http_json(
                    f"https://api.modrinth.com/v2/project/{quote(project_id)}"
                )
                if isinstance(value, Mapping):
                    detail = value
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"modrinth_project:{project_id}:{type(exc).__name__}: {exc}"
                )
        projects.append(
            {
                "project_id": project_id,
                "slug": hit.get("slug"),
                "title": hit.get("title"),
                "description": hit.get("description"),
                "author": hit.get("author"),
                "versions": list(hit.get("versions", [])),
                "downloads": hit.get("downloads"),
                "license": hit.get("license"),
                "project_url": (
                    f"https://modrinth.com/mod/{hit.get('slug') or project_id}"
                ),
                "source_url": detail.get("source_url"),
                "issues_url": detail.get("issues_url"),
                "body": detail.get("body"),
                "detail_retrieved": include_details and bool(detail),
            }
        )
    return projects, errors


def _github_adaptive_search(
    query: str,
    *,
    seed_repositories: Sequence[Any] = (),
    search_if_needed: bool = True,
) -> dict[str, Any]:
    evidence = adaptive_github_evidence(
        query,
        http_json=lambda url: _http_json(url, github=True),
        http_text=_http_text,
        source_document=_source_document,
        seed_repositories=seed_repositories,
        search_if_needed=search_if_needed,
    )
    documents = [dict(item) for item in evidence.documents]
    actual = sum(
        1
        for item in documents
        if str(item.get("source_type") or "").startswith("github_")
    )
    error_text = " ".join(str(item) for item in evidence.errors).casefold()
    if (
        "rate limit" in error_text
        or "http error 403" in error_text
        or "429" in error_text
    ):
        provider_status = "rate_limited"
    elif "422" in error_text or "search limit" in error_text:
        provider_status = "provider_limited"
    elif actual:
        provider_status = "available"
    elif evidence.errors:
        provider_status = "error"
    else:
        provider_status = "ok_zero"
    return {
        "status": "available" if actual else "unavailable",
        "provider_status": provider_status,
        "query": query,
        "repositories": list(evidence.repositories),
        "documents": documents,
        "errors": list(evidence.errors),
        "search_queries": list(evidence.search_queries),
        "search_requests": evidence.search_requests,
        "source_requests": evidence.source_requests,
        "source_bytes": evidence.source_bytes,
        "coverage_score": evidence.coverage_score,
        "saturation_reason": str(evidence.saturation_reason or ""),
        "actual_source_document_count": actual,
    }


def _github_targeted_search(
    query: str, *, seed_repositories: Sequence[Any] = ()
) -> dict[str, Any]:
    """Inspect only enough source to resolve one concrete planning gap."""

    repositories: list[tuple[str, str]] = []
    for value in seed_repositories:
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and len(value) >= 2
        ):
            ref = (
                str(value[0]).strip(),
                str(value[1]).strip().removesuffix(".git"),
            )
            if all(ref) and ref not in repositories:
                repositories.append(ref)
    errors: list[str] = []
    search_requests = 0
    search_queries: list[str] = []
    saturation_reason = "seed_repositories"
    if not repositories:
        discovery = discover_repositories(
            query, http_json=lambda url: _http_json(url, github=True)
        )
        repositories.extend(discovery.repositories)
        errors.extend(discovery.errors)
        search_requests = discovery.search_requests
        search_queries.extend(discovery.search_queries)
        saturation_reason = discovery.saturation_reason

    documents: list[dict[str, Any]] = []
    source_requests = 0
    source_bytes = 0
    covered = 0.0
    request_remaining = 8
    for owner, repo in repositories[:2]:
        if request_remaining < 2:
            break
        evidence = retrieve_repository_documents(
            owner,
            repo,
            query,
            http_json=lambda url: _http_json(url, github=True),
            http_text=_http_text,
            source_document=_source_document,
            request_budget=min(4, request_remaining),
            byte_budget=256 * 1024,
            coverage_target=0.5,
        )
        request_remaining -= evidence.requests_used
        source_requests += evidence.requests_used
        source_bytes += evidence.source_bytes
        covered = max(covered, float(evidence.coverage_score))
        errors.extend(evidence.errors)
        documents.extend(dict(item) for item in evidence.documents)
        saturation_reason = evidence.saturation_reason
        if documents and covered >= 0.5:
            break

    error_text = " ".join(errors).casefold()
    if "rate limit" in error_text or "403" in error_text or "429" in error_text:
        provider_status = "rate_limited"
    elif documents:
        provider_status = "available"
    elif errors:
        provider_status = "error"
    else:
        provider_status = "ok_zero"
    return {
        "status": "available" if documents else "unavailable",
        "provider_status": provider_status,
        "query": query,
        "repositories": repositories[:2],
        "documents": documents,
        "errors": errors,
        "search_queries": search_queries,
        "search_requests": search_requests,
        "source_requests": source_requests,
        "source_bytes": source_bytes,
        "coverage_score": covered,
        "saturation_reason": saturation_reason,
        "actual_source_document_count": len(documents),
    }


def _coverage_score(
    query: str, documents: Sequence[Mapping[str, Any]]
) -> float:
    terms = set(_tokens(query))
    if not documents:
        return 0.0
    if not terms:
        return 1.0
    haystack: set[str] = set()
    for document in documents:
        haystack.update(_tokens(str(document.get("title", ""))))
        haystack.update(_tokens(str(document.get("content", ""))[:8_000]))
    return round(len(terms & haystack) / max(1, len(terms)), 4)


def _external_retrieval(
    query: str, versions: Sequence[str], *, mode: str = "source"
) -> dict[str, Any]:
    variants = _query_variants(query)
    projects_by_id: dict[str, dict[str, Any]] = {}
    documents_by_id: dict[str, dict[str, Any]] = {}
    seed_repositories: list[tuple[str, str]] = []
    errors: list[str] = []
    include_details = mode != _PLANNING_DISCOVERY

    for variant in variants:
        projects, variant_errors = _modrinth_search(
            variant,
            versions,
            include_details=include_details,
        )
        errors.extend(variant_errors)
        for project in projects:
            project_id = str(project.get("project_id", ""))
            if project_id:
                projects_by_id.setdefault(project_id, project)
            body = str(project.get("body") or "")
            if body and project_id:
                document = _source_document(
                    source_id=f"modrinth:{project_id}",
                    title=str(project.get("title") or project_id),
                    url=str(project.get("project_url") or ""),
                    content=body,
                    source_type="modrinth_project_body",
                    metadata={
                        "project_id": project_id,
                        "author": project.get("author"),
                        "versions": project.get("versions"),
                        "downloads": project.get("downloads"),
                        "license": project.get("license"),
                        "source_url": project.get("source_url"),
                    },
                )
                documents_by_id.setdefault(str(document["source_id"]), document)
            repo_ref = _github_repo_from_url(str(project.get("source_url") or ""))
            if repo_ref is not None and repo_ref not in seed_repositories:
                seed_repositories.append(repo_ref)

    if mode == _PLANNING_DISCOVERY:
        github = {
            "provider_status": "deferred_until_specific_gap_or_reuse",
            "repositories": [],
            "documents": [],
            "errors": [],
            "search_queries": [],
            "search_requests": 0,
            "source_requests": 0,
            "source_bytes": 0,
            "coverage_score": 0.0,
            "saturation_reason": "planning_discovery_uses_project_metadata",
        }
    elif mode == _PLANNING_GAP_SOURCE:
        github = _github_targeted_search(
            query, seed_repositories=tuple(seed_repositories)
        )
    else:
        github = _github_adaptive_search(
            query,
            seed_repositories=tuple(seed_repositories),
            search_if_needed=True,
        )

    errors.extend(str(item) for item in github.get("errors", ()))
    for document in github.get("documents", ()):
        if isinstance(document, Mapping):
            source_id = str(document.get("source_id") or "")
            if source_id:
                documents_by_id.setdefault(source_id, dict(document))
    documents = list(documents_by_id.values())
    github_source_count = sum(
        1
        for document in documents
        if str(document.get("source_type") or "").startswith("github_")
    )
    status = (
        "available"
        if documents
        else "metadata_only"
        if projects_by_id
        else "unavailable"
    )
    return {
        "schema_version": "mmm/external-grounded-rag-v2",
        "status": status,
        "retrieval_mode": mode,
        "query": query,
        "query_variants": list(variants),
        "github_search_queries": list(github.get("search_queries", ())),
        "providers": ["modrinth_public", "github_public_source"],
        "credentials_required": False,
        "corrective_search_used": mode == _PLANNING_GAP_SOURCE,
        "project_count": len(projects_by_id),
        "metadata_only_project_count": (
            len(projects_by_id) if not include_details else 0
        ),
        "source_repository_count": len(github.get("repositories", ())),
        "document_count": len(documents),
        "actual_source_document_count": len(documents),
        "github_source_document_count": github_source_count,
        "coverage_score": max(
            _coverage_score(query, documents),
            float(github.get("coverage_score") or 0.0),
        ),
        "projects": list(projects_by_id.values()),
        "documents": documents,
        "errors": errors,
        "github_retrieval": {
            "provider_status": str(github.get("provider_status") or "unknown"),
            "search_requests": int(github.get("search_requests") or 0),
            "source_requests": int(github.get("source_requests") or 0),
            "source_bytes": int(github.get("source_bytes") or 0),
            "saturation_reason": str(github.get("saturation_reason") or ""),
        },
    }


def _augment_bundle(
    agentic_module: Any,
    payload: Mapping[str, Any],
    *,
    versions: Sequence[str],
    local_index: Mapping[str, Any],
    external_queries: Sequence[str] = (),
    default_mode: str = "source",
) -> dict[str, Any]:
    result = dict(payload)
    raw_domains = result.get("domains", [])
    domains = [dict(item) for item in raw_domains if isinstance(item, Mapping)]
    allowed_external = {
        str(item).strip().casefold()
        for item in external_queries
        if str(item).strip()
    }
    jobs: list[tuple[int, int, str, str]] = []
    for domain_index, domain in enumerate(domains):
        queries = domain.get("queries", [])
        if not isinstance(queries, list):
            continue
        copied_queries = [
            dict(item) for item in queries if isinstance(item, Mapping)
        ]
        domain["queries"] = copied_queries
        mode = str(
            domain.get("retrieval_mode") or default_mode or "source"
        ).strip()
        if str(domain.get("domain_id") or "") == "request" and mode == "source":
            mode = _PLANNING_DISCOVERY
        for query_index, item in enumerate(copied_queries):
            query = str(item.get("query", "")).strip()
            if query and query.casefold() in allowed_external:
                jobs.append((domain_index, query_index, query, mode))

    def run(
        job: tuple[int, int, str, str]
    ) -> tuple[int, int, dict[str, Any]]:
        domain_index, query_index, query, mode = job
        return (
            domain_index,
            query_index,
            _external_retrieval(query, versions, mode=mode),
        )

    if jobs:
        with ThreadPoolExecutor(
            max_workers=max(1, min(4, len(jobs))),
            thread_name_prefix="mmm_grounded_rag",
        ) as pool:
            for domain_index, query_index, external in pool.map(run, jobs):
                domains[domain_index]["queries"][query_index][
                    "external_rag"
                ] = external

    result["domains"] = domains
    result["schema_version"] = "mmm/forced-pre-design-rag"
    result["local_index"] = dict(local_index)
    if local_index.get("status") == "available":
        result["code_index_status"] = "available"
        result["code_index_path"] = str(local_index.get("index_path", ""))
    external_payloads = [
        query.get("external_rag")
        for domain in domains
        for query in domain.get("queries", [])
        if isinstance(query, Mapping)
        and isinstance(query.get("external_rag"), Mapping)
    ]
    result["external_source_count"] = sum(
        int(item.get("actual_source_document_count", 0))
        for item in external_payloads
    )
    result["external_document_count"] = sum(
        int(item.get("document_count", 0)) for item in external_payloads
    )
    result["external_query_count"] = len(external_payloads)
    result["research_sha256"] = agentic_module._sha256(
        {key: value for key, value in result.items() if key != "research_sha256"}
    )
    return result


def install(agentic_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    original = agentic_module._forced_rag_bundle
    if getattr(original, "__mmm_grounded_rag__", False):
        _INSTALLED = True
        return

    def grounded_rag_bundle(
        router: Any, research_brief: Mapping[str, Any]
    ) -> dict[str, Any]:
        local_index = _ensure_local_index(agentic_module, router)
        payload = original(router, research_brief)
        versions = tuple(
            str(item) for item in payload.get("versions", []) if str(item).strip()
        )
        brief_schema = str(research_brief.get("schema_version") or "")
        default_mode = (
            _PLANNING_GAP_SOURCE
            if brief_schema == "mmm/corrective-retrieval-request-v1"
            else "source"
        )
        return _augment_bundle(
            agentic_module,
            payload,
            versions=versions,
            local_index=local_index,
            external_queries=_external_brief_queries(research_brief),
            default_mode=default_mode,
        )

    grounded_rag_bundle.__name__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__qualname__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__mmm_grounded_rag__ = True
    grounded_rag_bundle.__wrapped__ = original
    agentic_module._forced_rag_bundle = grounded_rag_bundle
    _INSTALLED = True


__all__ = ["install"]
