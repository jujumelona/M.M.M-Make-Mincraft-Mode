from __future__ import annotations

"""Research-grounded retrieval for pre-design planning.

The pre-design path must own retrieval end-to-end: build the local project index when it is
missing, search public mod metadata without credentials, follow source repositories, fetch
actual source files, and preserve provenance in the evidence ledger. Retrieval misses are
corrected by broadening providers; they are never repaired by asking the model to invent data.
"""

import base64
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

from .rag_index import ProjectRAGIndex

_HTTP_TIMEOUT_SECONDS = 12.0
_MAX_HTTP_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_TEXT_CHARS = 24_000
_MAX_SOURCE_FILES_PER_REPO = 4
_MAX_EXTERNAL_PROJECTS = 4
_MAX_SOURCE_REPOS_PER_QUERY = 2
_MAX_HTTP_CACHE_ITEMS = 256
_ALLOWED_SOURCE_SUFFIXES = (
    ".java",
    ".kt",
    ".kts",
    ".json",
    ".gradle",
    ".properties",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".mcfunction",
)
_TOKEN = re.compile(r"[^\W_]+(?:[.$:/_-][^\W_]+)*", flags=re.UNICODE)
_HTTP_CACHE_LOCK = threading.RLock()
_HTTP_CACHE: dict[tuple[str, bool], bytes] = {}
_INSTALLED = False


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


def _query_variants(query: str) -> tuple[str, ...]:
    """Deterministic multi-query expansion for recall without another model turn."""
    original = str(query).strip()
    normalized = re.sub(r"[._:/\\-]+", " ", original)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    implementation = f"{normalized} minecraft fabric mod source implementation".strip()
    return tuple(_dedupe((original, normalized, implementation))[:3])


def _workspace_root() -> Path:
    configured = os.environ.get("MMM_WORKSPACE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def _index_path(workspace: Path) -> Path:
    configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
    if configured:
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
    """Guarantee that local project retrieval has an index before forced RAG runs."""
    del router
    existing = agentic_module._existing_code_index()
    if existing is not None:
        return {
            "status": "available",
            "built": False,
            "index_path": str(existing),
        }

    workspace = _workspace_root()
    if not workspace.exists() or not workspace.is_dir():
        return {
            "status": "workspace_missing",
            "built": False,
            "index_path": "",
            "workspace": str(workspace),
        }

    index_path = _index_path(workspace)
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
    except Exception as exc:
        return {
            "status": "build_failed",
            "built": False,
            "index_path": str(index_path),
            "workspace": str(workspace),
            "error": f"{type(exc).__name__}: {exc}",
        }

    os.environ["MMM_PROJECT_RAG_INDEX"] = str(index_path)
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
    text = str(content)[:_MAX_SOURCE_TEXT_CHARS]
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "source_type": source_type,
        "content": text,
        "content_sha256": _content_sha256(text),
        "metadata": dict(metadata or {}),
    }


def _path_score(path: str, query: str) -> tuple[int, str]:
    folded = path.casefold()
    query_terms = set(_tokens(query))
    path_terms = set(_tokens(path.replace("/", " ")))
    score = 4 * len(query_terms & path_terms)
    if "/src/main/" in f"/{folded}":
        score += 4
    if folded.endswith((".java", ".kt")):
        score += 3
    if any(term in folded for term in ("entity", "item", "registry", "world", "network")):
        score += 2
    if "/test/" in f"/{folded}" or "/build/" in f"/{folded}":
        score -= 3
    return score, path


def _github_repo_documents(
    owner: str,
    repo: str,
    query: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    documents: list[dict[str, Any]] = []
    errors: list[str] = []
    repo_api = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}"
    try:
        repo_meta = _http_json(repo_api, github=True)
    except Exception as exc:
        return [], [f"github_repo:{owner}/{repo}:{type(exc).__name__}: {exc}"]

    branch = str(repo_meta.get("default_branch", "main") or "main")
    html_url = str(repo_meta.get("html_url", f"https://github.com/{owner}/{repo}"))

    try:
        readme = _http_json(f"{repo_api}/readme?ref={quote(branch)}", github=True)
        encoded = str(readme.get("content", "")).replace("\n", "")
        if encoded:
            content = base64.b64decode(encoded).decode("utf-8", errors="replace")
            documents.append(
                _source_document(
                    source_id=f"github:{owner}/{repo}:README",
                    title=f"{owner}/{repo} README",
                    url=str(readme.get("html_url", html_url)),
                    content=content,
                    source_type="github_readme",
                    metadata={"repository": f"{owner}/{repo}", "branch": branch},
                )
            )
    except Exception as exc:
        errors.append(f"github_readme:{owner}/{repo}:{type(exc).__name__}: {exc}")

    try:
        tree = _http_json(
            f"{repo_api}/git/trees/{quote(branch)}?recursive=1",
            github=True,
        )
        candidates = [
            str(item.get("path", ""))
            for item in tree.get("tree", [])
            if isinstance(item, Mapping)
            and item.get("type") == "blob"
            and str(item.get("path", "")).casefold().endswith(_ALLOWED_SOURCE_SUFFIXES)
        ]
        candidates.sort(key=lambda path: (-_path_score(path, query)[0], path))
        for path in candidates[:_MAX_SOURCE_FILES_PER_REPO]:
            raw_url = (
                f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repo)}/"
                f"{quote(branch, safe='')}/{quote(path, safe='/')}"
            )
            try:
                content = _http_text(raw_url)
            except Exception as exc:
                errors.append(
                    f"github_raw:{owner}/{repo}/{path}:{type(exc).__name__}: {exc}"
                )
                continue
            license_value = None
            if isinstance(repo_meta.get("license"), Mapping):
                license_value = repo_meta["license"].get("spdx_id")
            documents.append(
                _source_document(
                    source_id=f"github:{owner}/{repo}:{path}",
                    title=f"{owner}/{repo}:{path}",
                    url=f"{html_url}/blob/{quote(branch, safe='')}/{quote(path, safe='/')}",
                    content=content,
                    source_type="github_source",
                    metadata={
                        "repository": f"{owner}/{repo}",
                        "branch": branch,
                        "path": path,
                        "license": license_value,
                    },
                )
            )
    except Exception as exc:
        errors.append(f"github_tree:{owner}/{repo}:{type(exc).__name__}: {exc}")

    return documents, errors


def _modrinth_search(
    query: str,
    versions: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
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
    except Exception as exc:
        return [], [f"modrinth_search:{type(exc).__name__}: {exc}"]

    projects: list[dict[str, Any]] = []
    for hit in payload.get("hits", []) if isinstance(payload, Mapping) else []:
        if not isinstance(hit, Mapping):
            continue
        project_id = str(hit.get("project_id", "")).strip()
        if not project_id:
            continue
        try:
            detail = _http_json(
                f"https://api.modrinth.com/v2/project/{quote(project_id)}"
            )
        except Exception as exc:
            errors.append(f"modrinth_project:{project_id}:{type(exc).__name__}: {exc}")
            detail = {}
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
                "project_url": f"https://modrinth.com/mod/{hit.get('slug') or project_id}",
                "source_url": detail.get("source_url") if isinstance(detail, Mapping) else None,
                "issues_url": detail.get("issues_url") if isinstance(detail, Mapping) else None,
                "body": detail.get("body") if isinstance(detail, Mapping) else None,
            }
        )
    return projects, errors


def _github_repository_search(query: str) -> tuple[list[tuple[str, str]], list[str]]:
    params = urlencode(
        {
            "q": f"{query} minecraft fabric mod",
            "sort": "stars",
            "order": "desc",
            "per_page": 3,
        }
    )
    try:
        payload = _http_json(
            f"https://api.github.com/search/repositories?{params}",
            github=True,
        )
    except Exception as exc:
        return [], [f"github_repository_search:{type(exc).__name__}: {exc}"]
    repositories: list[tuple[str, str]] = []
    for item in payload.get("items", []) if isinstance(payload, Mapping) else []:
        if not isinstance(item, Mapping):
            continue
        full_name = str(item.get("full_name", "")).strip()
        if "/" not in full_name:
            continue
        owner, repo = full_name.split("/", 1)
        repositories.append((owner, repo))
    return repositories, []


def _coverage_score(query: str, documents: Sequence[Mapping[str, Any]]) -> float:
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


def _external_retrieval(query: str, versions: Sequence[str]) -> dict[str, Any]:
    variants = _query_variants(query)
    projects_by_id: dict[str, dict[str, Any]] = {}
    documents_by_id: dict[str, dict[str, Any]] = {}
    fetched_repos: set[tuple[str, str]] = set()
    errors: list[str] = []

    for variant in variants:
        projects, variant_errors = _modrinth_search(variant, versions)
        errors.extend(variant_errors)
        for project in projects:
            project_id = str(project.get("project_id", ""))
            projects_by_id.setdefault(project_id, project)
            body = str(project.get("body") or project.get("description") or "")
            if body:
                document = _source_document(
                    source_id=f"modrinth:{project_id}",
                    title=str(project.get("title") or project_id),
                    url=str(project.get("project_url") or ""),
                    content=body,
                    source_type="modrinth_project",
                    metadata={
                        "project_id": project_id,
                        "author": project.get("author"),
                        "versions": project.get("versions"),
                        "downloads": project.get("downloads"),
                        "license": project.get("license"),
                        "source_url": project.get("source_url"),
                    },
                )
                documents_by_id.setdefault(document["source_id"], document)

            repo_ref = _github_repo_from_url(str(project.get("source_url") or ""))
            if repo_ref is None or repo_ref in fetched_repos:
                continue
            if len(fetched_repos) >= _MAX_SOURCE_REPOS_PER_QUERY:
                continue
            fetched_repos.add(repo_ref)
            source_docs, source_errors = _github_repo_documents(
                repo_ref[0], repo_ref[1], query
            )
            errors.extend(source_errors)
            for document in source_docs:
                documents_by_id.setdefault(str(document["source_id"]), document)

    actual_source_docs = [
        document
        for document in documents_by_id.values()
        if document.get("source_type") in {"github_readme", "github_source"}
    ]
    corrective_search_used = False
    if not actual_source_docs:
        corrective_search_used = True
        repositories, repository_errors = _github_repository_search(query)
        errors.extend(repository_errors)
        for owner, repo in repositories[:_MAX_SOURCE_REPOS_PER_QUERY]:
            repo_ref = (owner, repo)
            if repo_ref in fetched_repos:
                continue
            fetched_repos.add(repo_ref)
            source_docs, source_errors = _github_repo_documents(owner, repo, query)
            errors.extend(source_errors)
            for document in source_docs:
                documents_by_id.setdefault(str(document["source_id"]), document)

    documents = list(documents_by_id.values())
    coverage = _coverage_score(query, documents)
    actual_source_count = sum(
        1
        for document in documents
        if document.get("source_type") in {"github_readme", "github_source"}
    )
    status = (
        "available"
        if actual_source_count
        else ("metadata_only" if documents else "unavailable")
    )
    return {
        "schema_version": "mmm/external-grounded-rag-v1",
        "status": status,
        "query": query,
        "query_variants": list(variants),
        "providers": ["modrinth_public", "github_public_source"],
        "credentials_required": False,
        "corrective_search_used": corrective_search_used,
        "project_count": len(projects_by_id),
        "source_repository_count": len(fetched_repos),
        "document_count": len(documents),
        "actual_source_document_count": actual_source_count,
        "coverage_score": coverage,
        "projects": list(projects_by_id.values()),
        "documents": documents,
        "errors": errors,
    }


def _augment_bundle(
    agentic_module: Any,
    payload: Mapping[str, Any],
    *,
    versions: Sequence[str],
    local_index: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    raw_domains = result.get("domains", [])
    domains = [dict(item) for item in raw_domains if isinstance(item, Mapping)]
    jobs: list[tuple[int, int, str]] = []
    for domain_index, domain in enumerate(domains):
        queries = domain.get("queries", [])
        if not isinstance(queries, list):
            continue
        copied_queries = [dict(item) for item in queries if isinstance(item, Mapping)]
        domain["queries"] = copied_queries
        for query_index, item in enumerate(copied_queries):
            query = str(item.get("query", "")).strip()
            if query:
                jobs.append((domain_index, query_index, query))

    def run(job: tuple[int, int, str]) -> tuple[int, int, dict[str, Any]]:
        domain_index, query_index, query = job
        return domain_index, query_index, _external_retrieval(query, versions)

    if jobs:
        with ThreadPoolExecutor(
            max_workers=max(1, min(4, len(jobs))),
            thread_name_prefix="mmm_grounded_rag",
        ) as pool:
            for domain_index, query_index, external in pool.map(run, jobs):
                domains[domain_index]["queries"][query_index]["external_rag"] = external

    result["domains"] = domains
    result["schema_version"] = "mmm/forced-pre-design-rag-v3"
    result["local_index"] = dict(local_index)
    if local_index.get("status") == "available":
        result["code_index_status"] = "available"
        result["code_index_path"] = str(local_index.get("index_path", ""))
    external_payloads = [
        query.get("external_rag")
        for domain in domains
        for query in domain.get("queries", [])
        if isinstance(query, Mapping) and isinstance(query.get("external_rag"), Mapping)
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
    """Bind grounded retrieval once to the pre-design forced-RAG owner."""
    global _INSTALLED
    if _INSTALLED:
        return

    original = agentic_module._forced_rag_bundle
    if getattr(original, "__mmm_grounded_rag__", False):
        _INSTALLED = True
        return

    def grounded_rag_bundle(
        router: Any,
        research_brief: Mapping[str, Any],
    ) -> dict[str, Any]:
        local_index = _ensure_local_index(agentic_module, router)
        payload = original(router, research_brief)
        versions = tuple(
            str(item) for item in payload.get("versions", []) if str(item).strip()
        )
        return _augment_bundle(
            agentic_module,
            payload,
            versions=versions,
            local_index=local_index,
        )

    grounded_rag_bundle.__name__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__qualname__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__mmm_grounded_rag__ = True
    grounded_rag_bundle.__wrapped__ = original
    agentic_module._forced_rag_bundle = grounded_rag_bundle
    _INSTALLED = True


__all__ = ["install"]
