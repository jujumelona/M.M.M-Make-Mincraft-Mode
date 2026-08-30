from __future__ import annotations

"""Bounded requirement/provider RAG coordinator shared by planning and reuse."""

import hashlib
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from . import research_grounded_rag_contract as _grounded


def _workers() -> int:
    raw = os.environ.get("MMM_GROUNDED_RAG_WORKERS", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 8
    return max(2, min(16, value))


def _text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedupe_text(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return tuple(out)


def _brief_queries(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    interesting = {
        "query", "statement", "semantic_statement", "objective", "capability",
        "requirement", "text", "purpose", "topic",
    }

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                walk(child, str(child_key).casefold())
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                walk(child, key)
            return
        if key in interesting and isinstance(node, str) and len(node.strip()) >= 3:
            found.append(node)

    walk(value)
    return _dedupe_text(found)[:24]


def _brief_versions(value: Any) -> tuple[str, ...]:
    found: list[str] = []

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                walk(child, str(child_key).casefold())
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                walk(child, key)
            return
        if key in {"version", "versions", "minecraft_version"} and isinstance(node, str):
            if node.strip():
                found.append(node.strip())

    walk(value)
    return _dedupe_text(found)[:4]


def _repository_key(document: Mapping[str, Any]) -> str:
    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        repo = str(metadata.get("repository") or "").strip()
        if repo:
            return repo.casefold()
    return ""


def _repo_url(repo: str) -> str:
    return f"https://github.com/{repo}" if repo else ""


class GroundedRAGCoordinator:
    """One process-wide bounded executor and deduplicated retrieval work graph."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=_workers(), thread_name_prefix="mmm-grounded-rag"
        )
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        self._donors_by_query: dict[str, list[dict[str, Any]]] = {}

    @property
    def max_workers(self) -> int:
        return int(getattr(self.executor, "_max_workers", _workers()))

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        return self.executor.submit(fn, *args, **kwargs)

    def _closure_documents(self, owner: str, repo: str, query: str) -> tuple[list[dict[str, Any]], list[str]]:
        documents, errors = _grounded._github_repo_documents(owner, repo, query)
        repo_api = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}"
        try:
            meta = _grounded._http_json(repo_api, github=True)
            branch = str(meta.get("default_branch", "main") or "main")
            tree = _grounded._http_json(
                f"{repo_api}/git/trees/{quote(branch)}?recursive=1", github=True
            )
            paths = [
                str(item.get("path", ""))
                for item in tree.get("tree", [])
                if isinstance(item, Mapping) and item.get("type") == "blob"
            ]
            closure: list[str] = []
            preferred_names = {
                "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
                "gradle.properties", "fabric.mod.json", "mods.toml", "neoforge.mods.toml",
            }
            for path in paths:
                folded = path.casefold()
                leaf = Path(path).name.casefold()
                if (
                    leaf in preferred_names
                    or "/src/test/" in f"/{folded}"
                    or "/src/gametest/" in f"/{folded}"
                    or "/src/main/resources/" in f"/{folded}"
                ):
                    closure.append(path)
            for path in closure[:10]:
                folded = path.casefold()
                source_id = f"github:{owner}/{repo}:{path}"
                if any(str(item.get("source_id")) == source_id for item in documents):
                    continue
                raw_url = (
                    f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repo)}/"
                    f"{quote(branch, safe='')}/{quote(path, safe='/')}"
                )
                try:
                    content = _grounded._http_text(raw_url)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"closure:{owner}/{repo}/{path}:{type(exc).__name__}: {exc}")
                    continue
                documents.append(
                    _grounded._source_document(
                        source_id=source_id,
                        title=f"{owner}/{repo}:{path}",
                        url=f"https://github.com/{owner}/{repo}/blob/{branch}/{path}",
                        content=content,
                        source_type="github_source_closure",
                        metadata={
                            "repository": f"{owner}/{repo}",
                            "branch": branch,
                            "path": path,
                            "closure_role": (
                                "test" if "/test/" in f"/{folded}"
                                else "resource" if "/resources/" in f"/{folded}"
                                else "build_dependency"
                            ),
                        },
                    )
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"closure_tree:{owner}/{repo}:{type(exc).__name__}: {exc}")
        return documents, errors

    def retrieve_many(
        self, queries: Sequence[str], versions: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        normalized_queries = _dedupe_text(tuple(queries))
        versions_key = tuple(_dedupe_text(tuple(str(v) for v in versions)))
        results: dict[str, dict[str, Any]] = {}
        pending_queries: list[str] = []
        with self._lock:
            for query in normalized_queries:
                cached = self._cache.get((query.casefold(), versions_key))
                if cached is None:
                    pending_queries.append(query)
                else:
                    results[query] = dict(cached)
        if not pending_queries:
            return results

        work: dict[Future[Any], tuple[str, str, str]] = {}
        for query in pending_queries:
            for variant in _grounded._query_variants(query):
                work[self.submit(_grounded._modrinth_search, variant, versions_key)] = (
                    query, "modrinth", variant
                )
                work[self.submit(_grounded._github_repository_search, variant)] = (
                    query, "github_search", variant
                )

        projects: dict[str, dict[str, dict[str, Any]]] = {q: {} for q in pending_queries}
        repos_by_query: dict[str, set[tuple[str, str]]] = {q: set() for q in pending_queries}
        errors_by_query: dict[str, list[str]] = {q: [] for q in pending_queries}
        for future in as_completed(work):
            query, provider, _variant = work[future]
            try:
                values, errors = future.result()
            except Exception as exc:  # noqa: BLE001
                errors_by_query[query].append(f"{provider}:{type(exc).__name__}: {exc}")
                continue
            errors_by_query[query].extend(str(item) for item in errors)
            if provider == "modrinth":
                for project in values:
                    project_id = str(project.get("project_id") or "")
                    if project_id:
                        projects[query].setdefault(project_id, dict(project))
                    repo_ref = _grounded._github_repo_from_url(str(project.get("source_url") or ""))
                    if repo_ref is not None:
                        repos_by_query[query].add(repo_ref)
            else:
                repos_by_query[query].update(values)

        repo_consumers: dict[tuple[str, str], set[str]] = {}
        for query, repo_refs in repos_by_query.items():
            ordered = sorted(repo_refs)[: max(2, int(getattr(_grounded, "_MAX_SOURCE_REPOS_PER_QUERY", 2)))]
            for repo_ref in ordered:
                repo_consumers.setdefault(repo_ref, set()).add(query)

        repo_work = {
            self.submit(self._closure_documents, owner, repo, next(iter(consumers))): (owner, repo)
            for (owner, repo), consumers in repo_consumers.items()
        }
        documents_by_query: dict[str, dict[str, dict[str, Any]]] = {
            q: {} for q in pending_queries
        }
        content_seen: set[str] = set()
        for future in as_completed(repo_work):
            repo_ref = repo_work[future]
            try:
                documents, repo_errors = future.result()
            except Exception as exc:  # noqa: BLE001
                documents, repo_errors = [], [f"repo:{repo_ref[0]}/{repo_ref[1]}:{type(exc).__name__}: {exc}"]
            consumers = repo_consumers.get(repo_ref, set())
            for query in consumers:
                errors_by_query[query].extend(repo_errors)
                for document in documents:
                    content_hash = str(document.get("content_sha256") or _text_hash(str(document.get("content") or "")))
                    dedup_key = f"{content_hash}:{str(document.get('source_id') or '')}"
                    if dedup_key in content_seen and str(document.get("source_id")) in documents_by_query[query]:
                        continue
                    content_seen.add(dedup_key)
                    documents_by_query[query].setdefault(str(document.get("source_id")), dict(document))

        for query in pending_queries:
            for project_id, project in projects[query].items():
                body = str(project.get("body") or project.get("description") or "")
                if body:
                    doc = _grounded._source_document(
                        source_id=f"modrinth:{project_id}",
                        title=str(project.get("title") or project_id),
                        url=str(project.get("project_url") or ""),
                        content=body,
                        source_type="modrinth_project",
                        metadata={
                            "project_id": project_id,
                            "source_url": project.get("source_url"),
                            "versions": project.get("versions"),
                        },
                    )
                    documents_by_query[query].setdefault(str(doc["source_id"]), doc)
            documents = list(documents_by_query[query].values())
            actual = sum(
                1 for item in documents
                if str(item.get("source_type", "")).startswith("github_")
            )
            payload = {
                "schema_version": "mmm/external-grounded-rag",
                "status": "available" if actual else ("metadata_only" if documents else "unavailable"),
                "query": query,
                "query_variants": list(_grounded._query_variants(query)),
                "providers": ["modrinth_public", "github_public_source"],
                "credentials_required": False,
                "corrective_search_used": bool(not projects[query]),
                "project_count": len(projects[query]),
                "source_repository_count": len(repos_by_query[query]),
                "document_count": len(documents),
                "actual_source_document_count": actual,
                "coverage_score": _grounded._coverage_score(query, documents),
                "projects": list(projects[query].values()),
                "documents": documents,
                "errors": errors_by_query[query],
                "work_graph": {
                    "key_space": "requirement_x_provider_x_query_purpose",
                    "bounded_workers": self.max_workers,
                    "nested_executor": False,
                },
            }
            with self._lock:
                self._cache[(query.casefold(), versions_key)] = dict(payload)
                self._donors_by_query[query.casefold()] = [
                    dict(item) for item in documents if _repository_key(item)
                ]
            results[query] = payload
        return results

    def repositories_for_capabilities(
        self, capabilities: Sequence[str], capability_graph: Mapping[str, Any] | None = None
    ) -> dict[str, tuple[str, ...]]:
        terms_by_capability: dict[str, list[str]] = {str(cap): [str(cap)] for cap in capabilities}
        if isinstance(capability_graph, Mapping):
            for item in capability_graph.get("search_terms", ()):
                if isinstance(item, Mapping):
                    cap = str(item.get("capability") or "")
                    if cap in terms_by_capability:
                        terms_by_capability[cap].extend(str(v) for v in item.get("terms", ()))
        with self._lock:
            donor_items = [(query, [dict(item) for item in docs]) for query, docs in self._donors_by_query.items()]
        result: dict[str, tuple[str, ...]] = {}
        for capability, terms in terms_by_capability.items():
            needles = set(_grounded._tokens(" ".join(terms)))
            scored: list[tuple[int, str]] = []
            seen: set[str] = set()
            for query, docs in donor_items:
                score = len(needles & set(_grounded._tokens(query)))
                if score <= 0:
                    continue
                for doc in docs:
                    repo = _repository_key(doc)
                    url = _repo_url(repo)
                    if url and url not in seen:
                        seen.add(url)
                        scored.append((score, url))
            scored.sort(key=lambda item: (-item[0], item[1]))
            result[capability] = tuple(url for _score, url in scored[:8])
        return result


_COORDINATOR = GroundedRAGCoordinator()
_INSTALLED = False


def _augment(
    agentic_module: Any,
    payload: Mapping[str, Any],
    *,
    local_index: Mapping[str, Any],
    external_by_query: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = dict(payload)
    domains = [dict(item) for item in result.get("domains", ()) if isinstance(item, Mapping)]
    donors: list[dict[str, Any]] = []
    for domain in domains:
        queries = [dict(item) for item in domain.get("queries", ()) if isinstance(item, Mapping)]
        domain["queries"] = queries
        for item in queries:
            query = str(item.get("query") or "").strip()
            external = external_by_query.get(query)
            if external is None:
                continue
            item["external_rag"] = dict(external)
            candidates = []
            for document in external.get("documents", ()):
                if not isinstance(document, Mapping):
                    continue
                repo = _repository_key(document)
                if not repo:
                    continue
                candidate = {
                    "requirement_id": str(item.get("requirement_id") or item.get("id") or ""),
                    "query": query,
                    "repository": repo,
                    "repository_url": _repo_url(repo),
                    "source_id": str(document.get("source_id") or ""),
                    "content_sha256": str(document.get("content_sha256") or ""),
                    "path": str((document.get("metadata") or {}).get("path") or "") if isinstance(document.get("metadata"), Mapping) else "",
                    "source_type": str(document.get("source_type") or ""),
                }
                candidates.append(candidate)
                donors.append(candidate)
            item["donor_candidates"] = candidates
    result["domains"] = domains
    result["schema_version"] = "mmm/forced-pre-design-rag"
    result["local_index"] = dict(local_index)
    if local_index.get("status") == "available":
        result["code_index_status"] = "available"
        result["code_index_path"] = str(local_index.get("index_path") or "")
    external_payloads = [
        item.get("external_rag")
        for domain in domains
        for item in domain.get("queries", ())
        if isinstance(item, Mapping) and isinstance(item.get("external_rag"), Mapping)
    ]
    result["external_source_count"] = sum(int(item.get("actual_source_document_count", 0)) for item in external_payloads)
    result["external_document_count"] = sum(int(item.get("document_count", 0)) for item in external_payloads)
    result["external_query_count"] = len(external_payloads)
    result["requirement_donor_candidates"] = donors
    result["retrieval_work_graph"] = {
        "key_space": "requirement_x_provider_x_query_purpose",
        "bounded_workers": _COORDINATOR.max_workers,
        "global_repo_dedup": True,
        "global_url_cache": True,
        "global_content_hash_dedup": True,
        "nested_executor": False,
    }
    result["research_sha256"] = agentic_module._sha256(
        {key: value for key, value in result.items() if key != "research_sha256"}
    )
    return result


def install(agentic_module: Any, reuse_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    current = agentic_module._forced_rag_bundle
    base = getattr(current, "__wrapped__", current)

    def forced_rag_bundle(router: Any, research_brief: Mapping[str, Any]) -> dict[str, Any]:
        pre_queries = _brief_queries(research_brief)
        pre_versions = _brief_versions(research_brief)
        original_future = _COORDINATOR.submit(base, router, research_brief)
        local_future = _COORDINATOR.submit(_grounded._ensure_local_index, agentic_module, router)
        pre_external = _COORDINATOR.retrieve_many(pre_queries, pre_versions) if pre_queries else {}
        payload = original_future.result()
        local_index = local_future.result()
        versions = tuple(str(item) for item in payload.get("versions", ()) if str(item).strip()) or pre_versions
        payload_queries = _dedupe_text(tuple(
            str(item.get("query") or "")
            for domain in payload.get("domains", ()) if isinstance(domain, Mapping)
            for item in domain.get("queries", ()) if isinstance(item, Mapping)
        ))
        missing = tuple(query for query in payload_queries if query not in pre_external)
        external = dict(pre_external)
        if missing:
            external.update(_COORDINATOR.retrieve_many(missing, versions))
        return _augment(
            agentic_module,
            payload,
            local_index=local_index,
            external_by_query=external,
        )

    forced_rag_bundle.__name__ = "_forced_rag_bundle_grounded"
    forced_rag_bundle.__qualname__ = "_forced_rag_bundle_grounded"
    forced_rag_bundle.__wrapped__ = base
    forced_rag_bundle.__mmm_grounded_rag__ = True
    forced_rag_bundle.__mmm_grounded_work_graph__ = True
    agentic_module._forced_rag_bundle = forced_rag_bundle

    original_discovery = reuse_module._parallel_donor_repository_discovery
    if not getattr(original_discovery, "__mmm_grounded_donors__", False):
        def donor_discovery(capabilities: Sequence[str], client: Any, *, capability_graph: Mapping[str, Any] | None = None):
            grounded = _COORDINATOR.repositories_for_capabilities(capabilities, capability_graph)
            public = original_discovery(capabilities, client, capability_graph=capability_graph)
            merged: dict[str, tuple[str, ...]] = {}
            for capability in capabilities:
                values = [*grounded.get(capability, ()), *public.get(capability, ())]
                merged[capability] = tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))
            return merged

        donor_discovery.__wrapped__ = original_discovery
        donor_discovery.__mmm_grounded_donors__ = True
        reuse_module._parallel_donor_repository_discovery = donor_discovery

    _grounded._external_retrieval = lambda query, versions: _COORDINATOR.retrieve_many((query,), versions)[query]
    _INSTALLED = True


def coordinator() -> GroundedRAGCoordinator:
    return _COORDINATOR


__all__ = ["GroundedRAGCoordinator", "coordinator", "install"]
