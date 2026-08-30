from __future__ import annotations

from pathlib import Path


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


# Allow seed-only source inspection after Modrinth and GitHub have run concurrently.
p = Path("minecraft_mod_ai/github_adaptive_retrieval.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "    seed_repositories: Sequence[Any] = (),\n) -> AdaptiveGitHubEvidence:",
    "    seed_repositories: Sequence[Any] = (),\n    search_if_needed: bool = True,\n) -> AdaptiveGitHubEvidence:",
)
old = '''            if discovery is None:
                discovery = discover_repositories(query, http_json=http_json)
                errors.extend(discovery.errors)
                for ref in discovery.repositories:
                    if ref not in seen_repos:
                        seen_repos.add(ref)
                        candidate_queue.append(ref)
                if candidate_index < len(candidate_queue):
                    continue
                reason = discovery.saturation_reason
            break
'''
new = '''            if discovery is None:
                if not search_if_needed:
                    reason = "seed_repositories_exhausted"
                    break
                discovery = discover_repositories(query, http_json=http_json)
                errors.extend(discovery.errors)
                for ref in discovery.repositories:
                    if ref not in seen_repos:
                        seen_repos.add(ref)
                        candidate_queue.append(ref)
                if candidate_index < len(candidate_queue):
                    continue
                reason = discovery.saturation_reason
            break
'''
if old not in s:
    raise SystemExit("adaptive seed-only anchor drifted")
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")


# Replace legacy GitHub retrieval with adaptive candidate generation + query-specific source inspection.
p = Path("minecraft_mod_ai/research_grounded_rag_contract.py")
s = p.read_text(encoding="utf-8")
s = s.replace("import base64\n", "", 1)
import_anchor = "from .rag_index import ProjectRAGIndex\n"
if import_anchor not in s:
    raise SystemExit("research import anchor drifted")
s = s.replace(
    import_anchor,
    '''from .github_adaptive_retrieval import (
    adaptive_github_evidence,
    discover_repositories,
    retrieve_repository_documents,
)
from .rag_index import ProjectRAGIndex
''',
    1,
)
s = s.replace("_MAX_SOURCE_FILES_PER_REPO = 4\n", "")
s = s.replace("_MAX_SOURCE_REPOS_PER_QUERY = 2\n", "")
if "_ALLOWED_SOURCE_SUFFIXES = (" in s:
    s = replace_between(s, "_ALLOWED_SOURCE_SUFFIXES = (", "_TOKEN = ", "")

if "def _path_score(" in s:
    s = replace_between(s, "def _path_score(", "def _github_repo_documents(", "")

repo_docs = '''def _github_repo_documents(
    owner: str,
    repo: str,
    query: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Query-specific source inspection with evidence/resource driven stopping."""

    evidence = retrieve_repository_documents(
        owner,
        repo,
        query,
        http_json=lambda url: _http_json(url, github=True),
        http_text=_http_text,
        source_document=_source_document,
    )
    return [dict(item) for item in evidence.documents], list(evidence.errors)
'''
s = replace_between(s, "def _github_repo_documents(", "def _modrinth_search(", repo_docs)

search_funcs = '''def _github_repository_search(query: str) -> tuple[list[tuple[str, str]], list[str]]:
    """High-recall repository candidate discovery; source inspection is separate."""

    discovery = discover_repositories(
        query,
        http_json=lambda url: _http_json(url, github=True),
    )
    return list(discovery.repositories), list(discovery.errors)


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
    return {
        "status": "available" if actual else "unavailable",
        "query": query,
        "repositories": list(evidence.repositories),
        "documents": documents,
        "errors": list(evidence.errors),
        "search_queries": list(evidence.search_queries),
        "search_requests": evidence.search_requests,
        "source_requests": evidence.source_requests,
        "source_bytes": evidence.source_bytes,
        "coverage_score": evidence.coverage_score,
        "saturation_reason": evidence.saturation_reason,
        "actual_source_document_count": actual,
    }
'''
s = replace_between(s, "def _github_repository_search(", "def _coverage_score(", search_funcs)

external = '''def _external_retrieval(query: str, versions: Sequence[str]) -> dict[str, Any]:
    variants = _query_variants(query)
    projects_by_id: dict[str, dict[str, Any]] = {}
    documents_by_id: dict[str, dict[str, Any]] = {}
    seed_repositories: list[tuple[str, str]] = []
    errors: list[str] = []

    for variant in variants:
        projects, variant_errors = _modrinth_search(variant, versions)
        errors.extend(variant_errors)
        for project in projects:
            project_id = str(project.get("project_id", ""))
            if project_id:
                projects_by_id.setdefault(project_id, project)
            body = str(project.get("body") or project.get("description") or "")
            if body and project_id:
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
                documents_by_id.setdefault(str(document["source_id"]), document)
            repo_ref = _github_repo_from_url(str(project.get("source_url") or ""))
            if repo_ref is not None and repo_ref not in seed_repositories:
                seed_repositories.append(repo_ref)

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
    actual_source_count = sum(
        1
        for document in documents
        if str(document.get("source_type") or "").startswith("github_")
    )
    status = (
        "available"
        if actual_source_count
        else ("metadata_only" if documents else "unavailable")
    )
    return {
        "schema_version": "mmm/external-grounded-rag",
        "status": status,
        "query": query,
        "query_variants": list(variants),
        "github_search_queries": list(github.get("search_queries", ())),
        "providers": ["modrinth_public", "github_public_source"],
        "credentials_required": False,
        "corrective_search_used": bool(github.get("search_queries")),
        "project_count": len(projects_by_id),
        "source_repository_count": len(github.get("repositories", ())),
        "document_count": len(documents),
        "actual_source_document_count": actual_source_count,
        "coverage_score": max(
            _coverage_score(query, documents),
            float(github.get("coverage_score") or 0.0),
        ),
        "projects": list(projects_by_id.values()),
        "documents": documents,
        "errors": errors,
        "github_retrieval": {
            "search_requests": int(github.get("search_requests") or 0),
            "source_requests": int(github.get("source_requests") or 0),
            "source_bytes": int(github.get("source_bytes") or 0),
            "saturation_reason": str(github.get("saturation_reason") or ""),
        },
    }
'''
s = replace_between(s, "def _external_retrieval(", "def _augment_bundle(", external)
p.write_text(s, encoding="utf-8")


# Coordinator: cache/dedup repo transport globally, but choose source evidence per query.
p = Path("minecraft_mod_ai/grounded_rag_runtime_contract.py")
s = p.read_text(encoding="utf-8")
s = s.replace("from pathlib import Path\n", "")
s = s.replace("from urllib.parse import quote\n", "")

closure = '''    def _closure_documents(
        self, owner: str, repo: str, query: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        # Compatibility seam. Adaptive retrieval owns source + build/test/resource
        # closure and has no fixed file-count cutoff.
        return _grounded._github_repo_documents(owner, repo, query)
'''
s = replace_between(s, "    def _closure_documents(", "    def retrieve_many(", closure)

retrieve = '''    def retrieve_many(
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
            work[self.submit(_grounded._github_adaptive_search, query)] = (
                query, "github", query
            )

        projects: dict[str, dict[str, dict[str, Any]]] = {q: {} for q in pending_queries}
        repositories_by_query: dict[str, list[tuple[str, str]]] = {
            q: [] for q in pending_queries
        }
        documents_by_query: dict[str, dict[str, dict[str, Any]]] = {
            q: {} for q in pending_queries
        }
        errors_by_query: dict[str, list[str]] = {q: [] for q in pending_queries}
        github_stats: dict[str, dict[str, Any]] = {q: {} for q in pending_queries}

        def register_document(query: str, document: Mapping[str, Any]) -> None:
            source_id = str(document.get("source_id") or "")
            if not source_id:
                return
            item = dict(document)
            content_hash = str(
                item.get("content_sha256")
                or _text_hash(str(item.get("content") or ""))
            )
            with self._lock:
                canonical = self._documents_by_hash.setdefault(content_hash, item)
            if canonical is not item and not item.get("content"):
                item["content"] = canonical.get("content", "")
            documents_by_query[query].setdefault(source_id, item)

        for future in as_completed(work):
            query, provider, _variant = work[future]
            try:
                returned = future.result()
            except Exception as exc:  # noqa: BLE001
                errors_by_query[query].append(f"{provider}:{type(exc).__name__}: {exc}")
                continue

            if provider == "modrinth":
                values, provider_errors = returned
                errors_by_query[query].extend(str(item) for item in provider_errors)
                for project in values:
                    project_id = str(project.get("project_id") or "")
                    if project_id:
                        projects[query].setdefault(project_id, dict(project))
                continue

            payload = dict(returned) if isinstance(returned, Mapping) else {}
            github_stats[query] = payload
            errors_by_query[query].extend(str(item) for item in payload.get("errors", ()))
            for raw_ref in payload.get("repositories", ()):
                if (
                    isinstance(raw_ref, Sequence)
                    and not isinstance(raw_ref, (str, bytes, bytearray))
                    and len(raw_ref) >= 2
                ):
                    ref = (str(raw_ref[0]), str(raw_ref[1]))
                    if ref not in repositories_by_query[query]:
                        repositories_by_query[query].append(ref)
            for document in payload.get("documents", ()):
                if isinstance(document, Mapping):
                    register_document(query, document)

        # Modrinth source URLs are high-confidence GitHub seeds discovered in parallel.
        # Inspect every new seed for this query; never share a file selection chosen for
        # another requirement just because the repository identity is the same.
        for query in pending_queries:
            seeds: list[tuple[str, str]] = []
            known = set(repositories_by_query[query])
            for project in projects[query].values():
                repo_ref = _grounded._github_repo_from_url(str(project.get("source_url") or ""))
                if repo_ref is not None and repo_ref not in known and repo_ref not in seeds:
                    seeds.append(repo_ref)
            if seeds:
                try:
                    seeded = _grounded._github_adaptive_search(
                        query,
                        seed_repositories=tuple(seeds),
                        search_if_needed=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors_by_query[query].append(f"github_seed:{type(exc).__name__}: {exc}")
                else:
                    errors_by_query[query].extend(str(item) for item in seeded.get("errors", ()))
                    for raw_ref in seeded.get("repositories", ()):
                        if (
                            isinstance(raw_ref, Sequence)
                            and not isinstance(raw_ref, (str, bytes, bytearray))
                            and len(raw_ref) >= 2
                        ):
                            ref = (str(raw_ref[0]), str(raw_ref[1]))
                            if ref not in repositories_by_query[query]:
                                repositories_by_query[query].append(ref)
                    for document in seeded.get("documents", ()):
                        if isinstance(document, Mapping):
                            register_document(query, document)

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
                    register_document(query, doc)

            documents = list(documents_by_query[query].values())
            actual = sum(
                1 for item in documents
                if str(item.get("source_type", "")).startswith("github_")
            )
            stats = github_stats[query]
            payload = {
                "schema_version": "mmm/external-grounded-rag",
                "status": "available" if actual else ("metadata_only" if documents else "unavailable"),
                "query": query,
                "query_variants": list(_grounded._query_variants(query)),
                "github_search_queries": list(stats.get("search_queries", ())),
                "providers": ["modrinth_public", "github_public_source"],
                "credentials_required": False,
                "corrective_search_used": bool(stats.get("search_queries")),
                "project_count": len(projects[query]),
                "source_repository_count": len(repositories_by_query[query]),
                "document_count": len(documents),
                "actual_source_document_count": actual,
                "coverage_score": max(
                    _grounded._coverage_score(query, documents),
                    float(stats.get("coverage_score") or 0.0),
                ),
                "projects": list(projects[query].values()),
                "documents": documents,
                "errors": errors_by_query[query],
                "github_retrieval": {
                    "search_requests": int(stats.get("search_requests") or 0),
                    "source_requests": int(stats.get("source_requests") or 0),
                    "source_bytes": int(stats.get("source_bytes") or 0),
                    "saturation_reason": str(stats.get("saturation_reason") or ""),
                },
                "work_graph": {
                    "key_space": "requirement_x_provider_x_query_purpose",
                    "bounded_workers": self.max_workers,
                    "nested_executor": False,
                    "repository_snapshot_dedup": True,
                    "query_specific_source_selection": True,
                    "fixed_file_count_cutoff": False,
                },
            }
            with self._lock:
                self._cache[(query.casefold(), versions_key)] = dict(payload)
                self._donors_by_query[query.casefold()] = [
                    dict(item) for item in documents if _repository_key(item)
                ]
            results[query] = payload
        return results
'''
s = replace_between(s, "    def retrieve_many(", "    def repositories_for_capabilities(", retrieve)

init_anchor = "        self._donors_by_query: dict[str, list[dict[str, Any]]] = {}\n"
if init_anchor not in s:
    raise SystemExit("coordinator init anchor drifted")
s = s.replace(
    init_anchor,
    init_anchor + "        self._documents_by_hash: dict[str, dict[str, Any]] = {}\n",
    1,
)
p.write_text(s, encoding="utf-8")


# Update the concurrency seam and prove same-repo source selection is query-specific.
p = Path("tests/test_grounded_rag_runtime_contract.py")
s = p.read_text(encoding="utf-8")
old = '''    def github(query):
        with lock:
            entered.append(("github", query))
        barrier.wait(timeout=5)
        return [], []

    monkeypatch.setattr(grounded, "_modrinth_search", modrinth)
    monkeypatch.setattr(grounded, "_github_repository_search", github)
'''
new = '''    def github(query, **kwargs):
        del kwargs
        with lock:
            entered.append(("github", query))
        barrier.wait(timeout=5)
        return {
            "repositories": [],
            "documents": [],
            "errors": [],
            "search_queries": [],
            "search_requests": 0,
            "source_requests": 0,
            "source_bytes": 0,
            "coverage_score": 0.0,
            "saturation_reason": "test",
        }

    monkeypatch.setattr(grounded, "_modrinth_search", modrinth)
    monkeypatch.setattr(grounded, "_github_adaptive_search", github)
'''
if old not in s:
    raise SystemExit("provider overlap test anchor drifted")
s = s.replace(old, new, 1)

if "def test_same_repository_source_selection_is_recomputed_per_query" not in s:
    s += '''\n\ndef test_same_repository_source_selection_is_recomputed_per_query(monkeypatch):
    coordinator = runtime.GroundedRAGCoordinator()
    calls: list[str] = []

    monkeypatch.setattr(grounded, "_modrinth_search", lambda query, versions: ([], []))

    def github(query, **kwargs):
        del kwargs
        calls.append(query)
        path = (
            "src/main/java/demo/TradeLedger.java"
            if "trade" in query
            else "src/main/java/demo/PlanetColony.java"
        )
        return {
            "repositories": [("owner", "shared-repo")],
            "documents": [
                {
                    "source_id": f"github:owner/shared-repo:{path}",
                    "content": f"class {path.rsplit('/', 1)[-1].removesuffix('.java')} {{}}",
                    "content_sha256": "sha256:" + query.replace(" ", "_"),
                    "source_type": "github_source",
                    "metadata": {"repository": "owner/shared-repo", "path": path},
                }
            ],
            "errors": [],
            "search_queries": [query],
            "search_requests": 1,
            "source_requests": 3,
            "source_bytes": 100,
            "coverage_score": 1.0,
            "saturation_reason": "evidence_coverage_satisfied",
        }

    monkeypatch.setattr(grounded, "_github_adaptive_search", github)
    result = coordinator.retrieve_many(("trade currency", "planet colony"), ())

    trade_ids = {item["source_id"] for item in result["trade currency"]["documents"]}
    planet_ids = {item["source_id"] for item in result["planet colony"]["documents"]}
    assert "github:owner/shared-repo:src/main/java/demo/TradeLedger.java" in trade_ids
    assert "github:owner/shared-repo:src/main/java/demo/PlanetColony.java" in planet_ids
    assert set(calls) == {"trade currency", "planet colony"}
    assert all(
        item["work_graph"]["query_specific_source_selection"] is True
        for item in result.values()
    )
    coordinator.executor.shutdown(wait=True, cancel_futures=True)
'''
p.write_text(s, encoding="utf-8")


# Architectural guard: the old recall-destroying patterns must not survive this patch.
for source_path in (
    Path("minecraft_mod_ai/research_grounded_rag_contract.py"),
    Path("minecraft_mod_ai/grounded_rag_runtime_contract.py"),
):
    text = source_path.read_text(encoding="utf-8")
    forbidden = (
        "_MAX_SOURCE_FILES_PER_REPO",
        "_MAX_SOURCE_REPOS_PER_QUERY",
        "closure[:10]",
        "next(iter(consumers))",
    )
    found = [needle for needle in forbidden if needle in text]
    if found:
        raise SystemExit(f"{source_path}: legacy recall cutoffs remain: {found}")
