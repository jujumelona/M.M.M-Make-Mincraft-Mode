from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
GH = ROOT / "minecraft_mod_ai" / "github_adaptive_retrieval.py"
RG = ROOT / "minecraft_mod_ai" / "research_grounded_rag_contract.py"
PD = ROOT / "minecraft_mod_ai" / "pre_design_research_pipeline.py"
TEST = ROOT / "tests" / "test_exhaustive_retrieval_contract.py"

GH_MARKER = "# === MMM EXHAUSTIVE RETRIEVAL OVERRIDES ==="
RG_MARKER = "# === MMM EXHAUSTIVE PUBLIC PROVIDER OVERRIDES ==="


def append_once(path: Path, marker: str, payload: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + payload.strip() + "\n", encoding="utf-8")


gh_override = r'''
# === MMM EXHAUSTIVE RETRIEVAL OVERRIDES ===
# Discovery completeness is controlled by reachable-frontier exhaustion, never by
# a local result/page/request/byte count or a coverage threshold.  Concurrency and
# transport timeouts may still be bounded because they do not change which frontier
# nodes are eventually visited.


def _retrieval_failure_kind(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".casefold()
    if "rate limit" in text or "secondary rate" in text or "429" in text:
        return "rate_limited"
    if "403" in text:
        return "rate_limited"
    if "422" in text or "1000" in text or "search limit" in text:
        return "provider_limit"
    return "error"


def discover_repositories(query: str, *, http_json: HttpJSON) -> RepositoryDiscovery:
    """Exhaust every reachable GitHub repository-search page for every query variant."""

    ladder = repository_query_ladder(query)
    repositories: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    errors: list[str] = []
    executed: list[str] = []
    requests = 0
    active: list[tuple[str, int]] = [(variant, 1) for variant in ladder]
    terminal = "frontier_exhausted"

    while active:
        next_pages: list[tuple[str, int]] = []
        provider_blocked = False
        for variant, page in active:
            requests += 1
            executed.append(variant if page == 1 else f"{variant} [page {page}]")
            try:
                # per_page is a transport page size, not a retrieval/result cutoff.
                payload = http_json(_github_search_url(variant, page=page, per_page=100))
            except Exception as exc:  # noqa: BLE001
                kind = _retrieval_failure_kind(exc)
                errors.append(
                    f"github_repository_search:{variant}:p{page}:{kind}:"
                    f"{type(exc).__name__}: {exc}"
                )
                if kind in {"rate_limited", "provider_limit"}:
                    terminal = kind
                    provider_blocked = True
                    break
                continue

            if not isinstance(payload, Mapping):
                errors.append(f"github_repository_search:{variant}:p{page}:error:invalid_payload")
                continue
            if bool(payload.get("incomplete_results")):
                terminal = "provider_limit"

            items = payload.get("items", [])
            if not isinstance(items, list):
                errors.append(f"github_repository_search:{variant}:p{page}:error:invalid_items")
                continue
            if not items:
                continue

            for item in items:
                if not isinstance(item, Mapping):
                    continue
                ref = _repo_tuple(str(item.get("full_name") or ""))
                if ref is None or ref in seen:
                    continue
                seen.add(ref)
                repositories.append(ref)

            # Do not stop merely because this page was duplicate-only.  A later page
            # can still contain a new repository.  Only an empty page exhausts this
            # query variant's reachable frontier.
            next_pages.append((variant, page + 1))

        if provider_blocked:
            break
        active = next_pages

    if not repositories and not errors and terminal == "frontier_exhausted":
        terminal = "ok_zero"
    return RepositoryDiscovery(
        repositories=tuple(repositories),
        errors=tuple(errors),
        search_queries=tuple(executed),
        search_requests=requests,
        saturation_reason=terminal,
    )


def _walk_complete_tree(
    repo_api: str,
    branch: str,
    *,
    http_json: HttpJSON,
) -> tuple[list[tuple[str, int | None]], list[str], int, str]:
    """Traverse Git trees to exhaustion; recursive-tree truncation triggers subtree walk."""

    errors: list[str] = []
    requests = 0
    try:
        requests += 1
        first = http_json(f"{repo_api}/git/trees/{quote(branch, safe='')}?recursive=1")
    except Exception as exc:  # noqa: BLE001
        kind = _retrieval_failure_kind(exc)
        return [], [f"github_tree:{kind}:{type(exc).__name__}: {exc}"], requests, kind

    if isinstance(first, Mapping) and not bool(first.get("truncated")):
        out: list[tuple[str, int | None]] = []
        raw = first.get("tree", [])
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, Mapping) or item.get("type") != "blob":
                continue
            path = str(item.get("path") or "").strip()
            if not path.casefold().endswith(_SOURCE_SUFFIXES):
                continue
            raw_size = item.get("size")
            size = int(raw_size) if isinstance(raw_size, int) and raw_size >= 0 else None
            out.append((path, size))
        return out, errors, requests, "frontier_exhausted"

    # GitHub's recursive tree response can be truncated.  Fall back to an unbounded
    # breadth-first subtree traversal using tree SHAs.  The visited set prevents cycles;
    # there is deliberately no depth or node-count stop.
    try:
        requests += 1
        root = http_json(f"{repo_api}/git/trees/{quote(branch, safe='')}")
    except Exception as exc:  # noqa: BLE001
        kind = _retrieval_failure_kind(exc)
        return [], [f"github_tree_root:{kind}:{type(exc).__name__}: {exc}"], requests, kind

    queue: list[tuple[str, str]] = [("", str(root.get("sha") or branch))]
    # If the root response already contains entries, process it without refetching.
    prefetched: dict[str, Any] = {str(root.get("sha") or branch): root}
    visited: set[str] = set()
    blobs: list[tuple[str, int | None]] = []
    terminal = "frontier_exhausted"

    while queue:
        prefix, tree_ref = queue.pop(0)
        if tree_ref in visited:
            continue
        visited.add(tree_ref)
        payload = prefetched.pop(tree_ref, None)
        if payload is None:
            try:
                requests += 1
                payload = http_json(f"{repo_api}/git/trees/{quote(tree_ref, safe='')}")
            except Exception as exc:  # noqa: BLE001
                kind = _retrieval_failure_kind(exc)
                errors.append(f"github_subtree:{tree_ref}:{kind}:{type(exc).__name__}: {exc}")
                if kind in {"rate_limited", "provider_limit"}:
                    terminal = kind
                    break
                continue
        raw = payload.get("tree", []) if isinstance(payload, Mapping) else []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("path") or "").strip()
            path = f"{prefix}/{name}".strip("/")
            item_type = str(item.get("type") or "")
            sha = str(item.get("sha") or "").strip()
            if item_type == "tree" and sha:
                queue.append((path, sha))
            elif item_type == "blob" and path.casefold().endswith(_SOURCE_SUFFIXES):
                raw_size = item.get("size")
                size = int(raw_size) if isinstance(raw_size, int) and raw_size >= 0 else None
                blobs.append((path, size))
    return blobs, errors, requests, terminal


def retrieve_repository_documents(
    owner: str,
    repo: str,
    query: str,
    *,
    http_json: HttpJSON,
    http_text: HttpText,
    source_document: SourceDocument,
    request_budget: int | None = None,
    byte_budget: int | None = None,
    coverage_target: float | None = None,
) -> RepositoryEvidence:
    """Fetch the complete relevant source/test/resource/build closure of one repository."""

    # Legacy budget arguments are intentionally ignored.  They remain in the signature
    # for API compatibility only and cannot terminate discovery.
    del request_budget, byte_budget, coverage_target
    errors: list[str] = []
    requests = 0
    source_bytes = 0
    query_features = _feature_set(query)
    covered: set[str] = set()
    documents: list[tuple[int, dict[str, Any]]] = []
    repo_api = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}"

    try:
        requests += 1
        meta = http_json(repo_api)
    except Exception as exc:  # noqa: BLE001
        kind = _retrieval_failure_kind(exc)
        return RepositoryEvidence(
            (owner, repo), (), (f"github_repo:{kind}:{type(exc).__name__}: {exc}",),
            requests, 0, frozenset(), 0.0, False, kind,
        )

    branch = str(meta.get("default_branch") or "main") if isinstance(meta, Mapping) else "main"
    html_url = str(meta.get("html_url") or f"https://github.com/{owner}/{repo}") if isinstance(meta, Mapping) else f"https://github.com/{owner}/{repo}"
    license_value = None
    if isinstance(meta, Mapping) and isinstance(meta.get("license"), Mapping):
        license_value = meta["license"].get("spdx_id")

    candidates, tree_errors, tree_requests, terminal = _walk_complete_tree(
        repo_api, branch, http_json=http_json
    )
    requests += tree_requests
    errors.extend(tree_errors)
    if terminal in {"rate_limited", "provider_limit"}:
        return RepositoryEvidence(
            (owner, repo), (), tuple(errors), requests, 0, frozenset(), 0.0, True, terminal,
        )

    # Ranking changes read order only.  Every candidate is fetched eventually.
    candidates.sort(key=lambda item: _path_score(item[0], query_features))
    for path, _advertised_size in candidates:
        raw_url = (
            f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repo)}/"
            f"{quote(branch, safe='')}/{quote(path, safe='/')}"
        )
        try:
            requests += 1
            content = http_text(raw_url)
        except Exception as exc:  # noqa: BLE001
            kind = _retrieval_failure_kind(exc)
            errors.append(f"github_raw:{owner}/{repo}/{path}:{kind}:{type(exc).__name__}: {exc}")
            if kind in {"rate_limited", "provider_limit"}:
                terminal = kind
                break
            continue
        source_bytes += len(content.encode("utf-8", errors="replace"))
        score, newly_covered = _content_score(path, content, query_features)
        covered.update(newly_covered)
        role = _path_role(path)
        documents.append(
            (
                score,
                source_document(
                    source_id=f"github:{owner}/{repo}:{path}",
                    title=f"{owner}/{repo}:{path}",
                    url=f"{html_url}/blob/{quote(branch, safe='')}/{quote(path, safe='/')}",
                    content=content,
                    source_type=("github_source" if role == "code" else "github_source_closure"),
                    metadata={
                        "repository": f"{owner}/{repo}",
                        "branch": branch,
                        "path": path,
                        "license": license_value,
                        "closure_role": role,
                        "query_evidence_score": score,
                    },
                ),
            )
        )

    ordered = tuple(
        document
        for _score, document in sorted(
            documents,
            key=lambda item: (-item[0], str(item[1].get("source_id") or "")),
        )
    )
    return RepositoryEvidence(
        repository=(owner, repo),
        documents=ordered,
        errors=tuple(errors),
        requests_used=requests,
        source_bytes=source_bytes,
        covered_features=frozenset(covered),
        coverage_score=round(_coverage(query_features, covered), 4),
        tree_truncated=False,
        saturation_reason=terminal,
    )


def adaptive_github_evidence(
    query: str,
    *,
    http_json: HttpJSON,
    http_text: HttpText,
    source_document: SourceDocument,
    seed_repositories: Sequence[Any] = (),
    search_if_needed: bool = True,
) -> AdaptiveGitHubEvidence:
    """Exhaust seed and discovered repository frontiers; coverage is diagnostic only."""

    query_features = _feature_set(query)
    documents: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    covered: set[str] = set()
    inspected: list[tuple[str, str]] = []
    seen_repos: set[tuple[str, str]] = set()
    source_requests = 0
    source_bytes = 0

    queue: list[tuple[str, str]] = []
    for value in seed_repositories:
        ref = _repo_tuple(value)
        if ref is not None and ref not in seen_repos:
            seen_repos.add(ref)
            queue.append(ref)

    discovery = RepositoryDiscovery((), (), (), 0, "search_not_needed")
    if search_if_needed:
        discovery = discover_repositories(query, http_json=http_json)
        errors.extend(discovery.errors)
        for ref in discovery.repositories:
            if ref not in seen_repos:
                seen_repos.add(ref)
                queue.append(ref)

    terminal = discovery.saturation_reason if search_if_needed else "frontier_exhausted"
    for owner, repo in queue:
        evidence = retrieve_repository_documents(
            owner,
            repo,
            query,
            http_json=http_json,
            http_text=http_text,
            source_document=source_document,
        )
        inspected.append((owner, repo))
        source_requests += evidence.requests_used
        source_bytes += evidence.source_bytes
        errors.extend(evidence.errors)
        covered.update(evidence.covered_features)
        for document in evidence.documents:
            source_id = str(document.get("source_id") or "")
            if source_id:
                documents.setdefault(source_id, dict(document))
        if evidence.saturation_reason in {"rate_limited", "provider_limit"}:
            terminal = evidence.saturation_reason
            break

    if terminal == "frontier_exhausted" and not inspected and not documents and not errors:
        terminal = "ok_zero"
    return AdaptiveGitHubEvidence(
        repositories=tuple(inspected),
        documents=tuple(documents.values()),
        errors=tuple(errors),
        search_queries=discovery.search_queries,
        search_requests=discovery.search_requests,
        source_requests=source_requests,
        source_bytes=source_bytes,
        coverage_score=round(_coverage(query_features, covered), 4),
        saturation_reason=terminal,
    )
'''

rg_override = r'''
# === MMM EXHAUSTIVE PUBLIC PROVIDER OVERRIDES ===
# These late-bound definitions intentionally replace the earlier transport helpers.
# They remove local cardinality/byte truncation from discovery and source ingestion.


def _query_variants(query: str) -> tuple[str, ...]:
    original = str(query).strip()
    normalized = re.sub(r"[._:/\\-]+", " ", original)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    implementation = f"{normalized} minecraft fabric mod source implementation".strip()
    return tuple(_dedupe((original, normalized, implementation)))


def _http_bytes(url: str, *, github: bool = False) -> bytes:
    cache_key = (url, github)
    with _HTTP_CACHE_LOCK:
        cached = _HTTP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    request = Request(url, headers=_request_headers(github=github))
    with urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read()
    with _HTTP_CACHE_LOCK:
        # Cache eviction affects only refetch cost; it never truncates retrieval.
        if len(_HTTP_CACHE) >= _MAX_HTTP_CACHE_ITEMS:
            _HTTP_CACHE.pop(next(iter(_HTTP_CACHE)))
        _HTTP_CACHE[cache_key] = payload
    return payload


def _source_document(
    *,
    source_id: str,
    title: str,
    url: str,
    content: str,
    source_type: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    text = str(content)
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "source_type": source_type,
        "content": text,
        "content_sha256": _content_sha256(text),
        "metadata": dict(metadata or {}),
    }


def _modrinth_search(
    query: str,
    versions: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Paginate Modrinth until its reported result frontier is exhausted."""

    errors: list[str] = []
    facets: list[list[str]] = [["project_type:mod"]]
    concrete_versions = [
        str(version).strip()
        for version in versions
        if str(version).strip()
        and str(version).strip() not in {"*", "target-neutral", "unknown"}
    ]
    if concrete_versions:
        facets.append([f"versions:{version}" for version in concrete_versions])

    projects: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    while True:
        params = urlencode(
            {
                "query": query,
                "offset": offset,
                # This is only transport page width; it is not a total-result cap.
                "limit": 100,
                "index": "relevance",
                "facets": json.dumps(facets, separators=(",", ":")),
            }
        )
        try:
            payload = _http_json(f"https://api.modrinth.com/v2/search?{params}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"modrinth_search:error:{type(exc).__name__}: {exc}")
            break
        hits = payload.get("hits", []) if isinstance(payload, Mapping) else []
        if not isinstance(hits, list) or not hits:
            break
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            project_id = str(hit.get("project_id", "")).strip()
            if not project_id or project_id in seen:
                continue
            seen.add(project_id)
            try:
                detail = _http_json(f"https://api.modrinth.com/v2/project/{quote(project_id)}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"modrinth_project:{project_id}:error:{type(exc).__name__}: {exc}")
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
        offset += len(hits)
        total_hits = payload.get("total_hits") if isinstance(payload, Mapping) else None
        if isinstance(total_hits, int) and offset >= total_hits:
            break
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
        1 for item in documents
        if str(item.get("source_type") or "").startswith("github_")
    )
    terminal = str(evidence.saturation_reason or "")
    if terminal in {"rate_limited", "provider_limit"}:
        provider_status = terminal
    elif terminal == "ok_zero" and not actual:
        provider_status = "ok_zero"
    elif evidence.errors and not actual:
        provider_status = "error"
    else:
        provider_status = "exhausted"
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
        "saturation_reason": terminal,
        "actual_source_document_count": actual,
    }
'''

append_once(GH, GH_MARKER, gh_override)
append_once(RG, RG_MARKER, rg_override)

# Remove the pre-design official-evidence top-four stop. Positive relevance, not an
# arbitrary ordinal, decides inclusion.
pd = PD.read_text(encoding="utf-8")
pd = pd.replace(
    "        selected.append(record)\n        if len(selected) >= 4:\n            break\n",
    "        selected.append(record)\n",
)
PD.write_text(pd, encoding="utf-8")

TEST.write_text(r'''from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from minecraft_mod_ai import github_adaptive_retrieval as gh
from minecraft_mod_ai import research_grounded_rag_contract as rg


def _doc(**kwargs):
    return dict(kwargs)


def test_repository_discovery_follows_duplicate_only_page_until_empty():
    calls = []

    def fake_json(url: str):
        calls.append(url)
        page = int(parse_qs(urlparse(url).query)["page"][0])
        # Every query-variant has a duplicate-only middle page and a new later page.
        if page == 1:
            return {"items": [{"full_name": "a/one"}], "incomplete_results": False}
        if page == 2:
            return {"items": [{"full_name": "a/one"}], "incomplete_results": False}
        if page == 3:
            return {"items": [{"full_name": "b/two"}], "incomplete_results": False}
        return {"items": [], "incomplete_results": False}

    result = gh.discover_repositories("space travel", http_json=fake_json)
    assert ("b", "two") in result.repositories
    assert result.saturation_reason == "frontier_exhausted"
    assert any("page=4" in url for url in calls)


def test_repository_discovery_distinguishes_provider_limit_from_exhaustion():
    def fake_json(_url: str):
        raise RuntimeError("HTTP Error 422: search limit reached")

    result = gh.discover_repositories("space travel", http_json=fake_json)
    assert result.saturation_reason == "provider_limit"
    assert result.errors


def test_repository_document_retrieval_ignores_legacy_cardinality_budgets():
    tree = {
        "truncated": False,
        "tree": [
            {"type": "blob", "path": "src/main/java/A.java", "size": 999999},
            {"type": "blob", "path": "src/test/java/ATest.java", "size": 999999},
            {"type": "blob", "path": "src/main/resources/x.json", "size": 999999},
        ],
    }

    def fake_json(url: str):
        if "/git/trees/" in url:
            return tree
        return {"default_branch": "main", "html_url": "https://github.com/a/b", "license": {"spdx_id": "MIT"}}

    result = gh.retrieve_repository_documents(
        "a", "b", "space travel", http_json=fake_json,
        http_text=lambda url: url,
        source_document=_doc,
        request_budget=0,
        byte_budget=0,
        coverage_target=0.0,
    )
    assert len(result.documents) == len(tree["tree"])
    assert result.saturation_reason == "frontier_exhausted"


def test_recursive_tree_truncation_falls_back_to_subtree_frontier():
    root_sha = "rootsha"
    child_sha = "childsha"

    def fake_json(url: str):
        if "?recursive=1" in url:
            return {"truncated": True, "tree": []}
        if url.endswith("/git/trees/main"):
            return {"sha": root_sha, "tree": [{"type": "tree", "path": "src", "sha": child_sha}]}
        if url.endswith(f"/git/trees/{child_sha}"):
            return {"sha": child_sha, "tree": [{"type": "blob", "path": "A.java", "size": 1}]}
        return {"default_branch": "main", "html_url": "https://github.com/a/b"}

    result = gh.retrieve_repository_documents(
        "a", "b", "space", http_json=fake_json,
        http_text=lambda _url: "class A {}", source_document=_doc,
    )
    assert any(str(item.get("source_id", "")).endswith("src/A.java") for item in result.documents)
    assert result.saturation_reason == "frontier_exhausted"


def test_source_document_keeps_complete_content():
    text = "x" * (rg._MAX_SOURCE_TEXT_CHARS + 1)
    doc = rg._source_document(
        source_id="x", title="x", url="https://example.invalid/x",
        content=text, source_type="test",
    )
    assert doc["content"] == text


def test_modrinth_paginates_to_reported_total(monkeypatch):
    offsets = []

    def fake_json(url: str, *, github: bool = False):
        del github
        if "/v2/project/" in url:
            return {}
        qs = parse_qs(urlparse(url).query)
        offset = int(qs.get("offset", ["0"])[0])
        offsets.append(offset)
        if offset == 0:
            return {"hits": [{"project_id": "p1", "slug": "p1"}], "total_hits": 2}
        return {"hits": [{"project_id": "p2", "slug": "p2"}], "total_hits": 2}

    monkeypatch.setattr(rg, "_http_json", fake_json)
    projects, errors = rg._modrinth_search("space", ())
    assert not errors
    assert [p["project_id"] for p in projects] == ["p1", "p2"]
    assert len(offsets) == len(projects)


def test_query_variants_are_not_top_n_sliced():
    variants = rg._query_variants("space.travel")
    assert variants == ("space.travel", "space travel", "space travel minecraft fabric mod source implementation")
''', encoding="utf-8")

print("patched exhaustive retrieval contract")
