from __future__ import annotations

"""Adaptive GitHub repository and source retrieval.

GitHub repository search is a candidate generator, not an evidence oracle.  The
retriever therefore separates high-recall repository discovery from query-specific
source inspection.  Repository and file cardinality are never truncated by a
"top N files/repos" rule.  Work stops only when evidence coverage is sufficient,
GitHub search saturates, or explicit host resource budgets are exhausted.

The module is transport-agnostic on purpose.  Callers inject the reviewed HTTP and
source-document helpers they already own, which keeps authentication, caching and
provenance policy in one place.
"""

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode

_TOKEN = re.compile(r"[A-Za-z0-9]+", re.UNICODE)
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


def _lexemes(value: str) -> tuple[str, ...]:
    """Split path/source identifiers before lexical retrieval scoring."""

    expanded = _CAMEL_BOUNDARY.sub(" ", str(value or ""))
    return tuple(_TOKEN.findall(expanded))

# Terms that describe the retrieval task rather than the implementation being sought.
# Removing these is what prevents an obligation sentence from becoming an AND-heavy
# GitHub repository query.
_QUERY_BOILERPLATE = frozenset(
    {
        "api",
        "behavior",
        "closure",
        "code",
        "compatibility",
        "dependency",
        "evidence",
        "fabric",
        "game",
        "gametest",
        "github",
        "implementation",
        "license",
        "loader",
        "mapping",
        "mappings",
        "mechanic",
        "minecraft",
        "mod",
        "mods",
        "player",
        "players",
        "provenance",
        "reusable",
        "source",
        "target",
        "testing",
        "validation",
        "world",
        "can",
        "the",
        "and",
        "for",
        "from",
        "with",
        "into",
        "that",
        "this",
    }
)

_SOURCE_SUFFIXES = (
    ".java",
    ".kt",
    ".kts",
    ".json",
    ".mcfunction",
    ".gradle",
    ".properties",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
)
_CODE_SUFFIXES = (".java", ".kt", ".kts", ".mcfunction")
_BUILD_NAMES = frozenset(
    {
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradle.properties",
        "fabric.mod.json",
        "mods.toml",
        "neoforge.mods.toml",
    }
)

HttpJSON = Callable[[str], Any]
HttpText = Callable[[str], str]
SourceDocument = Callable[..., dict[str, Any]]


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))
















def _dedupe_text(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            out.append(text)
    return tuple(out)


def _token_root(token: str) -> str:
    """Small deterministic morphology normalizer used only for retrieval recall."""

    value = token.casefold().strip()
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    for suffix in ("ization", "isation", "ations", "ation", "ments", "ment", "ing", "ers", "er", "ed"):
        if len(value) - len(suffix) >= 4 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def semantic_terms(query: str) -> tuple[str, ...]:
    """Return implementation-bearing query terms while dropping retrieval boilerplate."""

    values: list[str] = []
    seen: set[str] = set()
    normalized = re.sub(r"[._:/\\-]+", " ", str(query or ""))
    for raw in _lexemes(normalized):
        token = raw.casefold()
        if len(token) < 2 or token in _QUERY_BOILERPLATE:
            continue
        root = _token_root(token)
        if root in _QUERY_BOILERPLATE or root in seen:
            continue
        seen.add(root)
        values.append(token)
    return tuple(values)


def _feature_set(text: str) -> set[str]:
    features: set[str] = set()
    for raw in _lexemes(str(text or "")):
        token = raw.casefold()
        if len(token) < 2 or token in _QUERY_BOILERPLATE:
            continue
        features.add(_token_root(token))
    return features


def repository_query_ladder(query: str) -> tuple[str, ...]:
    """Produce progressively broader repository queries without a variant-count cap.

    GitHub repository search is AND-heavy.  A long obligation sentence is therefore
    the *worst* possible search query.  The ladder starts with the semantic phrase,
    then immediately probes each semantic anchor in Minecraft/Fabric contexts and as
    a plain repository term.  The caller's request budget decides how far the ladder
    is explored; this function never truncates it to an arbitrary top-N variant list.
    """

    terms = semantic_terms(query)
    if not terms:
        return ("minecraft fabric",)

    variants: list[str] = [" ".join(terms)]
    for term in terms:
        variants.append(f"{term} minecraft")
    for term in terms:
        variants.append(f"{term} fabric")
    variants.extend(terms)
    return _dedupe_text(variants)


def _github_search_url(query: str, *, page: int, per_page: int) -> str:
    params = urlencode(
        {
            "q": f"{query} in:name,description,readme fork:false archived:false",
            # Omitting sort preserves GitHub's best-match ranking instead of allowing
            # stars to promote popular but semantically unrelated repositories.
            "page": page,
            "per_page": per_page,
        }
    )
    return f"https://api.github.com/search/repositories?{params}"


def _repo_tuple(value: Any) -> tuple[str, str] | None:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) >= 2
    ):
        owner, repo = str(value[0]).strip(), str(value[1]).strip().removesuffix(".git")
        if owner and repo:
            return owner, repo
    text = str(value or "").strip()
    text = text.removeprefix("https://github.com/")
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        owner, repo = parts[0], parts[1].removesuffix(".git")
        if owner and repo:
            return owner, repo
    return None


@dataclass(frozen=True)
class RepositoryDiscovery:
    repositories: tuple[tuple[str, str], ...]
    errors: tuple[str, ...]
    search_queries: tuple[str, ...]
    search_requests: int
    saturation_reason: str




def _path_role(path: str) -> str:
    folded = f"/{path.casefold()}"
    leaf = PurePosixPath(path).name.casefold()
    if leaf in _BUILD_NAMES:
        return "build_dependency"
    if "/src/test/" in folded or "/src/gametest/" in folded or "/test/" in folded:
        return "test"
    if "/src/main/resources/" in folded or "/resources/" in folded:
        return "resource"
    if path.casefold().endswith(_CODE_SUFFIXES):
        return "code"
    return "support"


def _path_score(path: str, query_features: set[str]) -> tuple[int, int, int, str]:
    role = _path_role(path)
    role_priority = {
        "code": 0,
        "test": 1,
        "build_dependency": 2,
        "resource": 3,
        "support": 4,
    }[role]
    path_features = _feature_set(path.replace("/", " "))
    overlap = len(query_features & path_features)
    source_root = 1 if "/src/main/" in f"/{path.casefold()}" else 0
    return (-overlap, role_priority, -source_root, path)


def _content_score(path: str, content: str, query_features: set[str]) -> tuple[int, set[str]]:
    observed = _feature_set(path + "\n" + content)
    covered = query_features & observed
    role = _path_role(path)
    role_bonus = 4 if role == "code" else 2 if role == "test" else 1
    return 10 * len(covered) + role_bonus, covered


def _coverage(query_features: set[str], covered: set[str]) -> float:
    if not query_features:
        return 1.0
    return len(query_features & covered) / len(query_features)


@dataclass(frozen=True)
class RepositoryEvidence:
    repository: tuple[str, str]
    documents: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    requests_used: int
    source_bytes: int
    covered_features: frozenset[str]
    coverage_score: float
    tree_truncated: bool
    saturation_reason: str




@dataclass(frozen=True)
class AdaptiveGitHubEvidence:
    repositories: tuple[tuple[str, str], ...]
    documents: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    search_queries: tuple[str, ...]
    search_requests: int
    source_requests: int
    source_bytes: int
    coverage_score: float
    saturation_reason: str




__all__ = [
    "AdaptiveGitHubEvidence",
    "RepositoryDiscovery",
    "RepositoryEvidence",
    "adaptive_github_evidence",
    "discover_repositories",
    "repository_query_ladder",
    "retrieve_repository_documents",
    "semantic_terms",
]

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
