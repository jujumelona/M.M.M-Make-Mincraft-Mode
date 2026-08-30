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


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _search_request_budget() -> int:
    # A network-work budget, not a result-count cutoff.
    return _env_int("MMM_GITHUB_SEARCH_REQUEST_BUDGET", 10, minimum=1, maximum=200)


def _search_page_size() -> int:
    return _env_int("MMM_GITHUB_SEARCH_PAGE_SIZE", 50, minimum=10, maximum=100)


def _source_request_budget() -> int:
    # Covers repository metadata/tree requests and raw-source reads for one query.
    return _env_int("MMM_GITHUB_SOURCE_REQUEST_BUDGET", 96, minimum=4, maximum=4096)


def _source_byte_budget() -> int:
    return _env_int(
        "MMM_GITHUB_SOURCE_BYTE_BUDGET",
        4 * 1024 * 1024,
        minimum=64 * 1024,
        maximum=256 * 1024 * 1024,
    )


def _output_byte_budget() -> int:
    return _env_int(
        "MMM_GITHUB_EVIDENCE_OUTPUT_BYTE_BUDGET",
        512 * 1024,
        minimum=32 * 1024,
        maximum=32 * 1024 * 1024,
    )


def _coverage_target() -> float:
    return _env_float(
        "MMM_GITHUB_EVIDENCE_COVERAGE_TARGET",
        0.75,
        minimum=0.0,
        maximum=1.0,
    )


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 2:
            owner, repo = str(value[0]).strip(), str(value[1]).strip().removesuffix(".git")
            if owner and repo:
                return owner, repo
    text = str(value or "").strip()
    if text.startswith("https://github.com/"):
        text = text[len("https://github.com/") :]
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


def discover_repositories(query: str, *, http_json: HttpJSON) -> RepositoryDiscovery:
    """Discover repositories by broadening until search saturation or request budget."""

    ladder = repository_query_ladder(query)
    budget = _search_request_budget()
    page_size = _search_page_size()
    repositories: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    errors: list[str] = []
    executed: list[str] = []
    requests = 0

    # Breadth first: page 1 of broader semantic variants is more useful for recall
    # than page 2..N of one over-specific phrase.  Additional pages are explored only
    # after every reachable variant has had a first chance.
    active: list[tuple[str, int]] = [(variant, 1) for variant in ladder]
    reason = "search_ladder_exhausted"
    while active and requests < budget:
        next_pages: list[tuple[str, int]] = []
        for variant, page in active:
            if requests >= budget:
                reason = "search_request_budget_exhausted"
                break
            requests += 1
            executed.append(variant if page == 1 else f"{variant} [page {page}]")
            try:
                payload = http_json(_github_search_url(variant, page=page, per_page=page_size))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"github_repository_search:{variant}:p{page}:{type(exc).__name__}: {exc}")
                continue
            items = payload.get("items", []) if isinstance(payload, Mapping) else []
            if not isinstance(items, list):
                errors.append(f"github_repository_search:{variant}:p{page}:invalid_items")
                continue
            new_on_page = 0
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                full_name = str(item.get("full_name") or "").strip()
                ref = _repo_tuple(full_name)
                if ref is None or ref in seen:
                    continue
                seen.add(ref)
                repositories.append(ref)
                new_on_page += 1
            # Search saturation is evidence based: a short page or a page that adds
            # no new repository ends pagination for that semantic variant.
            if len(items) >= page_size and new_on_page:
                next_pages.append((variant, page + 1))
        else:
            active = next_pages
            continue
        break

    if requests >= budget and active:
        reason = "search_request_budget_exhausted"
    elif not repositories and not errors:
        reason = "search_saturated_without_candidates"
    return RepositoryDiscovery(
        repositories=tuple(repositories),
        errors=tuple(errors),
        search_queries=tuple(executed),
        search_requests=requests,
        saturation_reason=reason,
    )


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
    """Inspect a repository adaptively with no fixed file-count truncation."""

    request_limit = _source_request_budget() if request_budget is None else max(0, request_budget)
    byte_limit = _source_byte_budget() if byte_budget is None else max(0, byte_budget)
    target = _coverage_target() if coverage_target is None else max(0.0, min(1.0, coverage_target))
    errors: list[str] = []
    requests = 0
    source_bytes = 0
    query_features = _feature_set(query)
    covered: set[str] = set()
    fetched: list[tuple[int, dict[str, Any]]] = []
    repo_api = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}"

    if request_limit < 2:
        return RepositoryEvidence(
            repository=(owner, repo),
            documents=(),
            errors=("github_repository_budget:insufficient_for_metadata_and_tree",),
            requests_used=0,
            source_bytes=0,
            covered_features=frozenset(),
            coverage_score=0.0,
            tree_truncated=False,
            saturation_reason="source_request_budget_exhausted",
        )

    try:
        requests += 1
        meta = http_json(repo_api)
    except Exception as exc:  # noqa: BLE001
        return RepositoryEvidence(
            repository=(owner, repo),
            documents=(),
            errors=(f"github_repo:{owner}/{repo}:{type(exc).__name__}: {exc}",),
            requests_used=requests,
            source_bytes=0,
            covered_features=frozenset(),
            coverage_score=0.0,
            tree_truncated=False,
            saturation_reason="repository_metadata_failed",
        )

    branch = str(meta.get("default_branch", "main") or "main") if isinstance(meta, Mapping) else "main"
    html_url = (
        str(meta.get("html_url") or f"https://github.com/{owner}/{repo}")
        if isinstance(meta, Mapping)
        else f"https://github.com/{owner}/{repo}"
    )
    license_value = None
    if isinstance(meta, Mapping) and isinstance(meta.get("license"), Mapping):
        license_value = meta["license"].get("spdx_id")

    try:
        requests += 1
        tree = http_json(f"{repo_api}/git/trees/{quote(branch, safe='')}?recursive=1")
    except Exception as exc:  # noqa: BLE001
        return RepositoryEvidence(
            repository=(owner, repo),
            documents=(),
            errors=(f"github_tree:{owner}/{repo}:{type(exc).__name__}: {exc}",),
            requests_used=requests,
            source_bytes=0,
            covered_features=frozenset(),
            coverage_score=0.0,
            tree_truncated=False,
            saturation_reason="repository_tree_failed",
        )

    tree_truncated = bool(tree.get("truncated")) if isinstance(tree, Mapping) else False
    if tree_truncated:
        errors.append(f"github_tree:{owner}/{repo}:recursive_tree_truncated")

    candidates: list[tuple[str, int | None]] = []
    raw_tree = tree.get("tree", []) if isinstance(tree, Mapping) else []
    for item in raw_tree if isinstance(raw_tree, list) else []:
        if not isinstance(item, Mapping) or item.get("type") != "blob":
            continue
        path = str(item.get("path") or "").strip()
        if not path.casefold().endswith(_SOURCE_SUFFIXES):
            continue
        raw_size = item.get("size")
        size = int(raw_size) if isinstance(raw_size, int) and raw_size >= 0 else None
        candidates.append((path, size))
    candidates.sort(key=lambda item: _path_score(item[0], query_features))

    reason = "repository_source_exhausted"
    for path, advertised_size in candidates:
        if requests >= request_limit:
            reason = "source_request_budget_exhausted"
            break
        if advertised_size is not None and source_bytes + advertised_size > byte_limit:
            # The byte budget is global work control; skip oversized candidates and
            # continue looking for smaller evidence instead of treating this as a hit cap.
            continue
        raw_url = (
            f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repo)}/"
            f"{quote(branch, safe='')}/{quote(path, safe='/')}"
        )
        try:
            requests += 1
            content = http_text(raw_url)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"github_raw:{owner}/{repo}/{path}:{type(exc).__name__}: {exc}")
            continue
        content_bytes = len(content.encode("utf-8", errors="replace"))
        if source_bytes + content_bytes > byte_limit:
            reason = "source_byte_budget_exhausted"
            break
        source_bytes += content_bytes
        score, newly_covered = _content_score(path, content, query_features)
        covered.update(newly_covered)
        role = _path_role(path)
        document = source_document(
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
        )
        fetched.append((score, document))

        # Stop on actual evidence sufficiency, never because a file ordinal was reached.
        if role == "code" and _coverage(query_features, covered) >= target:
            reason = "evidence_coverage_satisfied"
            break

    # Keep the highest-value evidence under a payload-byte budget.  This bounds prompt
    # pressure without throwing away a relevant file merely because it was the 5th/20th
    # file inspected.
    output_limit = _output_byte_budget()
    output_bytes = 0
    selected: list[dict[str, Any]] = []
    for _score, document in sorted(
        fetched,
        key=lambda item: (
            -item[0],
            0 if str(item[1].get("source_type")) == "github_source" else 1,
            str(item[1].get("source_id") or ""),
        ),
    ):
        size = len(str(document.get("content") or "").encode("utf-8", errors="replace"))
        if selected and output_bytes + size > output_limit:
            continue
        if not selected and size > output_limit:
            # The caller's source_document helper already applies its own single-source
            # cap.  Preserve the best document rather than returning metadata only.
            selected.append(document)
            output_bytes += size
            continue
        selected.append(document)
        output_bytes += size

    return RepositoryEvidence(
        repository=(owner, repo),
        documents=tuple(selected),
        errors=tuple(errors),
        requests_used=requests,
        source_bytes=source_bytes,
        covered_features=frozenset(covered),
        coverage_score=round(_coverage(query_features, covered), 4),
        tree_truncated=tree_truncated,
        saturation_reason=reason,
    )


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


def adaptive_github_evidence(
    query: str,
    *,
    http_json: HttpJSON,
    http_text: HttpText,
    source_document: SourceDocument,
    seed_repositories: Sequence[Any] = (),
    search_if_needed: bool = True,
) -> AdaptiveGitHubEvidence:
    """Resolve one query from GitHub under coverage/saturation/resource contracts."""

    query_features = _feature_set(query)
    target = _coverage_target()
    request_remaining = _source_request_budget()
    bytes_remaining = _source_byte_budget()
    documents: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    covered: set[str] = set()
    inspected: list[tuple[str, str]] = []
    seen_repos: set[tuple[str, str]] = set()
    source_requests = 0
    source_bytes = 0

    seeds: list[tuple[str, str]] = []
    for value in seed_repositories:
        ref = _repo_tuple(value)
        if ref is not None and ref not in seen_repos:
            seen_repos.add(ref)
            seeds.append(ref)

    discovery: RepositoryDiscovery | None = None
    candidate_queue = list(seeds)
    candidate_index = 0
    reason = "candidate_exhausted"

    while True:
        if candidate_index >= len(candidate_queue):
            current_coverage = _coverage(query_features, covered)
            if current_coverage >= target and documents:
                reason = "evidence_coverage_satisfied"
                break
            if discovery is None:
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

        if request_remaining < 2:
            reason = "source_request_budget_exhausted"
            break
        if bytes_remaining <= 0:
            reason = "source_byte_budget_exhausted"
            break

        owner, repo = candidate_queue[candidate_index]
        candidate_index += 1
        evidence = retrieve_repository_documents(
            owner,
            repo,
            query,
            http_json=http_json,
            http_text=http_text,
            source_document=source_document,
            request_budget=request_remaining,
            byte_budget=bytes_remaining,
            coverage_target=target,
        )
        inspected.append((owner, repo))
        request_remaining = max(0, request_remaining - evidence.requests_used)
        bytes_remaining = max(0, bytes_remaining - evidence.source_bytes)
        source_requests += evidence.requests_used
        source_bytes += evidence.source_bytes
        errors.extend(evidence.errors)
        covered.update(evidence.covered_features)
        for document in evidence.documents:
            source_id = str(document.get("source_id") or "")
            if source_id:
                documents.setdefault(source_id, dict(document))
        if _coverage(query_features, covered) >= target and documents:
            reason = "evidence_coverage_satisfied"
            break

    if discovery is None:
        discovery = RepositoryDiscovery((), (), (), 0, "search_not_needed")
    return AdaptiveGitHubEvidence(
        repositories=tuple(inspected),
        documents=tuple(documents.values()),
        errors=tuple(errors),
        search_queries=discovery.search_queries,
        search_requests=discovery.search_requests,
        source_requests=source_requests,
        source_bytes=source_bytes,
        coverage_score=round(_coverage(query_features, covered), 4),
        saturation_reason=reason,
    )


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
