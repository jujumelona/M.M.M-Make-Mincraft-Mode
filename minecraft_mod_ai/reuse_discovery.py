from __future__ import annotations

"""Adaptive, capability-level discovery for reusable Minecraft source donors.

Discovery is intentionally cheap-first: query catalogues/repository metadata in
parallel, resolve upstream source origins, deduplicate by immutable repository
identity, then pass only a small coverage-diverse representative set to the
expensive source-transplant inspector.
"""

import atexit
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import httpx

_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_MINECRAFT_GAME_ID = 432
_HTTP_CLIENT_LOCK = Lock()
_HTTP_CLIENT: httpx.Client | None = None


@dataclass
class _RepositoryEvidence:
    repository: str
    capabilities: set[str] = field(default_factory=set)
    providers: set[str] = field(default_factory=set)
    hits: int = 0
    rank_score: float = 0.0


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _workers() -> int:
    return _env_int("MMM_REUSE_PARALLELISM", 12, minimum=1, maximum=64)


def _query_limit() -> int:
    return _env_int("MMM_REUSE_DISCOVERY_LIMIT", 16, minimum=4, maximum=50)


def _representative_limit() -> int:
    return _env_int("MMM_REUSE_REPRESENTATIVE_LIMIT", 16, minimum=4, maximum=48)


def _query_variant_limit() -> int:
    return _env_int("MMM_REUSE_QUERY_VARIANTS", 3, minimum=1, maximum=6)


def _minimum_candidates() -> int:
    return _env_int("MMM_REUSE_MIN_CANDIDATES_PER_CAPABILITY", 3, minimum=1, maximum=12)


def _pooled_http_client() -> httpx.Client:
    """Return one thread-safe connection pool for auxiliary discovery APIs."""

    global _HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
            workers = _workers()
            _HTTP_CLIENT = httpx.Client(
                timeout=10.0,
                follow_redirects=False,
                headers={"User-Agent": "MMM-reuse-discovery/1"},
                limits=httpx.Limits(
                    max_connections=max(8, workers * 2),
                    max_keepalive_connections=max(4, workers),
                    keepalive_expiry=30.0,
                ),
            )
        return _HTTP_CLIENT


def _close_pooled_http_client() -> None:
    global _HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        client = _HTTP_CLIENT
        _HTTP_CLIENT = None
    if client is not None and not client.is_closed:
        client.close()


atexit.register(_close_pooled_http_client)


def _curseforge_api_key() -> str:
    """Return the optional host-owned CurseForge credential without exposing it."""

    key = (
        os.environ.get("MMM_CURSEFORGE_API_KEY", "").strip()
        or os.environ.get("CURSEFORGE_API_KEY", "").strip()
    )
    if not key:
        try:
            import google.colab.userdata as _colab_userdata  # type: ignore

            key = str(_colab_userdata.get("CURSEFORGE_API_KEY") or "").strip()
        except Exception:
            key = ""
    return key


def _graph_search_terms(
    capabilities: Sequence[str],
    capability_graph: Mapping[str, Any] | None,
) -> dict[str, tuple[str, ...]]:
    result = {capability: (capability.replace(".", " "),) for capability in capabilities}
    if not isinstance(capability_graph, Mapping):
        return result
    raw = capability_graph.get("search_terms")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return result
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        capability = str(item.get("capability") or "").strip()
        if capability not in result:
            continue
        terms = item.get("terms")
        if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
            continue
        values: list[str] = []
        seen: set[str] = set()
        for term in terms:
            text = " ".join(str(term or "").split())
            folded = text.casefold()
            if text and folded not in seen:
                values.append(text)
                seen.add(folded)
        if values:
            result[capability] = tuple(values)
    return result


def _query_variants(capability: str, terms: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    seen_values: set[str] = set()
    semantic = " ".join(_TOKEN.findall(capability.replace(".", " ").replace("-", " ")))
    for raw in [*terms, semantic]:
        text = " ".join(str(raw or "").split())
        folded = text.casefold()
        if text and folded not in seen_values:
            values.append(text[:512])
            seen_values.add(folded)
    expanded: list[str] = []
    expanded_seen: set[str] = set()
    for text in values:
        expanded.append(text)
        expanded_seen.add(text.casefold())
        if len(expanded) >= _query_variant_limit():
            break
    if len(expanded) < _query_variant_limit() and semantic:
        for suffix in ("system", "implementation source"):
            query = f"{semantic} {suffix}"
            folded = query.casefold()
            if folded not in expanded_seen:
                expanded.append(query)
                expanded_seen.add(folded)
            if len(expanded) >= _query_variant_limit():
                break
    return tuple(expanded or (capability,))


def _github_repository(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}", text):
        return text.removesuffix(".git")
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() != "github.com":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    owner, repo = parts[0], parts[1].removesuffix(".git")
    candidate = f"{owner}/{repo}"
    return candidate if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}", candidate) else ""


def _candidate_repository(candidate: Mapping[str, Any]) -> str:
    return (
        _github_repository(candidate.get("repository"))
        or _github_repository(candidate.get("source_url"))
        or _github_repository(candidate.get("url"))
        or _github_repository(candidate.get("title"))
    )


def _resolve_modrinth_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[tuple[str, float]]:
    jobs: list[tuple[int, str]] = []
    direct: list[tuple[str, float]] = []
    for index, candidate in enumerate(candidates):
        repository = _candidate_repository(candidate)
        if repository:
            direct.append((repository, 1.0 / (1 + index)))
            continue
        api_url = str(candidate.get("api_url") or "").strip()
        if api_url.startswith("https://api.modrinth.com/v2/project/"):
            jobs.append((index, api_url))
    if not jobs:
        return direct

    client = _pooled_http_client()

    def resolve(job: tuple[int, str]) -> tuple[str, float]:
        index, url = job
        try:
            response = client.get(url, timeout=8.0)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, Mapping):
                return "", 0.0
            return _github_repository(value.get("source_url")), 1.0 / (1 + index)
        except Exception:
            return "", 0.0

    with ThreadPoolExecutor(max_workers=min(_workers(), len(jobs))) as pool:
        futures = [pool.submit(resolve, job) for job in jobs]
        for future in as_completed(futures):
            repository, score = future.result()
            if repository:
                direct.append((repository, score))
    return direct


def _ascii_search_query(value: str) -> str:
    """Project arbitrary Unicode text onto the ASCII-only CurseForge query field.

    This is intentionally script-neutral: Unicode normalization may preserve
    compatible Latin characters, while unsupported scripts fall back at the
    provider boundary instead of invoking a language-specific transliterator.
    """

    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9\s_-]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


def _search_curseforge(query: str, *, limit: int) -> list[tuple[str, float]]:
    api_key = _curseforge_api_key()
    if not api_key:
        return []
    clean_query = _ascii_search_query(query)
    if not clean_query or len(clean_query) < 2:
        clean_query = "minecraft mod"
    params = {
        "gameId": str(_MINECRAFT_GAME_ID),
        "searchFilter": clean_query[:80],
        "sortField": "2",
        "sortOrder": "desc",
        "pageSize": str(min(limit, 50)),
        "index": "0",
    }
    try:
        response = _pooled_http_client().get(
            "https://api.curseforge.com/v1/mods/search",
            params=params,
            headers={"Accept": "application/json", "x-api-key": api_key},
            timeout=10.0,
        )
        response.raise_for_status()
        raw = response.json()
    except Exception:
        return []
    data = raw.get("data") if isinstance(raw, Mapping) else None
    if not isinstance(data, list):
        return []
    result: list[tuple[str, float]] = []
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            continue
        links = item.get("links")
        source_url = links.get("sourceUrl") if isinstance(links, Mapping) else ""
        repository = _github_repository(source_url)
        if repository:
            result.append((repository, 1.0 / (1 + index)))
    return result


def discover_repositories_for_graph(
    capabilities: Sequence[str],
    client: Any,
    *,
    capability_graph: Mapping[str, Any] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Adaptive two-wave discovery with cross-capability representative selection."""

    ordered = tuple(dict.fromkeys(str(item).strip() for item in capabilities if str(item).strip()))
    result: dict[str, tuple[str, ...]] = {capability: () for capability in ordered}
    if not ordered:
        return result
    terms = _graph_search_terms(ordered, capability_graph)
    variants = {capability: _query_variants(capability, terms[capability]) for capability in ordered}
    evidence: dict[str, _RepositoryEvidence] = {}
    per_capability: dict[str, dict[str, float]] = {capability: {} for capability in ordered}

    def register(capability: str, provider: str, values: Sequence[tuple[str, float]]) -> None:
        for repository, rank in values:
            repository = _github_repository(repository)
            if not repository:
                continue
            cap_scores = per_capability[capability]
            cap_scores[repository] = max(cap_scores.get(repository, 0.0), rank)
            item = evidence.setdefault(repository, _RepositoryEvidence(repository=repository))
            item.capabilities.add(capability)
            item.providers.add(provider)
            item.hits += 1
            item.rank_score += rank

    def provider_search(capability: str, provider: str, query: str) -> tuple[str, str, list[tuple[str, float]]]:
        if provider == "curseforge":
            return capability, provider, _search_curseforge(query, limit=_query_limit())
        try:
            page = client.search(provider, query, limit=_query_limit(), target_profile="minecraft_mod")
        except Exception:
            return capability, provider, []
        raw = page.get("candidates") if isinstance(page, Mapping) else None
        candidates = [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []
        if provider == "modrinth":
            return capability, provider, _resolve_modrinth_candidates(candidates)
        values: list[tuple[str, float]] = []
        for index, candidate in enumerate(candidates):
            repository = _candidate_repository(candidate)
            if repository:
                values.append((repository, 1.0 / (1 + index)))
        return capability, provider, values

    providers = ["github", "modrinth"]
    if _curseforge_api_key():
        providers.append("curseforge")

    def run_wave(capability_set: Sequence[str], variant_index: int) -> None:
        jobs: list[tuple[str, str, str]] = []
        for capability in capability_set:
            query_list = variants[capability]
            if variant_index >= len(query_list):
                continue
            for provider in providers:
                jobs.append((capability, provider, query_list[variant_index]))
        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=min(_workers(), len(jobs)), thread_name_prefix="mmm-reuse-catalog") as pool:
            futures = [pool.submit(provider_search, *job) for job in jobs]
            for future in as_completed(futures):
                try:
                    capability, provider, values = future.result()
                except Exception:
                    continue
                register(capability, provider, values)

    run_wave(ordered, 0)
    for variant_index in range(1, _query_variant_limit()):
        uncovered = [
            capability
            for capability in ordered
            if len(per_capability[capability]) < _minimum_candidates()
        ]
        if not uncovered:
            break
        run_wave(uncovered, variant_index)

    selected = _representative_repositories(evidence, ordered)
    selected_set = set(selected)
    for capability in ordered:
        ranked = sorted(
            per_capability[capability],
            key=lambda repo: (
                repo in selected_set,
                len(evidence[repo].capabilities),
                len(evidence[repo].providers),
                per_capability[capability][repo],
                evidence[repo].rank_score,
                repo,
            ),
            reverse=True,
        )
        selected_for_capability = [repo for repo in ranked if repo in selected_set]
        if not selected_for_capability and ranked:
            selected_for_capability.append(ranked[0])
        result[capability] = tuple(selected_for_capability)
    return result


def _representative_repositories(
    evidence: Mapping[str, _RepositoryEvidence],
    capabilities: Sequence[str],
) -> tuple[str, ...]:
    """Greedy set-cover/facility-location approximation over source repositories."""

    remaining = set(capabilities)
    available = set(evidence)
    selected: list[str] = []
    limit = min(_representative_limit(), len(available))
    minimum = min(6, limit)
    while available and len(selected) < limit:
        best = max(
            available,
            key=lambda repo: (
                100.0 * len(evidence[repo].capabilities & remaining)
                + 10.0 * len(evidence[repo].capabilities)
                + 3.0 * len(evidence[repo].providers)
                + evidence[repo].rank_score,
                evidence[repo].hits,
                repo,
            ),
        )
        new_coverage = evidence[best].capabilities & remaining
        if not new_coverage and len(selected) >= minimum:
            break
        selected.append(best)
        available.remove(best)
        remaining -= new_coverage
        if not remaining and len(selected) >= minimum:
            break
    return tuple(selected)


__all__ = ["discover_repositories_for_graph"]
