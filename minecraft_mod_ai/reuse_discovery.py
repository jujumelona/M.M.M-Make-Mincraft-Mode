from __future__ import annotations

"""Adaptive, capability-level discovery for reusable Minecraft source donors.

Discovery queries CurseForge first, then Modrinth and GitHub, with independent
queries parallel within each provider stage. Upstream source identities are
deduplicated and representatives inspected first; every fallback candidate stays
available until source verification and reuse proof decide coverage.
"""

import atexit
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import httpx

from .canonical_capability_ontology import (
    atomic_capability_definitions,
    resolve_capabilities_from_phrase_structured,
    search_queries_for_capability,
)
from .curseforge_catalog import (
    curseforge_api_key as _curseforge_api_key,
)
from .curseforge_catalog import (
    curseforge_mod_metadata,
    curseforge_mods_class_id,
    curseforge_next_index,
)
from .deadline_executor import iter_completed_with_deadlines
from .ecosystem_discovery import provider_request_failure
from .source_intent_aliases import source_intent_aliases as _source_intent_aliases

_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_MINECRAFT_GAME_ID = 432
_PROVIDER_PRIORITY = {"curseforge": 0, "modrinth": 1, "sourcegraph": 1, "github": 2}
_HTTP_CLIENT_LOCK = Lock()
_HTTP_CLIENT: httpx.Client | None = None
_GITHUB_QUERY_BOILERPLATE = frozenset(
    {"minecraft", "mod", "mods", "custom", "implementation", "source", "fabric", "forge", "neoforge"}
)
_QUERY_ACTION_WORDS = frozenset({
    "gameplay", "feature", "system", "entity", "generation", "construction",
    "assembly", "crafting", "building", "create", "creation", "implement",
    "travel", "exploration", "recruitment", "upgrade", "upgrades",
})
_COMPOUND_SHIP_NOUNS = frozenset({
    "spaceship", "spacecraft", "airship", "starship", "warship", "battleship",
    "dropship", "shipyard",
})


def _singular_source_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _source_words(text: str) -> set[str]:
    """Match source vocabulary, including plurals and compound ship nouns."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).replace("_", " ")
    words = set(_TOKEN.findall(text.casefold()))
    roots = {_singular_source_word(word) for word in words}
    return words | roots | ({"ship"} if roots & _COMPOUND_SHIP_NOUNS else set())


def _short_source_queries(capability: str) -> tuple[str, ...]:
    # Narrow AND queries often miss repositories whose descriptions name the
    # underlying object rather than the whole requested mechanic. Broaden only
    # retrieval wording; the original feature and its source proof stay intact.
    semantic = capability.replace(".", " ").replace("_", " ").replace("-", " ")
    words = [word for word in _TOKEN.findall(semantic.casefold())
             if word not in _GITHUB_QUERY_BOILERPLATE | _QUERY_ACTION_WORDS]
    if len(_TOKEN.findall(semantic)) < 2 or not words:
        return ()
    noun = _singular_source_word(words[0])
    return (noun, "ship") if noun in _COMPOUND_SHIP_NOUNS else (noun,)


@dataclass
class _RepositoryEvidence:
    repository: str
    capabilities: set[str] = field(default_factory=set)
    providers: set[str] = field(default_factory=set)
    hits: int = 0
    rank_score: float = 0.0

    @property
    def provider_priority(self) -> int:
        return min((_PROVIDER_PRIORITY[provider] for provider in self.providers), default=2)


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


def _discovery_pages() -> int:
    return _env_int("MMM_REUSE_DISCOVERY_PAGES", 2, minimum=1, maximum=4)


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


def _append_query(values: list[str], seen: set[str], raw: Any) -> None:
    text = " ".join(str(raw or "").split())[:512]
    folded = text.casefold()
    if text and folded not in seen:
        values.append(text)
        seen.add(folded)


def _ontology_search_queries(capability: str) -> tuple[str, ...]:
    """Return retrieval phrasing without changing the approved capability graph.

    The authoritative planner owns which capabilities exist. This helper only
    translates one already-approved capability into catalogue-friendly queries.
    Canonical ontology matches are therefore safe retrieval aliases, not scope
    expansion or dependency invention.
    """

    values: list[str] = []
    seen: set[str] = set()
    exact_canonical = capability in atomic_capability_definitions()
    if exact_canonical:
        for query in search_queries_for_capability(capability):
            _append_query(values, seen, query)

    semantic = " ".join(
        _TOKEN.findall(
            capability.replace(".", " ")
            .replace("_", " ")
            .replace("-", " ")
            .replace(":", " ")
        )
    )
    try:
        resolution = resolve_capabilities_from_phrase_structured(semantic)
    except Exception:
        resolution = None
    if resolution is not None:
        for node in resolution.nodes:
            if node.origin not in {"explicit", "archetype_inferred"}:
                continue
            if str(node.capability_id).startswith(("unresolved:", "provisional:")):
                continue
            for query in search_queries_for_capability(node.capability_id):
                _append_query(values, seen, query)

    if not exact_canonical:
        for query in search_queries_for_capability(capability):
            _append_query(values, seen, query)
    return tuple(values)


def _specific_search_terms(terms: Sequence[str]) -> tuple[str, ...]:
    normalized = (" ".join(str(raw or "").split()) for raw in terms)
    return tuple(query for query in normalized if 2 <= len(query) <= 90)


def _ordered_source_queries(capability: str, terms: Sequence[str]) -> tuple[str, ...]:
    specific = _specific_search_terms(terms)
    bilingual = _source_intent_aliases(capability)
    aliases = _ontology_search_queries(capability)
    canonical = capability in atomic_capability_definitions()
    authored = tuple(query for query in specific if len(_TOKEN.findall(query)) <= 8)
    head = aliases[:1] if canonical else (
        bilingual if re.search(r"[가-힣]", capability) and bilingual else authored
    )
    compact = _short_source_queries(capability) if head and not bilingual else ()
    ordinary = () if capability.startswith("custom gameplay ") else specific
    semantic = " ".join(_TOKEN.findall(
        capability.replace(".", " ").replace("_", " ").replace("-", " ").replace(":", " ")))
    fallback = (f"minecraft {semantic} mod", f"{semantic} fabric mod") if semantic and len(semantic) <= 90 else ()
    return (*head, *compact, *(query for query in aliases if len(query) <= 90), *ordinary, *fallback)


def _query_variants(capability: str, terms: Sequence[str]) -> tuple[str, ...]:
    """Prioritize the authored phrase, source nouns and curated retrieval aliases."""
    values: list[str] = []
    seen: set[str] = set()
    for query in _ordered_source_queries(capability, terms):
        _append_query(values, seen, query)
    return tuple(values[:_query_variant_limit()] or ("minecraft mod",))


def _provider_query(provider: str, query: str) -> str:
    """Adapt one semantic query to the provider's search semantics.

    GitHub/Sourcegraph are code or repository discovery rather than catalogues.
    The GitHub client appends ``minecraft mod``. Passing phrases that already
    contain those words over-constrains queries. Keep at most two semantic
    terms for code search and GitHub; Modrinth retains its own full-text search.
    """

    query = re.sub(r'\bminecraft\s+(?:fabric|forge|neoforge)\b', 'minecraft', query, flags=re.IGNORECASE)
    query = re.sub(r'\b(?:fabric|forge|neoforge)\s+mods?\b', 'mod', query, flags=re.IGNORECASE)
    if provider not in {"github", "sourcegraph"}:
        return query
    tokens = [
        token
        for token in _TOKEN.findall(query.casefold())
        if token not in _GITHUB_QUERY_BOILERPLATE
    ]
    if not tokens:
        return query
    tokens = tokens[:2]
    return " ".join(tokens)


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


def _github_candidate_is_minecraft_mod(candidate: Mapping[str, Any]) -> bool:
    """Reject broad GitHub keyword hits before spending donor-inspection work.

    GitHub searches READMEs as well as repository names. A general Rust,
    security, or agent index can therefore match `minecraft mod` in its README
    without providing any mod sources. Accept repository-level Minecraft and
    modding signals, not a search-result position or a README-only match.
    Modrinth/CurseForge already scope results to Minecraft projects.
    """
    repository = _candidate_repository(candidate).casefold()
    name = repository.rsplit("/", 1)[-1]
    if name.startswith(("awesome-", "list-of-", "awesome_")):
        return False
    metadata = candidate.get("metadata")
    topics = metadata.get("topics", ()) if isinstance(metadata, Mapping) else ()
    description = (
        str(candidate.get("title") or "") + " "
        + str(candidate.get("summary") or "") + " "
        + " ".join(str(value) for value in topics) + " "
        + str(candidate.get("topics") or "")
    ).casefold()
    if not re.search(r"\b(minecraft|fabricmc|neoforge|quiltmc|mc[-_]?mod)\b", description):
        return False
    return bool(re.search(
        r"\b(mods?|modding|addons?|add-ons?|fabric|forge|quilt|"
        r"resource[ -]packs?|texture[ -]packs?|datapacks?|data[ -]packs?)\b",
        description,
    ))


def _candidate_repository(candidate: Mapping[str, Any]) -> str:
    return (
        _github_repository(candidate.get("repository"))
        or _github_repository(candidate.get("source_url"))
        or _github_repository(candidate.get("url"))
        or _github_repository(candidate.get("title"))
    )


def _candidate_relevance(candidate: Mapping[str, Any], capability: str) -> float:
    """Rank actual feature evidence, rather than crediting a hit with its query."""
    stop = _GITHUB_QUERY_BOILERPLATE | {
        "fabric", "forge", "neoforge", "quilt", "system", "feature", "generation",
        "the", "and", "for", "with", "a", "an", "to", "of",
    }
    terms = (_source_words(capability) | {
        word for alias in _source_intent_aliases(capability) for word in _source_words(alias)
    }) - stop
    if capability in atomic_capability_definitions():
        terms.update(word for query in _ontology_search_queries(capability)
                     for word in _source_words(query) if word not in stop)
    metadata = candidate.get("metadata")
    topics = metadata.get("topics", ()) if isinstance(metadata, Mapping) else ()
    text = " ".join(str(candidate.get(key) or "") for key in (
        "title", "summary", "description", "body", "repository", "source_url", "topics",
    )) + " " + " ".join(str(value) for value in topics)
    matched = terms & _source_words(text)
    return len(matched) / max(1, len(terms))


def _discovery_provider_error(provider: str, exc: BaseException, *, stage: str) -> dict[str, Any]:
    """Retain typed transport facts without exception prose or response bodies."""

    failure = provider_request_failure(exc, provider=provider)
    status = failure.status_code if failure is not None else getattr(exc, "http_status", None)
    status = status if isinstance(status, int) else None
    unavailable = bool(failure is not None and failure.backend_unavailable)
    unavailable = unavailable or provider == "curseforge" and status in {401, 403, 429}
    return {
        "provider": provider,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "http_status": status,
        "backend_unavailable": unavailable,
        "request_failure": failure.to_dict() if failure is not None else None,
        "error": (
            f"{provider} {stage} failed ({type(exc).__name__}"
            + (f", HTTP {status}" if status is not None else "") + ")."
        ),
    }


def _resolve_modrinth_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    with_metadata=False,
    errors: list[dict[str, Any]] | None = None,
) -> list:
    jobs: list[tuple[int, str]] = []
    direct = []
    for index, candidate in enumerate(candidates):
        repository = _candidate_repository(candidate)
        if repository:
            direct.append((repository, 1.0 / (1 + index), candidate))
            continue
        api_url = str(candidate.get("api_url") or "").strip()
        if api_url.startswith("https://api.modrinth.com/v2/project/"):
            jobs.append((index, api_url))
    if not jobs:
        return direct if with_metadata else [(repo, rank) for repo, rank, _meta in direct]

    client = _pooled_http_client()

    def resolve(job: tuple[int, str]):
        index, url = job
        try:
            response = client.get(url, timeout=8.0)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, Mapping):
                raise TypeError("Modrinth project detail was not an object.")
            return _github_repository(value.get("source_url")), 1.0 / (1 + index), {
                **candidates[index], "description": value.get("description", ""),
                "body": str(value.get("body") or "")[:6000],
            }, None
        except Exception as exc:  # noqa: BLE001 - preserve one failed detail alongside successful sources.
            return "", 0.0, {}, _discovery_provider_error("modrinth", exc, stage="source_resolution")

    for _job, result in iter_completed_with_deadlines(
        jobs,
        resolve,
        max_workers=min(_workers(), len(jobs)),
        stage="reuse-modrinth-source-resolution",
        sort_key=lambda item: item[0],
    ):
        repository, score, metadata, error = result
        if error is not None and errors is not None:
            errors.append(error)
        if repository:
            direct.append((repository, score, metadata))
    return direct if with_metadata else [(repo, rank) for repo, rank, _meta in direct]


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


class _CurseForgeSearchError(RuntimeError):
    """Provider failure containing no credential or response-body text."""

    def __init__(
        self, message: str, *, http_status: int | None = None, request_failure: Any = None
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.request_failure = request_failure


def _search_curseforge(
    query: str,
    *,
    limit: int,
    with_metadata=False,
    minecraft_version=None,
    loader=None,
) -> list:
    """Browse real Minecraft mod listings, not invented feature-name queries.

    Version/loader constraints are verified against pinned source during donor
    inspection; applying them here would lose useful porting references.
    Catalogue results without source links never acquire reuse authority.
    """
    api_key = _curseforge_api_key()
    if not api_key:
        return []
    client = _pooled_http_client()
    headers = {"Accept": "application/json", "x-api-key": api_key}
    try:
        category_response = client.get(
            "https://api.curseforge.com/v1/categories",
            params={"gameId": str(_MINECRAFT_GAME_ID)},
            headers=headers, timeout=10.0,
        )
        category_response.raise_for_status()
        category_payload = category_response.json()
        category_rows = category_payload.get("data") if isinstance(category_payload, Mapping) else None
        mods_class_id = curseforge_mods_class_id(category_rows)
    except (httpx.HTTPError, ValueError):
        # The search endpoint may remain functional even if categories do not.
        # Results without a known Mods class must not masquerade as exhaustive.
        mods_class_id = None

    result = []
    seen_ids: set[int] = set()
    page_size = 50  # Each browse page can expose candidates beyond a single shortlist.
    for sort_field in (2, 3, 11):  # popularity, updated, released
        index = 0
        for _ in range(_discovery_pages()):
            params = {
                "gameId": str(_MINECRAFT_GAME_ID),
                "sortField": str(sort_field),
                "sortOrder": "desc",
                "pageSize": str(page_size),
                "index": str(index),
            }
            if mods_class_id is not None:
                params["classId"] = str(mods_class_id)
            try:
                response = client.get(
                    "https://api.curseforge.com/v1/mods/search",
                    params=params, headers=headers, timeout=10.0,
                )
                response.raise_for_status()
                raw = response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                raise _CurseForgeSearchError(
                    f"CurseForge search failed (HTTP {status}).", http_status=status,
                    request_failure=provider_request_failure(exc, provider="curseforge"),
                ) from None
            except httpx.HTTPError as exc:
                raise _CurseForgeSearchError(
                    f"CurseForge search transport failed ({type(exc).__name__}).",
                    request_failure=provider_request_failure(exc, provider="curseforge"),
                ) from None
            except ValueError:
                raise _CurseForgeSearchError("CurseForge search returned invalid JSON.") from None

            rows = raw.get("data") if isinstance(raw, Mapping) else None
            if not isinstance(rows, list):
                raise _CurseForgeSearchError("CurseForge search returned no valid data list.")
            for position, item in enumerate(rows):
                if not isinstance(item, Mapping) or type(item.get("id")) is not int:
                    continue
                mod_id = item["id"]
                if mod_id in seen_ids:
                    continue
                if mods_class_id is not None and item.get("classId") not in (None, mods_class_id):
                    continue
                seen_ids.add(mod_id)
                links = item.get("links")
                source_url = links.get("sourceUrl") if isinstance(links, Mapping) else ""
                repository = _github_repository(source_url)
                result.append((repository, 1.0 / (1 + len(result)), {
                    "title": item.get("name", ""), "summary": item.get("summary", ""),
                    "repository": repository, "sort_field": sort_field,
                    "catalog_class_verified": mods_class_id is not None,
                    **curseforge_mod_metadata(
                        item, minecraft_version=minecraft_version, loader=loader
                    ),
                }))
            try:
                next_index = curseforge_next_index(
                    raw.get("pagination") if isinstance(raw, Mapping) else None,
                    index=index, page_size=page_size, row_count=len(rows),
                )
            except ValueError as exc:
                raise _CurseForgeSearchError(str(exc)) from None
            if next_index is None:
                break
            index = next_index
    return result if with_metadata else [(repo, rank) for repo, rank, _meta in result if repo]


def _search_sourcegraph(query: str, *, limit: int, with_metadata=False) -> list:
    """Find actual Java code in public GitHub projects using Sourcegraph's SSE API.

    A code hit is an untrusted candidate, not a proof of Minecraft feature
    compatibility. Missing public service access is a recoverable provider error.
    """
    tokens = [
        word for word in _TOKEN.findall(query.casefold())
        if re.fullmatch(r"[a-z][a-z0-9_]{3,40}", word)
        and word not in _GITHUB_QUERY_BOILERPLATE | _QUERY_ACTION_WORDS
    ]
    if not tokens:
        return []
    term = tokens[0]
    search = f'lang:Java type:file "net.minecraft" "{term}" count:{min(50, max(1, limit))} patternType:keyword'
    try:
        response = _pooled_http_client().get(
            "https://sourcegraph.com/.api/search/stream",
            params={"q": search, "v": "V3"},
            headers={"Accept": "text/event-stream"},
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Sourcegraph code search unavailable ({type(exc).__name__})."
        ) from None
    results = []
    seen: set[str] = set()
    completed = False
    for event in re.split(r"\r?\n\r?\n", response.text):
        event_type = ""
        fragments = []
        for line in event.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                fragments.append(line[5:].strip())
        if event_type == "done":
            completed = True
        if event_type != "matches" or not fragments:
            continue
        try:
            entries = __import__("json").loads("\n".join(fragments))
        except ValueError:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("type") != "content":
                continue
            source = str(entry.get("repository") or "")
            repository = _github_repository("https://" + source)
            path = str(entry.get("path") or "")
            if not repository or not path.endswith(".java") or repository in seen:
                continue
            seen.add(repository)
            matches = entry.get("lineMatches")
            snippets = [
                str(match.get("line") or "")[:300]
                for match in matches if isinstance(match, Mapping)
            ][:3] if isinstance(matches, list) else []
            metadata = {
                "repository": repository,
                "title": path.rsplit("/", 1)[-1],
                "summary": " ".join(snippets),
                "source_url": "https://github.com/" + repository,
                "source_id": "github:" + repository,
                "code_search_path": path,
                "code_search_query": term,
                "source_reuse_authority": "verification_required",
            }
            results.append((repository, 1.0 / (1 + len(results)), metadata))
    if not completed:
        raise RuntimeError("Sourcegraph returned an incomplete search stream.")
    return results if with_metadata else [(repo, rank) for repo, rank, _meta in results]


def discover_repositories_for_graph(
    capabilities: Sequence[str],
    client: Any,
    *,
    capability_graph: Mapping[str, Any] | None = None,
    minecraft_version: str | None = None,
    loader: str | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    providers: Sequence[str] | None = None,
    query_cache: dict[tuple[str, str, str, str], Any] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Discover requested provider stages without treating keyword hits as reuse.

    A caller may request one provider at a time and inspect its sources before
    paying for fallback discovery. The default retains the complete provider
    sequence. A caller-owned query cache preserves results and provider failures
    across lazy stages. Provider errors remain distinct from successful zero hits.
    """

    ordered = tuple(dict.fromkeys(str(item).strip() for item in capabilities if str(item).strip()))
    result: dict[str, tuple[str, ...]] = {capability: () for capability in ordered}
    if not ordered:
        return result
    terms = _graph_search_terms(ordered, capability_graph)
    variants = {capability: _query_variants(capability, terms[capability]) for capability in ordered}
    evidence: dict[str, _RepositoryEvidence] = {}
    per_capability: dict[str, dict[str, float]] = {capability: {} for capability in ordered}
    per_capability_priority: dict[str, dict[str, int]] = {capability: {} for capability in ordered}

    def register(capability: str, provider: str, values: Sequence) -> None:
        for repository, rank, metadata in values:
            relevance = _candidate_relevance(metadata, capability)
            # Search relevance orders inspection; a short catalogue description
            # cannot veto the actual code before it has been inspected.
            rank += 2 * relevance
            repository = _github_repository(repository)
            if not repository:
                continue
            cap_scores = per_capability[capability]
            cap_scores[repository] = max(cap_scores.get(repository, 0.0), rank)
            cap_priorities = per_capability_priority[capability]
            cap_priorities[repository] = min(
                cap_priorities.get(repository, _PROVIDER_PRIORITY[provider]),
                _PROVIDER_PRIORITY[provider],
            )
            item = evidence.setdefault(repository, _RepositoryEvidence(repository=repository))
            item.capabilities.add(capability)
            item.providers.add(provider)
            item.hits += 1
            item.rank_score += rank

    def provider_search(capability: str, provider: str, query: str):
        provider_query = _provider_query(provider, query)
        if provider == "curseforge":
            return capability, provider, _search_curseforge(provider_query, limit=_query_limit(),
                with_metadata=True, minecraft_version=minecraft_version, loader=loader), [], False
        if provider == "sourcegraph":
            return capability, provider, _search_sourcegraph(provider_query, limit=_query_limit(),
                with_metadata=True), [], False
        candidates = []
        errors: list[dict[str, Any]] = []
        cursor = ""
        try:
            # The user requires the requested MC release. Catalogue filters
            # narrow discovery; pinned source metadata verifies it independently.
            for _page_index in range(_discovery_pages()):
                page = client.search(
                    provider, provider_query, limit=_query_limit(), target_profile="minecraft_mod",
                    minecraft_version=minecraft_version, loader=loader, source_reference=True,
                    **({"cursor": cursor} if cursor else {}),
                )
                raw = page.get("candidates") if isinstance(page, Mapping) else None
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                    candidates.extend(item for item in raw if isinstance(item, Mapping))
                next_cursor = page.get("next_cursor", "") if isinstance(page, Mapping) else ""
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
        except Exception as exc:  # noqa: BLE001 - preserve prior pages and their incomplete retrieval status.
            errors.append(_discovery_provider_error(provider, exc, stage="search"))
        partial_search = bool(errors and candidates)
        if provider == "modrinth":
            values = _resolve_modrinth_candidates(candidates, with_metadata=True, errors=errors)
            return capability, provider, values, errors, partial_search or bool(values and errors)
        values = []
        for index, candidate in enumerate(candidates):
            if provider == "github" and not _github_candidate_is_minecraft_mod(candidate):
                continue
            repository = _candidate_repository(candidate)
            if repository:
                values.append((repository, 1.0 / (1 + index), candidate))
        return capability, provider, values, errors, partial_search

    if isinstance(providers, (str, bytes)):
        raise TypeError("Reuse discovery providers must be a sequence of provider names.")
    provider_order = tuple(dict.fromkeys(
        str(provider).strip().casefold()
        for provider in (providers if providers is not None else ("curseforge", "modrinth", "github"))
    ))
    if any(provider not in _PROVIDER_PRIORITY for provider in provider_order):
        raise ValueError("Reuse discovery received an unsupported provider.")
    if "curseforge" in provider_order and not _curseforge_api_key():
        provider_order = tuple(provider for provider in provider_order if provider != "curseforge")
        if diagnostics is not None:
            diagnostics.append({
                "provider": "curseforge", "provider_priority": _PROVIDER_PRIORITY["curseforge"],
                "query": "", "capabilities": list(ordered), "status": "not_configured",
                "error": "", "http_status": None, "repository_count": 0,
                "catalog_result_count": 0, "usable_source_repository_count": 0,
                "source_repository_missing_count": 0,
                "repositories": [], "repository_evidence": [],
                "minecraft_version": minecraft_version, "loader": loader,
                "relevance_policy": "rank_candidates_then_inspect_actual_source",
            })
    query_results: dict[tuple[str, str], list] = {}
    query_receipts: dict[tuple[str, str], dict[str, Any]] = {}
    provider_blocks: dict[str, dict[str, Any]] = {}
    cache_target = (str(minecraft_version or "").strip(), str(loader or "").strip().casefold())
    if query_cache is not None:
        for cached_key, cached in query_cache.items():
            if len(cached_key) != 4 or cached_key[2:] != cache_target or not isinstance(cached, Mapping):
                continue
            for error in cached.get("errors", ()):
                if isinstance(error, Mapping) and error.get("backend_unavailable"):
                    provider_blocks[cached_key[0]] = dict(error)
                    break

    def cache_key(provider: str, query: str) -> tuple[str, str, str, str]:
        return provider, query, *cache_target

    def consume_result(
        job: tuple[str, str, str],
        owners: Sequence[str],
        values: list,
        errors: list[dict[str, Any]],
        partial_results: bool,
        *,
        cache_hit: bool = False,
        deferred: bool = False,
    ) -> None:
        _capability, provider, query = job
        key = (provider, query)
        query_results[key] = values
        blocking = next((error for error in errors if error.get("backend_unavailable")), None)
        if blocking is not None:
            provider_blocks[provider] = blocking
        main_error = blocking or next(iter(errors), {})
        source_repository_rows = [
            _github_repository(repository) for repository, _rank, _metadata in values
        ]
        repositories = list(dict.fromkeys(repo for repo in source_repository_rows if repo))
        receipt = {
            "provider": provider, "query": query,
            "provider_priority": _PROVIDER_PRIORITY[provider],
            "capabilities": list(owners),
            "status": "provider_error" if errors else ("found" if values else "search_empty"),
            "error": "; ".join(str(error["error"]) for error in errors),
            "errors": [dict(error) for error in errors],
            "partial_results": partial_results,
            "backend_unavailable": blocking is not None,
            "request_failure": main_error.get("request_failure"),
            "http_status": main_error.get("http_status"),
            "deferred": deferred,
            "deferred_reason": "provider_backend_unavailable" if deferred else "",
            "cache_hit": cache_hit,
            "search_executed": not cache_hit and not deferred,
            "repository_count": len(repositories),
            "usable_source_repository_count": len(repositories),
            "source_repository_missing_count": sum(not repo for repo in source_repository_rows),
            **({"catalog_result_count": len(values)} if provider == "curseforge" else {}),
            "repositories": repositories,
            "repository_evidence": [
                {
                    "repository": _github_repository(repository),
                    "provider": provider,
                    "provider_priority": _PROVIDER_PRIORITY[provider],
                    **{
                        field: metadata[field]
                        for field in (
                            "source_id", "catalog_id", "catalog_url", "source_url",
                            "versions", "version_indexes", "requested_minecraft_version",
                            "requested_loader", "source_reuse_authority",
                        )
                        if field in metadata
                    },
                }
                for repository, _rank, metadata in values
            ],
            "minecraft_version": minecraft_version, "loader": loader,
            "relevance_policy": "rank_candidates_then_inspect_actual_source",
        }
        query_receipts[key] = receipt
        if diagnostics is not None:
            diagnostics.append(receipt)
        if query_cache is not None and not deferred and not cache_hit:
            query_cache[cache_key(provider, query)] = {
                "values": list(values), "errors": [dict(error) for error in errors],
                "partial_results": partial_results, "receipt": dict(receipt),
            }
        for capability in owners:
            register(capability, provider, values)

    def run_wave(capability_set: Sequence[str], variant_index: int, provider: str) -> None:
        subscribers: dict[tuple[str, str], list[str]] = {}
        for capability in capability_set:
            query_list = variants[capability]
            if variant_index >= len(query_list):
                continue
            query = "" if provider == "curseforge" else _provider_query(provider, query_list[variant_index])
            if provider == "curseforge" and variant_index != 0:
                continue
            key = (provider, query)
            subscribers.setdefault(key, []).append(capability)

        pending: list[tuple[str, str, str]] = []
        for key, owners in subscribers.items():
            if key in query_results:
                for capability in owners:
                    register(capability, provider, query_results[key])
                    prior_receipt = query_receipts[key]
                    if capability not in prior_receipt["capabilities"]:
                        prior_receipt["capabilities"].append(capability)
                continue
            query = key[1]
            job = (owners[0], provider, query)
            cached = query_cache.get(cache_key(provider, query)) if query_cache is not None else None
            if isinstance(cached, Mapping):
                consume_result(
                    job, owners, list(cached["values"]),
                    [dict(error) for error in cached["errors"]], bool(cached["partial_results"]),
                    cache_hit=True,
                )
            else:
                pending.append(job)

        jobs: list[tuple[str, str, str]] = []
        for job in pending:
            if provider in provider_blocks:
                consume_result(
                    job, subscribers[(provider, job[2])], [], [provider_blocks[provider]],
                    False, deferred=True,
                )
            else:
                jobs.append(job)
        if not jobs:
            return

        def safe_provider_search(
            job: tuple[str, str, str],
        ):
            capability, provider, query = job
            try:
                return provider_search(capability, provider, query)
            except Exception as exc:  # noqa: BLE001 - preserve provider failure as data for staged fallback.
                error = _discovery_provider_error(provider, exc, stage="search")
                return capability, provider, [], [error], False

        for _job, result_row in iter_completed_with_deadlines(
            jobs,
            safe_provider_search,
            max_workers=min(_workers(), len(jobs)),
            stage=f"reuse-{provider}-wave-{variant_index}",
            sort_key=lambda item: (item[0], item[1], item[2]),
        ):
            _capability, provider, values, errors, partial_results = result_row
            key = (_job[1], _job[2])
            consume_result(_job, subscribers[key], values, errors, partial_results)

    # An uninspected repository count is NOT reuse coverage. Previously three
    # keyword hits stopped all alternate queries, even when every source was
    # unrelated, unlicensed, or could not compile. Complete the small, bounded
    # set of semantic query variants before the inspector/proof makes the
    # authoritative reuse decision.
    for provider in provider_order:
        for variant_index in range(_query_variant_limit()):
            wave_capabilities = [
                capability for capability in ordered
                if variant_index < len(variants[capability])
            ]
            if wave_capabilities:
                run_wave(wave_capabilities, variant_index, provider)

    selected = _representative_repositories(evidence, ordered)
    selected_set = set(selected)
    for capability in ordered:
        ranked = sorted(
            per_capability[capability],
            key=lambda repo: (
                -per_capability_priority[capability][repo],
                repo in selected_set,
                len(evidence[repo].capabilities),
                len(evidence[repo].providers),
                per_capability[capability][repo],
                evidence[repo].rank_score,
                repo,
            ),
            reverse=True,
        )
        # Representatives are the first inspection batch. Keep every ranked
        # candidate so proof failure can continue to a different source donor.
        result[capability] = tuple(ranked)
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
                -evidence[repo].provider_priority,
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
