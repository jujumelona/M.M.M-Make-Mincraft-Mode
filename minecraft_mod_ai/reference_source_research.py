from __future__ import annotations

"""Target-neutral external reference retrieval for prompt understanding.

Reference research is identity-first and host-owned. Long natural-language research
questions are deliberately not treated as the only search key: the host first resolves the
named reference itself, then expands through encyclopedia search, Wikidata language links,
and repository evidence. Provider failures are retried in a bounded way and zero-result
searches trigger broader deterministic variants before the planning state may call the
research insufficient.
"""

import atexit
import hashlib
import os
import re
import threading
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

_TIMEOUT = 12.0
_UA = "MMM-ReferenceResearch/3.0 (+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"
_GITHUB_API = "https://api.github.com"
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_MAX_WIKI_PAGES = 5
_MAX_GITHUB_REPOS = 3
_MAX_WIKIDATA_ENTITIES = 4
_MAX_QUERY_VARIANTS = 8
_MAX_PROVIDER_ATTEMPTS = 2
_HTTP_MAX_CONNECTIONS = 12

_ReferenceProvider = Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]]
_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = threading.Lock()

_GENERIC_PREFIX_TERMS = frozenset(
    {
        "what",
        "how",
        "which",
        "documented",
        "systems",
        "system",
        "rules",
        "rule",
        "behavior",
        "behaviour",
        "define",
        "defines",
        "gameplay",
        "mechanics",
        "mechanic",
        "progression",
        "features",
        "feature",
    }
)


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _http_client() -> httpx.Client:
    """Return the process-owned client so reference requests reuse keep-alive connections."""
    global _HTTP_CLIENT
    client = _HTTP_CLIENT
    if client is not None:
        return client
    with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None:
            _HTTP_CLIENT = httpx.Client(
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
                limits=httpx.Limits(
                    max_connections=_HTTP_MAX_CONNECTIONS,
                    max_keepalive_connections=_HTTP_MAX_CONNECTIONS,
                ),
            )
            atexit.register(_HTTP_CLIENT.close)
        return _HTTP_CLIENT


def _json(url: str, *, headers: Mapping[str, str] | None = None) -> Any:
    response = _http_client().get(
        url,
        headers={"Accept": "application/json", **dict(headers or {})},
    )
    response.raise_for_status()
    return response.json()


def _body(url: str, *, headers: Mapping[str, str] | None = None) -> str:
    response = _http_client().get(url, headers=dict(headers or {}))
    response.raise_for_status()
    return response.content.decode("utf-8", errors="replace")


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": _UA,
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _identity_text(value: Any) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+|[가-힣]+", str(value or "").casefold())
    )


def _identity_tokens(value: Any) -> tuple[str, ...]:
    return tuple(token for token in _identity_text(value).split() if token)


def _identity_matches(anchor: Any, source_value: Any) -> bool:
    anchor_tokens = _identity_tokens(anchor)
    source_tokens = _identity_tokens(source_value)
    if not anchor_tokens or not source_tokens:
        return False
    anchor_compact = "".join(anchor_tokens)
    widths = {
        len(anchor_tokens),
        max(1, len(anchor_tokens) - 1),
        len(anchor_tokens) + 1,
    }
    for width in sorted(widths):
        if width > len(source_tokens):
            continue
        for start in range(len(source_tokens) - width + 1):
            window = source_tokens[start : start + width]
            if window == anchor_tokens:
                return True
            if len(anchor_compact) >= 4 and "".join(window) == anchor_compact:
                return True
    return False


def _identity_variants(value: Any) -> list[str]:
    raw = _text(value)
    if not raw:
        return []
    normalized = _identity_text(raw)
    compact = "".join(normalized.split())
    variants = [raw, normalized]
    if len(compact) >= 4:
        variants.append(compact)
    return list(dict.fromkeys(item for item in variants if item))


def _infer_reference_names(queries: Sequence[str]) -> list[str]:
    """Infer the stable named prefix shared by host-authored reference queries.

    Planning-state reference queries are intentionally identity-anchored. Their common
    prefix is therefore a safer recovery key than any individual long research question.
    The inference is deterministic and refuses generic question prefixes.
    """
    token_rows = [_identity_tokens(query) for query in queries if _text(query)]
    if len(token_rows) < 2:
        return []
    common: list[str] = []
    for values in zip(*token_rows):
        if len(set(values)) != 1:
            break
        token = values[0]
        if not common and token in _GENERIC_PREFIX_TERMS:
            return []
        if token in _GENERIC_PREFIX_TERMS:
            break
        common.append(token)
        if len(common) >= 6:
            break
    candidate = " ".join(common).strip()
    return [candidate] if candidate else []


def _reference_search_plan(
    queries: Sequence[str], reference_names: Sequence[str] = ()
) -> tuple[list[str], list[str]]:
    authored = list(dict.fromkeys(_text(query) for query in queries if _text(query)))
    anchors = list(
        dict.fromkeys(
            _text(name)
            for name in reference_names
            if _text(name)
        )
    )
    if not anchors:
        anchors = _infer_reference_names(authored)

    candidates: list[str] = []
    for anchor in anchors:
        candidates.extend(_identity_variants(anchor))
    candidates.extend(authored)
    for anchor in anchors:
        candidates.extend(
            (
                f"{anchor} gameplay",
                f"{anchor} mechanics",
                f"{anchor} progression",
            )
        )
    plan = list(dict.fromkeys(_text(value) for value in candidates if _text(value)))
    return anchors, plan[:_MAX_QUERY_VARIANTS]


def _query_tokens(query: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9_+.#/-]{3,}|[가-힣]{2,}", query)
    }


def _relevant(query: str, title: str, body: str) -> bool:
    wanted = _query_tokens(query)
    if not wanted:
        return False
    available = _query_tokens(title + " " + body[:12000])
    return bool(wanted & available)


def _wikipedia_languages(value: str) -> tuple[str, ...]:
    """Search the authored language first and English as a bridge language."""
    languages: list[str] = []
    if re.search(r"[가-힣]", value):
        languages.append("ko")
    elif re.search(r"[ぁ-ゟ゠-ヿ]", value):
        languages.append("ja")
    elif re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value):
        languages.append("zh")
    if "en" not in languages:
        languages.append("en")
    return tuple(languages)


def _record_matches_anchor(record: Mapping[str, Any], anchors: Sequence[str]) -> bool:
    if not anchors:
        return True
    values = (
        record.get("title"),
        record.get("url"),
        record.get("source_id"),
    )
    return any(
        _identity_matches(anchor, value)
        for anchor in anchors
        for value in values
    )


def _wikipedia_extract_titles(
    language: str,
    titles: Sequence[str],
    *,
    query: str,
    search_mode: str,
    identity_anchor: str = "",
    wikidata_entity_id: str = "",
) -> list[dict[str, Any]]:
    clean_titles = list(dict.fromkeys(_text(title) for title in titles if _text(title)))
    if not clean_titles:
        return []
    api = f"https://{language}.wikipedia.org/w/api.php"
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "titles": "|".join(clean_titles[:_MAX_WIKI_PAGES]),
        }
    )
    payload = _json(f"{api}?{params}")
    pages = payload.get("query", {}).get("pages", {}) if isinstance(payload, Mapping) else {}
    records: list[dict[str, Any]] = []
    for page in pages.values() if isinstance(pages, Mapping) else []:
        if not isinstance(page, Mapping) or "missing" in page:
            continue
        title = _text(page.get("title"))
        body = str(page.get("extract") or "").strip()
        url = str(page.get("fullurl") or "").strip()
        if len(body) < 120:
            continue
        page_id = str(page.get("pageid") or title)
        source_id = f"wikipedia:{language}:{page_id}"
        if wikidata_entity_id and identity_anchor:
            source_id = (
                f"wikidata:{wikidata_entity_id}:{identity_anchor}:"
                f"{language}:{page_id}"
            )
        records.append(
            {
                "source_id": source_id,
                "source_type": "reference_encyclopedia_body",
                "source_locator": url or f"wikipedia:{language}:{title}",
                "url": url,
                "title": title,
                "content": body,
                "content_sha256": _sha(body),
                "body_retrieved": True,
                "evidence_origin": (
                    "wikidata_sitelink_wikipedia_extract"
                    if wikidata_entity_id
                    else "wikipedia_page_extract"
                ),
                "metadata": {
                    "provider": "wikidata" if wikidata_entity_id else "wikipedia",
                    "language": language,
                    "query": query,
                    "search_mode": search_mode,
                    "identity_anchor": identity_anchor,
                    "wikidata_entity_id": wikidata_entity_id,
                },
            }
        )
    return records


def _wikipedia_search_titles(query: str, language: str) -> list[str]:
    api = f"https://{language}.wikipedia.org/w/api.php"
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": _MAX_WIKI_PAGES,
            "utf8": 1,
        }
    )
    search = _json(f"{api}?{params}")
    rows = search.get("query", {}).get("search", []) if isinstance(search, Mapping) else []
    return [
        _text(row.get("title"))
        for row in rows
        if isinstance(row, Mapping) and _text(row.get("title"))
    ][: _MAX_WIKI_PAGES]


def _wikipedia_sources(
    queries: Sequence[str], anchors: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_values = list(queries) or list(anchors)
    language_basis = " ".join([*anchors, *plan_values])
    languages = _wikipedia_languages(language_basis)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    search_requests = 0
    source_requests = 0
    language_receipts: dict[str, Any] = {}

    for language in languages:
        language_records: list[dict[str, Any]] = []
        language_errors: list[str] = []

        # Round 1: resolve the reference identity directly as an encyclopedia title.
        for anchor in anchors:
            try:
                variants = _identity_variants(anchor)
                source_requests += 1
                found = _wikipedia_extract_titles(
                    language,
                    variants,
                    query=anchor,
                    search_mode="identity_exact_title",
                    identity_anchor=anchor,
                )
                language_records.extend(
                    record for record in found if _record_matches_anchor(record, [anchor])
                )
            except Exception as exc:
                language_errors.append(f"exact:{type(exc).__name__}:{exc}")

        # Round 2+: broaden deterministically only after identity lookup has been attempted.
        for query in plan_values:
            if len(language_records) >= _MAX_WIKI_PAGES:
                break
            try:
                search_requests += 1
                titles = _wikipedia_search_titles(query, language)
                if not titles:
                    continue
                source_requests += 1
                found = _wikipedia_extract_titles(
                    language,
                    titles,
                    query=query,
                    search_mode="expanded_search",
                )
                for record in found:
                    if anchors:
                        if _record_matches_anchor(record, anchors):
                            language_records.append(record)
                    elif _relevant(query, str(record.get("title") or ""), str(record.get("content") or "")):
                        language_records.append(record)
            except Exception as exc:
                language_errors.append(f"search:{type(exc).__name__}:{exc}")

        records.extend(language_records)
        language_receipts[language] = {
            "provider": f"wikipedia_{language}",
            "status": "available" if language_records or not language_errors else "error",
            "result_count": len(language_records),
            "errors": language_errors[:4],
        }
        errors.extend(f"{language}:{error}" for error in language_errors)

    unique = _dedupe(records)
    return unique, {
        "provider": "wikipedia",
        "status": "available" if unique or not errors else "error",
        "languages": list(languages),
        "result_count": len(unique),
        "search_requests": search_requests,
        "source_requests": source_requests,
        "language_receipts": language_receipts,
        "errors": errors[:6],
        "policy": "identity_exact_then_bounded_expansion",
    }


def _wikidata_sources(
    queries: Sequence[str], anchors: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del queries
    if not anchors:
        return [], {
            "provider": "wikidata",
            "status": "skipped_no_identity_anchor",
            "result_count": 0,
            "search_requests": 0,
            "source_requests": 0,
        }

    languages = _wikipedia_languages(" ".join(anchors))
    entity_anchors: dict[str, str] = {}
    errors: list[str] = []
    search_requests = 0
    source_requests = 0

    for anchor in anchors:
        for language in languages:
            try:
                params = urllib.parse.urlencode(
                    {
                        "action": "wbsearchentities",
                        "format": "json",
                        "search": anchor,
                        "language": language,
                        "uselang": language,
                        "limit": _MAX_WIKIDATA_ENTITIES,
                    }
                )
                search_requests += 1
                payload = _json(f"{_WIKIDATA_API}?{params}")
                rows = payload.get("search", []) if isinstance(payload, Mapping) else []
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, Mapping):
                        continue
                    entity_id = _text(row.get("id"))
                    label = _text(row.get("label"))
                    aliases = row.get("aliases")
                    alias_values = (
                        [_text(value) for value in aliases if _text(value)]
                        if isinstance(aliases, list)
                        else []
                    )
                    if not entity_id:
                        continue
                    if _identity_matches(anchor, label) or any(
                        _identity_matches(anchor, value) for value in alias_values
                    ):
                        entity_anchors.setdefault(entity_id, anchor)
                    if len(entity_anchors) >= _MAX_WIKIDATA_ENTITIES:
                        break
            except Exception as exc:
                errors.append(f"search:{language}:{type(exc).__name__}:{exc}")

    if not entity_anchors:
        return [], {
            "provider": "wikidata",
            "status": "available" if not errors else "error",
            "languages": list(languages),
            "result_count": 0,
            "search_requests": search_requests,
            "source_requests": source_requests,
            "errors": errors[:6],
            "policy": "identity_to_language_sitelink_bridge",
        }

    try:
        params = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(entity_anchors),
                "props": "sitelinks",
            }
        )
        source_requests += 1
        payload = _json(f"{_WIKIDATA_API}?{params}")
        entities = payload.get("entities", {}) if isinstance(payload, Mapping) else {}
    except Exception as exc:
        errors.append(f"entities:{type(exc).__name__}:{exc}")
        entities = {}

    records: list[dict[str, Any]] = []
    preferred_languages = list(dict.fromkeys([*languages, "en", "ko", "ja", "zh"]))
    for entity_id, anchor in entity_anchors.items():
        entity = entities.get(entity_id) if isinstance(entities, Mapping) else None
        sitelinks = entity.get("sitelinks") if isinstance(entity, Mapping) else None
        if not isinstance(sitelinks, Mapping):
            continue
        for language in preferred_languages:
            site = sitelinks.get(f"{language}wiki")
            title = _text(site.get("title")) if isinstance(site, Mapping) else ""
            if not title:
                continue
            try:
                source_requests += 1
                records.extend(
                    _wikipedia_extract_titles(
                        language,
                        [title],
                        query=anchor,
                        search_mode="wikidata_sitelink",
                        identity_anchor=anchor,
                        wikidata_entity_id=entity_id,
                    )
                )
            except Exception as exc:
                errors.append(
                    f"sitelink:{entity_id}:{language}:{type(exc).__name__}:{exc}"
                )
            if len(records) >= _MAX_WIKI_PAGES:
                break
        if len(records) >= _MAX_WIKI_PAGES:
            break

    unique = _dedupe(records)
    return unique, {
        "provider": "wikidata",
        "status": "available" if unique or not errors else "error",
        "languages": preferred_languages,
        "result_count": len(unique),
        "search_requests": search_requests,
        "source_requests": source_requests,
        "errors": errors[:6],
        "policy": "identity_to_language_sitelink_bridge",
    }


def _github_reference_sources(
    queries: Sequence[str], anchors: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    headers = _github_headers()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    search_requests = 0
    source_requests = 0
    plan = list(dict.fromkeys([*anchors, *queries]))[:4]

    for query in plan:
        if len(records) >= _MAX_GITHUB_REPOS:
            break
        params = urllib.parse.urlencode(
            {
                "q": query + " in:name,description,readme,topics",
                "per_page": _MAX_GITHUB_REPOS,
            }
        )
        try:
            search_requests += 1
            payload = _json(f"{_GITHUB_API}/search/repositories?{params}", headers=headers)
        except Exception as exc:
            errors.append(f"search:{type(exc).__name__}:{exc}")
            continue
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        for repository in items[:_MAX_GITHUB_REPOS] if isinstance(items, list) else []:
            if not isinstance(repository, Mapping):
                continue
            full_name = _text(repository.get("full_name"))
            if not full_name or "/" not in full_name:
                continue
            try:
                source_requests += 1
                readme = _json(f"{_GITHUB_API}/repos/{full_name}/readme", headers=headers)
                download_url = str(readme.get("download_url") or "") if isinstance(readme, Mapping) else ""
                if not download_url:
                    continue
                body = _body(download_url, headers={"User-Agent": _UA}).strip()
            except Exception as exc:
                errors.append(f"{full_name}:{type(exc).__name__}:{exc}")
                continue
            title = full_name
            if len(body) < 120 or not _relevant(query, title, body):
                continue
            record = {
                "source_id": f"github-reference:{full_name}",
                "source_type": "reference_repository_body",
                "source_locator": f"github:{full_name}",
                "url": f"https://github.com/{full_name}",
                "title": title,
                "content": body,
                "content_sha256": _sha(body),
                "body_retrieved": True,
                "evidence_origin": "github_reference_readme",
                "metadata": {
                    "provider": "github",
                    "repository": full_name,
                    "query": query,
                },
            }
            if anchors and not _record_matches_anchor(record, anchors):
                continue
            records.append(record)
            if len(records) >= _MAX_GITHUB_REPOS:
                break

    unique = _dedupe(records)
    return unique, {
        "provider": "github_reference",
        "status": "available" if unique or not errors else "error",
        "result_count": len(unique),
        "search_requests": search_requests,
        "source_requests": source_requests,
        "errors": errors[:6],
        "policy": "independent_identity_anchored_provider",
    }


def _dedupe(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        record = dict(raw)
        key = str(record.get("content_sha256") or record.get("source_id") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def _retrieve_provider(
    provider: str,
    function: _ReferenceProvider,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str] | None]:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_PROVIDER_ATTEMPTS + 1):
        try:
            found, raw_receipt = function()
            receipt = {**dict(raw_receipt), "attempts": attempt}
            return found, receipt, None
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    return (
        [],
        {
            "provider": provider,
            "status": "error",
            "result_count": 0,
            "attempts": _MAX_PROVIDER_ATTEMPTS,
        },
        {"provider": provider, "error": f"{type(last_error).__name__}: {last_error}"},
    )


def retrieve_reference_grounded_evidence(queries: Sequence[str]) -> dict[str, Any]:
    """Run identity-first bounded expansion before declaring reference research empty."""
    query_list = list(dict.fromkeys(_text(raw) for raw in queries if _text(raw)))
    anchors, search_plan = _reference_search_plan(query_list)
    provider_specs: tuple[tuple[str, _ReferenceProvider], ...] = (
        ("wikipedia", lambda: _wikipedia_sources(search_plan, anchors)),
        ("wikidata", lambda: _wikidata_sources(search_plan, anchors)),
        ("github_reference", lambda: _github_reference_sources(search_plan, anchors)),
    )

    with ThreadPoolExecutor(
        max_workers=len(provider_specs),
        thread_name_prefix="mmm-reference-provider",
    ) as executor:
        futures = [
            executor.submit(_retrieve_provider, provider, function)
            for provider, function in provider_specs
        ]
        provider_results = [future.result() for future in futures]

    records: list[dict[str, Any]] = []
    providers: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for (provider, _), (found, receipt, error) in zip(
        provider_specs, provider_results, strict=True
    ):
        records.extend(found)
        providers[provider] = receipt
        if error is not None:
            errors.append(error)

    unique = _dedupe(records)
    combined_query = "\n".join(query_list)
    return {
        "schema_version": "mmm/reference-grounded-evidence-v3",
        "retrieval_strategy": "identity_first_bounded_expansion",
        "inferred_reference_names": anchors,
        "search_plan": search_plan,
        "queries": [
            {
                "query": query_list[0] if query_list else "",
                "authored_queries": query_list,
                "query_sha256": _sha(combined_query),
                "evidence_records": unique,
                "content_record_count": len(unique),
                "provider_receipts": providers,
                "retrieval_errors": errors,
                "provider_policy": "independent_parallel_after_identity_first_expansion",
                "search_plan": search_plan,
            }
        ]
        if query_list
        else [],
    }


__all__ = ["retrieve_reference_grounded_evidence"]
