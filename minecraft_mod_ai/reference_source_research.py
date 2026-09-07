from __future__ import annotations

"""Target-neutral external reference retrieval for prompt understanding.

Minecraft ecosystem retrieval intentionally filters for Minecraft mods. Reference-driven
requests need a different source path before any Minecraft implementation decision exists.
Wikipedia and GitHub are independent, host-owned reference providers. Every authored query
runs against both providers; no provider is selected because another provider failed or
returned no evidence.
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
_UA = "MMM-ReferenceResearch/2.0 (+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"
_GITHUB_API = "https://api.github.com"
_MAX_WIKI_PAGES = 3
_MAX_GITHUB_REPOS = 2
_MAX_QUERY_WORKERS = 4
_HTTP_MAX_CONNECTIONS = 12

_ReferenceProvider = Callable[[str], tuple[list[dict[str, Any]], dict[str, Any]]]
_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = threading.Lock()


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


def _wikipedia_languages(query: str) -> tuple[str, ...]:
    """Search the language implied by the authored reference plus English."""
    languages: list[str] = []
    if re.search(r"[가-힣]", query):
        languages.append("ko")
    elif re.search(r"[ぁ-ゟ゠-ヿ]", query):
        languages.append("ja")
    elif re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", query):
        languages.append("zh")
    if "en" not in languages:
        languages.append("en")
    return tuple(languages)


def _wikipedia_language_sources(
    query: str,
    language: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    titles = [
        _text(row.get("title"))
        for row in rows
        if isinstance(row, Mapping) and _text(row.get("title"))
    ][: _MAX_WIKI_PAGES]
    if not titles:
        return [], {
            "provider": f"wikipedia_{language}",
            "status": "available",
            "result_count": 0,
        }

    extract_params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "titles": "|".join(titles),
        }
    )
    payload = _json(f"{api}?{extract_params}")
    pages = payload.get("query", {}).get("pages", {}) if isinstance(payload, Mapping) else {}
    records: list[dict[str, Any]] = []
    for page in pages.values() if isinstance(pages, Mapping) else []:
        if not isinstance(page, Mapping):
            continue
        title = _text(page.get("title"))
        body = str(page.get("extract") or "").strip()
        url = str(page.get("fullurl") or "").strip()
        if len(body) < 120 or not _relevant(query, title, body):
            continue
        records.append(
            {
                "source_id": f"wikipedia:{language}:{page.get('pageid', title)}",
                "source_type": "reference_encyclopedia_body",
                "source_locator": url or f"wikipedia:{language}:{title}",
                "url": url,
                "title": title,
                "content": body,
                "content_sha256": _sha(body),
                "body_retrieved": True,
                "evidence_origin": "wikipedia_page_extract",
                "metadata": {
                    "provider": "wikipedia",
                    "language": language,
                    "query": query,
                },
            }
        )
    return records, {
        "provider": f"wikipedia_{language}",
        "status": "available",
        "result_count": len(records),
    }


def _wikipedia_sources(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    errors: list[str] = []
    for language in _wikipedia_languages(query):
        try:
            found, receipt = _wikipedia_language_sources(query, language)
            records.extend(found)
            receipts[language] = receipt
        except Exception as exc:
            receipts[language] = {
                "provider": f"wikipedia_{language}",
                "status": "error",
                "result_count": 0,
            }
            errors.append(f"{language}:{type(exc).__name__}:{exc}")
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record.get("content_sha256") or record.get("source_id") or "")
        if key and key not in seen:
            seen.add(key)
            unique.append(record)
    return unique, {
        "provider": "wikipedia",
        "status": "available" if unique or not errors else "error",
        "languages": list(_wikipedia_languages(query)),
        "result_count": len(unique),
        "language_receipts": receipts,
        "errors": errors[:3],
    }


def _github_reference_sources(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    headers = _github_headers()
    params = urllib.parse.urlencode(
        {
            "q": query + " in:name,description,readme,topics",
            "per_page": _MAX_GITHUB_REPOS,
        }
    )
    payload = _json(f"{_GITHUB_API}/search/repositories?{params}", headers=headers)
    items = payload.get("items", []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for repository in items[:_MAX_GITHUB_REPOS] if isinstance(items, list) else []:
        if not isinstance(repository, Mapping):
            continue
        full_name = _text(repository.get("full_name"))
        if not full_name or "/" not in full_name:
            continue
        try:
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
        records.append(
            {
                "source_id": f"github-reference:{full_name}",
                "source_type": "reference_repository_body",
                "source_locator": f"github:{full_name}",
                "url": f"https://github.com/{full_name}",
                "title": title,
                "content": body,
                "content_sha256": _sha(body),
                "body_retrieved": True,
                "evidence_origin": "github_reference_readme",
                "metadata": {"provider": "github", "repository": full_name, "query": query},
            }
        )
    return records, {
        "provider": "github_reference",
        "status": "available",
        "result_count": len(records),
        "errors": errors[:3],
        "policy": "independent_provider",
    }


def _retrieve_provider(
    query: str,
    provider: str,
    function: _ReferenceProvider,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str] | None]:
    try:
        found, receipt = function(query)
        return found, receipt, None
    except Exception as exc:
        return (
            [],
            {
                "provider": provider,
                "status": "error",
                "result_count": 0,
            },
            {"provider": provider, "error": f"{type(exc).__name__}: {exc}"},
        )


def _retrieve_query_row(query: str) -> dict[str, Any]:
    """Retrieve one query from every fixed provider without failover semantics."""
    provider_specs: tuple[tuple[str, _ReferenceProvider], ...] = (
        ("wikipedia", _wikipedia_sources),
        ("github_reference", _github_reference_sources),
    )
    with ThreadPoolExecutor(
        max_workers=len(provider_specs),
        thread_name_prefix="mmm-reference-provider",
    ) as executor:
        futures = [
            executor.submit(_retrieve_provider, query, provider, function)
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

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record.get("content_sha256") or record.get("source_id") or "")
        if key and key not in seen:
            seen.add(key)
            unique.append(record)
    return {
        "query": query,
        "query_sha256": _sha(query),
        "evidence_records": unique,
        "content_record_count": len(unique),
        "provider_receipts": providers,
        "retrieval_errors": errors,
        "provider_policy": "independent_parallel",
    }


def retrieve_reference_grounded_evidence(queries: Sequence[str]) -> dict[str, Any]:
    """Retrieve independent queries concurrently while preserving authored query order."""
    query_list = [_text(raw) for raw in queries]
    query_list = [query for query in query_list if query]

    if len(query_list) <= 1:
        rows = [_retrieve_query_row(query) for query in query_list]
    else:
        workers = min(_MAX_QUERY_WORKERS, len(query_list))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mmm-reference",
        ) as executor:
            rows = list(executor.map(_retrieve_query_row, query_list))

    return {
        "schema_version": "mmm/reference-grounded-evidence-v2",
        "queries": rows,
    }


__all__ = ["retrieve_reference_grounded_evidence"]