from __future__ import annotations

"""Target-neutral external reference retrieval for prompt understanding.

Minecraft ecosystem retrieval intentionally filters for Minecraft mods. Reference-driven
requests need a different source path before any Minecraft implementation decision exists.
Encyclopedic sources are authoritative for reference semantics; GitHub README search is a
last-resort fallback only when Wikipedia returns no claim-bearing body. It is never run in
parallel with a successful encyclopedia lookup.
"""

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_TIMEOUT = 12.0
_UA = "MMM-ReferenceResearch/2.0 (+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"
_GITHUB_API = "https://api.github.com"
_MAX_WIKI_PAGES = 3
_MAX_GITHUB_REPOS = 2

_ReferenceProvider = Callable[[str], tuple[list[dict[str, Any]], dict[str, Any]]]


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _json(url: str, *, headers: Mapping[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": _UA, **dict(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


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
    """Search the language implied by the authored reference, then English as fallback."""
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
            request = urllib.request.Request(download_url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
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
                "evidence_origin": "github_reference_readme_fallback",
                "metadata": {"provider": "github", "repository": full_name, "query": query},
            }
        )
    return records, {
        "provider": "github_reference",
        "status": "available",
        "result_count": len(records),
        "errors": errors[:3],
        "policy": "wikipedia_empty_fallback_only",
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


def retrieve_reference_grounded_evidence(queries: Sequence[str]) -> dict[str, Any]:
    """Retrieve encyclopedia evidence first; GitHub is a no-result fallback per query."""
    query_list = [_text(raw) for raw in queries]
    query_list = [query for query in query_list if query]
    rows: list[dict[str, Any]] = []

    for query in query_list:
        records: list[dict[str, Any]] = []
        providers: dict[str, Any] = {}
        errors: list[dict[str, str]] = []

        wiki_found, wiki_receipt, wiki_error = _retrieve_provider(
            query, "wikipedia", _wikipedia_sources
        )
        records.extend(wiki_found)
        providers["wikipedia"] = wiki_receipt
        if wiki_error is not None:
            errors.append(wiki_error)

        if wiki_found:
            providers["github_reference"] = {
                "provider": "github_reference",
                "status": "skipped_wikipedia_has_evidence",
                "result_count": 0,
                "policy": "wikipedia_empty_fallback_only",
            }
        else:
            github_found, github_receipt, github_error = _retrieve_provider(
                query, "github_reference", _github_reference_sources
            )
            records.extend(github_found)
            providers["github_reference"] = github_receipt
            if github_error is not None:
                errors.append(github_error)

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            key = str(record.get("content_sha256") or record.get("source_id") or "")
            if key and key not in seen:
                seen.add(key)
                unique.append(record)
        rows.append(
            {
                "query": query,
                "query_sha256": _sha(query),
                "evidence_records": unique,
                "content_record_count": len(unique),
                "provider_receipts": providers,
                "retrieval_errors": errors,
                "provider_policy": "wikipedia_then_github_empty_fallback",
            }
        )

    return {
        "schema_version": "mmm/reference-grounded-evidence-v2",
        "queries": rows,
    }


__all__ = ["retrieve_reference_grounded_evidence"]
