from __future__ import annotations

"""Bounded external source-body acquisition for pre-design RAG.

The base pre-design retriever owns the approved query plan and local/project evidence.
This contract augments only those already-approved queries with real external source
bodies. Search metadata and snippets are never promoted to evidence.
"""

import base64
import hashlib
import math
import os
import re
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False
_DEFAULT_MAX_QUERIES = 12
_HARD_MAX_QUERIES = 20
_MAX_REPOSITORIES_PER_QUERY = 3
_GITHUB_API = "https://api.github.com"
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
_STOP = {
    "fabric",
    "minecraft",
    "mod",
    "mods",
    "mode",
    "source",
    "project",
    "implementation",
}


def _max_queries() -> int:
    raw = os.environ.get("MMM_PREDESIGN_EXTERNAL_SOURCE_QUERIES", "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_MAX_QUERIES
    except ValueError:
        value = _DEFAULT_MAX_QUERIES
    return max(1, min(value, _HARD_MAX_QUERIES))


def _clean_query(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _query_terms(query: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in _WORD.findall(query)
        if len(token) >= 3
    }
    specific = {token for token in tokens if token not in _STOP}
    return specific or tokens


def _body_relevant(query: str, body: str) -> bool:
    wanted = _query_terms(query)
    if not wanted:
        return bool(body.strip())
    body_terms = {
        token.casefold()
        for token in _WORD.findall(body)
        if len(token) >= 3
    }
    return bool(wanted & body_terms)


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "mmm-pre-design-source-rag",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _status_error(response: Any, *, operation: str) -> str:
    status = int(getattr(response, "status_code", 0) or 0)
    remaining = str(getattr(response, "headers", {}).get("x-ratelimit-remaining", ""))
    reset = str(getattr(response, "headers", {}).get("x-ratelimit-reset", ""))
    suffix = ""
    if remaining or reset:
        suffix = f" rate_remaining={remaining or '?'} reset={reset or '?'}"
    return f"github {operation} HTTP {status}{suffix}".strip()


def _decode_readme(payload: Mapping[str, Any]) -> str:
    encoded = str(payload.get("content") or "")
    encoding = str(payload.get("encoding") or "").casefold()
    if not encoded or encoding != "base64":
        return ""
    try:
        raw = base64.b64decode(encoded, validate=False)
    except Exception:
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) < 40:
        return ""
    return text


def _retrieve_github_source_body(query: str) -> dict[str, Any]:
    """Search GitHub and return at most one verified README body for *query*."""

    import httpx

    search_requests = 0
    source_requests = 0
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    headers = _headers()

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
            search_requests += 1
            response = client.get(
                f"{_GITHUB_API}/search/repositories",
                params={"q": query, "per_page": _MAX_REPOSITORIES_PER_QUERY},
            )
            if response.status_code != 200:
                errors.append(_status_error(response, operation="repository search"))
                status = "rate_limited" if response.status_code in {403, 429} else "unavailable"
                return {
                    "records": records,
                    "search_requests": search_requests,
                    "source_requests": source_requests,
                    "provider_status": status,
                    "saturation_reason": "provider_limited" if status == "rate_limited" else "search_failed",
                    "errors": errors,
                }
            try:
                payload = response.json()
            except Exception as exc:
                errors.append(f"github repository search JSON decode failed: {type(exc).__name__}: {exc}")
                return {
                    "records": records,
                    "search_requests": search_requests,
                    "source_requests": source_requests,
                    "provider_status": "unavailable",
                    "saturation_reason": "search_response_invalid",
                    "errors": errors,
                }
            items = payload.get("items") if isinstance(payload, Mapping) else None
            repositories = [item for item in (items or ()) if isinstance(item, Mapping)]
            if not repositories:
                return {
                    "records": records,
                    "search_requests": search_requests,
                    "source_requests": source_requests,
                    "provider_status": "available",
                    "saturation_reason": "search_exhausted_no_repository",
                    "errors": errors,
                }

            for repository in repositories[:_MAX_REPOSITORIES_PER_QUERY]:
                full_name = str(repository.get("full_name") or "").strip()
                if not full_name or "/" not in full_name:
                    continue
                source_requests += 1
                readme = client.get(f"{_GITHUB_API}/repos/{full_name}/readme")
                if readme.status_code != 200:
                    # A repository without a README is not a source-body success. Continue
                    # to the next search result instead of promoting search metadata.
                    if readme.status_code not in {404}:
                        errors.append(_status_error(readme, operation=f"README fetch {full_name}"))
                    continue
                try:
                    readme_payload = readme.json()
                except Exception as exc:
                    errors.append(
                        f"github README JSON decode failed for {full_name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if not isinstance(readme_payload, Mapping):
                    continue
                body = _decode_readme(readme_payload)
                if not body or not _body_relevant(query, body):
                    continue
                locator = str(readme_payload.get("html_url") or repository.get("html_url") or "").strip()
                if not locator:
                    continue
                digest = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
                records.append(
                    {
                        "source_id": "github:"
                        + full_name
                        + ":"
                        + str(readme_payload.get("sha") or digest.removeprefix("sha256:")),
                        "source_type": "github_source_body",
                        "source_locator": locator,
                        "url": locator,
                        "title": full_name,
                        "content": body,
                        "content_sha256": digest,
                        "body_retrieved": True,
                        "metadata": {
                            "repository": full_name,
                            "default_branch": str(repository.get("default_branch") or ""),
                            "readme_path": str(readme_payload.get("path") or "README"),
                            "query": query,
                        },
                        "retrieval": {
                            "provider": "github",
                            "provider_status": "available",
                            "body_retrieved": True,
                        },
                    }
                )
                break
    except Exception as exc:
        errors.append(f"github source acquisition failed: {type(exc).__name__}: {exc}")
        return {
            "records": records,
            "search_requests": search_requests,
            "source_requests": source_requests,
            "provider_status": "unavailable",
            "saturation_reason": "transport_failure",
            "errors": errors,
        }

    return {
        "records": records,
        "search_requests": search_requests,
        "source_requests": source_requests,
        "provider_status": "available",
        "saturation_reason": (
            "source_body_retrieved"
            if records
            else "repositories_found_no_claim_bearing_source_body"
        ),
        "errors": errors,
    }


def _stable_queries(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for raw in value:
        query = _clean_query(raw)
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            result.append(query)
    return result


def _planned_requirement_query_keys(payload: Mapping[str, Any]) -> set[str]:
    """Select one approved retrieval query per authored requirement when available."""

    if str(payload.get("schema_version") or "") == "mmm/corrective-retrieval-request-v1":
        return {
            query.casefold()
            for domain in payload.get("domains", ())
            if isinstance(domain, Mapping)
            for query in _stable_queries(domain.get("queries"))
        }

    selected: set[str] = set()
    domains = payload.get("domains")
    for domain in domains if isinstance(domains, list) else ():
        if not isinstance(domain, Mapping) or str(domain.get("domain_id") or "") != "request":
            continue
        requirements = _stable_queries(domain.get("requirements"))
        prompt = requirements[0] if requirements else ""
        if not prompt:
            continue
        try:
            from . import authored_scope_research_contract as authored_scope

            catalog = authored_scope._active_catalog(prompt)
        except Exception:
            catalog = None
        rows = catalog.get("requirements") if isinstance(catalog, Mapping) else None
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            planned = _stable_queries(raw.get("search_queries"))
            if planned:
                selected.add(planned[0].casefold())
    return selected


def _fallback_query_keys(bundle: Mapping[str, Any], limit: int) -> set[str]:
    all_queries: list[str] = []
    for domain in bundle.get("domains", ()) if isinstance(bundle.get("domains"), list) else ():
        if not isinstance(domain, Mapping):
            continue
        for row in domain.get("queries", ()) if isinstance(domain.get("queries"), list) else ():
            if isinstance(row, Mapping):
                query = _clean_query(row.get("query"))
                if query:
                    all_queries.append(query)
    if len(all_queries) <= limit:
        return {query.casefold() for query in all_queries}
    stride = max(1, math.floor(len(all_queries) / limit))
    chosen = all_queries[::stride][:limit]
    return {query.casefold() for query in chosen}


def _augment_bundle(payload: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(bundle)
    raw_domains = bundle.get("domains")
    if not isinstance(raw_domains, list):
        return result

    limit = _max_queries()
    selected = _planned_requirement_query_keys(payload)
    if not selected:
        selected = _fallback_query_keys(bundle, limit)
    selected = set(list(selected)[:limit])

    augmented_domains: list[Any] = []
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, Mapping):
            augmented_domains.append(raw_domain)
            continue
        domain = dict(raw_domain)
        raw_rows = raw_domain.get("queries")
        rows: list[Any] = []
        for raw_row in raw_rows if isinstance(raw_rows, list) else ():
            if not isinstance(raw_row, Mapping):
                rows.append(raw_row)
                continue
            row = dict(raw_row)
            query = _clean_query(row.get("query"))
            if query.casefold() in selected:
                receipt = _retrieve_github_source_body(query)
                external = row.get("external_rag")
                external_map = dict(external) if isinstance(external, Mapping) else {}
                existing_records = external_map.get("records")
                records = list(existing_records) if isinstance(existing_records, list) else []
                records.extend(
                    record
                    for record in receipt.get("records", ())
                    if isinstance(record, Mapping)
                )
                external_map["records"] = records
                external_map["github_retrieval"] = {
                    "provider_status": str(receipt.get("provider_status") or "unavailable"),
                    "saturation_reason": str(receipt.get("saturation_reason") or ""),
                    "search_requests": int(receipt.get("search_requests") or 0),
                    "source_requests": int(receipt.get("source_requests") or 0),
                }
                errors = external_map.get("errors")
                merged_errors = list(errors) if isinstance(errors, list) else []
                merged_errors.extend(str(item) for item in receipt.get("errors", ()) if str(item))
                external_map["errors"] = merged_errors
                row["external_rag"] = external_map
            rows.append(row)
        domain["queries"] = rows
        augmented_domains.append(domain)
    result["domains"] = augmented_domains
    return result


def install(pre_design_rag_module: Any) -> None:
    global _INSTALLED
    current = pre_design_rag_module._forced_rag_bundle
    if getattr(current, "_mmm_external_source_body_v1", False):
        _INSTALLED = True
        return

    @wraps(current)
    def forced_with_external_sources(
        router: Any, research_brief: Mapping[str, Any]
    ) -> dict[str, Any]:
        local_bundle = current(router, research_brief)
        if not isinstance(local_bundle, Mapping):
            return local_bundle
        return _augment_bundle(research_brief, local_bundle)

    forced_with_external_sources._mmm_external_source_body_v1 = True  # type: ignore[attr-defined]
    forced_with_external_sources.__wrapped__ = current  # type: ignore[attr-defined]
    pre_design_rag_module._forced_rag_bundle = forced_with_external_sources
    _INSTALLED = True


__all__ = ["install"]
