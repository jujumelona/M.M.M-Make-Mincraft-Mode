from __future__ import annotations

"""Bounded external source-body acquisition for pre-design RAG.

The base pre-design retriever owns the approved query plan and local/project evidence.
This contract augments only those already-approved queries with real external source
bodies. Search metadata and snippets are never promoted to evidence.
"""

import base64
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False
_DEFAULT_MAX_QUERIES = 20
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


def _github_token() -> str:
    return os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()


def _max_queries() -> int:
    raw = os.environ.get("MMM_PREDESIGN_EXTERNAL_SOURCE_QUERIES", "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_MAX_QUERIES
    except ValueError:
        value = _DEFAULT_MAX_QUERIES
    # This value is a fallback batch budget only. Approved requirement coverage
    # is never truncated to fit a provider request count. Provider rate limits are
    # observed dynamically and recorded as provider state instead.
    return max(1, min(value, _HARD_MAX_QUERIES))


def _clean_query(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _emit_source_trace(event: str, **fields: Any) -> None:
    print(
        "PRE-DESIGN RAG TRACE: "
        + json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def _repository_search_query(query: str) -> str:
    # Repository search is AND-like.  Feeding the full natural-language retrieval query
    # over-constrains discovery.  Keep only the first two domain-specific terms plus
    # minecraft, while preserving the original query in every receipt.
    ordered: list[str] = []
    for token in _WORD.findall(query):
        folded = token.casefold()
        if len(folded) < 3 or folded in _STOP or folded in ordered:
            continue
        ordered.append(folded)
    compact = ordered[:2]
    if "minecraft" not in compact:
        compact.append("minecraft")
    return " ".join(compact) or "minecraft"


def _query_terms(query: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in _WORD.findall(query)
        if len(token) >= 3
    }
    specific = {token for token in tokens if token not in _STOP}
    return specific or tokens


_GENERIC_QUERY_TERMS = frozenset({
    "build", "building", "create", "custom", "craft", "crafting", "make",
    "minecraft", "fabric", "forge", "neoforge", "mod", "mods", "mode",
    "source", "implementation", "system", "feature", "space", "game",
})
_GENERIC_REPOSITORY_MARKERS = (
    "studentsatbuild", "student zone", "awesome-minecraft", "awesome minecraft",
    "stockmarket", "stock market", "mindcraft-bots", "minecraft bot", "mineflayer",
    "llm agent", "learning path", "bootcamp", "tutorial collection", "games list",
)


def _specific_query_terms(query: str) -> set[str]:
    return {term for term in _query_terms(query) if term not in _GENERIC_QUERY_TERMS}


def _term_overlap(wanted: set[str], available: set[str]) -> bool:
    for left in wanted:
        for right in available:
            if left == right:
                return True
            if min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left)):
                return True
            # Capability words often differ only by morphology in repository metadata
            # (seasonal/seasons, colony/colonization, planting/plant).  Accept a long
            # semantic stem while still requiring Minecraft-ecosystem gating separately.
            common = 0
            for lch, rch in zip(left, right):
                if lch != rch:
                    break
                common += 1
            if common >= 5 and (len(left) - common <= 4 or len(right) - common <= 4):
                return True
    return False


def _repository_candidate_relevant(query: str, repository: Mapping[str, Any]) -> bool:
    full_name = str(repository.get("full_name") or "").strip()
    description = str(repository.get("description") or "").strip()
    topics = repository.get("topics")
    topic_text = " ".join(str(item) for item in topics) if isinstance(topics, list) else ""
    folded = " ".join((full_name, description, topic_text)).casefold()
    if not folded or any(marker in folded for marker in _GENERIC_REPOSITORY_MARKERS):
        return False
    if "minecraft" not in folded:
        return False
    terms = {token.casefold() for token in _WORD.findall(folded) if len(token) >= 3}
    specific = _specific_query_terms(query)
    if specific and not _term_overlap(specific, terms):
        return False
    return True


def _body_relevant(query: str, body: str) -> bool:
    wanted = _query_terms(query)
    body_terms = {token.casefold() for token in _WORD.findall(body) if len(token) >= 3}
    specific = {term for term in wanted if term not in _GENERIC_QUERY_TERMS}
    if specific and not _term_overlap(specific, body_terms):
        return False
    if not specific and wanted and not _term_overlap(wanted, body_terms):
        return False
    folded = body.casefold()
    ecosystem_markers = (
        "fabric.mod.json", "fabric api", "fabricmc", "minecraft mod",
        "mod for minecraft", "forge mod", "neoforge", "mods.toml",
        "architectury", "curseforge", "modrinth", "minecraftversion",
    )
    if not any(marker in folded for marker in ecosystem_markers):
        return False
    if any(marker in folded for marker in _GENERIC_REPOSITORY_MARKERS) and not any(
        marker in folded for marker in ("fabric.mod.json", "mods.toml", "architectury")
    ):
        return False
    return True


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "mmm-pre-design-source-rag",
    }
    token = _github_token()
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
    search_query = _repository_search_query(query)
    _emit_source_trace(
        "github_query_start",
        query=query,
        repository_search_query=search_query,
        authenticated=bool(_github_token()),
        max_repository_candidates=_MAX_REPOSITORIES_PER_QUERY,
    )

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
            search_requests += 1
            response = client.get(
                f"{_GITHUB_API}/search/repositories",
                params={
                    "q": search_query + " in:name,description,readme,topics",
                    "per_page": _MAX_REPOSITORIES_PER_QUERY,
                },
            )
            remaining = str(response.headers.get("x-ratelimit-remaining", ""))
            reset = str(response.headers.get("x-ratelimit-reset", ""))
            _emit_source_trace(
                "github_search_response",
                query=query,
                repository_search_query=search_query,
                http_status=response.status_code,
                rate_remaining=remaining,
                rate_reset=reset,
            )
            if response.status_code != 200:
                error = _status_error(response, operation="repository search")
                errors.append(error)
                status = "rate_limited" if response.status_code in {403, 429} else "unavailable"
                _emit_source_trace(
                    "github_search_failed",
                    query=query,
                    error=error,
                    provider_status=status,
                )
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
                error = f"github repository search JSON decode failed: {type(exc).__name__}: {exc}"
                errors.append(error)
                _emit_source_trace("github_search_decode_failed", query=query, error=error)
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
            _emit_source_trace(
                "github_search_candidates",
                query=query,
                candidate_count=len(repositories),
                total_count=(payload.get("total_count") if isinstance(payload, Mapping) else None),
                candidates=[str(item.get("full_name") or "") for item in repositories[:_MAX_REPOSITORIES_PER_QUERY]],
            )
            if not repositories:
                return {
                    "records": records,
                    "search_requests": search_requests,
                    "source_requests": source_requests,
                    "provider_status": "available",
                    "saturation_reason": "search_exhausted_no_repository",
                    "errors": errors,
                }

            for candidate_index, repository in enumerate(repositories[:_MAX_REPOSITORIES_PER_QUERY]):
                full_name = str(repository.get("full_name") or "").strip()
                if not full_name or "/" not in full_name:
                    _emit_source_trace(
                        "github_repository_skipped",
                        query=query,
                        candidate_index=candidate_index,
                        reason="invalid_full_name",
                    )
                    continue
                if not _repository_candidate_relevant(query, repository):
                    _emit_source_trace(
                        "github_repository_skipped",
                        query=query,
                        candidate_index=candidate_index,
                        repository=full_name,
                        reason="repository_not_minecraft_mod_query_relevant",
                    )
                    continue
                _emit_source_trace(
                    "github_repository_selected_for_body",
                    query=query,
                    candidate_index=candidate_index,
                    repository=full_name,
                )
                source_requests += 1
                readme = client.get(f"{_GITHUB_API}/repos/{full_name}/readme")
                _emit_source_trace(
                    "github_readme_response",
                    query=query,
                    repository=full_name,
                    http_status=readme.status_code,
                    rate_remaining=str(readme.headers.get("x-ratelimit-remaining", "")),
                    rate_reset=str(readme.headers.get("x-ratelimit-reset", "")),
                )
                if readme.status_code != 200:
                    if readme.status_code not in {404}:
                        errors.append(_status_error(readme, operation=f"README fetch {full_name}"))
                    _emit_source_trace(
                        "github_readme_rejected",
                        query=query,
                        repository=full_name,
                        reason="http_not_200",
                        http_status=readme.status_code,
                    )
                    continue
                try:
                    readme_payload = readme.json()
                except Exception as exc:
                    error = f"github README JSON decode failed for {full_name}: {type(exc).__name__}: {exc}"
                    errors.append(error)
                    _emit_source_trace("github_readme_rejected", query=query, repository=full_name, reason="json_decode_failed", error=error)
                    continue
                if not isinstance(readme_payload, Mapping):
                    _emit_source_trace("github_readme_rejected", query=query, repository=full_name, reason="payload_not_mapping")
                    continue
                body = _decode_readme(readme_payload)
                if not body:
                    _emit_source_trace("github_readme_rejected", query=query, repository=full_name, reason="empty_or_too_short_body")
                    continue
                if not _body_relevant(query, body):
                    _emit_source_trace(
                        "github_readme_rejected",
                        query=query,
                        repository=full_name,
                        reason="body_not_query_relevant",
                        body_chars=len(body),
                        body_sha256="sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    )
                    continue
                locator = str(readme_payload.get("html_url") or repository.get("html_url") or "").strip()
                if not locator:
                    _emit_source_trace("github_readme_rejected", query=query, repository=full_name, reason="missing_locator")
                    continue
                digest = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
                record = {
                    "source_id": "github:" + full_name + ":" + str(readme_payload.get("sha") or digest.removeprefix("sha256:")),
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
                records.append(record)
                _emit_source_trace(
                    "github_source_body_admitted",
                    query=query,
                    repository=full_name,
                    source_id=record["source_id"],
                    locator=locator,
                    body_chars=len(body),
                    body_sha256=digest,
                )
                break
    except Exception as exc:
        error = f"github source acquisition failed: {type(exc).__name__}: {exc}"
        errors.append(error)
        _emit_source_trace("github_transport_failure", query=query, error=error)
        return {
            "records": records,
            "search_requests": search_requests,
            "source_requests": source_requests,
            "provider_status": "unavailable",
            "saturation_reason": "transport_failure",
            "errors": errors,
        }

    saturation = "source_body_retrieved" if records else "repositories_found_no_claim_bearing_source_body"
    _emit_source_trace(
        "github_query_complete",
        query=query,
        search_requests=search_requests,
        source_requests=source_requests,
        source_body_count=len(records),
        saturation_reason=saturation,
        errors=errors,
    )
    return {
        "records": records,
        "search_requests": search_requests,
        "source_requests": source_requests,
        "provider_status": "available",
        "saturation_reason": saturation,
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


def _planned_requirement_query_keys(payload: Mapping[str, Any]) -> list[str]:
    # Stable one-query-per-authored-requirement first pass.
    if str(payload.get("schema_version") or "") == "mmm/corrective-retrieval-request-v1":
        result: list[str] = []
        seen: set[str] = set()
        for domain in payload.get("domains", ()):
            if not isinstance(domain, Mapping):
                continue
            for query in _stable_queries(domain.get("queries")):
                key = query.casefold()
                if key not in seen:
                    seen.add(key)
                    result.append(query)
        return result

    result: list[str] = []
    seen: set[str] = set()
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
            queries = _stable_queries(raw.get("search_queries"))
            if not queries:
                continue
            query = queries[0]
            key = query.casefold()
            if key not in seen:
                seen.add(key)
                result.append(query)
    return result
def _fallback_query_keys(bundle: Mapping[str, Any], limit: int) -> set[str]:
    del limit
    all_queries: list[str] = []
    for domain in bundle.get("domains", ()) if isinstance(bundle.get("domains"), list) else ():
        if not isinstance(domain, Mapping):
            continue
        for row in domain.get("queries", ()) if isinstance(domain.get("queries"), list) else ():
            if isinstance(row, Mapping):
                query = _clean_query(row.get("query"))
                if query:
                    all_queries.append(query)
    return {query.casefold() for query in all_queries}


def _safe_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    for key in ("source_id", "content_sha256", "source_locator", "url"):
        value = str(record.get(key) or "").strip()
        if value:
            return key, value
    return "content", hashlib.sha256(
        str(record.get("content") or "").encode("utf-8")
    ).hexdigest()


def _repository_identity(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        repository = str(metadata.get("repository") or "").strip()
        if repository:
            return repository
    return str(record.get("title") or record.get("source_locator") or "").strip()


def _augment_bundle(payload: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(bundle)
    raw_domains = bundle.get("domains")
    if not isinstance(raw_domains, list):
        return result

    limit = _max_queries()
    planned = _planned_requirement_query_keys(payload)
    selection_origin = "approved_requirement_queries"
    if not planned:
        planned = sorted(_fallback_query_keys(bundle, limit))
        selection_origin = "bounded_fallback_queries"
    # The normal authored plan contributes one first-pass query per requirement.
    # Never silently starve later requirements because an unrelated global cap was hit.
    selected_order = list(dict.fromkeys(planned))
    selected = {query.casefold() for query in selected_order}
    provider_rate_limited = False
    _emit_source_trace(
        "external_query_plan",
        selection_origin=selection_origin,
        configured_limit=limit,
        authenticated=bool(_github_token()),
        selected_count=len(selected),
        selected_queries=selected_order,
    )

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
                if provider_rate_limited:
                    receipt = {
                        "records": [],
                        "search_requests": 0,
                        "source_requests": 0,
                        "provider_status": "rate_limited",
                        "saturation_reason": "skipped_after_provider_rate_limit",
                        "errors": ["github provider already reported rate limit in this pre-design pass"],
                    }
                    _emit_source_trace(
                        "external_query_skipped",
                        query=query,
                        reason="provider_already_rate_limited",
                    )
                else:
                    _emit_source_trace("external_query_selected", query=query, reason=selection_origin)
                    receipt = _retrieve_github_source_body(query)
                    if str(receipt.get("provider_status") or "").casefold() == "rate_limited":
                        provider_rate_limited = True
                external = row.get("external_rag")
                external_map = dict(external) if isinstance(external, Mapping) else {}
                existing_records_raw = external_map.get("records")
                existing_records = [
                    dict(record)
                    for record in (
                        existing_records_raw if isinstance(existing_records_raw, list) else ()
                    )
                    if isinstance(record, Mapping)
                ]
                records = list(existing_records)
                seen = {_record_identity(record) for record in records}
                added_records: list[dict[str, Any]] = []
                for raw_record in receipt.get("records", ()):
                    if not isinstance(raw_record, Mapping):
                        continue
                    record = dict(raw_record)
                    identity = _record_identity(record)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    records.append(record)
                    added_records.append(record)

                added_count = len(added_records)
                external_map["records"] = records
                existing_actual = _safe_count(external_map.get("actual_source_document_count"))
                existing_documents = _safe_count(external_map.get("document_count"))
                baseline_records = len(existing_records)
                external_map["actual_source_document_count"] = (
                    max(existing_actual, baseline_records) + added_count
                )
                external_map["document_count"] = (
                    max(existing_documents, baseline_records) + added_count
                )
                repositories = {
                    identity
                    for identity in (_repository_identity(record) for record in records)
                    if identity
                }
                external_map["source_repository_count"] = max(
                    _safe_count(external_map.get("source_repository_count")),
                    len(repositories),
                )
                raw_providers = external_map.get("providers")
                providers = _stable_queries(raw_providers)
                if "github" not in {provider.casefold() for provider in providers}:
                    providers.append("github")
                external_map["providers"] = providers
                if added_count:
                    external_map["status"] = "available"
                    external_map.setdefault("credentials_required", False)
                elif not str(external_map.get("status") or "").strip():
                    external_map["status"] = str(
                        receipt.get("provider_status") or "unavailable"
                    )
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
            else:
                _emit_source_trace(
                    "external_query_skipped",
                    query=query,
                    reason="not_selected_by_bounded_query_plan",
                )
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
