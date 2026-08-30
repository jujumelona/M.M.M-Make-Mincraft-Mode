from __future__ import annotations

"""Compatibility facade for the side-effect-free pre-design RAG quality stack."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .pre_design_rag_corrective import (
    _correction_queries as _correction_queries,
)
from .pre_design_rag_corrective import (
    _quality_research_document_domain as _quality_research_document_domain,
)
from .pre_design_rag_corrective import (
    _read_and_verify_document as _read_and_verify_document,
)
from .pre_design_rag_fusion import (
    _is_retrieval_query as _is_retrieval_query,
)
from .pre_design_rag_fusion import _record_content as _record_content
from .pre_design_rag_fusion import (
    fuse_grounded_domain_evidence as _fuse_grounded_domain_evidence,
)
from .pre_design_rag_support import _verify_page_claims as _verify_page_claims

_INSTALLED = False

_PROVENANCE_FIELDS = (
    "source_locator",
    "url",
    "path",
    "file_path",
    "document_id",
    "source_id",
    "source_key",
    "page_ref",
)
_INCOMPLETE_FLAGS = (
    "content_omitted",
    "source_content_omitted",
    "omitted",
    "content_truncated",
    "truncated",
    "text_truncated",
    "payload_truncated",
    "queue_truncated",
    "tree_truncated",
    "pagination_incomplete",
    "request_budget_exhausted",
    "round_limit_reached",
)
_BODY_RETRIEVAL_FLAGS = (
    "body_retrieved",
    "source_body_retrieved",
    "raw_retrieved",
    "blob_retrieved",
)
_FATAL_STATUS_VALUES = {
    "error",
    "failed",
    "forbidden",
    "rate_limited",
    "timeout",
    "timed_out",
    "unavailable",
}
_METADATA_ONLY_SOURCE_MARKERS = (
    "search_result",
    "metadata_only",
    "metadata-only",
    "snippet_only",
    "snippet-only",
)
_DISCOVERY_ONLY_SOURCE_TYPES = {"modrinth_project"}
_RELEVANCE_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
_RELEVANCE_STOPWORDS = {
    "api",
    "code",
    "example",
    "fabric",
    "forge",
    "implementation",
    "java",
    "minecraft",
    "mod",
    "mods",
    "project",
    "source",
}


def _stable_text_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for raw in value:
        text = " ".join(str(raw or "").split()).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _approved_requirement_query_obligations(
    domain: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Recover the frozen requirement->query plan for the live request domain.

    The central research brief intentionally carries a compact flat query list. During the
    guarded planning call the approved requirement catalog is still the semantic authority,
    so this boundary binds every flat query back to the requirement that authorized it.
    No requirement meaning is reconstructed from query text.
    """

    if str(domain.get("domain_id") or "") != "request":
        return ()
    raw_requirements = domain.get("requirements")
    authored_requirements = _stable_text_list(raw_requirements)
    if not authored_requirements:
        return ()
    prompt = authored_requirements[0]

    from . import authored_scope_research_contract as authored_scope

    catalog = authored_scope._active_catalog(prompt)
    if not isinstance(catalog, Mapping):
        return ()
    catalog_requirements = catalog.get("requirements")
    if not isinstance(catalog_requirements, list) or not catalog_requirements:
        return ()

    domain_queries = _stable_text_list(domain.get("queries"))
    domain_query_keys = {query.casefold() for query in domain_queries}
    obligations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in catalog_requirements:
        if not isinstance(raw, Mapping):
            continue
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id or requirement_id in seen_ids:
            raise ValueError(
                "approved requirement retrieval provenance contains a missing/duplicate "
                f"requirement id: {requirement_id!r}"
            )
        planned_queries = [
            query
            for query in _stable_text_list(raw.get("search_queries"))
            if _is_retrieval_query(query, raw_prompt=prompt)
        ]
        if not planned_queries:
            raise ValueError(
                "approved requirement has no valid retrieval queries: "
                f"{requirement_id}"
            )
        missing = [
            query for query in planned_queries if query.casefold() not in domain_query_keys
        ]
        if missing:
            raise ValueError(
                "pre-design retrieval query provenance drift for approved requirement "
                f"{requirement_id}: {missing}"
            )
        seen_ids.add(requirement_id)
        obligations.append(
            {
                "requirement_id": requirement_id,
                "queries": planned_queries,
            }
        )
    return tuple(obligations)


def _mapping_layers(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    layers: list[Mapping[str, Any]] = []
    queue: list[Mapping[str, Any]] = [value]
    seen: set[int] = set()
    while queue:
        layer = queue.pop(0)
        marker = id(layer)
        if marker in seen:
            continue
        seen.add(marker)
        layers.append(layer)
        for key in ("metadata", "retrieval", "github_retrieval", "external_rag"):
            nested = layer.get(key)
            if isinstance(nested, Mapping):
                queue.append(nested)
    return tuple(layers)


def _flag(value: Mapping[str, Any], name: str) -> bool:
    for layer in _mapping_layers(value):
        if name in layer:
            raw = layer.get(name)
            if isinstance(raw, str):
                return raw.strip().casefold() in {"1", "true", "yes", "on"}
            return bool(raw)
    return False


def _explicit_false(value: Mapping[str, Any], name: str) -> bool:
    for layer in _mapping_layers(value):
        if name not in layer:
            continue
        raw = layer.get(name)
        if isinstance(raw, str):
            return raw.strip().casefold() in {"0", "false", "no", "off"}
        return raw is False
    return False


def _failure_status(value: Mapping[str, Any]) -> str:
    for layer in _mapping_layers(value):
        for key in (
            "status",
            "provider_status",
            "github_provider_status",
            "retrieval_status",
        ):
            status = str(layer.get(key) or "").strip().casefold()
            if status in _FATAL_STATUS_VALUES:
                return f"{key}:{status}"
    return ""


def _retrieval_errors(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for layer in _mapping_layers(value):
        raw = layer.get("retrieval_errors")
        if raw is None:
            raw = layer.get("errors")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            errors.extend(str(item).strip() for item in raw if str(item).strip())
        elif isinstance(raw, str) and raw.strip():
            errors.append(raw.strip())
        for key in ("retrieval_error", "error"):
            text = str(layer.get(key) or "").strip()
            if text:
                errors.append(text)
    return errors


def _incomplete_reason(value: Mapping[str, Any]) -> str:
    for name in _INCOMPLETE_FLAGS:
        if _flag(value, name):
            return name
    for layer in _mapping_layers(value):
        saturation = str(layer.get("saturation_reason") or "").strip().casefold()
        if any(marker in saturation for marker in _INCOMPLETE_FLAGS):
            return f"saturation_reason:{saturation}"
    return ""


def _record_locator(record: Mapping[str, Any]) -> str:
    for layer in _mapping_layers(record):
        for field in _PROVENANCE_FIELDS:
            text = str(layer.get(field) or "").strip()
            if text:
                return text
    return ""


def _metadata_only_record(record: Mapping[str, Any]) -> bool:
    if _flag(record, "metadata_only") or _flag(record, "snippet_only"):
        return True
    source_types = [
        str(record.get(key) or "").strip().casefold()
        for key in ("source_type", "source_kind", "record_type")
        if str(record.get(key) or "").strip()
    ]
    if any(source_type in _DISCOVERY_ONLY_SOURCE_TYPES for source_type in source_types):
        return True
    joined = " ".join(source_types)
    return any(marker in joined for marker in _METADATA_ONLY_SOURCE_MARKERS)


def _source_body(record: Mapping[str, Any]) -> str:
    """Return only a verified source body; never promote snippets/excerpts/metadata."""

    if _metadata_only_record(record):
        return ""
    if _retrieval_errors(record) or _failure_status(record) or _incomplete_reason(record):
        return ""
    if any(_explicit_false(record, flag) for flag in _BODY_RETRIEVAL_FLAGS):
        return ""
    if not _record_locator(record):
        return ""
    return _record_content(record)


def _query_terms(query: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in _RELEVANCE_WORD.findall(query)
        if len(token) >= 3
    }
    specific = {token for token in tokens if token not in _RELEVANCE_STOPWORDS}
    return specific or tokens


def _body_relevant_to_query(query: str, body: str) -> bool:
    wanted = _query_terms(query)
    if not wanted:
        return bool(body.strip())
    body_tokens = {
        token.casefold()
        for token in _RELEVANCE_WORD.findall(body)
        if len(token) >= 3
    }
    return bool(wanted & body_tokens)


def _query_row_is_complete(row: Mapping[str, Any]) -> bool:
    return not (
        _retrieval_errors(row)
        or _failure_status(row)
        or _incomplete_reason(row)
    )


def _requirement_evidence_sufficiency(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> dict[str, Any] | None:
    obligations = _approved_requirement_query_obligations(domain)
    if not obligations:
        return None

    query_has_content: dict[str, bool] = {}
    query_receipts: dict[str, dict[str, Any]] = {}
    raw_rows = grounded.get("queries")
    rows = raw_rows if isinstance(raw_rows, list) else []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        query = " ".join(str(raw.get("query") or "").split()).strip()
        if not query:
            continue
        raw_records = raw.get("evidence_records")
        records = raw_records if isinstance(raw_records, list) else []
        row_complete = _query_row_is_complete(raw)
        usable_ids: list[str] = []
        rejected_count = 0
        for index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                rejected_count += 1
                continue
            body = _source_body(record) if row_complete else ""
            if not body or not _body_relevant_to_query(query, body):
                rejected_count += 1
                continue
            usable_ids.append(
                str(record.get("evidence_id") or record.get("source_id") or "")
                or f"record:{index}"
            )
        has_content = bool(usable_ids)
        key = query.casefold()
        query_has_content[key] = query_has_content.get(key, False) or has_content
        receipt = query_receipts.setdefault(
            key,
            {
                "query": query,
                "usable_source_body_ids": [],
                "rejected_record_count": 0,
                "complete_retrieval": False,
            },
        )
        receipt["complete_retrieval"] = bool(
            receipt["complete_retrieval"] or row_complete
        )
        receipt["rejected_record_count"] += rejected_count
        for evidence_id in usable_ids:
            if evidence_id not in receipt["usable_source_body_ids"]:
                receipt["usable_source_body_ids"].append(evidence_id)

    requirement_receipts: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for obligation in obligations:
        requirement_id = str(obligation["requirement_id"])
        queries = [str(query) for query in obligation["queries"]]
        queries_with_content = [
            query for query in queries if query_has_content.get(query.casefold(), False)
        ]
        sufficient = bool(queries_with_content)
        if not sufficient:
            unresolved.append(requirement_id)
        requirement_receipts.append(
            {
                "requirement_id": requirement_id,
                "queries": queries,
                "queries_with_content": queries_with_content,
                "sufficient": sufficient,
            }
        )

    receipt = {
        "schema_version": "mmm/pre-design-requirement-evidence-sufficiency-v1",
        "validation_version": 3,
        "authority": "approved_requirement_retrieval_plan",
        "evidence_validation": "verified_source_body",
        "required_requirement_count": len(requirement_receipts),
        "satisfied_requirement_count": len(requirement_receipts) - len(unresolved),
        "unresolved_requirement_ids": unresolved,
        "sufficient": not unresolved,
        "requirements": requirement_receipts,
        "query_evidence_receipts": list(query_receipts.values()),
    }
    if unresolved:
        raise ValueError(
            "pre-design evidence is insufficient for approved requirements: "
            + ", ".join(unresolved)
        )
    return receipt


def fuse_grounded_domain_evidence(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse evidence only after every approved authored requirement has real support."""

    requirement_sufficiency = _requirement_evidence_sufficiency(domain, grounded)
    result = _fuse_grounded_domain_evidence(domain, grounded)
    if requirement_sufficiency is not None:
        result["requirement_sufficiency"] = requirement_sufficiency
    return result


def install() -> None:
    """Compatibility no-op; canonical research owners call the stack directly."""

    global _INSTALLED
    _INSTALLED = True


__all__ = [
    "_approved_requirement_query_obligations",
    "_correction_queries",
    "_is_retrieval_query",
    "_quality_research_document_domain",
    "_read_and_verify_document",
    "_requirement_evidence_sufficiency",
    "_source_body",
    "_verify_page_claims",
    "fuse_grounded_domain_evidence",
    "install",
]
