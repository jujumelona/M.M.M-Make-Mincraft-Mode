from __future__ import annotations

"""Compatibility facade for the side-effect-free pre-design RAG quality stack."""

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

    The central research brief intentionally carries a compact flat query list.  During the
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


def _requirement_evidence_sufficiency(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> dict[str, Any] | None:
    obligations = _approved_requirement_query_obligations(domain)
    if not obligations:
        return None

    query_has_content: dict[str, bool] = {}
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
        has_content = any(
            isinstance(record, Mapping) and bool(_record_content(record))
            for record in records
        )
        key = query.casefold()
        query_has_content[key] = query_has_content.get(key, False) or has_content

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
        "authority": "approved_requirement_retrieval_plan",
        "required_requirement_count": len(requirement_receipts),
        "satisfied_requirement_count": len(requirement_receipts) - len(unresolved),
        "unresolved_requirement_ids": unresolved,
        "sufficient": not unresolved,
        "requirements": requirement_receipts,
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
    "_verify_page_claims",
    "fuse_grounded_domain_evidence",
    "install",
]
