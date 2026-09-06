from __future__ import annotations

"""Host-grounded pre-design research projection.

Retrieval and source materialization remain host-owned. Pre-design no longer asks the
planner model to summarize those pages: exact source excerpts are sufficient provenance
for later target and implementation work. A domain is sufficient only when at least one
claim-bearing materialized source excerpt exists.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .research_evidence_state import record_grounded_evidence

_STOP_TERMS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "minecraft",
        "fabric",
        "mod",
        "mods",
        "implementation",
        "system",
        "feature",
    }
)


def _tokens(value: Any) -> set[str]:
    folded = re.sub(r"[_./:+-]+", " ", str(value or "").casefold())
    return {
        token
        for token in re.findall(r"[a-z0-9]+|[가-힣]{2,}", folded)
        if len(token) > 1 and token not in _STOP_TERMS
    }


def _domain_terms(domain: Mapping[str, Any]) -> set[str]:
    values = [str(domain.get("objective") or "")]
    for key in ("requirements", "queries"):
        raw = domain.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            values.extend(str(item) for item in raw if str(item).strip())
    result: set[str] = set()
    for value in values:
        result.update(_tokens(value))
    return result


def _exact_excerpt(content: str, wanted: set[str]) -> tuple[str, int]:
    text = str(content or "")
    candidates = [
        chunk.strip()
        for chunk in re.split(r"(?:\r?\n){2,}|(?<=[.!?])\s+(?=[A-Z0-9가-힣])", text)
        if chunk.strip()
    ]
    if not candidates:
        return "", 0
    ranked = [
        (len(wanted & _tokens(chunk)), len(_tokens(chunk)), -index, chunk)
        for index, chunk in enumerate(candidates)
    ]
    score, _specificity, _order, selected = max(ranked)
    return selected, max(0, int(score))


def _source_unit(page: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(page.get("content") or "")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None
    if isinstance(value, Mapping) and str(value.get("content") or "").strip():
        return dict(value)
    return {
        "source_id": "",
        "source_type": "",
        "url": "",
        "title": "",
        "content_sha256": "",
        "content": raw,
    }


def _grounded_evidence_cards(
    project_rag: Any,
    document: Mapping[str, Any],
    domain: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build deterministic exact-excerpt evidence cards from materialized pages."""
    reader = getattr(project_rag, "_read_evidence_pages", None)
    if not callable(reader):
        return []
    try:
        pages = reader(document)
    except Exception:
        return []
    wanted = _domain_terms(domain)
    cards: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for raw in pages if isinstance(pages, Sequence) else ():
        if not isinstance(raw, Mapping):
            continue
        page_ref = str(raw.get("page_ref") or "").strip()
        unit = _source_unit(raw)
        source_content = str(unit.get("content") or "")
        if not page_ref or page_ref in seen_refs or not source_content.strip():
            continue
        excerpt, score = _exact_excerpt(source_content, wanted)
        if not excerpt:
            continue
        seen_refs.add(page_ref)
        cards.append(
            {
                "page_ref": page_ref,
                "source_id": str(unit.get("source_id") or ""),
                "source_type": str(unit.get("source_type") or ""),
                "source_url": str(unit.get("url") or ""),
                "source_title": str(unit.get("title") or ""),
                "source_content_sha256": str(unit.get("content_sha256") or ""),
                "exact_excerpt": excerpt,
                "domain_term_overlap": score,
                "verification": "host_exact_substring_from_materialized_source_page",
                "semantic_claim": False,
            }
        )
    return cards


def _claims_from_grounded_cards(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose host exact excerpts through the claim/ref transport contract."""
    claims: list[dict[str, Any]] = []
    for card in cards:
        page_ref = str(card.get("page_ref") or "").strip()
        excerpt = str(card.get("exact_excerpt") or "").strip()
        if not page_ref or not excerpt:
            continue
        claims.append(
            {
                "claim": excerpt,
                "evidence_refs": [page_ref],
                "grounding": "host_exact_substring_from_materialized_source_page",
                "semantic_claim": False,
            }
        )
    return claims


def research_document_domain(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return host-grounded evidence directly; never spend a model turn synthesizing it."""
    del agentic_module, router, trace_metadata
    cards = _grounded_evidence_cards(project_rag, document, domain)
    claims = _claims_from_grounded_cards(cards)
    reader = getattr(project_rag, "_read_evidence_pages", None)
    try:
        pages = reader(document) if callable(reader) else ()
    except Exception:
        pages = ()
    source_body_count = sum(
        1
        for page in pages
        if isinstance(page, Mapping) and str(page.get("content") or "").strip()
    )
    sufficient = bool(claims)
    status = "supported" if sufficient else "no_relevant_external_evidence"
    failure_reason = "" if sufficient else "no_claim_bearing_source_body"
    note = {
        "domain_id": str(domain.get("domain_id") or "unknown"),
        "research_mode": "advisory_predesign",
        "claims": claims,
        "gaps": [] if sufficient else [
            "No claim-bearing materialized source body was available for this required research domain."
        ],
        "next_queries": [],
        "procedures": [],
        "sufficient": sufficient,
        "fixed_point": False,
        "checkpoint": {
            "status": "complete" if sufficient else "blocked",
            "reason": failure_reason,
        },
        "research_failures": [] if sufficient else [failure_reason],
        "source_body_count": source_body_count,
        "model_called": False,
        "model_grounded_claim_count": 0,
        "host_grounded_evidence_card_count": len(cards),
        "grounded_evidence_cards": cards,
        "research_evidence_status": status,
        "evidence_extraction_status": (
            "host_exact_evidence_available" if sufficient else failure_reason
        ),
        "page_local_diagnostics": [
            "model_synthesis_skipped_host_evidence_preserved"
        ],
    }
    record_grounded_evidence(
        prompt,
        source_body_count=source_body_count,
        evidence_card_count=len(cards),
    )
    return note


def _root_page_claims(
    notes: Sequence[Mapping[str, Any]], *, page_ref: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for note in notes:
        for raw in note.get("claims", ()) if isinstance(note, Mapping) else ():
            if not isinstance(raw, Mapping):
                continue
            claim = dict(raw)
            if not str(claim.get("claim") or "").strip():
                continue
            claim["evidence_refs"] = [page_ref]
            result.append(claim)
    return result


def _merge_page_notes(
    domain_id: str,
    page_notes: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    gaps: list[str] = []
    next_queries: list[str] = []
    procedures: list[Any] = []
    for page_ref, notes in page_notes:
        claims.extend(_root_page_claims(notes, page_ref=page_ref))
        for note in notes:
            if not isinstance(note, Mapping):
                continue
            for value in note.get("gaps", ()):
                text = str(value).strip()
                if text and text not in gaps:
                    gaps.append(text)
            for value in note.get("next_queries", ()):
                text = str(value).strip()
                if text and text not in next_queries:
                    next_queries.append(text)
            for value in note.get("procedures", ()):
                if value not in procedures:
                    procedures.append(value)
    sufficient = bool(claims)
    if not sufficient and not gaps:
        gaps.append("No evidence-backed design claim was extracted from the host-issued pages.")
    return {
        "domain_id": domain_id,
        "claims": claims,
        "gaps": [] if sufficient else gaps,
        "next_queries": next_queries,
        "procedures": procedures,
        "sufficient": sufficient,
        "fixed_point": False,
    }


__all__ = [
    "research_document_domain",
    "_grounded_evidence_cards",
    "_claims_from_grounded_cards",
    "_root_page_claims",
    "_merge_page_notes",
]
