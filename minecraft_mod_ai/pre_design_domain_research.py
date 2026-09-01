from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Canonical pre-design research entrypoint. The previous corrective/page-gap state
# machine is intentionally not on the execution path.
from .small_model_predesign_research import research_document_domain


def _root_page_claims(notes: Sequence[Mapping[str, Any]], *, page_ref: str) -> list[dict[str, Any]]:
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


__all__ = ["research_document_domain", "_root_page_claims", "_merge_page_notes"]
