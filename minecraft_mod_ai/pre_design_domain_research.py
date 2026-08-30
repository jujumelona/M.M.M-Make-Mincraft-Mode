from __future__ import annotations

"""Canonical host-owned pre-design domain research.

This module is the direct owner of the pre-design domain research path. Retrieval-quality
work is called from here rather than installed as a late runtime monkey-patch: initial
multi-query evidence is fused before reading, corrective retrieval remains bounded and
claim-support verified, and the final validation view contains every exact host page used
by the accepted claims.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _root_page_claims(
    notes: list[dict[str, Any]],
    *,
    page_ref: str,
) -> list[dict[str, Any]]:
    """Bind extracted claims to the root host page, never to model-invented refs."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for note in notes:
        if not isinstance(note, Mapping):
            continue
        for raw_claim in note.get("claims", ()):
            if not isinstance(raw_claim, Mapping):
                continue
            text = str(raw_claim.get("claim", "")).strip()
            if not text or text in seen or not page_ref:
                continue
            seen.add(text)
            result.append({"claim": text, "evidence_refs": [page_ref]})
    return result


def _stable_text(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, (list, tuple)):
        return result
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _merge_page_notes(
    domain_id: str,
    page_results: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Legacy-compatible page merge helper retained for direct unit callers."""

    claims: list[dict[str, Any]] = []
    gaps: list[str] = []
    next_queries: list[str] = []
    procedures: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()
    seen_procedures: set[str] = set()

    for root_ref, notes in page_results:
        for claim in _root_page_claims(notes, page_ref=root_ref):
            key = (claim["claim"], root_ref)
            if key in seen_claims:
                continue
            seen_claims.add(key)
            claims.append(claim)
        for note in notes:
            if not isinstance(note, Mapping):
                continue
            for gap in _stable_text(note.get("gaps")):
                if gap not in gaps:
                    gaps.append(gap)
            for query in _stable_text(note.get("next_queries")):
                if query not in next_queries:
                    next_queries.append(query)
            for raw_procedure in note.get("procedures", ()):
                if not isinstance(raw_procedure, Mapping):
                    continue
                procedure = dict(raw_procedure)
                procedure["evidence_refs"] = [root_ref]
                key = repr(sorted(procedure.items(), key=lambda item: str(item[0])))
                if key in seen_procedures:
                    continue
                seen_procedures.add(key)
                procedures.append(procedure)

    sufficient = bool(claims)
    if not sufficient:
        gaps.append(
            "No evidence-backed design-relevant claim was extracted from the host-owned "
            "pre-design evidence pages."
        )
    return {
        "domain_id": domain_id,
        "claims": claims,
        "gaps": gaps,
        "next_queries": next_queries,
        "procedures": procedures,
        "sufficient": sufficient,
    }


def _load_materialized_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(document.get("raw_path", ""))).expanduser()
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Pre-design materialized evidence is missing: {raw_path}"
        )
    value = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(
            f"Pre-design materialized evidence root must be an object: {raw_path}"
        )
    return value


def _fuse_initial_document(
    project_rag: Any,
    *,
    domain_id: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse all initial rewritten-query/provider evidence before the model reads pages."""

    from .pre_design_rag_quality_contract import fuse_grounded_domain_evidence

    evidence = _load_materialized_evidence(document)
    grounded = evidence.get("grounded_rag")
    if not isinstance(grounded, Mapping):
        raise ValueError(
            f"Pre-design evidence for domain {domain_id!r} has no grounded_rag object."
        )
    fused = fuse_grounded_domain_evidence(domain, grounded)
    evidence["grounded_rag"] = fused
    return project_rag._materialize_domain_evidence_document(domain_id, evidence)


def _combined_validation_document(
    project_rag: Any,
    *,
    domain_id: str,
    note: Mapping[str, Any],
    working_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a host-only validation page set spanning initial and corrective evidence.

    The accepted claims retain their original page refs. This document is not new evidence;
    it is a deterministic validation projection of the already-written evidence ledger so
    the pipeline's canonical grounding validator can verify corrective-page refs too.
    """

    ledger = note.get("evidence_ledger")
    ledger_path = Path(
        str(ledger.get("path", "")) if isinstance(ledger, Mapping) else ""
    ).expanduser()
    if not ledger_path.is_file():
        raise FileNotFoundError(
            f"Pre-design quality evidence ledger is missing: {ledger_path}"
        )

    records: list[dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Invalid pre-design evidence ledger row in {ledger_path}"
                )
            page_ref = str(raw.get("page_ref", "")).strip()
            content = str(raw.get("content", ""))
            if not page_ref:
                raise ValueError(
                    f"Pre-design evidence ledger row has no page_ref in {ledger_path}"
                )
            records.append(
                {
                    "schema_version": "mmm/research-evidence-page-v1",
                    "domain_id": domain_id,
                    "unit_id": str(raw.get("unit_id", "")),
                    "part_index": raw.get("part_index"),
                    "part_count": raw.get("part_count"),
                    "content": content,
                    "page_ref": page_ref,
                }
            )
    if not records:
        raise ValueError(
            f"Pre-design quality evidence ledger is empty: {ledger_path}"
        )

    page_count = len(records)
    for page_index, record in enumerate(records):
        record["page_index"] = page_index
        record["page_count"] = page_count

    checkpoint = note.get("checkpoint")
    checkpoint_dir = Path(
        str(checkpoint.get("checkpoint_dir", ""))
        if isinstance(checkpoint, Mapping)
        else ""
    ).expanduser()
    if not str(checkpoint_dir):
        raise ValueError("Pre-design quality note has no checkpoint directory.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pages_path = checkpoint_dir / "combined-validation-pages.jsonl"
    pages_text = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )
    project_rag._atomic_write_text(pages_path, pages_text)

    result = dict(working_document)
    result.update(
        {
            "schema_version": "mmm/research-evidence-validation-view-v1",
            "domain_id": domain_id,
            "pages_path": str(pages_path),
            "page_count": page_count,
            "page_chars": int(
                working_document.get("page_chars")
                or getattr(project_rag, "_EVIDENCE_PAGE_CHARS", 1800)
            ),
            "page_bytes": int(
                working_document.get("page_bytes")
                or getattr(project_rag, "_EVIDENCE_PAGE_CHARS", 1800)
            ),
            "model_projection": "quality_evidence_ledger_validation_view",
        }
    )
    return result


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
    """Run fused, corrective, support-verified research without runtime rebinding."""

    from . import pre_design_research_pipeline as pipeline
    from .pre_design_rag_quality_contract import _quality_research_document_domain

    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    working_document = _fuse_initial_document(
        project_rag,
        domain_id=domain_id,
        domain=domain,
        document=document,
    )
    note = _quality_research_document_domain(
        pipeline,
        agentic_module,
        project_rag,
        router,
        prompt=prompt,
        domain=domain,
        document=working_document,
        trace_metadata=trace_metadata,
    )
    validation_document = _combined_validation_document(
        project_rag,
        domain_id=domain_id,
        note=note,
        working_document=working_document,
    )

    if isinstance(document, dict):
        document.clear()
        document.update(validation_document)
    return note


__all__ = ["research_document_domain"]
