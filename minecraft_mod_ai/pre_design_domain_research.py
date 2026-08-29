from __future__ import annotations

"""Canonical host-owned pre-design domain research.

The pre-design phase must not turn raw evidence receipts into an apparently successful
research note without first extracting concrete, page-grounded claims.  This module
owns that phase contract directly: every evidence page is read through the bounded
page protocol, every surviving claim is bound to the exact host-issued root page ref,
and only provenance-valid claims may be checkpointed as complete.
"""

from collections.abc import Mapping
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
                # Procedures are only reusable when they carry the same host provenance.
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
    """Read all pages losslessly and checkpoint only host-provenance-valid research."""

    del trace_metadata
    domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
    pages = project_rag._read_evidence_pages(document)
    allowed_refs = frozenset(
        str(page.get("page_ref", "")).strip()
        for page in pages
        if str(page.get("page_ref", "")).strip()
    )
    legacy_key = project_rag._domain_checkpoint_key(
        router,
        prompt=prompt,
        domain=domain,
        document=document,
    )
    domain_key = project_rag._sha256(
        {
            "base_domain_key": legacy_key,
            "research_policy": "host-grounded-lossless-pages",
        }
    ).removeprefix("sha256:")

    with project_rag._domain_lock(domain_key):
        cached = project_rag._read_complete_manifest(
            agentic_module,
            domain_key,
            domain_id,
        )
        if isinstance(cached, Mapping):
            try:
                agentic_module._validate_sufficient_research(
                    cached,
                    allowed_refs=allowed_refs,
                )
            except agentic_module.SpecValidationError:
                cached = None
        if isinstance(cached, Mapping):
            return dict(cached)

        failures: list[dict[str, str]] = []
        page_results: list[tuple[str, list[dict[str, Any]]]] = []
        project_rag._emit_research_progress(
            "domain_start",
            domain_id=domain_id,
            page_count=len(pages),
            evidence_document=project_rag._prompt_document_receipt(document),
            evidence_pages_path=document.get("pages_path"),
            evidence_raw_path=document.get("raw_path"),
            checkpoint_dir=str(project_rag._checkpoint_dir(domain_key)),
            grounding_policy="host-grounded-lossless-pages",
        )

        for page_index, page in enumerate(pages):
            page_ref = str(page.get("page_ref", "")).strip()
            project_rag._emit_research_progress(
                "page_start",
                domain_id=domain_id,
                page_index=page_index + 1,
                page_count=len(pages),
                page_ref=page_ref,
            )
            notes = project_rag._read_page_losslessly(
                agentic_module,
                router,
                prompt=prompt,
                domain=domain,
                document=document,
                page=page,
                domain_key=domain_key,
                progress_label=f"domain {domain_id} page {page_index + 1}/{len(pages)}",
                failures=failures,
            )
            page_results.append((page_ref, notes))

        summary = _merge_page_notes(domain_id, page_results)
        claims = list(summary["claims"])
        catalog = project_rag._materialize_claim_catalog(domain_key, domain_id, claims)
        evidence_ledger = project_rag._materialize_evidence_ledger(
            domain_key,
            domain_id,
            pages,
        )
        failure_reasons: list[str] = []
        if failures:
            failure_reasons.append("bounded page extraction failure")
        if not claims:
            failure_reasons.append("zero grounded claims")

        status = "failed" if failure_reasons else "complete"
        note: dict[str, Any] = {
            **summary,
            "evidence_document": project_rag._prompt_document_receipt(document),
            "claim_catalog": catalog,
            "evidence_ledger": evidence_ledger,
            "checkpoint": {
                "schema_version": "mmm/research-domain-checkpoint",
                "request_sha256": "sha256:" + domain_key,
                "status": status,
                "manifest_path": str(project_rag._manifest_path(domain_key)),
                "checkpoint_dir": str(project_rag._checkpoint_dir(domain_key)),
            },
        }
        if failures:
            note["research_failures"] = list(failures)
            note["gaps"] = [
                *list(note.get("gaps", ())),
                *(f"{item['unit']}: {item['error']}" for item in failures),
            ]
        if failure_reasons:
            note["sufficient"] = False
            note["fixed_point"] = True
            note["failure_reasons"] = failure_reasons
        else:
            # This validation runs before the complete manifest is written.  A missing or
            # invented evidence ref can therefore never become a reusable checkpoint.
            agentic_module._validate_sufficient_research(
                note,
                allowed_refs=allowed_refs,
            )

        project_rag._write_manifest(
            domain_key,
            status=status,
            note=note,
            failures=failures,
        )
        if status != "complete":
            project_rag._emit_research_progress(
                "domain_failure",
                domain_id=domain_id,
                status=status,
                failure_reasons=failure_reasons,
                failures=failures,
                claim_catalog=catalog,
                evidence_ledger=evidence_ledger,
                note=note,
            )
            raise project_rag._BoundedResearchOutputError(
                "pre-design research failed closed for domain "
                f"{domain_id!r}: {'; '.join(failure_reasons)}"
            )

        project_rag._emit_research_progress(
            "domain_complete",
            domain_id=domain_id,
            status=status,
            claim_count=len(claims),
            procedure_count=len(note.get("procedures", ())),
            page_count=len(pages),
            failure_count=0,
            claim_catalog=catalog,
            evidence_ledger=evidence_ledger,
        )
        return note


__all__ = ["research_document_domain"]
