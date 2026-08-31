from __future__ import annotations

"""Bounded corrective retrieval loop for pre-design research."""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .pre_design_rag_fusion import (
    _is_retrieval_query,
    _record_content,
    _stable_text,
    fuse_grounded_domain_evidence,
)
from .pre_design_rag_support import (
    _claim_candidates,
    _merge_verified_notes,
    _verify_page_claims,
)

_DEFAULT_CORRECTIVE_ROUNDS = 2
_MAX_CORRECTIVE_ROUNDS = 4
_QUALITY_SCHEMA = "mmm/pre-design-rag-quality-v3"
_VERIFIED_FIXED_POINT = "verified_claims_sufficient"


def _corrective_round_limit() -> int:
    try:
        value = int(
            os.environ.get(
                "MMM_PREDESIGN_CORRECTIVE_ROUNDS",
                str(_DEFAULT_CORRECTIVE_ROUNDS),
            ).strip()
        )
    except ValueError:
        value = _DEFAULT_CORRECTIVE_ROUNDS
    return max(0, min(value, _MAX_CORRECTIVE_ROUNDS))


def _correction_queries(
    values: Any,
    *,
    seen: set[str],
    raw_prompt: str,
    limit: int = 4,
) -> list[str]:
    result: list[str] = []
    for query in _stable_text(values):
        key = query.casefold()
        if key in seen or not _is_retrieval_query(query, raw_prompt=raw_prompt):
            continue
        seen.add(key)
        result.append(query)
        if len(result) >= limit:
            break
    return result


def _generate_gap_queries(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    domain: Mapping[str, Any],
    gaps: Sequence[str],
    prior_queries: Sequence[str],
    seen: set[str],
    raw_prompt: str,
    progress_label: str,
) -> list[str]:
    # Queries are the executable contract. Small diagnostic fields such as
    # ``sufficient``/``gaps`` are harmless model annotations and must not discard an
    # otherwise valid corrective search plan before the parser can extract it.
    query_array_schema = {
        "type": "array",
        "minItems": 1,
        "maxItems": 4,
        "items": {"type": "string", "minLength": 4, "maxLength": 180},
        "uniqueItems": True,
    }
    schema = {
        "type": "object",
        "properties": {
            "queries": query_array_schema,
            # Qwen commonly emits this semantically equivalent key. Accept it at the
            # host parser boundary instead of discarding a valid retrieval plan before
            # any external source request can run.
            "search_queries": query_array_schema,
        },
        "additionalProperties": True,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Write 1-4 concise English search queries that retrieve the missing "
                "evidence for the already-approved Minecraft-mod requirement. Do not "
                "change the requirement, select a donor, copy the raw request, or "
                "repeat an already searched query."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "domain_id": str(domain.get("domain_id") or ""),
                    "objective": str(domain.get("objective") or ""),
                    "requirements": list(domain.get("requirements") or ()),
                    "evidence_gap": list(gaps)[-8:],
                    "already_searched": list(prior_queries)[-12:],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]

    def parse(raw: str) -> list[str]:
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise agentic_module.SpecValidationError(
                f"corrective query planner returned invalid JSON: {exc}"
            ) from exc
        queries = value.get("queries") if isinstance(value, Mapping) else None
        if not isinstance(queries, list) and isinstance(value, Mapping):
            queries = value.get("search_queries")
        if not isinstance(queries, list):
            raise agentic_module.SpecValidationError(
                "corrective query planner omitted queries/search_queries"
            )
        result = _correction_queries(
            queries,
            seen=seen,
            raw_prompt=raw_prompt,
        )
        if not result:
            raise agentic_module.SpecValidationError(
                "corrective query planner returned no new executable query"
            )
        return result

    return project_rag._generate_bounded(
        agentic_module,
        router,
        messages=messages,
        response_schema=schema,
        parser=parse,
        progress_label=progress_label + " corrective-query",
    )


def _read_and_verify_document(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    domain_key: str,
    failures: list[dict[str, str]],
    round_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    pages = project_rag._read_evidence_pages(document)
    results: list[dict[str, Any]] = []
    rejected_count = 0
    domain_id = str(domain.get("domain_id") or "").strip() or "unknown"

    for page_index, page in enumerate(pages):
        page_ref = str(page.get("page_ref") or "").strip()
        notes = project_rag._read_page_losslessly(
            agentic_module,
            router,
            prompt=prompt,
            domain=domain,
            document=document,
            page=page,
            domain_key=domain_key,
            progress_label=(
                f"domain {domain_id} round {round_index} "
                f"page {page_index + 1}/{len(pages)}"
            ),
            failures=failures,
        )
        candidates = _claim_candidates(
            [note for note in notes if isinstance(note, Mapping)]
        )
        try:
            verified, rejected = _verify_page_claims(
                agentic_module,
                project_rag,
                router,
                domain_id=domain_id,
                page=page,
                claims=candidates,
                progress_label=(
                    f"domain {domain_id} round {round_index} "
                    f"page {page_index + 1}/{len(pages)}"
                ),
            )
        except Exception as exc:
            failures.append(
                {
                    "unit": f"support:{round_index}:{page_index}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            verified, rejected = [], candidates

        rejected_count += len(rejected)
        gaps: list[str] = []
        next_queries: list[str] = []
        procedures: list[dict[str, Any]] = []
        for note in notes:
            if not isinstance(note, Mapping):
                continue
            for value in _stable_text(note.get("gaps")):
                if value not in gaps:
                    gaps.append(value)
            for value in _stable_text(note.get("next_queries")):
                if value not in next_queries:
                    next_queries.append(value)
            procedures.extend(
                dict(value)
                for value in note.get("procedures", ())
                if isinstance(value, Mapping)
            )
        gaps.extend(
            "Claim rejected by page-support verification; retrieve stronger evidence: "
            + claim
            for claim in rejected
        )
        results.append(
            {
                "_host_page_ref": page_ref,
                "domain_id": domain_id,
                "claims": verified,
                "gaps": gaps,
                "next_queries": next_queries,
                "procedures": procedures,
                "sufficient": bool(verified) and not gaps,
            }
        )
    return pages, results, rejected_count


def _quality_research_document_domain(
    pipeline_module: Any,
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    prompt: str,
    domain: Mapping[str, Any],
    document: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Retrieve -> assess -> correct until supported evidence or a real fixed point."""

    del trace_metadata
    domain_id = str(domain.get("domain_id") or "").strip() or "unknown"
    base_key = project_rag._domain_checkpoint_key(
        router,
        prompt=prompt,
        domain=domain,
        document=document,
    )
    domain_key = project_rag._sha256(
        {
            "base_domain_key": base_key,
            "research_policy": "corrective-fusion-claim-support-v3",
        }
    ).removeprefix("sha256:")

    with project_rag._domain_lock(domain_key):
        cached = project_rag._read_complete_manifest(
            agentic_module,
            domain_key,
            domain_id,
        )
        cached_quality = cached.get("quality_contract") if isinstance(cached, Mapping) else None
        if (
            isinstance(cached, Mapping)
            and isinstance(cached_quality, Mapping)
            and cached_quality.get("schema_version") == _QUALITY_SCHEMA
            and cached_quality.get("fixed_point_reason") == _VERIFIED_FIXED_POINT
            and cached.get("sufficient") is True
        ):
            refs = frozenset(
                str(ref)
                for ref in cached.get("evidence_page_refs", ())
                if str(ref).strip()
            )
            if refs:
                try:
                    agentic_module._validate_sufficient_research(
                        cached,
                        allowed_refs=refs,
                    )
                    return dict(cached)
                except agentic_module.SpecValidationError:
                    pass

        failures: list[dict[str, str]] = []
        documents = [dict(document)]
        all_pages: list[dict[str, Any]] = []
        all_notes: list[dict[str, Any]] = []
        seen_documents = {str(document.get("document_sha256") or "")}
        searched = [
            str(query).strip()
            for query in domain.get("queries", ())
            if str(query).strip()
        ]
        seen_queries = {query.casefold() for query in searched}
        history: list[dict[str, Any]] = []
        rejected_total = 0
        fixed_point = ""
        max_rounds = _corrective_round_limit()
        round_index = 0
        active_summary = _merge_verified_notes(domain_id, [])

        while round_index <= max_rounds:
            pages, notes, rejected = _read_and_verify_document(
                agentic_module,
                project_rag,
                router,
                prompt=prompt,
                domain=domain,
                document=documents[-1],
                domain_key=domain_key,
                failures=failures,
                round_index=round_index,
            )
            all_pages.extend(pages)
            all_notes.extend(notes)
            rejected_total += rejected
            active_summary = _merge_verified_notes(domain_id, notes)
            active_gaps = list(active_summary.get("gaps") or ())
            unseen = _correction_queries(
                active_summary.get("next_queries"),
                seen=seen_queries,
                raw_prompt=prompt,
            )

            if failures:
                fixed_point = "bounded_extraction_or_support_verification_failure"
                break
            if active_summary.get("claims") and not active_gaps:
                fixed_point = _VERIFIED_FIXED_POINT
                break
            if round_index >= max_rounds:
                fixed_point = "corrective_round_limit_reached"
                break
            if not unseen and active_gaps:
                try:
                    unseen = _generate_gap_queries(
                        agentic_module,
                        project_rag,
                        router,
                        domain=domain,
                        gaps=active_gaps,
                        prior_queries=searched,
                        seen=seen_queries,
                        raw_prompt=prompt,
                        progress_label=f"domain {domain_id} round {round_index}",
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "unit": f"corrective-query:{round_index}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    unseen = []
            if not unseen:
                fixed_point = (
                    "unresolved_evidence_gaps"
                    if active_gaps
                    else "no_support_verified_claims"
                )
                break

            searched.extend(unseen)
            correction_domain = dict(domain)
            correction_domain["queries"] = unseen
            bundle = project_rag._forced_rag_bundle(
                router,
                {
                    "domains": [correction_domain],
                    "schema_version": "mmm/corrective-retrieval-request-v1",
                },
            )
            grounded = pipeline_module._grounded_domain_evidence(
                agentic_module,
                domain_id,
                bundle,
            )
            fused = fuse_grounded_domain_evidence(domain, grounded)
            records = [
                record
                for row in fused.get("queries", ())
                if isinstance(row, Mapping)
                for record in row.get("evidence_records", ())
                if isinstance(record, Mapping) and _record_content(record)
            ]
            history.append(
                {
                    "round": round_index + 1,
                    "queries": unseen,
                    "unique_content_records": len(records),
                    "fusion": dict(fused.get("fusion") or {}),
                }
            )
            if not records:
                fixed_point = "corrective_retrieval_returned_no_claim_bearing_content"
                break

            next_document = project_rag._materialize_domain_evidence_document(
                domain_id,
                {"grounded_rag": fused},
            )
            digest = str(next_document.get("document_sha256") or "")
            if not digest or digest in seen_documents:
                fixed_point = "corrective_retrieval_repeated_existing_evidence"
                break
            seen_documents.add(digest)
            documents.append(dict(next_document))
            round_index += 1

        accumulated = _merge_verified_notes(domain_id, all_notes)
        summary = dict(accumulated)
        # Historical gaps are diagnostics, not live obligations. Terminal sufficiency is
        # decided from the latest evidence round so a gap that was actually resolved does
        # not remain forever, while a current unresolved gap can never be hidden by an old
        # verified claim.
        summary["gaps"] = list(active_summary.get("gaps") or ())
        summary["next_queries"] = list(active_summary.get("next_queries") or ())
        claims = list(summary["claims"])
        catalog = project_rag._materialize_claim_catalog(
            domain_key,
            domain_id,
            claims,
        )
        ledger = project_rag._materialize_evidence_ledger(
            domain_key,
            domain_id,
            all_pages,
        )
        page_refs = [
            str(page.get("page_ref") or "").strip()
            for page in all_pages
            if str(page.get("page_ref") or "").strip()
        ]
        reasons: list[str] = []
        if failures:
            reasons.append("bounded extraction/support verification failure")
        if not claims:
            reasons.append("zero support-verified grounded claims")
        if fixed_point != _VERIFIED_FIXED_POINT:
            reasons.append(
                "corrective retrieval did not reach verified sufficiency: "
                + (fixed_point or "no_terminal_state")
            )
        if summary.get("gaps"):
            reasons.append("unresolved evidence gaps remain")
        status = "failed" if reasons else "complete"

        note: dict[str, Any] = {
            **summary,
            "evidence_document": project_rag._prompt_document_receipt(document),
            "corrective_evidence_documents": [
                project_rag._prompt_document_receipt(item)
                for item in documents[1:]
            ],
            "evidence_page_refs": page_refs,
            "claim_catalog": catalog,
            "evidence_ledger": ledger,
            "quality_contract": {
                "schema_version": _QUALITY_SCHEMA,
                "fusion": "verified_source_body+exact_content_dedupe+query_rank+rrf",
                "corrective_retrieval": True,
                "claim_support": "model_entailment+host_exact_quote",
                "corrective_round_limit": max_rounds,
                "corrective_rounds_executed": len(history),
                "correction_history": history,
                "rejected_claim_count": rejected_total,
                "fixed_point_reason": fixed_point,
                "active_gap_count": len(summary.get("gaps") or ()),
                "donor_selection_performed": False,
                "runtime_rebinding": False,
            },
            "checkpoint": {
                "schema_version": "mmm/research-domain-checkpoint-v5",
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
        if reasons:
            note["sufficient"] = False
            note["fixed_point"] = True
            note["failure_reasons"] = reasons
        else:
            note["sufficient"] = True
            note["fixed_point"] = False
            agentic_module._validate_sufficient_research(
                note,
                allowed_refs=frozenset(page_refs),
            )

        project_rag._write_manifest(
            domain_key,
            status=status,
            note=note,
            failures=failures,
        )
        if status != "complete":
            raise project_rag._BoundedResearchOutputError(
                "pre-design corrective research failed closed for domain "
                f"{domain_id!r}: {'; '.join(reasons)}; fixed_point={fixed_point}"
            )
        return note


__all__ = [
    "_correction_queries",
    "_quality_research_document_domain",
    "_read_and_verify_document",
]
