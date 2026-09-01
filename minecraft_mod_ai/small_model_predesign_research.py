from __future__ import annotations

# Small-model-safe, host-owned pre-design research.
# User-authored gameplay requirements are already authoritative. External RAG is
# implementation evidence, never permission to design the requested feature.

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PROTOCOL = "mmm/small-model-predesign-evidence-v3"
_STOP = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
    "minecraft", "fabric", "mod", "mods", "mode", "requested", "user",
    "implementation", "system", "game", "feature",
}


def _terms(values: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for token in re.findall(r"[A-Za-z0-9_+.#/-]+|[가-힣]{2,}", str(value)):
            folded = token.casefold()
            if len(folded) >= 3 and folded not in _STOP:
                result.add(folded)
    return result


def _page_score(page: Mapping[str, Any], domain: Mapping[str, Any]) -> int:
    wanted = _terms(
        [
            str(domain.get("objective") or ""),
            *(str(x) for x in domain.get("queries", ()) if str(x).strip()),
        ]
    )
    if not wanted:
        return 0
    have = _terms([str(page.get("content") or "")])
    return len(wanted & have)


def _candidate_pages(
    pages: Sequence[Mapping[str, Any]], domain: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Order every page by relevance without dropping zero-score evidence."""

    scored = [
        (max(0, _page_score(page, domain)), index, dict(page))
        for index, page in enumerate(pages)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [page for _score, _index, page in scored]


def _exact_span(content: str, proposed: str) -> str:
    quote = str(proposed or "").strip().strip('"').strip("'")
    if not quote:
        return ""
    if quote in content:
        return quote
    words = [piece for piece in re.split(r"\s+", quote) if piece]
    if not words:
        return ""
    pattern = r"\s+".join(re.escape(piece) for piece in words)
    match = re.search(pattern, content, flags=re.MULTILINE)
    return match.group(0) if match else ""


def _batch_messages(
    domain: Mapping[str, Any], pages: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    sources: list[str] = []
    for page in pages:
        page_ref = str(page.get("page_ref") or "").strip()
        content = str(page.get("content") or "")
        if not page_ref or not content.strip():
            continue
        sources.append(
            f"SOURCE PAGE_REF={page_ref}\n{content}\nEND SOURCE PAGE_REF={page_ref}"
        )
    return [
        {
            "role": "system",
            "content": (
                "Read only the tagged SOURCES. Extract every directly useful implementation "
                "fact supported by an exact source span. Output one line per fact exactly as "
                "EVIDENCE<TAB>PAGE_REF<TAB>EXACT_QUOTE<TAB>IMPLEMENTATION_INSIGHT. "
                "PAGE_REF must name the source containing EXACT_QUOTE, and EXACT_QUOTE must "
                "be copied from that source rather than paraphrased. If no useful supported "
                "fact exists output only NONE. No JSON, Markdown, code fences, analysis, "
                "headings, sufficiency flags, search queries, or extra prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "OBJECTIVE\n"
                + str(domain.get("objective") or "")
                + "\n\n"
                + "\n\n".join(sources)
            ),
        },
    ]


def _planner_output_reserve(router: Any) -> int:
    try:
        config = router.registry.role(router.profile, "planner")
        return max(0, int(getattr(config, "max_new_tokens", 0) or 0))
    except Exception:
        return 0


def _live_accounting(router: Any, messages: Sequence[Mapping[str, Any]]) -> Any | None:
    counter = getattr(router, "input_context_accounting", None)
    if not callable(counter):
        return None
    return counter(
        "planner",
        messages,
        response_format="text",
        response_schema=None,
        tool_stage="research",
        enable_tools=False,
    )


def _capacity_batches(
    router: Any,
    *,
    domain: Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Pack all pages by live tokenizer/context capacity, never a fixed page count."""

    ordered = [
        dict(page)
        for page in pages
        if str(page.get("page_ref") or "").strip()
        and str(page.get("content") or "").strip()
    ]
    if not ordered:
        return [], []

    diagnostics: list[str] = []
    reserve = _planner_output_reserve(router)
    try:
        probe = _live_accounting(router, _batch_messages(domain, ordered[:1]))
    except Exception as exc:
        diagnostics.append(f"exact_input_accounting_failure:{type(exc).__name__}:{exc}")
        probe = None
    if probe is None:
        diagnostics.append("exact_input_accounting_unavailable;all_pages_kept_in_one_batch")
        return [ordered], diagnostics

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for index, page in enumerate(ordered):
        trial = [*current, page]
        try:
            accounting = _live_accounting(router, _batch_messages(domain, trial))
        except Exception as exc:
            diagnostics.append(f"exact_input_accounting_failure:{type(exc).__name__}:{exc}")
            accounting = None
        if accounting is None:
            if current:
                batches.append(current)
            remaining = ordered[index:]
            if remaining:
                batches.append(remaining)
            diagnostics.append("exact_input_accounting_lost;remaining_pages_kept_together")
            return batches, diagnostics

        input_tokens = int(getattr(accounting, "input_tokens"))
        context_tokens = int(getattr(accounting, "context_tokens"))
        if input_tokens + reserve <= context_tokens:
            current = trial
            continue

        if current:
            batches.append(current)
            current = [page]
            try:
                single = _live_accounting(router, _batch_messages(domain, current))
            except Exception as exc:
                diagnostics.append(f"exact_input_accounting_failure:{type(exc).__name__}:{exc}")
                single = None
            if single is None:
                batches.append(current)
                current = []
                continue
            if int(getattr(single, "input_tokens")) + reserve > int(
                getattr(single, "context_tokens")
            ):
                diagnostics.append(
                    "source_page_exceeds_live_context:" + str(page.get("page_ref") or "")
                )
                current = []
        else:
            diagnostics.append(
                "source_page_exceeds_live_context:" + str(page.get("page_ref") or "")
            )
    if current:
        batches.append(current)
    return batches, diagnostics


def _extract_batch(
    router: Any,
    *,
    domain: Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], int]:
    contents = {
        str(page.get("page_ref") or "").strip(): str(page.get("content") or "")
        for page in pages
        if str(page.get("page_ref") or "").strip()
        and str(page.get("content") or "").strip()
    }
    if not contents:
        return [], ["empty_host_batch"], 0
    messages = _batch_messages(domain, pages)
    try:
        raw = router.generate_text(
            "planner",
            messages,
            response_format="text",
            response_schema=None,
            tool_stage="research",
            enable_tools=False,
        )
    except Exception as exc:
        return [], [f"model_read_failure:{type(exc).__name__}:{exc}"], 1

    claims: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_line in str(raw or "").splitlines():
        line = raw_line.strip()
        if not line or line.casefold() == "none":
            continue
        parts = line.split("\t", 3)
        if len(parts) != 4 or parts[0].strip().casefold() != "evidence":
            diagnostics.append("ignored_malformed_model_line")
            continue
        page_ref = parts[1].strip()
        content = contents.get(page_ref)
        if content is None:
            diagnostics.append("rejected_unknown_page_ref")
            continue
        exact = _exact_span(content, parts[2])
        insight = " ".join(parts[3].split()).strip()
        if not exact:
            diagnostics.append("rejected_non_exact_quote")
            continue
        if not insight:
            insight = "Implementation reference: " + " ".join(exact.split())
        key = (insight, exact, page_ref)
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            {
                "claim": insight,
                "evidence_refs": [page_ref],
                "support_quote": exact,
                "support_verification": "host_exact_quote_from_small_model_line",
            }
        )
    return claims, diagnostics, 1


def _load_grounded(document: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(document.get("raw_path") or "")).expanduser()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _source_body_count(evidence: Mapping[str, Any]) -> int:
    """Count claim-bearing source bodies without trusting a generated envelope."""
    grounded = evidence.get("grounded_rag")
    queries = grounded.get("queries") if isinstance(grounded, Mapping) else None
    count = 0
    for query in queries if isinstance(queries, list) else []:
        if not isinstance(query, Mapping):
            continue
        records = query.get("evidence_records")
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, Mapping):
                continue
            if str(record.get("content") or record.get("body") or record.get("text") or "").strip():
                count += 1
    return count


def _authoritative_requirements(domain: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for value in domain.get("requirements", ()):
        text = " ".join(str(value or "").split()).strip()
        if text and text not in result:
            result.append(text)
    if not result:
        objective = " ".join(str(domain.get("objective") or "").split()).strip()
        if objective:
            result.append(objective)
    return result


def _document_receipt(project_rag: Any, document: Mapping[str, Any]) -> dict[str, Any]:
    receipt = getattr(project_rag, "_prompt_document_receipt", None)
    if callable(receipt):
        value = receipt(document)
        return dict(value) if isinstance(value, Mapping) else {"value": value}
    return {
        "schema_version": "mmm/research-evidence-document-receipt-v1",
        "domain_id": str(document.get("domain_id") or ""),
        "document_sha256": str(document.get("document_sha256") or ""),
        "page_count": int(document.get("page_count") or 0),
        "raw_path": str(document.get("raw_path") or ""),
        "pages_path": str(document.get("pages_path") or ""),
        "model_unit_count": int(document.get("model_unit_count") or 0),
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
    del agentic_module, prompt, trace_metadata
    domain_id = str(domain.get("domain_id") or "").strip() or "unknown"

    working_document = dict(document)
    evidence = _load_grounded(document)
    source_body_count = _source_body_count(evidence)
    grounded = evidence.get("grounded_rag") if isinstance(evidence, Mapping) else None
    if source_body_count > 0 and isinstance(grounded, Mapping):
        try:
            from .pre_design_rag_quality_contract import fuse_grounded_domain_evidence

            evidence["grounded_rag"] = fuse_grounded_domain_evidence(domain, grounded)
            source_body_count = _source_body_count(evidence)
            working_document = project_rag._materialize_domain_evidence_document(
                domain_id, evidence
            )
        except Exception:
            working_document = dict(document)

    if isinstance(document, dict):
        document.clear()
        document.update(working_document)

    model_unit_count = int(working_document.get("model_unit_count") or 0)
    projection_is_empty = source_body_count == 0 or model_unit_count == 0
    if projection_is_empty:
        pages: list[dict[str, Any]] = []
    else:
        try:
            pages = project_rag._read_evidence_pages(working_document)
        except Exception:
            pages = []

    claims: list[dict[str, Any]] = []
    diagnostics: list[str] = (
        ["no_claim_bearing_source_bodies;model_not_called"] if projection_is_empty else []
    )
    model_call_count = 0
    batches, batch_diagnostics = _capacity_batches(
        router,
        domain=domain,
        pages=_candidate_pages(pages, domain),
    )
    diagnostics.extend(batch_diagnostics)
    for batch in batches:
        extracted, batch_notes, calls = _extract_batch(
            router,
            domain=domain,
            pages=batch,
        )
        claims.extend(extracted)
        diagnostics.extend(batch_notes)
        model_call_count += calls

    unique: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()
    for claim in claims:
        refs = claim.get("evidence_refs") if isinstance(claim, Mapping) else None
        ref = str(refs[0]) if isinstance(refs, list) and refs else ""
        key = (str(claim.get("claim") or ""), ref)
        if key[0] and key[1] and key not in seen_claims:
            seen_claims.add(key)
            unique.append(dict(claim))

    page_refs = [
        str(page.get("page_ref") or "").strip()
        for page in pages
        if str(page.get("page_ref") or "").strip()
    ]
    evidence_status = "supported" if unique else "no_relevant_external_evidence"
    requirement_fallback = [] if unique else _authoritative_requirements(domain)
    return {
        "domain_id": domain_id,
        "claims": unique,
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
        "fixed_point": False,
        "research_mode": "advisory_predesign",
        "research_evidence_status": evidence_status,
        "authoritative_requirement_fallback": requirement_fallback,
        "source_body_count": source_body_count,
        "model_called": model_call_count > 0,
        "model_call_count": model_call_count,
        "page_local_diagnostics": list(dict.fromkeys(diagnostics)),
        "evidence_page_refs": page_refs,
        "evidence_document": _document_receipt(project_rag, working_document),
        "quality_contract": {
            "schema_version": _PROTOCOL,
            "model_role": "semantic_evidence_extraction_only",
            "host_role": (
                "retrieval_state+source_refs+quote_verification+sufficiency+serialization"
            ),
            "model_json": False,
            "model_corrective_queries": False,
            "capacity_boundary": "live_adapter_input_tokens+configured_output<=live_context",
            "page_local_uncertainty_blocks_design": False,
            "missing_external_evidence_blocks_design": False,
            "zero_source_body_model_calls": 0,
        },
        "checkpoint": {
            "schema_version": "mmm/research-domain-checkpoint-v9",
            "status": "complete",
        },
    }


__all__ = ["research_document_domain"]
