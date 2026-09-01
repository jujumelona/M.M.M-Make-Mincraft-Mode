from __future__ import annotations

# Small-model-safe, host-owned pre-design research.
# User-authored gameplay requirements are already authoritative. External RAG is
# implementation evidence, never permission to design the requested feature.

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PROTOCOL = "mmm/small-model-predesign-evidence-v1"
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
        return 1
    have = _terms([str(page.get("content") or "")])
    return len(wanted & have)


def _candidate_pages(
    pages: Sequence[Mapping[str, Any]], domain: Mapping[str, Any]
) -> list[dict[str, Any]]:
    scored = [
        (max(0, _page_score(page, domain)), index, dict(page))
        for index, page in enumerate(pages)
    ]
    positive = [item for item in scored if item[0] > 0]
    selected = positive if positive else scored[: min(4, len(scored))]
    selected.sort(key=lambda item: (-item[0], item[1]))
    return [page for _score, _index, page in selected]


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


def _extract_page(
    router: Any,
    *,
    domain: Mapping[str, Any],
    page: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    content = str(page.get("content") or "")
    page_ref = str(page.get("page_ref") or "").strip()
    if not content.strip() or not page_ref:
        return [], ["empty_host_page"]
    messages = [
        {
            "role": "system",
            "content": (
                "Read SOURCE only. Extract at most 3 implementation facts useful for the "
                "Minecraft mod design. Output one line per useful fact exactly as "
                "EVIDENCE<TAB>EXACT_QUOTE<TAB>IMPLEMENTATION_INSIGHT. "
                "EXACT_QUOTE must be copied from SOURCE, not paraphrased. "
                "If nothing useful exists output only NONE. No JSON, Markdown, code fences, "
                "analysis, headings, IDs, sufficiency flags, search queries, or extra prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "OBJECTIVE\n"
                + str(domain.get("objective") or "")
                + "\n\nSOURCE\n"
                + content
            ),
        },
    ]
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
        return [], [f"model_read_failure:{type(exc).__name__}:{exc}"]

    claims: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in str(raw or "").splitlines():
        line = raw_line.strip()
        if not line or line.casefold() == "none":
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[0].strip().casefold() != "evidence":
            diagnostics.append("ignored_malformed_model_line")
            continue
        exact = _exact_span(content, parts[1])
        insight = " ".join(parts[2].split()).strip()
        if not exact or len("".join(exact.split())) < 8:
            diagnostics.append("rejected_non_exact_quote")
            continue
        if not insight:
            insight = "Implementation reference: " + " ".join(exact.split())
        key = (insight, exact)
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
        if len(claims) >= 3:
            break
    return claims, diagnostics


def _load_grounded(document: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(document.get("raw_path") or "")).expanduser()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


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
    grounded = evidence.get("grounded_rag") if isinstance(evidence, Mapping) else None
    if isinstance(grounded, Mapping):
        try:
            from .pre_design_rag_quality_contract import fuse_grounded_domain_evidence

            evidence["grounded_rag"] = fuse_grounded_domain_evidence(domain, grounded)
            working_document = project_rag._materialize_domain_evidence_document(
                domain_id, evidence
            )
        except Exception:
            working_document = dict(document)

    if isinstance(document, dict):
        document.clear()
        document.update(working_document)

    projection_is_empty = (
        "model_unit_count" in working_document
        and int(working_document.get("model_unit_count") or 0) == 0
    )
    if projection_is_empty:
        pages = []
    else:
        try:
            pages = project_rag._read_evidence_pages(working_document)
        except Exception:
            pages = []

    claims: list[dict[str, Any]] = []
    diagnostics: list[str] = (
        ["no_claim_bearing_source_bodies"] if projection_is_empty else []
    )
    for page in _candidate_pages(pages, domain):
        extracted, page_diagnostics = _extract_page(router, domain=domain, page=page)
        claims.extend(extracted)
        diagnostics.extend(page_diagnostics)

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
            "page_local_uncertainty_blocks_design": False,
            "missing_external_evidence_blocks_design": False,
        },
        "checkpoint": {
            "schema_version": "mmm/research-domain-checkpoint-v7",
            "status": "complete",
        },
    }


__all__ = ["research_document_domain"]
