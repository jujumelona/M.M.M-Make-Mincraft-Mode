from __future__ import annotations

"""Claim/page support verification for pre-design research."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .pre_design_rag_fusion import _sha256_text, _stable_text

_MIN_QUOTE_CHARS = 8


def _emit_support_trace(event: str, **fields: Any) -> None:
    print(
        "PRE-DESIGN RAG TRACE: "
        + json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def _support_schema(count: int) -> dict[str, Any]:
    """Keep structured generation permissive; the host performs the strict validation."""

    verdict_item = {
        "type": "object",
        "properties": {
            "claim_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": max(0, count - 1),
            },
            "supported": {"type": "boolean"},
            "support_quote": {"type": "string"},
            "quote": {"type": "string"},
            "support": {"type": "string"},
        },
        "required": ["claim_index"],
        "additionalProperties": True,
    }
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": verdict_item,
            },
            "claims": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": verdict_item,
            },
            "sufficient": {"type": "boolean"},
            "gaps": {},
            "research_note": {},
        },
        "additionalProperties": True,
    }


def _claim_candidates(notes: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for note in notes:
        for raw in note.get("claims", ()):
            if not isinstance(raw, Mapping):
                continue
            claim = " ".join(str(raw.get("claim") or "").split()).strip()
            if claim and claim not in result:
                result.append(claim)
    return result


def _normalize_verdicts(value: Any, count: int) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    raw = value.get("verdicts")
    if not isinstance(raw, list):
        raw = value.get("claims")
    if not isinstance(raw, list) or len(raw) != count:
        return []
    result: list[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return []
        result.append(item)
    return result


def _verify_page_claims(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    domain_id: str,
    page: Mapping[str, Any],
    claims: Sequence[str],
    progress_label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep a claim only when the page entails it and provides an exact host quote."""

    if not claims:
        return [], []
    content = str(page.get("content") or "")
    page_ref = str(page.get("page_ref") or "").strip()
    messages = [
        {
            "role": "system",
            "content": (
                "Judge each claim only against the supplied host-owned evidence page. "
                "External knowledge is forbidden. Return one decision for every supplied "
                "claim_index. Mark supported only when the page entails the material "
                "proposition. For a supported claim copy the shortest exact contiguous "
                "supporting quote from the page; otherwise return false and an empty quote. "
                "Preferred shape is {verdicts:[{claim_index,supported,support_quote}]}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "domain_id": domain_id,
                    "page_ref": page_ref,
                    "claims": [
                        {"claim_index": index, "claim": claim}
                        for index, claim in enumerate(claims)
                    ],
                    "evidence_page_content": content,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]

    def parse(raw: str) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise agentic_module.SpecValidationError(
                f"claim support verifier returned invalid JSON: {exc}"
            ) from exc
        verdicts = _normalize_verdicts(value, len(claims))
        if not verdicts:
            raise agentic_module.SpecValidationError(
                "claim support verifier must return one decision per claim"
            )
        indexed: dict[int, Mapping[str, Any]] = {}
        for verdict in verdicts:
            index = verdict.get("claim_index")
            if (
                type(index) is not int
                or not 0 <= index < len(claims)
                or index in indexed
            ):
                raise agentic_module.SpecValidationError(
                    "claim support verdict indices are invalid"
                )
            indexed[index] = verdict
        if set(indexed) != set(range(len(claims))):
            raise agentic_module.SpecValidationError(
                "claim support verifier omitted a claim"
            )

        accepted: list[dict[str, Any]] = []
        rejected: list[str] = []
        for index, claim in enumerate(claims):
            verdict = indexed[index]
            supported = verdict.get("supported")
            quote = str(
                verdict.get("support_quote")
                or verdict.get("quote")
                or verdict.get("support")
                or ""
            )
            if type(supported) is not bool:
                # Some local models omit the boolean but provide/omit an exact support
                # quote.  Treat that only as a formatting alias; host exact-quote
                # validation below remains the authority.
                supported = bool(quote.strip())
            if not supported:
                _emit_support_trace(
                    "claim_support_rejected",
                    page_ref=page_ref,
                    claim_index=index,
                    claim=claim,
                    reason="model_marked_unsupported",
                )
                rejected.append(claim)
                continue
            if (
                not quote
                or quote not in content
                or len("".join(quote.split())) < _MIN_QUOTE_CHARS
            ):
                _emit_support_trace(
                    "claim_support_rejected",
                    page_ref=page_ref,
                    claim_index=index,
                    claim=claim,
                    reason=(
                        "missing_quote"
                        if not quote
                        else "quote_not_exact_page_substring"
                        if quote not in content
                        else "quote_too_short"
                    ),
                    quote_chars=len(quote),
                )
                rejected.append(claim)
                continue
            _emit_support_trace(
                "claim_support_accepted",
                page_ref=page_ref,
                claim_index=index,
                claim=claim,
                quote_chars=len(quote),
                quote_sha256=_sha256_text(quote),
            )
            accepted.append(
                {
                    "claim": claim,
                    "evidence_refs": [page_ref],
                    "support_quote": quote,
                    "support_quote_sha256": _sha256_text(quote),
                    "support_verification": "model_entailment+host_exact_quote",
                }
            )
        return accepted, rejected

    return project_rag._generate_bounded(
        agentic_module,
        router,
        messages=messages,
        response_schema=_support_schema(len(claims)),
        parser=parse,
        progress_label=progress_label + " claim-support",
    )


def _merge_verified_notes(
    domain_id: str,
    notes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    gaps: list[str] = []
    next_queries: list[str] = []
    procedures: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()

    for note in notes:
        page_ref = str(note.get("_host_page_ref") or "").strip()
        for raw in note.get("claims", ()):
            if not isinstance(raw, Mapping):
                continue
            claim = str(raw.get("claim") or "").strip()
            refs = [
                str(ref).strip()
                for ref in raw.get("evidence_refs", ())
                if str(ref).strip()
            ]
            key = (claim, refs[0] if refs else page_ref)
            if claim and refs and key not in seen_claims:
                seen_claims.add(key)
                claims.append(dict(raw))
        for value in _stable_text(note.get("gaps")):
            if value not in gaps:
                gaps.append(value)
        for value in _stable_text(note.get("next_queries")):
            if value not in next_queries:
                next_queries.append(value)

        verified_page = any(
            isinstance(raw, Mapping)
            and raw.get("support_verification")
            == "model_entailment+host_exact_quote"
            for raw in note.get("claims", ())
        )
        if verified_page:
            for raw in note.get("procedures", ()):
                if not isinstance(raw, Mapping):
                    continue
                procedure = dict(raw)
                procedure["evidence_refs"] = [page_ref] if page_ref else []
                if procedure not in procedures:
                    procedures.append(procedure)

    return {
        "domain_id": domain_id,
        "claims": claims,
        "gaps": gaps,
        "next_queries": next_queries,
        "procedures": procedures,
        "sufficient": bool(claims),
    }


__all__ = [
    "_claim_candidates",
    "_merge_verified_notes",
    "_verify_page_claims",
]
