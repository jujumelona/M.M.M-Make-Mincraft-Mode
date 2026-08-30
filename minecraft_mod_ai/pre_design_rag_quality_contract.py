from __future__ import annotations

"""Corrective quality layer for pre-design grounded research."""

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

_INSTALLED = False
_MARKER = "_mmm_pre_design_rag_quality_v1"
_RRF_K = 60.0
_DEFAULT_CORRECTIVE_ROUNDS = 2
_MAX_CORRECTIVE_ROUNDS = 4
_QUERY_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
_SUPPORT_QUOTE_MIN_NONSPACE = 8


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_text(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def _query_tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(str(value or ""))
        if len(token) >= 2
    }


def _is_retrieval_query(value: Any, *, raw_prompt: str = "") -> bool:
    query = " ".join(str(value or "").split()).strip()
    if not query or len(query) > 180:
        return False
    try:
        query.encode("ascii")
    except UnicodeEncodeError:
        return False
    words = _QUERY_WORD.findall(query)
    if len(words) < 2 or len(words) > 24:
        return False
    if raw_prompt and query.casefold() == " ".join(raw_prompt.split()).strip().casefold():
        return False
    return True


def _record_content(record: Mapping[str, Any]) -> str:
    for field in ("content", "excerpt", "snippet", "text", "body"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _record_key(record: Mapping[str, Any]) -> str:
    digest = str(record.get("content_sha256") or "").strip()
    if digest:
        return "content:" + digest.casefold()
    content = _record_content(record)
    if content:
        return "content:" + _sha256_text(content)
    source_id = str(record.get("source_id") or record.get("document_id") or "").strip()
    url = str(record.get("url") or "").strip()
    return "locator:" + _sha256_text(source_id + "\n" + url)


def _provider_family(record: Mapping[str, Any]) -> str:
    section = str(record.get("retrieval_section") or "").strip().casefold()
    source_type = str(record.get("source_type") or "").strip().casefold()
    source_id = str(record.get("source_id") or "").strip().casefold()
    url = str(record.get("url") or "").strip().casefold()
    if "github" in source_type or source_id.startswith("github:") or "github.com/" in url:
        return "github"
    if "modrinth" in source_type or source_id.startswith("modrinth:") or "modrinth.com/" in url:
        return "modrinth"
    if section == "code_rag":
        return "project_code"
    if section == "project_rag":
        return "project_reference"
    return section or "unknown"


def _record_relevance(query: str, record: Mapping[str, Any]) -> float:
    query_terms = _query_tokens(query)
    if not query_terms:
        return 0.0
    searchable = " ".join(
        (
            str(record.get("title") or ""),
            str(record.get("source_id") or ""),
            str(record.get("url") or ""),
            _record_content(record)[:12_000],
        )
    )
    record_terms = _query_tokens(searchable)
    return len(query_terms & record_terms) / max(1, len(query_terms))


def fuse_grounded_domain_evidence(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> dict[str, Any]:
    del domain
    raw_queries = grounded.get("queries")
    query_rows = [row for row in raw_queries or [] if isinstance(row, Mapping)]
    aggregates: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    duplicate_count = 0

    for row in query_rows:
        query = str(row.get("query") or "").strip()
        raw_records = [
            item
            for item in row.get("evidence_records", ())
            if isinstance(item, Mapping) and _record_content(item)
        ]
        per_query: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for input_rank, record in enumerate(raw_records, start=1):
            key = _record_key(record)
            if key in per_query:
                duplicate_count += 1
                continue
            per_query[key] = (input_rank, record)
        ranked = sorted(
            per_query.items(),
            key=lambda item: (
                -_record_relevance(query, item[1][1]),
                item[1][0],
                item[0],
            ),
        )
        trace.append(
            {
                "query": query,
                "query_sha256": str(row.get("query_sha256") or _sha256_text(query)),
                "input_record_count": len(raw_records),
                "unique_record_count": len(ranked),
                "github_provider_status": str(row.get("github_provider_status") or "not_requested"),
                "github_saturation_reason": str(row.get("github_saturation_reason") or ""),
                "retrieval_errors": list(row.get("retrieval_errors") or ()),
            }
        )
        for rank, (key, (_input_rank, record)) in enumerate(ranked, start=1):
            relevance = _record_relevance(query, record)
            aggregate = aggregates.get(key)
            if aggregate is None:
                aggregate = {
                    "record": dict(record),
                    "rrf_score": 0.0,
                    "max_query_coverage": 0.0,
                    "matched_queries": [],
                    "provider_families": set(),
                    "query_hits": 0,
                }
                aggregates[key] = aggregate
            else:
                duplicate_count += 1
            aggregate["rrf_score"] += 1.0 / (_RRF_K + rank)
            aggregate["max_query_coverage"] = max(float(aggregate["max_query_coverage"]), relevance)
            if query and query not in aggregate["matched_queries"]:
                aggregate["matched_queries"].append(query)
                aggregate["query_hits"] += 1
            aggregate["provider_families"].add(_provider_family(record))

    fused_records: list[dict[str, Any]] = []
    for key, aggregate in aggregates.items():
        record = dict(aggregate["record"])
        query_hits = int(aggregate["query_hits"])
        families = sorted(str(item) for item in aggregate["provider_families"])
        score = (
            float(aggregate["rrf_score"])
            + float(aggregate["max_query_coverage"])
            + 0.05 * max(0, query_hits - 1)
            + 0.02 * max(0, len(families) - 1)
        )
        record["retrieval_fusion"] = {
            "record_key": key,
            "rrf_score": round(float(aggregate["rrf_score"]), 8),
            "max_query_coverage": round(float(aggregate["max_query_coverage"]), 6),
            "query_hits": query_hits,
            "matched_queries": list(aggregate["matched_queries"]),
            "provider_families": families,
            "combined_score": round(score, 8),
        }
        fused_records.append(record)
    fused_records.sort(
        key=lambda record: (
            -float(record.get("retrieval_fusion", {}).get("combined_score", 0.0)),
            _record_key(record),
        )
    )
    source_queries = [str(row.get("query") or "").strip() for row in query_rows]
    source_queries = [query for query in source_queries if query]
    query_coverage = (
        sum(1 for row in trace if int(row["unique_record_count"]) > 0) / len(trace)
        if trace
        else 0.0
    )
    families = sorted(
        {
            family
            for record in fused_records
            for family in record.get("retrieval_fusion", {}).get("provider_families", ())
        }
    )
    fused_query = {
        "query": "domain fused evidence",
        "query_sha256": _sha256_text("\n".join(source_queries)),
        "source_queries": source_queries,
        "evidence_records": fused_records,
        "content_record_count": len(fused_records),
        "github_record_count": sum(
            1
            for record in fused_records
            if "github" in record.get("retrieval_fusion", {}).get("provider_families", ())
        ),
        "github_provider_status": (
            "available"
            if any(str(row.get("github_provider_status") or "") == "available" for row in trace)
            else "not_requested"
        ),
        "github_saturation_reason": "",
        "retrieval_errors": [
            str(error)
            for row in trace
            for error in row.get("retrieval_errors", ())
        ][:12],
    }
    result = dict(grounded)
    result["queries"] = [fused_query]
    result["retrieval_trace"] = trace
    result["fusion"] = {
        "schema_version": "mmm/pre-design-evidence-fusion-v1",
        "algorithm": "exact_content_dedupe+query_local_lexical_rank+rrf",
        "query_count": len(query_rows),
        "queries_with_content": sum(1 for row in trace if row["unique_record_count"]),
        "query_coverage_ratio": round(query_coverage, 6),
        "unique_record_count": len(fused_records),
        "duplicate_record_count": duplicate_count,
        "provider_families": families,
        "all_unique_records_preserved": True,
    }
    return result


def _support_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_index": {"type": "integer", "minimum": 0, "maximum": max(0, count - 1)},
                        "supported": {"type": "boolean"},
                        "support_quote": {"type": "string"},
                    },
                    "required": ["claim_index", "supported", "support_quote"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def _claim_candidates(notes: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for note in notes:
        for raw in note.get("claims", ()):
            if not isinstance(raw, Mapping):
                continue
            claim = " ".join(str(raw.get("claim") or "").split()).strip()
            if claim and claim not in seen:
                seen.add(claim)
                result.append(claim)
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
    if not claims:
        return [], []
    content = str(page.get("content") or "")
    page_ref = str(page.get("page_ref") or "").strip()
    messages = [
        {
            "role": "system",
            "content": (
                "Judge each claim ONLY against the supplied host-owned evidence page. "
                "A claim is supported only when the page entails its material proposition. "
                "For supported claims copy the shortest exact contiguous supporting quote. "
                "If merely related, metadata-only, ambiguous, or unsupported, return false "
                "and an empty quote. External knowledge is forbidden."
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
            raise agentic_module.SpecValidationError(f"claim support verifier returned invalid JSON: {exc}") from exc
        verdicts = value.get("verdicts") if isinstance(value, Mapping) else None
        if not isinstance(verdicts, list) or len(verdicts) != len(claims):
            raise agentic_module.SpecValidationError("claim support verifier must return exactly one verdict per claim")
        by_index: dict[int, Mapping[str, Any]] = {}
        for verdict in verdicts:
            if not isinstance(verdict, Mapping):
                raise agentic_module.SpecValidationError("claim support verdict must be an object")
            index = verdict.get("claim_index")
            if type(index) is not int or not 0 <= index < len(claims) or index in by_index:
                raise agentic_module.SpecValidationError("claim support verdict indices are invalid or duplicated")
            by_index[index] = verdict
        if set(by_index) != set(range(len(claims))):
            raise agentic_module.SpecValidationError("claim support verifier omitted a claim")
        accepted: list[dict[str, Any]] = []
        rejected: list[str] = []
        for index, claim in enumerate(claims):
            verdict = by_index[index]
            supported = verdict.get("supported")
            if type(supported) is not bool:
                raise agentic_module.SpecValidationError("claim support verdict supported must be boolean")
            quote = str(verdict.get("support_quote") or "")
            if not supported:
                rejected.append(claim)
                continue
            if not quote or quote not in content:
                raise agentic_module.SpecValidationError("supported claim must carry an exact contiguous quote from the page")
            if len("".join(quote.split())) < _SUPPORT_QUOTE_MIN_NONSPACE:
                raise agentic_module.SpecValidationError("supported claim quote is too short to establish material support")
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


def _merge_verified_notes(domain_id: str, page_notes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    gaps: list[str] = []
    next_queries: list[str] = []
    procedures: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()
    seen_procedures: set[str] = set()
    for note in page_notes:
        page_ref = str(note.get("_host_page_ref") or "").strip()
        for claim in note.get("claims", ()):
            if not isinstance(claim, Mapping):
                continue
            text = str(claim.get("claim") or "").strip()
            refs = [str(ref).strip() for ref in claim.get("evidence_refs", ()) if str(ref).strip()]
            key = (text, refs[0] if refs else page_ref)
            if text and refs and key not in seen_claims:
                seen_claims.add(key)
                claims.append(dict(claim))
        for gap in _stable_text(note.get("gaps")):
            if gap not in gaps:
                gaps.append(gap)
        for query in _stable_text(note.get("next_queries")):
            if query not in next_queries:
                next_queries.append(query)
        verified_page = any(
            isinstance(claim, Mapping)
            and str(claim.get("support_verification") or "") == "model_entailment+host_exact_quote"
            for claim in note.get("claims", ())
        )
        if verified_page:
            for raw in note.get("procedures", ()):
                if not isinstance(raw, Mapping):
                    continue
                procedure = dict(raw)
                procedure["evidence_refs"] = [page_ref] if page_ref else []
                key = json.dumps(procedure, ensure_ascii=False, sort_keys=True, default=str)
                if key not in seen_procedures:
                    seen_procedures.add(key)
                    procedures.append(procedure)
    return {
        "domain_id": domain_id,
        "claims": claims,
        "gaps": gaps,
        "next_queries": next_queries,
        "procedures": procedures,
        "sufficient": bool(claims),
    }


def _corrective_round_limit() -> int:
    raw = os.environ.get("MMM_PREDESIGN_CORRECTIVE_ROUNDS", str(_DEFAULT_CORRECTIVE_ROUNDS)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_CORRECTIVE_ROUNDS
    return max(0, min(value, _MAX_CORRECTIVE_ROUNDS))


def _correction_queries(values: Any, *, seen: set[str], raw_prompt: str, limit: int = 4) -> list[str]:
    result: list[str] = []
    for value in _stable_text(values):
        if not _is_retrieval_query(value, raw_prompt=raw_prompt):
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _gap_query_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"type": "string", "minLength": 4, "maxLength": 180},
                "uniqueItems": True,
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }


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
    messages = [
        {
            "role": "system",
            "content": (
                "Write 1-4 concise ENGLISH search queries that retrieve the missing evidence "
                "for this already-approved Minecraft-mod requirement. Do not change the "
                "requirement, select a donor mod, copy the raw user request, or repeat an "
                "already-searched query."
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
            raise agentic_module.SpecValidationError(f"corrective query planner returned invalid JSON: {exc}") from exc
        queries = value.get("queries") if isinstance(value, Mapping) else None
        if not isinstance(queries, list):
            raise agentic_module.SpecValidationError("corrective query planner must return a queries array")
        planned = _correction_queries(queries, seen=seen, raw_prompt=raw_prompt, limit=4)
        if not planned:
            raise agentic_module.SpecValidationError("corrective query planner returned no new executable English query")
        return planned

    return project_rag._generate_bounded(
        agentic_module,
        router,
        messages=messages,
        response_schema=_gap_query_schema(),
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
    normalized_notes: list[dict[str, Any]] = []
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
            progress_label=f"domain {domain_id} round {round_index} page {page_index + 1}/{len(pages)}",
            failures=failures,
        )
        candidates = _claim_candidates([note for note in notes if isinstance(note, Mapping)])
        try:
            verified, rejected = _verify_page_claims(
                agentic_module,
                project_rag,
                router,
                domain_id=domain_id,
                page=page,
                claims=candidates,
                progress_label=f"domain {domain_id} round {round_index} page {page_index + 1}/{len(pages)}",
            )
        except Exception as exc:
            failures.append({"unit": f"support:{round_index}:{page_index}", "error": f"{type(exc).__name__}: {exc}"})
            verified, rejected = [], candidates
        rejected_count += len(rejected)
        gaps: list[str] = []
        next_queries: list[str] = []
        procedures: list[dict[str, Any]] = []
        for note in notes:
            if not isinstance(note, Mapping):
                continue
            for gap in _stable_text(note.get("gaps")):
                if gap not in gaps:
                    gaps.append(gap)
            for query in _stable_text(note.get("next_queries")):
                if query not in next_queries:
                    next_queries.append(query)
            for procedure in note.get("procedures", ()):
                if isinstance(procedure, Mapping):
                    procedures.append(dict(procedure))
        for claim in rejected:
            gaps.append("Claim rejected by page-support verification; retrieve stronger evidence: " + claim)
        normalized_notes.append(
            {
                "_host_page_ref": page_ref,
                "domain_id": domain_id,
                "claims": verified,
                "gaps": gaps,
                "next_queries": next_queries,
                "procedures": procedures,
                "sufficient": bool(verified),
            }
        )
    return pages, normalized_notes, rejected_count


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
    del trace_metadata
    domain_id = str(domain.get("domain_id") or "").strip() or "unknown"
    base_key = project_rag._domain_checkpoint_key(router, prompt=prompt, domain=domain, document=document)
    domain_key = project_rag._sha256({"base_domain_key": base_key, "research_policy": "corrective-fusion-claim-support-v1"}).removeprefix("sha256:")
    with project_rag._domain_lock(domain_key):
        cached = project_rag._read_complete_manifest(agentic_module, domain_key, domain_id)
        if isinstance(cached, Mapping):
            quality = cached.get("quality_contract")
            refs = {str(ref) for ref in cached.get("evidence_page_refs", ()) if str(ref).strip()}
            if isinstance(quality, Mapping) and quality.get("schema_version") == "mmm/pre-design-rag-quality-v1" and refs:
                try:
                    agentic_module._validate_sufficient_research(cached, allowed_refs=frozenset(refs))
                except agentic_module.SpecValidationError:
                    cached = None
            else:
                cached = None
        if isinstance(cached, Mapping):
            return dict(cached)

        failures: list[dict[str, str]] = []
        documents: list[dict[str, Any]] = [dict(document)]
        all_pages: list[dict[str, Any]] = []
        all_notes: list[dict[str, Any]] = []
        seen_document_hashes = {str(document.get("document_sha256") or "").strip()}
        searched_queries = [str(query).strip() for query in domain.get("queries", ()) if str(query).strip()]
        seen_queries = {query.casefold() for query in searched_queries}
        correction_history: list[dict[str, Any]] = []
        rejected_claim_count = 0
        fixed_point_reason = ""
        max_rounds = _corrective_round_limit()
        round_index = 0

        while round_index <= max_rounds:
            current_document = documents[-1]
            pages, notes, rejected = _read_and_verify_document(
                agentic_module,
                project_rag,
                router,
                prompt=prompt,
                domain=domain,
                document=current_document,
                domain_key=domain_key,
                failures=failures,
                round_index=round_index,
            )
            all_pages.extend(pages)
            all_notes.extend(notes)
            rejected_claim_count += rejected
            summary = _merge_verified_notes(domain_id, all_notes)
            unseen = _correction_queries(summary.get("next_queries"), seen=seen_queries, raw_prompt=prompt)
            if failures:
                fixed_point_reason = "bounded_extraction_or_support_verification_failure"
                break
            if round_index >= max_rounds:
                fixed_point_reason = "corrective_round_limit_reached" if unseen or summary.get("gaps") else "no_unseen_corrective_query"
                break
            if not unseen and summary.get("gaps"):
                try:
                    unseen = _generate_gap_queries(
                        agentic_module,
                        project_rag,
                        router,
                        domain=domain,
                        gaps=list(summary.get("gaps") or ()),
                        prior_queries=searched_queries,
                        seen=seen_queries,
                        raw_prompt=prompt,
                        progress_label=f"domain {domain_id} round {round_index}",
                    )
                except Exception as exc:
                    project_rag._emit_research_progress(
                        "corrective_query_generation_failure",
                        domain_id=domain_id,
                        corrective_round=round_index,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    unseen = []
            if not unseen:
                fixed_point_reason = "verified_claims_sufficient" if summary.get("claims") else "no_valid_unseen_corrective_query"
                break
            searched_queries.extend(unseen)
            correction_domain = dict(domain)
            correction_domain["queries"] = unseen
            bundle = project_rag._forced_rag_bundle(
                router,
                {"domains": [correction_domain], "schema_version": "mmm/corrective-retrieval-request-v1"},
            )
            grounded = pipeline_module._grounded_domain_evidence(agentic_module, domain_id, bundle)
            fused = fuse_grounded_domain_evidence(domain, grounded)
            fused_records = [
                record
                for row in fused.get("queries", ())
                if isinstance(row, Mapping)
                for record in row.get("evidence_records", ())
                if isinstance(record, Mapping) and _record_content(record)
            ]
            correction_history.append(
                {
                    "round": round_index + 1,
                    "queries": unseen,
                    "unique_content_records": len(fused_records),
                    "fusion": dict(fused.get("fusion") or {}) if isinstance(fused.get("fusion"), Mapping) else {},
                }
            )
            if not fused_records:
                fixed_point_reason = "corrective_retrieval_returned_no_claim_bearing_content"
                break
            correction_document = project_rag._materialize_domain_evidence_document(domain_id, {"grounded_rag": fused})
            document_hash = str(correction_document.get("document_sha256") or "").strip()
            if not document_hash or document_hash in seen_document_hashes:
                fixed_point_reason = "corrective_retrieval_repeated_existing_evidence"
                break
            seen_document_hashes.add(document_hash)
            documents.append(dict(correction_document))
            round_index += 1

        summary = _merge_verified_notes(domain_id, all_notes)
        claims = list(summary["claims"])
        catalog = project_rag._materialize_claim_catalog(domain_key, domain_id, claims)
        evidence_ledger = project_rag._materialize_evidence_ledger(domain_key, domain_id, all_pages)
        page_refs = [str(page.get("page_ref") or "").strip() for page in all_pages if str(page.get("page_ref") or "").strip()]
        failure_reasons: list[str] = []
        if failures:
            failure_reasons.append("bounded extraction/support verification failure")
        if not claims:
            failure_reasons.append("zero support-verified grounded claims")
        status = "failed" if failure_reasons else "complete"
        note: dict[str, Any] = {
            **summary,
            "evidence_document": project_rag._prompt_document_receipt(document),
            "corrective_evidence_documents": [project_rag._prompt_document_receipt(item) for item in documents[1:]],
            "evidence_page_refs": page_refs,
            "claim_catalog": catalog,
            "evidence_ledger": evidence_ledger,
            "quality_contract": {
                "schema_version": "mmm/pre-design-rag-quality-v1",
                "fusion": "exact_content_dedupe+query_local_lexical_rank+rrf",
                "corrective_retrieval": True,
                "claim_support": "model_entailment+host_exact_quote",
                "corrective_round_limit": max_rounds,
                "corrective_rounds_executed": len(correction_history),
                "correction_history": correction_history,
                "rejected_claim_count": rejected_claim_count,
                "fixed_point_reason": fixed_point_reason,
                "donor_selection_performed": False,
            },
            "checkpoint": {
                "schema_version": "mmm/research-domain-checkpoint-v4",
                "request_sha256": "sha256:" + domain_key,
                "status": status,
                "manifest_path": str(project_rag._manifest_path(domain_key)),
                "checkpoint_dir": str(project_rag._checkpoint_dir(domain_key)),
            },
        }
        if failures:
            note["research_failures"] = list(failures)
            note["gaps"] = [*list(note.get("gaps", ())), *(f"{item['unit']}: {item['error']}" for item in failures)]
        if failure_reasons:
            note["sufficient"] = False
            note["fixed_point"] = True
            note["failure_reasons"] = failure_reasons
        else:
            note["sufficient"] = True
            note["fixed_point"] = bool(fixed_point_reason and fixed_point_reason != "verified_claims_sufficient")
            agentic_module._validate_sufficient_research(note, allowed_refs=frozenset(page_refs))
        project_rag._write_manifest(domain_key, status=status, note=note, failures=failures)
        if status != "complete":
            raise project_rag._BoundedResearchOutputError(
                "pre-design corrective research failed closed for domain "
                f"{domain_id!r}: {'; '.join(failure_reasons)}; fixed_point={fixed_point_reason}"
            )
        return note


def _sanitize_pre_design_brief(previous: Any, prompt: str) -> dict[str, Any]:
    brief = previous(prompt)
    if not isinstance(brief, Mapping):
        return brief
    result = dict(brief)
    raw_domains = result.get("domains")
    if not isinstance(raw_domains, list):
        return result
    domains: list[dict[str, Any]] = []
    for raw in raw_domains:
        if not isinstance(raw, Mapping):
            continue
        domain = dict(raw)
        queries = [str(query).strip() for query in domain.get("queries", ()) if _is_retrieval_query(query, raw_prompt=prompt)]
        queries = list(dict.fromkeys(queries))
        if not queries:
            from . import pre_design_research_pipeline as pipeline
            raise pipeline.PreDesignResearchFailure(
                "Approved pre-design requirement produced no rewritten English retrieval query after raw-prompt/non-ASCII filtering."
            )
        domain["queries"] = queries
        domains.append(domain)
    result["domains"] = domains
    return result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from . import agentic_pre_design_rag as project_rag
    from . import pre_design_research_pipeline as pipeline

    previous_domain_evidence = pipeline._domain_document_evidence
    if not getattr(previous_domain_evidence, _MARKER, False):
        def domain_evidence(agentic: Any, domain_id: str, deterministic: Mapping[str, Any], *, grounded_bundle: Mapping[str, Any]) -> dict[str, Any]:
            result = previous_domain_evidence(agentic, domain_id, deterministic, grounded_bundle=grounded_bundle)
            grounded = result.get("grounded_rag")
            if isinstance(grounded, Mapping):
                result["grounded_rag"] = fuse_grounded_domain_evidence({"domain_id": domain_id}, grounded)
            return result
        setattr(domain_evidence, _MARKER, True)
        domain_evidence.__wrapped__ = previous_domain_evidence
        pipeline._domain_document_evidence = domain_evidence

    previous_page_messages = project_rag._research_page_messages
    if not getattr(previous_page_messages, _MARKER, False):
        def page_messages(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
            messages = [dict(item) for item in previous_page_messages(*args, **kwargs)]
            if messages:
                messages[0]["content"] = str(messages[0].get("content") or "") + (
                    " When evidence is missing or ambiguous, next_queries must contain concise English retrieval queries, not the raw user sentence."
                )
            return messages
        setattr(page_messages, _MARKER, True)
        page_messages.__wrapped__ = previous_page_messages
        project_rag._research_page_messages = page_messages

    previous_research = pipeline.research_document_domain
    if not getattr(previous_research, _MARKER, False):
        def research_document_domain(agentic_module: Any, project_rag_module: Any, router: Any, *, prompt: str, domain: Mapping[str, Any], document: Mapping[str, Any], trace_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
            return _quality_research_document_domain(
                pipeline,
                agentic_module,
                project_rag_module,
                router,
                prompt=prompt,
                domain=domain,
                document=document,
                trace_metadata=trace_metadata,
            )
        setattr(research_document_domain, _MARKER, True)
        research_document_domain.__wrapped__ = previous_research
        pipeline.research_document_domain = research_document_domain

    previous_grounding = pipeline._validate_document_grounding
    if not getattr(previous_grounding, _MARKER, False):
        def validate_document_grounding(agentic: Any, project_rag_module: Any, note: Mapping[str, Any], document: Mapping[str, Any], *, domain_id: str) -> None:
            quality = note.get("quality_contract")
            refs = {str(ref).strip() for ref in note.get("evidence_page_refs", ()) if str(ref).strip()}
            if isinstance(quality, Mapping) and quality.get("schema_version") == "mmm/pre-design-rag-quality-v1" and refs:
                agentic._validate_sufficient_research(note, allowed_refs=frozenset(refs))
                return
            previous_grounding(agentic, project_rag_module, note, document, domain_id=domain_id)
        setattr(validate_document_grounding, _MARKER, True)
        validate_document_grounding.__wrapped__ = previous_grounding
        pipeline._validate_document_grounding = validate_document_grounding

    previous_brief = pipeline._pre_design_brief
    if not getattr(previous_brief, _MARKER, False):
        def pre_design_brief(prompt: str) -> dict[str, Any]:
            return _sanitize_pre_design_brief(previous_brief, prompt)
        setattr(pre_design_brief, _MARKER, True)
        pre_design_brief.__wrapped__ = previous_brief
        pipeline._pre_design_brief = pre_design_brief

    _INSTALLED = True


__all__ = ["fuse_grounded_domain_evidence", "install"]
