from __future__ import annotations

"""Pure multi-query/provider fusion primitives for pre-design retrieval."""

import hashlib
import os
import re
from collections.abc import Mapping
from typing import Any

_RRF_K = 60.0
_QUERY_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
_SOURCE_BODY_FIELDS = ("content", "body", "text")
_GENERIC_QUERY_TERMS = {
    "minecraft",
    "mod",
    "mods",
    "source",
    "implementation",
    "mechanic",
    "system",
    "feature",
}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)).strip())
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _evidence_byte_budget() -> int:
    # This is a prompt/evidence budget, not a source-count cutoff. The fusion layer
    # keeps cross-query coverage first, then spends remaining bytes on the highest
    # scoring evidence. It prevents retrieval breadth from becoming hundreds of LLM
    # page reads.
    return _env_int(
        "MMM_PREDESIGN_EVIDENCE_BYTE_BUDGET",
        48 * 1024,
        minimum=12 * 1024,
        maximum=512 * 1024,
    )


def _excerpt_char_budget() -> int:
    return _env_int(
        "MMM_PREDESIGN_EVIDENCE_EXCERPT_CHARS",
        3_200,
        minimum=800,
        maximum=12_000,
    )


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_text(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _is_retrieval_query(value: Any, *, raw_prompt: str = "") -> bool:
    query = " ".join(str(value or "").split()).strip()
    if not query or len(query) > 180:
        return False
    try:
        query.encode("ascii")
    except UnicodeEncodeError:
        return False
    words = _QUERY_WORD.findall(query)
    if not 2 <= len(words) <= 24:
        return False
    return not raw_prompt or query.casefold() != " ".join(raw_prompt.split()).casefold()


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(str(value or ""))
        if len(token) >= 2
    }


def _query_relevance_tokens(query: str) -> set[str]:
    """Return claim-bearing query terms instead of generic retrieval scaffolding."""

    all_tokens = _tokens(query)
    distinctive = all_tokens - _GENERIC_QUERY_TERMS
    return distinctive or all_tokens


def _record_content(record: Mapping[str, Any]) -> str:
    """Return source-body text only; snippets/excerpts are never evidence bodies."""

    for field in _SOURCE_BODY_FIELDS:
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
    locator = (
        str(record.get("source_id") or record.get("document_id") or "")
        + "\n"
        + str(record.get("url") or "")
    )
    return "locator:" + _sha256_text(locator)


def _provider_family(record: Mapping[str, Any]) -> str:
    source_type = str(record.get("source_type") or "").casefold()
    source_id = str(record.get("source_id") or "").casefold()
    url = str(record.get("url") or "").casefold()
    section = str(record.get("retrieval_section") or "").casefold()
    if "github" in source_type or source_id.startswith("github:") or "github.com/" in url:
        return "github"
    if "modrinth" in source_type or source_id.startswith("modrinth:") or "modrinth.com/" in url:
        return "modrinth"
    return {
        "code_rag": "project_code",
        "project_rag": "project_reference",
    }.get(section, section or "unknown")


def _relevance(query: str, record: Mapping[str, Any]) -> float:
    wanted = _query_relevance_tokens(query)
    if not wanted:
        return 0.0
    searchable = " ".join(
        (
            str(record.get("title") or ""),
            str(record.get("source_id") or ""),
            str(record.get("url") or ""),
            _record_content(record)[:12_000],
        )
    )
    return len(wanted & _tokens(searchable)) / max(1, len(wanted))


def _record_is_query_relevant(query: str, record: Mapping[str, Any]) -> bool:
    """Require at least one non-generic query term before a page can enter fusion.

    Ranking can order weak records, but breadth preservation must never resurrect a
    record whose title/url/body shares no claim-bearing term with the retrieval query.
    This is a zero-relevance rejection, not a fixed score threshold.
    """

    return _relevance(query, record) > 0.0


def _evidence_excerpt(content: str, queries: list[str]) -> str:
    """Keep a contiguous source excerpt around the strongest query-bearing term."""

    text = str(content or "").strip()
    limit = _excerpt_char_budget()
    if len(text) <= limit:
        return text
    folded = text.casefold()
    terms: list[str] = []
    for query in queries:
        for token in _TOKEN.findall(query):
            term = token.casefold()
            if len(term) >= 4 and term not in _GENERIC_QUERY_TERMS and term not in terms:
                terms.append(term)
    terms.sort(key=len, reverse=True)
    positions = [folded.find(term) for term in terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    half = limit // 2
    start = max(0, center - half)
    end = min(len(text), start + limit)
    start = max(0, end - limit)
    excerpt = text[start:end]
    if start:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt += "…"
    return excerpt


def _bounded_records(
    records: list[dict[str, Any]], source_queries: list[str]
) -> tuple[list[dict[str, Any]], int]:
    budget = _evidence_byte_budget()
    prepared: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        fusion = dict(record.get("retrieval_fusion") or {})
        matched = [str(item) for item in fusion.get("matched_queries", ()) if str(item)]
        original = _record_content(record)
        projection = _evidence_excerpt(original, matched or source_queries)
        original_digest = str(record.get("content_sha256") or "").strip() or _sha256_text(original)

        # Canonicalize the model-facing evidence to one trusted body field. Keeping
        # ``body``/``text`` beside a bounded projection would silently retain the full
        # payload, while keeping ``snippet``/``excerpt`` would let discovery metadata be
        # mistaken for source evidence by downstream code.
        record["content"] = projection
        for field in ("body", "text", "snippet", "excerpt"):
            record.pop(field, None)
        if projection != original:
            record["source_content_sha256"] = original_digest
            record["content_sha256"] = _sha256_text(projection)
        else:
            record["content_sha256"] = original_digest

        fusion["original_content_chars"] = len(original)
        fusion["selected_content_chars"] = len(projection)
        fusion["evidence_projection"] = "query_centered_contiguous_source_body"
        record["retrieval_fusion"] = fusion
        prepared.append(record)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    covered_queries: set[str] = set()
    used = 0

    def try_add(record: dict[str, Any]) -> bool:
        nonlocal used
        key = _record_key(record)
        if key in selected_keys:
            return False
        size = len(_record_content(record).encode("utf-8", errors="replace"))
        if selected and used + size > budget:
            return False
        selected.append(record)
        selected_keys.add(key)
        used += size
        covered_queries.update(
            str(item)
            for item in record["retrieval_fusion"].get("matched_queries", ())
            if str(item)
        )
        return True

    # First preserve breadth: one strongest relevant record for each research question.
    # Zero-relevance records have already been removed and therefore cannot be revived
    # simply because a noisy provider returned something for that query.
    for query in source_queries:
        if query in covered_queries:
            continue
        for record in prepared:
            if query in record["retrieval_fusion"].get("matched_queries", ()) and try_add(record):
                break

    # Then spend remaining bytes by global fusion score.
    for record in prepared:
        try_add(record)

    return selected, used


def fuse_grounded_domain_evidence(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> dict[str, Any]:
    """Dedupe/rank evidence, reject zero relevance, then recompose under a byte budget."""

    del domain
    rows = [row for row in grounded.get("queries", ()) if isinstance(row, Mapping)]
    merged: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    duplicates = 0
    zero_relevance_dropped = 0

    for row in rows:
        query = str(row.get("query") or "").strip()
        raw_all = [
            record
            for record in row.get("evidence_records", ())
            if isinstance(record, Mapping) and _record_content(record)
        ]
        raw = [record for record in raw_all if _record_is_query_relevant(query, record)]
        zero_relevance_dropped += len(raw_all) - len(raw)
        unique: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for input_rank, record in enumerate(raw, 1):
            key = _record_key(record)
            if key in unique:
                duplicates += 1
            else:
                unique[key] = (input_rank, record)
        ranked = sorted(
            unique.items(),
            key=lambda item: (
                -_relevance(query, item[1][1]),
                item[1][0],
                item[0],
            ),
        )
        trace.append(
            {
                "query": query,
                "query_sha256": str(row.get("query_sha256") or _sha256_text(query)),
                "input_record_count": len(raw_all),
                "zero_relevance_dropped": len(raw_all) - len(raw),
                "unique_record_count": len(ranked),
                "github_provider_status": str(
                    row.get("github_provider_status") or "not_requested"
                ),
                "github_saturation_reason": str(
                    row.get("github_saturation_reason") or ""
                ),
                "retrieval_errors": list(row.get("retrieval_errors") or ()),
            }
        )
        for rank, (key, (_input_rank, record)) in enumerate(ranked, 1):
            item = merged.get(key)
            if item is None:
                item = {
                    "record": dict(record),
                    "rrf": 0.0,
                    "coverage": 0.0,
                    "queries": [],
                    "providers": set(),
                }
                merged[key] = item
            else:
                duplicates += 1
            item["rrf"] += 1.0 / (_RRF_K + rank)
            item["coverage"] = max(float(item["coverage"]), _relevance(query, record))
            if query and query not in item["queries"]:
                item["queries"].append(query)
            item["providers"].add(_provider_family(record))

    ranked_records: list[dict[str, Any]] = []
    for key, item in merged.items():
        record = dict(item["record"])
        providers = sorted(str(value) for value in item["providers"])
        query_hits = len(item["queries"])
        score = (
            float(item["rrf"])
            + float(item["coverage"])
            + 0.05 * max(0, query_hits - 1)
            + 0.02 * max(0, len(providers) - 1)
        )
        record["retrieval_fusion"] = {
            "record_key": key,
            "rrf_score": round(float(item["rrf"]), 8),
            "max_query_coverage": round(float(item["coverage"]), 6),
            "query_hits": query_hits,
            "matched_queries": list(item["queries"]),
            "provider_families": providers,
            "combined_score": round(score, 8),
        }
        ranked_records.append(record)
    ranked_records.sort(
        key=lambda record: (
            -float(record["retrieval_fusion"]["combined_score"]),
            _record_key(record),
        )
    )

    source_queries = [
        str(row.get("query") or "").strip()
        for row in rows
        if str(row.get("query") or "").strip()
    ]
    records, selected_bytes = _bounded_records(ranked_records, source_queries)
    result = dict(grounded)
    result["queries"] = [
        {
            "query": "domain fused evidence",
            "query_sha256": _sha256_text("\n".join(source_queries)),
            "source_queries": source_queries,
            "evidence_records": records,
            "content_record_count": len(records),
            "github_record_count": sum(
                1
                for record in records
                if "github" in record["retrieval_fusion"]["provider_families"]
            ),
            "github_provider_status": (
                "available"
                if any(
                    str(row.get("github_provider_status") or "") == "available"
                    for row in rows
                )
                else "not_requested"
            ),
            "github_saturation_reason": "",
            "retrieval_errors": [
                str(error)
                for row in trace
                for error in row["retrieval_errors"]
            ][:12],
        }
    ]
    result["retrieval_trace"] = trace
    result["fusion"] = {
        "schema_version": "mmm/pre-design-evidence-fusion-v4",
        "algorithm": "source_body_only+zero_relevance_gate+exact_dedupe+query_rank+rrf+coverage_first_byte_budget",
        "query_count": len(rows),
        "queries_with_content": sum(1 for row in trace if row["unique_record_count"]),
        "query_coverage_ratio": (
            round(
                sum(1 for row in trace if row["unique_record_count"]) / len(trace),
                6,
            )
            if trace
            else 0.0
        ),
        "unique_record_count": len(ranked_records),
        "selected_record_count": len(records),
        "dropped_record_count": max(0, len(ranked_records) - len(records)),
        "zero_relevance_dropped_record_count": zero_relevance_dropped,
        "selected_content_bytes": selected_bytes,
        "evidence_byte_budget": _evidence_byte_budget(),
        "duplicate_record_count": duplicates,
        "provider_families": sorted(
            {
                family
                for record in records
                for family in record["retrieval_fusion"]["provider_families"]
            }
        ),
        "all_unique_records_preserved": len(records) == len(ranked_records),
    }
    return result


__all__ = [
    "_is_retrieval_query",
    "_record_content",
    "_record_is_query_relevant",
    "_relevance",
    "_sha256_text",
    "_stable_text",
    "fuse_grounded_domain_evidence",
]
