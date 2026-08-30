from __future__ import annotations

"""Pure multi-query/provider fusion primitives for pre-design retrieval."""

import hashlib
import re
from collections.abc import Mapping
from typing import Any

_RRF_K = 60.0
_QUERY_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")


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
    wanted = _tokens(query)
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


def fuse_grounded_domain_evidence(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact-content dedupe, query-local lexical rank, then RRF-style fusion."""

    del domain
    rows = [row for row in grounded.get("queries", ()) if isinstance(row, Mapping)]
    merged: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    duplicates = 0

    for row in rows:
        query = str(row.get("query") or "").strip()
        raw = [
            record
            for record in row.get("evidence_records", ())
            if isinstance(record, Mapping) and _record_content(record)
        ]
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
                "input_record_count": len(raw),
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

    records: list[dict[str, Any]] = []
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
        records.append(record)
    records.sort(
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
        "schema_version": "mmm/pre-design-evidence-fusion-v1",
        "algorithm": "exact_content_dedupe+query_local_lexical_rank+rrf",
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
        "unique_record_count": len(records),
        "duplicate_record_count": duplicates,
        "provider_families": sorted(
            {
                family
                for record in records
                for family in record["retrieval_fusion"]["provider_families"]
            }
        ),
        "all_unique_records_preserved": True,
    }
    return result


__all__ = [
    "_is_retrieval_query",
    "_record_content",
    "_sha256_text",
    "_stable_text",
    "fuse_grounded_domain_evidence",
]
