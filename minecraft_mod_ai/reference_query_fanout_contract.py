from __future__ import annotations

"""Bound reference retrieval at the query row, never at nested provider executors.

The host owns query fan-out.  Each row performs a deterministic Wikipedia-first lookup
and consults GitHub only when Wikipedia produced no usable body evidence.  This avoids
provider-level nested executors and prevents expensive repository retrieval when an
encyclopedia source already grounds the requested reference.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from types import ModuleType
from typing import Any

_MARKER = "_mmm_reference_query_fanout_contract"
_DEFAULT_QUERY_WORKERS = 3


def install(target_module: ModuleType | None = None) -> None:
    if target_module is None:
        from . import reference_source_research as target_module

    current = target_module.retrieve_reference_grounded_evidence
    if bool(getattr(current, _MARKER, False)):
        return

    original_wikipedia = target_module._wikipedia_sources
    original_github = target_module._github_reference_sources

    def wikipedia_one(query: str):
        anchors, search_plan = target_module._reference_search_plan([query])
        return original_wikipedia(search_plan, anchors)

    def github_one(query: str):
        anchors, search_plan = target_module._reference_search_plan([query])
        return original_github(search_plan, anchors)

    # The public-private helper surface is intentionally one query per call.  Tests and
    # host orchestration can therefore prove there is exactly one concurrency owner.
    target_module._wikipedia_sources = wikipedia_one
    target_module._github_reference_sources = github_one
    target_module._MAX_QUERY_WORKERS = int(
        getattr(target_module, "_MAX_QUERY_WORKERS", _DEFAULT_QUERY_WORKERS)
        or _DEFAULT_QUERY_WORKERS
    )

    def retrieve_query_row(query: str) -> dict[str, Any]:
        clean_query = target_module._text(query)
        if not clean_query:
            return {
                "query": "",
                "evidence_records": [],
                "content_record_count": 0,
                "provider_receipts": {},
                "retrieval_errors": [],
                "provider_policy": "wikipedia_then_github_fallback",
            }

        wikipedia_rows, wikipedia_receipt, wikipedia_error = target_module._retrieve_provider(
            "wikipedia",
            lambda: target_module._wikipedia_sources(clean_query),
        )
        receipts: dict[str, Any] = {"wikipedia": wikipedia_receipt}
        errors: list[dict[str, str]] = []
        if wikipedia_error is not None:
            errors.append(wikipedia_error)

        records = list(wikipedia_rows)
        if records:
            receipts["github_reference"] = {
                "provider": "github_reference",
                "status": "skipped_wikipedia_has_evidence",
                "result_count": 0,
                "attempts": 0,
                "policy": "wikipedia_empty_fallback_only",
            }
        else:
            github_rows, github_receipt, github_error = target_module._retrieve_provider(
                "github_reference",
                lambda: target_module._github_reference_sources(clean_query),
            )
            receipts["github_reference"] = github_receipt
            records.extend(github_rows)
            if github_error is not None:
                errors.append(github_error)

        unique = target_module._dedupe(records)
        _anchors, search_plan = target_module._reference_search_plan([clean_query])
        return {
            "query": clean_query,
            "authored_queries": [clean_query],
            "query_sha256": target_module._sha(clean_query),
            "evidence_records": unique,
            "content_record_count": len(unique),
            "provider_receipts": receipts,
            "retrieval_errors": errors,
            "provider_policy": "wikipedia_then_github_fallback",
            "search_plan": search_plan,
        }

    target_module._retrieve_query_row = retrieve_query_row

    @wraps(current)
    def retrieve_query_owned(queries):
        query_list = list(
            dict.fromkeys(
                target_module._text(raw)
                for raw in queries
                if target_module._text(raw)
            )
        )
        anchors, search_plan = target_module._reference_search_plan(query_list)
        if not query_list:
            return {
                "schema_version": "mmm/reference-grounded-evidence-v3",
                "retrieval_strategy": "query_owned_wikipedia_github_fallback",
                "inferred_reference_names": anchors,
                "search_plan": search_plan,
                "queries": [],
            }

        workers = max(
            1,
            min(
                len(query_list),
                int(getattr(target_module, "_MAX_QUERY_WORKERS", _DEFAULT_QUERY_WORKERS)),
            ),
        )
        if workers == 1:
            rows = [target_module._retrieve_query_row(query) for query in query_list]
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="mmm-reference-query",
            ) as executor:
                # executor.map preserves authored query order while still overlapping rows.
                rows = list(executor.map(target_module._retrieve_query_row, query_list))
        return {
            "schema_version": "mmm/reference-grounded-evidence-v3",
            "retrieval_strategy": "query_owned_wikipedia_github_fallback",
            "inferred_reference_names": anchors,
            "search_plan": search_plan,
            "queries": rows,
        }

    setattr(retrieve_query_owned, _MARKER, True)
    retrieve_query_owned.__wrapped__ = current
    target_module.retrieve_reference_grounded_evidence = retrieve_query_owned


__all__ = ["install"]
