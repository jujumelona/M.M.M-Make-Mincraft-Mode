from __future__ import annotations

"""Own reference-research concurrency at the query-row layer.

Reference providers stay sequential inside one authored query. Concurrency is spent only
across independent query rows, which avoids nested provider fan-out while retaining the
identity-first search plan. Query-row scheduling uses the shared deadline executor so one
stalled transport cannot make the whole research stage wait forever.
"""

import inspect
import threading
from typing import Any, Callable

from .deadline_executor import ParallelExecutionTimeout, iter_completed_with_deadlines


def install(reference_module: Any) -> None:
    if getattr(reference_module, "_mmm_query_parallelism_contract", False):
        return

    if not hasattr(reference_module, "_MAX_QUERY_WORKERS"):
        reference_module._MAX_QUERY_WORKERS = 3

    row_context = threading.local()

    def _invoke_provider(function: Callable[..., Any], query: str, search_plan, anchors):
        """Support canonical two-argument providers and one-argument test adapters."""
        try:
            parameter_count = len(inspect.signature(function).parameters)
        except (TypeError, ValueError):
            parameter_count = 2
        if parameter_count <= 1:
            return function(query)
        return function(search_plan, anchors)

    def _retrieve(
        provider: str,
        function: Callable[..., Any],
        query: str,
        search_plan,
        anchors,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str] | None]:
        attempts = max(1, int(getattr(reference_module, "_MAX_PROVIDER_ATTEMPTS", 1)))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                found, raw_receipt = _invoke_provider(
                    function,
                    query,
                    search_plan,
                    anchors,
                )
                receipt = {**dict(raw_receipt), "attempts": attempt}
                return list(found), receipt, None
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        return (
            [],
            {
                "provider": provider,
                "status": "error",
                "result_count": 0,
                "attempts": attempts,
            },
            {
                "provider": provider,
                "error": f"{type(last_error).__name__}: {last_error}",
            },
        )

    def _retrieve_query_row(query: str) -> dict[str, Any]:
        normalized = reference_module._text(query)
        anchors = tuple(getattr(row_context, "anchors", ()))
        if anchors:
            _, search_plan = reference_module._reference_search_plan([normalized], anchors)
        else:
            _, search_plan = reference_module._reference_search_plan([normalized])
        search_plan = tuple(search_plan or (normalized,))

        records: list[dict[str, Any]] = []
        providers: dict[str, Any] = {}
        errors: list[dict[str, str]] = []

        wiki_found, wiki_receipt, wiki_error = _retrieve(
            "wikipedia",
            reference_module._wikipedia_sources,
            normalized,
            search_plan,
            anchors,
        )
        records.extend(wiki_found)
        providers["wikipedia"] = wiki_receipt
        if wiki_error is not None:
            errors.append(wiki_error)

        if wiki_found:
            providers["wikidata"] = {
                "provider": "wikidata",
                "status": "skipped_wikipedia_has_evidence",
                "result_count": 0,
                "policy": "identity_bridge_only_after_wikipedia_miss",
            }
            providers["github_reference"] = {
                "provider": "github_reference",
                "status": "skipped_wikipedia_has_evidence",
                "result_count": 0,
                "policy": "wikipedia_empty_fallback_only",
            }
        else:
            if anchors:
                wikidata_found, wikidata_receipt, wikidata_error = _retrieve(
                    "wikidata",
                    reference_module._wikidata_sources,
                    normalized,
                    search_plan,
                    anchors,
                )
                records.extend(wikidata_found)
                providers["wikidata"] = wikidata_receipt
                if wikidata_error is not None:
                    errors.append(wikidata_error)
            else:
                providers["wikidata"] = {
                    "provider": "wikidata",
                    "status": "skipped_no_identity_anchor",
                    "result_count": 0,
                }

            github_found, github_receipt, github_error = _retrieve(
                "github_reference",
                reference_module._github_reference_sources,
                normalized,
                search_plan,
                anchors,
            )
            records.extend(github_found)
            providers["github_reference"] = github_receipt
            if github_error is not None:
                errors.append(github_error)

        unique = reference_module._dedupe(records)
        return {
            "query": normalized,
            "authored_queries": [normalized],
            "query_sha256": reference_module._sha(normalized),
            "evidence_records": unique,
            "content_record_count": len(unique),
            "provider_receipts": providers,
            "retrieval_errors": errors,
            "provider_policy": "sequential_identity_first_then_github_empty_fallback",
            "search_plan": list(search_plan),
        }

    def _timeout_row(query: str, exc: ParallelExecutionTimeout) -> dict[str, Any]:
        normalized = reference_module._text(query)
        error = (
            "ParallelExecutionTimeout: "
            f"{exc.deadline_kind} deadline exceeded while retrieving query row"
        )
        return {
            "query": normalized,
            "authored_queries": [normalized],
            "query_sha256": reference_module._sha(normalized),
            "evidence_records": [],
            "content_record_count": 0,
            "provider_receipts": {
                "reference_query": {
                    "provider": "reference_query",
                    "status": "error",
                    "result_count": 0,
                    "deadline_kind": exc.deadline_kind,
                }
            },
            "retrieval_errors": [
                {"provider": "reference_query", "error": error}
            ],
            "provider_policy": "sequential_identity_first_then_github_empty_fallback",
            "search_plan": [normalized] if normalized else [],
        }

    def retrieve_reference_grounded_evidence(queries) -> dict[str, Any]:
        query_list = list(
            dict.fromkeys(
                reference_module._text(raw)
                for raw in queries
                if reference_module._text(raw)
            )
        )
        anchors, global_search_plan = reference_module._reference_search_plan(query_list)
        indexed_queries = list(enumerate(query_list))
        rows_by_index: dict[int, dict[str, Any]] = {}
        deadline_error: ParallelExecutionTimeout | None = None

        def run_row(item: tuple[int, str]) -> tuple[int, dict[str, Any]]:
            index, query = item
            row_context.anchors = tuple(anchors)
            try:
                return index, reference_module._retrieve_query_row(query)
            finally:
                try:
                    delattr(row_context, "anchors")
                except AttributeError:
                    pass

        if indexed_queries:
            workers = min(
                max(1, int(getattr(reference_module, "_MAX_QUERY_WORKERS", 3))),
                len(indexed_queries),
            )
            try:
                for _item, (index, row) in iter_completed_with_deadlines(
                    indexed_queries,
                    run_row,
                    max_workers=workers,
                    stage="mmm-reference-query",
                    sort_key=lambda item: item[0],
                ):
                    rows_by_index[index] = row
            except ParallelExecutionTimeout as exc:
                deadline_error = exc

        if deadline_error is not None:
            for index, query in indexed_queries:
                if index not in rows_by_index:
                    rows_by_index[index] = _timeout_row(query, deadline_error)

        rows = [rows_by_index[index] for index, _query in indexed_queries]
        return {
            "schema_version": "mmm/reference-grounded-evidence-v3",
            "retrieval_strategy": "identity_first_bounded_query_rows",
            "inferred_reference_names": list(anchors),
            "search_plan": list(global_search_plan),
            "queries": rows,
        }

    reference_module._retrieve_query_row = _retrieve_query_row
    reference_module.retrieve_reference_grounded_evidence = retrieve_reference_grounded_evidence
    reference_module._mmm_query_parallelism_contract = True


__all__ = ["install"]