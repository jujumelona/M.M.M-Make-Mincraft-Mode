from __future__ import annotations

"""Install bounded query-level reference retrieval without provider fan-out.

Independent authored queries may overlap because they are separate work items.  Within
one query, however, providers are sequential: Wikipedia is authoritative enough to stop
that row, and GitHub is used only when Wikipedia produced no evidence.  The richer
identity-wide Wikidata expansion remains owned by the existing implementation whenever a
stable common identity anchor can be inferred across the authored questions.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

_MARKER = "_mmm_query_level_reference_retrieval"
_DEFAULT_QUERY_WORKERS = 3


def _call_query_provider(module: Any, function: Callable[..., Any], query: str) -> Any:
    """Call legacy one-query providers and current sequence/anchor providers safely."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(query)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        parameter
        for parameter in positional
        if parameter.default is parameter.empty
    ]
    if len(required) >= 2:
        return function([query], [])
    return function(query)


def install(module: Any) -> None:
    if getattr(module.retrieve_reference_grounded_evidence, _MARKER, False):
        return

    identity_retrieve = module.retrieve_reference_grounded_evidence
    if not hasattr(module, "_MAX_QUERY_WORKERS"):
        module._MAX_QUERY_WORKERS = _DEFAULT_QUERY_WORKERS

    def skipped(provider: str, status: str) -> dict[str, Any]:
        return {
            "provider": provider,
            "status": status,
            "result_count": 0,
            "attempts": 0,
        }

    def retrieve_query_row(query: str) -> dict[str, Any]:
        wikipedia, wiki_receipt, wiki_error = module._retrieve_provider(
            "wikipedia",
            lambda: _call_query_provider(module, module._wikipedia_sources, query),
        )
        records = list(wikipedia)
        providers: dict[str, Any] = {"wikipedia": wiki_receipt}
        errors: list[dict[str, str]] = []
        if wiki_error is not None:
            errors.append(wiki_error)

        if records:
            providers["wikidata"] = skipped(
                "wikidata", "skipped_wikipedia_has_evidence"
            )
            providers["github_reference"] = skipped(
                "github_reference", "skipped_wikipedia_has_evidence"
            )
        else:
            # Unanchored query rows deliberately do not fan out to Wikidata.  The
            # identity-wide path below still uses Wikidata when a stable named anchor is
            # inferred across authored questions.
            providers["wikidata"] = skipped(
                "wikidata", "skipped_no_shared_identity_anchor"
            )
            github, github_receipt, github_error = module._retrieve_provider(
                "github_reference",
                lambda: _call_query_provider(
                    module, module._github_reference_sources, query
                ),
            )
            records.extend(github)
            providers["github_reference"] = github_receipt
            if github_error is not None:
                errors.append(github_error)

        unique = module._dedupe(records)
        return {
            "query": query,
            "authored_queries": [query],
            "query_sha256": module._sha(query),
            "evidence_records": unique,
            "content_record_count": len(unique),
            "provider_receipts": providers,
            "retrieval_errors": errors,
            "provider_policy": "sequential_identity_first_then_github_empty_fallback",
            "search_plan": [query],
        }

    def retrieve_reference_grounded_evidence(
        queries: Sequence[str],
    ) -> dict[str, Any]:
        query_list = list(
            dict.fromkeys(module._text(raw) for raw in queries if module._text(raw))
        )
        anchors, search_plan = module._reference_search_plan(query_list)
        if anchors:
            return identity_retrieve(query_list)
        if not query_list:
            return {
                "schema_version": "mmm/reference-grounded-evidence-v3",
                "retrieval_strategy": "query_rows_bounded_parallel",
                "inferred_reference_names": [],
                "search_plan": [],
                "queries": [],
            }

        indexed = tuple(enumerate(query_list))
        completed: dict[int, dict[str, Any]] = {}
        try:
            for (index, _query), row in module.iter_completed_with_deadlines(
                indexed,
                lambda item: module._retrieve_query_row(item[1]),
                max_workers=max(1, min(len(indexed), int(module._MAX_QUERY_WORKERS))),
                stage="mmm-reference-query",
                sort_key=lambda item: item[0],
            ):
                completed[index] = row
        except module.ParallelExecutionTimeout as exc:
            timed_out = exc.item
            timed_out_index = (
                int(timed_out[0])
                if isinstance(timed_out, tuple) and timed_out
                else -1
            )
            raise RuntimeError(
                "Reference query retrieval deadline exceeded: "
                f"index={timed_out_index}; kind={exc.deadline_kind}"
            ) from exc

        rows = [completed[index] for index in range(len(query_list))]
        return {
            "schema_version": "mmm/reference-grounded-evidence-v3",
            "retrieval_strategy": "query_rows_bounded_parallel",
            "inferred_reference_names": [],
            "search_plan": search_plan,
            "queries": rows,
        }

    setattr(retrieve_query_row, _MARKER, True)
    setattr(retrieve_reference_grounded_evidence, _MARKER, True)
    retrieve_reference_grounded_evidence.__wrapped__ = identity_retrieve
    module._retrieve_query_row = retrieve_query_row
    module.retrieve_reference_grounded_evidence = retrieve_reference_grounded_evidence


__all__ = ["install"]
