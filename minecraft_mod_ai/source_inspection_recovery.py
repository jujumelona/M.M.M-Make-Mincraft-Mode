"""Reuse per-run immutable failure evidence without re-fetching broken archives."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def cached_archive(cache: Mapping, url: str, error_type):
    value = cache[url]
    if isinstance(value, error_type):
        raise error_type(str(value))
    return value


def cache_archive_failure(cache: dict, url: str, error: Exception, error_type) -> None:
    if (isinstance(error, error_type)
            and str(error).startswith(("SOURCE_ARCHIVE_", "SOURCE_MEMBER_"))
            and url not in cache):
        cache[url] = error


def usable_source_links(candidates: Sequence[Mapping], unusable: set[str]) -> list[Mapping]:
    return [
        hit for hit in candidates
        if hit.get("source_url") and str(hit["source_url"]) not in unusable
    ]


def unusable_source_links(receipts: Sequence[Mapping]) -> set[str]:
    return {
        str(item["source_url"]) for item in receipts if item.get("source_url")
        and (item.get("inspection_status") == "license_blocked"
             or (item.get("inspection_status") == "inspection_error"
                 and str(item.get("error") or "").startswith((
                     "SOURCE_ARCHIVE_", "SOURCE_MEMBER_",
                 ))))
    }

def inspect_shortlist_until_reference(capability, available_links, *, router, cache, inspect, max_candidates=2):
    """Check a second source only when the first lacks a verified semantic reference."""
    receipts = []
    for candidate in available_links[:max_candidates]:
        current = inspect(capability, [candidate], router=router, cache=cache, max_candidates=1)
        receipts.extend(current)
        if any(
            row.get("inspection_status") == "inspected"
            and (row.get("semantic") or {}).get("status") == "semantic_reviewed"
            and (row.get("semantic") or {}).get("intent") in {"reference", "adapt", "reuse"}
            for row in current
        ):
            break
    return receipts
