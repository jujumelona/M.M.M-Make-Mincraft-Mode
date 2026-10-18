"""Reuse per-run immutable failure evidence without re-fetching broken archives."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def cached_archive(cache: Mapping, url: str):
    from .curseforge_semantic_source_review import SourceInspectionError

    value = cache[url]
    if isinstance(value, SourceInspectionError):
        raise SourceInspectionError(str(value))
    return value


def cache_archive_failure(cache: dict, url: str, error: Exception) -> None:
    from .curseforge_semantic_source_review import SourceInspectionError

    if (isinstance(error, SourceInspectionError)
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
