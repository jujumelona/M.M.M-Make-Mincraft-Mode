from __future__ import annotations

"""Install strict byte-bounded source-observation paging for small coder turns.

ProjectIndex pages are independently bounded, but converting those pages into exact
source observations adds provenance metadata and global anchors. That second envelope
must be budgeted again or a small-model request can exceed the host-selected context
budget even though every underlying index page was valid.
"""

from functools import wraps
from types import ModuleType
from typing import Any

_MARKER = "_mmm_exact_source_observation_budget_v1"
_CONTEXT_BUDGET_MARKER = "_mmm_atomic_source_context_budget_v1"
_MIN_FRAGMENT_BYTES = 128
_PAGE_RESERVE_BYTES = 128
_ATOMIC_SOURCE_CONTEXT_BYTES = 4 * 1024


def _utf8_fragments(text: str, max_bytes: int) -> tuple[tuple[int, bytes], ...]:
    """Split UTF-8 text without breaking code points and retain byte offsets."""

    limit = max(_MIN_FRAGMENT_BYTES, int(max_bytes))
    fragments: list[tuple[int, bytes]] = []
    chars: list[str] = []
    size = 0
    start = 0
    for char in text:
        encoded = char.encode("utf-8")
        if chars and size + len(encoded) > limit:
            payload = "".join(chars).encode("utf-8")
            fragments.append((start, payload))
            start += len(payload)
            chars = []
            size = 0
        chars.append(char)
        size += len(encoded)
    if chars:
        fragments.append((start, "".join(chars).encode("utf-8")))
    return tuple(fragments)


def _split_records(
    target: ModuleType,
    records: list[dict[str, Any]],
    byte_budget: int,
) -> list[dict[str, Any]]:
    """Make every exact-source record small enough to coexist with page metadata."""

    fragment_budget = max(_MIN_FRAGMENT_BYTES, min(1024, byte_budget // 5))
    result: list[dict[str, Any]] = []
    for record in records:
        if target._json_size(record) <= fragment_budget + 512:
            result.append(record)
            continue
        text = str(record.get("text", ""))
        base_start = int(record.get("content_start_bytes", 0) or 0)
        for relative_start, payload in _utf8_fragments(text, fragment_budget):
            result.append(
                target._exact_observation(
                    path=str(record.get("path", "")),
                    sha256=str(record.get("sha256", "")),
                    start=base_start + relative_start,
                    content=payload,
                    source_page=int(record.get("source_page_index", 0) or 0),
                )
            )
    return result


def _bounded_pages(
    target: ModuleType,
    ledger: dict[str, Any],
    *,
    query: str,
    byte_budget: int,
) -> tuple[dict[str, Any], ...]:
    if type(byte_budget) is not int or byte_budget < 1024:
        raise target.CustomModuleGenerationError(
            "Source-observation byte budget must be an integer >= 1024."
        )

    records = _split_records(target, list(ledger["records"]), byte_budget)
    query_tokens = target._query_tokens(query)
    ranked = sorted(
        records,
        key=lambda record: (
            -target._observation_score(record, query_tokens),
            record["path"],
            record["content_start_bytes"],
            record["observation_id"],
        ),
    )

    safe_budget = max(1024, byte_budget - _PAGE_RESERVE_BYTES)
    anchors: list[dict[str, Any]] = []
    anchor_target = max(1024, safe_budget * 3 // 5)
    for record in ranked:
        candidate = target._observation_page_payload(
            receipt=ledger["receipt"],
            page_index=0,
            page_count=999999,
            anchors=[*anchors, record],
            records=[],
            complete=False,
        )
        if target._json_size(candidate) <= anchor_target:
            anchors.append(record)

    if ranked and not anchors:
        candidate = target._observation_page_payload(
            receipt=ledger["receipt"],
            page_index=0,
            page_count=999999,
            anchors=[ranked[0]],
            records=[],
            complete=False,
        )
        if target._json_size(candidate) > safe_budget:
            raise target.CustomModuleGenerationError(
                "Exact-source provenance metadata cannot fit the configured coder context budget."
            )
        anchors.append(ranked[0])

    anchor_ids = {record["observation_id"] for record in anchors}
    remaining = [
        record for record in ranked if record["observation_id"] not in anchor_ids
    ]
    pages: list[dict[str, Any]] = []
    cursor = 0

    while cursor < len(remaining) or not pages:
        page_records: list[dict[str, Any]] = []
        while cursor < len(remaining):
            candidate_records = [*page_records, remaining[cursor]]
            candidate = target._observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=999999,
                anchors=anchors,
                records=candidate_records,
                complete=False,
            )
            if target._json_size(candidate) > safe_budget:
                break
            page_records.append(remaining[cursor])
            cursor += 1

        if cursor < len(remaining) and not page_records:
            if anchors:
                demoted = anchors.pop()
                anchor_ids.discard(demoted["observation_id"])
                remaining.insert(cursor, demoted)
                continue
            raise target.CustomModuleGenerationError(
                "Exact-source observation cannot fit the configured coder context budget."
            )

        pages.append(
            target._observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=0,
                anchors=anchors,
                records=page_records,
                complete=False,
            )
        )
        if cursor >= len(remaining):
            break

    page_count = len(pages)
    for index, page in enumerate(pages):
        page["page_count"] = page_count
        page["complete"] = index == page_count - 1
        if target._json_size(page) > byte_budget:
            raise target.CustomModuleGenerationError(
                "Host source-observation context page exceeded its byte budget after finalization."
            )
    return tuple(pages)


def _install_context_budget(target_module: ModuleType) -> None:
    """Keep the initial source page atomic even when the live model context is huge."""

    current = target_module._coder_project_context_budget
    if bool(getattr(current, _CONTEXT_BUDGET_MARKER, False)):
        return

    @wraps(current)
    def atomic_source_context_budget(
        router: Any,
        policy: Any,
        *,
        fast_mode: bool,
    ) -> int:
        live_budget = int(current(router, policy, fast_mode=fast_mode))
        host_budget = max(1024, int(getattr(policy, "model_context_bytes", 1024)))
        return min(_ATOMIC_SOURCE_CONTEXT_BYTES, host_budget, max(1024, live_budget))

    setattr(atomic_source_context_budget, _CONTEXT_BUDGET_MARKER, True)
    target_module._coder_project_context_budget = atomic_source_context_budget


def install(target_module: ModuleType | None = None) -> None:
    """Install the source-page and initial-context budgets exactly once."""

    if target_module is None:
        from . import custom_module_generator as target_module

    _install_context_budget(target_module)

    current = target_module._observation_context_pages
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def strict_observation_context_pages(
        ledger: dict[str, Any],
        *,
        query: str,
        byte_budget: int,
    ) -> tuple[dict[str, Any], ...]:
        pages = current(ledger, query=query, byte_budget=byte_budget)
        if all(target_module._json_size(page) <= byte_budget for page in pages):
            return pages
        return _bounded_pages(
            target_module,
            ledger,
            query=query,
            byte_budget=byte_budget,
        )

    setattr(strict_observation_context_pages, _MARKER, True)
    target_module._observation_context_pages = strict_observation_context_pages


__all__ = ["install"]
