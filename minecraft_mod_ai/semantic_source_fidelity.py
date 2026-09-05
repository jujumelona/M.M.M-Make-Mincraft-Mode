from __future__ import annotations

"""Language-neutral source-fidelity contract for semantic extraction.

The semantic model may decide *meaning*, but it may not silently discard authored text.
For each host-owned source clause its returned semantic leaves must form a non-overlapping
partition of all authored non-whitespace/non-punctuation characters.  This is deliberately
language-neutral: the host does not maintain Korean/English stop-word lists or phrase
special cases.  Context/connective text is assigned to the adjacent semantic span rather
than being dropped.

A failed partition is repairable semantic-model output.  The bounded semantic batching
owner may make one diagnostic-guided retry; after that it must fail closed rather than
inventing missing semantics.
"""

import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _authored_character(character: str) -> bool:
    if character.isspace():
        return False
    category = unicodedata.category(character)
    return not category.startswith(("P", "Z"))


def _ranges(positions: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not positions:
        return ()
    ordered = sorted(set(int(position) for position in positions))
    start = previous = ordered[0]
    result: list[tuple[int, int]] = []
    for position in ordered[1:]:
        if position == previous + 1:
            previous = position
            continue
        result.append((start, previous + 1))
        start = previous = position
    result.append((start, previous + 1))
    return tuple(result)


def _diagnostic(
    code: str,
    *,
    clause_index: int,
    text: str,
    absolute_clause_start: int,
    positions: Sequence[int],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spans = []
    for start, end in _ranges(positions):
        local_start = max(0, start - absolute_clause_start)
        local_end = max(local_start, end - absolute_clause_start)
        spans.append(
            {
                "char_start": start,
                "char_end": end,
                "text": text[local_start:local_end],
            }
        )
    payload: dict[str, Any] = {
        "error_code": code,
        "clause_index": clause_index,
        "spans": spans,
    }
    if details:
        payload["details"] = dict(details)
    return payload


def validate_semantic_source_partition(
    nodes: Sequence[Mapping[str, Any]],
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return diagnostics when semantic source spans drop or double-own authored text."""

    diagnostics: list[dict[str, Any]] = []
    nodes_by_clause: dict[int, list[Mapping[str, Any]]] = {}
    for node in nodes:
        clause_index = int(node.get("source_clause_index", -1))
        nodes_by_clause.setdefault(clause_index, []).append(node)

    for clause in clauses:
        clause_index = int(clause["clause_index"])
        clause_start = int(clause["char_start"])
        clause_end = int(clause["char_end"])
        text = str(clause["text"])
        authored_positions = {
            clause_start + offset
            for offset, character in enumerate(text)
            if _authored_character(character)
        }
        ownership: dict[int, list[int]] = {}

        for node_index, node in enumerate(nodes_by_clause.get(clause_index, ())):
            start = int(node.get("source_start", -1))
            end = int(node.get("source_end", -1))
            if not (clause_start <= start < end <= clause_end):
                diagnostics.append(
                    {
                        "error_code": "REQ_SOURCE_PARTITION_BOUNDS",
                        "clause_index": clause_index,
                        "node_index": node_index,
                        "source_start": start,
                        "source_end": end,
                        "expected_clause_span": [clause_start, clause_end],
                    }
                )
                continue
            for position in range(start, end):
                local = position - clause_start
                if 0 <= local < len(text) and _authored_character(text[local]):
                    ownership.setdefault(position, []).append(node_index)

        uncovered = sorted(authored_positions - set(ownership))
        if uncovered:
            diagnostics.append(
                _diagnostic(
                    "REQ_SOURCE_PARTITION_GAP",
                    clause_index=clause_index,
                    text=text,
                    absolute_clause_start=clause_start,
                    positions=uncovered,
                    details={
                        "contract": (
                            "semantic source spans must collectively account for every "
                            "authored non-whitespace/non-punctuation character"
                        )
                    },
                )
            )

        overlaps = sorted(
            position for position, owners in ownership.items() if len(owners) > 1
        )
        if overlaps:
            diagnostics.append(
                _diagnostic(
                    "REQ_SOURCE_PARTITION_OVERLAP",
                    clause_index=clause_index,
                    text=text,
                    absolute_clause_start=clause_start,
                    positions=overlaps,
                    details={
                        "contract": (
                            "each authored character has exactly one semantic owner; "
                            "split adjacent behaviors instead of nesting source anchors"
                        )
                    },
                )
            )

    return tuple(diagnostics)


_FIDELITY_SYSTEM_RULE = (
    "SOURCE-FIDELITY CONTRACT: source_anchor is not a keyword label. Across all semantic "
    "leaves for each supplied clause, choose exact contiguous source anchors whose spans "
    "partition every authored non-whitespace/non-punctuation character exactly once. "
    "Include connective/context text in one adjacent span; do not leave source words "
    "unowned and do not overlap anchors. Split independently observable behaviors before "
    "classification. capability_id must represent every authored behavior inside its own "
    "span; never substitute a prerequisite/base entity/state capability for a directly "
    "authored interaction merely because the prerequisite is also needed. The host will "
    "add prerequisites after semantic approval."
)


def _augment_messages(
    messages: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    instruction = _FIDELITY_SYSTEM_RULE
    if diagnostics:
        instruction += (
            " Previous output violated the host source-fidelity gate. Repair only the "
            "semantic decomposition/classification and source anchors using these host "
            "diagnostics: "
            + _canonical(list(diagnostics))
        )
    if result and str(result[0].get("role") or "") == "system":
        result[0]["content"] = str(result[0].get("content") or "") + "\n\n" + instruction
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result


class _TextRouterProxy:
    def __init__(self, router: Any, diagnostics: Sequence[Mapping[str, Any]]) -> None:
        self._router = router
        self._diagnostics = tuple(diagnostics)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def generate_text(self, role: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any:
        return self._router.generate_text(
            role,
            _augment_messages(messages, self._diagnostics),
            **kwargs,
        )


class _ToolRouterProxy(_TextRouterProxy):
    def generate_tool_decision(
        self,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Any:
        return self._router.generate_tool_decision(
            role,
            _augment_messages(messages, self._diagnostics),
            **kwargs,
        )


def fidelity_router(
    router: Any,
    *,
    diagnostics: Sequence[Mapping[str, Any]] = (),
) -> Any:
    """Wrap a router without changing its tool-vs-text routing capability surface."""

    if callable(getattr(router, "generate_tool_decision", None)):
        return _ToolRouterProxy(router, diagnostics)
    return _TextRouterProxy(router, diagnostics)


__all__ = [
    "fidelity_router",
    "validate_semantic_source_partition",
]
