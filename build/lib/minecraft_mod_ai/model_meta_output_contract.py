from __future__ import annotations

"""Reject internal reasoning and contradictions in accepted design fields."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .research_evidence_state import current_grounded_evidence

_META_MARKERS = (
    re.compile(r"<\s*/?\s*think\b", re.IGNORECASE),
    re.compile(r"\bi\s+need\s+to\b", re.IGNORECASE),
    re.compile(r"\bi\s+should\b", re.IGNORECASE),
    re.compile(r"\bthe\s+user\s+wants\b", re.IGNORECASE),
    re.compile(r"\bthe\s+user\s+asked\b", re.IGNORECASE),
    re.compile(r"\bbranch[- ]policy\b", re.IGNORECASE),
    re.compile(r"\bmain[- ]only\s+(?:branch|policy)\b", re.IGNORECASE),
    re.compile(r"\brepository\s+(?:branch|policy)\s+(?:check|review)\b", re.IGNORECASE),
)
_BLANKET_NO_EVIDENCE_MARKERS = (
    re.compile(r"\bno_relevant_external_evidence\b", re.IGNORECASE),
    re.compile(
        r"\bno\s+(?:relevant\s+)?external\s+evidence\s+(?:was\s+|is\s+)?found\s+for\s+(?:any|all)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:relevant\s+)?external\s+evidence\s+(?:exists|is available)\s+for\s+(?:any|all)\b",
        re.IGNORECASE,
    ),
)
_GUARDED_FIELDS = frozenset(
    {
        "title",
        "pitch",
        "core_loop",
        "progression",
        "combat",
        "mod_context",
        "modules",
        "assets",
        "acceptance_tests",
        "art_direction",
    }
)


def _text_values(value: Any):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _text_values(child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _text_values(child)


def contains_internal_model_meta(value: Any) -> bool:
    return any(
        pattern.search(text)
        for text in _text_values(value)
        for pattern in _META_MARKERS
    )


def contradicts_grounded_research(value: Any) -> bool:
    """Reject only blanket absence claims that contradict host materialization.

    A model may still state that evidence is insufficient for one specific fact. What it
    cannot do is overwrite a request-local host receipt containing grounded source bodies
    with a global ``no_relevant_external_evidence`` conclusion.
    """

    state = current_grounded_evidence()
    if not state.available:
        return False
    return any(
        pattern.search(text)
        for text in _text_values(value)
        for pattern in _BLANKET_NO_EVIDENCE_MARKERS
    )


def assert_design_field_clean(field: str, value: Any) -> None:
    if field not in _GUARDED_FIELDS:
        return
    if contains_internal_model_meta(value):
        raise ValueError(
            f"game_design.{field} contains internal model reasoning/meta output and cannot be accepted"
        )
    if contradicts_grounded_research(value):
        state = current_grounded_evidence()
        raise ValueError(
            f"game_design.{field} contradicts host-grounded RAG evidence "
            f"(source_bodies={state.source_body_count}, evidence_cards={state.evidence_card_count})"
        )


__all__ = [
    "assert_design_field_clean",
    "contains_internal_model_meta",
    "contradicts_grounded_research",
]
