from __future__ import annotations

"""Bounded, source-only semantic partition proposal.

The model is not a planning authority.  It may only copy contiguous source substrings into
ordered segments when punctuation/newlines do not already provide structural boundaries.
The host resolves every copied substring back to the original request and rejects any
paraphrase, omission, overlap, or non-whitespace gap.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class SourcePartitionError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSegment:
    start: int
    end: int
    text: str


_PREFIX = "SEGMENT:"


def _parse_copied_segments(text: str) -> tuple[str, ...]:
    segments: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith(_PREFIX):
            continue
        value = line[len(_PREFIX) :].strip()
        if value:
            segments.append(value)
    if not segments:
        raise SourcePartitionError("semantic source partition returned no copied segments")
    return tuple(segments)


def _locate_exact_partition(prompt: str, copied: Sequence[str]) -> tuple[SourceSegment, ...]:
    cursor = 0
    result: list[SourceSegment] = []
    for index, segment in enumerate(copied):
        if not segment:
            raise SourcePartitionError(f"semantic source segment {index} is empty")
        start = prompt.find(segment, cursor)
        if start < 0:
            raise SourcePartitionError(
                f"semantic source segment {index} is not an exact copy of the request"
            )
        gap = prompt[cursor:start]
        if gap.strip():
            raise SourcePartitionError(
                f"semantic source segment {index} omitted non-whitespace request text"
            )
        end = start + len(segment)
        result.append(SourceSegment(start=start, end=end, text=segment))
        cursor = end
    if prompt[cursor:].strip():
        raise SourcePartitionError("semantic source partition omitted trailing request text")
    if len(result) > 1:
        for previous, current in zip(result, result[1:]):
            gap = prompt[previous.end : current.start]
            if not gap or not gap.isspace():
                raise SourcePartitionError(
                    "semantic source partition may split only at existing whitespace boundaries"
                )
    return tuple(result)


def propose_source_partition(router: Any, prompt: str) -> tuple[SourceSegment, ...]:
    """Use one bounded model turn to propose source boundaries and verify them exactly."""

    if not str(prompt).strip():
        raise SourcePartitionError("semantic source partition requires a non-empty request")
    generate = getattr(router, "generate_text", None)
    if not callable(generate):
        raise SourcePartitionError("semantic source partition requires a planner text backend")

    response = generate(
        "planner",
        [
            {
                "role": "system",
                "content": (
                    "Partition one user request into the smallest independently observable "
                    "gameplay requirements. You are only a source-boundary locator. Copy text "
                    "verbatim; do not translate, paraphrase, summarize, label capabilities, "
                    "invent behavior, or omit any non-whitespace text. Output only one or more "
                    "lines in the form `SEGMENT: <exact contiguous source substring>`. Preserve "
                    "the original order. Boundaries may occur only at whitespace already present "
                    "in the request. If the request is already one atomic behavior, return one "
                    "SEGMENT line containing the exact request text."
                ),
            },
            {
                "role": "user",
                "content": "SOURCE REQUEST:\n" + prompt,
            },
        ],
        response_format="text",
        tool_stage="planning",
        enable_tools=False,
    )
    return _locate_exact_partition(prompt, _parse_copied_segments(response))


__all__ = ["SourcePartitionError", "SourceSegment", "propose_source_partition"]
