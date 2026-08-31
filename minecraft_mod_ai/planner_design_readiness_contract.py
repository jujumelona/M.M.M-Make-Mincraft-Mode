from __future__ import annotations

"""Fail-closed requirement-to-design readiness without a second design owner.

The canonical game-design producer, Markdown parser, module representation, and coverage
validator live in ``agentic_research_game_design``. This contract only exposes readiness
helpers for callers that already depend on them and preserves the lossless semantic-span
fix. It must not mutate the design schema, replace section messages, replace the section
parser, or wrap the design generator.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import agentic_research_game_design as _design
from . import evidence_first_planning as _evidence

_INSTALLED = False


def _active_requirement_ledger(prompt: str) -> tuple[dict[str, Any], ...]:
    return _design._active_requirement_ledger(prompt)


def _nonempty_text_list(value: Any, *, field: str) -> list[str]:
    return _design._nonempty_text_list(value, field=field)


def _strict_validate_section_types(
    section: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    ledger_ids = tuple(
        item["requirement_id"]
        for item in _active_requirement_ledger("")
        if str(item.get("requirement_id") or "").strip()
    )
    _design._validate_section_types(
        section,
        fields,
        requirement_ids=ledger_ids,
    )


def _validate_design_coverage(
    design: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _design._validate_requirement_coverage(design, ledger)


def _assert_module_trace_schema() -> None:
    """Assert the canonical owner already contains the required module representation."""
    for section_id, _fields, properties in _design._SECTION_SPECS:
        if section_id != "modules_and_assets":
            continue
        modules = properties.get("modules")
        if not isinstance(modules, Mapping):
            raise RuntimeError("modules_and_assets schema lost its modules object")
        item_schema = modules.get("items")
        if not isinstance(item_schema, Mapping):
            raise RuntimeError("modules schema lost its item object")
        item_properties = item_schema.get("properties")
        required = item_schema.get("required")
        if not isinstance(item_properties, Mapping) or not isinstance(required, list):
            raise RuntimeError("modules schema is not contract-compatible")
        expected = {
            "plugin_id",
            "status",
            "reason",
            "requirement_refs",
            "implementation_obligations",
        }
        missing_properties = sorted(expected - set(item_properties))
        missing_required = sorted(expected - set(required))
        if missing_properties or missing_required:
            raise RuntimeError(
                "canonical modules schema is incomplete: "
                f"missing_properties={missing_properties} "
                f"missing_required={missing_required}"
            )
        return
    raise RuntimeError("modules_and_assets section schema was not found")


def _lossless_semantic_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Semantic line spans whose offsets remain exact for LF, CRLF, and CR."""
    spans: list[tuple[int, int]] = []
    line_offset = 0
    for raw_with_eol in prompt.splitlines(keepends=True):
        raw_line = raw_with_eol.rstrip("\r\n")
        line_start = line_offset
        line_offset += len(raw_with_eol)
        stripped = re.sub(r"^[\s\-\*•▶●]+|^\s*\d+\.\s*", "", raw_line)
        if not stripped.strip():
            continue
        matched_any = False
        for match in _evidence._SEMANTIC_BOUNDARY.finditer(raw_line):
            start = line_start + match.start()
            end = line_start + match.end()
            inner = raw_line[match.start() : match.end()]
            inner_stripped = re.sub(
                r"^[\s\-\*•▶●]+|^\s*\d+\.\s*",
                "",
                inner,
            )
            if not inner_stripped.strip():
                continue
            leading = len(inner) - len(inner.lstrip())
            bullet_stripped = inner.lstrip()
            bullet_cleaned = re.sub(
                r"^[\-\*•▶●]+|^\d+\.\s*",
                "",
                bullet_stripped,
            )
            start += leading + (len(bullet_stripped) - len(bullet_cleaned))
            while start < end and prompt[start].isspace():
                start += 1
            while end > start and prompt[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
                matched_any = True
        if not matched_any:
            leading = len(raw_line) - len(raw_line.lstrip())
            bullet_stripped = raw_line.lstrip()
            bullet_cleaned = re.sub(
                r"^[\-\*•▶●]+|^\d+\.\s*",
                "",
                bullet_stripped,
            )
            start = line_start + leading + (len(bullet_stripped) - len(bullet_cleaned))
            end = line_start + len(raw_line.rstrip())
            while start < end and prompt[start].isspace():
                start += 1
            if start < end:
                spans.append((start, end))
    if not spans and prompt.strip():
        spans.append((len(prompt) - len(prompt.lstrip()), len(prompt.rstrip())))
    return tuple(spans)


def _install_lossless_source_offsets() -> None:
    if getattr(_evidence._semantic_spans, "__mmm_crlf_lossless__", False):
        return
    _lossless_semantic_spans.__mmm_crlf_lossless__ = True  # type: ignore[attr-defined]
    _evidence._semantic_spans = _lossless_semantic_spans


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _assert_module_trace_schema()
    _install_lossless_source_offsets()
    _INSTALLED = True


__all__ = ["install"]
