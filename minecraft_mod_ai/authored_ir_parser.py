from __future__ import annotations

"""Strict parser for the canonical authored design document.

This module never falls back to generic implementation units. It only recognizes
the fixed authored document section ids, lowers only roles actually present in the
approved document, and keeps nested concern headings inside their owning section.
"""

import re
from collections.abc import Mapping
from typing import Any

from .authored_section_ids import (
    DOCUMENT_SECTION_ORDER,
    DOCUMENT_SECTION_SET,
    EXECUTION_SECTION_ORDER,
    EXECUTION_SECTION_SET,
)


class AuthoredDesignSchemaError(ValueError):
    pass


_SECTION_ALIASES = {
    "개요": "overview",
    "overview": "overview",
    "행동_계약": "behavior_contract",
    "상태_모델": "state_model",
    "알고리즘": "algorithm",
    "통합": "integration",
    "권한_및_네트워크": "authority_and_network",
    "지속성": "persistence",
    "자원_및_ui": "resources_and_ui",
    "실패_및_제한": "failure_and_limits",
    "재사용_평가": "reuse_assessment",
    "검증": "verification",
    "결론": "conclusion",
}


def section_slug(value: str) -> str:
    text = re.sub(r"^\s*\d+[.)]\s*", "", str(value or "").strip()).casefold()
    text = text.strip("*_ `" + chr(96))
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_가-힣]+", "_", text)
    return text.strip("_").casefold()


def authored_section_id(title: str) -> str:
    raw = re.sub(r"^\s*\d+[.)]\s*", "", str(title or "").strip())
    for inner in reversed(re.findall(r"\(([^()]*)\)", raw)):
        token = _SECTION_ALIASES.get(section_slug(inner), section_slug(inner))
        if token in DOCUMENT_SECTION_SET:
            return token
    token = section_slug(raw)
    token = _SECTION_ALIASES.get(token, token)
    return token if token in DOCUMENT_SECTION_SET else ""


def parse_markdown_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
    if not match:
        return None
    title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip("*_ `" + chr(96))
    return len(match.group(1)), title


def _requirement_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    key = str(item[0] or "")
    match = re.fullmatch(r"R(\d+)", key)
    return (int(match.group(1)) if match else 10**9, key)


def _requirement_concern_label(value: str) -> str:
    text = str(value or "")
    if not text.startswith("- ") or ":" not in text:
        return ""
    return section_slug(text[2:].split(":", 1)[0].strip())


def slice_concern_requirements(
    raw: Mapping[str, Any],
    *,
    concern: str,
) -> dict[str, str]:
    """Return the exact canonical source slice owned by one authored concern."""
    ordered = [
        (str(key), str(value))
        for key, value in sorted(dict(raw or {}).items(), key=_requirement_sort_key)
    ]
    if not ordered:
        return {}

    target = section_slug(concern)
    anchor = next(
        (
            index
            for index, (_key, value) in enumerate(ordered)
            if _requirement_concern_label(value) == target
        ),
        -1,
    )

    headings = [
        (key, value)
        for key, value in ordered[: anchor if anchor >= 0 else len(ordered)]
        if value.lstrip().startswith("## ")
    ]
    selected: list[tuple[str, str]] = headings[-1:] if headings else []

    if anchor < 0:
        target_words = target.replace("_", " ")
        for key, value in ordered:
            lowered = value.casefold()
            if target in lowered or target_words in lowered:
                selected.append((key, value))
        return dict(selected)

    selected.append(ordered[anchor])
    for key, value in ordered[anchor + 1:]:
        if value.startswith("## "):
            break
        sibling = _requirement_concern_label(value)
        if sibling:
            if sibling == target:
                selected.append((key, value))
                continue
            break
        selected.append((key, value))
    return dict(selected)


def _legacy_canonical_section_depth(text: str) -> int | None:
    """Recognize only the old planner template shape; never relax generic depth rules."""

    records: list[tuple[int, int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        heading = parse_markdown_heading(line)
        if heading is None:
            continue
        depth, title = heading
        records.append((line_number, depth, title, authored_section_id(title)))

    canonical = [record for record in records if record[3]]
    if len(canonical) < 2:
        return None
    first_line, first_depth, _first_title, first_section = canonical[0]
    if first_section != "overview" or first_depth <= 1:
        return None

    wrapper_depth = first_depth - 1
    has_document_wrapper = any(
        line_number < first_line
        and depth == wrapper_depth
        and not section
        for line_number, depth, _title, section in records
    )
    if not has_document_wrapper:
        return None

    trailing = canonical[1:]
    if not trailing or any(
        depth != wrapper_depth
        for _line, depth, _title, _section in trailing
    ):
        return None
    return first_depth


def _append_ref(
    by_section: dict[str, list[str]], active: str, req_id: str
) -> None:
    if active in EXECUTION_SECTION_SET:
        by_section.setdefault(active, []).append(req_id)


def _enter_section(
    section: str,
    *,
    seen: list[str],
    by_section: dict[str, list[str]],
    req_id: str,
    last_index: int,
) -> tuple[str, int]:
    if section in seen:
        raise AuthoredDesignSchemaError(
            f"IMPLEMENTATION_IR_AUTHORED_SECTION_DUPLICATE: {section}"
        )
    index = DOCUMENT_SECTION_ORDER.index(section)
    if index <= last_index:
        raise AuthoredDesignSchemaError(
            f"IMPLEMENTATION_IR_AUTHORED_SECTION_ORDER: {section}"
        )
    seen.append(section)
    _append_ref(by_section, section, req_id)
    return section, index


def _handle_heading(
    heading: tuple[int, str],
    *,
    section_depth: int | None,
    active_section: str,
    seen_sections: list[str],
    by_section: dict[str, list[str]],
    req_id: str,
    last_index: int,
) -> tuple[int | None, str, int]:
    depth, title = heading
    section = authored_section_id(title)
    if section_depth is None:
        if not section:
            return None, active_section, last_index
        active, index = _enter_section(
            section, seen=seen_sections, by_section=by_section, req_id=req_id,
            last_index=last_index,
        )
        return depth, active, index
    if depth > section_depth:
        _append_ref(by_section, active_section, req_id)
        return section_depth, active_section, last_index
    if depth < section_depth:
        raise AuthoredDesignSchemaError(
            f"IMPLEMENTATION_IR_AUTHORED_SECTION_DEPTH: {title} uses depth {depth}, expected {section_depth}"
        )
    if not section:
        raise AuthoredDesignSchemaError(
            f"IMPLEMENTATION_IR_AUTHORED_UNKNOWN_SECTION: {title}"
        )
    active, index = _enter_section(
        section, seen=seen_sections, by_section=by_section, req_id=req_id,
        last_index=last_index,
    )
    return section_depth, active, index


def _collect_section_refs(text: str) -> dict[str, list[str]]:
    section_depth: int | None = None
    active_section = ""
    seen_sections: list[str] = []
    by_section: dict[str, list[str]] = {}
    last_index = -1
    legacy_section_depth = _legacy_canonical_section_depth(text)
    for idx, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        req_id = f"R{idx}"
        heading = parse_markdown_heading(line)
        if heading is None:
            _append_ref(by_section, active_section, req_id)
            continue
        if (
            legacy_section_depth is not None
            and heading[0] == legacy_section_depth - 1
            and authored_section_id(heading[1])
        ):
            heading = (legacy_section_depth, heading[1])
        section_depth, active_section, last_index = _handle_heading(
            heading,
            section_depth=section_depth,
            active_section=active_section,
            seen_sections=seen_sections,
            by_section=by_section,
            req_id=req_id,
            last_index=last_index,
        )
    return by_section


def decompose_canonical_authored_units(
    text: str, requirements: Mapping[str, str]
) -> list[dict[str, Any]]:
    if not requirements:
        raise AuthoredDesignSchemaError("IMPLEMENTATION_IR_AUTHORED_DESIGN_EMPTY")
    by_section = _collect_section_refs(text)
    planned_roles = [
        role for role in EXECUTION_SECTION_ORDER
        if by_section.get(role)
    ]
    if not planned_roles:
        raise AuthoredDesignSchemaError(
            "IMPLEMENTATION_IR_AUTHORED_EXECUTION_EMPTY"
        )
    return [
        {
            "unit_id": role,
            "title": role,
            "requirements": {
                req: requirements[req]
                for req in dict.fromkeys(by_section[role])
            },
            "context_requirements": {},
        }
        for role in planned_roles
    ]


__all__ = [
    "AuthoredDesignSchemaError",
    "authored_section_id",
    "decompose_canonical_authored_units",
    "parse_markdown_heading",
    "slice_concern_requirements",
]
