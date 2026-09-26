from __future__ import annotations

"""Strict parser for the canonical authored design document.

This module never falls back to generic implementation units. It only recognizes
the fixed authored document section ids and keeps nested concern headings inside
their owning canonical section.
"""

import re
from collections.abc import Mapping
from typing import Any

from .authored_execution_schema import (
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
    for idx, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        req_id = f"R{idx}"
        heading = parse_markdown_heading(line)
        if heading is None:
            _append_ref(by_section, active_section, req_id)
            continue
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
    missing = [
        section for section in EXECUTION_SECTION_ORDER
        if not by_section.get(section)
    ]
    if missing:
        raise AuthoredDesignSchemaError(
            "IMPLEMENTATION_IR_AUTHORED_SECTION_MISSING: " + ", ".join(missing)
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
        for role in EXECUTION_SECTION_ORDER
    ]


__all__ = [
    "AuthoredDesignSchemaError",
    "authored_section_id",
    "decompose_canonical_authored_units",
    "parse_markdown_heading",
]
