from __future__ import annotations

"""Host-owned canonicalization for authored design Markdown.

The planner may write free Markdown, but production must never depend on the model
remembering a heading protocol. This module projects malformed/legacy prose into the
same canonical execution vocabulary consumed by authored_ir_parser. The projection is
deterministic, preserves authored section bodies, and never manufactures omitted
optional execution roles. Explicit nested role contracts are promoted when present.
"""

import hashlib
import re
from collections import Counter
from typing import Any

class AuthoredDesignSchemaError(ValueError):
    pass

_SECTION_ALIASES = {
    "개요": "overview", "overview": "overview",
    "행동_계약": "behavior_contract", "상태_모델": "state_model",
    "알고리즘": "algorithm", "통합": "integration",
    "권한_및_네트워크": "authority_and_network", "영속성": "persistence",
    "자원_및_ui": "resources_and_ui", "실패_및_제한": "failure_and_limits",
    "재사용_평가": "reuse_assessment", "검증": "verification", "결론": "conclusion",
}

def _section_slug(value: str) -> str:
    text = re.sub(r"^\s*\d+[.)]\s*", "", str(value or "").strip()).casefold()
    text = text.strip("*_ `" + chr(96))
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_\uac00-\ud7a3]+", "_", text)
    return text.strip("_").casefold()

def authored_section_id(title: str) -> str:
    raw = re.sub(r"^\s*\d+[.)]\s*", "", str(title or "").strip())
    for inner in reversed(re.findall(r"\(([^()]*)\)", raw)):
        token = _SECTION_ALIASES.get(_section_slug(inner), _section_slug(inner))
        if token in DOCUMENT_SECTION_SET:
            return token
    token = _section_slug(raw)
    token = _SECTION_ALIASES.get(token, token)
    return token if token in DOCUMENT_SECTION_SET else ""

def parse_markdown_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
    if not match:
        return None
    title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip("*_ `" + chr(96))
    return len(match.group(1)), title

def decompose_canonical_authored_units(text: str, requirements: Any) -> list[dict[str, Any]]:
    return [{"unit_id": s, "title": s, "requirements": dict(requirements or {}), "context_requirements": {}} for s in EXECUTION_SECTION_ORDER]
from .authored_section_ids import (
    DOCUMENT_SECTION_ORDER,
    DOCUMENT_SECTION_SET,
    EXECUTION_SECTION_ORDER,
    EXECUTION_SECTION_SET,
)


class AuthoredDocumentContractError(ValueError):
    pass


def _source_requirements(text: str) -> dict[str, str]:
    return {
        f"R{index}": line
        for index, line in enumerate(str(text or "").splitlines(), start=1)
        if line.strip()
    }


def _canonical_heading_records(
    text: str,
) -> tuple[list[str], list[tuple[int, int, str, str]]]:
    lines = str(text or "").splitlines(keepends=True)
    records: list[tuple[int, int, str, str]] = []
    fence = ""
    for index, line in enumerate(lines):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not line[marker.end():].strip()
            ):
                fence = ""
            continue
        if marker:
            fence = marker[1]
            continue
        parsed = parse_markdown_heading(line.rstrip("\r\n"))
        if parsed is None:
            continue
        depth, title = parsed
        section = authored_section_id(title)
        if section in DOCUMENT_SECTION_SET:
            records.append((index, depth, title, section))
    return lines, records


def _demote_peer_headings(text: str) -> str:
    """Keep arbitrary authored subheadings without letting them become peer sections."""

    result: list[str] = []
    fence = ""
    for line in str(text or "").splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            result.append(line)
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not line[marker.end():].strip()
            ):
                fence = ""
            continue
        if marker:
            fence = marker[1]
            result.append(line)
            continue
        parsed = parse_markdown_heading(line.rstrip("\r\n"))
        if parsed is None or parsed[0] > 2:
            result.append(line)
            continue
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        result.append("### " + parsed[1] + ending)
    return "".join(result)


def _section_bodies(
    text: str,
) -> tuple[str, dict[str, list[str]], Counter[str]]:
    lines, records = _canonical_heading_records(text)
    if not records:
        return "", {"behavior_contract": [_demote_peer_headings(text)]}, Counter()

    preamble = "".join(lines[: records[0][0]])
    bodies: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    for position, (start, _depth, _title, section) in enumerate(records):
        end = records[position + 1][0] if position + 1 < len(records) else len(lines)
        body = _demote_peer_headings("".join(lines[start + 1 : end]))
        bodies.setdefault(section, []).append(body)
        counts[section] += 1
    return preamble, bodies, counts


def _canonical_planned_roles(text: str) -> set[str]:
    try:
        units = decompose_canonical_authored_units(text, _source_requirements(text))
    except AuthoredDesignSchemaError:
        return set()
    return {str(unit["unit_id"]) for unit in units}


def _strict_ready(text: str) -> bool:
    planned = _canonical_planned_roles(text)
    return bool(planned) and not _nested_execution_labels(text, planned)


_NESTED_SECTION_BULLET = re.compile(
    r"^(?P<indent>[ \t]*)[-*+]\s+"
    r"(?P<label>[0-9A-Za-z_가-힣-]+)\s*:\s*(?P<tail>.*?)\s*(?:\r?\n)?$"
)
_ANY_BULLET = re.compile(r"^(?P<indent>[ \t]*)[-*+]\s+")


def _indent_width(value: str) -> int:
    return len(value.expandtabs(4))


def _nested_section_label(label: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_가-힣]+", "_", str(label or "").strip()).strip("_").casefold()
    if slug in EXECUTION_SECTION_SET:
        return slug
    return next(
        (section for section in EXECUTION_SECTION_ORDER if slug.startswith(section + "_")),
        "",
    )


def _nested_execution_labels(
    source: str, existing: set[str]
) -> set[str]:
    found: set[str] = set()
    fence = ""
    for line in str(source or "").splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = ""
            continue
        if fence:
            continue
        match = _NESTED_SECTION_BULLET.match(line)
        section = _nested_section_label(match["label"]) if match else ""
        if section and section not in existing:
            found.add(section)
    return found


def _nested_block_end(lines: list[str], start: int, base_indent: int) -> int:
    for position in range(start + 1, len(lines)):
        line = lines[position]
        if parse_markdown_heading(line.rstrip("\r\n")) is not None:
            return position
        bullet = _ANY_BULLET.match(line)
        if bullet and _indent_width(bullet["indent"]) <= base_indent:
            return position
    return len(lines)


def _nested_block_body(lines: list[str], start: int, match: re.Match[str]) -> tuple[str, int]:
    end = _nested_block_end(lines, start, _indent_width(match["indent"]))
    tail = str(match["tail"] or "").strip()
    body = "".join(lines[start + 1 : end]).strip("\r\n")
    if tail:
        body = tail + ("\n" + body if body else "")
    return body, end


def _promote_nested_execution_sections(
    source: str,
    bodies: dict[str, list[str]],
) -> list[str]:
    missing_at_source = {
        section
        for section in EXECUTION_SECTION_ORDER
        if not any(chunk.strip() for chunk in bodies.get(section, ()))
    }
    if not missing_at_source:
        return []

    lines = str(source or "").splitlines(keepends=True)
    promoted: dict[str, list[str]] = {}
    fence = ""
    position = 0
    while position < len(lines):
        line = lines[position]
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = ""
            position += 1
            continue
        if fence:
            position += 1
            continue
        match = _NESTED_SECTION_BULLET.match(line)
        section = _nested_section_label(match["label"]) if match else ""
        if not match or section not in missing_at_source:
            position += 1
            continue
        body, end = _nested_block_body(lines, position, match)
        if body:
            promoted.setdefault(section, []).append(body)
        position = max(position + 1, end)

    for section, chunks in promoted.items():
        bodies.setdefault(section, []).extend(chunks)
    return [section for section in EXECUTION_SECTION_ORDER if section in promoted]


def _missing_execution_sections(bodies: dict[str, list[str]]) -> list[str]:
    return [
        section
        for section in EXECUTION_SECTION_ORDER
        if not any(chunk.strip() for chunk in bodies.get(section, ()))
    ]


def _render_canonical_document(
    preamble: str,
    bodies: dict[str, list[str]],
) -> str:
    parts = [preamble.rstrip()] if preamble.strip() else []
    for section in DOCUMENT_SECTION_ORDER:
        chunks = [
            chunk.strip("\r\n")
            for chunk in bodies.get(section, ())
            if chunk.strip()
        ]
        if not chunks:
            continue
        body = "\n\n".join(chunks).strip()
        parts.append(f"## {section}\n{body}".rstrip())
    return "\n\n".join(parts).rstrip() + "\n"


def _validate_normalized_document(text: str) -> None:
    try:
        decompose_canonical_authored_units(text, _source_requirements(text))
    except AuthoredDesignSchemaError as exc:
        raise AuthoredDocumentContractError(
            "AUTHORED_DOCUMENT_CANONICALIZATION_FAILED: " + str(exc)
        ) from exc


def _normalization_report(
    source: str,
    normalized: str,
    *,
    missing: list[str],
    duplicates: list[str],
    promoted: list[str],
) -> dict[str, Any]:
    source_bytes = source.encode("utf-8")
    normalized_bytes = normalized.encode("utf-8")
    return {
        "schema_version": "mmm/authored-document-normalization-v1",
        "policy": "host_canonical_execution_sections",
        "source_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        "normalized_sha256": "sha256:" + hashlib.sha256(normalized_bytes).hexdigest(),
        "source_bytes": len(source_bytes),
        "normalized_bytes": len(normalized_bytes),
        "missing_execution_sections": missing,
        "merged_duplicate_sections": duplicates,
        "promoted_nested_sections": promoted,
    }


def normalize_authored_document(
    text: str,
) -> tuple[str, dict[str, Any] | None]:
    """Return a production-safe canonical view plus provenance when bytes changed."""

    source = str(text or "")
    if not source.strip():
        raise AuthoredDocumentContractError("AUTHORED_DOCUMENT_EMPTY")
    if _strict_ready(source):
        return source, None

    preamble, bodies, counts = _section_bodies(source)
    promoted = _promote_nested_execution_sections(source, bodies)
    missing = _missing_execution_sections(bodies)
    duplicates = sorted(section for section, count in counts.items() if count > 1)
    normalized = _render_canonical_document(preamble, bodies)
    _validate_normalized_document(normalized)
    return normalized, _normalization_report(
        source,
        normalized,
        missing=missing,
        duplicates=duplicates,
        promoted=promoted,
    )


def assert_authored_document_ready(text: str) -> None:
    if not _strict_ready(str(text or "")):
        raise AuthoredDocumentContractError("AUTHORED_DOCUMENT_NOT_CANONICAL")


__all__ = [
    "AuthoredDocumentContractError",
    "assert_authored_document_ready",
    "normalize_authored_document",
]
