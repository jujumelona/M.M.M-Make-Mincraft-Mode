from __future__ import annotations

"""Text-native research-note protocol with host-owned parsing.

Model research turns produce Markdown. JSON remains acceptable only as a legacy host input
for persisted checkpoints/tests; no model-facing caller is required to generate JSON.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .external_procedural_skill_contract import _sanitize_procedure
from .spec import SpecValidationError

_NONE = frozenset({"", "none", "null", "n/a", "없음", "-"})


def instructions(domain_id: str) -> str:
    return (
        "Return Markdown, not JSON and not a code block. Use exactly these level-2 headings: "
        "## domain_id, ## sufficient, ## claims, ## gaps, ## next_queries, ## procedures. "
        f"Under ## domain_id write exactly {domain_id!r}. Under ## sufficient write true or false. "
        "Under ## claims, use one level-3 heading per claim and lines '- text: <claim>' and "
        "'- evidence_refs: <comma-separated host-issued refs>'. Under ## gaps and ## next_queries "
        "use bullets, or '- none'. Under ## procedures, use one level-3 heading per reusable "
        "procedure. A procedure uses '- activate_when:', '- contraindications:', '- steps:', "
        "'- constraints:', '- output_contract:', '- evidence_refs:', '- confidence:', "
        "'- requires:', and '- provides:'. List-valued procedure fields may be comma-separated "
        "or followed by indented bullets. If there is no evidence-grounded reusable procedure, "
        "write '- none'. Never invent evidence refs."
    )


def _split_csv(value: str) -> list[str]:
    text = str(value or "").strip().strip("[](){}")
    if text.casefold() in _NONE:
        return []
    return list(
        dict.fromkeys(
            item.strip().strip("`'\"")
            for item in re.split(r"\s*[,;，；]\s*", text)
            if item.strip().strip("`'\"")
        )
    )


def _strip_bullet(line: str) -> str:
    return re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line.strip()).strip()


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in str(text or "").splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", raw)
        if match:
            current = re.sub(r"[^a-z0-9]+", "_", match.group(1).casefold()).strip("_")
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(raw)
    return sections


def _list_section(lines: Sequence[str]) -> list[str]:
    values: list[str] = []
    for raw in lines:
        if raw.lstrip().startswith("### "):
            continue
        value = _strip_bullet(raw)
        if value and value.casefold() not in _NONE:
            values.append(value)
    return list(dict.fromkeys(values))


def _parse_claims(lines: Sequence[str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        text = str(current.get("claim") or "").strip()
        if text:
            claims.append(
                {
                    "claim": text,
                    "evidence_refs": list(current.get("evidence_refs") or []),
                }
            )
        current = None

    for raw in lines:
        heading = re.match(r"^\s*###\s+(.+?)\s*$", raw)
        if heading:
            flush()
            current = {"claim": "", "evidence_refs": []}
            continue
        value = _strip_bullet(raw)
        if not value or value.casefold() in _NONE:
            continue
        if current is None:
            current = {"claim": "", "evidence_refs": []}
        if ":" in value:
            key, item = value.split(":", 1)
            key = key.strip().casefold().replace(" ", "_")
            item = item.strip()
            if key in {"text", "claim", "claim_text"}:
                current["claim"] = item
                continue
            if key in {"evidence_refs", "evidence_ref", "source_refs", "source_ref"}:
                current["evidence_refs"] = _split_csv(item)
                continue
        if not current.get("claim"):
            current["claim"] = value
    flush()
    return claims


def _parse_procedures(lines: Sequence[str], domain_id: str) -> list[dict[str, Any]]:
    procedures: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_list: str | None = None
    list_fields = {
        "activate_when",
        "contraindications",
        "steps",
        "constraints",
        "evidence_refs",
        "requires",
        "provides",
    }

    def flush() -> None:
        nonlocal current, active_list
        if current is not None:
            sanitized = _sanitize_procedure(current, domain_id)
            if sanitized is not None:
                procedures.append(sanitized)
        current = None
        active_list = None

    for raw in lines:
        heading = re.match(r"^\s*###\s+(.+?)\s*$", raw)
        if heading:
            flush()
            name = heading.group(1).strip().strip("`")
            if name.casefold() not in _NONE:
                current = {"name": name}
            continue
        value = _strip_bullet(raw)
        if not value or value.casefold() in _NONE:
            continue
        if current is None:
            continue
        if ":" in value:
            key, item = value.split(":", 1)
            key = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            item = item.strip()
            if key in list_fields:
                current[key] = _split_csv(item)
                active_list = key if not item else None
                continue
            if key == "output_contract":
                current[key] = item
                active_list = None
                continue
            if key == "confidence":
                try:
                    current[key] = float(item)
                except ValueError:
                    current[key] = 0.0
                active_list = None
                continue
        if active_list is not None:
            current.setdefault(active_list, []).append(value)
    flush()
    return procedures


def normalize(value: Mapping[str, Any], domain_id: str) -> dict[str, Any]:
    note = value.get("research_note") if isinstance(value.get("research_note"), Mapping) else value
    claims: list[dict[str, Any]] = []
    for raw in note.get("claims", []) if isinstance(note.get("claims"), list) else []:
        if isinstance(raw, Mapping):
            text = str(raw.get("claim") or raw.get("text") or raw.get("claim_text") or raw.get("content") or "").strip()
            refs = raw.get("evidence_refs")
            if isinstance(refs, list):
                evidence_refs = [str(ref).strip() for ref in refs if str(ref).strip()]
            else:
                evidence_refs = _split_csv(str(raw.get("evidence_ref") or raw.get("source_ref") or ""))
            if text:
                claims.append({"claim": text, "evidence_refs": evidence_refs})
        elif isinstance(raw, str) and raw.strip():
            claims.append({"claim": raw.strip(), "evidence_refs": []})
    procedures: list[dict[str, Any]] = []
    raw_procedures = note.get("procedures")
    if isinstance(raw_procedures, list):
        for raw in raw_procedures:
            if isinstance(raw, Mapping):
                procedure = _sanitize_procedure(raw, domain_id)
                if procedure is not None:
                    procedures.append(procedure)
    return {
        "domain_id": domain_id,
        "claims": claims,
        "gaps": [str(item).strip() for item in note.get("gaps", []) if str(item).strip()]
        if isinstance(note.get("gaps"), list)
        else [],
        "next_queries": [str(item).strip() for item in note.get("next_queries", []) if str(item).strip()]
        if isinstance(note.get("next_queries"), list)
        else [],
        "sufficient": bool(note.get("sufficient", False)),
        "procedures": procedures,
    }


def parse(text: str, domain_id: str) -> dict[str, Any]:
    """Parse the native Markdown protocol; legacy JSON is accepted only for host compatibility."""

    raw = str(text or "").strip()
    if not raw:
        raise SpecValidationError("research note output is empty")
    if raw.startswith("{"):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, Mapping):
            return normalize(value, domain_id)

    sections = _sections(raw)
    required = {"domain_id", "sufficient", "claims", "gaps", "next_queries", "procedures"}
    missing = sorted(required - set(sections))
    if missing:
        raise SpecValidationError(
            "research Markdown omitted required heading(s): " + ", ".join(missing)
        )
    rendered_domain = " ".join(_strip_bullet(line) for line in sections["domain_id"] if _strip_bullet(line)).strip()
    if rendered_domain and rendered_domain != domain_id:
        raise SpecValidationError(
            f"research note domain_id {rendered_domain!r} does not match {domain_id!r}"
        )
    sufficient_text = " ".join(
        _strip_bullet(line) for line in sections["sufficient"] if _strip_bullet(line)
    ).strip().casefold()
    if sufficient_text not in {"true", "false"}:
        raise SpecValidationError("research Markdown ## sufficient must be exactly true or false")
    return {
        "domain_id": domain_id,
        "claims": _parse_claims(sections["claims"]),
        "gaps": _list_section(sections["gaps"]),
        "next_queries": _list_section(sections["next_queries"]),
        "sufficient": sufficient_text == "true",
        "procedures": _parse_procedures(sections["procedures"], domain_id),
    }


__all__ = ["instructions", "normalize", "parse"]
