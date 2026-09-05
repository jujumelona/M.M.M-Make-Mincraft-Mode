from __future__ import annotations

"""Requirement-specific external research receipt collection and retrieval."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .research_requirement_schema import FACETS, FACET_HINTS, STOPWORDS

_MAX_EVIDENCE_PER_REQUIREMENT = 12
_EVIDENCE_KEYS = frozenset(
    {
        "source_id",
        "source_ref",
        "url",
        "uri",
        "path",
        "claim",
        "statement",
        "summary",
        "status",
        "version",
        "minecraft_version",
        "loader",
        "loaders",
        "mappings",
        "module_id",
        "api",
        "symbol",
        "symbols",
        "kind",
        "reason",
        "rationale",
        "capability",
        "capabilities",
        "requirement_ref",
        "requirement_refs",
        "task_ref",
        "task_id",
        "repository",
        "commit_sha",
        "license_id",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _receipt_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, raw in value.items():
        if str(key) not in _EVIDENCE_KEYS:
            continue
        if isinstance(raw, (str, int, float, bool)) or raw is None:
            summary[str(key)] = raw
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            scalars = [item for item in raw if isinstance(item, (str, int, float, bool))]
            if scalars:
                summary[str(key)] = scalars
    return summary


def _collect_receipts(value: Any, *, path: str, output: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        summary = _receipt_summary(value)
        if summary:
            identity = {"path": path, "summary": summary}
            output.append(
                {
                    "evidence_ref": "evidence:" + _sha(identity)[7:23],
                    "path": path,
                    "summary": summary,
                }
            )
        for key, child in value.items():
            _collect_receipts(child, path=f"{path}.{key}", output=output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _collect_receipts(child, path=f"{path}[{index}]", output=output)


def evidence_catalog(
    research_brief: Any,
    technical_evidence: Any,
    game_design: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw: list[dict[str, Any]] = []
    _collect_receipts(research_brief, path="research_brief", output=raw)
    _collect_receipts(technical_evidence, path="technical_evidence", output=raw)
    for key in (
        "_existing_project_inventory",
        "_existing_snapshot",
        "_platform_selection",
        "_platform_evidence",
        "_pre_design_research",
        "_reuse_plan",
    ):
        if key in game_design:
            _collect_receipts(game_design[key], path=f"game_design.{key}", output=raw)
    deduped: dict[str, dict[str, Any]] = {}
    for receipt in raw:
        deduped.setdefault(str(receipt["evidence_ref"]), receipt)
    return tuple(deduped.values())


def _requirement_terms(requirement: Mapping[str, Any]) -> tuple[str, ...]:
    raw = " ".join(
        [
            str(requirement.get("capability") or ""),
            str(requirement.get("statement") or ""),
            " ".join(_strings(requirement.get("provides"))),
            " ".join(_strings(requirement.get("gameplay_capabilities"))),
            " ".join(_strings(requirement.get("implementation_capabilities"))),
        ]
    ).casefold()
    terms = [
        token
        for token in re.findall(r"[a-z0-9_]+", raw)
        if len(token) >= 4 and token not in STOPWORDS
    ]
    return tuple(dict.fromkeys(terms))


def _explicit_requirement_binding(
    receipt: Mapping[str, Any], requirement: Mapping[str, Any]
) -> bool:
    searchable = _canonical(receipt).casefold()
    requirement_id = str(requirement.get("requirement_id") or "").casefold()
    capability = str(requirement.get("capability") or "").casefold()
    return bool(
        (requirement_id and requirement_id in searchable)
        or (capability and capability in searchable)
    )


def requirement_score(
    receipt: Mapping[str, Any], requirement: Mapping[str, Any]
) -> int:
    searchable = _canonical(receipt).casefold()
    if _explicit_requirement_binding(receipt, requirement):
        return 100
    return sum(1 for term in _requirement_terms(requirement) if term in searchable)


def facet_score(receipt: Mapping[str, Any], facet: str) -> int:
    searchable = _canonical(receipt).casefold()
    return sum(1 for hint in FACET_HINTS[facet] if hint in searchable)


def facet_relevant_refs(
    evidence: Sequence[Mapping[str, Any]],
    requirement: Mapping[str, Any],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for facet in FACETS:
        ranked: list[tuple[int, str]] = []
        for receipt in evidence:
            req_score = requirement_score(receipt, requirement)
            f_score = facet_score(receipt, facet)
            if f_score <= 0:
                continue
            if baseline[facet]["disposition"] == "not_applicable" and req_score <= 0:
                continue
            ref = str(receipt.get("evidence_ref") or "")
            if ref:
                ranked.append((req_score * 10 + f_score, ref))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        result[facet] = tuple(
            ref for _score, ref in ranked[:_MAX_EVIDENCE_PER_REQUIREMENT]
        )
    return result


def requirement_evidence_window(
    evidence: Sequence[Mapping[str, Any]],
    relevant_refs: Mapping[str, Sequence[str]],
) -> tuple[Mapping[str, Any], ...]:
    wanted = {ref for refs in relevant_refs.values() for ref in refs}
    if not wanted:
        return ()
    by_ref = {str(item.get("evidence_ref") or ""): item for item in evidence}
    ordered = sorted(
        wanted,
        key=lambda ref: (
            -sum(ref in refs for refs in relevant_refs.values()),
            ref,
        ),
    )
    return tuple(
        by_ref[ref]
        for ref in ordered[:_MAX_EVIDENCE_PER_REQUIREMENT]
        if ref in by_ref
    )


__all__ = [
    "evidence_catalog",
    "facet_relevant_refs",
    "requirement_evidence_window",
]
