from __future__ import annotations

"""Evidence-backed derivation of implementation requirements.

Authored requirements remain immutable.  This module performs a second, research-aware
requirements-engineering pass and records implementation obligations that are implied by
repository/API/runtime evidence but were not necessarily written by the user.  Every
facet receives an explicit disposition so omission cannot silently mean "not needed".

The model-facing contract is deliberately bounded for small local models: the host owns
the facet matrix, selects a small evidence window for one facet at a time, and accepts one
native tool decision per facet. Free-form model JSON is not part of this protocol.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .evidence_first_planning import validate_evidence_first_plan

SCHEMA = "mmm/research-derived-requirements-v1"
FACETS = (
    "state_lifecycle",
    "interfaces_integration",
    "persistence_reload",
    "server_network_authority",
    "registration_data_resources",
    "failure_edge_cases",
    "verification_testing",
)
_DISPOSITIONS = frozenset({"derived", "already_covered", "not_applicable", "unresolved"})
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
        "loader",
        "module_id",
        "api",
        "symbol",
        "reason",
        "rationale",
    }
)
_MAX_EVIDENCE_PER_FACET = 12
_MAX_EVIDENCE_VALUE_CHARS = 360
_FACET_HINTS: dict[str, tuple[str, ...]] = {
    "state_lifecycle": ("lifecycle", "state", "init", "tick", "update", "cleanup", "dispose"),
    "interfaces_integration": ("interface", "integration", "api", "method", "hook", "event", "callback"),
    "persistence_reload": ("persist", "save", "load", "reload", "serialize", "codec", "nbt", "data"),
    "server_network_authority": ("server", "client", "network", "packet", "sync", "authority", "multiplayer"),
    "registration_data_resources": ("registry", "register", "resource", "datapack", "tag", "recipe", "asset", "data"),
    "failure_edge_cases": ("error", "failure", "missing", "invalid", "edge", "exception", "fallback"),
    "verification_testing": ("test", "verify", "verification", "validation", "assert", "unit", "regression"),
}
_FACET_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "disposition": {
            "type": "string",
            "enum": ["derived", "already_covered", "not_applicable", "unresolved"],
        },
        "statement": {"type": "string"},
        "rationale": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "acceptance": {"type": "array", "items": {"type": "string"}},
        "implementation_obligations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "disposition",
        "statement",
        "rationale",
        "evidence_refs",
        "acceptance",
        "implementation_obligations",
    ],
    "additionalProperties": False,
}


class ResearchRequirementError(ValueError):
    """Raised when research cannot close every required derivation facet."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


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


def _evidence_catalog(
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
    ):
        if key in game_design:
            _collect_receipts(game_design[key], path=f"game_design.{key}", output=raw)
    deduped: dict[str, dict[str, Any]] = {}
    for receipt in raw:
        deduped.setdefault(str(receipt["evidence_ref"]), receipt)
    return tuple(deduped.values())


def _require_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ResearchRequirementError(f"derived requirement field {field!r} must be non-empty")
    return text


def _bounded_text(value: Any, *, limit: int = _MAX_EVIDENCE_VALUE_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _facet_evidence_window(
    evidence: Sequence[Mapping[str, Any]],
    *,
    facet: str,
) -> tuple[Mapping[str, Any], ...]:
    """Rank and cap evidence before it reaches the model context."""

    hints = _FACET_HINTS[facet]
    scored: list[tuple[int, str, Mapping[str, Any]]] = []
    for receipt in evidence:
        searchable = _canonical(receipt).casefold()
        score = sum(1 for hint in hints if hint in searchable)
        scored.append((score, str(receipt.get("evidence_ref") or ""), receipt))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in scored[:_MAX_EVIDENCE_PER_FACET])


def _render_evidence_window(evidence: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for receipt in evidence:
        lines.append(f"- evidence_ref: {_bounded_text(receipt.get('evidence_ref'), limit=96)}")
        lines.append(f"  path: {_bounded_text(receipt.get('path'), limit=220)}")
        summary = receipt.get("summary")
        if not isinstance(summary, Mapping):
            continue
        for key in sorted(summary):
            raw = summary[key]
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                raw = "; ".join(_bounded_text(item, limit=120) for item in raw[:4])
            lines.append(f"  {key}: {_bounded_text(raw)}")
    return "\n".join(lines)


def _requirement_prompt(requirement: Mapping[str, Any]) -> str:
    fields: list[str] = []
    for key in ("requirement_id", "title", "summary", "description", "statement"):
        if requirement.get(key) is not None:
            fields.append(f"{key}: {_bounded_text(requirement.get(key), limit=700)}")
    return "\n".join(fields)[:1600]


def _model_facets(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Resolve host-owned facets one at a time with bounded native tool decisions."""

    parent = _require_text(requirement.get("requirement_id"), field="parent_requirement_ref")
    requirement_text = _requirement_prompt(requirement)
    decisions: list[Mapping[str, Any]] = []
    for facet in FACETS:
        selected = _facet_evidence_window(evidence, facet=facet)
        system = (
            "You are the research-to-requirements authority for a Minecraft mod production planner. "
            "The authored requirement and facet name are host-owned and immutable. Evaluate only "
            f"the fixed facet '{facet}'. Do not rename, replace, add, or omit the facet. Choose one "
            "disposition: derived, already_covered, not_applicable, or unresolved. A derived decision "
            "must cite supplied evidence_ref values and include an observable acceptance check plus "
            "concrete implementation obligations. Use unresolved whenever the bounded evidence is "
            "insufficient. Never invent platform, loader, API, version, lifecycle, or repository facts."
        )
        user = (
            f"Parent requirement: {parent}\n{requirement_text}\n\n"
            f"Fixed facet: {facet}\n"
            "Bounded evidence window; cite only refs shown below:\n"
            f"{_render_evidence_window(selected) or '- no evidence selected'}"
        )
        try:
            result = router.generate_tool_decision(
                "planner",
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tool_name="record_research_facet_decision",
                description="Record one decision for the single host-fixed research facet.",
                parameters=_FACET_TOOL_PARAMETERS,
            )
        except Exception as exc:  # fail closed: downstream small agents must not inherit missing design work
            raise ResearchRequirementError(
                f"research-derived facet decision failed for {parent!r} facet {facet!r}"
            ) from exc
        if not isinstance(result, Mapping):
            raise ResearchRequirementError(
                f"research-derived facet decision for {parent!r} facet {facet!r} is not an object"
            )
        decisions.append(
            {
                "facet": facet,
                "disposition": result.get("disposition"),
                "statement": result.get("statement"),
                "rationale": result.get("rationale"),
                "evidence_refs": result.get("evidence_refs"),
                "acceptance": result.get("acceptance"),
                "implementation_obligations": result.get("implementation_obligations"),
            }
        )
    return decisions


def derive_research_requirements(
    router: Any,
    *,
    prompt: str,
    evidence_plan: Mapping[str, Any],
    research_brief: Any,
    technical_evidence: Any,
    game_design: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce a complete research derivation matrix for every authored requirement."""

    validate_evidence_first_plan(evidence_plan, prompt=prompt)
    request_catalog = evidence_plan.get("request_catalog")
    if not isinstance(request_catalog, Mapping):
        raise ResearchRequirementError("evidence plan has no request catalog")
    requirements = request_catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ResearchRequirementError("evidence plan has no authored requirements")

    evidence = _evidence_catalog(research_brief, technical_evidence, game_design)
    allowed_refs = {str(item["evidence_ref"]) for item in evidence}
    if not evidence:
        raise ResearchRequirementError(
            "research-derived requirements require repository/API/runtime evidence receipts"
        )

    decisions: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise ResearchRequirementError("authored requirement is not an object")
        parent = _require_text(requirement.get("requirement_id"), field="parent_requirement_ref")
        raw_facets = _model_facets(router, requirement=requirement, evidence=evidence)
        by_facet: dict[str, Mapping[str, Any]] = {}
        for item in raw_facets:
            facet = str(item.get("facet") or "").strip()
            if facet not in FACETS or facet in by_facet:
                raise ResearchRequirementError(
                    f"requirement {parent!r} has an invalid or duplicate facet {facet!r}"
                )
            by_facet[facet] = item
        missing = [facet for facet in FACETS if facet not in by_facet]
        if missing:
            raise ResearchRequirementError(
                f"requirement {parent!r} omitted derivation facets: {missing}"
            )

        for facet in FACETS:
            item = by_facet[facet]
            disposition = str(item.get("disposition") or "").strip().casefold()
            if disposition not in _DISPOSITIONS:
                raise ResearchRequirementError(
                    f"requirement {parent!r} facet {facet!r} has invalid disposition {disposition!r}"
                )
            rationale = _require_text(item.get("rationale"), field=f"{parent}.{facet}.rationale")
            evidence_refs = _strings(item.get("evidence_refs"))
            if not set(evidence_refs) <= allowed_refs:
                unknown = sorted(set(evidence_refs) - allowed_refs)
                raise ResearchRequirementError(
                    f"requirement {parent!r} facet {facet!r} cites unknown evidence: {unknown}"
                )
            acceptance = _strings(item.get("acceptance"))
            obligations = _strings(item.get("implementation_obligations"))
            statement = str(item.get("statement") or "").strip()
            if disposition == "derived":
                if not statement or not evidence_refs or not acceptance or not obligations:
                    raise ResearchRequirementError(
                        f"derived facet {parent}.{facet} lacks statement/evidence/acceptance/obligations"
                    )
            elif disposition == "unresolved":
                unresolved.append(f"{parent}:{facet}")
            decision = {
                "derived_requirement_id": "derived_" + _sha(
                    {"parent": parent, "facet": facet, "statement": statement, "disposition": disposition}
                )[7:27],
                "parent_requirement_ref": parent,
                "provenance_role": "logically_derived",
                "facet": facet,
                "disposition": disposition,
                "statement": statement,
                "rationale": rationale,
                "evidence_refs": list(evidence_refs),
                "acceptance": list(acceptance),
                "implementation_obligations": list(obligations),
            }
            decisions.append(decision)

    if unresolved:
        raise ResearchRequirementError(
            "research could not close required implementation facets: " + ", ".join(unresolved)
        )

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA,
        "prompt_sha256": request_catalog.get("prompt_sha256"),
        "evidence_catalog": list(evidence),
        "facet_decisions": decisions,
        "ledger_sha256": "",
    }
    ledger["ledger_sha256"] = _sha({key: value for key, value in ledger.items() if key != "ledger_sha256"})
    return ledger


def attach_derived_requirement_ledger(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the immutable sidecar to PlanIR without changing authored requirements."""

    result = json.loads(_canonical(plan))
    result["derived_requirement_ledger"] = json.loads(_canonical(ledger))
    result["plan_sha256"] = ""
    result["plan_sha256"] = _sha(result)
    validate_evidence_first_plan(result)
    return result


__all__ = [
    "FACETS",
    "ResearchRequirementError",
    "attach_derived_requirement_ledger",
    "derive_research_requirements",
]
