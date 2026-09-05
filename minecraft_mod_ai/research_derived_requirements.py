from __future__ import annotations

"""Evidence-backed derivation of implementation requirements.

Authored requirements remain immutable.  This module performs a second, research-aware
requirements-engineering pass and records implementation obligations that are implied by
repository/API/runtime evidence but were not necessarily written by the user.  Every
facet receives an explicit disposition so omission cannot silently mean "not needed".
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


def _model_facets(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    system = (
        "You are the research-to-requirements authority for a Minecraft mod production planner. "
        "The authored requirement is immutable. Determine implementation requirements that are "
        "logically required by the supplied repository/API/runtime evidence. Do NOT add generic "
        "features merely because mods often have them. For each required facet return exactly one "
        "disposition: derived, already_covered, not_applicable, or unresolved. A derived item must "
        "cite one or more supplied evidence_ref values and include an observable acceptance check "
        "plus concrete implementation obligations. not_applicable and already_covered require a "
        "specific rationale. Use unresolved when the evidence is insufficient; never guess.\n\n"
        "Return JSON only: {\"facets\":[{\"facet\":...,\"disposition\":...,"
        "\"statement\":...,\"rationale\":...,\"evidence_refs\":[...],"
        "\"acceptance\":[...],\"implementation_obligations\":[...]}]}."
    )
    payload = {
        "parent_requirement": dict(requirement),
        "required_facets": list(FACETS),
        "evidence_catalog": list(evidence),
    }
    try:
        raw = router.generate_text(
            "planner",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": _canonical(payload)},
            ],
            response_format="json",
            enable_tools=False,
        )
        parsed = json.loads(raw)
    except Exception as exc:  # fail closed: downstream small agents must not inherit missing design work
        raise ResearchRequirementError(
            f"research-derived requirement analysis failed for {requirement.get('requirement_id')!r}"
        ) from exc
    facets = parsed.get("facets") if isinstance(parsed, Mapping) else None
    if not isinstance(facets, list):
        raise ResearchRequirementError("research-derived requirement response has no facets list")
    return [item for item in facets if isinstance(item, Mapping)]


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
