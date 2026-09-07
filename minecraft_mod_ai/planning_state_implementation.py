from __future__ import annotations

"""Host-owned detailed-plan compilation from grounded research.

The model never authors the plan container, evidence identifiers, or a large JSON/tool
payload. Host code owns structure and validation; the small model writes one bounded
plain-text engineering section at a time.
"""

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

from .model_concurrency import router_native_model_parallelism
from .planner_operation import planner_operation
from .planning_detail_contract import validate_detailed_plan_grounding, validate_evidence_refs
from .planning_detail_template import (
    DETAIL_FIELDS,
    DETAIL_SLOT_GUIDANCE,
    normalize_required_sections,
    validate_worksheet,
)
from .planning_state_contract import validate_planning_state


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _requirement_decisions(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in state.get("decisions", [])
        if isinstance(item, Mapping) and item.get("decision_type") == "requirement"
    ]


def _implementation_evidence(
    state: Mapping[str, Any], requirement_ref: str
) -> list[Mapping[str, Any]]:
    research_ids = {
        str(item.get("research_id") or "")
        for item in state.get("research_queue", [])
        if isinstance(item, Mapping)
        and str(item.get("requirement_ref") or "") == requirement_ref
        and item.get("status") == "complete"
    }
    rows: list[Mapping[str, Any]] = []
    for item in state.get("evidence", []):
        if (
            isinstance(item, Mapping)
            and str(item.get("research_ref") or "") in research_ids
            and item.get("sufficient") is True
        ):
            rows.append(
                {
                    "research_ref": _text(item.get("research_ref")),
                    "claims": deepcopy(item.get("claims") or []),
                    "evidence_refs": [
                        _text(ref) for ref in item.get("evidence_refs", []) if _text(ref)
                    ],
                    "sufficient": True,
                    "source": _text(item.get("source")),
                }
            )
    return rows


def _allowed_refs(evidence: list[Mapping[str, Any]]) -> set[str]:
    return {
        _text(ref)
        for item in evidence
        for ref in item.get("evidence_refs", [])
        if _text(ref)
    }


def _requirement_grounding(
    state: Mapping[str, Any], requirement_ref: str
) -> tuple[list[Mapping[str, Any]], set[str]]:
    evidence = _implementation_evidence(state, requirement_ref)
    allowed = _allowed_refs(evidence)
    if not evidence or not allowed:
        raise ValueError(
            f"DETAILED_PLAN_EVIDENCE: {requirement_ref} has no sufficient grounded implementation evidence"
        )
    return evidence, allowed


def _preflight_detailed_planning(
    state: Mapping[str, Any], requirements: list[Mapping[str, Any]]
) -> None:
    seen: set[str] = set()
    for requirement in requirements:
        requirement_ref = _text(requirement.get("requirement_id"))
        if not requirement_ref or requirement_ref in seen:
            raise ValueError(
                "DETAILED_PLAN_REQUIREMENTS: requirement IDs must be non-empty and unique"
            )
        seen.add(requirement_ref)
        _requirement_grounding(state, requirement_ref)


def _validate_refs(
    refs: Any,
    allowed: set[str],
    *,
    field: str,
    require: bool = True,
) -> list[str]:
    return validate_evidence_refs(refs, allowed, field=field, require=require)


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _evidence_context(evidence: list[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for item in evidence:
        research_ref = _text(item.get("research_ref"))
        claims = item.get("claims") or []
        claim_text = " | ".join(_text(claim) for claim in claims if _text(claim))
        if claim_text:
            rows.append(f"- {research_ref}: {claim_text}")
    return "\n".join(rows) or "- Grounded evidence exists, but no claim prose is available."


def _plain_section(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    section: str,
    evidence: list[Mapping[str, Any]],
) -> str:
    checklist = "; ".join(DETAIL_SLOT_GUIDANCE[section])
    statement = _text(requirement.get("statement"))
    messages = [
        {
            "role": "system",
            "content": (
                "Write one bounded engineering-design section for a small-model planning pipeline. "
                "Return plain prose only: no JSON, YAML, XML, tool/function call, code fence, object keys, "
                "or evidence identifiers. The host owns all structure and provenance. Do not invent API names, "
                "symbols, versions, dependencies, repository paths, or source facts. Keep the design abstract "
                "where target facts are not established. Be concrete about actors, state, branches, limits, "
                "failure behavior, and observable outcomes that belong to this section."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {statement}\n"
                f"Section: {section}\n"
                f"Purpose: {DETAIL_FIELDS[section]}\n"
                f"Checklist: {checklist}\n"
                "Grounded research context (context only; do not emit evidence IDs):\n"
                f"{_evidence_context(evidence)}\n"
                "Write only the section specification."
            ),
        },
    ]
    with planner_operation(f"detailed_section:{section}", output_tokens=900):
        raw = router.generate_text(
            "planner",
            messages,
            response_format="text",
            enable_tools=False,
        )
    value = str(raw or "").strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
    if value.lstrip().startswith(("{", "[")):
        raise ValueError(
            f"DETAILED_PLAN_TEXT_BOUNDARY: {section} returned a structured payload instead of prose"
        )
    value = _text(value)
    if len(value) < 24:
        raise ValueError(f"DETAILED_PLAN_SECTION: {section} returned no concrete specification")
    return value


def _host_derived_capabilities(worksheet: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "capability": f"Preserve the {section} contract: {_text(row.get('specification'))}",
            "constraint_evidence_refs": [],
        }
        for section, row in worksheet.items()
    ]


def _host_derived_obligations(worksheet: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "obligation": f"Implement and verify the {section} contract: {_text(row.get('specification'))}",
            "constraint_evidence_refs": [],
        }
        for section, row in worksheet.items()
    ]


def _host_derived_checks(requirement: Mapping[str, Any], worksheet: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    acceptance = requirement.get("acceptance")
    checks = [
        _text(item)
        for item in acceptance
        if _text(item)
    ] if isinstance(acceptance, list) else []
    if not checks:
        verification = worksheet.get("verification", {})
        specification = _text(verification.get("specification"))
        if specification:
            checks = [specification]
    if not checks:
        checks = [f"Given the requirement is exercised, verify the observable behavior: {_text(requirement.get('statement'))}"]
    return [
        {"check": check, "constraint_evidence_refs": []}
        for check in checks
    ]


def _compile_requirement_plan(
    router: Any,
    state: Mapping[str, Any],
    requirement: Mapping[str, Any],
    required_sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected_sections = normalize_required_sections(required_sections)
    requirement_ref = _text(requirement.get("requirement_id"))
    evidence, allowed = _requirement_grounding(state, requirement_ref)

    worksheet_raw: dict[str, dict[str, Any]] = {}
    for section in selected_sections:
        worksheet_raw[section] = {
            "specification": _plain_section(
                router,
                requirement=requirement,
                section=section,
                evidence=evidence,
            ),
            "constraint_evidence_refs": [],
        }
    worksheet = validate_worksheet(worksheet_raw, allowed, selected_sections)

    plan = {
        "requirement_ref": requirement_ref,
        "required_detail_sections": list(selected_sections),
        "engineering_worksheet": worksheet,
        "implementation_capabilities": _host_derived_capabilities(worksheet),
        "implementation_obligations": _host_derived_obligations(worksheet),
        "artifact_obligations": [],
        "grounded_bindings": [],
        "reuse_candidates": [],
        "verification_obligations": _host_derived_checks(requirement, worksheet),
    }
    validate_detailed_plan_grounding(plan, allowed)
    return plan


def _host_section_selection(
    requirements: list[Mapping[str, Any]],
    required_sections_by_requirement: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    requirement_ids = [str(item.get("requirement_id") or "") for item in requirements]
    if required_sections_by_requirement is None:
        return {
            requirement_id: normalize_required_sections()
            for requirement_id in requirement_ids
        }
    if not isinstance(required_sections_by_requirement, Mapping):
        raise ValueError("DETAILED_PLAN_SECTIONS: host selection must be a requirement mapping")
    unknown = set(str(key) for key in required_sections_by_requirement) - set(requirement_ids)
    if unknown:
        raise ValueError(
            "DETAILED_PLAN_SECTIONS: selection cites unknown requirement(s): "
            + ", ".join(sorted(unknown))
        )
    return {
        requirement_id: normalize_required_sections(
            required_sections_by_requirement.get(requirement_id)
        )
        for requirement_id in requirement_ids
    }


def compile_detailed_implementation_plans(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    required_sections_by_requirement: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    validate_planning_state(state, prompt=prompt)
    value = deepcopy(dict(state))
    requirements = _requirement_decisions(value)
    if not requirements:
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: no researched requirements exist")

    _preflight_detailed_planning(value, requirements)
    section_selection = _host_section_selection(requirements, required_sections_by_requirement)
    workers = min(len(requirements), router_native_model_parallelism(router))
    if workers <= 1:
        compiled = [
            _compile_requirement_plan(
                router,
                value,
                requirement,
                section_selection[str(requirement.get("requirement_id") or "")],
            )
            for requirement in requirements
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="planning-detail") as pool:
            futures = [
                pool.submit(
                    _compile_requirement_plan,
                    router,
                    value,
                    requirement,
                    section_selection[str(requirement.get("requirement_id") or "")],
                )
                for requirement in requirements
            ]
            compiled = [future.result() for future in futures]

    detailed: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for index, plan in enumerate(compiled, start=1):
        decision_id = f"detail_{index:03d}"
        detailed.append(
            {"decision_id": decision_id, "decision_type": "detailed_implementation_plan", **plan}
        )
        coverage.append(
            {
                "requirement_ref": plan["requirement_ref"],
                "status": "covered",
                "detailed_plan_ref": decision_id,
            }
        )

    value["decisions"] = [
        item
        for item in value.get("decisions", [])
        if not (
            isinstance(item, Mapping)
            and item.get("decision_type") == "detailed_implementation_plan"
        )
    ] + detailed
    value["coverage"] = coverage
    blocking = [
        item
        for item in value.get("unresolved", [])
        if isinstance(item, Mapping) and item.get("status") != "resolved"
    ]
    value["plan_ready"] = not blocking and len(coverage) == len(requirements)
    if not value["plan_ready"]:
        raise ValueError("DETAILED_PLAN_NOT_READY: unresolved planning obligations remain")

    result = _rehash(value)
    validate_planning_state(result, prompt=prompt)
    return result


__all__ = ["compile_detailed_implementation_plans"]
