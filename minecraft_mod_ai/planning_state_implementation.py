from __future__ import annotations

"""Host-owned detailed-plan compilation from grounded research.

Each requirement is authored as one schema-constrained engineering worksheet. The host
owns requirement selection, worksheet sections, evidence identifiers, validation, and
plan assembly. The model never chooses its own response shape and detailed planning does
not depend on free-form section boundaries or reasoning-label parsing.
"""

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from typing import Any

from .model_concurrency import router_native_model_parallelism
from .planner_operation import planner_operation
from .root_cause_trace import emit_root_cause
from .planning_detail_contract import validate_detailed_plan_grounding
from .planning_detail_template import (
    normalize_required_sections,
    validate_worksheet,
    worksheet_prompt,
    worksheet_schema,
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


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _evidence_context(evidence: list[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for item in evidence:
        research_ref = _text(item.get("research_ref"))
        refs = ", ".join(
            _text(ref) for ref in item.get("evidence_refs", []) if _text(ref)
        )
        claims = item.get("claims") or []
        claim_text = " | ".join(_text(claim) for claim in claims if _text(claim))
        source = _text(item.get("source"))
        rows.append(
            f"- research_ref={research_ref}; evidence_refs=[{refs}]; "
            f"source={source or 'unspecified'}; claims={claim_text or 'no claim prose'}"
        )
    return "\n".join(rows)


def _worksheet_messages(
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    statement = _text(requirement.get("statement"))
    acceptance = requirement.get("acceptance")
    acceptance_rows = (
        [_text(item) for item in acceptance if _text(item)]
        if isinstance(acceptance, list)
        else []
    )
    acceptance_text = "\n".join(f"- {row}" for row in acceptance_rows) or "- none supplied"
    return [
        {
            "role": "system",
            "content": (
                "Complete exactly one engineering worksheet for one requirement. "
                "Return only the JSON object required by the supplied response schema. "
                "Do not emit analysis, reasoning, commentary, markdown, code fences, or keys "
                "outside that schema. The host owns the section set and provenance. "
                "Do not invent target API names, symbols, versions, repository paths, external "
                "facts, or evidence identifiers. Use only evidence_refs shown in the grounded "
                "context, and use an empty constraint_evidence_refs array when evidence does not "
                "constrain an authored design decision."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirement: {statement}\n"
                "Acceptance observations supplied by the requirement:\n"
                f"{acceptance_text}\n"
                "Grounded implementation evidence:\n"
                f"{_evidence_context(evidence)}\n\n"
                f"{worksheet_prompt(selected_sections)}\n"
                "Fill the schema once as one internally consistent worksheet."
            ),
        },
    ]


def _structured_output_text(exc: BaseException) -> str:
    return str(getattr(exc, "output", "") or "")


def _compile_requirement_worksheet(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
    allowed: set[str],
) -> dict[str, Any]:
    """Generate and validate exactly one schema-owned worksheet for one requirement."""

    requirement_ref = _text(requirement.get("requirement_id"))
    schema = worksheet_schema(selected_sections)
    messages = _worksheet_messages(requirement, selected_sections, evidence)
    raw = ""
    try:
        with planner_operation("detailed_worksheet"):
            raw = router.generate_text(
                "planner",
                messages,
                response_format="json",
                response_schema=schema,
                enable_tools=False,
            )
        decoded = json.loads(raw)
        return validate_worksheet(decoded, allowed, selected_sections)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raw_text = raw or _structured_output_text(exc)
        emit_root_cause(
            "detailed_worksheet_failure",
            stage="planning_state",
            operation="detailed_worksheet",
            gate="structured_worksheet_validation",
            result="FAIL",
            reason=f"{type(exc).__name__}: {exc}",
            details={
                "requirement_ref": requirement_ref,
                "selected_sections": list(selected_sections),
                "raw_output": raw_text,
                "raw_output_chars": len(raw_text),
                "response_format": "json",
                "response_schema": schema,
                "allowed_evidence_refs": sorted(allowed),
                "parser_rule": (
                    "one JSON worksheet matching the host-owned response schema; "
                    "all sections and evidence refs are validated by the host"
                ),
            },
            exc=exc,
        )
        raise


def _host_derived_capabilities(
    worksheet: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "capability": f"Preserve the {section} contract: {_text(row.get('specification'))}",
            "constraint_evidence_refs": [],
        }
        for section, row in worksheet.items()
    ]


def _host_derived_obligations(
    worksheet: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "obligation": f"Implement and verify the {section} contract: {_text(row.get('specification'))}",
            "constraint_evidence_refs": [],
        }
        for section, row in worksheet.items()
    ]


def _host_derived_checks(
    requirement: Mapping[str, Any], worksheet: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    acceptance = requirement.get("acceptance")
    checks = (
        [_text(item) for item in acceptance if _text(item)]
        if isinstance(acceptance, list)
        else []
    )
    if not checks:
        verification = worksheet.get("verification", {})
        specification = _text(verification.get("specification"))
        if specification:
            checks = [specification]
    if not checks:
        checks = [
            "Given the requirement is exercised, verify the observable behavior: "
            + _text(requirement.get("statement"))
        ]
    return [{"check": check, "constraint_evidence_refs": []} for check in checks]


def _assemble_requirement_plan(
    requirement: Mapping[str, Any],
    requirement_ref: str,
    selected_sections: tuple[str, ...],
    worksheet: Mapping[str, Mapping[str, Any]],
    allowed: set[str],
) -> dict[str, Any]:
    validated_worksheet = validate_worksheet(worksheet, allowed, selected_sections)
    plan = {
        "requirement_ref": requirement_ref,
        "required_detail_sections": list(selected_sections),
        "engineering_worksheet": validated_worksheet,
        "implementation_capabilities": _host_derived_capabilities(validated_worksheet),
        "implementation_obligations": _host_derived_obligations(validated_worksheet),
        "artifact_obligations": [],
        "grounded_bindings": [],
        "reuse_candidates": [],
        "verification_obligations": _host_derived_checks(
            requirement, validated_worksheet
        ),
    }
    validate_detailed_plan_grounding(plan, allowed)
    return plan


def _compile_requirement_plan(
    router: Any,
    state: Mapping[str, Any],
    requirement: Mapping[str, Any],
    required_sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected_sections = normalize_required_sections(required_sections)
    requirement_ref = _text(requirement.get("requirement_id"))
    evidence, allowed = _requirement_grounding(state, requirement_ref)
    worksheet = _compile_requirement_worksheet(
        router,
        requirement=requirement,
        selected_sections=selected_sections,
        evidence=evidence,
        allowed=allowed,
    )
    return _assemble_requirement_plan(
        requirement,
        requirement_ref,
        selected_sections,
        worksheet,
        allowed,
    )


def _compile_requirement_plans_parallel(
    router: Any,
    state: Mapping[str, Any],
    requirements: list[Mapping[str, Any]],
    section_selection: Mapping[str, tuple[str, ...]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Run independent requirement worksheets in parallel."""

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="planning-detail") as pool:
        futures = [
            pool.submit(
                _compile_requirement_plan,
                router,
                state,
                requirement,
                section_selection[_text(requirement.get("requirement_id"))],
            )
            for requirement in requirements
        ]
        return [future.result() for future in futures]


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
        compiled = _compile_requirement_plans_parallel(
            router,
            value,
            requirements,
            section_selection,
            workers=workers,
        )

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
