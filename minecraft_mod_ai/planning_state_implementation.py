from __future__ import annotations

"""Host-owned detailed-plan compilation from grounded research.

The model never authors the plan container, evidence identifiers, or a large JSON/tool
payload. Host code owns structure and validation. Detailed planning follows the semantic
sections selected by research/host policy: each selected section is one complete work
unit. Token counts do not decide decomposition, retries, or fallback behavior.
"""

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import re
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


_META_REASONING_TAG_RE = re.compile(
    r"^\s*<\s*think(?:ing)?\b[^>]*>.*?<\s*/\s*think(?:ing)?\s*>\s*",
    re.IGNORECASE | re.DOTALL,
)
_META_REASONING_LABEL_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:thinking\s+process|reasoning|analysis)\s*(?:\*\*|__)?\s*:\s*",
    re.IGNORECASE,
)
_FINAL_OUTPUT_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:final(?:\s+(?:answer|specification))?|specification|answer)\s*(?:\*\*|__)?\s*:\s*",
    re.IGNORECASE,
)
_CONTINUITY_CONTEXT_MAX_CHARS = 8_000
_CONTINUITY_ELLIPSIS = " … "


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


def _strip_leading_meta_reasoning(raw: Any, section: str) -> str:
    value = str(raw or "").strip()
    while True:
        stripped = _META_REASONING_TAG_RE.sub("", value, count=1).strip()
        if stripped == value:
            break
        value = stripped

    if _META_REASONING_LABEL_RE.match(value):
        final_output = _FINAL_OUTPUT_LABEL_RE.search(value)
        if final_output is None:
            raise ValueError(
                f"DETAILED_PLAN_META_REASONING: {section} returned reasoning without an explicit final-output boundary"
            )
        value = value[final_output.end():].strip()
    return value


def _normalize_section_text(raw: Any, section: str) -> str:
    value = _strip_leading_meta_reasoning(raw, section)
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


def _bounded_continuity_excerpt(value: str, limit: int) -> str:
    normalized = _text(value)
    if limit <= 0:
        return ""
    if len(normalized) <= limit:
        return normalized
    if limit <= len(_CONTINUITY_ELLIPSIS):
        return normalized[:limit]
    body_budget = limit - len(_CONTINUITY_ELLIPSIS)
    head_budget = (body_budget * 2) // 3
    tail_budget = body_budget - head_budget
    if tail_budget <= 0:
        return normalized[:head_budget] + _CONTINUITY_ELLIPSIS
    return (
        normalized[:head_budget]
        + _CONTINUITY_ELLIPSIS
        + normalized[-tail_budget:]
    )


def _continuity_context(specifications: Mapping[str, str]) -> str:
    if not specifications:
        return "- No earlier section has been authored for this requirement."

    normalized = [
        (str(section), _text(specification))
        for section, specification in specifications.items()
    ]
    full = "\n".join(
        f"- {section}: {specification}"
        for section, specification in normalized
    )
    if len(full) <= _CONTINUITY_CONTEXT_MAX_CHARS:
        return full

    label_overhead = sum(len(f"- {section}: ") for section, _ in normalized)
    newline_overhead = max(0, len(normalized) - 1)
    available = max(
        0,
        _CONTINUITY_CONTEXT_MAX_CHARS - label_overhead - newline_overhead,
    )
    share, remainder = divmod(available, len(normalized))
    rows = [
        f"- {section}: {_bounded_continuity_excerpt(specification, share + (index < remainder))}"
        for index, (section, specification) in enumerate(normalized)
    ]
    return "\n".join(rows)


def _plain_section(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    section: str,
    evidence: list[Mapping[str, Any]],
    prior_specifications: Mapping[str, str] | None = None,
) -> str:
    checklist = "; ".join(DETAIL_SLOT_GUIDANCE[section])
    statement = _text(requirement.get("statement"))
    continuity = _continuity_context(prior_specifications or {})
    messages = [
        {
            "role": "system",
            "content": (
                "Write exactly one complete semantic engineering-design section for a small-model planning pipeline. "
                "The host has already chosen this section because it is a meaningful planning unit; do not split or "
                "resize the work based on token length. Return plain prose only: no JSON, YAML, XML, tool/function "
                "call, code fence, object keys, or evidence identifiers. The host owns all structure and provenance. "
                "Do not invent API names, symbols, versions, dependencies, repository paths, or source facts. Keep "
                "the design abstract where target facts are not established. Be concrete about actors, state, branches, "
                "limits, failure behavior, and observable outcomes that belong to this section."
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
                "Earlier completed semantic sections for continuity only:\n"
                f"{continuity}\n"
                "Write only the complete section specification."
            ),
        },
    ]
    with planner_operation(f"detailed_section:{section}"):
        raw = router.generate_text(
            "planner",
            messages,
            response_format="text",
            enable_tools=False,
        )
    return _normalize_section_text(raw, section)


def _compile_requirement_specifications(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    selected_sections: tuple[str, ...],
    evidence: list[Mapping[str, Any]],
) -> dict[str, str]:
    """Author the host-selected semantic sections once, in dependency-preserving order.

    There is deliberately no token-derived batching, token escalation, repair retry, or
    fallback path here. The selected section names are the decomposition boundary.
    Earlier completed sections are passed forward as continuity context so a small model
    can keep one requirement coherent without authoring a monolithic response.
    """

    specifications: dict[str, str] = {}
    seen: set[str] = set()
    for section in selected_sections:
        specification = _plain_section(
            router,
            requirement=requirement,
            section=section,
            evidence=evidence,
            prior_specifications=specifications,
        )
        normalized = _text(specification).casefold()
        if normalized in seen:
            raise ValueError(
                f"DETAILED_PLAN_DUPLICATE_SECTION: {section} duplicated an earlier semantic section"
            )
        specifications[section] = specification
        seen.add(normalized)
    return specifications


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
    specifications: Mapping[str, str],
    allowed: set[str],
) -> dict[str, Any]:
    worksheet_raw = {
        section: {
            "specification": specifications[section],
            "constraint_evidence_refs": [],
        }
        for section in selected_sections
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


def _compile_requirement_plan(
    router: Any,
    state: Mapping[str, Any],
    requirement: Mapping[str, Any],
    required_sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected_sections = normalize_required_sections(required_sections)
    requirement_ref = _text(requirement.get("requirement_id"))
    evidence, allowed = _requirement_grounding(state, requirement_ref)
    specifications = _compile_requirement_specifications(
        router,
        requirement=requirement,
        selected_sections=selected_sections,
        evidence=evidence,
    )
    return _assemble_requirement_plan(
        requirement,
        requirement_ref,
        selected_sections,
        specifications,
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
    """Run independent requirements in parallel; preserve section order inside each one."""

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