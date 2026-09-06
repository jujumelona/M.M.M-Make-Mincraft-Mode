from __future__ import annotations

"""Detailed-plan compilation between grounded research and coder lowering.

One small-model decision is made per researched user-visible requirement. Host code owns
readiness, section applicability, evidence closure and target-fact validation. Authored
design obligations are intentionally distinct from externally grounded bindings: design
may be new, while API/version/repository/dependency/source facts must cite evidence.
"""

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

from .model_concurrency import router_native_model_parallelism
from .planner_operation import planner_operation
from .planning_detail_contract import (
    GROUNDED_BINDING_KINDS,
    REUSE_MODES,
    validate_detailed_plan_grounding,
    validate_evidence_refs,
)
from .planning_detail_template import (
    WORKSHEET_SCHEMA,
    normalize_required_sections,
    validate_worksheet,
    worksheet_prompt,
    worksheet_schema,
)
from .planning_state_contract import validate_planning_state

_TOOL = "submit_detailed_implementation_plan"
_GROUNDED_BINDING_KINDS = set(GROUNDED_BINDING_KINDS)
_REUSE_MODES = set(REUSE_MODES)


def _constraint_refs_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "uniqueItems": True,
        "description": (
            "Evidence that constrains this authored design row. Empty is valid when the row is a new design decision rather than an external fact."
        ),
        "items": {"type": "string"},
    }


_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "engineering_worksheet": WORKSHEET_SCHEMA,
        "implementation_capabilities": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "description": "Concrete technical capability the authored implementation must possess; never a vague feature label.",
                    },
                    "constraint_evidence_refs": _constraint_refs_schema(),
                },
                "required": ["capability", "constraint_evidence_refs"],
                "additionalProperties": False,
            },
        },
        "implementation_obligations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "obligation": {
                        "type": "string",
                        "description": "One executable authored obligation with actor/owner, action, condition and observable result where applicable.",
                    },
                    "constraint_evidence_refs": _constraint_refs_schema(),
                },
                "required": ["obligation", "constraint_evidence_refs"],
                "additionalProperties": False,
            },
        },
        "artifact_obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "description": "Artifact class such as source component, registry/data resource, generated resource, test, config or migration artifact; do not invent a target path or symbol here.",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Why this authored artifact class is required and which observable behavior or implementation invariant it realizes.",
                    },
                    "constraint_evidence_refs": _constraint_refs_schema(),
                },
                "required": ["kind", "purpose", "constraint_evidence_refs"],
                "additionalProperties": False,
            },
        },
        "grounded_bindings": {
            "type": "array",
            "description": (
                "Externally verifiable implementation facts only. Every API/symbol/version/repository/dependency/source-behavior claim belongs here and requires evidence."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": sorted(_GROUNDED_BINDING_KINDS),
                    },
                    "fact": {
                        "type": "string",
                        "description": "One concrete externally verifiable fact established by supplied evidence; never an authored design preference.",
                    },
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                },
                "required": ["kind", "fact", "evidence_refs"],
                "additionalProperties": False,
            },
        },
        "reuse_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_ref": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": sorted(_REUSE_MODES),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Specific compatibility/provenance/semantic reason for the evidence-backed reuse verdict, including what must change when adapting.",
                    },
                },
                "required": ["evidence_ref", "mode", "reason"],
                "additionalProperties": False,
            },
        },
        "verification_obligations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "check": {
                        "type": "string",
                        "description": "Observable Given/When/Then-style proof obligation with expected result; include success, rejection and relevant boundary cases.",
                    },
                    "constraint_evidence_refs": _constraint_refs_schema(),
                },
                "required": ["check", "constraint_evidence_refs"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "engineering_worksheet",
        "implementation_capabilities",
        "implementation_obligations",
        "artifact_obligations",
        "grounded_bindings",
        "reuse_candidates",
        "verification_obligations",
    ],
    "additionalProperties": False,
}

_SMALL_MODEL_PLAN_PROTOCOL = """SMALL-MODEL DETAILED-PLAN PROTOCOL
1. Scope lock: plan exactly the supplied requirement. Do not redesign unrelated requirements or host-owned target coordinates.
2. Evidence pass: read every supplied implementation_research record first; distinguish proved external facts from examples, authored proposals and unresolved target bindings.
3. Worksheet pass: fill exactly the host-required engineering_worksheet sections using the canonical checklists. These specifications are authored design contracts, not quotations from research.
4. Capability pass: name concrete technical capabilities. Use constraint_evidence_refs only when research constrains the capability; an empty array is valid for a new design decision.
5. Obligation pass: decompose implementation into independently executable authored obligations. State owner/action/condition/result instead of 'implement/support/handle X' alone.
6. Artifact pass: enumerate only artifact classes required by the authored design. Do not invent target paths, symbols or identifiers.
7. Binding pass: put every claimed API, symbol, version compatibility, repository fact, dependency or source behavior into grounded_bindings with one or more allowed evidence refs. Never put a design choice there.
8. Reuse pass: classify each relevant source as reuse/adapt/reference_only/new_required and explain compatibility plus the adaptation boundary. Reuse verdicts are evidence-backed facts.
9. Verification pass: prove authored success, rejection/failure and important boundaries. Add reload, multiplayer/authority and resource checks when those branches apply.
10. Fail closed: if research cannot establish an external target fact, omit that factual binding and keep the design abstract instead of fabricating one.
"""


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _parameters_for_sections(required_sections: Iterable[str] | None) -> dict[str, Any]:
    """Build one tool schema from the same trusted section selection used by prompt/validation."""

    selected = normalize_required_sections(required_sections)
    parameters = deepcopy(_PARAMETERS)
    parameters["properties"]["engineering_worksheet"] = worksheet_schema(selected)
    return parameters


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
    return [
        item
        for item in state.get("evidence", [])
        if isinstance(item, Mapping)
        and str(item.get("research_ref") or "") in research_ids
        and item.get("sufficient") is True
    ]


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
    """Reject deterministic readiness failures before the first expensive model call."""

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


def _constraint_refs(refs: Any, allowed: set[str], *, field: str) -> list[str]:
    return _validate_refs(refs, allowed, field=field, require=False)


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _compile_requirement_plan(
    router: Any,
    state: Mapping[str, Any],
    requirement: Mapping[str, Any],
    required_sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compile and validate one independent requirement without mutating shared state."""

    selected_sections = normalize_required_sections(required_sections)
    requirement_ref = str(requirement.get("requirement_id") or "")
    evidence, allowed = _requirement_grounding(state, requirement_ref)

    context = {
        "requirement": deepcopy(dict(requirement)),
        "implementation_research": deepcopy(evidence),
        "allowed_evidence_refs": sorted(allowed),
        "host_required_worksheet_sections": list(selected_sections),
    }
    messages = [
        {
            "role": "system",
            "content": (
                _SMALL_MODEL_PLAN_PROTOCOL
                + "\n"
                + worksheet_prompt(selected_sections)
                + "\nGROUNDING RULES:\n"
                "- This is a fill-and-verify task, not a redesign of requirement scope.\n"
                "- Authored behavior, algorithms, state machines, constants and verification scenarios are design decisions; do not attach unrelated evidence merely to satisfy a field.\n"
                "- constraint_evidence_refs may be empty. Populate them only when supplied evidence constrains that design row.\n"
                "- Every API/symbol/version/repository/dependency/source-behavior claim belongs in grounded_bindings and every grounded binding must cite allowed evidence.\n"
                "- Never invent target paths, API names, symbols, dependencies, versions or identifiers absent from evidence.\n"
                "- Reuse mode must reflect what the cited source really supports; a retrieved pattern is not proof that it can be copied unchanged.\n"
                "- Verification checks prove the authored requirement; compilation alone never proves behavior.\n"
                "- Fill exactly the host-required worksheet sections; never add a branch the host omitted.\n"
                "- Prefer explicit numbers, units, state owners, branch conditions and expected outcomes over vague quality adjectives."
            ),
        },
        {"role": "user", "content": str(context)},
    ]
    with planner_operation("detailed_implementation_plan", output_tokens=6144):
        raw = router.generate_tool_decision(
            "planner",
            messages,
            tool_name=_TOOL,
            parameters=_parameters_for_sections(selected_sections),
            description=(
                "Fill the host-required authored engineering design, evidence-backed external bindings, reuse assessment and verification obligations for exactly one requirement."
            ),
        )
    if not isinstance(raw, Mapping):
        raise ValueError("DETAILED_PLAN_MODEL: planner returned a non-object")

    worksheet = validate_worksheet(
        raw.get("engineering_worksheet"),
        allowed,
        selected_sections,
    )

    capabilities: list[dict[str, Any]] = []
    for item in raw.get("implementation_capabilities", []):
        if not isinstance(item, Mapping) or not _text(item.get("capability")):
            raise ValueError("DETAILED_PLAN_CAPABILITY: invalid capability item")
        capabilities.append(
            {
                "capability": _text(item.get("capability")),
                "constraint_evidence_refs": _constraint_refs(
                    item.get("constraint_evidence_refs"),
                    allowed,
                    field="capability_constraint",
                ),
            }
        )

    obligations: list[dict[str, Any]] = []
    for item in raw.get("implementation_obligations", []):
        if not isinstance(item, Mapping) or not _text(item.get("obligation")):
            raise ValueError("DETAILED_PLAN_OBLIGATION: invalid obligation item")
        obligations.append(
            {
                "obligation": _text(item.get("obligation")),
                "constraint_evidence_refs": _constraint_refs(
                    item.get("constraint_evidence_refs"),
                    allowed,
                    field="obligation_constraint",
                ),
            }
        )
    if not capabilities or not obligations:
        raise ValueError(
            f"DETAILED_PLAN_EMPTY: {requirement_ref} lacks concrete capabilities/obligations"
        )

    artifacts: list[dict[str, Any]] = []
    for item in raw.get("artifact_obligations", []):
        if not isinstance(item, Mapping):
            continue
        kind, purpose = _text(item.get("kind")), _text(item.get("purpose"))
        if not kind or not purpose:
            raise ValueError("DETAILED_PLAN_ARTIFACT: kind and purpose are required")
        artifacts.append(
            {
                "kind": kind,
                "purpose": purpose,
                "constraint_evidence_refs": _constraint_refs(
                    item.get("constraint_evidence_refs"),
                    allowed,
                    field="artifact_constraint",
                ),
            }
        )

    grounded_bindings: list[dict[str, Any]] = []
    for item in raw.get("grounded_bindings", []):
        if not isinstance(item, Mapping):
            raise ValueError("DETAILED_PLAN_BINDING: invalid grounded binding")
        kind = _text(item.get("kind"))
        fact = _text(item.get("fact"))
        if kind not in _GROUNDED_BINDING_KINDS or not fact:
            raise ValueError("DETAILED_PLAN_BINDING: invalid kind or missing fact")
        grounded_bindings.append(
            {
                "kind": kind,
                "fact": fact,
                "evidence_refs": _validate_refs(
                    item.get("evidence_refs"), allowed, field="binding"
                ),
            }
        )

    reuse: list[dict[str, Any]] = []
    for item in raw.get("reuse_candidates", []):
        if not isinstance(item, Mapping):
            continue
        ref = _text(item.get("evidence_ref"))
        mode = _text(item.get("mode"))
        reason = _text(item.get("reason"))
        if mode not in _REUSE_MODES or not reason:
            raise ValueError("DETAILED_PLAN_REUSE: invalid reuse verdict or missing rationale")
        _validate_refs([ref], allowed, field="reuse")
        reuse.append({"evidence_ref": ref, "mode": mode, "reason": reason})

    checks: list[dict[str, Any]] = []
    for item in raw.get("verification_obligations", []):
        if not isinstance(item, Mapping) or not _text(item.get("check")):
            raise ValueError("DETAILED_PLAN_VERIFICATION: invalid verification item")
        checks.append(
            {
                "check": _text(item.get("check")),
                "constraint_evidence_refs": _constraint_refs(
                    item.get("constraint_evidence_refs"),
                    allowed,
                    field="verification_constraint",
                ),
            }
        )
    if not checks:
        raise ValueError(
            f"DETAILED_PLAN_VERIFICATION: {requirement_ref} has no verification plan"
        )

    plan = {
        "requirement_ref": requirement_ref,
        "required_detail_sections": list(selected_sections),
        "engineering_worksheet": worksheet,
        "implementation_capabilities": capabilities,
        "implementation_obligations": obligations,
        "artifact_obligations": artifacts,
        "grounded_bindings": grounded_bindings,
        "reuse_candidates": reuse,
        "verification_obligations": checks,
    }
    validate_detailed_plan_grounding(plan, allowed)
    return plan


def _host_section_selection(
    requirements: list[Mapping[str, Any]],
    required_sections_by_requirement: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """Normalize trusted per-requirement selections before any concurrent model call."""

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
    """Compile researched requirements using only host-owned worksheet applicability."""

    validate_planning_state(state, prompt=prompt)
    value = deepcopy(dict(state))
    requirements = _requirement_decisions(value)
    if not requirements:
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: no researched requirements exist")

    _preflight_detailed_planning(value, requirements)
    section_selection = _host_section_selection(
        requirements, required_sections_by_requirement
    )

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
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="planning-detail"
        ) as pool:
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
            {
                "decision_id": decision_id,
                "decision_type": "detailed_implementation_plan",
                **plan,
            }
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
