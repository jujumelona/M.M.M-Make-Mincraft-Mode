from __future__ import annotations

"""Evidence-backed detailed-plan template between research and coder lowering.

One small-model decision is made per researched user-visible requirement. Every concrete
implementation obligation must cite grounded evidence already stored in the planning
state. Host code owns readiness and refuses semantic-only or evidence-free handoffs.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .planner_operation import planner_operation
from .planning_detail_template import WORKSHEET_SCHEMA, validate_worksheet, worksheet_prompt
from .planning_state_contract import validate_planning_state

_TOOL = "submit_detailed_implementation_plan"
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
                        "description": "Concrete implementation capability required by this requirement; never a vague feature label.",
                    },
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                },
                "required": ["capability", "evidence_refs"],
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
                        "description": "One executable obligation with actor/owner, action, condition and observable result where applicable.",
                    },
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                },
                "required": ["obligation", "evidence_refs"],
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
                        "description": "Artifact class such as source symbol, registry/data resource, generated resource, test, config or migration artifact.",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Why this artifact is required and which observable behavior or implementation invariant it realizes.",
                    },
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                },
                "required": ["kind", "purpose", "evidence_refs"],
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
                        "enum": ["reuse", "adapt", "reference_only", "new_required"],
                    },
                    "reason": {
                        "type": "string",
                        "description": "Specific compatibility/provenance/semantic reason for the reuse verdict, including what must change when adapting.",
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
                    "evidence_refs": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                },
                "required": ["check", "evidence_refs"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "engineering_worksheet",
        "implementation_capabilities",
        "implementation_obligations",
        "artifact_obligations",
        "reuse_candidates",
        "verification_obligations",
    ],
    "additionalProperties": False,
}

_SMALL_MODEL_PLAN_PROTOCOL = """SMALL-MODEL DETAILED-PLAN PROTOCOL
1. Scope lock: plan exactly the supplied requirement. Do not redesign unrelated requirements or host-owned target coordinates.
2. Evidence pass: read every supplied implementation_research record first; distinguish proved facts from examples, proposals and unresolved target bindings.
3. Worksheet pass: fill all ten engineering_worksheet sections using the canonical checklist. Never copy one generic sentence into multiple sections.
4. Capability pass: name the concrete technical capabilities the implementation must possess; each one needs allowed evidence.
5. Obligation pass: decompose implementation into independently executable obligations. State owner/action/condition/result instead of 'implement/support/handle X' alone.
6. Artifact pass: enumerate only artifacts actually required by the grounded design. Do not invent paths, symbols, IDs or APIs that evidence does not establish.
7. Reuse pass: classify each relevant source as reuse/adapt/reference_only/new_required and explain compatibility plus the adaptation boundary.
8. Verification pass: prove authored success, rejection/failure and important boundaries. Add reload, multiplayer/authority and resource checks when those branches apply.
9. Consistency pass: cross-check state transitions against algorithm, integration, network/persistence, artifacts and verification. Resolve contradictions before submission.
10. Fail closed: if evidence cannot safely establish an implementation detail, keep it explicitly unresolved instead of fabricating a target-specific fact.
"""


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _requirement_decisions(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in state.get("decisions", [])
        if isinstance(item, Mapping) and item.get("decision_type") == "requirement"
    ]


def _implementation_evidence(state: Mapping[str, Any], requirement_ref: str) -> list[Mapping[str, Any]]:
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


def _validate_refs(refs: Any, allowed: set[str], *, field: str, require: bool = True) -> list[str]:
    if not isinstance(refs, list):
        raise ValueError(f"DETAILED_PLAN_{field.upper()}: evidence_refs must be an array")
    values = list(dict.fromkeys(_text(ref) for ref in refs if _text(ref)))
    if require and not values:
        raise ValueError(f"DETAILED_PLAN_{field.upper()}: grounded evidence is required")
    unknown = [ref for ref in values if ref not in allowed]
    if unknown:
        raise ValueError(
            f"DETAILED_PLAN_{field.upper()}: unknown evidence refs: " + ", ".join(unknown)
        )
    return values


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def compile_detailed_implementation_plans(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile all researched requirements into concrete evidence-backed plan slices."""

    validate_planning_state(state, prompt=prompt)
    value = deepcopy(dict(state))
    requirements = _requirement_decisions(value)
    if not requirements:
        raise ValueError("DETAILED_PLAN_REQUIREMENTS: no researched requirements exist")

    detailed: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_ref = str(requirement.get("requirement_id") or "")
        evidence = _implementation_evidence(value, requirement_ref)
        allowed = _allowed_refs(evidence)
        if not evidence or not allowed:
            raise ValueError(
                f"DETAILED_PLAN_EVIDENCE: {requirement_ref} has no sufficient grounded implementation evidence"
            )
        context = {
            "requirement": deepcopy(dict(requirement)),
            "implementation_research": deepcopy(evidence),
            "allowed_evidence_refs": sorted(allowed),
            "engineering_worksheet_contract": worksheet_prompt(),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    _SMALL_MODEL_PLAN_PROTOCOL
                    + "\n"
                    + worksheet_prompt()
                    + "\nGROUNDING RULES:\n"
                    "- This is a fill-and-verify task, not a redesign task.\n"
                    "- Never invent API names, files, symbols, dependencies, versions, identifiers or implementation mechanisms absent from evidence.\n"
                    "- Every implementation capability and implementation obligation must cite one or more supplied allowed evidence refs.\n"
                    "- Artifact entries must state a concrete artifact purpose supported by evidence; omit artifacts that are not required.\n"
                    "- Reuse mode must reflect what the cited source really supports; a retrieved pattern is not proof that it can be copied unchanged.\n"
                    "- Verification checks must prove the user-visible requirement or an evidence-backed invariant; compilation alone never proves behavior.\n"
                    "- An inapplicable worksheet section must explain why using cited evidence; never silently omit it.\n"
                    "- New algorithms and proposed identifiers are design decisions, not retrieved facts.\n"
                    "- Keep unverified target-specific bindings explicitly separate from source examples.\n"
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
                parameters=_PARAMETERS,
                description=(
                    "Fill the complete grounded engineering worksheet and concrete implementation, artifact, reuse and verification obligations for exactly one requirement."
                ),
            )
        if not isinstance(raw, Mapping):
            raise ValueError("DETAILED_PLAN_MODEL: planner returned a non-object")

        worksheet = validate_worksheet(raw.get("engineering_worksheet"), allowed)
        capabilities: list[dict[str, Any]] = []
        for item in raw.get("implementation_capabilities", []):
            if not isinstance(item, Mapping) or not _text(item.get("capability")):
                raise ValueError("DETAILED_PLAN_CAPABILITY: invalid capability item")
            capabilities.append(
                {
                    "capability": _text(item.get("capability")),
                    "evidence_refs": _validate_refs(item.get("evidence_refs"), allowed, field="capability"),
                }
            )
        obligations: list[dict[str, Any]] = []
        for item in raw.get("implementation_obligations", []):
            if not isinstance(item, Mapping) or not _text(item.get("obligation")):
                raise ValueError("DETAILED_PLAN_OBLIGATION: invalid obligation item")
            obligations.append(
                {
                    "obligation": _text(item.get("obligation")),
                    "evidence_refs": _validate_refs(item.get("evidence_refs"), allowed, field="obligation"),
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
                    "evidence_refs": _validate_refs(item.get("evidence_refs"), allowed, field="artifact"),
                }
            )

        reuse: list[dict[str, Any]] = []
        for item in raw.get("reuse_candidates", []):
            if not isinstance(item, Mapping):
                continue
            ref = _text(item.get("evidence_ref"))
            mode = _text(item.get("mode"))
            reason = _text(item.get("reason"))
            if mode not in {"reuse", "adapt", "reference_only", "new_required"} or not reason:
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
                    "evidence_refs": _validate_refs(
                        item.get("evidence_refs"), allowed, field="verification", require=False
                    ),
                }
            )
        if not checks:
            raise ValueError(f"DETAILED_PLAN_VERIFICATION: {requirement_ref} has no verification plan")

        detailed.append(
            {
                "decision_id": f"detail_{len(detailed) + 1:03d}",
                "decision_type": "detailed_implementation_plan",
                "requirement_ref": requirement_ref,
                "engineering_worksheet": worksheet,
                "implementation_capabilities": capabilities,
                "implementation_obligations": obligations,
                "artifact_obligations": artifacts,
                "reuse_candidates": reuse,
                "verification_obligations": checks,
            }
        )
        coverage.append(
            {
                "requirement_ref": requirement_ref,
                "status": "covered",
                "detailed_plan_ref": detailed[-1]["decision_id"],
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
    return _rehash(value)


__all__ = ["compile_detailed_implementation_plans"]
