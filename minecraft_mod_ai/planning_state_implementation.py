from __future__ import annotations

"""Evidence-backed detailed-plan template between research and coder lowering.

One small-model decision is made per researched user-visible requirement.  Every concrete
implementation obligation must cite grounded evidence already stored in the planning
state.  Host code owns readiness and refuses semantic-only or evidence-free handoffs.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .planner_operation import planner_operation
from .planning_state_contract import validate_planning_state

_TOOL = "submit_detailed_implementation_plan"
_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "implementation_capabilities": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string"},
                    "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string"}},
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
                    "obligation": {"type": "string"},
                    "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string"}},
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
                    "kind": {"type": "string"},
                    "purpose": {"type": "string"},
                    "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string"}},
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
                    "mode": {"type": "string", "enum": ["reuse", "adapt", "reference_only", "new_required"]},
                    "reason": {"type": "string"},
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
                    "check": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["check", "evidence_refs"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "implementation_capabilities",
        "implementation_obligations",
        "artifact_obligations",
        "reuse_candidates",
        "verification_obligations",
    ],
    "additionalProperties": False,
}


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
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Fill one detailed implementation-plan template from the supplied grounded evidence. "
                    "This is not a redesign task. Do not invent API names, files, symbols, dependencies, or "
                    "implementation mechanisms absent from the evidence. Each capability and implementation "
                    "obligation must cite one or more supplied evidence refs. Artifact entries must describe "
                    "a concrete artifact purpose supported by evidence. Reuse mode must reflect what the cited "
                    "source actually supports. Verification checks must prove the user-visible requirement or "
                    "an evidence-backed implementation invariant."
                ),
            },
            {"role": "user", "content": str(context)},
        ]
        with planner_operation("detailed_implementation_plan", output_tokens=2048):
            raw = router.generate_tool_decision(
                "planner",
                messages,
                tool_name=_TOOL,
                parameters=_PARAMETERS,
                description="Submit one grounded detailed implementation plan for a single requirement.",
            )
        if not isinstance(raw, Mapping):
            raise ValueError("DETAILED_PLAN_MODEL: planner returned a non-object")

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
        item for item in value.get("decisions", [])
        if not (isinstance(item, Mapping) and item.get("decision_type") == "detailed_implementation_plan")
    ] + detailed
    value["coverage"] = coverage
    blocking = [
        item
        for item in value.get("unresolved", [])
        if isinstance(item, Mapping)
        and item.get("status") != "resolved"
        and item.get("resolution_route") != "user_only"
    ]
    value["plan_ready"] = not blocking and len(coverage) == len(requirements)
    if not value["plan_ready"]:
        raise ValueError("DETAILED_PLAN_NOT_READY: unresolved planning obligations remain")
    return _rehash(value)


__all__ = ["compile_detailed_implementation_plans"]
