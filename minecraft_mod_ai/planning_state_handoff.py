from __future__ import annotations

"""Lower a plan-ready planning-state SSOT into the existing evidence-plan catalog.

The legacy catalog remains a downstream interchange shape, not a semantic authority.
Every implementation capability/obligation in it originates from the detailed grounded
planning state and carries its research provenance.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from . import evidence_first_planning as _evidence
from .planning_state_contract import validate_planning_state


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _requirements(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item for item in state.get("decisions", [])
        if isinstance(item, Mapping) and item.get("decision_type") == "requirement"
    ]


def _details(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("requirement_ref") or ""): item
        for item in state.get("decisions", [])
        if isinstance(item, Mapping)
        and item.get("decision_type") == "detailed_implementation_plan"
        and str(item.get("requirement_ref") or "")
    }


def _source_span(prompt: str, requirement: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    quote = str(requirement.get("prompt_quote") or "").strip()
    if quote:
        start = prompt.find(quote)
        if start < 0:
            raise ValueError("PLANNING_HANDOFF_SOURCE: requirement prompt quote is stale")
        text = quote
    else:
        goal = state.get("goal")
        source = goal.get("source") if isinstance(goal, Mapping) else None
        if not isinstance(source, Mapping):
            raise ValueError("PLANNING_HANDOFF_SOURCE: derived requirement has no authored goal anchor")
        start = int(source.get("char_start"))
        text = str(source.get("text") or "")
        if prompt[start : start + len(text)] != text:
            raise ValueError("PLANNING_HANDOFF_SOURCE: authored goal anchor is stale")
    return {
        "source_id": "requested_prompt",
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
        "text_sha256": _evidence._sha(text),
    }


def _implementation_queries(state: Mapping[str, Any], requirement_ref: str) -> list[str]:
    values: list[str] = []
    for item in state.get("research_queue", []) if isinstance(state.get("research_queue"), list) else []:
        if not isinstance(item, Mapping) or str(item.get("requirement_ref") or "") != requirement_ref:
            continue
        for query in item.get("queries", []) if isinstance(item.get("queries"), list) else []:
            text = _text(query)
            if text and text not in values:
                values.append(text)
    return values


def _flatten_detail(detail: Mapping[str, Any], key: str, value_key: str) -> list[str]:
    raw = detail.get(key)
    if not isinstance(raw, list):
        return []
    return list(
        dict.fromkeys(
            _text(item.get(value_key))
            for item in raw
            if isinstance(item, Mapping) and _text(item.get(value_key))
        )
    )


def build_request_catalog_from_planning_state(prompt: str, state: Mapping[str, Any]) -> dict[str, Any]:
    validate_planning_state(state, prompt=prompt)
    if state.get("plan_ready") is not True:
        raise ValueError("PLANNING_HANDOFF_READY: request catalog requires a plan-ready state")

    requirements = _requirements(state)
    details = _details(state)
    if not requirements or len(details) != len(requirements):
        raise ValueError("PLANNING_HANDOFF_COVERAGE: every requirement needs one detailed plan")

    output: list[dict[str, Any]] = []
    for index, requirement in enumerate(requirements):
        requirement_id = str(requirement.get("requirement_id") or "")
        detail = details.get(requirement_id)
        if detail is None:
            raise ValueError(f"PLANNING_HANDOFF_DETAIL: missing detail for {requirement_id}")
        statement = _text(requirement.get("statement"))
        implementation_capabilities = _flatten_detail(
            detail, "implementation_capabilities", "capability"
        )
        implementation_obligations = _flatten_detail(
            detail, "implementation_obligations", "obligation"
        )
        if not implementation_capabilities or not implementation_obligations:
            raise ValueError(
                f"PLANNING_HANDOFF_SEMANTIC_ONLY: {requirement_id} has no concrete implementation detail"
            )
        capability = "researched." + _evidence._sha(
            {"requirement": statement, "index": index}
        )[7:23]
        artifacts = []
        for artifact in detail.get("artifact_obligations", []) if isinstance(detail.get("artifact_obligations"), list) else []:
            if not isinstance(artifact, Mapping):
                continue
            artifacts.append(
                {
                    "kind": _text(artifact.get("kind")),
                    "purpose": _text(artifact.get("purpose")),
                    "status": "REQUIRED_DESIGN_AND_GENERATION",
                    "evidence_refs": list(artifact.get("evidence_refs") or []),
                }
            )
        acceptance = list(
            dict.fromkeys(
                [
                    _text(item)
                    for item in requirement.get("acceptance", [])
                    if _text(item)
                ]
                + _flatten_detail(detail, "verification_obligations", "check")
            )
        )
        output.append(
            {
                "requirement_id": requirement_id,
                "capability": capability,
                "statement": statement,
                "semantic_statement": statement,
                "mandatory": True,
                "provenance_role": "authored" if requirement.get("prompt_quote") else "grounded_reference_derivation",
                "source_span": _source_span(prompt, requirement, state),
                "evidence_refs": list(requirement.get("evidence_refs") or []),
                "derived_from": list(requirement.get("evidence_refs") or []),
                "depends_on": [],
                "provides": [_evidence._canonical_capability(capability)],
                "gameplay_capabilities": [capability],
                "implementation_capabilities": implementation_capabilities,
                "implementation_obligations": implementation_obligations,
                "artifact_task_ids": [
                    _evidence._stable_id(
                        "task", implementation,
                        {"requirement_id": requirement_id, "layer": "researched_implementation"},
                    )
                    for implementation in implementation_capabilities
                ],
                "semantic_type": "researched_gameplay_requirement",
                "unlock_policy": {
                    "required_capabilities": [],
                    "required_requirement_refs": [],
                    "optional_capabilities": [],
                    "optional_requirement_refs": [],
                    "policy": "grounded_planning_state_only",
                },
                "artifact_obligations": artifacts,
                "design_resolution_obligations": implementation_obligations,
                "runtime_acceptance": list(acceptance),
                "semantic_status": "RESOLVED",
                "unresolved_spans": [],
                "acceptance": acceptance,
                "observable_behavior": {
                    "given": "the researched requirement preconditions are established",
                    "when": statement,
                    "then": acceptance[0] if acceptance else statement,
                },
                "template_profile": {
                    "template_id": "grounded_researched_requirement",
                    "architecture_owner": "planning_state",
                },
                "search_queries": _implementation_queries(state, requirement_id),
                "reuse_candidates": deepcopy(detail.get("reuse_candidates") or []),
                "detailed_plan_ref": str(detail.get("decision_id") or ""),
            }
        )

    catalog: dict[str, Any] = {
        "prompt_sha256": _evidence._sha(prompt),
        "prompt_char_length": len(prompt),
        "purpose": _text(state.get("goal", {}).get("statement") if isinstance(state.get("goal"), Mapping) else prompt),
        "requirements": output,
        "constraints": [],
        "non_goals": [],
        "deployment_expectations": [],
        "requirement_graph": {
            "node_ids": [item["requirement_id"] for item in output],
            "edges": [],
        },
        "dependency_provenance": [],
        "semantic_audit": {
            "status": "APPROVED",
            "authored_clause_count": len(output),
            "covered_clause_count": len(output),
            "unresolved_clause_count": 0,
            "unsupported_design_choice_count": 0,
            "normal_model_turns": "bounded_prompt_and_research_templates",
            "semantic_model_turns": "bounded",
            "retrieval_model_turns": "bounded_per_information_need",
            "generation_policy": "prompt_first_grounded_state_machine",
            "model_generated_planning_json": True,
            "source_grounding_owner": "host_validated_evidence",
            "capability_id_owner": "host_handoff",
            "dependency_owner": "host_downstream_plan",
            "implementation_architecture_owner": "grounded_detailed_plan",
            "research_query_owner": "bounded_query_compiler",
        },
        "planning_state_sha256": state.get("state_sha256"),
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = _evidence._hash_without(catalog, "catalog_sha256")
    _evidence._validate_request_catalog(catalog, prompt=prompt)
    return catalog


__all__ = ["build_request_catalog_from_planning_state"]
