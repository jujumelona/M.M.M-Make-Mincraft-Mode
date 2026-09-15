from __future__ import annotations

"""Lower a plan-ready planning-state SSOT into the existing evidence-plan catalog.

The legacy catalog remains a downstream interchange shape, not a semantic authority.
Every implementation capability/obligation originates from the detailed grounded planning
state. Prompt provenance is resolved from structural ``prompt_refs`` rather than a second,
fragile free-text quote contract.
"""

from collections.abc import Mapping
from typing import Any

from . import evidence_first_planning as _evidence
from .acceptance_contracts import canonical_public_acceptance
from .planning_handoff_contract import project_detailed_plan_for_request_catalog
from .planning_state_contract import validate_planning_state



def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _requirements(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in state.get("decisions", [])
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


def _sufficient_refs(state: Mapping[str, Any]) -> set[str]:
    return {
        _text(ref)
        for item in state.get("evidence", [])
        if isinstance(item, Mapping) and item.get("sufficient") is True
        for ref in item.get("evidence_refs", [])
        if _text(ref)
    }


def _prompt_source_by_ref(
    state: Mapping[str, Any], ref: str
) -> Mapping[str, Any] | None:
    if ref == "goal":
        goal = state.get("goal")
        source = goal.get("source") if isinstance(goal, Mapping) else None
        return source if isinstance(source, Mapping) else None
    for item in (
        state.get("known", []) if isinstance(state.get("known"), list) else []
    ):
        if not isinstance(item, Mapping) or str(item.get("known_id") or "") != ref:
            continue
        source = item.get("source")
        return source if isinstance(source, Mapping) else None
    return None


def _validated_source_span(
    prompt: str, source: Mapping[str, Any] | None
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        start = source.get("char_start")
        end = source.get("char_end")
        text = str(source.get("text") or "")
        if (
            type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(prompt)
            and prompt[start:end] == text
        ):
            return {
                "source_id": "requested_prompt",
                "char_start": start,
                "char_end": end,
                "text": text,
                "text_sha256": _evidence._sha(text),
            }

    # Old checkpoints may contain the pre-fix -1/empty receipt. The immutable raw prompt
    # is still authoritative, so use it as the broad source anchor instead of crashing or
    # fabricating a quote.
    if not prompt:
        raise ValueError("PLANNING_HANDOFF_SOURCE: request prompt is empty")
    return {
        "source_id": "requested_prompt",
        "char_start": 0,
        "char_end": len(prompt),
        "text": prompt,
        "text_sha256": _evidence._sha(prompt),
    }


def _source_span(
    prompt: str, requirement: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    prompt_refs = (
        [_text(ref) for ref in requirement.get("prompt_refs", []) if _text(ref)]
        if isinstance(requirement.get("prompt_refs"), list)
        else []
    )
    for ref in prompt_refs:
        source = _prompt_source_by_ref(state, ref)
        if source is not None:
            return _validated_source_span(prompt, source)

    # Evidence-derived reference requirements can legitimately have no prompt_ref. They
    # still belong to this request, so anchor their legacy interchange record to the goal
    # receipt (or, for old checkpoints, to the entire immutable prompt).
    goal = state.get("goal")
    source = goal.get("source") if isinstance(goal, Mapping) else None
    return _validated_source_span(
        prompt, source if isinstance(source, Mapping) else None
    )


def _implementation_queries(state: Mapping[str, Any], requirement_ref: str) -> list[str]:
    values: list[str] = []
    for item in (
        state.get("research_queue", [])
        if isinstance(state.get("research_queue"), list)
        else []
    ):
        if (
            not isinstance(item, Mapping)
            or str(item.get("requirement_ref") or "") != requirement_ref
        ):
            continue
        for query in (
            item.get("queries", []) if isinstance(item.get("queries"), list) else []
        ):
            text = _text(query)
            if text and text not in values:
                values.append(text)
    return values


def _semantic_capability(requirement: Mapping[str, Any]) -> str:
    capability = _text(requirement.get("semantic_capability")).casefold()
    if not capability:
        raise ValueError(
            "PLANNING_HANDOFF_CAPABILITY: requirement has no valid canonical semantic "
            f"capability: {capability or '<empty>'}"
        )
    return capability


def build_request_catalog_from_planning_state(
    prompt: str, state: Mapping[str, Any]
) -> dict[str, Any]:
    validate_planning_state(state, prompt=prompt)
    if state.get("plan_ready") is not True:
        raise ValueError(
            "PLANNING_HANDOFF_READY: request catalog requires a plan-ready state"
        )

    requirements = _requirements(state)
    details = _details(state)
    if not requirements or len(details) != len(requirements):
        raise ValueError(
            "PLANNING_HANDOFF_COVERAGE: every requirement needs one detailed plan"
        )

    sufficient_refs = _sufficient_refs(state)
    output: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "")
        detail = details.get(requirement_id)
        if detail is None:
            raise ValueError(
                f"PLANNING_HANDOFF_DETAIL: missing detail for {requirement_id}"
            )
        statement = _text(requirement.get("statement"))
        projection = project_detailed_plan_for_request_catalog(
            detail, sufficient_refs
        )
        implementation_capabilities = projection["implementation_capabilities"]
        implementation_obligations = projection["implementation_obligations"]
        acceptance = list(
            canonical_public_acceptance(
                list(requirement.get("acceptance", []))
                + list(projection["verification_checks"]),
                reject_invalid=True,
            )
        )
        capability = _semantic_capability(requirement)
        prompt_refs = (
            [_text(ref) for ref in requirement.get("prompt_refs", []) if _text(ref)]
            if isinstance(requirement.get("prompt_refs"), list)
            else []
        )
        output.append(
            {
                "requirement_id": requirement_id,
                "capability": capability,
                "statement": statement,
                "semantic_statement": statement,
                "mandatory": True,
                "provenance_role": (
                    "authored" if prompt_refs else "grounded_reference_derivation"
                ),
                "source_span": _source_span(prompt, requirement, state),
                "evidence_refs": list(requirement.get("evidence_refs") or []),
                "derived_from": list(requirement.get("evidence_refs") or []),
                "depends_on": [],
                "provides": [_evidence._canonical_capability(capability)],
                "gameplay_capabilities": [capability],
                **projection,
                "artifact_task_ids": [
                    _evidence._stable_id(
                        "task",
                        implementation,
                        {
                            "requirement_id": requirement_id,
                            "layer": "researched_implementation",
                        },
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
                "detailed_plan_ref": str(detail.get("decision_id") or ""),
            }
        )

    catalog: dict[str, Any] = {
        "prompt_sha256": _evidence._sha(prompt),
        "prompt_char_length": len(prompt),
        "purpose": _text(
            state.get("goal", {}).get("statement")
            if isinstance(state.get("goal"), Mapping)
            else prompt
        ),
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
            "capability_id_owner": "planning_state_requirement",
            "dependency_owner": "host_downstream_plan",
            "implementation_architecture_owner": "grounded_detailed_plan",
            "research_query_owner": "bounded_query_compiler",
        },
        "planning_state_sha256": state.get("state_sha256"),
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = _evidence._hash_without(
        catalog, "catalog_sha256"
    )
    _evidence._validate_request_catalog(catalog, prompt=prompt)
    return catalog


__all__ = ["build_request_catalog_from_planning_state"]
