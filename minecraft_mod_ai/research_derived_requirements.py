from __future__ import annotations

"""Deterministic evidence-backed implementation closure.

External research can strengthen provenance, but no language-model decision is required
to finish the implementation plan. Missing host-required facets are converted directly
into owned implementation obligations so research absence or malformed model output can
never destroy the plan.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .evidence_first_planning import validate_evidence_first_plan
from .research_requirement_evidence import evidence_catalog, facet_relevant_refs
from .research_requirement_plan_slice import (
    facet_owner,
    host_facet_baseline,
    requirement_task_slice,
)
from .research_requirement_schema import FACETS
from .research_requirement_template import (
    FACET_TEMPLATE_GUIDANCE,
    build_host_planning_context,
)

SCHEMA = "mmm/research-derived-requirements-v4"


class ResearchRequirementError(ValueError):
    """Host-plan corruption diagnostic; never represents a model-generation failure."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


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


def _retained_components(plan: Mapping[str, Any], parent: str) -> tuple[str, ...]:
    for decision in plan.get("reuse_decisions", ()):
        if (
            isinstance(decision, Mapping)
            and decision.get("requirement_ref") == parent
            and decision.get("action") == "retain"
        ):
            return _strings(decision.get("component_refs"))
    return ()


def _baseline_for_requirement(
    plan: Mapping[str, Any],
    requirement: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline = host_facet_baseline(requirement, tasks)
    parent = str(requirement.get("requirement_id") or "")
    retained = _retained_components(plan, parent)
    if retained:
        for item in baseline.values():
            if item["disposition"] == "missing":
                item["disposition"] = "already_covered"
                item["statement"] = f"Verified retained components own this facet for {parent}."
                item["rationale"] = "Frozen retain receipt: " + ", ".join(retained)
    return baseline


def _fallback_owner(tasks: Sequence[Mapping[str, Any]], facet: str) -> str:
    owner = facet_owner(tasks, facet)
    if owner:
        return owner
    # PlanIR already split work by implementation capability. If a required cross-cutting
    # facet has no specialized task, bind it to the final requirement-owned task rather
    # than creating another planning/model round trip.
    for task in reversed(tasks):
        task_id = str(task.get("task_id") or "").strip()
        if task_id:
            return task_id
    return ""


def _host_derived_decision(
    *,
    parent: str,
    facet: str,
    baseline: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    evidence_refs: Sequence[str],
) -> dict[str, Any]:
    disposition = str(baseline.get("disposition") or "")
    owner = _fallback_owner(tasks, facet)
    guidance = FACET_TEMPLATE_GUIDANCE[facet]["purpose"]
    refs = list(dict.fromkeys(str(ref) for ref in evidence_refs if str(ref)))

    if disposition == "missing":
        if not owner:
            raise ResearchRequirementError(
                f"host PlanIR has no execution task for required facet {parent}:{facet}"
            )
        obligations = [guidance]
        acceptance = [
            f"Verify {facet.replace('_', ' ')} for {parent} while exercising the authored requirement."
        ]
        rationale = (
            "Host capability/template contract requires this facet and the frozen task "
            "slice did not own it explicitly. The compiler binds it to an existing "
            "requirement task."
        )
        if refs:
            rationale += " Relevant research evidence is attached by reference."
        selected_disposition = "derived"
    else:
        obligations = list(_strings(baseline.get("implementation_obligations")))
        acceptance = list(_strings(baseline.get("acceptance")))
        rationale = str(baseline.get("rationale") or "Host baseline owns this facet.")
        selected_disposition = disposition or "not_applicable"

    statement = str(baseline.get("statement") or "").strip() or guidance
    record: dict[str, Any] = {
        "derived_requirement_id": "derived_" + _sha(
            {
                "parent": parent,
                "facet": facet,
                "statement": statement,
                "disposition": selected_disposition,
            }
        )[7:27],
        "parent_requirement_ref": parent,
        "provenance_role": "logically_derived",
        "facet": facet,
        "disposition": selected_disposition,
        "statement": statement,
        "rationale": rationale,
        "evidence_refs": refs,
        "acceptance": acceptance,
        "implementation_obligations": obligations,
    }
    if selected_disposition == "derived":
        record["owner_task_ref"] = owner
    return record


def derive_research_requirements(
    router: Any,
    *,
    prompt: str,
    evidence_plan: Mapping[str, Any],
    research_brief: Any,
    technical_evidence: Any,
    game_design: Mapping[str, Any],
) -> dict[str, Any]:
    """Close every implementation facet with host logic and evidence references only."""
    validate_evidence_first_plan(evidence_plan, prompt=prompt)
    request_catalog = evidence_plan.get("request_catalog")
    if not isinstance(request_catalog, Mapping):
        raise ResearchRequirementError("evidence plan has no request catalog")
    raw_requirements = request_catalog.get("requirements")
    requirements = [
        item for item in raw_requirements if isinstance(item, Mapping)
    ] if isinstance(raw_requirements, list) else []
    if not requirements:
        raise ResearchRequirementError("evidence plan has no authored requirements")

    try:
        planning_context = build_host_planning_context(router, game_design)
    except Exception:
        planning_context = {
            "schema_version": "mmm/host-planning-context-v1",
            "authority": "host_only",
            "target": dict(game_design.get("_platform_selection", {}).get("target") or {})
            if isinstance(game_design.get("_platform_selection"), Mapping)
            else {},
        }

    try:
        evidence = tuple(evidence_catalog(research_brief, technical_evidence, game_design))
    except Exception:
        evidence = ()

    decisions: list[dict[str, Any]] = []
    required_facets_closed = 0
    for requirement in requirements:
        parent = str(requirement.get("requirement_id") or "").strip()
        if not parent:
            raise ResearchRequirementError("authored requirement has no requirement_id")
        tasks = requirement_task_slice(evidence_plan, parent)
        baseline = _baseline_for_requirement(evidence_plan, requirement, tasks)
        try:
            relevant = facet_relevant_refs(evidence, requirement, baseline)
        except Exception:
            relevant = {}

        for facet in FACETS:
            item = _host_derived_decision(
                parent=parent,
                facet=facet,
                baseline=baseline[facet],
                tasks=tasks,
                evidence_refs=(relevant.get(facet, ()) if isinstance(relevant, Mapping) else ()),
            )
            if item["disposition"] == "derived":
                required_facets_closed += 1
            decisions.append(item)

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA,
        "prompt_sha256": request_catalog.get("prompt_sha256"),
        "host_template": {
            "authority": "host_only",
            "facet_order": list(FACETS),
            "planning_context": planning_context,
            "model_generated_planning_json": False,
        },
        "evidence_catalog": list(evidence),
        "facet_decisions": decisions,
        "augmentation_events": [],
        "degraded_augmentation_facets": [],
        "model_call_policy": {
            "unit": "none",
            "evidence_bearing_slots": 0,
            "actual_calls_including_retries": 0,
            "max_attempts_per_evidence_bearing_facet": 0,
            "research_absence_behavior": "continue_with_host_template",
            "model_may_mutate_host_fields": False,
        },
        "host_closure": {
            "required_facets_closed": required_facets_closed,
            "execution_owner_policy": "specialized_owner_else_last_requirement_task",
        },
        "ledger_sha256": "",
    }
    ledger["ledger_sha256"] = _sha(
        {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    )
    return ledger


def attach_derived_requirement_ledger(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
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
