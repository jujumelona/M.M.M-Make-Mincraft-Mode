from __future__ import annotations

"""Make design-decision provenance explicit and prohibit heuristic selection claims.

Task architecture may legitimately derive implementation branches from a requirement, but
that does not mean the user authored that architecture or that the planner compared and
selected one design alternative over others.  This contract records branch decisions as
derived architecture and leaves ``selected_design_alternatives`` empty unless a future
explicit comparative-evidence contract supplies candidates and a selection receipt.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from . import evidence_first_planning as _planning
from . import task_artifact_contract as _task_contract

_INSTALLED = False


def _branch_evidence(
    plan: Mapping[str, Any], predicate: str, requirement_ref: str
) -> list[str]:
    branches = plan.get("branch_predicates")
    if not isinstance(branches, Mapping):
        return []
    branch = branches.get(predicate)
    if not isinstance(branch, Mapping):
        return []
    by_requirement = branch.get("requirement_evidence_refs")
    if isinstance(by_requirement, Mapping):
        return list(_planning._strings(by_requirement.get(requirement_ref)))
    return []


def _explicit_selected_alternatives(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept selection claims only from an explicit comparative-evidence receipt."""

    raw = plan.get("design_alternative_evaluations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    selected: list[dict[str, Any]] = []
    for evaluation in raw:
        if not isinstance(evaluation, Mapping):
            continue
        candidates = evaluation.get("candidates")
        selected_id = str(evaluation.get("selected_candidate_id") or "").strip()
        evidence_refs = list(_planning._strings(evaluation.get("evidence_refs")))
        requirement_refs = list(_planning._strings(evaluation.get("requirement_refs")))
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes, bytearray))
            or len(candidates) < 2
            or not selected_id
            or not evidence_refs
            or not requirement_refs
        ):
            continue
        candidate_ids = {
            str(item.get("candidate_id") or "").strip()
            for item in candidates
            if isinstance(item, Mapping) and str(item.get("candidate_id") or "").strip()
        }
        if selected_id not in candidate_ids:
            continue
        selected.append(
            {
                "decision_id": _planning._stable_id(
                    "design", selected_id, {"requirements": requirement_refs, "evidence": evidence_refs}
                ),
                "provenance_role": "selected_design_alternative",
                "authority": "comparative_evidence",
                "requirement_refs": requirement_refs,
                "selection": selected_id,
                "candidate_ids": sorted(candidate_ids),
                "evidence_refs": evidence_refs,
                "selection_receipt_sha256": str(
                    evaluation.get("evaluation_sha256")
                    or _planning._sha(dict(evaluation))
                ),
            }
        )
    return selected


def _design_resolution(plan: Mapping[str, Any]) -> dict[str, Any]:
    architecture: list[dict[str, Any]] = []
    obligations: list[dict[str, Any]] = []
    seen_architecture: set[tuple[str, str]] = set()
    seen_obligations: set[str] = set()

    tasks = plan.get("tasks")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            task_id = str(task.get("task_id") or "").strip()
            requirement_refs = list(_planning._strings(task.get("requirement_refs")))
            predicates = list(_planning._strings(task.get("conditional_predicates")))
            for requirement_ref in requirement_refs:
                for predicate in predicates:
                    key = (requirement_ref, predicate)
                    if key in seen_architecture:
                        continue
                    seen_architecture.add(key)
                    evidence_refs = _branch_evidence(plan, predicate, requirement_ref)
                    architecture.append(
                        {
                            "decision_id": _planning._stable_id(
                                "architecture",
                                predicate,
                                {"requirement": requirement_ref},
                            ),
                            "provenance_role": "derived_architecture_decision",
                            "authority": "implementation_only",
                            "requirement_refs": [requirement_ref],
                            "predicate": predicate,
                            "evidence_refs": evidence_refs,
                            "source_task_refs": [task_id] if task_id else [],
                            "status": (
                                "EVIDENCE_BOUND"
                                if evidence_refs
                                else "UNRESOLVED_PROVENANCE"
                            ),
                            "semantic_effect": "none",
                        }
                    )

            artifacts = task.get("artifact_obligations")
            if not isinstance(artifacts, Sequence) or isinstance(
                artifacts, (str, bytes, bytearray)
            ):
                continue
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                artifact_id = str(artifact.get("artifact_id") or "").strip()
                if not artifact_id or artifact_id in seen_obligations:
                    continue
                seen_obligations.add(artifact_id)
                obligations.append(
                    {
                        "obligation_id": artifact_id,
                        "provenance_role": "implementation_obligation",
                        "authority": "compiled_task_architecture",
                        "requirement_refs": list(
                            _planning._strings(
                                artifact.get("requirement_refs") or requirement_refs
                            )
                        ),
                        "task_ref": str(artifact.get("task_ref") or task_id),
                        "kind": artifact.get("kind"),
                        "locator": artifact.get("locator"),
                        "status": artifact.get("status"),
                        "reason": "compiled task architecture requires this artifact; it is not an authored gameplay mandate",
                    }
                )

    selected = _explicit_selected_alternatives(plan)
    unresolved = [
        {
            "decision_id": item["decision_id"],
            "requirement_refs": list(item["requirement_refs"]),
            "predicate": item["predicate"],
            "reason": "derived architecture lacks exact requirement-scoped evidence",
        }
        for item in architecture
        if item["status"] == "UNRESOLVED_PROVENANCE"
    ]
    return {
        "schema_version": "mmm/design-resolution-v2",
        "selected_design_alternatives": selected,
        "derived_architecture_decisions": architecture,
        "implementation_obligations": obligations,
        "unresolved_design_provenance": unresolved,
        "selection_policy": {
            "minimum_candidates_for_selection": 2,
            "comparative_evidence_required": True,
            "requirement_binding_required": True,
            "heuristic_single_candidate_selection_allowed": False,
        },
        "policy": (
            "Authored requirements remain the sole gameplay authority. Conditional branches and "
            "artifact obligations are implementation-only derivations. A selected design alternative "
            "requires an explicit candidate set, requirement bindings, and comparative evidence."
        ),
    }


def install_design_resolution_provenance_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _task_contract._design_resolution = _design_resolution
    _INSTALLED = True


__all__ = [
    "_design_resolution",
    "_explicit_selected_alternatives",
    "install_design_resolution_provenance_contract",
]
