from __future__ import annotations

"""Host-owned templates for bounded research-to-requirement augmentation.

The language model never owns target versions, requirement identity, facet identity,
task identity, baseline acceptance, or template structure.  Host code freezes all of
those values and exposes exactly one semantic decision slot at a time.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .research_requirement_schema import FACETS

TEMPLATE_SCHEMA = "mmm/research-requirement-slot-v1"
HOST_CONTEXT_SCHEMA = "mmm/host-planning-context-v1"
DECISIONS = ("no_addition", "add_obligation", "insufficient_evidence")

FACET_TEMPLATE_GUIDANCE: dict[str, dict[str, str]] = {
    "state_lifecycle": {
        "purpose": (
            "Close initialization, ownership, state transitions, update/tick behavior, "
            "terminal states, and cleanup for this requirement."
        ),
        "model_question": (
            "Does the supplied external evidence require lifecycle behavior that is not "
            "already present in the immutable host task slice?"
        ),
    },
    "interfaces_integration": {
        "purpose": (
            "Close caller/callee boundaries, hooks, events, services, menus/screens, and "
            "cross-system integration contracts owned by this requirement."
        ),
        "model_question": (
            "Does the supplied evidence require an integration contract missing from the "
            "host baseline?"
        ),
    },
    "persistence_reload": {
        "purpose": (
            "Close persisted state, serialization format, save/load/reload behavior, and "
            "restart compatibility where this requirement owns persistent data."
        ),
        "model_question": (
            "Does the supplied evidence impose persistence or reload behavior not already "
            "represented by the host tasks?"
        ),
    },
    "server_network_authority": {
        "purpose": (
            "Close authoritative side, validation boundary, payload/synchronization rules, "
            "and multiplayer consistency without granting clients server authority."
        ),
        "model_question": (
            "Does the supplied evidence require networking or authority work missing from "
            "the host baseline?"
        ),
    },
    "registration_data_resources": {
        "purpose": (
            "Close registries, identifiers, recipes, loot/tags, worldgen, data generation, "
            "models, language entries, and other required data/resource artifacts."
        ),
        "model_question": (
            "Does the supplied evidence require a registration/data/resource artifact not "
            "already owned by the host tasks?"
        ),
    },
    "failure_edge_cases": {
        "purpose": (
            "Close invalid input, missing dependency/resource, boundary-state rejection, "
            "recovery, replacement/removal, and other observable failure behavior."
        ),
        "model_question": (
            "Does the supplied evidence expose a concrete failure case that the host "
            "baseline does not yet require?"
        ),
    },
    "verification_testing": {
        "purpose": (
            "Close deterministic compile/runtime/GameTest or equivalent observable checks "
            "that prove the requirement and its failure behavior."
        ),
        "model_question": (
            "Does the supplied evidence require a verification check not already present "
            "in the host acceptance contract?"
        ),
    },
}

FACET_AUGMENTATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "rationale": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "implementation_obligations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "acceptance": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "decision",
        "rationale",
        "evidence_refs",
        "implementation_obligations",
        "acceptance",
    ],
    "additionalProperties": False,
}


def _canonical_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def build_host_planning_context(
    router: Any,
    game_design: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze notebook/session target decisions after host platform resolution."""

    selection = game_design.get("_platform_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("host planning context requires resolved _platform_selection")
    target = selection.get("target")
    if not isinstance(target, Mapping) or not target:
        raise ValueError("host planning context requires a resolved platform target receipt")

    requested_version = getattr(router, "_mmm_requested_minecraft_version", None)
    requested_loader = getattr(router, "_mmm_requested_loader", None)
    return {
        "schema_version": HOST_CONTEXT_SCHEMA,
        "authority": "host_only",
        "target": _canonical_copy(dict(target)),
        "target_selection": {
            "source": str(selection.get("source") or ""),
            "explicit_version": bool(selection.get("explicit_version")),
            "explicit_loader": bool(selection.get("explicit_loader")),
            "preserved_existing_target": bool(
                selection.get("preserved_existing_target")
            ),
            "migration_requested": bool(selection.get("migration_requested")),
        },
        "requested_constraints": {
            "minecraft_version": (
                str(requested_version) if requested_version is not None else None
            ),
            "loader": str(requested_loader) if requested_loader is not None else None,
            "minecraft_version_source": (
                "user_notebook" if requested_version is not None else "host_selector"
            ),
            "loader_source": (
                "user_notebook" if requested_loader is not None else "host_selector"
            ),
        },
    }


def build_facet_slot(
    *,
    planning_context: Mapping[str, Any],
    requirement: Mapping[str, Any],
    facet: str,
    baseline: Mapping[str, Any],
    task_slice: Sequence[Mapping[str, Any]],
    evidence_catalog: Sequence[Mapping[str, Any]],
    allowed_evidence_refs: Sequence[str],
) -> dict[str, Any]:
    """Build one immutable host template plus one small-model decision slot."""

    if facet not in FACETS:
        raise ValueError(f"unknown research facet: {facet!r}")
    guidance = FACET_TEMPLATE_GUIDANCE[facet]
    return {
        "schema_version": TEMPLATE_SCHEMA,
        "host_owned": {
            "planning_context": _canonical_copy(dict(planning_context)),
            "requirement": {
                "requirement_id": str(requirement.get("requirement_id") or ""),
                "capability": requirement.get("capability"),
                "statement": str(requirement.get("statement") or "")[:1200],
                "implementation_capabilities": [
                    str(item)
                    for item in requirement.get("implementation_capabilities", ())
                    if str(item).strip()
                ],
                "artifact_obligations": [
                    str(item)
                    for item in requirement.get("artifact_obligations", ())
                    if str(item).strip()
                ],
                "acceptance": [
                    str(item)
                    for item in requirement.get("acceptance", ())
                    if str(item).strip()
                ],
            },
            "facet": facet,
            "facet_purpose": guidance["purpose"],
            "host_baseline": _canonical_copy(dict(baseline)),
            "host_task_slice": _canonical_copy(list(task_slice)),
            "allowed_evidence_refs": list(dict.fromkeys(allowed_evidence_refs)),
            "evidence_catalog": _canonical_copy(list(evidence_catalog)),
        },
        "model_slot": {
            "question": guidance["model_question"],
            "allowed_decisions": list(DECISIONS),
            "rules": [
                "no_addition: host baseline is sufficient; obligations and acceptance must be empty.",
                "add_obligation: cite supplied allowed evidence and return only concrete missing obligations plus observable acceptance checks.",
                "insufficient_evidence: supplied evidence exposes a real missing obligation but is insufficient to specify it safely; obligations and acceptance must be empty.",
                "Never repeat or alter host-owned facet, versions, loader, mappings, IDs, paths, task identity, or baseline fields.",
                "Never invent an evidence ref, API, repository, version, or implementation detail not supported by the supplied evidence.",
            ],
            "output_fields_only": [
                "decision",
                "rationale",
                "evidence_refs",
                "implementation_obligations",
                "acceptance",
            ],
        },
    }


__all__ = [
    "DECISIONS",
    "FACET_AUGMENTATION_RESPONSE_SCHEMA",
    "FACET_TEMPLATE_GUIDANCE",
    "HOST_CONTEXT_SCHEMA",
    "TEMPLATE_SCHEMA",
    "build_facet_slot",
    "build_host_planning_context",
]
