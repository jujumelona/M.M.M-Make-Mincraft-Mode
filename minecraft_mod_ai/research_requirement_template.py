from __future__ import annotations

"""Host-owned templates for bounded research-to-requirement augmentation.

The language model never owns target versions, requirement identity, facet identity,
task identity, baseline acceptance, or template structure. Host code freezes all of
those values and exposes exactly one semantic decision slot at a time. The slot is
intentionally verbose about what must be checked so a small model can make a narrow,
well-grounded decision without having to invent its own review procedure.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .research_requirement_schema import FACETS

TEMPLATE_SCHEMA = "mmm/research-requirement-slot-v1"
HOST_CONTEXT_SCHEMA = "mmm/host-planning-context-v1"
DECISIONS = ("no_addition", "add_obligation", "insufficient_evidence")

FACET_TEMPLATE_GUIDANCE: dict[str, dict[str, Any]] = {
    "state_lifecycle": {
        "purpose": (
            "Close initialization, ownership, state variables, legal transitions, update/tick behavior, "
            "terminal states, reset/removal/unload behavior and cleanup for this requirement."
        ),
        "model_question": (
            "After comparing the immutable host baseline against the supplied evidence, is any concrete "
            "state ownership, initialization, transition, mutation, termination or cleanup obligation missing?"
        ),
        "checks": (
            "state owner and scope",
            "default/initial state",
            "legal transitions and guards",
            "tick/update mutation",
            "terminal/reset/removal state",
            "cleanup and interruption behavior",
            "invariants and invalid transitions",
        ),
    },
    "interfaces_integration": {
        "purpose": (
            "Close caller/callee boundaries, hooks, events, services, registries, menus/screens, initialization "
            "order and cross-system integration contracts owned by this requirement."
        ),
        "model_question": (
            "Does the evidence prove a required integration boundary, hook, service call, registration ordering "
            "or side placement that the host baseline does not yet require?"
        ),
        "checks": (
            "entry hook/event/service",
            "caller/callee responsibilities",
            "initialization/registration order",
            "common/server/client placement",
            "cross-module data flow",
            "public API versus mixin/access requirement",
            "compatibility/extension boundary",
        ),
    },
    "persistence_reload": {
        "purpose": (
            "Close persisted state ownership, serialization shape, defaults, dirty/save triggers, load/reload, "
            "migration and malformed/stale data behavior where this requirement owns persistent data."
        ),
        "model_question": (
            "Does the evidence require persistence, reload, migration or malformed-data handling not already "
            "represented by the immutable host task slice?"
        ),
        "checks": (
            "persisted fields and owner/scope",
            "codec/serialization contract",
            "absent-data defaults",
            "save/dirty trigger",
            "load/restart behavior",
            "schema migration",
            "malformed/stale data handling",
            "death/copy/dimension transfer when relevant",
        ),
    },
    "server_network_authority": {
        "purpose": (
            "Close authoritative logical side, client/server trust boundary, payload direction, validation, "
            "permissions/rate limits, synchronization, join/reconnect and multiplayer consistency."
        ),
        "model_question": (
            "Does the evidence require an authority, validation, payload, synchronization or reconnect obligation "
            "missing from the host baseline, or prove that a networking branch is unnecessary?"
        ),
        "checks": (
            "authoritative side",
            "client-originated input validation",
            "payload direction and purpose",
            "permission/ownership/distance/rate validation",
            "sync recipients and trigger",
            "join/reconnect/resync",
            "stale/replayed/malformed payload",
            "client prediction versus server truth",
        ),
    },
    "registration_data_resources": {
        "purpose": (
            "Close registries, stable identifiers, recipes, loot, tags, data generation, models, blockstates, "
            "language, worldgen, assets and other required data/resource artifacts."
        ),
        "model_question": (
            "Does the evidence prove a concrete registration, data, generated resource, client asset or worldgen "
            "artifact missing from the host baseline?"
        ),
        "checks": (
            "registry identity",
            "recipe/loot/tag/data requirements",
            "models/blockstates/language",
            "worldgen configured/placed/biome binding",
            "generated versus authored ownership",
            "resource path/identifier constraints",
            "missing-resource fallback",
        ),
    },
    "failure_edge_cases": {
        "purpose": (
            "Close invalid input, missing dependency/resource, boundary values, duplicate invocation, partial "
            "failure, interruption, rollback/cleanup, concurrency and bounded-resource behavior."
        ),
        "model_question": (
            "Does the evidence expose a specific observable failure, limit or recovery case that the host baseline "
            "does not yet require?"
        ),
        "checks": (
            "invalid input rejection",
            "missing dependency/resource",
            "boundary values",
            "duplicate/repeated request",
            "partial mutation rollback",
            "unload/disconnect/removal interruption",
            "concurrency/reentrancy",
            "rate/size/count/tick-time bound",
        ),
    },
    "verification_testing": {
        "purpose": (
            "Close deterministic static/compile/runtime/GameTest or equivalent observable checks that prove the "
            "user-visible requirement, rejection behavior, state invariants and applicable reload/network/resource branches."
        ),
        "model_question": (
            "Does the evidence require an observable success, rejection, boundary, reload, multiplayer or resource "
            "verification check not already present in the immutable host acceptance contract?"
        ),
        "checks": (
            "success Given/When/Then",
            "rejection/failure Given/When/Then",
            "boundary values",
            "state invariant/transition",
            "reload/restart when applicable",
            "multiplayer authority/resync when applicable",
            "resource/data load when applicable",
            "target compile/static prerequisite",
        ),
    },
}

FACET_AUGMENTATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "One bounded evidence-backed decision for one immutable requirement facet.",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "rationale": {
            "type": "string",
            "minLength": 12,
            "description": (
                "Concise comparison of evidence against the immutable baseline. Name the exact missing concern or "
                "explain why the baseline already closes it; do not use generic quality language."
            ),
        },
        "evidence_refs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
            "description": "Only host-supplied allowed evidence refs directly supporting this facet decision.",
        },
        "implementation_obligations": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "Concrete missing obligation with owner/action/condition/result; no vague 'handle properly' wording.",
            },
        },
        "acceptance": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "Observable check with condition/action/expected result proving the added obligation.",
            },
        },
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


def _string_items(value: Any) -> list[str]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return []
    return [str(item) for item in values if str(item).strip()]


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

    if target.get("host_facts_json"):
        from .target_contract import target_contract_from_mapping
        from .resolved_version_context import ResolvedVersionContext

        resolved = target_contract_from_mapping(target).version_context
        supplied = ResolvedVersionContext.from_dict(selection.get("resolved_version_context", {}))
        resolved.assert_context(supplied.context_id)
        return {
            "schema_version": HOST_CONTEXT_SCHEMA, "authority": "host_only",
            "target": resolved.to_dict()["target"],
            "resolved_version_context": resolved.to_dict(),
            "context_id": resolved.context_id,
        }

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
    """Build one immutable host template plus one detailed small-model decision slot."""

    if facet not in FACETS:
        raise ValueError(f"unknown research facet: {facet!r}")
    guidance = FACET_TEMPLATE_GUIDANCE[facet]
    checks = [str(item) for item in guidance["checks"]]
    return {
        "schema_version": TEMPLATE_SCHEMA,
        "host_owned": {
            "planning_context": _canonical_copy(dict(planning_context)),
            "requirement": {
                "requirement_id": str(requirement.get("requirement_id") or ""),
                "capability": requirement.get("capability"),
                "statement": str(requirement.get("statement") or "")[:1200],
                "implementation_capabilities": _string_items(
                    requirement.get("implementation_capabilities")
                ),
                "artifact_obligations": _string_items(
                    requirement.get("artifact_obligations")
                ),
                "acceptance": _string_items(requirement.get("acceptance")),
            },
            "facet": facet,
            "facet_purpose": guidance["purpose"],
            "host_baseline": _canonical_copy(dict(baseline)),
            "host_task_slice": _canonical_copy(list(task_slice)),
            "allowed_evidence_refs": list(dict.fromkeys(allowed_evidence_refs)),
            "evidence_catalog": _canonical_copy(list(evidence_catalog)),
        },
        "model_slot": {
            "question": (
                str(guidance["model_question"])
                + " Before deciding, explicitly inspect: "
                + "; ".join(checks)
                + "."
            ),
            "allowed_decisions": list(DECISIONS),
            "rules": [
                "Read the immutable requirement, host baseline, host task slice and all supplied evidence before deciding; do not answer from the facet name alone.",
                "Compare each facet checklist concern against the baseline. Add only a gap that is concrete, requirement-local and directly supported by supplied evidence.",
                "no_addition: the host baseline already closes every evidence-backed concern in this facet; implementation_obligations and acceptance must be empty.",
                "add_obligation: cite supplied allowed evidence and return only concrete missing obligations plus observable acceptance checks; each obligation must state owner/action/condition/result where applicable.",
                "insufficient_evidence: evidence exposes a real missing concern but cannot specify it safely; obligations and acceptance must be empty and rationale must say exactly what evidence is missing.",
                "Never repeat, rewrite or alter host-owned facet, target versions, loader, mappings, IDs, paths, task identity, dependency edges, baseline obligations or baseline acceptance.",
                "Never invent an evidence ref, API, repository, dependency, version, symbol, identifier, path, constant, behavior or implementation mechanism not supported by the supplied evidence.",
                "A source example is evidence about a pattern, not proof that the exact target binding is valid. Keep target-specific unknowns explicit.",
                "Do not convert preferences, optional polish or adjacent features into mandatory obligations unless the evidence and current requirement require them.",
                "Acceptance must be observable and falsifiable. Compile success alone is not acceptance for a user-visible behavior.",
                "Use concrete terms and conditions; avoid vague words such as robust, proper, appropriate, seamless, comprehensive or handle correctly unless followed by measurable behavior.",
                "Before submission, verify decision, rationale, evidence_refs, obligations and acceptance are mutually consistent and contain no unsupported claim.",
                "Facet checklist for this decision: " + "; ".join(checks) + ".",
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
