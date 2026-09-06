from __future__ import annotations

"""One engineering worksheet shared by planning and coding.

The worksheet deliberately keeps a compact, stable wire shape for small models while
making the meaning of every slot explicit. Host code owns the canonical section list and
may narrow it only through an explicit trusted selection. With no selection, all ten
sections are required. The model never decides which sections apply.

A worksheet specification is an authored design contract, not a retrieved fact. Evidence
references are therefore optional constraints on that design. Target/API/version/source
facts are represented separately by the detailed-plan grounded-binding contract.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

DETAIL_SLOT_GUIDANCE: dict[str, tuple[str, ...]] = {
    "behavior_contract": (
        "actors and authoritative owner",
        "trigger or entry condition",
        "preconditions and eligibility rules",
        "all inputs with type, unit, range, default and source",
        "all outputs and externally visible side effects",
        "success postconditions",
        "rejection/no-op postconditions",
        "ordering, timing, cooldown or frequency semantics when relevant",
        "explicit boundaries and non-goals",
    ),
    "state_model": (
        "every owned state variable and owning component",
        "type, unit, default and valid domain for each state value",
        "legal state transitions with trigger and guard",
        "invariants that must hold before and after each transition",
        "initialization and construction behavior",
        "tick/update/lifecycle mutation rules",
        "reset, death, removal, unload and cleanup behavior",
        "concurrency/reentrancy assumptions when state may be touched from multiple paths",
    ),
    "algorithm": (
        "ordered operations from entry to observable result",
        "branch predicates and the action for every branch",
        "formulae, thresholds, units and rounding/clamping rules",
        "iteration order and termination condition",
        "determinism/randomness source and seed ownership",
        "boundary values and empty/null/missing cases",
        "atomicity requirements for multi-step mutations",
        "algorithmic cost or bounded-work expectation",
    ),
    "integration": (
        "entry hook/event/callback/service boundary",
        "caller and callee responsibilities",
        "verified target APIs/symbols distinguished from design requirements",
        "registry or initialization order dependencies",
        "cross-module inputs/outputs and dependency direction",
        "side-only versus common/server-safe placement",
        "compatibility assumptions and extension points",
        "fallback when a desired public hook is unavailable",
    ),
    "authority_and_network": (
        "authoritative logical side for every mutable gameplay decision",
        "client prediction/presentation boundary",
        "packet direction and exact purpose when networking applies",
        "payload fields, validation and trust boundary",
        "permission, ownership, distance/rate-limit and replay checks when applicable",
        "state synchronization recipients and trigger",
        "join/reconnect/resync behavior",
        "disconnect, stale packet and malformed payload behavior",
        "explicit reason when networking is inapplicable",
    ),
    "persistence": (
        "which state persists and its storage owner/scope",
        "serialization keys/shape or the requirement to bind them after target research",
        "defaults for absent data",
        "save/dirty/update trigger",
        "load and restart behavior",
        "schema/version migration policy",
        "malformed, stale or partially missing data behavior",
        "copy/clone/death/dimension-transfer semantics when relevant",
        "explicit reason when persistence is inapplicable",
    ),
    "resources_and_ui": (
        "all required registries, identifiers and data resources",
        "recipes, loot, tags, models, blockstates, language and worldgen data as applicable",
        "client assets and generated-versus-authored ownership",
        "menu/screen/container interaction contract when applicable",
        "server-authoritative data exposed to UI",
        "resource paths or path-binding requirements without inventing unsupported names",
        "missing-resource fallback and validation",
        "accessibility/localization or tooltip feedback required by observable behavior",
        "explicit reason for inapplicable resource/UI branches",
    ),
    "failure_and_limits": (
        "invalid user/input states and rejection result",
        "missing dependency, registry entry, resource or target binding behavior",
        "duplicate/repeated invocation behavior",
        "partial-failure rollback or cleanup",
        "unload/removal/disconnect/restart interruption behavior",
        "concurrency/reentrancy hazards",
        "rate, size, count, tick-time or memory bounds",
        "logging/diagnostic signal required for non-user-visible failures",
        "fail-closed conditions where inventing a fallback would change semantics",
    ),
    "reuse_assessment": (
        "source/evidence locator and exact relevant implementation pattern",
        "what can transfer unchanged",
        "what must be adapted for the selected target and authored semantics",
        "API/version/loader/mappings compatibility",
        "dependency and transitive-dependency impact",
        "license or provenance constraint when supplied by evidence",
        "ownership/path collision risk with the current project",
        "missing evidence that prevents direct reuse",
        "verdict: reuse, adapt, reference-only or new implementation required",
    ),
    "verification": (
        "at least one success Given/When/Then observation",
        "at least one rejection or failure observation",
        "boundary-value checks for authored limits",
        "state transition/invariant checks",
        "persistence reload/restart check when applicable",
        "multiplayer/authority/resync check when applicable",
        "resource/data loading check when applicable",
        "target compile/static gate required before runtime claims",
        "observable expected result and how the host can measure it",
        "explicit mapping from important behavior invariants to checks",
    ),
}

DETAIL_FIELDS = {
    "behavior_contract": "Freeze the complete externally observable behavior contract.",
    "state_model": "Freeze owned state, lifecycle, transitions and invariants.",
    "algorithm": "Freeze the deterministic ordered implementation logic and edge handling.",
    "integration": "Freeze platform/module integration boundaries without inventing target APIs.",
    "authority_and_network": "Freeze logical-side authority, validation and synchronization semantics.",
    "persistence": "Freeze persistence ownership, codec lifecycle, migration and malformed-data behavior.",
    "resources_and_ui": "Freeze required registries, data/resources, assets and UI interaction contracts.",
    "failure_and_limits": "Freeze rejection, recovery, cleanup, concurrency and resource-limit behavior.",
    "reuse_assessment": "Freeze evidence-backed reuse/adaptation boundaries and compatibility constraints.",
    "verification": "Freeze executable/observable proof obligations for success, failure and boundaries.",
}

WORKSHEET_SECTIONS: tuple[str, ...] = tuple(DETAIL_FIELDS)
CORE_WORKSHEET_SECTIONS: tuple[str, ...] = (
    "behavior_contract",
    "state_model",
    "algorithm",
    "integration",
    "failure_and_limits",
    "reuse_assessment",
    "verification",
)
CONDITIONAL_WORKSHEET_SECTIONS: tuple[str, ...] = (
    "authority_and_network",
    "persistence",
    "resources_and_ui",
)

WORKSHEET_INSTRUCTIONS: tuple[str, ...] = (
    "Work on exactly one user-visible requirement; do not redesign neighboring requirements.",
    "Read all supplied evidence before filling any section. Use constraint_evidence_refs only when retrieved evidence actually constrains the authored design; an empty list is valid.",
    "Fill all ten sections. Never use a bare N/A, none, TODO, TBD, unknown, same-as-above, or generic placeholder.",
    "Write a distinct section-specific specification for every section; copying one generic answer across multiple sections is invalid.",
    "For an inapplicable concern, state the concrete design reason it is inapplicable. Cite evidence only when that conclusion depends on an external fact.",
    "Separate retrieved facts from design decisions. Proposed identifiers, algorithms, paths, constants or behavior rules are authored design, not evidence-backed facts.",
    "Use exact actors, state owners, triggers, inputs, outputs, branches, units, limits and observable postconditions instead of adjectives such as robust, proper, appropriate or handle correctly.",
    "Do not silently widen scope. Every claimed behavior must belong to the current requirement or be a necessary dependency established by the planning state.",
    "Treat compile/static checks as necessary but insufficient: verification must also prove the user-visible runtime behavior and relevant failure paths.",
    "Before submission, cross-check that state, algorithm, integration, persistence/network branches and verification describe one internally consistent design.",
)

_MIN_SPECIFICATION_CHARS = 24


def normalize_required_sections(
    required_sections: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return a validated host-owned section selection in canonical order.

    ``None`` is deliberately fail-safe and means every canonical section. An explicit
    selection may omit only host-proven inapplicable conditional branches. Core sections
    are mandatory for every detailed plan. Model output must never be used as
    ``required_sections``.
    """

    if required_sections is None:
        return WORKSHEET_SECTIONS
    if isinstance(required_sections, (str, bytes)):
        raise ValueError("DETAILED_PLAN_SECTIONS: selection must be an iterable of section names")

    requested = list(required_sections)
    if not requested:
        raise ValueError("DETAILED_PLAN_SECTIONS: explicit selection cannot be empty")
    if any(not isinstance(key, str) or not key for key in requested):
        raise ValueError("DETAILED_PLAN_SECTIONS: every selected section must be a non-empty string")
    if len(set(requested)) != len(requested):
        raise ValueError("DETAILED_PLAN_SECTIONS: duplicate section selection")

    unknown = set(requested) - set(WORKSHEET_SECTIONS)
    if unknown:
        raise ValueError(
            "DETAILED_PLAN_SECTIONS: unknown section(s): " + ", ".join(sorted(unknown))
        )
    missing_core = set(CORE_WORKSHEET_SECTIONS) - set(requested)
    if missing_core:
        raise ValueError(
            "DETAILED_PLAN_SECTIONS: core section(s) cannot be omitted: "
            + ", ".join(key for key in CORE_WORKSHEET_SECTIONS if key in missing_core)
        )
    requested_set = set(requested)
    return tuple(key for key in WORKSHEET_SECTIONS if key in requested_set)


def _section_description(key: str) -> str:
    checklist = "; ".join(DETAIL_SLOT_GUIDANCE[key])
    return f"{DETAIL_FIELDS[key]} Explicitly cover: {checklist}."


def _instructions_for_sections(selected: tuple[str, ...]) -> tuple[str, ...]:
    if selected == WORKSHEET_SECTIONS:
        return WORKSHEET_INSTRUCTIONS

    instructions = list(WORKSHEET_INSTRUCTIONS)
    instructions[2] = (
        f"Fill exactly the {len(selected)} host-required sections shown below and do not add omitted sections. "
        "Never use a bare N/A, none, TODO, TBD, unknown, same-as-above, or generic placeholder."
    )
    return tuple(instructions)


def worksheet_prompt(required_sections: Iterable[str] | None = None) -> str:
    """Return canonical instructions for only the host-required worksheet sections."""

    selected = normalize_required_sections(required_sections)
    rows = ["ENGINEERING WORKSHEET — mandatory completion protocol:"]
    rows.extend(
        f"{index}. {rule}"
        for index, rule in enumerate(_instructions_for_sections(selected), start=1)
    )
    rows.append("Host-required section checklists:")
    for key in selected:
        rows.append(f"- {key}: " + "; ".join(DETAIL_SLOT_GUIDANCE[key]))
    return "\n".join(rows)


def _worksheet_section_schema(key: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": _section_description(key),
        "properties": {
            "specification": {
                "type": "string",
                "minLength": _MIN_SPECIFICATION_CHARS,
                "description": (
                    _section_description(key)
                    + " Write a self-contained, section-specific authored design contract. Use explicit 'inapplicable because ...' reasoning when needed; never emit a bare placeholder or reuse another section's answer."
                ),
            },
            "constraint_evidence_refs": {
                "type": "array",
                "uniqueItems": True,
                "description": (
                    "Evidence references supplied by the host that constrain this authored design section. "
                    "Use an empty array when the section is a design decision rather than an external fact."
                ),
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["specification", "constraint_evidence_refs"],
        "additionalProperties": False,
    }


def worksheet_schema(required_sections: Iterable[str] | None = None) -> dict[str, Any]:
    """Build the response schema for an explicit host-owned section selection."""

    selected = normalize_required_sections(required_sections)
    return {
        "type": "object",
        "description": (
            "Complete engineering worksheet for exactly the host-required sections. "
            "Specifications are authored design contracts; constraint_evidence_refs only record "
            "external evidence that actually constrains those designs. Section answers must be distinct."
        ),
        "properties": {key: _worksheet_section_schema(key) for key in selected},
        "required": list(selected),
        "additionalProperties": False,
    }


WORKSHEET_SCHEMA = worksheet_schema()

_PLACEHOLDERS = {
    "n/a",
    "na",
    "none",
    "not applicable",
    "not-applicable",
    "todo",
    "tbd",
    "unknown",
    "same as above",
    "same-as-above",
}


def validate_worksheet(
    value: Any,
    allowed_refs: set[str],
    required_sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected = normalize_required_sections(required_sections)
    if not isinstance(value, Mapping) or set(value) != set(selected):
        raise ValueError(
            "DETAILED_PLAN_WORKSHEET: exactly the host-required engineering sections must be filled"
        )

    seen_specifications: dict[str, str] = {}
    normalized: dict[str, Any] = {}
    for key in selected:
        row = value[key]
        if not isinstance(row, Mapping):
            raise ValueError(f"DETAILED_PLAN_WORKSHEET: {key} is not an object")
        specification = " ".join(str(row.get("specification") or "").split()).strip()
        normalized_specification = specification.casefold()
        if (
            len(specification) < _MIN_SPECIFICATION_CHARS
            or normalized_specification in _PLACEHOLDERS
        ):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: {key} has no concrete specification"
            )
        duplicate_of = seen_specifications.get(normalized_specification)
        if duplicate_of is not None:
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: {key} duplicates {duplicate_of}; every section requires a section-specific specification"
            )
        seen_specifications[normalized_specification] = key

        refs = row.get("constraint_evidence_refs")
        if not isinstance(refs, list):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: {key} constraint_evidence_refs must be an array"
            )
        ref_values = [str(ref).strip() for ref in refs if str(ref).strip()]
        if (
            len(ref_values) != len(refs)
            or len(set(ref_values)) != len(ref_values)
            or any(ref not in allowed_refs for ref in ref_values)
        ):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: {key} has invalid constraint evidence"
            )
        normalized[key] = {
            "specification": specification,
            "constraint_evidence_refs": ref_values,
        }
    return deepcopy(normalized)


__all__ = [
    "CONDITIONAL_WORKSHEET_SECTIONS",
    "CORE_WORKSHEET_SECTIONS",
    "DETAIL_FIELDS",
    "DETAIL_SLOT_GUIDANCE",
    "WORKSHEET_INSTRUCTIONS",
    "WORKSHEET_SCHEMA",
    "WORKSHEET_SECTIONS",
    "normalize_required_sections",
    "validate_worksheet",
    "worksheet_prompt",
    "worksheet_schema",
]
