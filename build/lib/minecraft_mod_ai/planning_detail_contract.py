from __future__ import annotations

"""Shared detailed-plan grounding contract.

Authored design rows may cite zero or more evidence constraints. Externally verifiable
facts and reuse claims must cite sufficient grounded evidence. Keeping this validation in
one neutral module prevents compiler/state-invariant schema drift.
"""

from collections.abc import Mapping
from typing import Any

GROUNDED_BINDING_KINDS = frozenset(
    {
        "api_symbol",
        "version_compatibility",
        "repository_fact",
        "dependency",
        "source_behavior",
    }
)
REUSE_MODES = frozenset({"reuse", "adapt", "reference_only", "new_required"})

_AUTHORED_COLLECTIONS = (
    "implementation_capabilities",
    "implementation_obligations",
    "artifact_obligations",
    "verification_obligations",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def validate_evidence_refs(
    refs: Any,
    allowed: set[str],
    *,
    field: str,
    require: bool,
) -> list[str]:
    """Normalize evidence refs and reject blanks, duplicates, or non-grounded refs."""

    if not isinstance(refs, list):
        raise ValueError(f"DETAILED_PLAN_{field.upper()}: evidence refs must be an array")
    normalized = [_text(ref) for ref in refs]
    if any(not ref for ref in normalized):
        raise ValueError(f"DETAILED_PLAN_{field.upper()}: evidence refs contain an empty value")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"DETAILED_PLAN_{field.upper()}: duplicate evidence refs")
    if require and not normalized:
        raise ValueError(f"DETAILED_PLAN_{field.upper()}: grounded evidence is required")
    unknown = [ref for ref in normalized if ref not in allowed]
    if unknown:
        raise ValueError(
            f"DETAILED_PLAN_{field.upper()}: unknown evidence refs: " + ", ".join(unknown)
        )
    return normalized


def validate_detailed_plan_grounding(
    plan: Mapping[str, Any],
    sufficient_refs: set[str],
) -> None:
    """Validate the evidence semantics shared by compiler and canonical-state checks."""

    for collection in _AUTHORED_COLLECTIONS:
        rows = plan.get(collection, [])
        if not isinstance(rows, list):
            raise ValueError(f"PROMPT_STATE_DECISION: {collection} must be an array")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"PROMPT_STATE_DECISION: {collection}[{index}] must be an object"
                )
            validate_evidence_refs(
                row.get("constraint_evidence_refs"),
                sufficient_refs,
                field=f"{collection}_{index}_constraint",
                require=False,
            )

    bindings = plan.get("grounded_bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("PROMPT_STATE_DECISION: grounded_bindings must be an array")
    for index, row in enumerate(bindings):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"PROMPT_STATE_DECISION: grounded_bindings[{index}] must be an object"
            )
        kind = _text(row.get("kind"))
        fact = _text(row.get("fact"))
        if kind not in GROUNDED_BINDING_KINDS or not fact:
            raise ValueError(
                f"PROMPT_STATE_DECISION: grounded_bindings[{index}] has invalid kind/fact"
            )
        validate_evidence_refs(
            row.get("evidence_refs"),
            sufficient_refs,
            field=f"grounded_binding_{index}",
            require=True,
        )

    reuse = plan.get("reuse_candidates", [])
    if not isinstance(reuse, list):
        raise ValueError("PROMPT_STATE_DECISION: reuse_candidates must be an array")
    for index, row in enumerate(reuse):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"PROMPT_STATE_DECISION: reuse_candidates[{index}] must be an object"
            )
        mode = _text(row.get("mode"))
        reason = _text(row.get("reason"))
        if mode not in REUSE_MODES or not reason:
            raise ValueError(
                f"PROMPT_STATE_DECISION: reuse_candidates[{index}] has invalid mode/reason"
            )
        validate_evidence_refs(
            [row.get("evidence_ref")],
            sufficient_refs,
            field=f"reuse_candidate_{index}",
            require=True,
        )


__all__ = [
    "GROUNDED_BINDING_KINDS",
    "REUSE_MODES",
    "validate_detailed_plan_grounding",
    "validate_evidence_refs",
]
