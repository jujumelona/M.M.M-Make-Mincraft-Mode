"""Invalidate legacy prose-derived plans while preserving source research checkpoints."""
from collections.abc import Mapping
from copy import deepcopy

from .planning_state_contract import _hash_without
from .root_cause_trace import emit_root_cause


def refresh_worksheet_checkpoint(state):
    decisions = state.get("decisions", [])
    legacy = any(
        isinstance(decision, Mapping)
        and decision.get("decision_type") == "detailed_implementation_plan"
        and isinstance(decision.get("engineering_worksheet"), Mapping)
        and any(isinstance(row, Mapping) and isinstance(row.get("specification"), str)
                for row in decision["engineering_worksheet"].values())
        for decision in decisions
    )
    if not legacy:
        return state
    # Never re-sign a tampered checkpoint merely because it uses the old schema.
    if state.get("state_sha256") != _hash_without(state, "state_sha256"):
        raise ValueError("PROMPT_STATE_HASH: legacy worksheet checkpoint hash mismatch")
    result = deepcopy(dict(state))
    result["decisions"] = [row for row in result["decisions"] if not (
        isinstance(row, Mapping) and row.get("decision_type") == "detailed_implementation_plan"
    )]
    result["coverage"] = []
    result["plan_ready"] = False
    result["state_sha256"] = _hash_without(result, "state_sha256")
    emit_root_cause(
        "legacy_worksheet_invalidated", stage="planning_state", result="PASS",
        operation="refresh_worksheet_checkpoint",
        details={"removed_details": len(decisions) - len(result["decisions"]),
                 "reason": "Prose worksheets require regeneration under fixed record templates.",
                 "research_preserved": True},
    )
    return result
