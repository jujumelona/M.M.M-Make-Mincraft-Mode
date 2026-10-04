"""Reject obsolete planning checkpoints; no legacy migration path is retained."""

from collections.abc import Mapping


def refresh_worksheet_checkpoint(state):
    decisions = state.get("decisions", [])
    obsolete = any(
        isinstance(decision, Mapping)
        and decision.get("decision_type") == "detailed_implementation_plan"
        and decision.get("worksheet_contract") != "authored_concern_records"
        for decision in decisions
    ) or any(
        isinstance(row, Mapping)
        and row.get("schema_version") != "mmm/detail-criterion-records"
        for row in state.get("detail_progress", [])
    )
    if obsolete:
        raise ValueError(
            "PLANNING_CHECKPOINT_OBSOLETE: legacy worksheet/prose planning state is "
            "unsupported; restart planning from the current canonical contract"
        )
    return state
