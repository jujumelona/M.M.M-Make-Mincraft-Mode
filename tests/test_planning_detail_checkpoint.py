from copy import deepcopy
import pytest
from minecraft_mod_ai.planning_detail_checkpoint import refresh_worksheet_checkpoint
from minecraft_mod_ai.planning_state_contract import _hash_without


def _legacy():
    state = {"decisions": [
        {"decision_type": "requirement", "requirement_id": "r"},
        {"decision_type": "detailed_implementation_plan", "engineering_worksheet": {
            "behavior_contract": {"specification": "Old prose contract"}}},
    ], "evidence": [{"source": "preserve me"}], "coverage": [{"requirement_ref": "r"}],
        "plan_ready": True}
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def test_legacy_details_are_regenerated_without_losing_research():
    state = _legacy()
    original = deepcopy(state)
    result = refresh_worksheet_checkpoint(state)
    assert state == original
    assert result["decisions"] == [state["decisions"][0]]
    assert result["evidence"] == state["evidence"]
    assert result["coverage"] == [] and result["plan_ready"] is False
    assert result["state_sha256"] == _hash_without(result, "state_sha256")
    assert refresh_worksheet_checkpoint(result) is result


def test_legacy_checkpoint_integrity_is_checked_before_regeneration():
    state = _legacy()
    state["evidence"] = []
    with pytest.raises(ValueError, match="hash mismatch"):
        refresh_worksheet_checkpoint(state)
