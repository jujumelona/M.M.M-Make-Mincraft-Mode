from __future__ import annotations

"""REG-022 / REG-DESIGN-001: model design choices cannot become authored requirements."""

from copy import deepcopy

import pytest

from minecraft_mod_ai.design_requirement_contract import _validate_requirement_coverage
from minecraft_mod_ai.spec import SpecValidationError


APPROVED = [{
    "requirement_id": "req_ship",
    "authored_text": "Let the player upgrade the ship's speed.",
    "semantic_statement": "Ship speed is upgradeable.",
}]


def _design(*refs):
    return {"modules": [{
        "plugin_id": "ship_performance",
        "requirement_refs": list(refs),
        "implementation_obligations": [
            "Designer-proposed initial ship speed of 900 units: not an authored requirement."
        ],
    }]}


def test_unapproved_numeric_default_cannot_be_promoted_to_requirement_identity():
    approved = deepcopy(APPROVED)
    with pytest.raises(SpecValidationError, match="unknown requirement refs"):
        _validate_requirement_coverage(
            _design("req_ship", "req_ship_speed_900"), approved
        )
    assert approved == APPROVED


def test_designer_choice_stays_under_original_requirement_without_inventing_an_id():
    result = _validate_requirement_coverage(_design("req_ship"), deepcopy(APPROVED))
    binding = result["_requirement_design_bindings"]
    assert binding["requirement_ids"] == ["req_ship"]
    assert [row["requirement_id"] for row in binding["bindings"]] == ["req_ship"]
    assert "req_ship_speed_900" not in binding["requirement_ids"]
    assert result["modules"][0]["implementation_obligations"]
