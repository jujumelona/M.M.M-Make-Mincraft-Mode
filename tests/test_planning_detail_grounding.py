from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS, validate_worksheet
from minecraft_mod_ai.planning_state_implementation import _PARAMETERS, _validate_refs


def _authored_worksheet() -> dict[str, dict[str, object]]:
    return {
        key: {
            "specification": (
                f"{key} has a distinct authored implementation contract with an owner, "
                "condition, boundary, and observable result."
            ),
            "constraint_evidence_refs": [],
        }
        for key in WORKSHEET_SECTIONS
    }


def test_authored_design_sections_do_not_require_fake_evidence() -> None:
    sheet = _authored_worksheet()

    assert validate_worksheet(sheet, {"source:1"}) == sheet


def test_constraint_evidence_must_still_be_host_allowed_when_present() -> None:
    sheet = _authored_worksheet()
    sheet["algorithm"]["constraint_evidence_refs"] = ["model:guess"]

    with pytest.raises(ValueError, match="invalid constraint evidence"):
        validate_worksheet(sheet, {"source:1"})


def test_grounded_external_fact_requires_real_allowed_evidence() -> None:
    assert _validate_refs(["source:1"], {"source:1"}, field="binding") == ["source:1"]

    with pytest.raises(ValueError, match="grounded evidence is required"):
        _validate_refs([], {"source:1"}, field="binding")
    with pytest.raises(ValueError, match="unknown evidence refs"):
        _validate_refs(["model:guess"], {"source:1"}, field="binding")


def test_tool_schema_keeps_design_constraints_separate_from_grounded_bindings() -> None:
    properties = _PARAMETERS["properties"]
    capability_refs = properties["implementation_capabilities"]["items"]["properties"][
        "constraint_evidence_refs"
    ]
    binding_refs = properties["grounded_bindings"]["items"]["properties"]["evidence_refs"]

    assert "minItems" not in capability_refs
    assert binding_refs["minItems"] == 1
    assert "grounded_bindings" in _PARAMETERS["required"]
