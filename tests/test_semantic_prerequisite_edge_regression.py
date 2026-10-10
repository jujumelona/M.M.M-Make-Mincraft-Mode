from __future__ import annotations

"""REG-021/REG-SEM-001: host-approved prerequisite edges must survive planning."""

import pytest

from minecraft_mod_ai.minecraft_requirement_dependencies import (
    bind_selected_feature_dependencies,
)


def _catalog():
    return {
        "requirements": [
            {"requirement_id": "req_farm", "capability": "resource.farming", "depends_on": []},
            {"requirement_id": "req_ship", "capability": "spacecraft.construction", "depends_on": ["req_farm"]},
            {"requirement_id": "req_launch", "capability": "space.launch", "depends_on": ["req_ship"]},
        ],
    }


def test_space_progression_prerequisite_graph_cannot_collapse_to_zero_edges():
    original = _catalog()
    result = bind_selected_feature_dependencies(original)
    edges = result["requirement_graph"]["edges"]
    assert {(e["from_requirement_ref"], e["to_requirement_ref"]) for e in edges} == {
        ("req_farm", "req_ship"),
        ("req_ship", "req_launch"),
    }
    requirements = {r["requirement_id"]: r for r in result["requirements"]}
    assert requirements["req_launch"]["unlock_policy"]["required_requirement_refs"] == ["req_ship"]
    assert requirements["req_launch"]["dependency_provenance"]["required_requirement_refs"] == ["req_ship"]
    assert original["requirements"][1]["depends_on"] == ["req_farm"]


def test_missing_space_progression_prerequisite_is_rejected_not_dropped():
    catalog = _catalog()
    catalog["requirements"][2]["depends_on"] = ["req_ship_missing"]
    with pytest.raises(ValueError, match="unknown dependency refs"):
        bind_selected_feature_dependencies(catalog)


def test_cyclic_space_progression_prerequisites_are_rejected():
    catalog = _catalog()
    catalog["requirements"][0]["depends_on"] = ["req_launch"]
    with pytest.raises(ValueError, match="cycle"):
        bind_selected_feature_dependencies(catalog)
