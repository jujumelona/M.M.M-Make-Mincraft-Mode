from __future__ import annotations

from minecraft_mod_ai.authored_content_contract import (
    CONTENT_GRAPH_DRIVER_CONCERNS,
    CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS,
    content_owned_refs,
)
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def test_registry_binding_runs_after_concrete_content_discovery() -> None:
    assert CONTENT_GRAPH_DRIVER_CONCERNS == (
        "assets",
        "interactions",
        "displayed_state",
        "data_resources",
        "registries",
    )
    assert CONTENT_GRAPH_DRIVER_CONCERNS[-2:] == (
        "data_resources",
        "registries",
    )


def test_engineering_resource_rows_are_host_content_constraints() -> None:
    assert {"paths", "registries", "data_resources"} <= set(
        CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS
    )


def _resource_section(concern: str, rows: list[dict]) -> dict:
    specification = {
        name: []
        for name in DETAIL_RECORDS["resources_and_ui"]
    }
    specification[concern] = rows
    specification["inapplicable_concerns"] = []
    return {
        "resources_and_ui": {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }


def test_registry_constraint_is_externally_owned_without_synthetic_entity() -> None:
    structured = _resource_section(
        "registries",
        [{
            "purpose": "ore tags",
            "identifier": "space_mode_resources_ore_tags",
            "binding_requirement": "bind resource types without inventing content",
        }],
    )

    assert content_owned_refs(structured, {}) == frozenset({
        "resources_and_ui.registries",
    })


def test_data_resource_constraint_is_externally_owned_without_synthetic_entity() -> None:
    structured = _resource_section(
        "data_resources",
        [{
            "kind": "tag",
            "purpose": "ore resource metadata",
            "owner": "space_mode",
        }],
    )

    assert content_owned_refs(structured, {}) == frozenset({
        "resources_and_ui.data_resources",
    })
