from __future__ import annotations

from minecraft_mod_ai import evidence_first_planning as planning
from minecraft_mod_ai.requirement_branch_scope_contract import (
    _branches_for_requirement,
    _scoped_branch_predicates,
)


def _requirement(requirement_id: str, capability: str, statement: str) -> dict:
    return {
        "requirement_id": requirement_id,
        "capability": capability,
        "statement": statement,
        "source_span": {"text": statement},
        "provides": [f"capability:{capability}"],
        "gameplay_capabilities": [capability],
        "implementation_capabilities": [],
    }


def test_client_render_branch_is_scoped_to_activating_requirement():
    requirements = [
        _requirement(
            "REQ_MACHINE",
            "machine",
            "A machine processes ore automatically on the server.",
        ),
        _requirement(
            "REQ_GUI",
            "gui",
            "A client screen shows the player's controls.",
        ),
    ]

    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )

    client = branches["needs_client_render"]
    assert client["status"] == "ACTIVE"
    assert client["requirement_status"]["REQ_MACHINE"] == "NOT_APPLICABLE"
    assert client["requirement_status"]["REQ_GUI"] == "ACTIVE"
    assert client["evidence_refs"] == ["REQ_GUI"]


def test_machine_steps_do_not_inherit_sibling_gui_branch():
    requirements = [
        _requirement("REQ_MACHINE", "machine", "A machine processes ore."),
        _requirement("REQ_GUI", "gui", "Show a client screen."),
    ]
    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )

    machine_branches = _branches_for_requirement(branches, "REQ_MACHINE")
    machine_steps = planning._semantic_steps("machine", machine_branches)
    names = {step.name for step in machine_steps}

    assert "menu_contract" not in names
    assert "client_screen" not in names
    assert "resource_binding" in names


def test_same_machine_requirement_can_activate_its_own_screen_branch():
    requirements = [
        _requirement(
            "REQ_MACHINE",
            "machine",
            "The machine has a client screen for its processing state.",
        )
    ]
    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )

    machine_branches = _branches_for_requirement(branches, "REQ_MACHINE")
    names = {step.name for step in planning._semantic_steps("machine", machine_branches)}

    assert "menu_contract" in names
    assert "client_screen" in names


def test_multiple_loader_topology_is_intentionally_global_architecture_branch():
    requirements = [
        _requirement("REQ_A", "item", "Add an item."),
        _requirement("REQ_B", "block", "Add a block."),
    ]
    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric", "neoforge"]}},
    )

    loader = branches["needs_loader_leaf"]
    assert loader["requirement_status"] == {"REQ_A": "ACTIVE", "REQ_B": "ACTIVE"}
    assert all(
        "target-topology:multiple-loaders" in refs
        for refs in loader["requirement_evidence_refs"].values()
    )


def test_generated_resource_component_only_activates_matching_requirement():
    requirements = [
        _requirement("REQ_RECIPE", "recipe", "Provide a recipe."),
        _requirement("REQ_CHAT", "chat_command", "Provide a chat command."),
    ]
    components = [
        {
            "component_id": "component_recipe",
            "kind": "generated_resource",
            "provides": ["capability:recipe"],
        }
    ]

    branches = _scoped_branch_predicates(
        requirements,
        components,
        {"project_topology": {"loaders": ["fabric"]}},
    )

    datagen = branches["needs_datagen"]
    assert datagen["requirement_status"]["REQ_RECIPE"] == "ACTIVE"
    assert datagen["requirement_status"]["REQ_CHAT"] == "NOT_APPLICABLE"
