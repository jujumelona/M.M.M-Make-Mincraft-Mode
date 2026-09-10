from __future__ import annotations

from minecraft_mod_ai.requirement_branch_scope_contract import (
    _branches_for_requirement,
    _scoped_branch_predicates,
)
from minecraft_mod_ai.structural_minecraft_runtime_contract import structural_steps_for_requirement


def _requirement(
    requirement_id: str,
    capability: str,
    statement: str,
    *artifacts: str,
) -> dict:
    return {
        "requirement_id": requirement_id,
        "capability": capability,
        "statement": statement,
        "source_span": {"text": statement},
        "provides": [f"capability:{capability}"],
        "gameplay_capabilities": [capability],
        "implementation_capabilities": [],
        "artifact_obligations": [{"kind": kind} for kind in artifacts],
    }


def test_client_render_branch_is_scoped_to_exact_activating_requirement():
    requirements = [
        _requirement(
            "REQ_RESOURCE",
            "resource.mining",
            "The player mines a registered resource on the server.",
            "loot",
        ),
        _requirement(
            "REQ_GUI",
            "ui.menu",
            "A client menu shows the player's controls.",
            "screen",
        ),
    ]

    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )

    client = branches["needs_client_render"]
    assert client["status"] == "ACTIVE"
    assert client["requirement_status"]["REQ_RESOURCE"] == "NOT_APPLICABLE"
    assert client["requirement_status"]["REQ_GUI"] == "ACTIVE"
    assert client["evidence_refs"] == ["REQ_GUI"]


def test_resource_steps_do_not_inherit_sibling_gui_branch():
    resource = _requirement("REQ_RESOURCE", "resource.mining", "Mine a resource.", "loot")
    gui = _requirement("REQ_GUI", "ui.menu", "Show a client menu.", "screen")
    requirements = [resource, gui]
    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )

    resource_branches = _branches_for_requirement(branches, "REQ_RESOURCE")
    resource_steps = structural_steps_for_requirement(resource)

    assert resource_branches["needs_client_render"]["status"] == "NOT_APPLICABLE"
    assert any("needs_datagen" in step.branch_features for step in resource_steps)
    assert all("needs_client_render" not in step.branch_features for step in resource_steps)


def test_ui_requirement_activates_its_own_client_branch():
    requirement = _requirement(
        "REQ_GUI",
        "ui.menu",
        "The requested client menu presents authoritative state.",
        "screen",
    )
    branches = _scoped_branch_predicates(
        [requirement],
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )

    gui_branches = _branches_for_requirement(branches, "REQ_GUI")
    steps = structural_steps_for_requirement(requirement)

    assert gui_branches["needs_client_render"]["status"] == "ACTIVE"
    assert any("needs_client_render" in step.branch_features for step in steps)


def test_multiple_loader_topology_is_intentionally_global_architecture_branch():
    requirements = [
        _requirement("REQ_A", "item.weapon", "Add a weapon item.", "item"),
        _requirement("REQ_B", "resource.mining", "Add a mineable resource.", "block"),
    ]
    branches = _scoped_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric", "neoforge"]}},
    )

    loader = branches["needs_loader_leaf"]
    assert loader["requirement_status"] == {"REQ_A": "ACTIVE", "REQ_B": "ACTIVE"}
    assert all(
        "target-topology:multiple-loader-modules" in refs
        for refs in loader["requirement_evidence_refs"].values()
    )


def test_generated_resource_component_only_activates_matching_requirement():
    requirements = [
        _requirement("REQ_RECIPE", "crafting.recipe", "Provide a recipe."),
        _requirement("REQ_CHAT", "custom.semantic.chat_command", "Provide a chat command."),
    ]
    components = [
        {
            "component_id": "component_recipe",
            "kind": "generated_resource",
            "provides": ["capability:crafting.recipe"],
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
