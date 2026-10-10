from __future__ import annotations

from minecraft_mod_ai.authored_production import _normalize_content_build_dependencies
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.content_design_graph import _strip_semantic_content_build_dependencies


def _cycle_modules() -> tuple[ProductionModule, ...]:
    gui = "ship_construction_gui_screen_contract_ui_menu_unlock_001"
    segment = "ship_segment_item_block_entity_002"
    target = "spaceport_planet_target_list_gui_economy_system_001"
    return (
        ProductionModule(
            module_id=gui,
            kind="gui",
            config={
                "requires": [segment],
                "executable_relations": [{"relation": "requires", "target": segment}],
            },
            depends_on=(segment,),
        ),
        ProductionModule(
            module_id=segment,
            kind="item",
            config={
                "requires": [target],
                "executable_relations": [{"relation": "requires", "target": target}],
            },
            depends_on=(target,),
        ),
        ProductionModule(
            module_id=target,
            kind="gui",
            config={
                "requires": [gui],
                "executable_relations": [{"relation": "requires", "target": gui}],
            },
            depends_on=(gui,),
        ),
    )


def test_explicit_requires_edges_are_not_erased_to_hide_a_build_cycle() -> None:
    normalized = _strip_semantic_content_build_dependencies(_cycle_modules())

    # Explicit "requires" is a build prerequisite until independent evidence
    # proves otherwise. Dropping a cycle here would mask a scheduler defect.
    assert [module.depends_on for module in normalized] == [
        ("ship_segment_item_block_entity_002",),
        ("spaceport_planet_target_list_gui_economy_system_001",),
        ("ship_construction_gui_screen_contract_ui_menu_unlock_001",),
    ]
    assert normalized[0].config["requires"] == [
        "ship_segment_item_block_entity_002"
    ]
    assert normalized[0].config["executable_relations"] == [{
        "relation": "requires",
        "target": "ship_segment_item_block_entity_002",
    }]


def test_saved_requires_cycle_remains_visible_to_scheduler_without_fact_proof() -> None:
    original = _cycle_modules()
    normalized = _normalize_content_build_dependencies(original, ())

    assert [module.depends_on for module in normalized] == [
        module.depends_on for module in original
    ]
