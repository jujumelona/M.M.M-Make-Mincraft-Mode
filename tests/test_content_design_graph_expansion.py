import json
from copy import deepcopy

from minecraft_mod_ai.atomic_design_pipeline import compile_atomic_design
from minecraft_mod_ai.fact_source_reuse import (
    FactReuseClassifier,
    ReuseMode,
)
from minecraft_mod_ai.implementation_fact import FactType, ImplementationFact


class ExpansionGraphRouter:
    def __init__(self, *, nodes=None, edges=None, capabilities=None, properties=None):
        self.calls = []
        self.nodes = nodes or []
        self.edges = edges or []
        self.capabilities = capabilities or {}
        self.properties = properties or {}

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, **kwargs):
        self.calls.append(tool_name)
        context = json.loads(messages[-1]["content"])
        accepted = context["accepted_records"]

        if tool_name == "submit_design_content_entity":
            rows = self.nodes
        elif tool_name == "submit_design_content_relation":
            rows = self.edges
        elif tool_name == "submit_design_content_capability":
            eid = context["entity"]["entity_id"]
            cap = self.capabilities.get(eid, "ITEM_EXISTS")
            rows = [{"fact_type": cap}]
        elif tool_name == "submit_design_content_property":
            eid = context["entity"]["entity_id"]
            props = self.properties.get(
                eid,
                [
                    {"property": "display_name", "value": eid.replace("_", " ").title()},
                    {"property": "shape", "value": "faceted chunk"},
                ],
            )
            rows = props
        elif tool_name == "submit_design_decision":
            rows = []
        else:
            raise AssertionError(tool_name)

        if len(accepted) < len(rows):
            return {
                "status": "record",
                "record": deepcopy(rows[len(accepted)]),
                "reason": "",
            }
        return {
            "status": "done" if rows else "not_applicable",
            "record": None,
            "reason": "No additional records required",
        }


def test_entity_exists_lowering_and_asset_mold():
    router = ExpansionGraphRouter(
        nodes=[{"entity_id": "space_boss", "kind": "boss", "role": "Boss mob"}],
        capabilities={"space_boss": "ENTITY_EXISTS"},
        properties={
            "space_boss": [
                {"property": "display_name", "value": "Space Boss"},
                {"property": "health", "value": "300"},
                {"property": "shape", "value": "humanoid titan"},
                {"property": "main_color", "value": "void purple"},
            ]
        },
    )
    design = compile_atomic_design("Space boss mob", router)
    assert len(design["modules"]) == 1
    boss_mod = design["modules"][0]
    assert boss_mod.module_id == "space_boss"
    assert boss_mod.kind == "entity"
    assert boss_mod.config["name"] == "Space Boss"
    assert boss_mod.config["health"] == "300"

    fact = next(f for f in design["_implementation_facts"] if f.subject == "space_boss")
    assert fact.fact_type == FactType.ENTITY_EXISTS

    assert len(design["assets"]) == 1
    asset = design["assets"][0]
    assert asset.kind == "entity"
    assert asset.width == 64
    assert asset.height == 64
    assert "mob texture map" in asset.prompt
    assert "void purple" in asset.prompt


def test_gui_panel_lowering_and_asset_mold():
    router = ExpansionGraphRouter(
        nodes=[{"entity_id": "fusion_gui", "kind": "screen", "role": "Machine interface"}],
        capabilities={"fusion_gui": "GUI_EXISTS"},
        properties={
            "fusion_gui": [
                {"property": "display_name", "value": "Fusion Chamber"},
                {"property": "screen_type", "value": "container_9x3"},
                {"property": "surface", "value": "dark metallic plate"},
            ]
        },
    )
    design = compile_atomic_design("Fusion GUI", router)
    gui_mod = design["modules"][0]
    assert gui_mod.kind == "gui"
    assert gui_mod.config["screen_type"] == "container_9x3"

    asset = design["assets"][0]
    assert asset.kind == "gui"
    assert asset.width == 256
    assert asset.height == 256
    assert "container interface panel" in asset.prompt


def test_machine_and_network_capabilities():
    router = ExpansionGraphRouter(
        nodes=[
            {"entity_id": "fusion_core", "kind": "machine", "role": "Energy generator"},
            {"entity_id": "power_sync", "kind": "packet", "role": "Sync power packet"},
        ],
        capabilities={
            "fusion_core": "BLOCK_ENTITY_EXISTS",
            "power_sync": "NETWORK_PACKET",
        },
        properties={
            "fusion_core": [
                {"property": "display_name", "value": "Fusion Core"},
                {"property": "sync_type", "value": "ticking_block_entity"},
            ],
            "power_sync": [
                {"property": "display_name", "value": "Power Sync"},
                {"property": "channel", "value": "space:power_sync"},
                {"property": "direction", "value": "s2c"},
            ],
        },
    )
    design = compile_atomic_design("Fusion machine and networking", router)
    kinds = {m.module_id: m.kind for m in design["modules"]}
    assert kinds["fusion_core"] == "machine"
    assert kinds["power_sync"] == "networking"


def test_expanded_relations_graph():
    router = ExpansionGraphRouter(
        nodes=[
            {"entity_id": "controller_item", "kind": "item", "role": "Handheld controller"},
            {"entity_id": "control_gui", "kind": "gui", "role": "Control panel"},
            {"entity_id": "machine_block", "kind": "machine", "role": "Fabricator machine"},
            {"entity_id": "sync_packet", "kind": "packet", "role": "Sync packet"},
            {"entity_id": "spawn_drone", "kind": "entity", "role": "Drone companion"},
            {"entity_id": "space_dim", "kind": "dimension", "role": "Orbit dimension"},
        ],
        capabilities={
            "controller_item": "ITEM_EXISTS",
            "control_gui": "GUI_EXISTS",
            "machine_block": "BLOCK_ENTITY_EXISTS",
            "sync_packet": "NETWORK_PACKET",
            "spawn_drone": "ENTITY_EXISTS",
            "space_dim": "DIMENSION",
        },
        properties={
            "controller_item": [{"property": "display_name", "value": "Remote Controller"}],
            "control_gui": [{"property": "display_name", "value": "Control UI"}],
            "machine_block": [{"property": "display_name", "value": "Fabricator"}],
            "sync_packet": [{"property": "display_name", "value": "Sync Packet"}],
            "spawn_drone": [{"property": "display_name", "value": "Drone"}],
            "space_dim": [{"property": "display_name", "value": "Space Orbit"}],
        },
        edges=[
            {"relation_type": "opens", "source_id": "controller_item", "target_id": "control_gui"},
            {"relation_type": "controls", "source_id": "controller_item", "target_id": "machine_block"},
            {"relation_type": "synchronizes", "source_id": "machine_block", "target_id": "sync_packet"},
            {"relation_type": "spawns", "source_id": "controller_item", "target_id": "spawn_drone"},
            {"relation_type": "transports_to", "source_id": "controller_item", "target_id": "space_dim"},
        ],
    )
    design = compile_atomic_design("Complex space machinery system", router)
    mod_map = {m.module_id: m for m in design["modules"]}

    # controller_item depends on control_gui, machine_block, spawn_drone, space_dim
    ctrl = mod_map["controller_item"]
    assert "control_gui" in ctrl.depends_on
    assert "machine_block" in ctrl.depends_on
    assert "spawn_drone" in ctrl.depends_on
    assert "space_dim" in ctrl.depends_on
    assert ctrl.config["opens_gui"] == "control_gui"
    assert ctrl.config["controls"] == "machine_block"
    assert ctrl.config["spawns"] == "spawn_drone"
    assert ctrl.config["transports_to"] == "space_dim"

    # machine_block synchronizes via sync_packet
    mach = mod_map["machine_block"]
    assert "sync_packet" in mach.depends_on
    assert mach.config["sync_packet"] == "sync_packet"

    # implementation facts for relations
    rel_facts = [f for f in design["_implementation_facts"] if f.fact_type == FactType.CONTENT_RELATION]
    assert len(rel_facts) == 5


def test_fact_source_reuse_classifier_index():
    index = {
        "symbols": {
            "RAW_LUNITE": {"file": "src/main/java/com/demo/ModItems.java"},
            "LUNITE_BLOCK": {"file": "src/main/java/com/demo/ModBlocks.java"},
        }
    }
    classifier = FactReuseClassifier(project_index=index)

    fact_exact = ImplementationFact(
        fact_id="raw_lunite.exists",
        fact_type=FactType.ITEM_EXISTS,
        subject="raw_lunite",
    )
    dec_exact = classifier.classify(fact_exact)
    assert dec_exact.mode == ReuseMode.REUSE
    assert dec_exact.matching_symbol == "RAW_LUNITE"
    assert "ModItems.java" in dec_exact.target_file

    fact_adapt = ImplementationFact(
        fact_id="lunite.exists",
        fact_type=FactType.ITEM_EXISTS,
        subject="lunite",
    )
    dec_adapt = classifier.classify(fact_adapt)
    assert dec_adapt.mode == ReuseMode.ADAPT

    fact_new = ImplementationFact(
        fact_id="solar_battery.exists",
        fact_type=FactType.ITEM_EXISTS,
        subject="solar_battery",
    )
    dec_new = classifier.classify(fact_new)
    assert dec_new.mode == ReuseMode.NEW


def test_fact_source_reuse_classifier_filesystem(tmp_path):
    java_dir = tmp_path / "src/main/java/com/demo"
    java_dir.mkdir(parents=True)
    items_file = java_dir / "ModItems.java"
    items_file.write_text(
        "package com.demo;\n"
        "public class ModItems {\n"
        "    public static final Item COPPER_WRENCH = null;\n"
        "}\n",
        encoding="utf-8",
    )

    classifier = FactReuseClassifier(project_root=tmp_path)
    fact = ImplementationFact(
        fact_id="copper_wrench.exists",
        fact_type=FactType.ITEM_EXISTS,
        subject="copper_wrench",
    )
    decision = classifier.classify(fact)
    assert decision.mode == ReuseMode.REUSE
    assert decision.matching_symbol == "COPPER_WRENCH"
