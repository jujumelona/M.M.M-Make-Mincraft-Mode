"""Block/BlockItem recipe contract: never confuse Block and Item registries."""
from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.artifact_ports import PortKind, PortRegistry, TypedPort
from minecraft_mod_ai.artifact_validators.resource_links import validate_resource_links
from minecraft_mod_ai.implementation_template_renderer import render_template
from minecraft_mod_ai.task_template_catalog import load_template


def _shaped_recipe() -> str:
    return json.dumps({
        "type": "minecraft:crafting_shaped",
        "pattern": ["FF", "FF"],
        "key": {"F": "mmm_debug_crystal:crystal_fragment"},
        "result": {"id": "mmm_debug_crystal:crystal_block", "count": 1},
    })


def test_block_template_registers_a_real_inventory_item_and_publishes_item_port():
    template = load_template("fabric/block/register_basic")
    names = {row["binding"]: row for row in template["produces"]}
    assert names["{{subject}}.block_registry_id"]["target_type"] == "Block"
    block_item = names["{{subject}}.registry_id"]
    assert block_item["kind"] == "REGISTRY_ID"
    assert block_item["target_type"] == "Item"
    output = render_template(template, {
        "java_constant": "CRYSTAL_BLOCK",
        "registry_path": "crystal_block",
        "mod_id": "mmm_debug_crystal",
        "package_path": "ai/minecraft/generated/mmm_debug_crystal",
        "subject": "crystal_block",
    })
    assert "new BlockItem(CRYSTAL_BLOCK" in output
    assert "BuiltInRegistries.ITEM" in output
    assert "ResourceKey.create(Registries.ITEM" in output


def test_recipe_result_rejects_block_only_port_then_accepts_actual_item_port():
    registry = PortRegistry()
    registry.publish(TypedPort(
        name="crystal_block.block_registry_id",
        port_kind=PortKind.REGISTRY_ID,
        target_type="Block",
        value="mmm_debug_crystal:crystal_block",
    ))
    registry.publish(TypedPort(
        name="crystal_fragment.registry_id",
        port_kind=PortKind.REGISTRY_ID,
        target_type="Item",
        value="mmm_debug_crystal:crystal_fragment",
    ))
    with pytest.raises(ValueError, match="RESOURCE_REFERENCE_MISSING"):
        validate_resource_links(_shaped_recipe(), {"registry_kind": "item"}, registry)
    registry.publish(TypedPort(
        name="crystal_block.registry_id",
        port_kind=PortKind.REGISTRY_ID,
        target_type="Item",
        value="mmm_debug_crystal:crystal_block",
    ))
    receipt = validate_resource_links(
        _shaped_recipe(), {"registry_kind": "item"}, registry
    )
    assert receipt["status"] == "PASS"
    assert "mmm_debug_crystal:crystal_block" in receipt["references"]


def test_native_26_x_registry_templates_use_existing_identifier_methods():
    """Regression: the exact generated symbols in the Colab compile failure."""
    inputs = {
        "mod_id": "mmm_debug_crystal",
        "registry_path": "crystal_block",
        "java_constant": "CRYSTAL_BLOCK",
        "package_path": "ai/minecraft/generated/mmm_debug_crystal",
        "subject": "crystal_block",
    }
    block_key = render_template(load_template("fabric/block/key"), inputs)
    block_registry = render_template(load_template("fabric/block/register_basic"), inputs)
    assert 'Identifier.fromNamespaceAndPath("mmm_debug_crystal", "crystal_block")' in block_key
    assert "Identifier.of(" not in block_key
    assert "ModBlockIds.CRYSTAL_BLOCK_KEY.identifier()" in block_registry
    assert "ModBlockIds.CRYSTAL_BLOCK_KEY.location()" not in block_registry

    item_key = render_template(load_template("fabric/item/key"), {
        **inputs,
        "registry_path": "crystal_fragment",
        "java_constant": "CRYSTAL_FRAGMENT",
        "subject": "crystal_fragment",
    })
    assert 'Identifier.fromNamespaceAndPath("mmm_debug_crystal", "crystal_fragment")' in item_key
    assert "Identifier.of(" not in item_key
