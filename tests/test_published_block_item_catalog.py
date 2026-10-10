"""Regression for the actual *shipped* HOST catalog, not just the rules builder."""
from minecraft_mod_ai.host_version_catalog import host_target
from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.implementation_fact import ImplementationFact
from minecraft_mod_ai.prompt_fact_types import FactType


def test_published_minecraft_26_2_catalog_generates_inventory_visible_block():
    context = host_target("26.2").version_context
    row = context.to_dict()["host_facts"]["leaf_bindings"]["minecraft/block/model"]
    assert row["implementation"]["extra_templates"] == [
        "minecraft/resource/item/client_block_item"
    ]

    block = ImplementationFact(
        fact_id="crystal_block.exists",
        fact_type=FactType.BLOCK_EXISTS,
        subject="crystal_block",
        source_clause="Register the crystal_block as a placeable inventory item.",
    )
    jobs = expand_facts_to_jobs(
        (block,),
        mod_id="mmm_debug_crystal",
        package_name="ai.minecraft.generated.mmm_debug_crystal",
        minecraft_version="26.2",
        version_context=context,
    )
    paths = {job.target_path for job in jobs}
    assert "src/main/resources/assets/mmm_debug_crystal/items/crystal_block.json" in paths
    assert "src/main/resources/assets/mmm_debug_crystal/models/block/crystal_block.json" in paths


def test_published_catalog_remains_stable_across_loads():
    first = host_target("26.2").version_context
    second = host_target("26.2").version_context
    assert first.context_id == second.context_id
