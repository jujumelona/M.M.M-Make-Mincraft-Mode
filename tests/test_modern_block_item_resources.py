"""Block inventory rendering must use the client item-model definition on 26.2."""
from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.implementation_fact import ImplementationFact
from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
from minecraft_mod_ai.prompt_fact_types import FactType
from minecraft_mod_ai.host_version_catalog import host_target


def test_modern_block_model_expands_client_block_item_resource():
    facts = build_version_facts("26.2", base_facts={})
    implementation = facts["leaf_bindings"]["minecraft/block/model"]["implementation"]
    assert implementation["template"] == "minecraft/resource/block/model_cube_all"
    assert "minecraft/resource/item/client_block_item" in implementation["extra_templates"]

    # Expand against the actual shipped version context. A SimpleNamespace
    # binding stub does not implement template admission and masks runtime drift.
    ctx = host_target("26.2").version_context
    block = ImplementationFact(
        fact_id="crystal_block.exists",
        fact_type=FactType.BLOCK_EXISTS,
        subject="crystal_block",
        source_clause="Create an inventory-visible crystal_block.",
    )
    jobs = expand_facts_to_jobs(
        (block,),
        mod_id="mmm_debug_crystal",
        package_name="example.crystal",
        minecraft_version="26.2",
        version_context=ctx,
    )
    templates = [job.template_id for job in jobs]
    assert templates.count("minecraft/resource/item/client_block_item") == 1
    assert templates.count("minecraft/resource/block/model_cube_all") == 1
