"""Block inventory rendering must use the client item-model definition on 26.2."""
from types import SimpleNamespace

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.implementation_fact import ImplementationFact
from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
from minecraft_mod_ai.prompt_fact_types import FactType


def test_modern_block_model_expands_client_block_item_resource():
    facts = build_version_facts("26.2", base_facts={})
    implementation = facts["leaf_bindings"]["minecraft/block/model"]["implementation"]
    assert implementation["template"] == "minecraft/resource/block/model_cube_all"
    assert "minecraft/resource/item/client_block_item" in implementation["extra_templates"]

    ctx = SimpleNamespace(
        minecraft="26.2",
        context_id="test-26.2",
        leaf_bindings=facts["leaf_bindings"],
    )
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
