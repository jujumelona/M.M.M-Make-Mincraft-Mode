import pytest

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.artifact_graph_executor import execute_artifact_graph
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact
from minecraft_mod_ai.host_version_catalog import host_target
from minecraft_mod_ai.resolved_version_context import VersionContextError
from minecraft_mod_ai.resource_fact_inputs import resource_inputs


def resource_fact(kind, value):
    return PromptFact(
        fact_id="resource", fact_type=kind, subject="conversion", value=value
    )


def run(tmp_path, fact):
    ctx = host_target("1.21.1").version_context
    jobs = expand_facts_to_jobs(
        [fact], mod_id="demo", package_name="org.demo",
        minecraft_version=ctx.minecraft, version_context=ctx,
    )
    return execute_artifact_graph(
        jobs,
        base_dir=tmp_path,
        context={
            "resolved_version_context": ctx.to_dict(),
            "known_registry_ids": {
                "item": ["minecraft:iron_ingot", "minecraft:iron_nugget"]
            }
        },
    )


@pytest.mark.parametrize(
    "kind,value",
    [
        (
            FactType.CRAFTING_RECIPE,
            {
                "kind": "shapeless",
                "ingredients": ["minecraft:iron_ingot"],
                "result_id": "minecraft:iron_nugget",
                "count": 9,
            },
        ),
        (
            FactType.CRAFTING_RECIPE,
            {
                "kind": "shaped",
                "pattern": ["X"],
                "key": {"X": "minecraft:iron_ingot"},
                "result_id": "minecraft:iron_nugget",
                "count": 9,
            },
        ),
        (
            FactType.SMELTING_RECIPE,
            {
                "cooking_type": "blasting",
                "ingredient": "minecraft:iron_ingot",
                "result_id": "minecraft:iron_nugget",
                "experience": 0.1,
                "cookingtime": 100,
            },
        ),
        (
            FactType.REGISTRY_TAG,
            {"registry_kind": "item", "members": ["minecraft:iron_ingot"]},
        ),
    ],
)
def test_unreviewed_resource_leaf_cannot_materialize(tmp_path, kind, value):
    # The 1.21.1 host has no admitted recipe/tag renderer. Its production
    # fail-closed gate must not be bypassed by a retired fixture.
    fact = resource_fact(kind, value)
    _, normalized = resource_inputs(fact, "demo")
    assert normalized["resource_references"]
    with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
        run(tmp_path, fact)
    assert not list(tmp_path.rglob("conversion.json"))


def test_unreviewed_registry_tag_refuses_before_any_write(tmp_path):
    fact = resource_fact(
        FactType.REGISTRY_TAG,
        {"registry_kind": "item", "members": ["minecraft:invented_item"]},
    )
    with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
        run(tmp_path, fact)
    assert not list(tmp_path.rglob("*.json"))


def test_manual_datagen_resource_cannot_be_mutated_by_unreviewed_leaf(tmp_path):
    generated = tmp_path / "src/main/generated/data/demo/tags/item/conversion.json"
    generated.parent.mkdir(parents=True)
    generated.write_text('{"values": []}')
    with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
        run(tmp_path, resource_fact(
            FactType.REGISTRY_TAG,
            {"registry_kind": "item", "members": ["minecraft:iron_ingot"]},
        ))
    assert generated.read_text() == '{"values": []}'


def test_existing_resource_is_not_overwritten(tmp_path):
    existing = tmp_path / "src/main/resources/data/demo/tags/item/conversion.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"values": []}')
    with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
        run(tmp_path, resource_fact(
            FactType.REGISTRY_TAG,
            {"registry_kind": "item", "members": ["minecraft:iron_ingot"]},
        ))
    assert existing.read_text() == '{"values": []}'


def test_shaped_recipe_rejects_unbound_pattern():
    # Semantic input rejection is independent of the target's leaf admission.
    fact = resource_fact(FactType.CRAFTING_RECIPE, {
        "kind": "shaped", "pattern": ["X"],
        "key": {"Y": "minecraft:iron_ingot"},
        "result_id": "minecraft:iron_nugget", "count": 1,
    })
    with pytest.raises(ValueError, match="RESOURCE_RECIPE_KEY"):
        resource_inputs(fact, "demo")


def test_recipe_requires_exact_distinct_local_ids_without_admission(tmp_path):
    facts = [
        PromptFact(fact_id=name, fact_type=FactType.ITEM_EXISTS, subject=name)
        for name in ("raw_ore", "gem")
    ]
    facts.append(resource_fact(FactType.CRAFTING_RECIPE, {
        "kind": "shapeless", "ingredients": ["raw_ore"],
        "result_id": "gem", "count": 1,
    }))
    _, normalized = resource_inputs(facts[-1], "demo")
    assert normalized["resource_references"] == ["demo:gem", "demo:raw_ore"]
    ctx = host_target("1.21.1").version_context
    with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
        expand_facts_to_jobs(
            facts, mod_id="demo", package_name="org.demo",
            minecraft_version=ctx.minecraft, version_context=ctx,
        )
    assert not list(tmp_path.rglob("conversion.json"))


def test_resource_template_rejects_target_context_mismatch():
    fact = resource_fact(
        FactType.REGISTRY_TAG,
        {"registry_kind": "item", "members": ["minecraft:iron_ingot"]},
    )
    # An explicitly resolved 1.21.1 binding must never silently lower a
    # requested 1.20.1 resource using the wrong registry/recipe API.
    ctx = host_target("1.21.1").version_context
    with pytest.raises(VersionContextError, match="VERSION_CONTEXT_MISMATCH"):
        expand_facts_to_jobs(
            [fact], mod_id="demo", package_name="org.demo",
            minecraft_version="1.20.1", version_context=ctx,
        )
