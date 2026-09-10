import json

import pytest

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.artifact_graph_executor import execute_artifact_graph
from minecraft_mod_ai.artifact_materializer import (
    MaterializeError,
    ensure_artifact_scaffolding,
)
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact


def resource_fact(kind, value):
    return PromptFact(
        fact_id="resource", fact_type=kind, subject="conversion", value=value
    )


def run(tmp_path, fact):
    jobs = expand_facts_to_jobs(
        [fact], mod_id="demo", package_name="org.demo", minecraft_version="1.21.2"
    )
    return execute_artifact_graph(
        jobs,
        base_dir=tmp_path,
        context={
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
def test_resource_leaf_materializes_real_json_without_model(tmp_path, kind, value):
    result = run(tmp_path, resource_fact(kind, value))
    assert result["status"] == "PASS"
    paths = list(tmp_path.rglob("conversion.json"))
    assert len(paths) == 1
    parsed = json.loads(paths[0].read_text())
    if "result" in parsed:
        assert type(parsed["result"]["count"]) is int
    if "ingredients" in parsed:
        assert isinstance(parsed["ingredients"], list)


def test_unknown_registry_reference_blocks_write(tmp_path):
    fact = resource_fact(
        FactType.REGISTRY_TAG,
        {"registry_kind": "item", "members": ["minecraft:invented_item"]},
    )
    with pytest.raises(ValueError, match="RESOURCE_REFERENCE_MISSING"):
        run(tmp_path, fact)
    assert not list(tmp_path.rglob("*.json"))


def test_manual_datagen_collision_blocks_write(tmp_path):
    generated = tmp_path / "src/main/generated/data/demo/tags/item/conversion.json"
    generated.parent.mkdir(parents=True)
    generated.write_text('{"values": []}')
    with pytest.raises(MaterializeError, match="OWNERSHIP_CONFLICT"):
        run(
            tmp_path,
            resource_fact(
                FactType.REGISTRY_TAG,
                {"registry_kind": "item", "members": ["minecraft:iron_ingot"]},
            ),
        )
    assert generated.read_text() == '{"values": []}'


def test_existing_resource_is_not_overwritten(tmp_path):
    fact = resource_fact(
        FactType.REGISTRY_TAG,
        {"registry_kind": "item", "members": ["minecraft:iron_ingot"]},
    )
    run(tmp_path, fact)
    path = next(tmp_path.rglob("conversion.json"))
    path.write_text('{"values": []}')
    with pytest.raises(ValueError, match="CHECKPOINT_TARGET_DRIFT"):
        run(tmp_path, fact)
    assert path.read_text() == '{"values": []}'


def test_shaped_recipe_rejects_unbound_pattern():
    with pytest.raises(ValueError, match="RESOURCE_RECIPE_KEY"):
        expand_facts_to_jobs(
            [
                resource_fact(
                    FactType.CRAFTING_RECIPE,
                    {
                        "kind": "shaped",
                        "pattern": ["X"],
                        "key": {"Y": "minecraft:iron_ingot"},
                        "result_id": "minecraft:iron_nugget",
                        "count": 1,
                    },
                )
            ],
            mod_id="demo",
            package_name="org.demo",
            minecraft_version="1.21.2",
        )


def test_recipe_consumes_two_distinct_local_item_ports(tmp_path):
    ensure_artifact_scaffolding(tmp_path, mod_id="demo", package_name="org.demo")
    facts = [
        PromptFact(fact_id=name, fact_type=FactType.ITEM_EXISTS, subject=name)
        for name in ("raw_ore", "gem")
    ]
    facts.append(
        resource_fact(
            FactType.CRAFTING_RECIPE,
            {
                "kind": "shapeless",
                "ingredients": ["raw_ore"],
                "result_id": "gem",
                "count": 1,
            },
        )
    )
    result = execute_artifact_graph(
        expand_facts_to_jobs(
            facts, mod_id="demo", package_name="org.demo", minecraft_version="1.21.2"
        ),
        base_dir=tmp_path,
    )
    assert result["status"] == "PASS"
    parsed = json.loads(
        (tmp_path / "src/main/resources/data/demo/recipe/conversion.json").read_text()
    )
    assert parsed["ingredients"] == ["demo:raw_ore"]
    assert parsed["result"]["id"] == "demo:gem"


def test_resource_template_rejects_unsupported_target():
    fact = resource_fact(
        FactType.REGISTRY_TAG,
        {"registry_kind": "item", "members": ["minecraft:iron_ingot"]},
    )
    with pytest.raises(ValueError, match="TARGET_UNSUPPORTED"):
        expand_facts_to_jobs(
            [fact], mod_id="demo", package_name="org.demo", minecraft_version="1.20.1"
        )
