from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_expansion import (
    ArtifactExpansionError,
    expand_facts_to_jobs,
    validate_expansion_catalog,
)
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact


def test_expansion_catalog_contains_only_executable_tested_leaves():
    validate_expansion_catalog()


def test_expand_item_and_explicit_stack_limit_without_defaults():
    facts = [
        PromptFact(
            fact_id="fact_001",
            fact_type=FactType.ITEM_EXISTS,
            subject="raw_lunite",
            source_clause="raw lunite item exists",
        ),
        PromptFact(
            fact_id="fact_002",
            fact_type=FactType.ITEM_STACK_LIMIT,
            subject="raw_lunite",
            value=16,
            source_clause="stack limit is 16",
        ),
    ]

    jobs = expand_facts_to_jobs(facts, mod_id="space", package_name="com.foo.space")

    assert len(jobs) == 7
    template_ids = [j.template_id for j in jobs]
    assert "fabric/item/key" in template_ids
    assert "fabric/item/register_basic" in template_ids
    assert "minecraft/resource/item/client_item" in template_ids
    assert "minecraft/resource/item/model_generated" in template_ids
    assert "fabric/item/lang_en" in template_ids
    assert "fabric/item/initializer" in template_ids
    assert "fabric/item/settings_max_stack" in template_ids

    key_job = next(j for j in jobs if j.template_id == "fabric/item/key")
    assert key_job.target_path == "src/main/java/com/foo/space/registry/ModItemIds.java"

    register_job = next(j for j in jobs if j.template_id == "fabric/item/register_basic")
    assert register_job.target_path == "src/main/java/com/foo/space/registry/ModItems.java"
    assert register_job.produces == (
        "raw_lunite.registry_id",
        "raw_lunite.java_symbol",
    )

    model_job = next(j for j in jobs if j.template_id == "minecraft/resource/item/model_generated")
    assert model_job.requires == ("raw_lunite.registry_id",)

    stack_job = next(j for j in jobs if j.template_id == "fabric/item/settings_max_stack")
    assert stack_job.requires == ("raw_lunite.java_symbol",)
    assert stack_job.deterministic_inputs["stack_limit"] == 16


def test_expand_facts_for_block_and_drop():
    facts = [
        PromptFact(
            fact_id="fact_001",
            fact_type=FactType.BLOCK_EXISTS,
            subject="lunite_ore",
            source_clause="lunite ore block exists",
        ),
        PromptFact(
            fact_id="fact_002",
            fact_type=FactType.BLOCK_DROP,
            subject="lunite_ore",
            object="raw_lunite",
            source_clause="drops raw lunite",
        ),
    ]

    jobs = expand_facts_to_jobs(facts, mod_id="space", package_name="com.foo.space")

    template_ids = [j.template_id for j in jobs]
    assert "fabric/block/key" in template_ids
    assert "fabric/block/register_basic" in template_ids
    assert "minecraft/resource/block/blockstate_simple" in template_ids
    assert "minecraft/resource/block/model_cube_all" in template_ids
    assert "fabric/block/lang_en" in template_ids
    assert "fabric/block/initializer" in template_ids
    assert "fabric/loot/block_drop" in template_ids


def test_missing_numeric_fact_is_not_replaced_by_magic_default():
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_FACT_VALUE_REQUIRED"):
        expand_facts_to_jobs(
            [
                PromptFact(
                    fact_id="fact_001",
                    fact_type=FactType.ITEM_STACK_LIMIT,
                    subject="raw_lunite",
                    value=None,
                    source_clause="make a special stack size",
                )
            ],
            mod_id="space",
            package_name="com.foo.space",
        )


def test_unimplemented_fact_fails_closed_instead_of_referencing_missing_template():
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_FACT_UNSUPPORTED"):
        expand_facts_to_jobs(
            [
                PromptFact(
                    fact_id="fact_001",
                    fact_type=FactType.ITEM_DURABILITY,
                    subject="raw_lunite",
                    value=100,
                    source_clause="durability is 100",
                )
            ],
            mod_id="space",
            package_name="com.foo.space",
        )


def test_invalid_registry_subject_is_rejected_before_render():
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_SUBJECT_INVALID"):
        expand_facts_to_jobs(
            [
                PromptFact(
                    fact_id="fact_001",
                    fact_type=FactType.ITEM_EXISTS,
                    subject="Raw Lunite",
                )
            ],
            mod_id="space",
            package_name="com.foo.space",
        )


def test_catalog_drives_targets_dependencies_and_scoped_ports(monkeypatch):
    import minecraft_mod_ai.artifact_expansion as expansion
    original = expansion.load_template

    def load(identifier):
        template = original(identifier)
        if identifier == "minecraft/resource/item/model_generated":
            template["target"]["file"] = "src/main/resources/assets/{{mod_id}}/models/item/custom_{{subject}}.json"
            template["dependencies"] = ["{{subject}}.java_symbol"]
            template["produces"][0]["binding"] = "{{subject}}.custom_model"
        return template

    monkeypatch.setattr(expansion, "load_template", load)
    jobs = expand_facts_to_jobs(
        [PromptFact(fact_id="one", fact_type=FactType.ITEM_EXISTS, subject="widget")],
        mod_id="sample", package_name="org.sample",
    )
    model = next(job for job in jobs if job.template_id == "minecraft/resource/item/model_generated")
    assert model.target_path.endswith("models/item/custom_widget.json")
    assert model.requires == ("widget.java_symbol",)
    assert model.produces == ("widget.custom_model",)


def test_block_drop_cannot_invent_a_drop_target():
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_DROP_TARGET_REQUIRED"):
        expand_facts_to_jobs(
            [PromptFact(fact_id="drop", fact_type=FactType.BLOCK_DROP, subject="ore")],
            mod_id="sample", package_name="org.sample",
        )
