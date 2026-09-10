from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact


def test_expand_facts_for_item_and_stack_limit():
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

    # Item exists generates register_basic, model_basic, lang_en
    # Stack limit generates settings_max_stack
    assert len(jobs) == 4
    template_ids = [j.template_id for j in jobs]
    assert "fabric/item/register_basic" in template_ids
    assert "fabric/item/model_basic" in template_ids
    assert "fabric/item/lang_en" in template_ids
    assert "fabric/item/settings_max_stack" in template_ids

    # Check deterministic inputs and targets
    register_job = next(j for j in jobs if j.template_id == "fabric/item/register_basic")
    assert register_job.target_path == "src/main/java/com/foo/space/registry/ModItems.java"
    assert register_job.deterministic_inputs["mod_id"] == "space"
    assert register_job.deterministic_inputs["java_constant"] == "RAW_LUNITE"
    assert register_job.deterministic_inputs["registry_path"] == "raw_lunite"

    settings_job = next(j for j in jobs if j.template_id == "fabric/item/settings_max_stack")
    assert settings_job.deterministic_inputs["stack_limit"] == 16


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
    assert "fabric/block/register_basic" in template_ids
    assert "fabric/block/blockstate_basic" in template_ids
    assert "fabric/block/model_cube_all" in template_ids
    assert "fabric/block/lang_en" in template_ids
    assert "fabric/loot/block_drop" in template_ids

    loot_job = next(j for j in jobs if j.template_id == "fabric/loot/block_drop")
    assert loot_job.deterministic_inputs["drop_item"] == "raw_lunite"
    assert "lunite_ore.block_registry_id" in loot_job.requires
    assert "raw_lunite.registry_id" in loot_job.requires
