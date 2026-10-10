from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_expansion import ArtifactExpansionError, expand_facts_to_jobs
from minecraft_mod_ai.content_design_contract import (
    PRIMARY_CONTENT_KINDS,
    relation_type_supported_for_content_pair,
)
from minecraft_mod_ai.content_design_graph import SlotFillError, _lower_drop_relation
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact


def _edge(source: str, target: str) -> dict:
    return {
        "source_id": source,
        "target_id": target,
        "relation_type": "drops",
        "parent_requirement": "req_space_loot",
    }


def test_loot_is_not_a_model_authored_primary_content_kind() -> None:
    # An interstellar UI event must not become an object-less ENTITY_LOOT.
    assert "entity_loot" not in PRIMARY_CONTENT_KINDS
    assert relation_type_supported_for_content_pair("drops", "entity", "item")
    assert not relation_type_supported_for_content_pair("drops", "entity", "entity_loot")


def test_entity_drop_compiles_to_entity_loot_with_registered_item_id() -> None:
    capabilities = {
        "space_alien": FactType.ENTITY_EXISTS,
        "alien_crystal": FactType.ITEM_EXISTS,
    }
    fact = _lower_drop_relation(_edge("space_alien", "alien_crystal"), capabilities, [])
    assert fact.fact_type is FactType.ENTITY_LOOT
    assert fact.subject == "space_alien"
    assert fact.object == "alien_crystal"
    assert fact.fact_id == "space_alien.drop"


def test_block_drop_compiles_as_block_drop_not_entity_loot() -> None:
    capabilities = {
        "moon_ore": FactType.BLOCK_EXISTS,
        "moon_dust": FactType.ITEM_EXISTS,
    }
    fact = _lower_drop_relation(_edge("moon_ore", "moon_dust"), capabilities, [])
    assert fact.fact_type is FactType.BLOCK_DROP


def test_loot_relations_reject_non_item_targets_and_duplicate_drop() -> None:
    capabilities = {
        "space_alien": FactType.ENTITY_EXISTS,
        "phantom_loot": FactType.ENTITY_LOOT,
        "alien_crystal": FactType.ITEM_EXISTS,
    }
    with pytest.raises(SlotFillError, match="CONTENT_RELATION_UNSUPPORTED"):
        _lower_drop_relation(_edge("space_alien", "phantom_loot"), capabilities, [])
    existing = _lower_drop_relation(_edge("space_alien", "alien_crystal"), capabilities, [])
    with pytest.raises(SlotFillError, match="CONTENT_DROP_CONFLICT"):
        _lower_drop_relation(_edge("space_alien", "alien_crystal"), capabilities, [existing])


def test_replay_invalid_space_ui_entity_loot_fails_before_materialization() -> None:
    # The user-provided proposal previously reached this stage with object=None.
    bad = PromptFact(
        fact_id="interstellar_unlock_event_ui_screen_content_renderer_gui_screen_.exists",
        fact_type=FactType.ENTITY_LOOT,
        subject="interstellar_unlock_event_ui_screen_content_renderer_gui_screen_",
        object=None,
    )
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_DROP_TARGET_REQUIRED"):
        expand_facts_to_jobs([bad], mod_id="space", package_name="org.space")


def test_unregistered_loot_item_and_missing_entity_owner_are_rejected() -> None:
    entity_loot = PromptFact(
        fact_id="alien.drop",
        fact_type=FactType.ENTITY_LOOT,
        subject="space_alien",
        object="alien_crystal",
    )
    item = PromptFact(
        fact_id="crystal.exists",
        fact_type=FactType.ITEM_EXISTS,
        subject="alien_crystal",
    )
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_DROP_TARGET_UNREGISTERED"):
        expand_facts_to_jobs([entity_loot], mod_id="space", package_name="org.space")
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_DROP_OWNER_UNREGISTERED"):
        expand_facts_to_jobs([item, entity_loot], mod_id="space", package_name="org.space")


def test_fully_declared_entity_loot_expands_with_bound_item() -> None:
    facts = [
        PromptFact(fact_id="alien.exists", fact_type=FactType.ENTITY_EXISTS, subject="space_alien"),
        PromptFact(fact_id="crystal.exists", fact_type=FactType.ITEM_EXISTS, subject="alien_crystal"),
        PromptFact(fact_id="alien.drop", fact_type=FactType.ENTITY_LOOT, subject="space_alien", object="alien_crystal"),
    ]
    jobs = expand_facts_to_jobs(facts, mod_id="space", package_name="org.space")
    loot_jobs = [job for job in jobs if job.template_id == "fabric/loot/entity_drop"]
    assert loot_jobs
    assert all(job.deterministic_inputs["drop_item"] == "alien_crystal" for job in loot_jobs)
