"""Regression cases for Korean-language source discovery and donor ranking.

These fixtures intentionally use synthetic provider records; they do not
assert that the named projects exist or are compatible with the target.
"""
from minecraft_mod_ai.source_intent_aliases import source_intent_terms, source_query_variants
from minecraft_mod_ai.curseforge_query import curseforge_catalogue_score


def test_korean_space_combat_search_expands_to_english():
    variants = source_query_variants("우주선 만들어서 외계인과 전투하는 모드")
    combined = " ".join(variants)
    assert "spaceship" in combined
    assert "alien" in combined
    assert "combat" in combined


def test_actual_catalogue_evidence_outweighs_provider_order_for_bilingual_intent():
    capability = "우주에서 외계인과 싸우는 모드"
    pets = {"name": "Alien Pets", "summary": "Tame friendly alien creatures"}
    combat = {"name": "Space Combat", "summary": "Fight alien invaders in spaceship battles"}
    assert curseforge_catalogue_score(
        capability, combat
    ) > curseforge_catalogue_score(capability, pets)


def test_reference_overlap_uses_actual_mod_description_not_requested_feature():
    capability = "우주선 외계인 전투"
    pets = {"name": "Alien Pets", "summary": "Tame friendly creatures"}
    combat = {"name": "Spaceship Combat", "summary": "Battle alien invaders"}
    assert curseforge_catalogue_score(
        capability, combat
    ) > curseforge_catalogue_score(capability, pets)


def test_aliases_do_not_claim_a_source_is_verified():
    assert {"alien", "combat"} <= set(source_intent_terms("외계인과 싸우는 모드"))
