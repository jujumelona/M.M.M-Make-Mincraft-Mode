"""Regression cases for Korean-language source discovery and donor ranking.

These fixtures intentionally use synthetic provider records; they do not
assert that the named projects exist or are compatible with the target.
"""
from minecraft_mod_ai.source_intent_aliases import source_intent_terms, source_query_variants
from minecraft_mod_ai.research_reuse_candidates import project_repository_candidates
from minecraft_mod_ai.grounded_source_reuse import _candidate_overlap


def test_korean_space_combat_search_expands_to_english():
    variants = source_query_variants("우주선 만들어서 외계인과 전투하는 모드")
    combined = " ".join(variants)
    assert "spaceship" in combined
    assert "alien" in combined
    assert "combat" in combined


def test_actual_evidence_outweighs_provider_order_for_bilingual_intent():
    domain = {
        "domain_id": "space_combat",
        "evidence_kinds": ["source_code"],
        "objective": "우주에서 외계인과 싸우는 모드",
        "requirements": ["외계인 전투"],
    }
    records = [
        {
            "source_id": "github:example/alien-pets",
            "title": "Alien Pets",
            "content": "Tame friendly alien creatures and collect pets.",
            "url": "https://github.com/example/alien-pets",
        },
        {
            "source_id": "github:example/space-combat",
            "title": "Space Combat",
            "content": "Fight alien invaders in spaceship battles.",
            "url": "https://github.com/example/space-combat",
        },
    ]
    grounded = {"queries": [{"query": "alien", "evidence_records": records}]}
    ranked = project_repository_candidates(domain, grounded)
    assert [r["repository"] for r in ranked][:2] == [
        "example/space-combat", "example/alien-pets"
    ]


def test_reference_overlap_uses_actual_source_not_requested_feature():
    capability = "우주선 외계인 전투"
    terms = ("우주선", "외계인", "전투")
    pets = {"evidence_text": "Alien Pets tame friendly creatures"}
    combat = {"evidence_text": "Spaceship combat against alien invaders"}
    assert _candidate_overlap(capability, terms, combat) > _candidate_overlap(
        capability, terms, pets
    )


def test_aliases_do_not_claim_a_source_is_verified():
    assert {"alien", "combat"} <= set(source_intent_terms("외계인과 싸우는 모드"))
