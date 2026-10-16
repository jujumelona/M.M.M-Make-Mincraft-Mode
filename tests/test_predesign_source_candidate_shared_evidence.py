"""Production handoff between grounded pre-design records and donor inspection."""
from __future__ import annotations

from minecraft_mod_ai.research_reuse_candidates import (
    merge_repository_candidates,
    planning_state_repository_cards,
    project_repository_candidates,
)


def _projected():
    domain = {
        "domain_id": "req-space-travel", "evidence_kinds": ["source_code"],
        "objective": "spaceship planet travel", "requirements": ["space travel"],
        "queries": ["planet travel mod"],
    }
    grounded = {"queries": [{
        "query": "planet travel mod", "query_sha256": "sha256:q1",
        "evidence_records": [
            {
                "source_id": "curseforge:42",
                "source_type": "curseforge_mod_body",
                "url": "https://www.curseforge.com/minecraft/mc-mods/planet-travel",
                "title": "Planet Travel",
                "content": "Planet travel and spaceship launch implementation",
                "content_sha256": "sha256:realbody",
                "evidence_origin": "curseforge_mod_body",
                "body_retrieved": True,
                "metadata": {"source_url": "https://github.com/example/planet-source"},
            },
            {
                "source_id": "modrinth:travel",
                "url": "https://modrinth.com/mod/planet-travel",
                "title": "Planet Travel",
                "content": "Players visit other planets",
                "content_sha256": "sha256:modrinthsummary",
                "metadata": {"source_url": "https://github.com/example/planet-source"},
            },
        ],
    }]}
    return project_repository_candidates(domain, grounded)


def test_prealigned_catalog_records_share_one_source_identity_and_provenance():
    candidates = _projected()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["repository"] == "example/planet-source"
    assert set(candidate["source_ids"]) == {"curseforge:42", "modrinth:travel"}
    assert {row["provider"] for row in candidate["catalog_evidence"]} == {
        "curseforge", "modrinth",
    }
    assert any(row["body_retrieved"] for row in candidate["catalog_evidence"])
    assert any(row["content_sha256"] == "sha256:realbody" for row in candidate["catalog_evidence"])
    assert all(row["source_reuse_authority"] == "verification_required"
               for row in candidate["catalog_evidence"])
    assert candidate["reference_only"] is True


def test_existing_state_cards_preserve_research_evidence_for_source_reuse():
    candidates = merge_repository_candidates([], _projected())
    state = {"_planning_state": {"repository_candidates": candidates}}
    cards = planning_state_repository_cards(state)
    assert len(cards) == 1
    card = cards[0]
    assert card["repository"] == "example/planet-source"
    assert len(card["catalog_evidence"]) == 2
    assert card["candidate_score"] > 0
    assert card["source_reuse_authority"] == "verification_required"
    assert card["explicit_reference"] is False
    assert "sha256:q1" in card["page_refs"][0]


def test_metadata_only_cannot_create_a_source_repository_candidate():
    domain = {"domain_id": "req", "evidence_kinds": ["source_code"],
              "objective": "planet travel"}
    grounded = {"queries": [{"query": "planet travel", "evidence_records": [
        {"source_id": "curseforge:999", "title": "Planet Travel",
         "content": "Planet travel", "metadata": {"source_url": ""}},
    ]}]}
    assert project_repository_candidates(domain, grounded) == []
