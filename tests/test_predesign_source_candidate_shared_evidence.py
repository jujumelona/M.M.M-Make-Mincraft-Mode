"""Production handoff between grounded pre-design records and donor inspection."""
from __future__ import annotations

from minecraft_mod_ai.research_reuse_candidates import (
    authored_design_repository_candidates,
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


def test_authored_reuse_resolver_accepts_host_stamped_research_candidates(monkeypatch):
    from minecraft_mod_ai import authored_reuse_bridge, grounded_source_reuse

    captured = []

    def capture(design):
        captured.append(design)
        return {"schema_version": "mmm/grounded-repository-reuse-plan-v2"}

    monkeypatch.setattr(grounded_source_reuse, "build_repository_reuse_plan", capture)
    receipts = _projected()
    outcome = authored_reuse_bridge.resolve_authored_source_reuse(
        "Build a spaceship travel mod", {},
        minecraft_version="26.1.2", loader="fabric",
        repository_candidates=receipts,
    )
    assert outcome["schema_version"] == "mmm/grounded-repository-reuse-plan-v2"
    assert len(captured) == 1
    assert captured[0]["_planning_state"]["repository_candidates"] == receipts
    assert captured[0]["_platform_selection"]["target"]["minecraft_version"] == "26.1.2"


def test_repository_candidate_handoff_rejects_unstamped_rows():
    polluted = {
        "_planning_state": {"repository_candidates": [
            {"repository": "example/unsafe",
             "origin": "model_claim",
             "reference_only": False,
             "source_reuse_authority": "compile_verified"},
        ]}
    }
    assert planning_state_repository_cards(polluted) == ()


def test_authored_content_design_combines_research_and_planning_without_duplication():
    rows = _projected()
    design = {
        "_planning_state": {"repository_candidates": rows},
        "_pre_design_research": {
            "deterministic": {"repository_candidates": rows},
        },
    }
    collected = authored_design_repository_candidates(design)
    assert len(collected) == 1
    assert len(collected[0]["catalog_evidence"]) == 2
    assert collected[0]["reference_only"] is True
    assert authored_design_repository_candidates({}) == []


def test_content_design_does_not_accept_arbitrary_unverified_candidate_authority():
    rows = [{"repository": "example/unsafe", "origin": "model_claim",
             "reference_only": False, "source_reuse_authority": "compile_verified"}]
    assert authored_design_repository_candidates(
        {"_pre_design_research": {"deterministic": {"repository_candidates": rows}}}
    ) == []
