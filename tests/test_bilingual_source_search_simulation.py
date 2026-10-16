"""Reproduce Korean request -> provider calls -> grounded donor rankings.

All HTTP calls are mock responses, not an invented claim about live availability.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

from minecraft_mod_ai import (
    pre_design_grounded_rag, research_reuse_candidates, reuse_discovery,
    grounded_source_reuse,
)
from minecraft_mod_ai.source_intent_aliases import (
    source_intent_aliases, source_intent_terms, source_query_variants,
)

REQUEST = "우주에서 외계인과 싸우는 모드"


def test_korean_combat_verbs_expand_into_actual_english_feature_queries():
    aliases = source_intent_aliases(REQUEST)
    assert "space" in aliases and "alien" in aliases and "combat" in aliases
    variants = source_query_variants(REQUEST)
    assert variants[0] == "alien space combat"
    assert variants == ("alien space combat", "alien combat", "space combat")
    assert {"space", "alien", "combat"} <= source_intent_terms(REQUEST)


def test_modrinth_provider_mock_finds_combat_source_after_alien_pets(monkeypatch):
    calls = []
    project_rows = {
        "pets": ("Alien Pets", "Tame alien creatures", "example/pets"),
        "combat": ("Space Combat", "Build spacecraft and fight enemies", "example/combat"),
        "invasion": ("Alien Invasion", "Aliens attack player bases", "example/invasion"),
        "explore": ("Spacecraft Exploration", "Explore space and planets", "example/explore"),
    }
    hits_by_phrase = {
        "alien space combat": ["pets", "combat"],
        "alien combat": ["combat", "invasion"],
        "space combat": ["explore", "combat"],
    }

    def fake_json(url, headers=None):
        query = parse_qs(urlsplit(url).query)
        if "/v2/search" in url:
            phrase = query["query"][0]
            calls.append(phrase)
            assert int(query["offset"][0]) == 0
            ids = hits_by_phrase[phrase]
            return {
                "total_hits": len(ids), "offset": 0,
                "hits": [
                    {"project_id": key, "slug": key,
                     "title": project_rows[key][0],
                     "description": project_rows[key][1]}
                    for key in ids
                ],
            }
        assert "/v2/projects" in url
        ids = json.loads(query["ids"][0])
        return [
            {"id": key, "title": project_rows[key][0],
             "body": project_rows[key][1],
             "source_url": "https://github.com/" + project_rows[key][2],
             "game_versions": ["1.21.1"], "loaders": ["fabric"]}
            for key in ids
        ]

    monkeypatch.setattr(pre_design_grounded_rag, "_json", fake_json)
    records, receipt = pre_design_grounded_rag._search_modrinth(REQUEST)
    assert calls == ["alien space combat", "alien combat", "space combat"]
    assert receipt["search_requests"] == 3
    assert len(records) == 4
    assert records[0]["title"] == "Space Combat"
    assert len({r["source_id"] for r in records}) == 4

    domain = {
        "domain_id": "request", "requirements": [REQUEST],
        "queries": [REQUEST], "evidence_kinds": ["minecraft_api"],
        "providers": ["modrinth"],
    }
    candidates = research_reuse_candidates.project_repository_candidates(
        domain, {"queries": [{
            "query": REQUEST,
            "evidence_records": records,
            "query_sha256": "sha256:actual-user-intent",
        }]},
    )
    assert candidates[0]["repository"] == "example/combat"
    assert candidates.index(next(c for c in candidates if c["repository"] == "example/combat")) < (
        candidates.index(next(c for c in candidates if c["repository"] == "example/pets"))
    )
    assert all(candidate["source_reuse_authority"] == "verification_required"
               for candidate in candidates)
    assert grounded_source_reuse._candidate_overlap(
        REQUEST, (REQUEST,), candidates[0],
    ) > grounded_source_reuse._candidate_overlap(
        REQUEST, (REQUEST,),
        next(row for row in candidates if row["repository"] == "example/pets"),
    )


def test_source_discovery_korean_request_credits_actual_combat_not_only_alien():
    pet = {"title": "Alien Pets", "summary": "Tame alien creatures",
           "source_url": "https://github.com/example/pets"}
    combat = {"title": "Space Combat", "summary": "Spaceships fight enemies in space",
              "source_url": "https://github.com/example/combat"}
    assert reuse_discovery._candidate_relevance(combat, REQUEST) > (
        reuse_discovery._candidate_relevance(pet, REQUEST)
    )


def test_hangul_name_does_not_fabricate_relevance_to_english_mod():
    irrelevant = {"title": "Inventory Sorting", "summary": "Organize your chests"}
    assert reuse_discovery._candidate_relevance(irrelevant, REQUEST) == 0.0
