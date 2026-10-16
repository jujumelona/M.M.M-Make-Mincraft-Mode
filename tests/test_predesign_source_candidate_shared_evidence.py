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


def test_actual_predesign_request_domain_is_source_code_eligible():
    from minecraft_mod_ai.pre_design_research_pipeline import _pre_design_brief

    brief = _pre_design_brief("Create a spaceship with planet travel")
    request = next(row for row in brief["domains"] if row["domain_id"] == "request")
    assert "minecraft_api" in request["evidence_kinds"]
    assert "curseforge" in request["providers"]
    assert "modrinth" in request["providers"]
    assert "github" not in request["providers"]
    grounded = {"queries": [{
        "query": "planet travel", "evidence_records": [{
            "source_id": "curseforge:42",
            "title": "Planet Travel",
            "content": "Spaceship launch and planet travel",
            "metadata": {"source_url": "https://github.com/example/planet-source"},
        }],
    }]}
    projected = project_repository_candidates(request, grounded)
    assert [row["repository"] for row in projected] == ["example/planet-source"]


def test_donor_cache_fingerprint_changes_with_new_grounded_evidence():
    from minecraft_mod_ai.authored_reuse_bridge import source_candidate_fingerprint

    candidates = _projected()
    before = source_candidate_fingerprint(candidates)
    assert before == source_candidate_fingerprint(_projected())
    updated = [dict(candidates[0])]
    updated[0]["catalog_evidence"] = [
        *updated[0]["catalog_evidence"],
        {"provider": "sourcegraph", "content_sha256": "sha256:new"},
    ]
    assert source_candidate_fingerprint(updated) != before
    assert source_candidate_fingerprint(()) != before


def test_live_handoff_projector_reuses_grounded_catalog_result_without_model(monkeypatch):
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline

    calls = []

    def fake_bundle(_backend, router, research_brief):
        calls.append((router, research_brief))
        return {"domains": [{
            "domain_id": "request",
            "queries": [{
                "query": "Create a spaceship travel mod",
                "external_rag": {"sources": [{
                    "source_id": "curseforge:42",
                    "title": "Planet Travel",
                    "url": "https://www.curseforge.com/minecraft/mc-mods/planet-travel",
                    "content": "Visit other planets with spaceship crafting",
                    "metadata": {
                        "source_url": "https://github.com/example/planet-source",
                    },
                }]},
            }],
        }]}

    monkeypatch.setattr(pipeline, "forced_rag_bundle", fake_bundle)
    receipt = pipeline.collect_source_candidate_handoff(
        object(), "Create a spaceship travel mod",
    )
    assert receipt["model_called"] is False
    assert receipt["candidate_count"] == 1
    assert receipt["repository_candidates"][0]["repository"] == "example/planet-source"
    assert receipt["repository_candidates"][0]["reference_only"] is True
    assert len(calls) == 1


def test_production_handoff_is_collected_once_and_reused_from_content_design(monkeypatch):
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline
    from minecraft_mod_ai.research_reuse_candidates import ensure_authored_source_handoff

    calls = []

    def fake_collect(_router, prompt):
        calls.append(prompt)
        return {
            "schema_version": "mmm/pre-design-source-candidate-handoff-v1",
            "prompt_sha256": pipeline._canonical_sha256(prompt),
            "repository_candidates": _projected(),
            "model_called": False,
        }

    monkeypatch.setattr(pipeline, "collect_source_candidate_handoff", fake_collect)
    original, initial = ensure_authored_source_handoff(
        {}, router=object(), prompt="planet travel",
    )
    restored, again = ensure_authored_source_handoff(
        original, router=object(), prompt="planet travel",
    )
    assert len(calls) == 1
    assert again == initial
    assert restored == original
    assert initial[0]["source_reuse_authority"] == "verification_required"


def test_handoff_discovery_failure_is_not_a_fake_success(monkeypatch):
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline
    from minecraft_mod_ai.research_reuse_candidates import ensure_authored_source_handoff

    def raise_error(_router, _prompt):
        raise OSError("network disconnected")

    monkeypatch.setattr(pipeline, "collect_source_candidate_handoff", raise_error)
    design, candidates = ensure_authored_source_handoff(
        {}, router=object(), prompt="planet travel",
    )
    assert candidates == []
    assert design["_pre_design_source_handoff"]["status"] == "provider_error"
    assert design["_pre_design_source_handoff"]["error_type"] == "OSError"


def test_outdated_prompt_handoff_is_discarded_before_candidate_projection(monkeypatch):
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline
    from minecraft_mod_ai.research_reuse_candidates import ensure_authored_source_handoff

    calls = []

    def replacement(_router, prompt):
        calls.append(prompt)
        return {
            "schema_version": "mmm/pre-design-source-candidate-handoff-v1",
            "prompt_sha256": pipeline._canonical_sha256(prompt),
            "repository_candidates": [],
            "model_called": False,
        }

    monkeypatch.setattr(pipeline, "collect_source_candidate_handoff", replacement)
    stale = {
        "_pre_design_source_handoff": {
            "schema_version": "mmm/pre-design-source-candidate-handoff-v1",
            "prompt_sha256": pipeline._canonical_sha256("old feature"),
            "repository_candidates": _projected(),
        },
    }
    content, candidates = ensure_authored_source_handoff(
        stale, router=object(), prompt="new unrelated feature",
    )
    assert calls == ["new unrelated feature"]
    assert candidates == []
    assert content["_pre_design_source_handoff"]["repository_candidates"] == []


def test_generic_architecture_background_is_not_nominated_as_a_donor(monkeypatch):
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline

    def fake_bundle(_backend, _router, research_brief):
        return {"domains": [{
            "domain_id": "request",
            "queries": [{
                "query": "A completely new spaceship feature",
                "external_rag": {"sources": []},
            }, {
                "query": "Minecraft Fabric architecture items entities dimensions world interaction networking persistence data components testing patterns",
                "external_rag": {"sources": [{
                    "source_id": "modrinth:unrelated",
                    "title": "Generic Network Utilities",
                    "content": "Utilities for Minecraft networking",
                    "metadata": {"source_url": "https://github.com/example/unrelated"},
                }]},
            }],
        }]}

    monkeypatch.setattr(pipeline, "forced_rag_bundle", fake_bundle)
    receipt = pipeline.collect_source_candidate_handoff(
        object(), "A completely new spaceship feature",
    )
    assert receipt["candidate_count"] == 0
    assert receipt["repository_candidates"] == []


def test_reference_changes_invalidate_cached_source_plan_identity():
    from minecraft_mod_ai.authored_reuse_bridge import (
        source_discovery_algorithm_version,
        source_reference_fingerprint,
    )

    assert source_discovery_algorithm_version()
    assert source_reference_fingerprint(()) == source_reference_fingerprint([])
    assert source_reference_fingerprint(("example/source-one",)) != (
        source_reference_fingerprint(("example/source-two",))
    )


def test_research_receipt_without_matching_prompt_hash_is_not_carried_to_new_request(monkeypatch):
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline
    from minecraft_mod_ai.research_reuse_candidates import ensure_authored_source_handoff

    prompt = "a completely different machine"
    monkeypatch.setattr(pipeline, "collect_source_candidate_handoff", lambda _r, p: {
        "schema_version": "mmm/pre-design-source-candidate-handoff-v1",
        "prompt_sha256": pipeline._canonical_sha256(p),
        "repository_candidates": [],
        "model_called": False,
    })
    stale = {"_pre_design_research": {
        "prompt_sha256": pipeline._canonical_sha256("old spaceship design"),
        "deterministic": {"repository_candidates": _projected()},
    }}
    content, rows = ensure_authored_source_handoff(stale, router=object(), prompt=prompt)
    assert rows == []
    assert "_pre_design_research" not in content
