from __future__ import annotations

from copy import deepcopy

import minecraft_mod_ai.agentic_research_game_design as agentic
import minecraft_mod_ai.pre_design_phase_contract as phase


def test_pre_design_does_not_expand_post_design_obligation_domains() -> None:
    prompt = (
        "자원을 모아 화폐를 얻고 거래해서 우주선 부품을 만들고 조립한 뒤 "
        "무기를 사고 우주선 성능을 업그레이드한다."
    )

    brief = agentic.normalize_research_brief(
        prompt,
        {"title": "pre-design research"},
    )

    domains = brief["domains"]
    assert [domain["domain_id"] for domain in domains] == ["pre_design_request"]
    providers = set(domains[0]["providers"])
    assert "official_docs" in providers
    assert "project_rag" in providers
    assert "modrinth" not in providers
    assert "github" not in providers
    assert all(not domain["domain_id"].startswith("obl_") for domain in domains)


def test_pre_design_ecosystem_donor_search_is_deferred() -> None:
    brief = phase._pre_design_candidate("우주선 모드를 설계해줘")
    result = agentic.collect_ecosystem_seed_bundle(
        "우주선 모드를 설계해줘",
        {},
        research_brief=brief,
        route_limit=12,
        page_builder=lambda *args, **kwargs: None,
        planning_seed_only=True,
    )

    assert result["status"] == "deferred_until_design_freeze"
    assert result["candidate_count"] == 0


def test_research_transport_schema_matches_host_normalization_surface() -> None:
    schema = deepcopy(agentic._RESEARCH_NOTE_SCHEMA)
    assert "required" not in schema
    assert schema["additionalProperties"] is True

    note = schema["properties"]["research_note"]
    assert "required" not in note
    assert note["additionalProperties"] is True
    assert "maxItems" not in note["properties"]["claims"]
    assert note["properties"]["claims"]["items"] == {}
    assert "maxItems" not in note["properties"]["gaps"]
    assert "maxItems" not in note["properties"]["next_queries"]


def test_host_parser_accepts_compact_qwen_claim_variants() -> None:
    raw = (
        '{"claims":["Fabric API 근거"],"gaps":[],"next_queries":[],'
        '"sufficient":true}'
    )
    note = agentic._parse_research_note(raw, "pre_design_request")

    assert note["domain_id"] == "pre_design_request"
    assert note["claims"] == [{"claim": "Fabric API 근거", "evidence_refs": []}]
    assert note["sufficient"] is True
