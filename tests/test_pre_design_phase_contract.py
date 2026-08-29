from __future__ import annotations

import minecraft_mod_ai.agentic_research_game_design as agentic
import minecraft_mod_ai.pre_design_research_pipeline as pipeline


def test_pre_design_does_not_expand_post_design_obligation_domains() -> None:
    prompt = (
        "자원을 모아 화폐를 얻고 거래해서 우주선 부품을 만들고 조립한 뒤 "
        "무기를 사고 우주선 성능을 업그레이드한다."
    )

    brief = pipeline._pre_design_brief(prompt)

    domains = brief["domains"]
    assert [domain["domain_id"] for domain in domains] == ["request"]
    providers = set(domains[0]["providers"])
    assert "official_docs" in providers
    assert "project_rag" in providers
    assert "modrinth" not in providers
    assert "github" not in providers
    assert all(not domain["domain_id"].startswith("obl_") for domain in domains)


def test_pre_design_donor_search_is_deferred_by_the_owning_pipeline() -> None:
    brief = pipeline._pre_design_brief("우주선 모드를 설계해줘")
    providers = {
        provider
        for domain in brief["domains"]
        for provider in domain.get("providers", [])
    }

    assert "modrinth" not in providers
    assert "github" not in providers
    assert set(pipeline._DETERMINISTIC_STAGES) == {
        "official_rag",
        "technology_radar",
        "forced_project_rag",
    }


def test_research_transport_schema_matches_host_normalization_surface() -> None:
    schema = agentic._RESEARCH_NOTE_SCHEMA
    assert "required" not in schema
    assert schema["additionalProperties"] is True

    note = schema["properties"]["research_note"]
    assert "required" not in note
    assert note["additionalProperties"] is True
    assert note["properties"]["claims"]["items"] == {}
    assert note["properties"]["gaps"]["items"] == {}
    assert note["properties"]["next_queries"]["items"] == {}
    assert note["properties"]["procedures"]["items"] == {}


def test_host_parser_accepts_compact_qwen_claim_variants() -> None:
    raw = (
        '{"claims":["Fabric API 근거"],"gaps":[],"next_queries":[],'
        '"sufficient":true}'
    )
    note = agentic._parse_research_note(raw, "request")

    assert note["domain_id"] == "request"
    assert note["claims"] == [{"claim": "Fabric API 근거", "evidence_refs": []}]
    assert note["sufficient"] is True
    assert note["procedures"] == []
