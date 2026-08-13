from __future__ import annotations

import hashlib

import minecraft_mod_ai.agentic_research_game_design as agentic
from minecraft_mod_ai.minecraft_knowledge_contract import (
    compile_minecraft_knowledge_plan,
    evaluate_route_coverage,
)


def _ids(plan: dict) -> set[str]:
    return {str(item["knowledge_id"]) for item in plan["requirements"]}


def _query_sha(query: str) -> str:
    return "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()


def _fake_research(plan: dict) -> dict:
    domains = list(plan["research_domains"])
    return {
        "research_brief": {"domains": domains},
        "deterministic": {
            "forced_project_rag": {
                "domains": [
                    {
                        "domain_id": domain["domain_id"],
                        "queries": [
                            {"query": query, "query_sha256": _query_sha(query)}
                            for query in domain["queries"]
                        ],
                    }
                    for domain in domains
                ]
            }
        },
        "domain_notes": [
            {
                "domain_id": domain["domain_id"],
                "claims": [],
                "gaps": [],
                "next_queries": [],
                "sufficient": True,
            }
            for domain in domains
        ],
    }


def test_boss_expands_host_owned_minecraft_dependency_graph() -> None:
    plan = compile_minecraft_knowledge_plan(
        "체력바와 특수 공격이 있는 보스 몬스터를 추가하고 드롭 아이템과 소리도 넣어줘."
    )
    ids = _ids(plan)
    assert {
        "platform.fabric_target",
        "platform.mappings",
        "project.structure",
        "registry.content",
        "entity.registration",
        "entity.attributes",
        "entity.ai_goals",
        "entity.tracked_data",
        "entity.spawn",
        "rendering.entity",
        "combat.damage",
        "boss.bossbar",
        "datagen.loot",
        "resources.sound",
        "custom.source_extension",
        "quality.compile",
        "quality.gametest",
        "quality.runtime",
        "release.packaging",
    } <= ids
    assert "item.registration" not in ids

    order = {value: index for index, value in enumerate(plan["knowledge_order"])}
    for item in plan["requirements"]:
        for dependency in item["depends_on"]:
            assert order[dependency.removeprefix("mk:")] < order[item["knowledge_id"]]

    mcp = {item["capability"] for item in plan["mcp_requirements"]}
    assert {"source_search", "mapping_resolution", "registry_lookup", "runtime_inspection"} <= mcp


def test_machine_gui_expands_container_network_and_persistence_dependencies() -> None:
    plan = compile_minecraft_knowledge_plan(
        "인벤토리가 있는 기계를 만들고 GUI 버튼을 누르면 서버에서 작동하게 해줘."
    )
    ids = _ids(plan)
    assert "machine_with_gui" in plan["features"]
    assert {
        "block.registration",
        "block_entity.lifecycle",
        "data.persistence",
        "inventory.container",
        "ui.screen_handler",
        "ui.client_screen",
        "networking.payloads",
        "networking.state_sync",
    } <= ids


def test_plain_hud_does_not_force_block_entity_inventory_or_networking() -> None:
    plan = compile_minecraft_knowledge_plan("HUD에 스태미나 바를 표시해줘.")
    ids = _ids(plan)
    assert "ui.hud" in ids
    assert "block_entity.lifecycle" not in ids
    assert "inventory.container" not in ids
    assert "ui.screen_handler" not in ids
    assert "networking.payloads" not in ids


def test_plain_entity_does_not_force_mob_ai_persistence_or_custom_packets() -> None:
    plan = compile_minecraft_knowledge_plan("새 엔티티를 추가하고 렌더링해줘.")
    ids = _ids(plan)
    assert {"entity.registration", "entity.tracked_data", "rendering.entity"} <= ids
    assert "entity.ai_goals" not in ids
    assert "entity.attributes" not in ids
    assert "data.persistence" not in ids
    assert "networking.payloads" not in ids


def test_unknown_mechanic_has_source_mapping_project_and_runtime_fallback() -> None:
    plan = compile_minecraft_knowledge_plan("완전히 새로운 특이한 게임 기믹을 추가해줘.")
    ids = _ids(plan)
    assert "custom_java" in plan["features"]
    assert {"custom.source_extension", "quality.gametest", "quality.runtime"} <= ids
    custom = next(
        item for item in plan["requirements"]
        if item["knowledge_id"] == "custom.source_extension"
    )
    assert custom["evidence"]["rag_queries"]
    assert {"source_search", "mapping_resolution", "mod_examples"} <= set(
        custom["evidence"]["mcp_capabilities"]
    )


def test_common_mod_features_have_dedicated_contracts() -> None:
    plan = compile_minecraft_knowledge_plan(
        "커스텀 상태 효과와 인챈트를 추가하고 주민 거래, 게임룰, 광석 생성도 넣어줘."
    )
    assert {
        "effects.status",
        "enchantment.custom",
        "villager.trade",
        "rules.gamerule",
        "worldgen.feature",
    } <= _ids(plan)


def test_pre_design_knowledge_does_not_pin_version_before_platform_resolution() -> None:
    plan = compile_minecraft_knowledge_plan(
        "Minecraft 1.21.1 Fabric에서 새 보스 몬스터를 추가해줘."
    )
    routed_text = "\n".join(
        [
            str(item["objective"])
            for item in plan["requirements"]
        ]
        + [
            str(query)
            for domain in plan["research_domains"]
            for query in domain["queries"]
        ]
    )
    assert "1.20.1" not in routed_text
    assert "1.21.1" not in routed_text
    assert "Java 17" not in routed_text
    assert "host-resolved" in routed_text


def test_runtime_normalizer_injects_minecraft_domains_before_forced_rag() -> None:
    assert getattr(agentic.normalize_research_brief, "_mmm_minecraft_knowledge_contract_v2", False)
    brief = agentic.normalize_research_brief(
        "보스 몬스터를 추가해줘.",
        {"title": "pre-design research"},
    )
    domain_ids = {str(item["domain_id"]) for item in brief["domains"]}
    assert "request" in domain_ids
    assert {"mk_entity", "mk_platform", "mk_project", "mk_quality"} <= domain_ids


def test_route_coverage_requires_actual_forced_rag_and_sufficient_research() -> None:
    plan = compile_minecraft_knowledge_plan("새 보스 몬스터를 추가해줘.")
    research = _fake_research(plan)
    coverage = evaluate_route_coverage(plan, research)
    assert coverage["status"] == "PASS"
    assert not coverage["blocking_requirement_refs"]
    assert all(item["status"] == "ROUTES_EXECUTED" for item in coverage["domains"])

    missing_domain = plan["research_domains"][0]["domain_id"]
    research["deterministic"]["forced_project_rag"]["domains"] = [
        item
        for item in research["deterministic"]["forced_project_rag"]["domains"]
        if item["domain_id"] != missing_domain
    ]
    blocked = evaluate_route_coverage(plan, research)
    assert blocked["status"] == "BLOCK"
    assert blocked["blocking_requirement_refs"]

    research = _fake_research(plan)
    research["domain_notes"][0]["sufficient"] = False
    unresolved = evaluate_route_coverage(plan, research)
    assert unresolved["status"] == "BLOCK"
    assert any(item["status"] == "RESEARCH_UNRESOLVED" for item in unresolved["domains"])



def test_route_coverage_accepts_terminal_fixed_point_with_deferred_gaps() -> None:
    plan = compile_minecraft_knowledge_plan("새 보스 몬스터를 추가해줘.")
    research = _fake_research(plan)
    domain = plan["research_domains"][0]
    note = next(
        item for item in research["domain_notes"]
        if item["domain_id"] == domain["domain_id"]
    )
    note["sufficient"] = False
    note["fixed_point"] = True

    coverage = evaluate_route_coverage(plan, research)

    assert coverage["status"] == "PASS"
    assert not coverage["blocking_requirement_refs"]
    assert set(domain["requirements"]) <= set(coverage["deferred_requirement_refs"])
    receipt = next(
        item for item in coverage["domains"]
        if item["domain_id"] == domain["domain_id"]
    )
    assert receipt["status"] == "ROUTES_EXECUTED_WITH_GAPS"
    assert receipt["research_agent_sufficient"] is False
    assert receipt["research_agent_fixed_point"] is True
