from __future__ import annotations

from minecraft_mod_ai.minecraft_knowledge_contract import (
    compile_minecraft_knowledge_plan,
    compact_plan,
)
from tests.planning_authority_fixtures import request_catalog


def test_broad_authored_request_survives_feature_routing_without_keyword_patches() -> None:
    prompt = (
        "약한 몹부터 강한 몹과 보스를 만들어줘.\n"
        "아이템/장비/드롭을 넣고 경험치/레벨업/성장 시스템도 만들어줘.\n"
        "장비 강화 시스템도 필요해."
    )
    requirements = [
        {
            "requirement_id": "req_combat_progression",
            "capability": "combat.enemy_progression",
            "statement": "약한 몹부터 강한 몹과 보스를 만든다.",
            "source_text": "약한 몹부터 강한 몹과 보스를 만들어줘.",
        },
        {
            "requirement_id": "req_items_growth",
            "capability": "progression.items_and_growth",
            "statement": "아이템/장비/드롭과 경험치/레벨업/성장 시스템을 만든다.",
            "source_text": "아이템/장비/드롭을 넣고 경험치/레벨업/성장 시스템도 만들어줘.",
        },
        {
            "requirement_id": "req_enhancement",
            "capability": "progression.equipment_enhancement",
            "statement": "장비 강화 시스템을 만든다.",
            "source_text": "장비 강화 시스템도 필요해.",
        },
    ]
    catalog_fixture = request_catalog(prompt, requirements)

    plan = compile_minecraft_knowledge_plan(
        prompt,
        {"_evidence_request_catalog": catalog_fixture},
    )
    catalog = plan["authored_request_catalog"]
    authored = plan["authored_requirements"]

    rendered = "\n".join(item["statement"] for item in authored)
    for authored_term in (
        "약한 몹",
        "강한 몹",
        "보스",
        "아이템",
        "장비",
        "드롭",
        "경험치",
        "레벨업",
        "성장",
        "강화",
    ):
        assert authored_term in rendered

    assert len(authored) == len(catalog["requirements"]) == 3
    assert all(item["state"] == "PRESERVED_FOR_RESEARCH" for item in authored)
    assert plan["policy"]["request_completeness_owner"] == "evidence_request_catalog"
    assert plan["policy"]["feature_detection_role"] == "routing_hint_only"
    assert plan["policy"]["authored_requirements_may_be_dropped"] is False

    for item in authored:
        span = item["source_span"]
        assert prompt[span["char_start"] : span["char_end"]] == span["text"]
        assert span["text_sha256"].startswith("sha256:")

    bounded = compact_plan(plan)
    assert bounded["authored_requirements"] == authored
    assert bounded["authored_request_catalog_sha256"] == catalog["catalog_sha256"]
