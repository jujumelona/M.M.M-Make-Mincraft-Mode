from __future__ import annotations

from minecraft_mod_ai.small_model_research_extensions_contract import (
    _memory_route,
    _project_capability_payload,
)


def test_memory_route_scales_only_when_needed() -> None:
    assert _memory_route("find registry id", "general", 6) == ("exact", 3)
    assert _memory_route("fix compile error in registry", "repair", 6) == (
        "targeted",
        6,
    )
    deep_query = "\n".join(
        [
            "cross-file integration dependency chain regression",
            *[f"constraint_{index} symbol_{index}" for index in range(80)],
        ]
    )
    tier, limit = _memory_route(deep_query, "repair", 6)
    assert tier == "deep"
    assert limit >= 8


def test_instruction_projection_keeps_only_frontier_skills() -> None:
    payload = {
        "eligible_skills": [
            {
                "name": "rag",
                "model_tools": ["search_code_rag"],
                "validators": ["evidence"],
            },
            {
                "name": "runtime",
                "model_tools": ["run_playtest"],
                "validators": ["runtime"],
            },
            {
                "name": "host-only",
                "model_tools": [],
                "host_owned_tools": ["package_release"],
            },
        ]
    }
    projected = _project_capability_payload(
        payload,
        exposed_tools=frozenset({"search_code_rag"}),
    )
    assert [item["name"] for item in projected["eligible_skills"]] == ["rag"]
    receipt = projected["instruction_projection"]
    assert receipt["candidate_skill_count"] == 3
    assert receipt["selected_skill_count"] == 1
    assert receipt["policy"] == "prompt_projection_only_authorization_unchanged"
