from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_authority import build_authoritative_request_catalog
from minecraft_mod_ai.planning_state_contract import build_initial_planning_state
from minecraft_mod_ai.planning_state_research import _compile_queries, _research_brief


class _Router:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


def test_reference_prompt_creates_reference_and_scope_research_without_mod_guessing() -> None:
    prompt = "메이플스토리 모드 만들어줘"
    router = _Router(
        [
            {
                "goal": {"statement": "메이플스토리를 참조한 Minecraft 모드를 만든다", "source_quote": prompt},
                "known": [],
                "references": [
                    {
                        "name": "메이플스토리",
                        "source_quote": "메이플스토리",
                        "what_must_be_learned": "실제 게임을 구성하는 문서화된 핵심 시스템과 규칙",
                    }
                ],
                "scope_status": "unspecified",
                "unresolved": [],
            }
        ]
    )

    state = build_initial_planning_state(router, prompt)

    assert state["original_prompt"] == prompt
    assert state["references"][0]["name"] == "메이플스토리"
    reasons = {item["reason"] for item in state["unresolved"]}
    assert reasons == {"reference_semantics", "scope"}
    reference = next(item for item in state["unresolved"] if item["reason"] == "reference_semantics")
    scope = next(item for item in state["unresolved"] if item["reason"] == "scope")
    assert reference["resolution_route"] == "reference_research"
    assert reference["source_kinds"] == ["reference_sources", "web_sources"]
    assert scope["resolution_route"] == "default_policy"
    assert state["plan_ready"] is False
    assert all(item["queries"] == [] for item in state["research_queue"])


def test_model_cannot_route_scope_to_user_only_to_bypass_research_policy() -> None:
    prompt = "무언가 큰 모드 만들어줘"
    router = _Router(
        [
            {
                "goal": {"statement": "큰 모드를 만든다", "source_quote": prompt},
                "known": [],
                "references": [],
                "scope_status": "unspecified",
                "unresolved": [
                    {
                        "question": "범위가 무엇인가",
                        "reason": "scope",
                        "blocks": ["requirement_selection"],
                        "information_needed": "구현 범위",
                        "resolution_route": "user_only",
                        "source_kinds": [],
                    }
                ],
            }
        ]
    )

    state = build_initial_planning_state(router, prompt)
    scope = next(item for item in state["unresolved"] if item["reason"] == "scope")
    assert scope["resolution_route"] == "default_policy"
    assert scope["source_kinds"] == []
    assert scope["status"] == "resolved"
    assert any(
        item.get("unresolved_id") == scope["unresolved_id"]
        and item.get("basis") == "host_default_policy"
        for item in state["resolved"]
    )


def test_reference_query_compiler_is_explicitly_target_neutral() -> None:
    router = _Router([{"queries": ["메이플스토리 gameplay systems"]}])
    state = {"references": [{"name": "메이플스토리"}]}
    research = {
        "objective": "메이플스토리의 실제 시스템을 조사한다",
        "information_needed": "문서화된 게임 시스템과 상호 관계",
        "source_kinds": ["reference_sources", "web_sources"],
    }

    queries = _compile_queries(router, state, research)

    assert queries == ["메이플스토리 gameplay systems"]
    system = router.calls[0][1][0]["content"]
    assert "never turn it into a '<name> Minecraft mod' query" in system


def test_reference_and_implementation_research_use_different_validated_routes() -> None:
    reference_state = {
        "unresolved": [{"question": "reference?", "status": "open"}],
        "research_queue": [
            {
                "research_id": "r_001",
                "objective": "research reference",
                "information_needed": "reference behavior",
                "source_kinds": ["reference_sources", "web_sources"],
                "queries": ["MapleStory gameplay systems"],
                "status": "pending",
            }
        ],
    }
    reference_brief, reference_ids = _research_brief("MapleStory mod", reference_state)
    domain = reference_brief["domains"][0]
    assert reference_ids == {"r_001"}
    assert "gameplay_reference" in domain["evidence_kinds"]
    assert "wikipedia" in domain["providers"]

    implementation_state = {
        "unresolved": [{"question": "implementation?", "status": "open"}],
        "research_queue": [
            {
                "research_id": "r_002",
                "objective": "research implementation",
                "information_needed": "actual Minecraft implementation and reuse",
                "source_kinds": [
                    "repository",
                    "existing_mods",
                    "minecraft_docs",
                    "minecraft_source",
                    "project_rag",
                ],
                "queries": ["Minecraft persistent progression implementation"],
                "status": "pending",
            }
        ],
    }
    implementation_brief, reference_ids = _research_brief("persistent progression", implementation_state)
    domain = implementation_brief["domains"][0]
    assert reference_ids == set()
    assert {"source_code", "minecraft_api", "local_project"}.issubset(domain["evidence_kinds"])
    assert {"github", "modrinth", "curseforge", "official_docs", "project_rag"}.issubset(domain["providers"])


def test_authoritative_catalog_has_no_raw_prompt_fallback() -> None:
    with pytest.raises(ValueError, match="PLANNING_AUTHORITY_STATE_REQUIRED"):
        build_authoritative_request_catalog("메이플스토리 모드 만들어줘")
