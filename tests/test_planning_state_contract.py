from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_authority import build_authoritative_request_catalog
from minecraft_mod_ai.planning_state_contract import MODEL_PARAMETERS, build_initial_planning_state
from minecraft_mod_ai.planning_state_research import _compile_queries, _research_brief
from minecraft_mod_ai.reference_source_research import _wikipedia_languages


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


def test_model_cannot_author_scope_or_route_it_to_bypass_host_policy() -> None:
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
                        "information_needed": "구현 범위",
                    }
                ],
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="PROMPT_STATE_UNRESOLVED: model cannot author reason 'scope'",
    ):
        build_initial_planning_state(router, prompt)


@pytest.mark.parametrize(
    "reason",
    ["repository_fact", "minecraft_api", "implementation_method", "compatibility"],
)
def test_prompt_model_cannot_create_downstream_engineering_unknowns(reason: str) -> None:
    prompt = "우주선을 부품별로 제작하고 업그레이드하는 우주 모드"
    router = _Router(
        [
            {
                "goal": {"statement": prompt},
                "known": [{"statement": "우주선을 부품별로 제작하고 업그레이드한다"}],
                "references": [],
                "scope_status": "partial",
                "unresolved": [
                    {
                        "question": "구현 세부가 무엇인가",
                        "reason": reason,
                        "information_needed": "구현 세부",
                    }
                ],
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match=f"PROMPT_STATE_UNRESOLVED: model cannot author reason '{reason}'",
    ):
        build_initial_planning_state(router, prompt)


def test_prompt_model_schema_cannot_author_blocker_topology() -> None:
    unresolved = MODEL_PARAMETERS["properties"]["unresolved"]["items"]
    assert "blocks" not in unresolved["properties"]
    assert set(unresolved["properties"]["reason"]["enum"]) == {
        "external_fact",
        "contradiction",
        "user_preference",
    }


def test_creative_unspecified_values_do_not_become_prompt_research() -> None:
    prompt = (
        "우주모드: 자원 파밍과 거래로 우주선 부품을 만들고 업그레이드해서 "
        "행성 탐사, 외계인 전투, 식민지화를 할 수 있게 해줘"
    )
    router = _Router(
        [
            {
                "goal": {"statement": "우주 진출과 행성 활동이 가능한 진행형 우주 모드"},
                "known": [
                    {"statement": "자원 파밍과 거래가 있다"},
                    {"statement": "우주선을 부품별로 제작하고 업그레이드한다"},
                    {"statement": "행성 탐사, 외계인 전투, 식민지화가 가능하다"},
                ],
                "references": [],
                "scope_status": "unspecified",
                "unresolved": [],
            }
        ]
    )

    state = build_initial_planning_state(router, prompt)

    assert {item["reason"] for item in state["unresolved"]} == {"scope"}
    assert not any(
        item["reason"] in {"implementation_method", "minecraft_api", "repository_fact", "compatibility"}
        for item in state["unresolved"]
    )
    assert not any(
        item["resolution_route"] == "implementation_research"
        for item in state["unresolved"]
    )


def test_nonexact_model_quote_falls_back_to_valid_full_prompt_receipt() -> None:
    prompt = "중력이 주기적으로 뒤집히는 모드 만들어줘"
    router = _Router(
        [
            {
                "goal": {
                    "statement": "주기적으로 중력이 반전되는 모드",
                    "source_quote": "정확히 원문에 없는 요약",
                },
                "known": [],
                "references": [],
                "scope_status": "explicit",
                "unresolved": [],
            }
        ]
    )

    state = build_initial_planning_state(router, prompt)
    source = state["goal"]["source"]

    assert source["char_start"] == 0
    assert source["char_end"] == len(prompt)
    assert source["text"] == prompt
    assert source["verification"] == "full_prompt_fallback"


def test_reference_query_compiler_is_host_owned_and_target_neutral() -> None:
    router = _Router([])
    state = {"references": [{"name": "메이플스토리"}]}
    research = {
        "objective": "메이플스토리의 실제 시스템을 조사한다",
        "information_needed": "문서화된 게임 시스템과 상호 관계",
        "source_kinds": ["reference_sources", "web_sources"],
    }

    queries = _compile_queries(router, state, research)

    assert queries[0] == "메이플스토리 문서화된 게임 시스템과 상호 관계"
    assert all("Minecraft mod" not in query for query in queries)
    assert router.calls == []


def test_reference_wikipedia_search_uses_authored_language_before_english() -> None:
    assert _wikipedia_languages("메이플스토리 gameplay systems") == ("ko", "en")
    assert _wikipedia_languages("MapleStory gameplay systems") == ("en",)


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
    assert domain["providers"] == ["wikipedia", "external_mcp"]

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
    assert {"dependency", "source_code", "minecraft_api", "local_project"}.issubset(
        domain["evidence_kinds"]
    )
    assert domain["providers"][:2] == ["curseforge", "modrinth"]
    assert "github" not in domain["providers"]
    assert {"official_docs", "project_rag"}.issubset(domain["providers"])


def test_authoritative_catalog_has_no_raw_prompt_fallback() -> None:
    with pytest.raises(ValueError, match="PLANNING_AUTHORITY_STATE_REQUIRED"):
        build_authoritative_request_catalog("메이플스토리 모드 만들어줘")
