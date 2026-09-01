from __future__ import annotations

import json

import pytest

import minecraft_mod_ai.agentic_research_game_design as agentic
from minecraft_mod_ai import game_design
from minecraft_mod_ai.spec import SpecValidationError


class _SectionRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role,
        messages,
        *,
        media_paths=(),
        response_format="text",
        response_schema=None,
        tool_stage=None,
        enable_tools=True,
    ):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": tuple(media_paths),
                "response_format": response_format,
                "response_schema": response_schema,
                "tool_stage": tool_stage,
                "enable_tools": enable_tools,
            }
        )
        text = str(messages[-1]["content"])
        marker = "\n\nSECTION\n"
        section_id = text.split(marker, 1)[1].split("\n", 1)[0].strip()
        bodies = {
            "identity_and_loop": """## title
연구 기반 모드
## pitch
검색 근거를 바탕으로 설계한다.
## core_loop
- 탐색하고 상호작용한다
""",
            "systems_and_progression": """## progression
- 기능을 단계적으로 해금한다
## combat
none
## mod_context
none
""",
            "modules_and_assets": """## modules
none
## assets
none
""",
            "quality_and_art": """## acceptance_tests
- 요청한 핵심 루프가 게임 내에서 동작한다
## art_direction
none
""",
        }
        return bodies[section_id]


class _ResearchRouter:
    def __init__(self, evidence_ref: str = "forced_project_rag") -> None:
        self.calls: list[dict[str, object]] = []
        self.evidence_ref = evidence_ref

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        return json.dumps(
            {
                "research_note": {
                    "domain_id": "request",
                    "claims": [
                        {
                            "claim": "요청은 target-neutral Minecraft 모드 기능 설계로 조사할 수 있다.",
                            "evidence_refs": [self.evidence_ref],
                        }
                    ],
                    "gaps": [],
                    "next_queries": ["정확한 API 이름은 target freeze 뒤 검증"],
                    "procedures": [],
                    "sufficient": True,
                }
            },
            ensure_ascii=False,
        )


def _deterministic_research() -> dict[str, object]:
    return {
        "official_rag": {"status": "deferred_until_platform_selected"},
        "technology_radar": {
            "status": "deferred_until_target_freeze",
            "target_frozen": False,
        },
        "forced_project_rag": {
            "schema_version": "mmm/forced-pre-design-rag-v2",
            "research_sha256": "sha256:forced",
            "domain_count": 1,
            "query_count": 1,
            "project_source_count": 6,
            "domains": [
                {
                    "domain_id": "request",
                    "queries": [{"query": "feature", "sources": [{"source_id": "fixture"}]}],
                }
            ],
        },
    }


def test_sectioned_game_design_uses_host_owned_field_compiler() -> None:
    router = _SectionRouter()
    research = {
        "research_brief": {"domains": []},
        "domain_notes": [],
        "deterministic": {},
        "errors": [],
    }
    result = agentic.generate_sectioned_game_design(
        game_design,
        router,
        "연구를 먼저 하고 모드를 설계해줘",
        research=research,
    )
    assert len(router.calls) == len(agentic._SECTION_SPECS) == 4
    assert all(call["role"] == "planner" for call in router.calls)
    assert all(call["response_format"] == "text" for call in router.calls)
    assert all(call["response_schema"] is None for call in router.calls)
    assert all(call["tool_stage"] == "game_design" for call in router.calls)
    assert all(call["enable_tools"] is False for call in router.calls)
    for (section_id, fields, _properties), call in zip(
        agentic._SECTION_SPECS,
        router.calls,
        strict=True,
    ):
        system = str(call["messages"][0]["content"])
        user = str(call["messages"][1]["content"])
        assert "Write design content as Markdown, not JSON" in system
        assert "single coherent response" in system
        assert f"\n\nSECTION\n{section_id}\n" in user
        for field in fields:
            assert f"## {field}" in user
    assert result["title"] == "연구 기반 모드"
    assert result["core_loop"]
    assert result["acceptance_tests"]
    assert "art_direction" not in result


def test_missing_markdown_headings_can_never_abort_identity_section() -> None:
    class HeadinglessRouter:
        def generate_text(self, role, messages, **kwargs):
            del role, messages, kwargs
            return "계절 작물과 요리를 연결하는 플레이 경험"

    section = agentic._generate_section(
        HeadinglessRouter(),
        prompt="계절마다 다른 작물을 재배하고 요리하는 모드를 만들어줘.",
        section_id="identity_and_loop",
        fields=("title", "pitch", "core_loop"),
        research={},
        media_paths=(),
        trace_metadata=None,
    )
    assert set(section) == {"title", "pitch", "core_loop"}
    assert section["title"]
    assert section["pitch"]
    assert section["core_loop"]


def test_research_domain_legacy_facade_is_host_owned() -> None:
    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model called: {name}")

    result = agentic._research_domain_with_agent(
        NeverModel(),
        prompt="기능을 조사해서 설계해줘",
        domain={"domain_id": "request", "objective": "요청 조사", "queries": ["minecraft mod feature"]},
        deterministic=_deterministic_research(),
        trace_metadata=None,
    )
    assert result["sufficient"] is True
    assert result["research_mode"] == "advisory_predesign"
    assert result["quality_contract"]["model_json"] is False


def test_sufficient_research_rejects_empty_and_invented_refs() -> None:
    note = {
        "domain_id": "request",
        "claims": [{"claim": "unsupported", "evidence_refs": []}],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
    }
    with pytest.raises(SpecValidationError, match="no host-issued evidence_ref"):
        agentic._validate_sufficient_research(
            note,
            allowed_refs=frozenset({"forced_project_rag"}),
        )

    note["claims"][0]["evidence_refs"] = ["minecraft_api:invented"]
    with pytest.raises(SpecValidationError, match="unverified evidence_refs"):
        agentic._validate_sufficient_research(
            note,
            allowed_refs=frozenset({"forced_project_rag"}),
        )


def test_domain_slice_issues_refs_only_for_real_host_evidence() -> None:
    prompt_slice = agentic._domain_evidence_slice(
        "request",
        _deterministic_research(),
    )

    assert "evidence_ref" not in prompt_slice["official_rag"]
    assert "evidence_ref" not in prompt_slice["technology_radar"]
    assert prompt_slice["forced_project_rag"]["evidence_ref"] == "forced_project_rag"
    assert agentic._allowed_research_refs(prompt_slice) == frozenset(
        {"forced_project_rag"}
    )


def test_domain_slice_bounds_forced_receipt_without_materializing_document() -> None:
    huge_forced = "F" * 40_000
    forced = {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "research_sha256": "sha256:forced",
        "query_count": 1,
        "project_source_count": 6,
        "code_index_status": "not_indexed",
        "domains": [
            {
                "domain_id": "request",
                "queries": [{"query": "forced", "content": huge_forced}],
            }
        ],
    }
    prompt_slice = agentic._domain_evidence_slice(
        "request",
        {
            "official_rag": {
                "status": "deferred_until_platform_selected",
                "domains": [],
            },
            "forced_project_rag": forced,
        },
    )

    rendered = json.dumps(prompt_slice, ensure_ascii=False, sort_keys=True)
    assert huge_forced not in rendered
    assert set(prompt_slice) == {"official_rag", "forced_project_rag"}
    assert prompt_slice["forced_project_rag"]["research_sha256"] == "sha256:forced"
    assert prompt_slice["forced_project_rag"]["project_source_count"] == 6
    assert prompt_slice["forced_project_rag"]["evidence_ref"] == "forced_project_rag"
    assert "evidence_document" not in prompt_slice


def test_runtime_binding_uses_native_research_first_owner() -> None:
    assert getattr(game_design.GameDesignPlanner.plan, "_mmm_host_owned_template", False)
    assert not hasattr(game_design._generate_game_design_once, "__wrapped__")
    assert not hasattr(game_design.GameDesignPlanner._plan_sharded_request, "__wrapped__")
