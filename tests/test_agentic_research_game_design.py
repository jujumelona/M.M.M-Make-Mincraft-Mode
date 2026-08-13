from __future__ import annotations

import json
from types import SimpleNamespace

import minecraft_mod_ai.agentic_research_game_design as agentic
import minecraft_mod_ai.game_design as game_design


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
        required = tuple(response_schema["properties"]["section"]["required"])
        values = {
            "title": "연구 기반 모드",
            "pitch": "검색 근거를 바탕으로 설계한다.",
            "core_loop": ["탐색하고 상호작용한다"],
            "progression": ["기능을 단계적으로 해금한다"],
            "combat": {},
            "mod_context": {},
            "modules": [],
            "assets": [],
            "acceptance_tests": ["요청한 핵심 루프가 게임 내에서 동작한다"],
            "art_direction": {},
        }
        return json.dumps(
            {"section": {field: values[field] for field in required}},
            ensure_ascii=False,
        )


class _ResearchRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        return json.dumps(
            {
                "research_note": {
                    "domain_id": "request",
                    "claims": [
                        {
                            "claim": "요청은 일반 Minecraft 모드 기능 설계다.",
                            "evidence_refs": ["official_rag:request"],
                        }
                    ],
                    "gaps": [],
                    "next_queries": [],
                    "sufficient": True,
                }
            },
            ensure_ascii=False,
        )


def test_sectioned_game_design_uses_four_small_schema_calls() -> None:
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

    assert len(router.calls) == 4
    assert all(call["role"] == "planner" for call in router.calls)
    assert all(call["response_format"] == "json" for call in router.calls)
    assert all(call["enable_tools"] is False for call in router.calls)
    assert all(call["response_schema"] for call in router.calls)
    assert result["title"] == "연구 기반 모드"
    assert result["core_loop"]
    assert result["acceptance_tests"]
    assert "art_direction" not in result


def test_research_domain_uses_research_tools_before_declaring_sufficient(monkeypatch) -> None:
    router = _ResearchRouter()

    class _Trace:
        def __init__(self, *args, **kwargs):
            self.attempts = []

        def record_attempt(self, **kwargs):
            self.attempts.append(kwargs)

        def record_success(self, value):
            self.success = value

    monkeypatch.setattr(agentic, "PlannerStageTrace", _Trace)

    result = agentic._research_domain_with_agent(
        router,
        prompt="기능을 조사해서 설계해줘",
        domain={
            "domain_id": "request",
            "objective": "요청 조사",
            "requirements": ["기능"],
            "evidence_kinds": ["minecraft_api"],
            "queries": ["minecraft mod feature"],
            "providers": ["official_docs", "project_rag", "github"],
            "depends_on": [],
        },
        deterministic={"official_rag": {"domains": []}},
        trace_metadata=None,
    )

    assert result["sufficient"] is True
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["tool_stage"] == "research"
    assert call["enable_tools"] is True
    assert call["response_format"] == "json"
    assert call["response_schema"] == agentic._RESEARCH_NOTE_SCHEMA


def test_runtime_binding_marks_game_design_as_research_first() -> None:
    assert getattr(
        game_design.GameDesignPlanner.plan,
        "_mmm_agentic_research_sectioned",
        False,
    )
