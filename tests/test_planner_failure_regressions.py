from __future__ import annotations

import json

from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai.agent_capability_context import (
    filter_tool_schemas_for_role,
    target_neutral_research_scope,
)
from minecraft_mod_ai.structured_repair_contract import _generate_section_local


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_names(schemas) -> set[str]:
    return {str(item["function"]["name"]) for item in schemas}


class _Router:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        if not self.outputs:
            raise AssertionError("unexpected planner generation")
        return self.outputs.pop(0)


def test_adaptive_progression_repair_is_schema_constrained_and_freezes_siblings() -> None:
    properties = {
        "progression": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "combat": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
        "mod_context": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    router = _Router(
        [
            json.dumps(
                {
                    "section": {
                        "progression": {"levels": ["1", "2"]},
                        "combat": {"boss": ["server-authoritative"]},
                        "mod_context": {"scope": ["maple-style progression"]},
                    }
                }
            ),
            json.dumps(
                {
                    "repair": {
                        "progression": ["레벨 성장", "장비 강화", "보스 단계 해금"]
                    }
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = _generate_section_local(
        router,
        prompt="메이플 스타일 성장 시스템을 설계해줘",
        section_id="systems_and_progression",
        fields=("progression", "combat", "mod_context"),
        properties=properties,
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert result["progression"] == ["레벨 성장", "장비 강화", "보스 단계 해금"]
    assert result["combat"] == {"boss": ["server-authoritative"]}
    assert result["mod_context"] == {"scope": ["maple-style progression"]}
    assert len(router.calls) == 2
    assert router.calls[0]["response_format"] == "text"
    assert router.calls[0]["response_schema"] is None
    assert router.calls[1]["response_format"] == "json"
    repair = router.calls[1]["response_schema"]["properties"]["repair"]
    assert repair["required"] == ["progression"]
    assert repair["properties"]["progression"]["type"] == "array"


def test_target_neutral_research_hides_donor_and_target_compatibility_tools() -> None:
    schemas = (
        _schema("inspect_modrinth_project"),
        _schema("inspect_github_repository"),
        _schema("discover_ecosystem_resources"),
        _schema("assess_technology_compatibility"),
        _schema("search_project_rag"),
        _schema("search_code_rag"),
        _schema("external_mcp_capabilities"),
        _schema("external_mcp_schema"),
        _schema("external_mcp_call"),
    )

    with target_neutral_research_scope():
        filtered = filter_tool_schemas_for_role("research", "planner", schemas)

    names = _tool_names(filtered)
    assert "inspect_modrinth_project" not in names
    assert "inspect_github_repository" not in names
    assert "discover_ecosystem_resources" not in names
    assert "assess_technology_compatibility" not in names
    assert "search_project_rag" not in names
    assert "search_code_rag" in names
    assert {
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    } <= names


def test_ungrounded_sufficient_research_reaches_evidence_frontier_fixed_point(
    monkeypatch,
) -> None:
    class _Trace:
        def __init__(self, *args, **kwargs):
            pass

        def record_attempt(self, **kwargs):
            pass

        def record_success(self, value):
            raise AssertionError("ungrounded research must not be accepted")

    monkeypatch.setattr(agentic, "PlannerStageTrace", _Trace)
    router = _Router(
        [
            json.dumps(
                {
                    "research_note": {
                        "domain_id": "request",
                        "claims": [{"claim": "첫 표현", "evidence_refs": []}],
                        "gaps": [],
                        "next_queries": [],
                        "procedures": [],
                        "sufficient": True,
                    }
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "research_note": {
                        "domain_id": "request",
                        "claims": [{"claim": "말만 바꾼 둘째 표현", "evidence_refs": []}],
                        "gaps": [],
                        "next_queries": [],
                        "procedures": [],
                        "sufficient": True,
                    }
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = agentic._research_domain_with_agent(
        router,
        prompt="기능을 조사해줘",
        domain={
            "domain_id": "request",
            "objective": "target-neutral research",
            "requirements": ["기능"],
            "queries": ["feature"],
        },
        deterministic={
            "forced_project_rag": {
                "project_source_count": 1,
                "domains": [
                    {
                        "domain_id": "request",
                        "queries": [
                            {
                                "query": "feature",
                                "sources": [{"source_id": "fixture"}],
                            }
                        ],
                    }
                ],
            }
        },
        trace_metadata=None,
    )

    assert result["sufficient"] is False
    assert result["fixed_point"] is True
    assert len(router.calls) == 2
