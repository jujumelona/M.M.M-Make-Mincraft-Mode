from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai import pre_design_research_pipeline as pipeline
from minecraft_mod_ai.agent_capability_context import (
    filter_tool_schemas_for_role,
    target_neutral_research_scope,
)
from minecraft_mod_ai.pre_design_research_pipeline import PreDesignResearchFailure


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


def test_document_grounding_rejects_invented_page_ref(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    document = project_rag._materialize_domain_evidence_document(
        "request",
        {
            "official_rag": {
                "documents": [
                    {
                        "document_id": "fixture",
                        "content": "real host-owned target-neutral evidence",
                    }
                ]
            }
        },
    )
    pages = project_rag._read_evidence_pages(document)
    assert pages

    invented = {
        "domain_id": "request",
        "claims": [
            {
                "claim": "unsupported claim",
                "evidence_refs": ["sha256:invented#page=1/1"],
            }
        ],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
    }
    with pytest.raises(PreDesignResearchFailure, match="lossless evidence pages"):
        pipeline._validate_document_grounding(
            agentic,
            project_rag,
            invented,
            document,
            domain_id="request",
        )

    grounded = {
        **invented,
        "claims": [
            {
                "claim": "grounded claim",
                "evidence_refs": [pages[0]["page_ref"]],
            }
        ],
    }
    pipeline._validate_document_grounding(
        agentic,
        project_rag,
        grounded,
        document,
        domain_id="request",
    )
