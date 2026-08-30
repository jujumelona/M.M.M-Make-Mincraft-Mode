from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import authored_scope_research_contract as authored
from minecraft_mod_ai import grounded_rag_runtime_contract as runtime
from minecraft_mod_ai import pre_design_rag_corrective as corrective
from minecraft_mod_ai import research_grounded_rag_contract as grounded


def test_approved_queries_reach_public_source_retrieval(monkeypatch) -> None:
    prompt = "우주선을 부위마다 만들고 다른 행성을 탐사하는 모드"
    planned_queries = [
        "minecraft mod modular spacecraft construction",
        "minecraft mod alien planet exploration",
    ]
    monkeypatch.setattr(
        authored,
        "_active_catalog",
        lambda value: {
            "requirements": [
                {
                    "requirement_id": "req_spacecraft",
                    "search_queries": planned_queries,
                }
            ]
        }
        if value == prompt
        else None,
    )
    candidate = {
        "domains": [
            {
                "domain_id": "request",
                "providers": ["official_docs", "project_rag", "external_mcp", "runtime"],
                "queries": [prompt],
            }
        ]
    }

    rewritten = authored._rewrite_pre_design_candidate(prompt, candidate)
    domain = rewritten["domains"][0]

    assert domain["queries"] == planned_queries
    assert prompt not in domain["queries"]
    assert "github" in domain["providers"]
    assert "modrinth" in domain["providers"]

    grounded_queries = grounded._external_brief_queries(rewritten)
    runtime_queries = runtime._external_brief_queries(rewritten)
    assert grounded_queries == tuple(planned_queries)
    assert runtime_queries == tuple(planned_queries)

    calls: list[str] = []

    def fake_external_retrieval(query: str, versions):
        del versions
        calls.append(query)
        return {
            "schema_version": "mmm/external-grounded-rag-v1",
            "status": "available",
            "query": query,
            "actual_source_document_count": 1,
            "document_count": 1,
            "documents": [
                {
                    "source_id": f"github:example/{query}",
                    "source_type": "github_source",
                    "url": "https://github.com/example/mod/blob/main/Example.java",
                    "content": f"source evidence for {query}",
                }
            ],
        }

    monkeypatch.setattr(grounded, "_external_retrieval", fake_external_retrieval)
    payload = {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "versions": [],
        "domains": [
            {
                "domain_id": "request",
                "queries": [{"query": query} for query in planned_queries],
            }
        ],
    }
    result = grounded._augment_bundle(
        SimpleNamespace(_sha256=lambda value: "sha256:test"),
        payload,
        versions=(),
        local_index={"status": "workspace_unconfigured"},
        external_queries=grounded_queries,
    )

    assert calls == planned_queries
    assert result["external_query_count"] == len(planned_queries)
    assert result["external_source_count"] == len(planned_queries)
    for query_row in result["domains"][0]["queries"]:
        document = query_row["external_rag"]["documents"][0]
        assert document["content"].startswith("source evidence for minecraft mod")


def test_corrective_query_keeps_queries_when_model_adds_diagnostics() -> None:
    captured: dict[str, object] = {}

    class Agentic:
        class SpecValidationError(ValueError):
            pass

    class ProjectRag:
        @staticmethod
        def _generate_bounded(
            agentic_module,
            router,
            *,
            messages,
            response_schema,
            parser,
            progress_label,
        ):
            del agentic_module, router, messages, progress_label
            captured["schema"] = response_schema
            return parser(
                json.dumps(
                    {
                        "sufficient": False,
                        "gaps": "Initial evidence contained metadata but no substantive source.",
                        "queries": [
                            "minecraft mod fabric modular spacecraft source",
                            "minecraft mod fabric planet colonization implementation",
                        ],
                    }
                )
            )

    seen = {"minecraft mod initial query"}
    result = corrective._generate_gap_queries(
        Agentic,
        ProjectRag,
        object(),
        domain={"domain_id": "request", "requirements": ["spacecraft"]},
        gaps=["Need substantive implementation evidence."],
        prior_queries=["minecraft mod initial query"],
        seen=seen,
        raw_prompt="우주모드",
        progress_label="test",
    )

    assert captured["schema"]["additionalProperties"] is True
    assert result == [
        "minecraft mod fabric modular spacecraft source",
        "minecraft mod fabric planet colonization implementation",
    ]
