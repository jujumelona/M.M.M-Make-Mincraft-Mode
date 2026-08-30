from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import authored_scope_research_contract as authored
from minecraft_mod_ai import pre_design_rag_corrective as corrective
from minecraft_mod_ai import research_grounded_rag_contract as grounded


def test_pre_design_executes_small_discovery_set_without_deep_source_crawl(monkeypatch) -> None:
    prompt = "우주선을 부위마다 만들고 다른 행성을 탐사하는 모드"
    planned_queries = [f"minecraft mod mechanic {index} discovery" for index in range(30)]
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

    discovery_queries = grounded._external_brief_queries(rewritten)
    assert len(discovery_queries) == 6
    assert discovery_queries[0] == planned_queries[0]
    assert discovery_queries[-1] == planned_queries[-1]

    calls: list[tuple[str, str]] = []

    def fake_external_retrieval(query: str, versions, *, mode: str = "source"):
        del versions
        calls.append((query, mode))
        return {
            "schema_version": "mmm/external-grounded-rag-v1",
            "status": "available",
            "retrieval_mode": mode,
            "query": query,
            "actual_source_document_count": 0,
            "document_count": 1,
            "documents": [
                {
                    "source_id": f"modrinth:example/{query}",
                    "source_type": "modrinth_project",
                    "url": "https://modrinth.com/mod/example",
                    "content": f"project description for {query}",
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
        external_queries=discovery_queries,
    )

    assert len(calls) == 6
    assert all(mode == "planning_discovery" for _, mode in calls)
    assert result["external_query_count"] == 6
    assert result["external_source_count"] == 0
    assert sum(
        1
        for row in result["domains"][0]["queries"]
        if "external_rag" in row
    ) == 6


def test_corrective_retrieval_uses_targeted_source_mode(monkeypatch) -> None:
    query = "minecraft modular spacecraft assembly source"
    brief = {
        "schema_version": "mmm/corrective-retrieval-request-v1",
        "domains": [
            {
                "domain_id": "request",
                "providers": ["github", "modrinth"],
                "queries": [query],
            }
        ],
    }
    assert grounded._external_brief_queries(brief) == (query,)
    calls: list[str] = []

    def fake_external_retrieval(value: str, versions, *, mode: str = "source"):
        del versions
        calls.append(mode)
        return {
            "status": "available",
            "query": value,
            "actual_source_document_count": 1,
            "document_count": 1,
            "documents": [
                {
                    "source_id": "github:example/Spacecraft.java",
                    "source_type": "github_source",
                    "content": "targeted spacecraft assembly evidence",
                }
            ],
        }

    monkeypatch.setattr(grounded, "_external_retrieval", fake_external_retrieval)
    payload = {
        "domains": [
            {
                "domain_id": "request",
                "queries": [{"query": query}],
            }
        ]
    }
    grounded._augment_bundle(
        SimpleNamespace(_sha256=lambda value: "sha256:test"),
        payload,
        versions=(),
        local_index={"status": "workspace_unconfigured"},
        external_queries=(query,),
        default_mode="planning_gap_source",
    )
    assert calls == ["planning_gap_source"]


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
