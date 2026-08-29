from __future__ import annotations

import json
from types import SimpleNamespace

import minecraft_mod_ai.agentic_pre_design_rag as project_rag
import minecraft_mod_ai.agentic_research_game_design as agentic


def _brief():
    return {
        "domains": [
            {"domain_id": "gameplay", "queries": ["Fabric item registry", "Fabric GameTest"]},
            {"domain_id": "assets", "queries": ["Minecraft resource assets"]},
        ]
    }


def test_forced_rag_searches_every_research_query(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    router = SimpleNamespace()

    bundle = project_rag._forced_rag_bundle(router, _brief())

    assert bundle["query_count"] == 3
    assert bundle["domain_count"] == 2
    assert bundle["project_source_count"] > 0
    assert bundle["code_index_status"] == "not_indexed"
    queries = [item for domain in bundle["domains"] for item in domain["queries"]]
    assert len(queries) == 3
    assert all(item["project_rag"]["sources"] for item in queries)
    assert all(item["code_rag"]["status"] == "not_indexed" for item in queries)


def test_explicit_target_limits_pre_design_rag_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    router = SimpleNamespace(_mmm_requested_minecraft_version="1.21.1")

    bundle = project_rag._forced_rag_bundle(router, _brief())

    assert bundle["versions"] == ["1.21.1"]
    for domain in bundle["domains"]:
        for query in domain["queries"]:
            sources = query["project_rag"]["sources"]
            assert sources
            assert all(source["matched_version"] == "1.21.1" for source in sources)


def test_legacy_pre_design_collector_is_not_runtime_owner() -> None:
    assert not hasattr(agentic, "collect_pre_design_research")
    assert not getattr(agentic.generate_sectioned_game_design, "_mmm_agentic_research_sectioned", False)


def test_domain_evidence_is_bounded_receipt_not_raw_page_payload() -> None:
    huge_gameplay = "gameplay-evidence-" + ("G" * 20_000)
    huge_assets = "asset-evidence-" + ("A" * 20_000)
    forced = {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "query_count": 2,
        "research_sha256": "sha256:forced",
        "project_source_count": 12,
        "code_index_status": "not_indexed",
        "domains": [
            {"domain_id": "gameplay", "queries": [{"query": "gameplay", "content": huge_gameplay}]},
            {"domain_id": "assets", "queries": [{"query": "assets", "content": huge_assets}]},
        ],
    }

    value = agentic._domain_evidence_slice("gameplay", {"forced_project_rag": forced})

    assert set(value) == {"forced_project_rag"}
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    assert huge_gameplay not in rendered
    assert huge_assets not in rendered
    assert "domains" not in rendered
    assert value["forced_project_rag"]["research_sha256"] == "sha256:forced"
    assert value["forced_project_rag"]["project_source_count"] == 12


class _ToolResearchRouter:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return json.dumps(
            {
                "research_note": {
                    "domain_id": "request",
                    "claims": [
                        {
                            "claim": "Target-neutral architecture can be researched before exact version selection.",
                            "evidence_refs": ["tool:official_docs"],
                        }
                    ],
                    "gaps": [],
                    "next_queries": ["Verify exact mappings after target freeze"],
                    "procedures": [],
                    "sufficient": True,
                }
            }
        )


def test_domain_research_uses_tools_and_does_not_require_target_version(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(tmp_path))
    router = _ToolResearchRouter()

    note = agentic._research_domain_with_agent(
        router,
        prompt="design a space mod",
        domain={"domain_id": "request", "queries": ["space mod architecture"]},
        deterministic={
            "technology_radar": {
                "status": "deferred_until_target_freeze",
                "target_frozen": False,
            }
        },
        trace_metadata=None,
    )

    assert note["sufficient"] is True
    assert note["claims"]
    assert len(router.calls) == 1
    _role, messages, kwargs = router.calls[0]
    assert kwargs["tool_stage"] == "research"
    assert kwargs["enable_tools"] is True
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "intentionally not frozen" in rendered
    assert "Do not ask the user" in rendered
