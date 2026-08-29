from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.agentic_pre_design_rag as project_rag
import minecraft_mod_ai.agentic_research_game_design as agentic


def _brief():
    return {
        "domains": [
            {
                "domain_id": "gameplay",
                "queries": ["Fabric item registry", "Fabric GameTest"],
            },
            {
                "domain_id": "assets",
                "queries": ["Minecraft resource assets"],
            },
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
    assert not getattr(
        agentic.generate_sectioned_game_design,
        "_mmm_agentic_research_sectioned",
        False,
    )


def test_forced_project_rag_is_scoped_to_current_domain() -> None:
    value = agentic._domain_evidence_slice(
        "gameplay",
        {
            "forced_project_rag": {
                "schema_version": "mmm/forced-pre-design-rag-v2",
                "query_count": 2,
                "domains": [
                    {"domain_id": "gameplay", "queries": [{"query": "gameplay"}]},
                    {"domain_id": "assets", "queries": [{"query": "assets"}]},
                ],
            }
        },
    )

    forced = value["forced_project_rag"]
    assert forced["domain_id"] == "gameplay"
    assert forced["queries"] == [{"query": "gameplay"}]
    assert "domains" not in forced
