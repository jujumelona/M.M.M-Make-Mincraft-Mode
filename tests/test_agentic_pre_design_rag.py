from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.agentic_pre_design_rag as hardening
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

    bundle = hardening._forced_rag_bundle(router, _brief())

    assert bundle["query_count"] == 3
    assert bundle["domain_count"] == 2
    assert bundle["project_source_count"] > 0
    assert bundle["code_index_status"] == "not_indexed"
    queries = [
        item
        for domain in bundle["domains"]
        for item in domain["queries"]
    ]
    assert len(queries) == 3
    assert all(item["project_rag"]["sources"] for item in queries)
    assert all(item["code_rag"]["status"] == "not_indexed" for item in queries)


def test_explicit_target_limits_pre_design_rag_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    router = SimpleNamespace(_mmm_requested_minecraft_version="1.21.1")

    bundle = hardening._forced_rag_bundle(router, _brief())

    assert bundle["versions"] == ["1.21.1"]
    for domain in bundle["domains"]:
        for query in domain["queries"]:
            sources = query["project_rag"]["sources"]
            assert sources
            assert all(source["matched_version"] == "1.21.1" for source in sources)


def test_runtime_hardening_is_bound_and_compacts_section_receipts() -> None:
    assert getattr(
        agentic.collect_pre_design_research,
        "_mmm_forced_pre_design_rag_v1",
        False,
    )

    compact = agentic._research_receipt(
        {
            "schema_version": "example-v1",
            "radar_sha256": "sha256:abc",
            "requirements": [{"huge": "payload"}],
            "errors": [{"huge": "payload"}],
            "query_count": 7,
        }
    )
    assert compact == {
        "schema_version": "example-v1",
        "radar_sha256": "sha256:abc",
        "query_count": 7,
    }
