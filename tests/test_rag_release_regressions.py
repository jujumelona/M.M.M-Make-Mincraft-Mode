"""Regression tests for grounded RAG discovery and authored release integrity."""

from __future__ import annotations

from minecraft_mod_ai.authored_reuse_bridge import authored_capability_graph
from minecraft_mod_ai.final_artifact import _authored_feature_semantic_findings
from minecraft_mod_ai.reuse_discovery import (
    _github_candidate_is_minecraft_mod,
    _query_variants,
)


def test_github_candidate_rejects_unrelated_readme_keyword_hits() -> None:
    for repository, summary in (
        ("rust-unofficial/awesome-rust", "Curated Rust ecosystem list"),
        ("gmh5225/awesome-game-security", "Security tools for games"),
        ("lobehub/lobe-chat-agents", "AI chat agents"),
        ("ferrumc-rs/ferrumc", "Minecraft server written in Rust"),
    ):
        assert not _github_candidate_is_minecraft_mod(
            {"title": repository, "summary": summary}
        )


def test_github_candidate_keeps_actual_mod_and_resource_pack() -> None:
    for name, summary in (
        ("author/spacecraft", "A Minecraft Fabric mod with ships and planets"),
        ("author/textures", "Minecraft texture pack for custom blocks"),
    ):
        assert _github_candidate_is_minecraft_mod(
            {"title": name, "summary": summary}
        )


def test_authored_graph_extracts_actual_structured_capabilities() -> None:
    structured = {
        "reuse_assessment": {
            "specification": {
                "adaptations": [{"part": "space_ship_fabrication"}],
            },
        },
        "resources_and_ui": {
            "specification": {
                "assets": [{"purpose": "planet_ore_registration"}],
            },
        },
    }
    graph = authored_capability_graph("우주선 제작과 광물 파밍", structured)
    assert "minecraft fabric space ship fabrication" in graph["nodes"]
    assert "minecraft fabric planet ore registration" in graph["nodes"]
    assert graph["retrieval_policy_version"] == "structured-intent-v2"


def test_source_queries_prefer_structured_terms_over_long_request(monkeypatch) -> None:
    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "3")
    query = "custom gameplay " + ("very long worldbuilding request " * 15)
    variants = _query_variants(
        query,
        ("minecraft spaceship mod", "minecraft planetary ore mod", query),
    )
    assert variants[0] == "minecraft spaceship mod"
    assert variants[1] == "minecraft planetary ore mod"
    assert all(len(value) <= 90 for value in variants)


def test_authored_coverage_rejects_nonfunctional_gui_placeholder(tmp_path) -> None:
    source = (
        tmp_path / "src" / "client" / "java" / "demo"
        / "generated" / "FabricationGuiScreenRegistration.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        'public class FabricationGuiScreenRegistration { '
        'String title = "Planned: Implement gameplay obligation ship assembly"; }',
        encoding="utf-8",
    )
    findings = _authored_feature_semantic_findings(tmp_path, ())
    assert any("nonfunctional requirement placeholder" in item for item in findings)


def test_authored_coverage_does_not_reject_real_gui_label(tmp_path) -> None:
    source = (
        tmp_path / "src" / "client" / "java" / "demo"
        / "generated" / "FabricationGuiScreenRegistration.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        'public class FabricationGuiScreenRegistration { '
        'String title = "Fabricate ship module"; }',
        encoding="utf-8",
    )
    assert _authored_feature_semantic_findings(tmp_path, ()) == []



def test_reuse_discovery_passes_resolved_target_to_search(monkeypatch) -> None:
    from minecraft_mod_ai.reuse_discovery import discover_repositories_for_graph

    monkeypatch.setenv("MMM_REUSE_QUERY_VARIANTS", "1")

    class StubClient:
        def __init__(self) -> None:
            self.calls = []

        def search(self, provider, query, *, minecraft_version=None, loader=None, **kwargs):
            self.calls.append((provider, minecraft_version, loader))
            if provider == "github":
                return {"candidates": [{
                    "title": "author/spacecraft",
                    "summary": "A Minecraft Fabric mod with planetary flight",
                }]}
            return {"candidates": []}

    client = StubClient()
    discovered = discover_repositories_for_graph(
        ("minecraft fabric spacecraft",), client,
        minecraft_version="1.21.1", loader="fabric",
    )
    assert "author/spacecraft" in discovered["minecraft fabric spacecraft"]
    assert client.calls
    assert all(version == "1.21.1" and loader == "fabric" for _, version, loader in client.calls)
