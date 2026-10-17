"""Regression tests for grounded RAG discovery and authored release integrity."""

from __future__ import annotations

from minecraft_mod_ai.authored_reuse_bridge import authored_capability_graph
from minecraft_mod_ai.final_artifact import _authored_feature_semantic_findings
from minecraft_mod_ai.curseforge_query import (
    curseforge_search_queries, curseforge_catalogue_score,
)


def test_curseforge_filters_irrelevant_title_and_description():
    requested = "spaceship aliens and planetary travel"
    assert curseforge_catalogue_score(
        requested, {"name": "Spaceship Explorer", "summary": "Explore planets"}
    ) > 0
    assert curseforge_catalogue_score(
        requested, {"name": "Unrelated Server Library", "summary": "Networking"}
    ) == 0



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
    assert graph["retrieval_policy_version"] == "structured-intent-v3-exact-source-partial-fit"


def test_source_queries_prefer_short_specific_terms():
    queries = curseforge_search_queries(
        "custom gameplay " + ("long worldbuilding request " * 15),
        search_terms=("minecraft spaceship mod", "minecraft planetary ore mod"),
    )
    assert queries
    assert "spaceship" in queries
    assert all(value and len(value) <= 90 for value in queries)



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



