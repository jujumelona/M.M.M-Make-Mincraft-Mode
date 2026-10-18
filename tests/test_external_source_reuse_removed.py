"""Prevent legacy external-mod source reuse paths from re-entering production."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_external_donor_calls_in_active_pipeline():
    forbidden = {
        "complete_planner.py": ("authored_reuse_bridge", "research_reuse_candidates", "_host_source_reuse"),
        "complete_orchestrator.py": ("curseforge_reuse_production", "materialize_verified_authored_sources", "source_reuse_audit"),
        "pre_design_research_pipeline.py": ("collect_source_candidate_handoff", "research_reuse_candidates"),
        "catalog_first_grounded_rag.py": ("_search_curseforge", "_search_modrinth"),
        "mcp_server.py": ("read_reuse_source",),
        "production_tools.py": ("def read_reuse_source",),
    }
    for filename, symbols in forbidden.items():
        source = (ROOT / "minecraft_mod_ai" / filename).read_text(encoding="utf-8")
        for symbol in symbols:
            assert symbol not in source, (filename, symbol)


def test_notebook_has_no_external_mod_credentials_or_reference_urls():
    notebook = (ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8")
    assert "CURSEFORGE_API_KEY" not in notebook
    assert "REFERENCE_MOD_URLS" not in notebook
