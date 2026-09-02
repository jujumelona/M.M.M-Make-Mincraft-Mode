from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"


def test_receipt_native_platform_evidence_has_no_retired_optimizer_or_semantic_candidate_cap() -> None:
    source = (PACKAGE / "platform_evidence_pipeline.py").read_text(encoding="utf-8")
    reuse = (PACKAGE / "reuse_planner.py").read_text(encoding="utf-8")

    assert "platform_optimizer" not in source
    assert "platform_optimizer" not in reuse
    assert "MMM_PLATFORM_CANDIDATE_LIMIT" not in source
    assert "MMM_PLATFORM_CANDIDATE_LIMIT" not in reuse
    assert "_parallel_support_matrix" not in reuse
    assert "_parallel_deep" not in reuse
    assert "fresh_only" not in reuse


def test_modrinth_search_is_exhaustive_transport_paging_not_semantic_top_k() -> None:
    source = (PACKAGE / "platform_evidence_pipeline.py").read_text(encoding="utf-8")
    assert "def _search_modrinth_exhaustive" in source
    assert "total_hits" in source
    assert "offset += len(hits)" in source
    assert "if not hits or offset >= total" in source


def test_retired_pipeline_hardening_monkeypatch_modules_are_absent() -> None:
    for name in (
        "pipeline_hardening.py",
        "pipeline_hardening_v2.py",
        "pipeline_hardening_v4.py",
        "pipeline_hardening_v7.py",
    ):
        assert not (PACKAGE / name).exists()


def test_machine_pack_metadata_preserves_three_value_contract(monkeypatch) -> None:
    from minecraft_mod_ai import platform_live_discovery as live

    monkeypatch.setattr(
        live,
        "_mojang_pack_versions",
        lambda version: ("61", "46"),
    )
    monkeypatch.setattr(
        live,
        "_mojang_target_url",
        lambda version: f"https://piston-meta.mojang.com/{version}.json",
    )

    data_pack, resource_pack, source_url = live._official_pack_versions("1.21.1")
    assert data_pack == "61"
    assert resource_pack == "46"
    assert source_url.endswith("/1.21.1.json")


def test_lossless_page_research_has_single_canonical_owner() -> None:
    from minecraft_mod_ai import pre_design_domain_research as domain_research

    assert callable(domain_research.research_document_domain)
    assert domain_research.research_document_domain.__module__ == (
        "minecraft_mod_ai.pre_design_domain_research"
    )
