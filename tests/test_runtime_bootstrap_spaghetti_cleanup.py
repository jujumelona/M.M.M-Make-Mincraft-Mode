from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"


def _text(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def test_runtime_bootstrap_contains_composition_not_inline_runtime_patches() -> None:
    source = _text("runtime_bootstrap.py")

    assert "complete_orchestrator.synthesize_audio_files =" not in source
    assert "central_research._bounded_text" not in source
    assert "research_coordinator.discover_seed_bundle =" not in source
    assert "CompactingAdapter" not in source
    assert "_generate_with_compaction" not in source
    assert source.count("install_context_compaction(model_router)") == 1


def test_small_model_research_drops_obsolete_private_compatibility_wrappers() -> None:
    source = _text("small_model_research_contract.py")

    assert "central_module._bounded_text" not in source
    assert "domain_with_query_pages" not in source
    assert "_mmm_lossless_query_pages" not in source
    assert "page_builder or ecosystem_module.discover_seed_bundle" in source


def test_context_compaction_has_one_explicit_owner() -> None:
    source = _text("small_model_context_compaction_contract.py")

    assert "def install(model_router_module" in source
    assert "CompactingAdapter(adapter)" in source
    assert "_mmm_lossless_context_compaction" in source
    assert "cls._generate_with_tools = generate_with_compaction" in source
