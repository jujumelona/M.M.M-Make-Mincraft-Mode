from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from minecraft_mod_ai import api
from minecraft_mod_ai.platform_catalog import adapter_for_target, provider_for_loader
from minecraft_mod_ai.platform_optimizer import TargetEvidence

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"


def _evidence(adapter, *, coverage: int, requested: int, freshness: float) -> TargetEvidence:
    capabilities = tuple(f"cap-{index}" for index in range(requested))
    return TargetEvidence(
        adapter=adapter,
        requested_capabilities=capabilities,
        covered_capabilities=capabilities[:coverage],
        exact_projects=tuple(f"project-{index}" for index in range(coverage)),
        exact_versions=coverage,
        verified_hash_files=coverage,
        dependency_edges=0,
        maintenance_signals=coverage,
        adoption=1_000_000 if adapter.minecraft_version == "1.21.1" else 1,
        freshness=freshness,
        evidence_quality=1.0,
        integration_risk=0.0,
        residual_cost=requested - coverage,
        dependency_complexity=0,
    )


def test_reuse_coverage_beats_newer_more_popular_target() -> None:
    older = _evidence(
        adapter_for_target("1.20.1", "fabric"),
        coverage=11,
        requested=12,
        freshness=1.0,
    )
    newer = _evidence(
        adapter_for_target("1.21.1", "fabric"),
        coverage=4,
        requested=12,
        freshness=9_999_999_999.0,
    )
    assert older.rank_key > newer.rank_key


def test_missing_loader_provider_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="No executable platform provider"):
        provider_for_loader("neoforge")


def test_public_sessions_default_to_unpinned_target() -> None:
    legacy = inspect.signature(api.ModAISession.__init__)
    complete = inspect.signature(api.CompleteModAISession.__init__)
    assert legacy.parameters["minecraft_version"].default is None
    assert legacy.parameters["loader"].default is None
    assert complete.parameters["minecraft_version"].default is None
    assert complete.parameters["loader"].default is None


def test_no_historical_constructor_placeholder_or_newest_fallback() -> None:
    api_source = (PACKAGE / "api.py").read_text(encoding="utf-8")
    resolver = (PACKAGE / "platform_resolver.py").read_text(encoding="utf-8")
    live_rag = (PACKAGE / "platform_live_rag_contract.py").read_text(encoding="utf-8")
    assert 'minecraft_version: str = "1.20.1"' not in api_source
    assert 'minecraft_version: str = "1.20.1"' not in live_rag
    assert 'loader: str = "fabric"' not in live_rag
    assert "_choose_with_central_ai" not in resolver
    assert not (PACKAGE / "platform_api_contract.py").exists()
    assert not (PACKAGE / "platform_prompt_contract.py").exists()
    assert not (PACKAGE / "platform_selection_efficiency_contract.py").exists()


def test_platform_planning_only_binds_legacy_pipeline_target() -> None:
    source = (PACKAGE / "platform_planning_contract.py").read_text(encoding="utf-8")
    assert "_install_pipeline_target_binding" in source
    assert "platform_resolver.resolve_platform" in source
    assert "platform_resolver.retarget_proposal" in source
    assert "GameDesignPlanner.plan =" not in source
    assert "_platform_evidence" in source
    assert "platform_prompt_contract" in source  # behavior is folded here, not imported
