from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import component_registry
from minecraft_mod_ai.reuse_planner import ReuseDecision, decompose_capability_graph
from minecraft_mod_ai.source_transplant import _target_compatibility


def test_capability_graph_uses_behavior_not_whole_mod_theme() -> None:
    graph = decompose_capability_graph(
        "메이플스토리 같은 모드",
        design={"systems": [{"id": "trade"}, {"id": "shop"}]},
    )
    assert "trade.transaction" in graph.nodes
    assert "trade.validation" in graph.nodes
    assert "ui.shop_menu" in graph.nodes
    assert all("maple" not in item and "메이플" not in item for item in graph.nodes)


def test_reuse_value_is_saved_work_not_reuse_count() -> None:
    expensive = ReuseDecision(
        capability="trade.transaction",
        mode="source_transplant",
        confidence=0.95,
        fresh_implementation_cost=40.0,
        fresh_verification_cost=15.0,
        adaptation_cost=5.0,
        integration_cost=3.0,
        dependency_cost=2.0,
        reuse_verification_cost=4.0,
        uncertainty_penalty=1.0,
    )
    trivial_a = ReuseDecision(
        capability="lang.entry",
        mode="library",
        confidence=1.0,
        fresh_implementation_cost=2.0,
        fresh_verification_cost=1.0,
        integration_cost=1.5,
        reuse_verification_cost=0.5,
    )
    trivial_b = ReuseDecision(
        capability="tag.entry",
        mode="library",
        confidence=1.0,
        fresh_implementation_cost=2.0,
        fresh_verification_cost=1.0,
        integration_cost=1.5,
        reuse_verification_cost=0.5,
    )
    assert expensive.actual_reuse_gain > trivial_a.actual_reuse_gain + trivial_b.actual_reuse_gain


def test_missing_remote_reuse_manifest_is_normal_zero_candidates(monkeypatch) -> None:
    monkeypatch.setattr(component_registry, "_read_remote_manifest", lambda *_args: None)
    assert component_registry.load_verified_components() == ()


def test_exact_transplant_target_requires_version_and_loader_evidence() -> None:
    adapter = SimpleNamespace(minecraft_version="1.20.1", loader="fabric")
    assert _target_compatibility(
        "minecraft_version=1.20.1\ndepends fabricloader >=0.15",
        adapter=adapter,
    ) == "exact"
    assert _target_compatibility("minecraft_version=1.20.1", adapter=adapter) == "adapt"
